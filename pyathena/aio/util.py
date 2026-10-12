# Copyright 2025 The PyAthena authors
#
# Licensed under the MIT License.
# See LICENSE or https://opensource.org/licenses/MIT.
#
# SPDX-License-Identifier: MIT

"""Asyncio helpers for retrying AWS API calls and closing resources in a thread."""

from __future__ import annotations

import asyncio
import contextvars
import logging
import threading
from collections.abc import Callable
from typing import Any, Protocol, TypeVar, cast

from pyathena.util import RetryConfig, retry_api_call


class _Closable(Protocol):
    def close(self) -> Any: ...


_ClosableT = TypeVar("_ClosableT", bound=_Closable)


async def async_retry_api_call(
    func: Callable[..., Any],
    config: RetryConfig,
    logger: logging.Logger | None = None,
    *args: Any,
    stop_on: Callable[[BaseException], bool] | None = None,
    **kwargs: Any,
) -> Any:
    """Execute a function with retry logic in a thread to avoid blocking the event loop.

    Wraps ``retry_api_call`` with ``asyncio.to_thread()`` so that blocking
    boto3 calls do not block the asyncio event loop.

    Args:
        func: The AWS API function to call.
        config: RetryConfig instance specifying retry behavior.
        logger: Optional logger for retry attempt logging.
        *args: Positional arguments to pass to ``retry_api_call``.
        stop_on: Passed to ``retry_api_call``.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        The result of the successful function call.
    """
    return await asyncio.to_thread(
        retry_api_call, func, config, logger, *args, stop_on=stop_on, **kwargs
    )


async def close_in_thread(close: Callable[[], Any]) -> None:
    """Run a blocking close in the event loop's default executor.

    The close is submitted at once and awaited through ``asyncio.shield()``, so
    it runs to completion even if the caller is cancelled before or while it
    runs. ``asyncio.to_thread()`` would drop a close that is still queued when
    the caller is cancelled.

    Args:
        close: The blocking function that closes the resource.
    """
    await asyncio.shield(asyncio.get_running_loop().run_in_executor(None, close))


async def create_in_thread(
    factory: Callable[..., _ClosableT], *args: Any, **kwargs: Any
) -> _ClosableT:
    """Run a blocking constructor of a closable object in the default executor.

    Like ``asyncio.to_thread()``, the constructor runs in the event loop's
    default executor with the caller's context. If the caller is cancelled, a
    construction that has not started is skipped, an object constructed after
    the cancellation is closed by the constructing thread, and an object
    constructed just before it is closed in a new thread: nothing else can
    close it, and a close by the garbage collector could block the event
    loop's thread. The cleanup needs neither the event loop nor its executor,
    which may be shutting down.

    Args:
        factory: The constructor.
        *args: Positional arguments for ``factory``.
        **kwargs: Keyword arguments for ``factory``.

    Returns:
        The constructed object.
    """
    lock = threading.Lock()
    cancelled = threading.Event()
    constructed: list[_ClosableT] = []

    def construct() -> _ClosableT | None:
        if cancelled.is_set():
            return None
        obj = factory(*args, **kwargs)
        with lock:
            if not cancelled.is_set():
                constructed.append(obj)
                return obj
        obj.close()
        return None

    future = asyncio.get_running_loop().run_in_executor(
        None, contextvars.copy_context().run, construct
    )
    try:
        # Unlike asyncio.shield(), asyncio.wait() reports nothing when the
        # construction fails after the caller is cancelled.
        await asyncio.wait((future,))
    except asyncio.CancelledError:
        with lock:
            cancelled.set()
        # Retrieve a later construction error, which nobody else awaits.
        future.add_done_callback(lambda f: f.cancelled() or f.exception())
        if constructed:
            threading.Thread(target=constructed[0].close, name="create_in_thread-close").start()
        raise
    return cast(_ClosableT, future.result())
