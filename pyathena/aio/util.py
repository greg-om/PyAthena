# Copyright 2025 The PyAthena authors
#
# Licensed under the MIT License.
# See LICENSE or https://opensource.org/licenses/MIT.
#
# SPDX-License-Identifier: MIT

"""Asyncio helpers for retrying AWS API calls and closing resources in a thread."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from pyathena.util import RetryConfig, retry_api_call


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
