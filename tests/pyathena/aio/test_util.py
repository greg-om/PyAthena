# Copyright 2026 The PyAthena authors
#
# Licensed under the MIT License.
# See LICENSE or https://opensource.org/licenses/MIT.
#
# SPDX-License-Identifier: MIT
import asyncio
import contextvars
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from pyathena.aio.util import create_in_thread
from tests.pyathena.util import EVENT_TIMEOUT

_VALUE: contextvars.ContextVar[str] = contextvars.ContextVar("_VALUE", default="unset")


class _Resource:
    """A closable object that records its constructor and close threads."""

    def __init__(self, started=None, release=None):
        self.created_in = threading.get_ident()
        self.value = _VALUE.get()
        self.closed = threading.Event()
        self.closed_in = None
        if started is not None:
            started.set()
            assert release.wait(EVENT_TIMEOUT)

    def close(self):
        self.closed_in = threading.get_ident()
        self.closed.set()


async def test_create_in_thread():
    """The constructor runs in another thread with the caller's context."""
    _VALUE.set("caller")
    resource = await create_in_thread(_Resource)
    assert resource.created_in != threading.get_ident()
    assert resource.value == "caller"
    assert not resource.closed.is_set()


async def test_create_in_thread_cancelled_during_construction():
    """An object constructed after the caller was cancelled is closed in a thread."""
    started = threading.Event()
    release = threading.Event()
    created = []

    def factory():
        resource = _Resource(started, release)
        created.append(resource)
        return resource

    task = asyncio.create_task(create_in_thread(factory))
    assert await asyncio.to_thread(started.wait, EVENT_TIMEOUT)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release.set()

    assert await asyncio.to_thread(lambda: created and created[0].closed.wait(EVENT_TIMEOUT))
    assert created[0].closed_in != threading.get_ident()


async def test_create_in_thread_cancelled_before_construction():
    """A construction that has not started when the caller is cancelled is skipped."""
    loop = asyncio.get_running_loop()
    # One busy worker keeps the construction queued until the caller is cancelled.
    executor = ThreadPoolExecutor(max_workers=1)
    loop.set_default_executor(executor)
    busy = threading.Event()
    calls = []
    try:
        blocker = loop.run_in_executor(None, busy.wait, EVENT_TIMEOUT)
        task = asyncio.create_task(create_in_thread(lambda: calls.append(1) or _Resource()))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        busy.set()
        await blocker
        # The queued construction runs after the blocker and returns without calling factory.
        await loop.run_in_executor(None, lambda: None)
        assert calls == []
    finally:
        busy.set()
        executor.shutdown(wait=True)
