# Copyright 2026 The PyAthena authors
#
# Licensed under the MIT License.
# See LICENSE or https://opensource.org/licenses/MIT.
#
# SPDX-License-Identifier: MIT
import asyncio
import contextvars
import gc
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from pyathena.aio.util import create_in_thread
from tests.pyathena.util import EVENT_TIMEOUT

_VALUE: contextvars.ContextVar[str] = contextvars.ContextVar("_VALUE", default="unset")


class _Resource:
    """A closable object that records its constructor and close threads."""

    def __init__(self):
        self.created_in = threading.get_ident()
        self.value = _VALUE.get()
        self.closed = threading.Event()
        self.closed_in = None

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
    """An object constructed after the caller was cancelled is closed off the loop."""
    started = threading.Event()
    release = threading.Event()
    created = []

    def factory():
        resource = _Resource()
        created.append(resource)
        started.set()
        assert release.wait(EVENT_TIMEOUT)
        return resource

    task = asyncio.create_task(create_in_thread(factory))
    assert await asyncio.to_thread(started.wait, EVENT_TIMEOUT)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release.set()

    assert await asyncio.to_thread(created[0].closed.wait, EVENT_TIMEOUT)
    assert created[0].closed_in is not None
    assert created[0].closed_in != threading.get_ident()


async def test_create_in_thread_cancelled_construction_error():
    """A construction that fails after the caller was cancelled reports nothing."""
    loop = asyncio.get_running_loop()
    errors = []
    loop.set_exception_handler(lambda _, context: errors.append(context))
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def factory():
        started.set()
        assert release.wait(EVENT_TIMEOUT)
        finished.set()
        raise OSError("read failed")

    task = asyncio.create_task(create_in_thread(factory))
    assert await asyncio.to_thread(started.wait, EVENT_TIMEOUT)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release.set()
    assert await asyncio.to_thread(finished.wait, EVENT_TIMEOUT)
    # Let the executor future complete and be collected.
    for _ in range(3):
        await asyncio.sleep(0)
    await asyncio.to_thread(lambda: None)
    gc.collect()
    await asyncio.sleep(0)
    assert errors == []


async def test_create_in_thread_cancelled_after_construction(monkeypatch):
    """An object constructed before the cancellation reaches the caller is closed in a thread."""
    created = []

    def factory():
        resource = _Resource()
        created.append(resource)
        return resource

    async def cancelled_after(future):
        # The construction completes, then the caller is cancelled before it resumes.
        await future
        raise asyncio.CancelledError

    monkeypatch.setattr("pyathena.aio.util.asyncio.shield", cancelled_after)
    with pytest.raises(asyncio.CancelledError):
        await create_in_thread(factory)

    assert await asyncio.to_thread(created[0].closed.wait, EVENT_TIMEOUT)
    assert created[0].closed_in is not None
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


async def test_create_in_thread_cleanup_after_executor_shutdown():
    """An abandoned object is closed even if the default executor shuts down meanwhile."""
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=1)
    loop.set_default_executor(executor)
    started = threading.Event()
    release = threading.Event()
    created = []

    def factory():
        resource = _Resource()
        created.append(resource)
        started.set()
        assert release.wait(EVENT_TIMEOUT)
        return resource

    with ThreadPoolExecutor(max_workers=1) as waiter:
        try:
            task = asyncio.create_task(create_in_thread(factory))
            assert await loop.run_in_executor(waiter, started.wait, EVENT_TIMEOUT)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            # As at the end of asyncio.run(), while the construction still runs.
            executor.shutdown(wait=False)
            release.set()
            assert await loop.run_in_executor(waiter, created[0].closed.wait, EVENT_TIMEOUT)
        finally:
            release.set()
            executor.shutdown(wait=True)
