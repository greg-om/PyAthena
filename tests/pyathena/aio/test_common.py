# Copyright 2026 The PyAthena authors
#
# Licensed under the MIT License.
# See LICENSE or https://opensource.org/licenses/MIT.
#
# SPDX-License-Identifier: MIT
import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock

import pytest

from pyathena.aio.arrow.cursor import AioArrowCursor
from pyathena.aio.cursor import AioCursor, AioDictCursor
from pyathena.aio.pandas.cursor import AioPandasCursor
from pyathena.aio.polars.cursor import AioPolarsCursor
from pyathena.aio.s3fs.cursor import AioS3FSCursor
from pyathena.error import OperationalError, ProgrammingError
from tests.pyathena.util import EVENT_TIMEOUT

_ALL_CURSOR_CLASSES = [
    AioCursor,
    AioDictCursor,
    AioArrowCursor,
    AioPandasCursor,
    AioPolarsCursor,
    AioS3FSCursor,
]


def _result_set_closing_after_loop_responds() -> tuple[MagicMock, list[bool]]:
    """Return a result set whose ``close()`` waits for a callback on the running loop.

    Each close schedules a callback on the event loop, waits for it, and records
    whether it ran. A close on the event loop's thread blocks the loop, so the
    callback cannot run and the record is False.

    Returns:
        The mock result set, which is open until closed, and the list of records.
    """
    loop = asyncio.get_running_loop()
    closes: list[bool] = []

    def close():
        responded = threading.Event()
        loop.call_soon_threadsafe(responded.set)
        closes.append(responded.wait(EVENT_TIMEOUT))

    result_set = MagicMock(is_closed=False)
    result_set.close.side_effect = close
    return result_set, closes


class TestWithAsyncFetch:
    @pytest.mark.parametrize("cursor_class", _ALL_CURSOR_CLASSES)
    def test_sync_iteration_raises(self, cursor_class):
        cursor = cursor_class(
            connection=MagicMock(), converter=None, formatter=None, retry_config=None
        )
        with pytest.raises(
            TypeError, match=rf"'{cursor_class.__name__}' object is not iterable; use 'async for'"
        ):
            iter(cursor)

    @pytest.mark.parametrize(
        "cursor_class",
        [AioArrowCursor, AioPandasCursor, AioPolarsCursor, AioS3FSCursor],
    )
    async def test_fetch_runs_result_set_fetch(self, cursor_class):
        cursor = cursor_class(
            connection=MagicMock(), converter=None, formatter=None, retry_config=None
        )
        threads = []

        def fetch(rows):
            def record(*args):
                threads.append(threading.get_ident())
                return rows

            return record

        cursor.result_set = MagicMock()
        cursor.result_set.fetchone.side_effect = fetch((1,))
        cursor.result_set.fetchmany.side_effect = fetch([(2,), (3,)])
        cursor.result_set.fetchall.side_effect = fetch([(4,)])

        assert await cursor.fetchone() == (1,)
        assert await cursor.fetchmany(2) == [(2,), (3,)]
        assert await cursor.fetchall() == [(4,)]
        cursor.result_set.fetchmany.assert_called_once_with(2)
        # The synchronous fetches run off the event loop's thread.
        assert len(threads) == 3
        assert threading.get_ident() not in threads

    @pytest.mark.parametrize(
        "cursor_class",
        [AioArrowCursor, AioPandasCursor, AioPolarsCursor, AioS3FSCursor],
    )
    async def test_fetch_without_result_set(self, cursor_class):
        cursor = cursor_class(
            connection=MagicMock(), converter=None, formatter=None, retry_config=None
        )
        for fetch in (cursor.fetchone, cursor.fetchmany, cursor.fetchall):
            with pytest.raises(ProgrammingError, match=r"No result set\."):
                await fetch()

    @pytest.mark.parametrize("cursor_class", _ALL_CURSOR_CLASSES)
    async def test_async_iteration(self, cursor_class):
        cursor = cursor_class(
            connection=MagicMock(), converter=None, formatter=None, retry_config=None
        )
        cursor.result_set = MagicMock()
        # AioCursor's result set fetches asynchronously; the others run it in a thread.
        fetchone = AsyncMock if cursor_class in (AioCursor, AioDictCursor) else MagicMock
        cursor.result_set.fetchone = fetchone(side_effect=[(1,), (2,), None])

        rows = []
        async for row in cursor:
            rows.append(row)
            # Stop rather than loop forever if iteration never ends.
            if len(rows) > 2:
                break
        assert rows == [(1,), (2,)]

    @pytest.mark.parametrize("cursor_class", _ALL_CURSOR_CLASSES)
    async def test_context_exit_closes_off_event_loop(self, cursor_class):
        result_set, closes = _result_set_closing_after_loop_responds()
        async with cursor_class(
            connection=MagicMock(), converter=None, formatter=None, retry_config=None
        ) as cursor:
            cursor.result_set = result_set

        # The event loop ran a callback while the close waited.
        assert closes == [True]

    @pytest.mark.parametrize("cursor_class", _ALL_CURSOR_CLASSES)
    async def test_execute_closes_previous_result_set_off_event_loop(self, cursor_class):
        cursor = cursor_class(
            connection=MagicMock(), converter=None, formatter=None, retry_config=None
        )
        cursor.result_set, closes = _result_set_closing_after_loop_responds()
        cursor._execute = AsyncMock(side_effect=OperationalError("stop"))

        with pytest.raises(OperationalError, match="stop"):
            await cursor.execute("SELECT 1")

        assert closes == [True]
        assert cursor.result_set is None

    @pytest.mark.parametrize("cursor_class", _ALL_CURSOR_CLASSES)
    async def test_execute_cancelled_during_close_leaves_cursor_reset(self, cursor_class):
        cursor = cursor_class(
            connection=MagicMock(), converter=None, formatter=None, retry_config=None
        )
        closing = threading.Event()
        released = threading.Event()
        result_set = MagicMock(is_closed=False)
        result_set.close.side_effect = lambda: (closing.set(), released.wait(EVENT_TIMEOUT))
        cursor.result_set = result_set
        cursor.query_id = "previous"
        cursor._execute = AsyncMock(side_effect=OperationalError("stop"))

        task = asyncio.create_task(cursor.execute("SELECT 1"))
        assert await asyncio.to_thread(closing.wait, EVENT_TIMEOUT)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # The cursor no longer refers to the result set that is still closing.
        assert cursor.result_set is None
        assert cursor.query_id is None
        assert cursor.rowcount == -1
        released.set()
        cursor._execute.assert_not_called()

    @pytest.mark.parametrize("cursor_class", _ALL_CURSOR_CLASSES)
    async def test_executemany_closes_previous_result_set_off_event_loop(self, cursor_class):
        cursor = cursor_class(
            connection=MagicMock(), converter=None, formatter=None, retry_config=None
        )
        cursor.result_set, closes = _result_set_closing_after_loop_responds()

        await cursor.executemany("SELECT 1", [])

        assert closes == [True]
        assert cursor.result_set is None
        assert cursor.rowcount == 0

    @pytest.mark.parametrize("cursor_class", _ALL_CURSOR_CLASSES)
    async def test_executemany_failure_closes_result_set_off_event_loop(self, cursor_class):
        cursor = cursor_class(
            connection=MagicMock(), converter=None, formatter=None, retry_config=None
        )
        result_set, closes = _result_set_closing_after_loop_responds()
        result_set.rowcount = 1

        async def execute(operation, parameters, **kwargs):
            cursor.result_set = result_set

        def parameters():
            yield None
            raise ValueError("bad parameters")

        cursor.execute = execute
        with pytest.raises(ValueError, match="bad parameters"):
            await cursor.executemany("SELECT 1", parameters())

        assert closes == [True]
        assert cursor.result_set is None
