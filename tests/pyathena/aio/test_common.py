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

_ALL_CURSOR_CLASSES = [
    AioCursor,
    AioDictCursor,
    AioArrowCursor,
    AioPandasCursor,
    AioPolarsCursor,
    AioS3FSCursor,
]


def _result_set_closing_after(released: threading.Event, closes: list[tuple[int, bool]]):
    """Return a result set whose ``close()`` waits until ``released`` is set.

    Each close records its thread and whether ``released`` was set within
    five seconds. A close on the event loop's thread cannot observe a
    callback scheduled on that loop, so it times out.

    Args:
        released: Event that a callback on the event loop sets.
        closes: List that receives one ``(thread_id, released)`` record per close.

    Returns:
        A mock result set that is open until closed.
    """
    result_set = MagicMock(is_closed=False)
    result_set.close.side_effect = lambda: closes.append((threading.get_ident(), released.wait(5)))
    return result_set


class TestWithAsyncFetch:
    @pytest.mark.parametrize(
        "cursor_class",
        [
            AioCursor,
            AioDictCursor,
            AioArrowCursor,
            AioPandasCursor,
            AioPolarsCursor,
            AioS3FSCursor,
        ],
    )
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

    @pytest.mark.parametrize(
        "cursor_class",
        [
            AioCursor,
            AioDictCursor,
            AioArrowCursor,
            AioPandasCursor,
            AioPolarsCursor,
            AioS3FSCursor,
        ],
    )
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
        released = threading.Event()
        closes: list[tuple[int, bool]] = []
        async with cursor_class(
            connection=MagicMock(), converter=None, formatter=None, retry_config=None
        ) as cursor:
            cursor.result_set = _result_set_closing_after(released, closes)
            asyncio.get_running_loop().call_soon(released.set)

        assert len(closes) == 1
        thread, released_during_close = closes[0]
        assert thread != threading.get_ident()
        # The event loop ran the callback while the close waited.
        assert released_during_close

    @pytest.mark.parametrize("cursor_class", _ALL_CURSOR_CLASSES)
    async def test_execute_closes_previous_result_set_off_event_loop(self, cursor_class):
        cursor = cursor_class(
            connection=MagicMock(), converter=None, formatter=None, retry_config=None
        )
        released = threading.Event()
        closes: list[tuple[int, bool]] = []
        cursor.result_set = _result_set_closing_after(released, closes)
        cursor._execute = AsyncMock(side_effect=OperationalError("stop"))
        asyncio.get_running_loop().call_soon(released.set)

        with pytest.raises(OperationalError, match="stop"):
            await cursor.execute("SELECT 1")

        assert len(closes) == 1
        thread, released_during_close = closes[0]
        assert thread != threading.get_ident()
        assert released_during_close
        assert cursor.result_set is None

    @pytest.mark.parametrize("cursor_class", _ALL_CURSOR_CLASSES)
    async def test_executemany_closes_previous_result_set_off_event_loop(self, cursor_class):
        cursor = cursor_class(
            connection=MagicMock(), converter=None, formatter=None, retry_config=None
        )
        released = threading.Event()
        closes: list[tuple[int, bool]] = []
        cursor.result_set = _result_set_closing_after(released, closes)
        asyncio.get_running_loop().call_soon(released.set)

        await cursor.executemany("SELECT 1", [])

        assert len(closes) == 1
        thread, released_during_close = closes[0]
        assert thread != threading.get_ident()
        assert released_during_close
        assert cursor.result_set is None
        assert cursor.rowcount == 0

    @pytest.mark.parametrize("cursor_class", _ALL_CURSOR_CLASSES)
    async def test_executemany_failure_closes_result_set_off_event_loop(self, cursor_class):
        cursor = cursor_class(
            connection=MagicMock(), converter=None, formatter=None, retry_config=None
        )
        released = threading.Event()
        closes: list[tuple[int, bool]] = []

        async def execute(operation, parameters, **kwargs):
            cursor.result_set = _result_set_closing_after(released, closes)
            cursor.result_set.rowcount = 1

        def parameters():
            yield None
            asyncio.get_running_loop().call_soon(released.set)
            raise ValueError("bad parameters")

        cursor.execute = execute
        with pytest.raises(ValueError, match="bad parameters"):
            await cursor.executemany("SELECT 1", parameters())

        assert len(closes) == 1
        thread, released_during_close = closes[0]
        assert thread != threading.get_ident()
        assert released_during_close
        assert cursor.result_set is None
