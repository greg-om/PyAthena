# Copyright 2026 The PyAthena authors
#
# Licensed under the MIT License.
# See LICENSE or https://opensource.org/licenses/MIT.
#
# SPDX-License-Identifier: MIT

"""S3 transfers through PyAthena's synchronous and asyncio filesystems."""

from __future__ import annotations

import asyncio
import time
from functools import partial
from typing import Any

from botocore.config import Config

from pyathena import connect
from pyathena.filesystem.s3 import S3FileSystem
from pyathena.filesystem.s3_async import AioS3FileSystem
from pyathena.util import parse_output_location
from pyathena_bench.adapters import heartbeat
from pyathena_bench.cases import Case
from pyathena_bench.config import Settings


def read_object(fs: Any, path: str, block_size: int) -> int:
    """Read a whole object in one call, as parallel range requests of ``block_size``.

    Args:
        fs: The filesystem.
        path: Object to read.
        block_size: Size of each range request.

    Returns:
        The number of bytes read.
    """
    with fs.open(path, "rb", block_size=block_size) as stream:
        return len(stream.read())


def write_object(fs: Any, path: str, size: int, block_size: int) -> int:
    """Write an object of zero bytes as a multipart upload with parts of ``block_size``.

    Args:
        fs: The filesystem.
        path: Object to write.
        size: Size of the object in bytes.
        block_size: Size of each part, and of the zero bytes written at a time.

    Returns:
        The number of bytes written.
    """
    block = memoryview(bytes(block_size))
    with fs.open(path, "wb", block_size=block_size) as stream:
        remaining = size
        while remaining:
            remaining -= stream.write(block[: min(remaining, block_size)])
    return size


async def transfer(
    fs: Any, case: Case, source: str, destination: str, size: int, block_size: int
) -> int:
    """Run the case's operation in the way its API is used.

    The synchronous filesystem runs on the calling thread. As ``AioS3FSCursor``
    reads result files, the asyncio filesystem's file reads and writes run in
    ``asyncio.to_thread()``: its files dispatch their range requests and part
    uploads through ``S3AioExecutor`` to the filesystem's event loop. Its
    copy is a coroutine awaited on the trial's event loop.

    Args:
        fs: ``S3FileSystem`` for the ``sync`` API, ``AioS3FileSystem`` for ``aio``.
        case: Filesystem case; the transport is the operation, ``read``,
            ``write``, or ``copy``.
        source: Object to read or copy.
        destination: Object to write or copy to.
        size: Size of the object in bytes.
        block_size: Size of each range request, part, or copied part.

    Returns:
        The number of bytes read, written, or copied.
    """
    if case.transport == "copy":
        options = {"max_workers": case.max_workers, "block_size": block_size}
        if case.api == "aio":
            await fs._cp_file(source, destination, **options)
        else:
            fs.cp_file(source, destination, **options)
        return size
    operation = (
        partial(read_object, fs, source, block_size)
        if case.transport == "read"
        else partial(write_object, fs, destination, size, block_size)
    )
    if case.api == "aio":
        return await asyncio.to_thread(operation)
    return operation()


async def run_filesystem(
    settings: Settings, case: Case, session_: Any, source: str, destination: str, size: int
) -> dict[str, Any]:
    """Measure one transfer of a filesystem case in the running trial process.

    The event loop keeps Python's default executor, so ``asyncio.to_thread()``
    runs with its default number of threads. The filesystem shares a
    connection's S3 client, whose connection pool has the case's size.

    Args:
        settings: Configuration with the block size and region.
        case: Filesystem case.
        session_: boto3 session.
        source: Prepared object of ``size`` bytes, read or copied.
        destination: Object that a write or copy creates and the trial deletes
            after checking its size.
        size: Size of the object in bytes.

    Returns:
        A trial result with one transfer in ``queries``; its status is
        ``size_mismatch`` if the transferred or resulting size differs.

    Raises:
        ValueError: If the case has no max_workers or connection pool size.
    """
    if case.max_workers is None or case.pool_connections is None:
        raise ValueError("A filesystem case needs max_workers and pool_connections")
    setup_start = time.perf_counter()
    connection = connect(
        session=session_,
        region_name=settings.region,
        work_group=settings.workgroup,
        config=Config(connect_timeout=10, read_timeout=60),
        s3_config=Config(max_pool_connections=case.pool_connections),
    )
    filesystem_class = AioS3FileSystem if case.api == "aio" else S3FileSystem
    fs = filesystem_class(
        connection=connection, max_workers=case.max_workers, skip_instance_cache=True
    )
    samples: list[float] = []
    stop = asyncio.Event()
    monitor = asyncio.create_task(heartbeat(settings.heartbeat_interval, stop, samples))
    setup_seconds = time.perf_counter() - setup_start
    query: dict[str, Any]
    try:
        await asyncio.sleep(0)
        start = time.perf_counter()
        try:
            transferred = await transfer(
                fs, case, source, destination, size, settings.filesystem_block_size
            )
            elapsed = time.perf_counter() - start
            # Checked after the measurement: the written or copied object is complete.
            if case.transport != "read":
                bucket, key = parse_output_location(destination)
                transferred = connection.s3_client.head_object(Bucket=bucket, Key=key)[
                    "ContentLength"
                ]
                # Removed, so that the trials do not accumulate copies of the object.
                connection.s3_client.delete_object(Bucket=bucket, Key=key)
        except Exception as exc:
            query = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "total_seconds": time.perf_counter() - start,
            }
        else:
            query = {
                "status": "ok" if transferred == size else "size_mismatch",
                "bytes": transferred,
                "expected_bytes": size,
                "total_seconds": elapsed,
            }
    finally:
        stop.set()
        await monitor
        connection.close()
    return {
        "status": "ok" if query["status"] == "ok" else "error",
        "setup_seconds": setup_seconds,
        "batch_seconds": query["total_seconds"],
        "queries": [query],
        "successful_queries_per_second": (query["status"] == "ok") / query["total_seconds"],
        "loop_lag_seconds": samples if case.api == "aio" else None,
    }
