# Copyright 2026 The PyAthena authors
#
# Licensed under the MIT License.
# See LICENSE or https://opensource.org/licenses/MIT.
#
# SPDX-License-Identifier: MIT

import asyncio
import io
import threading

import pytest
from fsspec.implementations.memory import MemoryFileSystem
from pyathena_bench.cases import Case
from pyathena_bench.filesystem import read_object, transfer, write_object

from pyathena.filesystem.s3_path import S3Path


@pytest.fixture
def memory_fs():
    fs = MemoryFileSystem(skip_instance_cache=True)
    fs.store = {}
    fs.pseudo_dirs = [""]
    return fs


def test_written_object_has_the_size_in_blocks_and_reads_back_whole(memory_fs):
    block = memoryview(bytes(4))
    assert write_object(memory_fs, "/bucket/object", 10, block) == 10
    assert memory_fs.cat_file("/bucket/object") == bytes(10)
    assert read_object(memory_fs, "/bucket/object", 4) == 10


class RecordingFileSystem:
    """Records the thread of each operation, and the arguments of the multipart copy."""

    def __init__(self, asynchronous_copy):
        self.asynchronous_copy = asynchronous_copy
        self.threads = []
        self.copies = []

    def open(self, path, mode, block_size):
        self.threads.append(threading.get_ident())
        return io.BytesIO()

    def _copy_object_with_multipart_upload(self, source, destination, **kwargs):
        if not self.asynchronous_copy:
            self.copies.append((source, destination, kwargs))
            return None

        async def copy():
            # Recorded only when the coroutine runs.
            self.copies.append((source, destination, kwargs))

        return copy()


@pytest.mark.parametrize(
    ("api", "operation", "on_loop_thread"),
    [
        ("sync", "read", True),
        ("sync", "write", True),
        ("aio", "read", False),
        ("aio", "write", False),
    ],
)
def test_aio_file_transfers_run_in_a_thread_as_the_aio_cursor_reads(api, operation, on_loop_thread):
    fs = RecordingFileSystem(asynchronous_copy=False)
    case = Case("s3", api, operation, "bytes", suite="filesystem", max_workers=8)

    async def run():
        await transfer(fs, case, "s3://b/source", "s3://b/destination", 0, memoryview(bytes(1)))
        return threading.get_ident()

    loop_thread = asyncio.run(run())
    assert (fs.threads == [loop_thread]) is on_loop_thread


@pytest.mark.parametrize("api", ["sync", "aio"])
def test_copy_is_a_multipart_copy_with_the_case_workers_and_block_size(api):
    fs = RecordingFileSystem(asynchronous_copy=api == "aio")
    case = Case("s3", api, "copy", "bytes", suite="filesystem", max_workers=8)
    copied = asyncio.run(
        transfer(fs, case, "s3://b/source", "s3://b/destination", 20, memoryview(bytes(5)))
    )
    assert copied == 20
    assert fs.copies == [
        (
            S3Path("b", "source", None),
            S3Path("b", "destination", None),
            {"max_workers": 8, "block_size": 5},
        )
    ]
