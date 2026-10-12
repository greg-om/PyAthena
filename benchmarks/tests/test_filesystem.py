# Copyright 2026 The PyAthena authors
#
# Licensed under the MIT License.
# See LICENSE or https://opensource.org/licenses/MIT.
#
# SPDX-License-Identifier: MIT

import asyncio
import io
import threading
from types import SimpleNamespace

import pytest
from fsspec.implementations.memory import MemoryFileSystem
from pyathena_bench.cases import Case
from pyathena_bench.config import Settings
from pyathena_bench.filesystem import (
    read_object,
    run_filesystem,
    transfer,
    unsupported_reason,
    write_object,
)


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
    """Records the thread of each file and the arguments of each copy."""

    def __init__(self):
        self.threads = []
        self.copies = []

    def open(self, path, mode, block_size):
        self.threads.append(threading.get_ident())
        return io.BytesIO()

    def cp_file(self, path1, path2, **kwargs):
        self.copies.append(("cp_file", path1, path2, kwargs))

    async def _cp_file(self, path1, path2, **kwargs):
        # Recorded only when the coroutine runs.
        self.copies.append(("_cp_file", path1, path2, kwargs))


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
    fs = RecordingFileSystem()
    case = Case("s3", api, operation, "bytes", suite="filesystem", max_workers=8)

    async def run():
        await transfer(fs, case, "s3://b/source", "s3://b/destination", 0, memoryview(bytes(1)))
        return threading.get_ident()

    loop_thread = asyncio.run(run())
    assert (fs.threads == [loop_thread]) is on_loop_thread


@pytest.mark.parametrize(("api", "method"), [("sync", "cp_file"), ("aio", "_cp_file")])
def test_copy_passes_the_case_workers_and_block_size(api, method):
    fs = RecordingFileSystem()
    case = Case("s3", api, "copy", "bytes", suite="filesystem", max_workers=8)
    copied = asyncio.run(
        transfer(fs, case, "s3://b/source", "s3://b/destination", 20, memoryview(bytes(5)))
    )
    assert copied == 20
    assert fs.copies == [
        (method, "s3://b/source", "s3://b/destination", {"max_workers": 8, "block_size": 5})
    ]


@pytest.mark.parametrize(
    ("operation", "size", "runs"),
    [
        ("copy", 5 * 2**30, False),
        ("copy", 5 * 2**30 + 1, True),
        ("read", 2**20, True),
        ("write", 2**20, True),
    ],
)
def test_only_multipart_copies_run(operation, size, runs):
    case = Case("s3", "sync", operation, "bytes", suite="filesystem", max_workers=8)
    assert (unsupported_reason(case, size) is None) is runs


class FakeS3Client:
    def __init__(self, size, head_error=None):
        self.size = size
        self.head_error = head_error
        self.deleted = []

    def head_object(self, **params):
        if self.head_error:
            raise self.head_error
        return {"ContentLength": self.size}

    def delete_object(self, **params):
        self.deleted.append((params["Bucket"], params["Key"]))


class TestRunFilesystem:
    @pytest.fixture
    def s3_client(self, monkeypatch):
        """Replace the connection and filesystems; the fake client answers HeadObject."""
        client = FakeS3Client(10)
        connection = SimpleNamespace(s3_client=client, close=lambda: None)
        monkeypatch.setattr("pyathena_bench.filesystem.connect", lambda **kwargs: connection)
        for name in ("S3FileSystem", "AioS3FileSystem"):
            monkeypatch.setattr(
                f"pyathena_bench.filesystem.{name}", lambda **kwargs: RecordingFileSystem()
            )
        return client

    def run(self, operation):
        case = Case(
            "s3", "aio", operation, "bytes", suite="filesystem", max_workers=8, pool_connections=10
        )
        return asyncio.run(
            run_filesystem(Settings(), case, None, "s3://b/source", "s3://b/out/object", 10)
        )

    def test_written_object_is_checked_and_deleted(self, s3_client):
        result = self.run("write")
        assert result["status"] == "ok"
        assert result["queries"][0]["bytes"] == 10
        assert s3_client.deleted == [("b", "out/object")]

    def test_short_object_is_a_size_mismatch(self, s3_client):
        s3_client.size = 9
        result = self.run("copy")
        assert result["status"] == "error"
        assert result["queries"][0]["status"] == "size_mismatch"
        assert result["queries"][0]["bytes"] == 9

    def test_failed_size_check_is_recorded_as_the_transfer_error(self, s3_client):
        s3_client.head_error = RuntimeError("Forbidden")
        result = self.run("write")
        assert result["queries"][0]["status"] == "error"
        assert result["queries"][0]["error"] == "RuntimeError: Forbidden"
        assert "total_seconds" in result["queries"][0]
