# Copyright 2026 The PyAthena authors
#
# Licensed under the MIT License.
# See LICENSE or https://opensource.org/licenses/MIT.
#
# SPDX-License-Identifier: MIT

import gc
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, wait
from unittest.mock import MagicMock

import pyarrow as pa
import pyarrow.csv as pa_csv
import pytest

from pyathena.filesystem.reader import ReadAheadStream
from pyathena.filesystem.s3 import S3FileSystem
from tests import ENV
from tests.pyathena.conftest import connect
from tests.pyathena.util import EVENT_TIMEOUT, ImmediateExecutor


class _RecordingRandomAccessFile:
    """A random-access file over bytes that records its reads."""

    def __init__(self, data, short=False, error_at=None, gate=None):
        self._data = data
        self._short = short
        self._error_at = error_at
        self._gate = gate
        self.gated_read_started = threading.Event()
        self.reads = []
        self.closed = False

    def size(self):
        return len(self._data)

    def read_at(self, nbytes, offset):
        self.reads.append((offset, nbytes))
        if self._gate is not None and offset > 0:
            self.gated_read_started.set()
            assert self._gate.wait(EVENT_TIMEOUT)
        if offset == self._error_at:
            raise OSError("read failed")
        return self._data[offset : offset + nbytes - (1 if self._short else 0)]

    def close(self):
        self.closed = True


class _InterruptedFuture(Future):
    """A future whose first wait for the outcome is interrupted."""

    def __init__(self):
        super().__init__()
        self.interrupted = False

    def exception(self, timeout=None):
        if not self.interrupted:
            self.interrupted = True
            raise KeyboardInterrupt
        return super().exception(timeout)


class _InterruptedWaitExecutor(ImmediateExecutor):
    """An executor that runs each call when it is submitted, whose first wait for the
    outcome of the first call is interrupted."""

    def __init__(self, max_workers=None):
        super().__init__(max_workers)
        self.submitted = False

    def _new_future(self):
        if self.submitted:
            return super()._new_future()
        self.submitted = True
        return _InterruptedFuture()


class TestReadAheadStream:
    """Tests for reading a file sequentially with parallel range reads."""

    @pytest.mark.parametrize("size", [0, 1, 6, 7, 21, 25])
    @pytest.mark.parametrize("read_size", [1, 5, 7, 100])
    def test_read(self, size, read_size):
        data = bytes(range(256))[:size]
        file = _RecordingRandomAccessFile(data)
        stream = ReadAheadStream.from_arrow_file(file, part_size=7, max_parts=2)
        chunks = []
        while chunk := stream.read(read_size):
            # Every read is full except at the end of the file.
            assert len(chunk) == read_size or sum(map(len, chunks)) + len(chunk) == size
            chunks.append(chunk)
        assert b"".join(chunks) == data
        assert stream.tell() == size
        assert sorted(file.reads) == [(o, min(7, size - o)) for o in range(0, size, 7)]
        stream.close()
        assert file.closed

    def test_read_range(self):
        """A range-read function is called with the offset and size of each part."""
        data = bytes(range(25))
        ranges = []
        closed = []

        def read_range(offset, size):
            ranges.append((offset, size))
            return data[offset : offset + size]

        stream = ReadAheadStream(
            read_range, len(data), part_size=10, max_parts=2, on_close=lambda: closed.append(1)
        )
        assert stream.read() == data
        assert sorted(ranges) == [(0, 10), (10, 10), (20, 5)]
        stream.close()
        stream.close()
        assert closed == [1]

    def test_default_part_size(self, monkeypatch):
        monkeypatch.setattr("pyathena.filesystem.reader.ThreadPoolExecutor", ImmediateExecutor)
        monkeypatch.setattr(ReadAheadStream, "DEFAULT_PART_SIZE", 10)
        monkeypatch.setattr(ReadAheadStream, "DEFAULT_MAX_PARTS", 2)
        file = _RecordingRandomAccessFile(bytes(100))
        stream = ReadAheadStream.from_arrow_file(file)
        assert stream.read(1) == bytes(1)
        assert file.reads == [(0, 10), (10, 10), (20, 10)]
        stream.close()

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"part_size": 0}, "part_size must be positive, got 0."),
            ({"max_parts": 0}, "max_parts must be positive, got 0."),
        ],
    )
    @pytest.mark.filterwarnings("error::pytest.PytestUnraisableExceptionWarning")
    def test_invalid_options(self, kwargs, message):
        closed = []
        with pytest.raises(ValueError, match=message):
            ReadAheadStream(lambda o, s: b"", 10, on_close=lambda: closed.append(1), **kwargs)
        # The stream that failed to initialize is closed without an error when it is
        # finalized, and without calling on_close.
        gc.collect()
        assert closed == []
        file = _RecordingRandomAccessFile(bytes(10))
        with pytest.raises(ValueError, match=message):
            ReadAheadStream.from_arrow_file(file, **kwargs)
        # The file is closed when the stream cannot be created.
        assert file.closed

    def test_from_arrow_file_size_error(self):
        file = MagicMock()
        file.size.side_effect = OSError("HEAD failed")
        with pytest.raises(OSError, match="HEAD failed"):
            ReadAheadStream.from_arrow_file(file)
        file.close.assert_called_once_with()

    def test_parts_in_flight(self, monkeypatch):
        """At most max_parts parts are requested ahead of the one being consumed."""
        monkeypatch.setattr("pyathena.filesystem.reader.ThreadPoolExecutor", ImmediateExecutor)
        file = _RecordingRandomAccessFile(bytes(100))
        stream = ReadAheadStream.from_arrow_file(file, part_size=10, max_parts=3)
        assert file.reads == []
        assert stream.read(1) == bytes(1)
        # Three parts were requested, the first is being consumed, and one more follows.
        assert file.reads == [(0, 10), (10, 10), (20, 10), (30, 10)]
        assert stream.read(9) == bytes(9)
        assert len(file.reads) == 4
        stream.close()

    def test_short_read(self):
        stream = ReadAheadStream.from_arrow_file(
            _RecordingRandomAccessFile(bytes(20), short=True), part_size=10, max_parts=2
        )
        with pytest.raises(OSError, match="Read 9 bytes of a range of 10 bytes"):
            stream.read(20)
        stream.close()

    def test_read_error(self):
        """An error is raised by the read that reaches the failed part."""
        file = _RecordingRandomAccessFile(bytes(30), error_at=10)
        stream = ReadAheadStream.from_arrow_file(file, part_size=10, max_parts=2)
        assert stream.read(10) == bytes(10)
        with pytest.raises(OSError, match="read failed"):
            stream.read(10)
        # A later read does not continue past the failed part.
        with pytest.raises(OSError, match="read failed"):
            stream.read(10)
        stream.close()
        assert file.closed

    def test_close_before_end(self):
        """Closing cancels the queued reads and closes the file."""
        file = _RecordingRandomAccessFile(bytes(1000))
        stream = ReadAheadStream.from_arrow_file(file, part_size=10, max_parts=2)
        assert stream.read(5) == bytes(5)
        stream.close()
        assert file.closed
        assert stream.closed
        assert len(file.reads) <= 3
        stream.close()
        with pytest.raises(ValueError, match="I/O operation on closed file"):
            stream.read(5)

    def test_interrupted_wait(self, monkeypatch):
        """An interrupt while waiting for a part leaves the part to the next read."""
        monkeypatch.setattr(
            "pyathena.filesystem.reader.ThreadPoolExecutor", _InterruptedWaitExecutor
        )
        data = bytes(range(30))
        file = _RecordingRandomAccessFile(data)
        stream = ReadAheadStream.from_arrow_file(file, part_size=10, max_parts=2)
        with pytest.raises(KeyboardInterrupt):
            stream.read(10)
        assert stream.read(10) == data[:10]
        assert stream.read(5) == data[10:15]
        assert sorted(file.reads) == [(0, 10), (10, 10), (20, 10)]
        stream.close()

    def test_interrupted_close(self, monkeypatch):
        """An interrupted close leaves the file open, and closing again waits for the
        running reads before closing the file."""
        gate = threading.Event()
        file = _RecordingRandomAccessFile(bytes(100), gate=gate)
        stream = ReadAheadStream.from_arrow_file(file, part_size=10, max_parts=2)
        interrupts = [KeyboardInterrupt()]

        def interrupted_wait(futures, *args, **kwargs):
            if interrupts:
                raise interrupts.pop()
            return wait(futures, *args, **kwargs)

        monkeypatch.setattr("pyathena.filesystem.reader.wait", interrupted_wait)
        closer = threading.Thread(target=stream.close, daemon=True)
        try:
            assert stream.read(10) == bytes(10)
            assert file.gated_read_started.wait(EVENT_TIMEOUT)
            with pytest.raises(KeyboardInterrupt):
                stream.close()
            assert not file.closed
            assert not stream.closed
            closer.start()
            closer.join(0.2)
            # A read of a later part is still blocked, so the retried close has not
            # closed the file.
            assert closer.is_alive()
            assert not file.closed
        finally:
            gate.set()
            if closer.is_alive():
                closer.join(EVENT_TIMEOUT)
        assert file.closed
        assert stream.closed

    def test_close_with_queued_reads(self, monkeypatch):
        """Closing cancels a part that no worker has started and does not wait for it."""
        monkeypatch.setattr(
            "pyathena.filesystem.reader.ThreadPoolExecutor",
            lambda max_workers: ThreadPoolExecutor(max_workers=1),
        )
        gate = threading.Event()
        file = _RecordingRandomAccessFile(bytes(100), gate=gate)
        stream = ReadAheadStream.from_arrow_file(file, part_size=10, max_parts=2)
        # The one worker blocks in the read of the second part, and the third waits.
        assert stream.read(10) == bytes(10)
        assert file.gated_read_started.wait(EVENT_TIMEOUT)
        closer = threading.Thread(target=stream.close, daemon=True)
        closer.start()
        closer.join(0.2)
        assert closer.is_alive()
        gate.set()
        closer.join(EVENT_TIMEOUT)
        assert not closer.is_alive()
        assert file.closed
        assert (20, 10) not in file.reads

    def test_close_waits_for_running_reads(self):
        """Closing waits for the reads in flight before closing the file."""
        gate = threading.Event()
        file = _RecordingRandomAccessFile(bytes(100), gate=gate)
        stream = ReadAheadStream.from_arrow_file(file, part_size=10, max_parts=2)
        closer = threading.Thread(target=stream.close)
        try:
            assert stream.read(10) == bytes(10)
            assert file.gated_read_started.wait(EVENT_TIMEOUT)
            closer.start()
            closer.join(0.2)
            # A read of a later part is still blocked, so close() has not returned.
            assert closer.is_alive()
            assert not file.closed
        finally:
            gate.set()
            if closer.is_alive():
                closer.join(EVENT_TIMEOUT)
        assert file.closed

    def test_csv(self):
        """pyarrow reads a CSV through the stream with its own block size."""
        data = b"a,b\n" + b"".join(f"{i},x{i}\n".encode() for i in range(10_000))
        file = pa.BufferReader(data)
        read_options = pa_csv.ReadOptions(block_size=4096)
        with pa.PythonFile(ReadAheadStream.from_arrow_file(file, 7, 3), mode="r") as source:
            table = pa_csv.read_csv(source, read_options=read_options)
        assert table.equals(pa_csv.read_csv(pa.BufferReader(data), read_options=read_options))
        assert file.closed

    def test_from_s3_invalid_options(self):
        fs = MagicMock()
        with pytest.raises(ValueError, match="part_size must be positive"):
            ReadAheadStream.from_s3(fs, "s3://bucket/key", part_size=0)
        # The object is not opened.
        fs.open.assert_not_called()

    @pytest.mark.parametrize(
        ("kwargs", "max_workers"),
        [
            ({}, 1),
            ({"max_workers": None}, 1),
            ({"max_workers": 4}, 4),
            ({"cache_type": "all", "cache_options": {"trim": False}}, 1),
        ],
        ids=["default", "none", "given", "cache_type"],
    )
    def test_from_s3_open_options(self, kwargs, max_workers):
        """The object is opened without a cache and with one worker, which reads each
        part with one request, unless the number of workers is given."""
        fs = MagicMock()
        fs.open.return_value.size = 0
        ReadAheadStream.from_s3(fs, "s3://bucket/key", part_size=10, **kwargs).close()
        fs.open.assert_called_once_with(
            "s3://bucket/key", "rb", max_workers=max_workers, cache_type="none"
        )
        fs.open.return_value.close.assert_called_once_with()

    @pytest.fixture(scope="class")
    def fs(self):
        return S3FileSystem(connect(), skip_instance_cache=True)

    @pytest.fixture
    def path(self, fs, request):
        path = (
            f"s3://{ENV.s3_staging_bucket}/{ENV.s3_staging_key}{ENV.schema}/"
            f"filesystem/{request.node.originalname}/{uuid.uuid4()}"
        )
        yield path
        if fs.exists(path):
            fs.rm(path)

    def test_from_s3(self, fs, path):
        data = bytes(range(256)) * 4
        fs.pipe_file(path, data)
        requests = []

        def record(params, **_):
            requests.append(params["Range"])

        events = fs.core.client.meta.events
        events.register("before-parameter-build.s3.GetObject", record)
        try:
            with ReadAheadStream.from_s3(fs, path, part_size=100, max_parts=3) as stream:
                assert stream.read() == data
        finally:
            events.unregister("before-parameter-build.s3.GetObject", record)
        # One request for every part, each as large as the part.
        assert sorted(requests, key=lambda r: int(r[6:].split("-")[0])) == [
            f"bytes={o}-{min(o + 100, len(data)) - 1}" for o in range(0, len(data), 100)
        ]

    def test_from_s3_overwritten(self, fs, path):
        """Parts are read from the object as it was when the stream was created."""
        fs.pipe_file(path, b"a" * 100)
        with ReadAheadStream.from_s3(fs, path, part_size=10, max_parts=1) as stream:
            assert stream.read(10) == b"a" * 10
            fs.pipe_file(path, b"b" * 100)
            with pytest.raises(OSError, match="pre-conditions"):
                stream.read()

    def test_from_s3_missing(self, fs, path):
        with pytest.raises(FileNotFoundError):
            ReadAheadStream.from_s3(fs, path)
