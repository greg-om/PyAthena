# Copyright 2026 The PyAthena authors
#
# Licensed under the MIT License.
# See LICENSE or https://opensource.org/licenses/MIT.
#
# SPDX-License-Identifier: MIT

from concurrent.futures import Future

import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.parquet as pq
import pytest
from fsspec.implementations.local import LocalFileSystem

from pyathena.arrow.reader import ParquetChunkReader, ReadAheadStream


class _ImmediateExecutor:
    """An executor that runs each call when it is submitted."""

    def __init__(self, max_workers=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def shutdown(self, wait=True, cancel_futures=False):
        pass

    def submit(self, fn, *args, **kwargs):
        future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as e:
            future.set_exception(e)
        return future


class TestParquetChunkReader:
    """Tests for reading Parquet files in tables of a fixed number of rows."""

    @pytest.fixture(params=["prefetch", "stream"])
    def max_prefetch_size(self, request, monkeypatch):
        """Read the files whole ahead of decoding, or open them for streaming."""
        if request.param == "stream":
            monkeypatch.setattr(ParquetChunkReader, "MAX_PREFETCH_SIZE", -1)
        return request.param

    @pytest.fixture
    def paths(self, tmp_path):
        """Three files of 3, 0, and 7 rows; the last has row groups of 2 rows."""
        paths = []
        for index, (start, rows, row_group_size) in enumerate(
            [(0, 3, None), (3, 0, None), (3, 7, 2)]
        ):
            path = tmp_path / f"{index}.parquet"
            table = pa.table(
                {
                    "id": pa.array(range(start, start + rows), pa.int64()),
                    "name": pa.array(
                        [f"name-{i}" for i in range(start, start + rows)], pa.string()
                    ),
                }
            )
            pq.write_table(table, path, row_group_size=row_group_size)
            paths.append(str(path))
        return paths

    @pytest.mark.parametrize(
        ("chunksize", "expected"),
        [(1, [1] * 10), (4, [4, 4, 2]), (5, [5, 5]), (10, [10]), (100, [10])],
    )
    def test_chunk_sizes(self, paths, max_prefetch_size, chunksize, expected):
        with ParquetChunkReader(LocalFileSystem(), paths, chunksize) as reader:
            tables = list(reader)
        assert [t.num_rows for t in tables] == expected
        combined = pa.concat_tables(tables)
        assert combined.column("id").to_pylist() == list(range(10))
        assert combined.column("name").to_pylist() == [f"name-{i}" for i in range(10)]

    def test_columns(self, paths, max_prefetch_size):
        with ParquetChunkReader(LocalFileSystem(), paths, 4, columns=["name"]) as reader:
            tables = list(reader)
        assert all(t.column_names == ["name"] for t in tables)
        assert pa.concat_tables(tables).column("name").to_pylist() == [
            f"name-{i}" for i in range(10)
        ]

    @pytest.mark.parametrize("columns", [["missing"], ["id", "missing"], ["name.x"]])
    def test_columns_not_top_level(self, paths, max_prefetch_size, columns):
        with (
            ParquetChunkReader(LocalFileSystem(), paths, 1, columns=columns) as reader,
            pytest.raises(ValueError, match="No top-level columns"),
        ):
            next(reader)

    @pytest.mark.parametrize("corrupt", [False, True], ids=["close", "error"])
    def test_releases_files_opened_ahead(self, paths, max_prefetch_size, monkeypatch, corrupt):
        """Files opened ahead are closed when reading stops early or fails."""
        if corrupt:
            # The third file fails while the files after it may be opened ahead.
            with open(paths[2], "rb") as f:
                valid = f.read()
            with open(paths[2], "wb") as f:
                f.write(valid[: len(valid) // 2] + b"\x00" * 64 + valid[-8:])
        paths = [*paths, paths[0], paths[0]]
        opened = []
        original_open = ParquetChunkReader._open

        def open_(self, path):
            source = original_open(self, path)
            opened.append(source)
            return source

        monkeypatch.setattr(ParquetChunkReader, "_open", open_)
        # Open ahead at once, so that a file opened ahead is never cancelled.
        monkeypatch.setattr("pyathena.arrow.reader.ThreadPoolExecutor", _ImmediateExecutor)
        reader = ParquetChunkReader(LocalFileSystem(), paths, 2)
        if corrupt:
            with pytest.raises((OSError, ValueError)):
                list(reader)
        else:
            next(reader)
            reader.close()
        # The file after the last one received was opened ahead.
        assert len(opened) == (4 if corrupt else 2)
        assert all(source.closed for source in opened)

    def test_open_failure_opens_no_further_file(self, paths, monkeypatch):
        """A file that fails to open raises before the next file is opened."""
        opened = []

        def open_(self, path):
            opened.append(path)
            raise OSError(f"cannot open {path}")

        monkeypatch.setattr(ParquetChunkReader, "_open", open_)
        # Open at once, so that an open submitted too early is not cancelled.
        monkeypatch.setattr("pyathena.arrow.reader.ThreadPoolExecutor", _ImmediateExecutor)
        with pytest.raises(OSError, match="cannot open"):
            next(ParquetChunkReader(LocalFileSystem(), paths, 2))
        assert opened == paths[:1]

    def test_no_files(self):
        assert list(ParquetChunkReader(LocalFileSystem(), [], 4)) == []

    @pytest.mark.parametrize("chunksize", [0, -1])
    def test_invalid_chunksize(self, paths, chunksize):
        with pytest.raises(ValueError, match="chunksize must be positive"):
            ParquetChunkReader(LocalFileSystem(), paths, chunksize)

    def test_close(self, paths, max_prefetch_size, monkeypatch):
        fs = LocalFileSystem()
        opened = []
        original_open = fs.open

        def open_(path, mode="rb", **kwargs):
            f = original_open(path, mode, **kwargs)
            opened.append(f)
            return f

        monkeypatch.setattr(fs, "open", open_)
        reader = ParquetChunkReader(fs, paths, 2)
        assert next(reader).num_rows == 2
        reader.close()
        assert opened
        assert all(f.closed for f in opened)
        assert list(reader) == []


class _RecordingFile:
    """A random-access file over bytes that records its reads."""

    def __init__(self, data, short=False, error_at=None):
        self._data = data
        self._short = short
        self._error_at = error_at
        self.reads = []
        self.closed = False

    def size(self):
        return len(self._data)

    def read_at(self, nbytes, offset):
        self.reads.append((offset, nbytes))
        if offset == self._error_at:
            raise OSError("read failed")
        return self._data[offset : offset + nbytes - (1 if self._short else 0)]

    def close(self):
        self.closed = True


class TestReadAheadStream:
    """Tests for reading a random-access file sequentially with parallel range reads."""

    @pytest.mark.parametrize("size", [0, 1, 6, 7, 21, 25])
    @pytest.mark.parametrize("read_size", [1, 5, 7, 100])
    def test_read(self, size, read_size):
        data = bytes(range(256))[:size]
        file = _RecordingFile(data)
        stream = ReadAheadStream(file, part_size=7, max_parts=2)
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

    def test_parts_in_flight(self, monkeypatch):
        """At most max_parts parts are requested ahead of the one being consumed."""
        monkeypatch.setattr("pyathena.arrow.reader.ThreadPoolExecutor", _ImmediateExecutor)
        file = _RecordingFile(bytes(100))
        stream = ReadAheadStream(file, part_size=10, max_parts=3)
        assert file.reads == []
        assert stream.read(1) == bytes(1)
        # Three parts were requested, the first is being consumed, and one more follows.
        assert file.reads == [(0, 10), (10, 10), (20, 10), (30, 10)]
        assert stream.read(9) == bytes(9)
        assert len(file.reads) == 4
        stream.close()

    def test_short_read(self):
        stream = ReadAheadStream(_RecordingFile(bytes(20), short=True), part_size=10, max_parts=2)
        with pytest.raises(OSError, match="Read 9 bytes of a range of 10 bytes"):
            stream.read(20)
        stream.close()

    def test_read_error(self):
        """An error is raised by the read that reaches the failed part."""
        file = _RecordingFile(bytes(30), error_at=10)
        stream = ReadAheadStream(file, part_size=10, max_parts=2)
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
        file = _RecordingFile(bytes(1000))
        stream = ReadAheadStream(file, part_size=10, max_parts=2)
        assert stream.read(5) == bytes(5)
        stream.close()
        assert file.closed
        assert stream.closed
        assert len(file.reads) <= 3
        stream.close()

    def test_csv(self):
        """pyarrow reads a CSV through the stream with its own block size."""
        data = b"a,b\n" + b"".join(f"{i},x{i}\n".encode() for i in range(10_000))
        file = pa.BufferReader(data)
        read_options = pa_csv.ReadOptions(block_size=4096)
        with pa.PythonFile(ReadAheadStream(file, 7, 3), mode="r") as source:
            table = pa_csv.read_csv(source, read_options=read_options)
        assert table.equals(pa_csv.read_csv(pa.BufferReader(data), read_options=read_options))
        assert file.closed
