# Copyright 2026 The PyAthena authors
#
# Licensed under the MIT License.
# See LICENSE or https://opensource.org/licenses/MIT.
#
# SPDX-License-Identifier: MIT

from concurrent.futures import Future

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fsspec.implementations.local import LocalFileSystem

from pyathena.arrow.reader import ParquetChunkReader


class _ImmediateExecutor:
    """An executor that runs each call when it is submitted."""

    def __init__(self, max_workers=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
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
