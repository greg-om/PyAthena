# Copyright 2026 The PyAthena authors
#
# Licensed under the MIT License.
# See LICENSE or https://opensource.org/licenses/MIT.
#
# SPDX-License-Identifier: MIT

"""Readers of the result files of the Arrow cursor."""

from __future__ import annotations

import io
from collections import abc, deque
from collections.abc import Generator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import closing
from typing import TYPE_CHECKING, Any, cast

from pyathena.util import override

if TYPE_CHECKING:
    from fsspec import AbstractFileSystem
    from pyarrow import NativeFile, RecordBatch, Table


class ParquetChunkReader(abc.Iterator):  # type: ignore[type-arg]
    """Iterator over Parquet files as Arrow tables of a fixed number of rows.

    The files are decoded one at a time, in the given order, one row group at a
    time, with ``pyarrow.parquet.ParquetFile.iter_batches()``. Memory use
    therefore depends on the chunk size and on one row group of the file being
    read, not on the size of the result. Every table has ``chunksize`` rows except
    the last, which has the remaining rows; rows carry over from one row group and
    file to the next.

    A file of at most ``MAX_PREFETCH_SIZE`` bytes is read whole with one
    ``cat_file()`` call, and the next file is read in a background thread while
    the current one is decoded. A larger file is opened with ``open()``, by default
    without the filesystem's read-ahead cache, and the column chunks of each row
    group are pre-buffered: pyarrow reads them in large coalesced ranges, which
    PyAthena's filesystem splits into parallel requests, and holds them until the
    row group is decoded.

    The files must have the same schema, as the files of one UNLOAD query do.

    Example:
        >>> with fs.open(manifest) as f:
        ...     paths = f.read().decode().split()
        >>> with ParquetChunkReader(fs, paths, chunksize=100_000) as reader:
        ...     for table in reader:
        ...         process(table)
    """

    #: The largest file that is read whole, ahead of decoding.
    MAX_PREFETCH_SIZE: int = 32 * 2**20

    def __init__(
        self,
        fs: AbstractFileSystem,
        paths: Sequence[str],
        chunksize: int,
        columns: Sequence[str] | None = None,
        cache_type: str | None = None,
    ) -> None:
        """Create a reader of the given files.

        Args:
            fs: The filesystem that opens the files.
            paths: The paths of the files, in reading order.
            chunksize: The number of rows of each table but the last.
            columns: The names of the top-level columns to read; all columns if None.
            cache_type: The fsspec cache type that opens a file larger than
                ``MAX_PREFETCH_SIZE``. If None, ``"none"``, which reads only the
                ranges that pyarrow requests.

        Raises:
            ValueError: If ``chunksize`` is not positive or ``columns`` is empty.
                Reading raises ``ValueError`` as well when a file has no top-level
                column of a name in ``columns``.
        """
        if chunksize < 1:
            raise ValueError(f"chunksize must be positive, got {chunksize}.")
        if columns is not None and not columns:
            raise ValueError("columns must not be empty.")
        self._fs = fs
        self._paths = list(paths)
        self._chunksize = chunksize
        self._columns = list(columns) if columns is not None else None
        self._cache_type = cache_type or "none"
        self._tables = self._read_tables()

    @override
    def __next__(self) -> Table:
        """Read the next table.

        Returns:
            The next ``chunksize`` rows, or the remaining rows at the end.

        Raises:
            StopIteration: When every row has been read.
        """
        return next(self._tables)

    @override
    def __iter__(self) -> ParquetChunkReader:
        """Return the reader itself."""
        return self

    def __enter__(self) -> ParquetChunkReader:
        """Return the reader, which the context closes."""
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Close the reader."""
        self.close()

    def close(self) -> None:
        """Stop reading and close the file being read; later calls read nothing."""
        self._tables.close()

    def _read_tables(self) -> Generator[Table]:
        """Cut the record batches of the files into tables of ``chunksize`` rows.

        Yields:
            The tables of rows, the last with the remaining rows.
        """
        import pyarrow as pa

        pending: list[RecordBatch] = []
        rows = 0
        with closing(self._read_batches()) as batches:
            for batch in batches:
                pending.append(batch)
                rows += batch.num_rows
                while rows >= self._chunksize:
                    table = pa.Table.from_batches(pending)
                    yield table.slice(0, self._chunksize)
                    rest = table.slice(self._chunksize)
                    pending, rows = rest.to_batches(), rest.num_rows
        if rows:
            yield pa.Table.from_batches(pending)

    def _read_batches(self) -> Generator[RecordBatch]:
        """Read the record batches of the files, in order, without empty batches.

        Yields:
            The record batches of at most ``chunksize`` rows.
        """
        with closing(self._open_files()) as sources:
            for source in sources:
                with source:
                    yield from self._read_file(source)

    def _read_file(self, source: Any) -> Generator[RecordBatch]:
        """Read the record batches of one file.

        Args:
            source: The file, as ``_open()`` opened it.

        Yields:
            The non-empty record batches of at most ``chunksize`` rows.

        Raises:
            ValueError: If the file has no top-level column of a name in ``columns``.
        """
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(source, pre_buffer=True)
        if self._columns is not None and (
            missing := set(self._columns) - set(parquet_file.schema_arrow.names)
        ):
            # pyarrow would select nested fields by dotted paths, or no column, whose
            # tables keep their rows when sliced.
            raise ValueError(f"No top-level columns {sorted(missing)}.")
        for row_group in range(parquet_file.num_row_groups):
            # One row group at a time, so that only its column chunks are pre-buffered.
            for batch in parquet_file.iter_batches(
                batch_size=self._chunksize, row_groups=[row_group], columns=self._columns
            ):
                if batch.num_rows:
                    yield batch

    def _open_files(self) -> Generator[Any]:
        """Open the files in order, each next file in a background thread.

        The receiver closes each file it receives. A file opened ahead and not
        received, because reading stopped or failed, is closed here.

        Yields:
            The files, as ``_open()`` opens them.
        """
        if not self._paths:
            return
        # One worker opens the one file read ahead: at most the file being decoded and
        # the next one are held, which bounds memory to two files.
        with ThreadPoolExecutor(max_workers=1) as executor:
            ahead: Future[Any] | None = executor.submit(self._open, self._paths[0])
            try:
                for next_path in [*self._paths[1:], None]:
                    # Open the next file only after this one opened.
                    source = cast("Future[Any]", ahead).result()
                    ahead = None if next_path is None else executor.submit(self._open, next_path)
                    yield source
            finally:
                if ahead is not None and not ahead.cancel() and ahead.exception() is None:
                    ahead.result().close()

    def _open(self, path: str) -> Any:
        """Open a file for ``ParquetFile``.

        Args:
            path: The path of the file.

        Returns:
            A buffer of the whole file if it has at most ``MAX_PREFETCH_SIZE``
            bytes, or else the file opened for reading with the cache type.
        """
        import pyarrow as pa

        if self._fs.size(path) <= self.MAX_PREFETCH_SIZE:
            return pa.BufferReader(self._fs.cat_file(path))
        return self._fs.open(path, "rb", cache_type=self._cache_type)


class ReadAheadStream(io.RawIOBase):
    """Sequential stream over a random-access file that reads ranges ahead in parallel.

    The file is read in parts of ``part_size`` bytes with ``read_at()``, and up
    to ``max_parts`` parts are read in a thread pool ahead of the consumer, so
    the transfer is not limited to one request at a time. Parts are returned in
    order, and at most ``max_parts`` parts plus the one being consumed are held
    in memory. A read that returns fewer bytes than requested raises
    ``OSError``, and an error from ``read_at()`` is raised by the read that
    reaches its part; every later read raises the same error.

    Closing the stream cancels the queued reads, waits for the running ones,
    and closes the file.

    Example:
        >>> stream = ReadAheadStream(fs.open_input_file(path), 8 * 2**20, 16)
        >>> with pyarrow.PythonFile(stream, mode="r") as f:
        ...     table = pyarrow.csv.read_csv(f)
    """

    def __init__(self, file: NativeFile, part_size: int, max_parts: int) -> None:
        """Initialize the stream without reading the file.

        Args:
            file: The file to read, whose ``read_at()`` may be called from
                several threads at once, as pyarrow's random-access files allow.
            part_size: Bytes requested by one ``read_at()`` call.
            max_parts: The most parts read ahead of the consumer at once.
        """
        super().__init__()
        self._file = file
        self._part_size = part_size
        self._max_parts = max_parts
        self._executor = ThreadPoolExecutor(max_workers=max_parts)
        self._parts: deque[tuple[int, Future[bytes]]] = deque()
        self._next_offset = 0
        self._position = 0
        self._buffer = memoryview(b"")
        self._error: BaseException | None = None
        # Last, so that close() works if it fails.
        self._size: int = file.size()

    @override
    def readable(self) -> bool:
        return True

    @override
    def tell(self) -> int:
        return self._position

    @override
    def readinto(self, buffer: Any) -> int:  # type: ignore[override]
        """Fill ``buffer`` from the parts read ahead.

        Args:
            buffer: A writable buffer.

        Returns:
            The number of bytes read, less than the size of ``buffer`` only at
            the end of the file.
        """
        view = memoryview(buffer).cast("B")
        filled = 0
        while filled < len(view):
            if not self._buffer and not self._next_part():
                break
            size = min(len(view) - filled, len(self._buffer))
            view[filled : filled + size] = self._buffer[:size]
            self._buffer = self._buffer[size:]
            filled += size
        self._position += filled
        return filled

    @override
    def close(self) -> None:
        if self.closed:
            return
        try:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._parts.clear()
            self._buffer = memoryview(b"")
        finally:
            self._file.close()
            super().close()

    def _next_part(self) -> bool:
        """Wait for the next part and make it the buffer being consumed.

        Returns:
            False at the end of the file.

        Raises:
            OSError: If the part has fewer bytes than requested.
        """
        if self._error is not None:
            raise self._error
        self._read_ahead()
        if not self._parts:
            return False
        expected, part = self._parts.popleft()
        try:
            data = part.result()
            if len(data) != expected:
                raise OSError(f"Read {len(data)} bytes of a range of {expected} bytes.")
        except BaseException as e:
            # The part is gone, so a later read must not continue past it.
            self._error = e
            raise
        self._buffer = memoryview(data)
        self._read_ahead()
        return True

    def _read_ahead(self) -> None:
        """Start reading parts until ``max_parts`` are in flight or the file is read."""
        while len(self._parts) < self._max_parts and self._next_offset < self._size:
            size = min(self._part_size, self._size - self._next_offset)
            part = self._executor.submit(self._file.read_at, size, self._next_offset)
            self._parts.append((size, part))
            self._next_offset += size
