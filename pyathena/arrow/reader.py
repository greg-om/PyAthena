# Copyright 2026 The PyAthena authors
#
# Licensed under the MIT License.
# See LICENSE or https://opensource.org/licenses/MIT.
#
# SPDX-License-Identifier: MIT

"""Reader of Parquet files in chunks of a fixed number of rows."""

from __future__ import annotations

from collections import abc, deque
from collections.abc import Generator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from pyathena.util import override

if TYPE_CHECKING:
    from fsspec import AbstractFileSystem
    from pyarrow import Table


class ParquetChunkReader(abc.Iterator):  # type: ignore[type-arg]
    """Iterator over Parquet files as Arrow tables of a fixed number of rows.

    The files are decoded one at a time, in the given order, with
    ``pyarrow.parquet.ParquetFile.iter_batches()``, without pre-buffering the row
    groups. Memory use therefore depends on the chunk size and the column chunks
    being decoded, not on the size of the result or its row groups. Every table
    has ``chunksize`` rows except the last, which has the remaining rows; rows
    carry over from one file to the next.

    A file of at most ``MAX_PREFETCH_SIZE`` bytes is read whole with one
    ``cat_file()`` call, and the next file is read in a background thread while
    the current one is decoded. Larger files are read in ``BUFFER_SIZE`` pieces
    through ``open()``.

    The files must have the same schema, as the files of one UNLOAD query do.

    Example:
        >>> with fs.open(manifest) as f:
        ...     paths = f.read().decode().split()
        >>> with ParquetChunkReader(fs, paths, chunksize=100_000) as reader:
        ...     for table in reader:
        ...         process(table)
    """

    #: Bytes that pyarrow reads at once from a column chunk of a larger file.
    BUFFER_SIZE: int = 8 * 2**20
    #: The largest file that is read whole, ahead of decoding.
    MAX_PREFETCH_SIZE: int = 32 * 2**20

    def __init__(
        self,
        fs: AbstractFileSystem,
        paths: Sequence[str],
        chunksize: int,
        columns: Sequence[str] | None = None,
    ) -> None:
        """Create a reader of the given files.

        Args:
            fs: The filesystem that opens the files.
            paths: The paths of the files, in reading order.
            chunksize: The number of rows of each table but the last.
            columns: The names of the top-level columns to read; all columns if None.

        Raises:
            ValueError: If ``chunksize`` is not positive or ``columns`` is empty.
        """
        if chunksize < 1:
            raise ValueError(f"chunksize must be positive, got {chunksize}.")
        if columns is not None and not columns:
            raise ValueError("columns must not be empty.")
        self._fs = fs
        self._paths = list(paths)
        self._chunksize = chunksize
        self._columns = list(columns) if columns is not None else None
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
        """Read the files and yield tables of ``chunksize`` rows.

        Yields:
            The tables of rows, the last with the remaining rows.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        batches: list[pa.RecordBatch] = []
        rows = 0
        paths = iter(self._paths)
        sources: deque[Future[Any]] = deque()
        with ThreadPoolExecutor(max_workers=1) as executor:
            try:
                if (path := next(paths, None)) is not None:
                    sources.append(executor.submit(self._open, path))
                while sources:
                    source = sources.popleft().result()
                    # Read the next file while this one is decoded.
                    if (path := next(paths, None)) is not None:
                        sources.append(executor.submit(self._open, path))
                    with source:
                        parquet_file = pq.ParquetFile(
                            source, pre_buffer=False, buffer_size=self.BUFFER_SIZE
                        )
                        for batch in parquet_file.iter_batches(
                            batch_size=self._chunksize, columns=self._columns
                        ):
                            if batch.num_rows == 0:
                                continue
                            batches.append(batch)
                            rows += batch.num_rows
                            while rows >= self._chunksize:
                                table = pa.Table.from_batches(batches)
                                yield table.slice(0, self._chunksize)
                                rest = table.slice(self._chunksize)
                                batches = rest.to_batches()
                                rows = rest.num_rows
            finally:
                for future in sources:
                    if not future.cancel() and future.exception() is None:
                        future.result().close()
        if rows:
            yield pa.Table.from_batches(batches)

    def _open(self, path: str) -> Any:
        """Open a file for ``ParquetFile``.

        Args:
            path: The path of the file.

        Returns:
            A buffer of the whole file if it has at most ``MAX_PREFETCH_SIZE``
            bytes, or else the file opened for reading.
        """
        import pyarrow as pa

        if self._fs.size(path) <= self.MAX_PREFETCH_SIZE:
            return pa.BufferReader(self._fs.cat_file(path))
        return self._fs.open(path, "rb")
