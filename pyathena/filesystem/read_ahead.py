# Copyright 2026 The PyAthena authors
#
# Licensed under the MIT License.
# See LICENSE or https://opensource.org/licenses/MIT.
#
# SPDX-License-Identifier: MIT

"""Sequential reading of a file with parallel range reads ahead of the consumer."""

from __future__ import annotations

import io
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, cast

from pyathena.util import override

if TYPE_CHECKING:
    from pyarrow import NativeFile

    from pyathena.filesystem.s3 import S3FileSystem


class ReadAheadStream(io.RawIOBase):
    """Sequential stream over a file that reads ranges ahead in parallel.

    The file is read in parts of ``part_size`` bytes with a range-read function,
    and up to ``max_parts`` parts are read in a thread pool ahead of the consumer,
    so the transfer is not limited to one request at a time. Parts are returned in
    order, and at most ``max_parts`` parts plus the one being consumed are held in
    memory. A read that returns fewer bytes than requested raises ``OSError``, and
    an error from the range-read function is raised by the read that reaches its
    part; every later read raises the same error.

    Closing the stream cancels the queued reads, waits for the running ones, and
    then calls ``on_close``.

    Example:
        >>> stream = ReadAheadStream.from_arrow_file(fs.open_input_file(path))
        >>> with pyarrow.PythonFile(stream, mode="r") as f:
        ...     table = pyarrow.csv.read_csv(f)
    """

    #: Bytes requested by one range read unless ``part_size`` is given.
    DEFAULT_PART_SIZE: int = 8 * 2**20
    #: The most parts read ahead at once unless ``max_parts`` is given.
    DEFAULT_MAX_PARTS: int = 16

    def __init__(
        self,
        read_range: Callable[[int, int], bytes],
        size: int,
        part_size: int | None = None,
        max_parts: int | None = None,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        """Initialize the stream without reading the file.

        Args:
            read_range: A function that reads ``size`` bytes of the file from
                ``offset``, called as ``read_range(offset, size)`` from several
                threads at once.
            size: The size of the file in bytes.
            part_size: Bytes requested by one ``read_range()`` call; if None,
                ``DEFAULT_PART_SIZE``.
            max_parts: The most parts read ahead of the consumer at once; if
                None, ``DEFAULT_MAX_PARTS``.
            on_close: A function called once when the stream is closed, after the
                running reads have finished, such as the ``close()`` of the file.

        Raises:
            ValueError: If ``part_size`` or ``max_parts`` is not positive.
        """
        super().__init__()
        # Set before the checks, for the close() that finalizing a stream that
        # failed to initialize calls; on_close is not called then.
        self._executor: ThreadPoolExecutor | None = None
        self._on_close: Callable[[], None] | None = None
        self._parts: deque[tuple[int, Future[bytes]]] = deque()
        self._buffer = memoryview(b"")
        part_size = self.DEFAULT_PART_SIZE if part_size is None else part_size
        max_parts = self.DEFAULT_MAX_PARTS if max_parts is None else max_parts
        if part_size < 1:
            raise ValueError(f"part_size must be positive, got {part_size}.")
        if max_parts < 1:
            raise ValueError(f"max_parts must be positive, got {max_parts}.")
        self._read_range = read_range
        self._size = size
        self._part_size = part_size
        self._max_parts = max_parts
        self._next_offset = 0
        self._position = 0
        self._error: BaseException | None = None
        self._executor = ThreadPoolExecutor(max_workers=max_parts)
        self._on_close = on_close

    @classmethod
    def from_arrow_file(
        cls, file: NativeFile, part_size: int | None = None, max_parts: int | None = None
    ) -> ReadAheadStream:
        """Read a pyarrow random-access file with ``read_at()``.

        Args:
            file: The file to read, such as one that
                ``pyarrow.fs.S3FileSystem.open_input_file()`` opened, whose
                ``read_at()`` may be called from several threads at once. The
                stream closes it when closed, or here if the stream cannot be
                created.
            part_size: Bytes requested by one ``read_at()`` call; if None,
                ``DEFAULT_PART_SIZE``.
            max_parts: The most parts read ahead at once; if None,
                ``DEFAULT_MAX_PARTS``.

        Returns:
            The stream over the file.
        """
        try:
            return cls(
                lambda offset, size: file.read_at(size, offset),
                file.size(),
                part_size,
                max_parts,
                file.close,
            )
        except BaseException:
            file.close()
            raise

    @classmethod
    def from_s3(
        cls,
        fs: S3FileSystem,
        path: str,
        part_size: int | None = None,
        max_parts: int | None = None,
        **kwargs: Any,
    ) -> ReadAheadStream:
        """Read an S3 object through PyAthena's S3 filesystem.

        The object is opened with ``fs.open()``, which looks it up once, and its
        parts are read with GetObject requests of the opened file: a version or
        ETag observed when opening, and the request parameters given in
        ``kwargs``, apply to every part.

        Args:
            fs: The filesystem that opens the object.
            path: The S3 path (s3://bucket/key) of the object.
            part_size: Bytes requested by one GetObject request; if None,
                ``DEFAULT_PART_SIZE``.
            max_parts: The most parts read ahead at once; if None,
                ``DEFAULT_MAX_PARTS``.
            **kwargs: Additional arguments passed to ``fs.open()``, such as
                ``version_id`` or ``s3_additional_kwargs``.

        Returns:
            The stream over the object.

        Raises:
            FileNotFoundError: If the object does not exist.
        """
        part_size = cls.DEFAULT_PART_SIZE if part_size is None else part_size
        # A part no larger than the block size is read with one request, not split
        # into parallel ones by the file.
        kwargs.setdefault("block_size", part_size)
        file = fs.open(path, "rb", **kwargs)
        try:
            return cls(
                lambda offset, size: file._fetch_range(offset, offset + size),
                file.size,
                part_size,
                max_parts,
                file.close,
            )
        except BaseException:
            file.close()
            raise

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
            if self._executor is not None:
                self._executor.shutdown(wait=True, cancel_futures=True)
            self._parts.clear()
            self._buffer = memoryview(b"")
        finally:
            try:
                if self._on_close is not None:
                    self._on_close()
            finally:
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
        executor = cast(ThreadPoolExecutor, self._executor)
        while len(self._parts) < self._max_parts and self._next_offset < self._size:
            size = min(self._part_size, self._size - self._next_offset)
            part = executor.submit(self._read_range, self._next_offset, size)
            self._parts.append((size, part))
            self._next_offset += size
