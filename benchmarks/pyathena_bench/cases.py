# Copyright 2026 The PyAthena authors
#
# Licensed under the MIT License.
# See LICENSE or https://opensource.org/licenses/MIT.
#
# SPDX-License-Identifier: MIT

"""Explicit capability matrix, including unsupported combinations."""

from __future__ import annotations

from dataclasses import dataclass, replace

from pyathena_bench.config import Settings

FAMILIES = ("cursor", "dict", "s3fs", "pandas", "arrow", "polars")
# pandas.read_sql_query() through a SQLAlchemy engine of each PyAthena driver.
SQLALCHEMY_FAMILIES = ("sqla_rest", "sqla_pandas", "sqla_arrow", "sqla_polars", "sqla_s3fs")
SHAPES = ("flat", "nested", "json")
# The SQLAlchemy drivers that read result files and can read UNLOAD results.
RESULT_FILE_SQLALCHEMY = {"sqla_pandas", "sqla_arrow", "sqla_polars"}


@dataclass(frozen=True)
class Case:
    family: str
    api: str = "sync"
    transport: str = "csv"
    output: str = "rows"
    arraysize: int = 1000
    chunksize: int | None = None
    shape: str = "flat"
    concurrency: int = 1
    suite: str = "single"
    unsupported: str | None = None
    # The pandas.read_csv()/read_parquet() dtype_backend given to execute().
    dtype_backend: str | None = None

    @property
    def id(self) -> str:
        backend = f"-{self.dtype_backend}" if self.dtype_backend else ""
        return (
            f"{self.suite}-{self.family}-{self.api}-{self.transport}-{self.output}-"
            f"a{self.arraysize}-c{self.chunksize or 0}{backend}-{self.shape}-n{self.concurrency}"
        )


def matrix(settings: Settings, suite: str, shape: str) -> list[Case]:
    """Return every case of a suite and result shape, including unsupported ones.

    The ``json`` shape adds a ``json`` column to the flat projection and covers
    only the native pandas and Wrangler reads of CSV results: Parquet cannot
    hold the ``json`` type.

    Args:
        settings: Configuration with the arraysizes, chunk sizes, and concurrency.
        suite: ``single``, ``concurrent``, or ``init``.
        shape: ``flat``, ``nested``, or ``json``.

    Returns:
        The cases in a stable order.

    Raises:
        ValueError: If the suite or shape is unknown.
    """
    if suite not in {"single", "concurrent", "init"} or shape not in SHAPES:
        raise ValueError("Unknown suite or shape")
    cases: list[Case] = []
    arraysizes = settings.arraysizes if suite == "single" else [1000]
    for family in FAMILIES:
        for api in ("sync", "thread", "aio"):
            cases.extend(Case(family, api, arraysize=a, shape=shape) for a in arraysizes)
            if family in {"pandas", "arrow", "polars"}:
                for transport in ("csv", "unload"):
                    chunks: list[int | None] = (
                        [None] if family == "arrow" else [None, *settings.chunksizes]
                    )
                    for chunk in chunks:
                        cases.append(
                            Case(family, api, transport, "native", chunksize=chunk, shape=shape)
                        )
                        if family == "pandas" and transport == "unload":
                            cases.append(
                                Case(
                                    family,
                                    api,
                                    transport,
                                    "native",
                                    chunksize=chunk,
                                    shape=shape,
                                    dtype_backend="pyarrow",
                                )
                            )
                cases.extend(
                    Case(family, api, "unload", arraysize=a, shape=shape) for a in arraysizes
                )
    for transport in ("csv", "ctas", "unload"):
        for chunk in (None, *settings.chunksizes):
            reason = (
                "Wrangler CSV does not support nested types"
                if shape == "nested" and transport == "csv"
                else None
            )
            cases.append(
                Case(
                    "wrangler",
                    transport=transport,
                    output="native",
                    chunksize=chunk,
                    shape=shape,
                    unsupported=reason,
                )
            )
    if suite == "single":
        for family in SQLALCHEMY_FAMILIES:
            transports = ["csv", "unload"] if family in RESULT_FILE_SQLALCHEMY else ["csv"]
            cases.extend(
                Case(family, transport=transport, output="native", chunksize=chunk, shape=shape)
                for transport in transports
                for chunk in (None, *settings.chunksizes)
            )
    if shape == "json":
        cases = [
            c
            for c in cases
            if c.family in {"pandas", "wrangler"}
            and c.transport == "csv"
            and c.output == "native"
            and c.dtype_backend is None
        ]
    if suite == "concurrent":
        # Keep the scaling suite bounded; chunk and arraysize sweeps belong to single.
        cases = [
            replace(c, suite=suite, concurrency=n)
            for c in cases
            if c.api in {"thread", "aio"}
            and c.arraysize == 1000
            and c.chunksize is None
            and c.dtype_backend is None
            and c.transport == "csv"
            and (
                (c.family in {"cursor", "dict", "s3fs"} and c.output == "rows")
                or (c.family in {"pandas", "arrow", "polars"} and c.output == "native")
            )
            for n in settings.concurrency
        ]
    elif suite == "init":
        cases = [
            replace(c, suite=suite, api=api)
            for c in cases
            if c.api == "sync"
            and c.arraysize == 1000
            and c.chunksize is None
            and c.dtype_backend is None
            and c.family != "wrangler"
            and (
                (c.family in {"cursor", "dict", "s3fs"} and c.output == "rows")
                or (c.family in {"pandas", "arrow", "polars"} and c.output == "native")
            )
            for api in ("direct", "to_thread")
        ]
    return cases
