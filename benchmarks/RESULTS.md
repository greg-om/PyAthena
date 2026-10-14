<!--
Copyright 2026 The PyAthena authors

Licensed under the MIT License.
See LICENSE or https://opensource.org/licenses/MIT.

SPDX-License-Identifier: MIT
-->

# Benchmark results

Each entry in `history/` records one run of the [benchmark harness](README.md), named by the run date and the short commit hash.
An entry lists the environment, the measured selection, the run record, findings, limits, and the result tables.
Compare entries only when they used the same instance type, architecture, and scales, because measurements depend on the host.

| Date | Commit | PyAthena | Hosts | Purchase | Scales | Entry |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-10-11 | `8a69ce03` | 4.0.0 | `r7g.2xlarge` (arm64), fleet of 12 | Spot | medium, large, xlarge | [2026-10-11-8a69ce03](history/2026-10-11-8a69ce03.md) |
| 2026-10-13 | `437276e1` | 4.1.0 release commit | `r7g.2xlarge` (arm64), fleet of 8 | Spot | medium, large, xlarge | [2026-10-13-437276e1](history/2026-10-13-437276e1.md) |
| 2026-10-13 | `437276e1` | 4.1.0 release commit, filesystem suite | `c8gn.2xlarge` (arm64), one host | Spot | small, medium, large, xlarge (objects) | [2026-10-13-437276e1-filesystem](history/2026-10-13-437276e1-filesystem.md) |

## Recommendations

These recommendations come from the runs of `437276e1` on `r7g.2xlarge` and `c8gn.2xlarge` hosts and the 4.0.0 run; the entries give the measured cases and their limits.

| Need | Use | Measured |
| --- | --- | --- |
| A DataFrame or Table of a large result | `unload=True` with ArrowCursor, PolarsCursor, or PandasCursor | 10 million flat rows in 18.6 to 19.2 seconds through UNLOAD, against 36.0 to 74.4 seconds through CSV; nested rows in 36.4 to 37.9 seconds with ArrowCursor, PolarsCursor, or PandasCursor with pyarrow dtypes, against 229.5 to 306.0 seconds through CSV |
| A pandas DataFrame with nested columns | PandasCursor with `unload=True` and `dtype_backend="pyarrow"` | 10 million nested rows in 37.7 seconds and 5,897 MiB, against 86.8 seconds and 19,664 MiB with the default dtypes |
| Bounded memory | `chunksize` with PandasCursor or PolarsCursor, with CSV or UNLOAD results | 10 million rows at 363 to 904 MiB with 10,000-row chunks, against 3,687 to 19,664 MiB without chunks; larger chunks use more memory |
| Python rows of a large result | ArrowCursor with `unload=True`, or S3FSCursor for constant memory | GetQueryResults paging through Cursor or DictCursor took 226 seconds for 1 million rows; ArrowCursor with `unload=True` returned 10 million flat rows in 27 seconds in the 4.0.0 run, and S3FSCursor in 329 seconds at 321 MiB |
| `pandas.read_sql_query()` | `awsathena+arrow` with `unload=true`, or `awsathena+polars` or `awsathena+pandas` with `unload=true` and `chunksize` for bounded memory | 1 million flat rows in 7.0 seconds through `awsathena+arrow`, against 230 seconds through the default `awsathena+rest` |
| A Wrangler-like pandas read | PandasCursor | 10 million flat rows in 19.0 seconds through UNLOAD and 74.4 seconds through CSV, against 21.4 and 95.3 seconds for Wrangler |
| Asynchronous code | Any of the synchronous, ThreadPool, or native asyncio APIs | Single-query timings differed by a median of 1.7% in the 4.0.0 run |
| Parallel S3 writes or copies with more workers than an event loop's default executor has threads, `min(32, os.cpu_count() + 4)` | `S3FileSystem` | `AioS3FileSystem` matched it for reads and below that worker count, and ran 40-worker writes at 0.75 to 0.83 times and copies at 0.30 to 0.35 times its throughput (#1159) |
