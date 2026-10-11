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
| 2026-10-11 | `8a69ce03` | 4.0.0 | `r7g.2xlarge` (arm64), 12 hosts | Spot | medium, large, xlarge | [2026-10-11-8a69ce03](history/2026-10-11-8a69ce03.md) |
