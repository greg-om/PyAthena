# Copyright 2026 The PyAthena authors
#
# Licensed under the MIT License.
# See LICENSE or https://opensource.org/licenses/MIT.
#
# SPDX-License-Identifier: MIT

import csv
import json

import pytest
from pyathena_bench.report import report, summarize


def test_summary_excludes_failures_and_warmups_but_reports_them():
    base = {
        "case_id": "test",
        "scale": "small",
        "status": "ok",
        "warmup": False,
        "queries": [{"total_seconds": 2}],
        "successful_queries_per_second": 0.5,
        "rss_peak_bytes": 100,
        "max_threads": 3,
        "loop_lag_seconds": [0.01],
    }
    trials = [
        base,
        {**base, "warmup": True, "queries": [{"total_seconds": 999}]},
        {**base, "status": "error", "queries": [{"status": "error", "error": "Access denied"}]},
        {
            "case_id": "unsupported",
            "scale": "small",
            "status": "unsupported",
            "reason": "Nested CSV",
        },
    ]
    rows = summarize(trials)
    assert rows[0]["median_seconds"] == 2
    assert rows[0]["successful_trials"] == rows[0]["failed_trials"] == 1
    assert rows[0]["notes"] == "Access denied"
    assert rows[1]["unsupported"]
    assert rows[1]["median_seconds"] is None


def test_summary_preserves_failed_warmup_and_cancellation_reasons():
    rows = summarize(
        [
            {
                "case_id": "case",
                "scale": "small",
                "status": "error",
                "warmup": True,
                "queries": [{"status": "row_count_mismatch", "rows": 2, "expected_rows": 10}],
                "cancellation_errors": ["q: StopQueryExecution denied"],
            }
        ]
    )
    assert rows[0]["failed_warmups"] == 1
    assert rows[0]["median_seconds"] is None
    assert "expected 10, got 2" in rows[0]["notes"]
    assert "Warmup: q: StopQueryExecution denied" in rows[0]["notes"]


class TestReport:
    def test_report_recovers_an_interrupted_trial_without_claiming_success(self, tmp_path):
        event = {
            "event": "trial_start",
            "trial": "interrupted",
            "case_id": "case",
            "scale": "small",
            "warmup": False,
        }
        (tmp_path / "events.jsonl").write_text(json.dumps(event) + "\n")
        (tmp_path / "trials.jsonl").write_text('{"trial":')
        with pytest.warns(UserWarning, match="incomplete final record"):
            report(tmp_path)
        with (tmp_path / "summary.csv").open() as stream:
            row = next(csv.DictReader(stream))
        assert row["failed_trials"] == "1"
        assert row["successful_trials"] == "0"
        assert row["median_seconds"] == ""

    def test_report_exposes_client_init_and_memory_increase(self, tmp_path):
        trial = {
            "case_id": "case",
            "scale": "small",
            "status": "ok",
            "queries": [
                {
                    "total_seconds": 20,
                    "post_completion_setup_seconds": 2,
                    "consume_seconds": 3,
                    "init_seconds": 1,
                }
            ],
            "successful_queries_per_second": 0.05,
            "rss_peak_bytes": 100,
            "rss_increase_bytes": 30,
            "max_threads": 3,
        }
        (tmp_path / "trials.jsonl").write_text(json.dumps(trial) + "\n")
        report(tmp_path)
        row = summarize([trial])[0]
        assert row["median_client_result_seconds"] == 5
        assert row["median_init_seconds"] == 1
        assert row["median_rss_increase_bytes"] == 30
        markdown = (tmp_path / "summary.md").read_text()
        assert "Client result (s)" in markdown
        assert "Init (s)" in markdown
        assert "RSS increase (B)" in markdown

    def test_report_shows_the_most_output_files_and_the_largest_file(self, tmp_path):
        base = {
            "case_id": "case",
            "scale": "small",
            "status": "ok",
            "queries": [{"total_seconds": 1}],
            "successful_queries_per_second": 1,
            "rss_peak_bytes": 1,
            "max_threads": 1,
        }
        trials = [
            {**base, "output_files": {"count": 30, "bytes": 60, "max_bytes": 4, "min_bytes": 1}},
            {**base, "output_files": {"count": 31, "bytes": 60, "max_bytes": 3, "min_bytes": 1}},
        ]
        (tmp_path / "trials.jsonl").write_text("".join(json.dumps(t) + "\n" for t in trials))
        report(tmp_path)
        row = summarize(trials)[0]
        assert (row["max_output_files"], row["max_output_file_bytes"]) == (31, 4)
        assert "| 31 | 4 |" in (tmp_path / "summary.md").read_text()
        assert summarize([base])[0]["max_output_files"] is None
        failed_listing = {**base, "output_files_error": "ClientError: denied"}
        row = summarize([trials[0], failed_listing])[0]
        assert (row["successful_trials"], row["max_output_files"]) == (2, 30)
        assert row["notes"] == "Output files not listed: ClientError: denied"
        # A warmup is not in the columns, so its listing failure is not noted either.
        row = summarize([trials[0], {**failed_listing, "warmup": True}])[0]
        assert row["notes"] == ""


def test_report_merges_attempts_and_drops_superseded_failures(tmp_path):
    def trial(repetition, status="ok"):
        return {
            "trial": f"t{repetition}-{status}",
            "case_id": "case",
            "scale": "small",
            "repetition": repetition,
            "warmup": False,
            "status": status,
            "queries": [{"total_seconds": 1}],
            "successful_queries_per_second": 1,
            "rss_peak_bytes": 1,
            "max_threads": 1,
        }

    first = tmp_path / "attempt-1"
    second = tmp_path / "attempt-2"
    first.mkdir()
    second.mkdir()
    (first / "trials.jsonl").write_text(
        json.dumps(trial(0)) + "\n" + json.dumps({**trial(1, "interrupted"), "queries": []}) + "\n"
    )
    # The interrupted attempt also started a trial that it never recorded.
    start = {"event": "trial_start", "trial": "lost", "case_id": "case", "scale": "small"}
    (first / "events.jsonl").write_text(json.dumps({**start, "repetition": 2}) + "\n")
    (second / "trials.jsonl").write_text(json.dumps(trial(1)) + "\n" + json.dumps(trial(2)) + "\n")
    report(tmp_path)
    with (tmp_path / "summary.csv").open() as stream:
        row = next(csv.DictReader(stream))
    assert (row["successful_trials"], row["failed_trials"]) == ("3", "0")
    # The superseded failures stay visible: the interrupted and the unrecorded trial.
    assert row["superseded_failures"] == "2"
    assert "Superseded: interrupted" in row["notes"]


def test_report_counts_a_repetition_recorded_by_two_attempts_once(tmp_path):
    record = {
        "case_id": "case",
        "scale": "small",
        "repetition": 0,
        "warmup": False,
        "status": "ok",
        "queries": [{"total_seconds": 1}],
        "successful_queries_per_second": 1,
        "rss_peak_bytes": 1,
        "max_threads": 1,
    }
    for attempt in (1, 2):
        directory = tmp_path / f"attempt-{attempt}"
        directory.mkdir()
        (directory / "trials.jsonl").write_text(
            json.dumps({**record, "trial": f"t{attempt}"}) + "\n"
        )
    report(tmp_path)
    with (tmp_path / "summary.csv").open() as stream:
        assert next(csv.DictReader(stream))["successful_trials"] == "1"
