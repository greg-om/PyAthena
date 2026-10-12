# Copyright 2026 The PyAthena authors
#
# Licensed under the MIT License.
# See LICENSE or https://opensource.org/licenses/MIT.
#
# SPDX-License-Identifier: MIT

import json
import os
import time
from dataclasses import replace
from pathlib import Path

import pytest
from pyathena_bench.cases import Case
from pyathena_bench.config import Settings
from pyathena_bench.runner import Observer, output_files, run, supervise


def successful_worker(pipe, payload):
    pipe.send({"event": "ready"})
    pipe.recv()
    time.sleep(0.05)
    pipe.send({"event": "result", "result": {"status": "ok"}})
    pipe.close()


def waiting_worker(pipe, payload):
    time.sleep(30)


def cache_writing_worker(pipe, payload):
    pipe.send({"event": "ready"})
    pipe.recv()
    cache = Path(os.environ["POLARS_TEMP_DIR"]) / "file-cache"
    cache.mkdir(parents=True)
    (cache / "object").write_bytes(b"x" * 4096)
    time.sleep(0.2)
    result = {"status": "ok", "temp_dir": os.environ["POLARS_TEMP_DIR"]}
    pipe.send({"event": "result", "result": result})
    pipe.close()


def test_observer_keeps_first_completion_and_query_id():
    events = []

    class Pipe:
        def send(self, value):
            events.append(value)

    observer = Observer(Pipe())
    context = {}
    observer.before_start(
        {"QueryString": "SELECT 1", "ResultConfiguration": {"OutputLocation": "s3://out/"}}, context
    )
    observer.started({"QueryExecutionId": "q"}, context)
    response = {
        "QueryExecution": {
            "QueryExecutionId": "q",
            "Status": {"State": "SUCCEEDED"},
            "Statistics": {"DataScannedInBytes": 123},
        }
    }
    observer.polled(response)
    observer.polled(response)
    assert [e["event"] for e in events] == ["query", "athena"]
    assert events[0]["output"] == "s3://out/"
    assert events[1]["statistics"]["DataScannedInBytes"] == 123


class TestRunner:
    def test_trial_uses_child_process_and_external_memory_samples(self, tmp_path):
        result = supervise(
            {"trial": "test"},
            tmp_path / "events.jsonl",
            replace(Settings(), timeout_seconds=20),
            successful_worker,
        )
        assert result["status"] == "ok"
        assert result["rss_peak_bytes"] >= result["rss_baseline_bytes"] > 0
        assert result["resource_samples"]
        events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
        assert events[0]["event"] == "ready"

    def test_each_trial_gets_a_measured_and_removed_polars_temp_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("POLARS_TEMP_DIR", "/parent/value")
        result = supervise(
            {"trial": "test"},
            tmp_path / "events.jsonl",
            replace(Settings(), timeout_seconds=20),
            cache_writing_worker,
        )
        assert result["status"] == "ok"
        assert result["temp_dir_peak_bytes"] >= 4096
        assert result["temp_dir"] != "/parent/value"
        assert not Path(result["temp_dir"]).exists()
        assert os.environ["POLARS_TEMP_DIR"] == "/parent/value"

    def test_timeout_does_not_become_a_successful_timing(self, tmp_path):
        result = supervise(
            {"trial": "test"},
            tmp_path / "events.jsonl",
            replace(Settings(), timeout_seconds=0.1),
            waiting_worker,
        )
        assert result["status"] == "timeout"

    @pytest.mark.parametrize("status", ["timeout", "memory_limit", "worker_exit", "error"])
    def test_failed_trial_stops_suite_before_any_more_queries(self, monkeypatch, tmp_path, status):
        calls = []
        monkeypatch.setattr(
            "pyathena_bench.runner.validate_manifest",
            lambda *args: {"ScratchDatabase": "scratch", "Bucket": "out"},
        )
        monkeypatch.setattr("pyathena_bench.runner.validate_inputs", lambda *args: None)
        monkeypatch.setattr("pyathena_bench.runner.environment", lambda *args: {})

        def fail(payload, events, settings):
            calls.append(payload)
            return {"status": status}

        monkeypatch.setattr("pyathena_bench.runner.supervise", fail)
        manifest = {
            "run_id": "a" * 32,
            "scales": {"small": {"table": "input", "rows": 10}},
            "fixtures": {},
        }
        assert not run(
            Settings(), manifest, [Case("cursor"), Case("dict")], ["small"], tmp_path / "run"
        )
        assert len(calls) == 1
        assert calls[0]["warmup"]
        result = json.loads((tmp_path / "run/trials.jsonl").read_text())
        assert result["status"] == status


class FakeS3Listing:
    """An S3 client whose paginated listing returns fixed pages."""

    def __init__(self, pages):
        self.pages = pages
        self.requests = []

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return self

    def paginate(self, **kwargs):
        self.requests.append(kwargs)
        return iter(self.pages)


def test_output_files_count_parquet_files_below_the_trial_prefix(monkeypatch):
    s3 = FakeS3Listing(
        [
            {
                "Contents": [
                    {"Key": "runs/r/trials/t/q.csv", "Size": 10},
                    {"Key": "runs/r/trials/t/q.csv.metadata", "Size": 1},
                    {"Key": "runs/r/trials/t/q-manifest.csv", "Size": 2},
                    {"Key": "runs/r/trials/t/unload/20261012_0001", "Size": 300},
                ]
            },
            {"Contents": [{"Key": "runs/r/trials/t/unload/20261012_0002", "Size": 100}]},
            {},
        ]
    )
    monkeypatch.setattr("pyathena_bench.runner.session", lambda settings: None)
    monkeypatch.setattr("pyathena_bench.runner.client", lambda session_, service: s3)
    assert output_files(Settings(), "s3://out/runs/r/trials/t/") == {
        "count": 2,
        "bytes": 400,
        "max_bytes": 300,
        "min_bytes": 100,
    }
    assert s3.requests == [{"Bucket": "out", "Prefix": "runs/r/trials/t/"}]


@pytest.mark.parametrize(
    ("case", "listed"),
    [(Case("pandas", transport="unload", output="native"), True), (Case("pandas"), False)],
)
def test_unload_trials_record_their_output_files(monkeypatch, tmp_path, case, listed):
    monkeypatch.setattr(
        "pyathena_bench.runner.validate_manifest",
        lambda *args: {"ScratchDatabase": "scratch", "Bucket": "out"},
    )
    monkeypatch.setattr("pyathena_bench.runner.validate_inputs", lambda *args: None)
    monkeypatch.setattr("pyathena_bench.runner.environment", lambda *args: {})
    monkeypatch.setattr("pyathena_bench.runner.supervise", lambda *args: {"status": "ok"})
    locations = []

    def summarize(settings, location):
        locations.append(location)
        return {"count": 1, "bytes": 5, "max_bytes": 5, "min_bytes": 5}

    monkeypatch.setattr("pyathena_bench.runner.output_files", summarize)
    manifest = {
        "run_id": "a" * 32,
        "scales": {"small": {"table": "input", "rows": 10}},
        "fixtures": {},
    }
    settings = replace(Settings(), warmups=0, repetitions=1)
    assert run(settings, manifest, [case], ["small"], tmp_path / "run")
    trial = json.loads((tmp_path / "run/trials.jsonl").read_text())
    assert ("output_files" in trial) is listed
    assert [loc.startswith(f"s3://out/runs/{'a' * 32}/trials/") for loc in locations] == (
        [True] if listed else []
    )
