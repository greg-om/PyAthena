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
from io import BytesIO
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


RUN_MANIFEST = {
    "run_id": "a" * 32,
    "scales": {"small": {"table": "input", "rows": 10}},
    "fixtures": {},
}


@pytest.fixture
def run_stubs(monkeypatch):
    """Replace the AWS calls of ``run()`` and return the S3 client it creates."""
    s3 = object()
    monkeypatch.setattr(
        "pyathena_bench.runner.validate_manifest",
        lambda *args: {"ScratchDatabase": "scratch", "Bucket": "out"},
    )
    monkeypatch.setattr("pyathena_bench.runner.validate_inputs", lambda *args: None)
    monkeypatch.setattr("pyathena_bench.runner.environment", lambda *args: {})
    monkeypatch.setattr("pyathena_bench.runner.session", lambda settings: None)
    monkeypatch.setattr("pyathena_bench.runner.client", lambda session_, service: s3)
    return s3


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
    def test_failed_trial_stops_suite_before_any_more_queries(
        self, monkeypatch, tmp_path, run_stubs, status
    ):
        calls = []

        def fail(payload, events, settings):
            calls.append(payload)
            return {"status": status}

        monkeypatch.setattr("pyathena_bench.runner.supervise", fail)
        assert not run(
            Settings(), RUN_MANIFEST, [Case("cursor"), Case("dict")], ["small"], tmp_path / "run"
        )
        assert len(calls) == 1
        assert calls[0]["warmup"]
        result = json.loads((tmp_path / "run/trials.jsonl").read_text())
        assert result["status"] == status


class FakeS3:
    """An S3 client with a fixed paginated listing and object bodies."""

    def __init__(self, pages, bodies=None, heads=None):
        self.pages = pages
        self.bodies = bodies or {}
        self.heads = heads or {}
        self.requests = []

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return self

    def paginate(self, **kwargs):
        self.requests.append(("list", kwargs))
        return iter(self.pages)

    def get_object(self, **kwargs):
        self.requests.append(("get", kwargs["Key"]))
        return {"Body": BytesIO(self.bodies[kwargs["Key"]].encode())}

    def head_object(self, **kwargs):
        self.requests.append(("head", kwargs["Bucket"], kwargs["Key"]))
        return {"ContentLength": self.heads[(kwargs["Bucket"], kwargs["Key"])]}


def test_output_files_follow_the_query_manifests():
    root = "runs/r/trials/t/"
    s3 = FakeS3(
        [
            {
                "Contents": [
                    {"Key": f"{root}q.csv", "Size": 10},
                    {"Key": f"{root}u-manifest.csv", "Size": 2},
                    {"Key": f"{root}u.metadata", "Size": 1},
                    {"Key": f"{root}unload/20261012_0001", "Size": 300},
                    # Not in a manifest, so not a data file of the query.
                    {"Key": f"{root}unload/leftover", "Size": 999},
                ]
            },
            {
                "Contents": [
                    {"Key": f"{root}unload/20261012_0002", "Size": 100},
                    {"Key": f"{root}tables/c-manifest.csv", "Size": 2},
                ]
            },
            {},
        ],
        bodies={
            f"{root}u-manifest.csv": (
                f"s3://out/{root}unload/20261012_0001\ns3://out/{root}unload/20261012_0002\n"
            ),
            f"{root}tables/c-manifest.csv": "s3://elsewhere/ctas/f1\n",
        },
        heads={("elsewhere", "ctas/f1"): 50},
    )
    assert output_files(s3, f"s3://out/{root}") == {
        "count": 3,
        "bytes": 450,
        "max_bytes": 300,
        "min_bytes": 50,
    }
    assert s3.requests[0] == ("list", {"Bucket": "out", "Prefix": root})
    assert ("head", "elsewhere", "ctas/f1") in s3.requests


def test_output_files_without_a_manifest_is_an_error():
    s3 = FakeS3([{"Contents": [{"Key": "runs/r/trials/t/q.csv", "Size": 10}]}])
    with pytest.raises(ValueError, match="No query manifest"):
        output_files(s3, "s3://out/runs/r/trials/t/")


@pytest.mark.parametrize(
    ("case", "listed"),
    [(Case("pandas", transport="unload", output="native"), True), (Case("pandas"), False)],
)
def test_unload_trials_record_their_output_files(monkeypatch, tmp_path, run_stubs, case, listed):
    monkeypatch.setattr("pyathena_bench.runner.supervise", lambda *args: {"status": "ok"})
    calls = []

    def summarize(s3, location):
        calls.append((s3, location))
        return {"count": 1, "bytes": 5, "max_bytes": 5, "min_bytes": 5}

    monkeypatch.setattr("pyathena_bench.runner.output_files", summarize)
    settings = replace(Settings(), warmups=1, repetitions=2)
    assert run(settings, RUN_MANIFEST, [case], ["small"], tmp_path / "run")
    trials = [json.loads(line) for line in (tmp_path / "run/trials.jsonl").read_text().splitlines()]
    # Measured trials only, with the run's one client.
    assert [("output_files" in t) for t in trials] == [False, listed, listed]
    assert len(calls) == (2 if listed else 0)
    assert all(
        s3 is run_stubs and location.startswith(f"s3://out/runs/{'a' * 32}/trials/")
        for s3, location in calls
    )
