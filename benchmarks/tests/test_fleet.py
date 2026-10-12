# Copyright 2026 The PyAthena authors
#
# Licensed under the MIT License.
# See LICENSE or https://opensource.org/licenses/MIT.
#
# SPDX-License-Identifier: MIT

import csv
import io
import json
import signal
import threading
import time
import urllib.error
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError
from pyathena_bench.__main__ import main
from pyathena_bench.config import Settings
from pyathena_bench.fleet import (
    Filters,
    JobWatch,
    Queue,
    SpotInterruptionError,
    expand_jobs,
    interruption_notice,
    merge_jobs,
    quiesce,
    retire_host,
    run_job,
    status,
    submit,
    suspend_rebalance,
    trial_evidence,
    work,
)

SETTINGS = Settings(scales={"small": 10000, "xlarge": 10000000})


class FakeS3:
    """In-memory S3 with conditional writes, as used by the fleet queue."""

    def __init__(self):
        self.objects = {}
        self.etags = {}
        self.writes = 0

    def put_object(self, **kwargs):
        key = kwargs["Key"]
        if kwargs.get("IfNoneMatch") == "*" and key in self.objects:
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        if "IfMatch" in kwargs and self.etags.get(key) != kwargs["IfMatch"]:
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        self.writes += 1
        self.objects[key] = kwargs["Body"]
        self.etags[key] = f'"{self.writes}"'
        return {"ETag": self.etags[key]}

    def get_object(self, **kwargs):
        if kwargs["Key"] not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {
            "Body": io.BytesIO(self.objects[kwargs["Key"]]),
            "ETag": self.etags.get(kwargs["Key"], '"0"'),
        }

    def delete_object(self, **kwargs):
        self.objects.pop(kwargs["Key"], None)
        self.etags.pop(kwargs["Key"], None)

    def upload_file(self, filename, bucket, key):
        self.put_object(Key=key, Body=Path(filename).read_bytes())

    def list_objects_v2(self, **kwargs):
        keys = [k for k in self.objects if k.startswith(kwargs["Prefix"])]
        return {"KeyCount": len(keys[: kwargs.get("MaxKeys", 1000)])}

    def get_paginator(self, name):
        def paginate(**kwargs):
            keys = [k for k in sorted(self.objects) if k.startswith(kwargs["Prefix"])]
            return [{"Contents": [{"Key": k} for k in keys]}]

        return SimpleNamespace(paginate=paginate)


def queue_with(jobs, tmp_path):
    s3 = FakeS3()
    queue = Queue(s3, "bucket", "run1")
    config = tmp_path / "config.toml"
    config.write_text("[scales]\nsmall = 10000\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    submit(queue, jobs, config, manifest)
    return s3, queue


def job(job_id, heavy=False, api_weight=1):
    return {
        "id": job_id,
        "args": [],
        "api_heavy": heavy,
        "api_weight": api_weight if heavy else 0,
        "pages": 0,
        "weight": 0,
    }


def no_queries(ids):
    return []


class TestExpandJobs:
    def test_row_cursor_jobs_are_split_by_arraysize_and_marked_api_heavy(self):
        jobs = expand_jobs(SETTINGS, ["single"], ["small"], ["flat"], Filters(family=["cursor"]))
        assert len(jobs) == 6
        assert all(j["api_heavy"] for j in jobs)
        assert {j["id"] for j in jobs} >= {
            "single-small-flat-cursor-sync-csv-rows-a100",
            "single-small-flat-cursor-aio-csv-rows-a1000",
        }
        assert jobs[0]["pages"] >= jobs[-1]["pages"]
        assert jobs[0]["args"][-2:] == ["--arraysize", "100"]

    def test_dataframe_chunk_variants_share_one_job(self):
        jobs = expand_jobs(
            SETTINGS,
            ["single"],
            ["small"],
            ["flat"],
            Filters(family=["pandas"], api=["sync"], transport=["csv"], output_kind="native"),
        )
        assert [j["id"] for j in jobs] == ["single-small-flat-pandas-sync-csv-native"]
        assert not jobs[0]["api_heavy"]
        assert "--arraysize" not in jobs[0]["args"]

    def test_long_api_trials_become_single_repetition_jobs(self):
        jobs = expand_jobs(
            SETTINGS,
            ["single"],
            ["xlarge"],
            ["flat"],
            Filters(family=["cursor"], api=["sync"]),
            split_pages=50000,
        )
        split = [j for j in jobs if "-a100-" in j["id"]]
        assert [j["id"][-3:] for j in split] == ["-r1", "-r2", "-r3", "-r4", "-r5"]
        assert all(j["args"][-4:] == ["--warmups", "0", "--repetitions", "1"] for j in split)
        assert [j["id"] for j in jobs if "-a1000" in j["id"]] == [
            "single-xlarge-flat-cursor-sync-csv-rows-a1000"
        ]

    def test_trial_overrides_are_carried_by_unsplit_jobs(self):
        jobs = expand_jobs(
            replace(SETTINGS, warmups=0, repetitions=3),
            ["single"],
            ["xlarge"],
            ["flat"],
            Filters(family=["cursor"], api=["sync"]),
            split_pages=50000,
            explicit_trials=True,
        )
        whole = [j for j in jobs if "-a1000" in j["id"]]
        assert whole[0]["args"][-4:] == ["--warmups", "0", "--repetitions", "3"]
        assert whole[0]["pages"] == 10001 * 3
        split = [j for j in jobs if "-a100-" in j["id"]]
        assert [j["id"][-3:] for j in split] == ["-r1", "-r2", "-r3"]
        assert all(j["args"][-4:] == ["--warmups", "0", "--repetitions", "1"] for j in split)

    def test_api_heavy_jobs_come_first(self):
        jobs = expand_jobs(
            SETTINGS, ["single"], ["small"], ["flat"], Filters(family=["cursor", "arrow"])
        )
        heavy = [j["api_heavy"] for j in jobs]
        assert heavy == sorted(heavy, reverse=True)


class TestExpandJobsArraysize:
    def test_row_output_jobs_keep_one_arraysize_for_every_family(self):
        jobs = expand_jobs(
            SETTINGS,
            ["single"],
            ["small"],
            ["flat"],
            Filters(family=["s3fs"], api=["sync"], arraysize=[100]),
        )
        assert [j["id"] for j in jobs] == ["single-small-flat-s3fs-sync-csv-rows-a100"]
        assert jobs[0]["args"][-2:] == ["--arraysize", "100"]
        assert not jobs[0]["api_heavy"]

    def test_initialization_validation_of_row_cursors_is_api_heavy(self):
        jobs = expand_jobs(
            SETTINGS, ["init"], ["small"], ["flat"], Filters(family=["cursor", "arrow"])
        )
        heavy = {j["id"]: (j["api_heavy"], j["api_weight"]) for j in jobs}
        assert heavy["init-small-flat-cursor-direct-csv-rows-a1000"] == (True, 1)
        assert heavy["init-small-flat-arrow-direct-csv-native"] == (False, 0)

    def test_page_estimate_includes_the_column_label_row(self):
        cursor = Filters(family=["cursor"], api=["direct"])
        jobs = expand_jobs(SETTINGS, ["init"], ["small"], ["flat"], cursor, split_pages=11)
        assert [j["id"] for j in jobs] == [
            f"init-small-flat-cursor-direct-csv-rows-a1000-r{repetition}"
            for repetition in range(1, 6)
        ]
        assert {j["pages"] for j in jobs} == {11}
        unsplit = expand_jobs(SETTINGS, ["init"], ["small"], ["flat"], cursor, split_pages=12)
        assert [j["id"] for j in unsplit] == ["init-small-flat-cursor-direct-csv-rows-a1000"]

    def test_concurrent_row_jobs_weigh_simultaneous_paging_queries(self):
        settings = Settings(scales={"small": 10000}, concurrency=[1, 10])
        jobs = expand_jobs(
            settings, ["concurrent"], ["small"], ["flat"], Filters(family=["cursor"], api=["aio"])
        )
        assert [(j["api_heavy"], j["api_weight"]) for j in jobs] == [(True, 10)]


class TestQueue:
    def test_submit_refuses_an_existing_queue_and_duplicate_ids(self, tmp_path):
        s3, queue = queue_with([job("a")], tmp_path)
        with pytest.raises(ValueError, match="already reserved"):
            submit(queue, [job("b")], tmp_path / "config.toml", tmp_path / "manifest.json")
        with pytest.raises(ValueError, match="unique"):
            submit(
                Queue(s3, "bucket", "run2"),
                [job("a"), job("a")],
                tmp_path / "config.toml",
                tmp_path / "manifest.json",
            )

    def test_workers_run_each_job_once_and_record_results(self, tmp_path):
        s3, queue = queue_with([job("heavy", heavy=True), job("light")], tmp_path)
        runs = []
        settled = []

        def runner(item, workdir, out, resume, stop):
            runs.append(item["id"])
            out.mkdir(parents=True)
            (out / "summary.csv").write_text("x")
            return 0 if item["id"] == "light" else 1

        assert work(
            queue,
            tmp_path / "w1",
            api_slots=1,
            poll_seconds=0,
            runner=runner,
            settle=lambda output: settled.append(output) or [],
        )
        assert (
            work(
                queue,
                tmp_path / "w2",
                api_slots=1,
                poll_seconds=0,
                runner=runner,
                settle=no_queries,
            )
            == []
        )
        assert sorted(runs) == ["heavy", "light"]
        assert [path.parent.name for path in settled] == ["heavy"]
        assert queue.key("results", "light", "attempt-1", "summary.csv") in s3.objects
        assert not queue.names("slots")
        summary = status(queue)
        assert summary["done"] == 2
        assert list(summary["failed"]) == ["heavy"]

    def test_api_heavy_jobs_wait_for_a_free_slot(self, tmp_path):
        s3, queue = queue_with([job("heavy", heavy=True), job("light")], tmp_path)
        other_host = queue.key("slots", "0")
        assert queue.acquire(other_host, {"host": "other"}, 600)
        order = []

        def runner(item, workdir, *args):
            order.append(item["id"])
            s3.delete_object(Bucket="bucket", Key=other_host)
            return 0

        work(queue, tmp_path / "w", api_slots=1, poll_seconds=0, runner=runner, settle=no_queries)
        assert order == ["light", "heavy"]


class TestMergeJobs:
    def test_merged_lists_follow_the_queue_order(self):
        heavy = {**job("heavy", heavy=True), "pages": 10}
        longer = {**job("longer"), "weight": 2}
        merged = merge_jobs([[job("shorter"), longer], [heavy]])
        assert [j["id"] for j in merged] == ["heavy", "longer", "shorter"]

    def test_duplicate_ids_across_lists_are_rejected(self):
        with pytest.raises(ValueError, match="unique"):
            merge_jobs([[job("a")], [job("a")]])


METADATA = {
    "instance-id": "i-123",
    "instance-type": "r7g.2xlarge",
    "placement/availability-zone": "us-west-2b",
    "instance-life-cycle": "spot",
}


def metadata_with(**tags):
    def read(path):
        if path.startswith("tags/instance/"):
            name = path.removeprefix("tags/instance/")
            if name not in tags:
                raise urllib.error.HTTPError(path, 404, "Not Found", None, None)
            return tags[name]
        return METADATA[path]

    return read


class FakeAutoScaling:
    """Auto Scaling client that records calls in order."""

    def __init__(self):
        self.calls = []

    def describe_auto_scaling_instances(self, **kwargs):
        self.calls.append(("describe", kwargs))
        return {"AutoScalingInstances": [{"AutoScalingGroupName": "fleet-group"}]}

    def suspend_processes(self, **kwargs):
        self.calls.append(("suspend", kwargs))

    def terminate_instance_in_auto_scaling_group(self, **kwargs):
        self.calls.append(("terminate", kwargs))


class TestRetireHost:
    def test_worker_start_suspends_zone_rebalancing(self):
        autoscaling = FakeAutoScaling()
        assert suspend_rebalance(autoscaling, metadata_with()) == "fleet-group"
        assert autoscaling.calls == [
            ("describe", {"InstanceIds": ["i-123"]}),
            (
                "suspend",
                {"AutoScalingGroupName": "fleet-group", "ScalingProcesses": ["AZRebalance"]},
            ),
        ]

    def test_finished_worker_records_its_host_and_terminates_it(self, tmp_path):
        s3, queue = queue_with([job("a")], tmp_path)
        autoscaling = FakeAutoScaling()
        assert retire_host(queue, autoscaling, ["a"], metadata=metadata_with(KeepHost="false"))
        assert autoscaling.calls == [
            ("terminate", {"InstanceId": "i-123", "ShouldDecrementDesiredCapacity": True}),
        ]
        record = json.loads(s3.objects[queue.key("hosts", "i-123.json")])
        assert record["ran"] == ["a"]
        assert record["kept"] is False
        assert (record["instance_type"], record["availability_zone"], record["lifecycle"]) == (
            "r7g.2xlarge",
            "us-west-2b",
            "spot",
        )

    def test_keep_tag_leaves_the_host_and_its_group_unchanged(self, tmp_path):
        s3, queue = queue_with([job("a")], tmp_path)
        autoscaling = FakeAutoScaling()
        assert not retire_host(queue, autoscaling, [], metadata=metadata_with(KeepHost="true"))
        assert autoscaling.calls == []
        assert json.loads(s3.objects[queue.key("hosts", "i-123.json")])["kept"] is True


class TestCommandLine:
    def test_jobs_command_prints_the_expanded_queue(self, tmp_path, capsys):
        config = tmp_path / "config.toml"
        config.write_text("[scales]\nsmall = 10000\n")
        argv = ["--config", str(config), "jobs", "--suite", "single", "--scale", "small"]
        assert main([*argv, "--family", "cursor", "--shape", "flat", "nested"]) == 0
        jobs = json.loads(capsys.readouterr().out)
        assert len(jobs) == 12
        assert {j["args"][5] for j in jobs} == {"flat", "nested"}

    def test_jobs_command_carries_trial_overrides(self, tmp_path, capsys):
        config = tmp_path / "config.toml"
        config.write_text("[scales]\nsmall = 10000\n")
        argv = ["--config", str(config), "jobs", "--suite", "single", "--scale", "small"]
        filters = ["--family", "pandas", "--api", "sync", "--transport", "csv"]
        assert main([*argv, *filters, "--output-kind", "native", "--repetitions", "3"]) == 0
        jobs = json.loads(capsys.readouterr().out)
        assert [j["args"][-4:] for j in jobs] == [["--warmups", "1", "--repetitions", "3"]]

    def test_plan_honors_arraysize_filter_and_repetition_overrides(self, tmp_path, capsys):
        config = tmp_path / "config.toml"
        config.write_text("[scales]\nsmall = 10000\n")
        argv = ["--config", str(config), "plan", "--suite", "single", "--scale", "small"]
        filters = ["--family", "cursor", "--api", "sync", "--arraysize", "1000"]
        assert main([*argv, *filters, "--warmups", "0", "--repetitions", "1"]) == 0
        plan = json.loads(capsys.readouterr().out)
        assert [c["arraysize"] for c in plan["cases"]] == [1000]
        assert plan["trials"] == 1


class TestQueueSafety:
    def test_interrupted_publication_can_be_repeated_and_is_not_consumed(self, tmp_path):
        s3 = FakeS3()
        queue = Queue(s3, "bucket", "run1")
        config = tmp_path / "config.toml"
        config.write_text("[scales]\nsmall = 10000\n")
        with pytest.raises(FileNotFoundError):
            submit(queue, [job("a")], config, tmp_path / "missing.json")
        with pytest.raises(ValueError, match="incompletely published"):
            work(queue, tmp_path / "w", api_slots=1, runner=lambda *a: 0, settle=no_queries)
        manifest = tmp_path / "manifest.json"
        manifest.write_text("{}")
        submit(queue, [job("a")], config, manifest)
        assert status(queue)["jobs"] == 1

    def test_failed_jobs_are_quiesced_and_unconfirmed_queries_stop_the_worker(self, tmp_path):
        _, queue = queue_with([job("first"), job("second")], tmp_path)
        runs = []

        def runner(item, workdir, out, resume, stop):
            runs.append(item["id"])
            out.mkdir(parents=True)
            (out / "events.jsonl").write_text(
                json.dumps({"event": "query", "query_id": "q1", "trial": "t"}) + "\n"
            )
            return 1

        def settle(output):
            assert trial_evidence(output)[0] == {"q1"}
            return ["q1: still RUNNING"]

        assert work(queue, tmp_path / "w", api_slots=1, runner=runner, settle=settle) == ["first"]
        assert runs == ["first"]
        assert status(queue)["failed"]["first"]["quiesce_errors"] == ["q1: still RUNNING"]

    def test_waiting_heavy_job_reserves_slots_from_later_heavy_jobs(self, tmp_path, monkeypatch):
        s3, queue = queue_with(
            [job("wide", heavy=True, api_weight=2), job("narrow", heavy=True)], tmp_path
        )
        other_host = queue.key("slots", "0")
        assert queue.acquire(other_host, {"host": "other"}, 600)
        attempts = []

        def sleep_and_release(seconds):
            attempts.append(sorted(queue.names("claims")))
            s3.delete_object(Bucket="bucket", Key=other_host)

        order = []

        def runner(item, workdir, *args):
            order.append(item["id"])
            return 0

        monkeypatch.setattr("pyathena_bench.fleet.time.sleep", sleep_and_release)
        work(queue, tmp_path / "w", api_slots=2, runner=runner, settle=no_queries)
        assert attempts == [[]]
        assert order == ["wide", "narrow"]

    def test_trial_evidence_reads_reported_queries_and_trial_prefixes(self, tmp_path):
        (tmp_path / "events.jsonl").write_text(
            "\n".join(
                [
                    json.dumps({"event": "trial_start", "output": "s3://b/runs/r/trials/t1/"}),
                    json.dumps({"event": "query", "query_id": "a"}),
                    json.dumps({"event": "athena", "query_id": "b"}),
                    "{incomplete",
                ]
            )
        )
        assert trial_evidence(tmp_path) == ({"a"}, ("s3://b/runs/r/trials/t1/",))
        assert trial_evidence(tmp_path / "missing") == (set(), ())

    def test_jobs_wider_than_the_api_slots_stop_the_worker_before_claiming(self, tmp_path):
        _, queue = queue_with([job("wide", heavy=True, api_weight=11)], tmp_path)
        with pytest.raises(ValueError, match="more than 10 API slots"):
            work(queue, tmp_path / "w", api_slots=10, runner=lambda *a: 0, settle=no_queries)
        assert not queue.names("claims")

    def test_slot_acquisition_errors_release_taken_slots(self, tmp_path):
        s3, queue = queue_with([], tmp_path)
        original = s3.put_object

        def flaky(**kwargs):
            if kwargs["Key"].endswith("slots/1"):
                raise ClientError({"Error": {"Code": "InternalError"}}, "PutObject")
            return original(**kwargs)

        s3.put_object = flaky
        with pytest.raises(ClientError):
            queue.acquire_slots(3, 2, {"host": "h"}, 600)
        assert not queue.names("slots")

    def test_overlapping_submissions_cannot_share_a_name(self, tmp_path):
        s3, queue = queue_with([job("a")], tmp_path)
        with pytest.raises(ValueError, match="reserved"):
            submit(queue, [job("b")], tmp_path / "config.toml", tmp_path / "manifest.json")
        assert json.loads(queue.read("jobs.json"))[0]["id"] == "a"


class FakeAthena:
    def __init__(self, states, stop_errors=()):
        self.states = states
        self.stop_errors = set(stop_errors)
        self.stopped = []

    def stop_query_execution(self, **kwargs):
        query_id = kwargs["QueryExecutionId"]
        self.stopped.append(query_id)
        if query_id in self.stop_errors:
            raise ClientError({"Error": {"Code": "InvalidRequestException"}}, "StopQueryExecution")
        self.states[query_id] = "CANCELLED"

    def get_query_execution(self, **kwargs):
        return {"QueryExecution": {"Status": {"State": self.states[kwargs["QueryExecutionId"]]}}}


class TestQuiesce:
    def test_unreported_queries_under_trial_prefixes_are_stopped(self, tmp_path, monkeypatch):
        (tmp_path / "events.jsonl").write_text(
            json.dumps({"event": "trial_start", "output": "s3://b/runs/r/trials/t1/"}) + "\n"
        )
        athena = FakeAthena({"unreported": "RUNNING"})
        found = {}

        def active(client_, workgroup, prefixes):
            found["prefixes"] = prefixes
            return {"unreported"}

        monkeypatch.setattr("pyathena_bench.fleet.session", lambda settings: None)
        monkeypatch.setattr("pyathena_bench.fleet.client", lambda session_, name: athena)
        monkeypatch.setattr("pyathena_bench.fleet.active_queries", active)
        assert quiesce(Settings(poll_interval=0.01), tmp_path) == []
        assert found["prefixes"] == ("s3://b/runs/r/trials/t1/",)
        assert athena.stopped == ["unreported"]

    def test_stop_errors_are_ignored_once_the_query_is_terminal(self, tmp_path, monkeypatch):
        (tmp_path / "events.jsonl").write_text(
            json.dumps({"event": "query", "query_id": "done"}) + "\n"
        )
        athena = FakeAthena({"done": "SUCCEEDED"}, stop_errors={"done"})
        monkeypatch.setattr("pyathena_bench.fleet.session", lambda settings: None)
        monkeypatch.setattr("pyathena_bench.fleet.client", lambda session_, name: athena)
        assert quiesce(Settings(poll_interval=0.01), tmp_path) == []


class TestFollowUpSafety:
    def test_a_queue_without_a_reservation_is_not_overwritten(self, tmp_path):
        s3, queue = queue_with([job("a")], tmp_path)
        s3.delete_object(Bucket="bucket", Key=queue.key("reservation.json"))
        with pytest.raises(ValueError, match="already exists"):
            submit(queue, [job("b")], tmp_path / "config.toml", tmp_path / "manifest.json")
        assert json.loads(queue.read("jobs.json"))[0]["id"] == "a"
        assert not queue.exists("reservation.json")

    def test_a_failed_existence_check_releases_the_reservation(self, tmp_path):
        s3 = FakeS3()
        queue = Queue(s3, "bucket", "run1")

        def failing(**kwargs):
            raise ClientError({"Error": {"Code": "InternalError"}}, "ListObjectsV2")

        s3.list_objects_v2 = failing
        with pytest.raises(ClientError):
            submit(queue, [job("a")], tmp_path / "config.toml", tmp_path / "manifest.json")
        assert queue.key("reservation.json") not in s3.objects

    def test_quiesce_errors_are_recorded_before_the_worker_stops(self, tmp_path):
        _, queue = queue_with([job("first"), job("second")], tmp_path)

        def settle(output):
            raise RuntimeError("Could not inspect query history")

        ran = work(queue, tmp_path / "w", api_slots=1, runner=lambda *a: 1, settle=settle)
        assert ran == ["first"]
        errors = status(queue)["failed"]["first"]["quiesce_errors"]
        assert errors == [
            "Could not quiesce the job: RuntimeError: Could not inspect query history"
        ]


def ok_trial(repetition):
    return {
        "trial": f"t{repetition}",
        "case_id": "case",
        "scale": "small",
        "repetition": repetition,
        "warmup": False,
        "status": "ok",
        "queries": [{"total_seconds": 1}],
        "successful_queries_per_second": 1,
        "rss_peak_bytes": 1,
        "max_threads": 1,
    }


class TestLeases:
    def test_only_one_worker_takes_over_an_expired_lease(self, tmp_path):
        _, queue = queue_with([], tmp_path)
        key = queue.key("claims", "a")
        assert queue.acquire(key, {"host": "dead"}, -1).body["attempt"] == 1
        stale = queue.read_lease(key)
        first = queue.acquire(key, {"host": "one"}, 600)
        assert first.body["attempt"] == 2
        assert queue.acquire(key, {"host": "two"}, 600) is None
        # A worker that read the expired lease before the takeover cannot overwrite it.
        assert queue.put(key, {"host": "late"}, IfMatch=stale.etag) is None
        assert queue.renew(first, 600)
        assert not queue.renew(stale, 600)
        assert queue.read_lease(key).body["host"] == "one"

    @pytest.mark.parametrize("operation", ["create", "takeover", "renew"])
    def test_a_retried_write_that_already_succeeded_keeps_the_lease(self, tmp_path, operation):
        s3, queue = queue_with([], tmp_path)
        key = queue.key("claims", "a")
        if operation != "create":
            lease = queue.acquire(key, {"host": "h"}, -1 if operation == "takeover" else 600)
        original = s3.put_object

        def lost_response(**kwargs):
            # The first write succeeds, but its response is lost and the SDK retries.
            original(**kwargs)
            return original(**kwargs)

        s3.put_object = lost_response
        if operation == "renew":
            assert queue.renew(lease, 600)
            assert queue.read_lease(key).etag == lease.etag
        else:
            taken = queue.acquire(key, {"host": "me"}, 600)
            assert taken.body["host"] == "me"
            assert taken.body["attempt"] == (1 if operation == "create" else 2)
            assert queue.read_lease(key).etag == taken.etag

    def test_live_claims_are_left_and_expired_ones_return_to_the_fleet(self, tmp_path):
        _, queue = queue_with([job("live"), job("expired")], tmp_path)
        queue.acquire(queue.key("claims", "live"), {"host": "busy"}, 600)
        queue.acquire(queue.key("claims", "expired"), {"host": "dead"}, -1)
        runs = []

        def runner(item, workdir, out, resume, stop):
            runs.append((item["id"], out.name, resume))
            return 0

        ran = work(
            queue, tmp_path / "w", api_slots=1, poll_seconds=0, runner=runner, settle=no_queries
        )
        assert ran == ["expired"]
        assert runs == [("expired", "attempt-2", [])]
        running = status(queue)["running"]
        assert running["live"] == {
            "host": "busy",
            "job": None,
            "attempt": 1,
            "state": "running",
            "expired": False,
        }

    def test_an_expired_slot_returns_to_the_fleet(self, tmp_path):
        _, queue = queue_with([job("heavy", heavy=True)], tmp_path)
        queue.acquire(queue.key("slots", "0"), {"host": "dead"}, -1)
        assert status(queue)["slots_in_use"]["0"]["expired"]
        ran = work(
            queue,
            tmp_path / "w",
            api_slots=1,
            poll_seconds=0,
            runner=lambda *a: 0,
            settle=no_queries,
        )
        assert ran == ["heavy"]
        assert not queue.names("slots")

    def test_watch_renews_leases_and_uploads_progress(self, tmp_path):
        _, queue = queue_with([], tmp_path)
        lease = queue.acquire(queue.key("claims", "a"), {"host": "h"}, 1)
        expires = lease.body["expires"]
        uploads = []
        with JobWatch(queue, [lease], 600, 0.01, lambda: uploads.append(1), lambda: False):
            deadline = time.monotonic() + 5
            while not uploads and time.monotonic() < deadline:
                time.sleep(0.01)
        assert uploads
        assert queue.read_lease(lease.key).body["expires"] > expires + 500

    def test_watch_stops_renewing_a_lease_taken_over_by_another_worker(self, tmp_path):
        _, queue = queue_with([], tmp_path)
        lease = queue.acquire(queue.key("claims", "a"), {"host": "h"}, -1)
        assert queue.acquire(lease.key, {"host": "other"}, 600)
        with JobWatch(queue, [lease], 600, 0.01, lambda: None, lambda: False) as watch:
            deadline = time.monotonic() + 5
            while not watch.lost and time.monotonic() < deadline:
                time.sleep(0.01)
        assert watch.lost == [lease.key]
        assert queue.read_lease(lease.key).body["host"] == "other"


class TestInterruption:
    def test_notice_comes_from_instance_metadata_or_a_simulation_file(self, tmp_path):
        def missing(path):
            raise urllib.error.HTTPError(path, 404, "Not Found", None, None)

        absent = tmp_path / "notice"
        assert not interruption_notice(missing, absent)
        assert interruption_notice(lambda path: '{"action": "terminate"}', absent)
        absent.touch()
        assert interruption_notice(missing, absent)

    def test_interrupted_job_is_resumed_by_another_worker_with_the_same_trials(self, tmp_path):
        s3, queue = queue_with([job("heavy", heavy=True), job("next")], tmp_path)
        notice = threading.Event()

        def interrupted_runner(item, workdir, out, resume, stop):
            out.mkdir(parents=True)
            (out / "trials.jsonl").write_text(json.dumps(ok_trial(0)) + "\n")
            notice.set()
            assert stop.wait(5)
            with (out / "trials.jsonl").open("a") as stream:
                stream.write(json.dumps({**ok_trial(1), "status": "interrupted"}) + "\n")
            return 1

        with pytest.raises(SpotInterruptionError) as raised:
            work(
                queue,
                tmp_path / "w1",
                api_slots=1,
                poll_seconds=0,
                runner=interrupted_runner,
                settle=no_queries,
                renew_seconds=0.01,
                notice=notice.is_set,
            )
        assert raised.value.ran == []
        claim = queue.read_lease(queue.key("claims", "heavy"))
        assert claim.body["state"] == "interrupted"
        assert not queue.names("slots")
        assert queue.key("results", "heavy", "attempt-1", "trials.jsonl") in s3.objects
        assert not queue.names("done")

        settled = []
        runs = []

        def resumed_runner(item, workdir, out, resume, stop):
            runs.append((item["id"], out.name, [p.name for p in resume]))
            out.mkdir(parents=True)
            if item["id"] == "heavy":
                (out / "trials.jsonl").write_text(
                    "".join(json.dumps(ok_trial(r)) + "\n" for r in (1, 2))
                )
            return 0

        ran = work(
            queue,
            tmp_path / "w2",
            api_slots=1,
            poll_seconds=0,
            runner=resumed_runner,
            settle=lambda output: settled.append(output.name) or [],
        )
        assert ran == ["heavy", "next"]
        assert runs == [("heavy", "attempt-2", ["attempt-1"]), ("next", "attempt-1", [])]
        # The interrupted attempt's queries are quiesced before the job continues.
        assert settled == ["attempt-1"]
        summary = s3.objects[queue.key("results", "heavy", "summary.csv")].decode()
        row = next(csv.DictReader(io.StringIO(summary)))
        assert (row["successful_trials"], row["failed_trials"]) == ("3", "0")
        assert json.loads(queue.read("done", "heavy"))["attempt"] == 2

    def test_a_notice_while_quiescing_an_earlier_attempt_skips_the_run(self, tmp_path):
        s3, queue = queue_with([job("a")], tmp_path)
        queue.acquire(queue.key("claims", "a"), {"host": "dead"}, -1)
        s3.put_object(Key=queue.key("results", "a", "attempt-1", "events.jsonl"), Body=b"")
        notice = threading.Event()

        def settle(output):
            notice.set()
            time.sleep(0.2)
            return []

        def runner(*args):
            pytest.fail("A job must not start after an interruption notice")

        with pytest.raises(SpotInterruptionError):
            work(
                queue,
                tmp_path / "w",
                api_slots=1,
                poll_seconds=0,
                runner=runner,
                settle=settle,
                renew_seconds=0.01,
                notice=notice.is_set,
            )
        lease = queue.read_lease(queue.key("claims", "a"))
        assert (lease.body["state"], lease.body["attempt"]) == ("interrupted", 2)

    def test_a_notice_between_jobs_stops_the_worker_without_claiming(self, tmp_path):
        _, queue = queue_with([job("a")], tmp_path)
        with pytest.raises(SpotInterruptionError):
            work(queue, tmp_path / "w", api_slots=1, runner=lambda *a: 0, notice=lambda: True)
        assert not queue.names("claims")

    def test_worker_waits_for_publication(self, tmp_path, monkeypatch):
        s3 = FakeS3()
        queue = Queue(s3, "bucket", "run1", "results/stack/")
        config = tmp_path / "config.toml"
        config.write_text("[scales]\nsmall = 10000\n")
        manifest = tmp_path / "manifest.json"
        manifest.write_text("{}")
        monkeypatch.setattr(
            "pyathena_bench.fleet.time.sleep",
            lambda seconds: submit(queue, [job("a")], config, manifest),
        )
        ran = work(
            queue, tmp_path / "w", api_slots=1, runner=lambda *a: 0, settle=no_queries, wait=True
        )
        assert ran == ["a"]
        assert all(key.startswith("results/stack/fleet/run1/") for key in s3.objects)


class FakeProcess:
    """A child process that runs until it receives SIGINT."""

    def __init__(self, command, stdout, stderr):
        self.command = command
        self.signals = []

    def poll(self):
        return 130 if self.signals else None

    def send_signal(self, signum):
        self.signals.append(signum)

    def wait(self, timeout=None):
        return 130


def test_run_job_resumes_earlier_attempts_and_interrupts_with_sigint(tmp_path, monkeypatch):
    processes = []

    def popen(command, stdout, stderr):
        processes.append(FakeProcess(command, stdout, stderr))
        return processes[-1]

    monkeypatch.setattr("pyathena_bench.fleet.subprocess.Popen", popen)
    stop = threading.Event()
    stop.set()
    earlier = tmp_path / "results" / "a" / "attempt-1"
    code = run_job(
        {"id": "a", "args": ["--suite", "single"]},
        tmp_path,
        tmp_path / "results" / "a" / "attempt-2",
        [earlier],
        stop,
    )
    assert code == 130
    [process] = processes
    assert process.signals == [signal.SIGINT]
    assert process.command[-4:] == ["--suite", "single", "--resume", str(earlier)]
    assert (tmp_path / "logs" / "a-attempt-2.log").exists()
