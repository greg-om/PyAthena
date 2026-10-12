# Copyright 2026 The PyAthena authors
#
# Licensed under the MIT License.
# See LICENSE or https://opensource.org/licenses/MIT.
#
# SPDX-License-Identifier: MIT

"""Distribute benchmark runs over identical hosts through an S3 job queue."""

from __future__ import annotations

import contextlib
import http.client
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any

from botocore.exceptions import ClientError

from pyathena_bench.aws import active_queries, client, session
from pyathena_bench.cases import Case, matrix
from pyathena_bench.config import Settings
from pyathena_bench.report import report

# Row cursors page through GetQueryResults, which has an account-wide request-rate quota;
# so does read_sql_query() through the REST driver.
API_FAMILIES = {"cursor", "dict", "sqla_rest"}
QUEUE_ROOT = "fleet"
# An instance tag with the value "true" keeps a host running after its worker exits.
KEEP_HOST_TAG = "KeepHost"
# A file that a worker treats like a Spot interruption notice, to rehearse interruptions.
SIMULATED_NOTICE = Path("interruption-notice")
METADATA_URL = "http://169.254.169.254/latest"
_logger = logging.getLogger(__name__)
# S3 errors of a conditional write whose condition failed.
PRECONDITION_ERRORS = {"PreconditionFailed", "ConditionalRequestConflict"}


@dataclass(frozen=True)
class Filters:
    """Case filters shared by ``plan``, ``run``, and ``jobs``."""

    family: list[str] | None = None
    api: list[str] | None = None
    transport: list[str] | None = None
    output_kind: str | None = None
    arraysize: list[int] | None = None
    # Chunk sizes to keep, with 0 for reads without chunks.
    chunksize: list[int] | None = None

    def select(self, cases: Iterable[Case]) -> list[Case]:
        """Return the cases that match every configured filter.

        Args:
            cases: Candidate cases.

        Returns:
            Matching cases in their original order.
        """
        return [
            c
            for c in cases
            if (not self.family or c.family in self.family)
            and (not self.api or c.api in self.api)
            and (not self.transport or c.transport in self.transport)
            and (not self.output_kind or c.output == self.output_kind)
            and (not self.arraysize or c.arraysize in self.arraysize)
            and (not self.chunksize or (c.chunksize or 0) in self.chunksize)
        ]


def api_pages(case: Case, rows: int) -> int:
    """Estimate GetQueryResults pages that one trial of a case requests.

    Args:
        case: Benchmark case.
        rows: Rows in the selected snapshot.

    The initialization suite counts too: it validates the constructed result
    set by reading every row.

    Returns:
        Page count summed over the trial's concurrent queries; 0 for cases that
        do not page through the API.
    """
    if case.family not in API_FAMILIES:
        return 0
    # The first page of a SELECT result also carries the column labels.
    return -(-(rows + 1) // case.arraysize) * case.concurrency


def job_order(job: dict[str, Any]) -> tuple[bool, int, int]:
    """Return the sort key that puts API-heavy jobs first and longer jobs earlier.

    Args:
        job: Job dictionary.

    Returns:
        Key for :func:`sorted`.
    """
    return (not job["api_heavy"], -job["pages"], -job["weight"])


def merge_jobs(job_lists: Iterable[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Combine job lists from several ``jobs`` invocations into one queue order.

    Args:
        job_lists: Job lists from :func:`expand_jobs`.

    Returns:
        All jobs, ordered as :func:`expand_jobs` orders one list.

    Raises:
        ValueError: If job IDs are not unique.
    """
    jobs = [job for jobs in job_lists for job in jobs]
    if len({j["id"] for j in jobs}) != len(jobs):
        raise ValueError("Job IDs must be unique")
    return sorted(jobs, key=job_order)


def expand_jobs(
    settings: Settings,
    suites: list[str],
    scales: list[str],
    shapes: list[str],
    filters: Filters,
    split_pages: int | None = None,
    explicit_trials: bool = False,
) -> list[dict[str, Any]]:
    """Split a selection into independent ``run`` invocations.

    A job covers one suite, scale, shape, family, API, transport, and output
    kind, and additionally one arraysize for row output. With a chunk size
    filter, each job runs and names the chunk sizes of its group's cases. Jobs that page through
    GetQueryResults are marked with the number of simultaneous paging queries
    so workers can bound the fleet's request rate. When one trial of such a job
    needs at least ``split_pages`` pages, each measured repetition becomes a
    separate single-trial job without a warmup.

    Args:
        settings: Configuration that defines the matrix and repetitions.
        suites: Suites to expand.
        scales: Scale names defined in the configuration.
        shapes: Result shapes to expand.
        filters: Case filters applied before grouping.
        split_pages: Per-trial page threshold for splitting repetitions.
        explicit_trials: Whether jobs that are not split carry the warmups and
            repetitions of ``settings`` as arguments, so a queue can mix trial
            counts under one stored configuration.

    Returns:
        Job dictionaries ordered with API-heavy jobs first and longer jobs earlier.

    Raises:
        ValueError: If no case matches the selection.
    """
    jobs: list[dict[str, Any]] = []
    for suite in suites:
        for scale in scales:
            rows = settings.scales[scale]
            for shape in shapes:
                groups: dict[tuple[Any, ...], list[Case]] = {}
                for case in filters.select(matrix(settings, suite, shape)):
                    arraysize = case.arraysize if case.output == "rows" else None
                    key = (case.family, case.api, case.transport, case.output, arraysize)
                    groups.setdefault(key, []).append(case)
                for (family, api, transport, output, arraysize), cases in groups.items():
                    base = f"{suite}-{scale}-{shape}-{family}-{api}-{transport}-{output}"
                    args = [
                        "--suite", suite, "--scale", scale, "--shape", shape,
                        "--family", family, "--api", api, "--transport", transport,
                        "--output-kind", output,
                    ]  # fmt: skip
                    if arraysize is not None:
                        base += f"-a{arraysize}"
                        args += ["--arraysize", str(arraysize)]
                    if filters.chunksize:
                        # The job runs its group's chunk sizes, which its ID names in
                        # order, so equal selections share an ID.
                        chunks = [str(c) for c in sorted({c.chunksize or 0 for c in cases})]
                        base += "-c" + "_".join(chunks)
                        args += ["--chunksize", *chunks]
                    runnable = [c for c in cases if c.unsupported is None]
                    pages = max((api_pages(c, rows) for c in runnable), default=0)
                    api_weight = max(
                        (c.concurrency for c in runnable if api_pages(c, rows)), default=0
                    )
                    weight = rows * max(len(runnable), 1)
                    if split_pages and pages >= split_pages:
                        jobs.extend(
                            {
                                "id": f"{base}-r{repetition}",
                                "args": [*args, "--warmups", "0", "--repetitions", "1"],
                                "api_heavy": True,
                                "api_weight": api_weight,
                                "pages": pages * len(runnable),
                                "weight": weight,
                            }
                            for repetition in range(1, settings.repetitions + 1)
                        )
                    else:
                        trials = settings.warmups + settings.repetitions
                        if explicit_trials:
                            args += [
                                "--warmups", str(settings.warmups),
                                "--repetitions", str(settings.repetitions),
                            ]  # fmt: skip
                        jobs.append(
                            {
                                "id": base,
                                "args": args,
                                "api_heavy": pages > 0,
                                "api_weight": api_weight,
                                "pages": pages * len(runnable) * trials,
                                "weight": weight * trials,
                            }
                        )
    if not jobs:
        raise ValueError("No cases match the selected filters")
    return sorted(jobs, key=job_order)


def now() -> str:
    """Return the current UTC time in ISO 8601 format.

    Returns:
        Timestamp with second precision.
    """
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class Lease:
    """A claim or API slot object held by one worker until it expires.

    The body records the owner, the attempt number, the state (``running``,
    ``interrupted``, or ``released``), and the times of the last write
    (``renewed``) and of the expiry, as Unix times. A worker renews its leases
    while it runs a job; another worker may take over a lease that has expired
    or was released. Objects without an expiry, written by earlier versions,
    never expire.
    """

    key: str
    body: dict[str, Any]
    etag: str

    def takeable(self, at: float) -> bool:
        """Return whether another worker may take over the lease.

        Args:
            at: Current Unix time.

        Returns:
            True if the lease was interrupted or expired before ``at``.
        """
        expires = self.body.get("expires")
        return self.body.get("state") in {"interrupted", "released"} or (
            expires is not None and expires < at
        )

    def stale(self, at: float) -> bool:
        """Return whether a running lease missed renewals but has not expired yet.

        A live worker renews a lease ten times per lease period, so a lease
        that has gone a fifth of its period without renewal belongs to a
        worker that has probably died.

        Args:
            at: Current Unix time.

        Returns:
            True if the lease is running, not takeable, and overdue for renewal.
        """
        renewed = self.body.get("renewed")
        expires = self.body.get("expires")
        if renewed is None or expires is None or self.takeable(at):
            return False
        return at - renewed > (expires - renewed) / 5


class Queue:
    """S3 layout of one named fleet run under ``<base>fleet/<name>/`` in a bucket."""

    def __init__(self, s3: Any, bucket: str, name: str, base: str = "") -> None:
        if not name or "/" in name:
            raise ValueError("Queue name must be a non-empty path segment")
        if base and not base.endswith("/"):
            raise ValueError("Queue base prefix must end with a slash")
        self.s3 = s3
        self.bucket = bucket
        self.prefix = f"{base}{QUEUE_ROOT}/{name}/"

    def key(self, *parts: str) -> str:
        """Return the object key for a queue path.

        Args:
            *parts: Path segments below the queue prefix.

        Returns:
            Object key.
        """
        return self.prefix + "/".join(parts)

    def put(self, key: str, body: dict[str, Any], **condition: str) -> str | None:
        """Write a JSON object, optionally only if a condition holds.

        Args:
            key: Object key.
            body: JSON-serializable content.
            **condition: ``IfNoneMatch="*"`` or ``IfMatch=<etag>``.

        Returns:
            The new ETag, or None if the condition failed.

        Raises:
            ClientError: For S3 errors other than a failed precondition.
        """
        try:
            response = self.s3.put_object(
                Bucket=self.bucket, Key=key, Body=json.dumps(body).encode(), **condition
            )
        except ClientError as error:
            if error.response["Error"]["Code"] in PRECONDITION_ERRORS:
                return None
            raise
        return str(response["ETag"])

    def put_if_absent(self, key: str, body: dict[str, Any]) -> bool:
        """Create an object only if no object exists at the key.

        Args:
            key: Object key.
            body: JSON-serializable content.

        Returns:
            True if this call created the object; False if it already existed.

        Raises:
            ClientError: For S3 errors other than a failed precondition.
        """
        return self.put(key, body, IfNoneMatch="*") is not None

    def read_lease(self, key: str) -> Lease | None:
        """Read a claim or slot object.

        Args:
            key: Object key.

        Returns:
            The lease, or None if the object does not exist.
        """
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=key)
        except ClientError as error:
            if error.response["Error"]["Code"] in {"NoSuchKey", "404"}:
                return None
            raise
        return Lease(key, json.loads(response["Body"].read()), str(response["ETag"]))

    def put_lease(self, key: str, body: dict[str, Any], **condition: str) -> Lease | None:
        """Write a lease conditionally, recognizing a write that already succeeded.

        When the response to a successful conditional write is lost, the SDK
        retries it, and the retry fails its condition. An object that holds
        exactly ``body`` is therefore this worker's own write.

        Args:
            key: Object key.
            body: Lease content, unique through its expiry time.
            **condition: ``IfNoneMatch="*"`` or ``IfMatch=<etag>``.

        Returns:
            The lease, or None if the condition failed for another writer's object.
        """
        etag = self.put(key, body, **condition)
        if etag is not None:
            return Lease(key, body, etag)
        current = self.read_lease(key)
        if current is not None and current.body == body:
            return current
        return None

    def acquire(self, key: str, owner: dict[str, Any], lease_seconds: float) -> Lease | None:
        """Take a lease on an object that is absent, expired, or interrupted.

        Creation and takeover are conditional writes, so exactly one worker
        obtains each lease. A takeover increments the attempt number.

        Args:
            key: Object key of the claim or slot.
            owner: Description of the worker and job.
            lease_seconds: Seconds until the lease expires unless renewed.

        Returns:
            The lease, or None if another worker holds it.
        """
        current = self.read_lease(key)
        at = time.time()
        body = {
            **owner,
            "attempt": 1,
            "state": "running",
            "renewed": at,
            "expires": at + lease_seconds,
        }
        if current is None:
            # Exactly one worker creates the object; another one leaves it.
            return self.put_lease(key, body, IfNoneMatch="*")
        if not current.takeable(at):
            return None
        body["attempt"] = current.body.get("attempt", 1) + 1
        return self.put_lease(key, body, IfMatch=current.etag)

    def renew(self, lease: Lease, lease_seconds: float, **changes: Any) -> bool:
        """Extend or update a lease if this worker still holds it.

        Args:
            lease: Lease obtained from :meth:`acquire`; updated in place.
            lease_seconds: Seconds from now until the lease expires.
            **changes: Other fields to change, such as ``state``.

        Returns:
            False if another worker took over the lease.
        """
        at = time.time()
        body = {**lease.body, "renewed": at, "expires": at + lease_seconds, **changes}
        written = self.put_lease(lease.key, body, IfMatch=lease.etag)
        if written is None:
            return False
        lease.body, lease.etag = written.body, written.etag
        return True

    def release_lease(self, lease: Lease) -> bool:
        """Release a lease that this worker still holds; see :meth:`end_lease`.

        Args:
            lease: Lease to release.

        Returns:
            False if another worker took over the lease.
        """
        return self.end_lease(lease, "released")

    def end_lease(self, lease: Lease, state: str) -> bool:
        """End a lease that this worker still holds, so another worker can take it.

        The object is overwritten with ``state`` rather than deleted, so that a
        lease another worker took over in the meantime stays intact. If a
        concurrent renewal by this worker gave the object a newer ETag, the
        current object is ended when it has the same owner and attempt.

        Args:
            lease: Lease to end.
            state: ``released`` or ``interrupted``.

        Returns:
            False if another worker took over the lease.
        """
        if self.renew(lease, 0, state=state, expires=0):
            return True
        current = self.read_lease(lease.key)
        if current is None or any(
            current.body.get(k) != lease.body.get(k) for k in ("host", "job", "attempt")
        ):
            return False
        return self.renew(current, 0, state=state, expires=0)

    def names(self, folder: str) -> set[str]:
        """List the object names directly below a queue folder.

        Args:
            folder: Folder name, such as ``claims`` or ``done``.

        Returns:
            Object names without the folder prefix.
        """
        names: set[str] = set()
        prefix = self.key(folder) + "/"
        for page in self.s3.get_paginator("list_objects_v2").paginate(
            Bucket=self.bucket, Prefix=prefix
        ):
            names.update(item["Key"][len(prefix) :] for item in page.get("Contents", []))
        return names

    def read(self, *parts: str) -> bytes:
        """Read a queue object.

        Args:
            *parts: Path segments below the queue prefix.

        Returns:
            Object content.
        """
        return self.s3.get_object(Bucket=self.bucket, Key=self.key(*parts))["Body"].read()

    def acquire_slots(
        self, slots: int, count: int, owner: dict[str, Any], lease_seconds: float
    ) -> list[Lease] | None:
        """Take API slots that bound simultaneous GetQueryResults paging.

        A slot whose lease expired, because its worker died, is taken over.

        Args:
            slots: Number of API slots in the fleet.
            count: Slots needed, one per simultaneous paging query.
            owner: Description stored in each slot object.
            lease_seconds: Seconds until each slot lease expires unless renewed.

        Returns:
            Slot leases, or None if not enough slots are free. Slots taken
            before a shortage or an error are released.

        Raises:
            ValueError: If the job needs more slots than the fleet has.
        """
        needed = max(count, 1)
        if needed > slots:
            raise ValueError(f"A job needs {needed} API slots, but only {slots} exist")
        taken: list[Lease] = []
        try:
            for index in range(slots):
                lease = self.acquire(self.key("slots", str(index)), owner, lease_seconds)
                if lease is not None:
                    taken.append(lease)
                    if len(taken) == needed:
                        return taken
        except BaseException:
            for lease in taken:
                self.release_lease(lease)
            raise
        for lease in taken:
            self.release_lease(lease)
        return None

    def exists(self, *parts: str) -> bool:
        """Return whether a queue object exists.

        Args:
            *parts: Path segments below the queue prefix.

        Returns:
            True if the object exists.
        """
        return bool(
            self.s3.list_objects_v2(Bucket=self.bucket, Prefix=self.key(*parts), MaxKeys=1).get(
                "KeyCount"
            )
        )

    def release(self, key: str) -> None:
        """Delete a queue object such as a slot.

        Args:
            key: Object key.
        """
        self.s3.delete_object(Bucket=self.bucket, Key=key)

    def upload_tree(self, directory: Path, *parts: str, pattern: str = "*") -> None:
        """Upload every file below a directory to a queue path.

        Args:
            directory: Local directory; a missing directory uploads nothing.
            *parts: Destination path segments below the queue prefix.
            pattern: Glob pattern of the files to upload, relative to ``directory``.
        """
        if not directory.exists():
            return
        for path in sorted(p for p in directory.rglob(pattern) if p.is_file()):
            self.s3.upload_file(
                str(path), self.bucket, self.key(*parts, path.relative_to(directory).as_posix())
            )

    def download_tree(self, directory: Path, *parts: str) -> list[str]:
        """Download every object below a queue path into a local directory.

        Args:
            directory: Local destination directory.
            *parts: Source path segments below the queue prefix.

        Returns:
            Relative paths of the downloaded files.
        """
        source = self.key(*parts) + "/"
        names = sorted(self.names("/".join(parts)))
        for name in names:
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(
                self.s3.get_object(Bucket=self.bucket, Key=source + name)["Body"].read()
            )
        return names


def submit(queue: Queue, jobs: list[dict[str, Any]], config: Path, manifest: Path) -> None:
    """Publish jobs, configuration, and manifest for a fleet run.

    Args:
        queue: Destination queue.
        jobs: Jobs from :func:`expand_jobs`.
        config: Configuration file every worker uses for its runs.
        manifest: Prepared manifest shared by every worker.

    Raises:
        ValueError: If the name is reserved or job IDs are not unique.
    """
    if len({j["id"] for j in jobs}) != len(jobs):
        raise ValueError("Job IDs must be unique")
    # The reservation keeps concurrent submissions from mixing files under one name.
    reservation = queue.key("reservation.json")
    if not queue.put_if_absent(reservation, {"host": socket.gethostname(), "at": now()}):
        raise ValueError("Queue name is already reserved; choose a new name")
    try:
        if queue.exists("queue.json"):
            # A published queue owns its name even without a reservation object.
            raise ValueError("Queue already exists; choose a new name")
        for name, path in (("config.toml", config), ("manifest.json", manifest)):
            queue.s3.upload_file(str(path), queue.bucket, queue.key(name))
        queue.s3.put_object(
            Bucket=queue.bucket,
            Key=queue.key("jobs.json"),
            Body=json.dumps(jobs, indent=1).encode(),
        )
        # Workers start only after this marker.
        if not queue.put_if_absent(queue.key("queue.json"), {"created": now(), "jobs": len(jobs)}):
            raise ValueError("Queue already exists; choose a new name")
    except BaseException:
        # Allow the same name to be published again after an interrupted upload.
        queue.release(reservation)
        raise


def run_job(
    job: dict[str, Any],
    workdir: Path,
    output: Path,
    resume: list[Path],
    stop: threading.Event,
) -> int:
    """Run one job attempt as a separate orchestrator process.

    When ``stop`` is set, the orchestrator receives SIGINT, which stops the
    running trial, cancels its reported queries, and records the trial as
    interrupted; it is killed if it has not exited a minute later. The
    orchestrator runs in its own process group, which is killed when it
    ends, so no trial process outlives it.

    Args:
        job: Job dictionary.
        workdir: Directory with the queue's ``config.toml`` and ``manifest.json``.
        output: New output directory for this attempt.
        resume: Output directories of earlier attempts whose completed trials
            are not repeated.
        stop: Event that interrupts the attempt.

    Returns:
        Process exit code.
    """
    (workdir / "logs").mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-m", "pyathena_bench", "--config", str(workdir / "config.toml"),
        "run", "--manifest", str(workdir / "manifest.json"), "--out", str(output), *job["args"],
    ]  # fmt: skip
    if resume:
        command += ["--resume", *map(str, resume)]
    with (workdir / "logs" / f"{job['id']}-{output.name}.log").open("w") as stream:
        process = subprocess.Popen(
            command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            while process.poll() is None:
                if stop.wait(1):
                    process.send_signal(signal.SIGINT)
                    try:
                        return process.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        return process.wait()
            return process.returncode
        finally:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(process.pid, signal.SIGKILL)


def trial_evidence(output: Path) -> tuple[set[str], tuple[str, ...]]:
    """Collect the query IDs and output prefixes that a job's trials used.

    Args:
        output: Job output directory with ``events.jsonl``.

    Returns:
        Reported query IDs and the S3 output prefixes of started trials; both
        are empty if the job wrote no events.
    """
    events = output / "events.jsonl"
    ids: set[str] = set()
    prefixes: set[str] = set()
    if not events.exists():
        return ids, ()
    for line in events.read_text().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event") == "query" and event.get("query_id"):
            ids.add(event["query_id"])
        elif event.get("event") == "trial_start" and event.get("output"):
            prefixes.add(event["output"])
    return ids, tuple(sorted(prefixes))


def quiesce(settings: Settings, output: Path) -> list[str]:
    """Stop a failed job's queries and wait until none is active.

    Besides the reported query IDs, the workgroup history is scanned for queries
    that write under the job's trial prefixes, which covers queries whose IDs
    were never reported before a trial process died.

    Args:
        settings: Configuration with the AWS region, workgroup, and timeouts.
        output: Job output directory with ``events.jsonl``.

    Returns:
        Queries that could not be confirmed as stopped, with the reason.
    """
    reported, prefixes = trial_evidence(output)
    athena = client(session(settings), "athena")
    ids = set(reported)
    if prefixes:
        ids |= active_queries(athena, settings.workgroup, prefixes)
    stop_errors: dict[str, str] = {}
    for query_id in sorted(ids):
        try:
            athena.stop_query_execution(QueryExecutionId=query_id)
        except ClientError as exc:
            stop_errors[query_id] = str(exc)
    errors = []
    deadline = time.monotonic() + settings.timeout_seconds
    for query_id in sorted(ids):
        while True:
            try:
                state = athena.get_query_execution(QueryExecutionId=query_id)["QueryExecution"][
                    "Status"
                ]["State"]
            except ClientError as exc:
                errors.append(f"{query_id}: {stop_errors.get(query_id, exc)}")
                break
            if state not in {"QUEUED", "RUNNING"}:
                break
            if time.monotonic() >= deadline:
                errors.append(f"{query_id}: still {state}; {stop_errors.get(query_id, '')}")
                break
            time.sleep(settings.poll_interval)
    return errors


class SpotInterruptionError(Exception):
    """A Spot interruption notice stopped the worker before every job was claimed."""

    def __init__(self, ran: list[str]) -> None:
        super().__init__("Stopped by a Spot interruption notice")
        self.ran = ran


class JobWatch:
    """Renew a running job's leases, upload its progress, and watch for interruption.

    One thread checks every ``poll_seconds`` for an interruption notice and
    for the claim's expiry, without S3 requests. Another calls ``progress``
    every ``poll_seconds`` and renews the leases every ``renew_seconds``.
    ``stop_requested`` is set on a notice (with ``interrupted``) and when the
    claim, the first lease, is lost: another worker took it over, or
    renewals failed or stalled until it expired. A lost lease is listed in
    ``lost`` and no longer renewed; losing an API slot does not stop the job.
    ``upload_lock`` serializes uploads of the attempt's files.
    """

    def __init__(
        self,
        queue: Queue,
        leases: list[Lease],
        lease_seconds: float,
        renew_seconds: float,
        progress: Callable[[], None],
        notice: Callable[[], bool],
        poll_seconds: float = 5.0,
    ) -> None:
        self.queue = queue
        self.claim = leases[0]
        self.leases = list(leases)
        self.lease_seconds = lease_seconds
        self.renew_seconds = renew_seconds
        self.progress = progress
        self.notice = notice
        self.poll_seconds = min(poll_seconds, renew_seconds)
        self.interrupted = threading.Event()
        self.stop_requested = threading.Event()
        self.lost: list[str] = []
        self.upload_lock = threading.Lock()
        self._stopped = threading.Event()
        self._threads = [
            threading.Thread(target=self._watch_notice, daemon=True),
            threading.Thread(target=self._renew, daemon=True),
        ]

    def __enter__(self) -> JobWatch:
        for thread in self._threads:
            thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        # After a notice or a lost lease, a renewal stuck in S3 must not delay the
        # release of the slots; a late renewal fails against the changed ETags.
        self.stop(10 if self.stop_requested.is_set() else None)

    def stop(self, timeout: float | None = None) -> None:
        """Stop the threads and wait for them.

        Args:
            timeout: Seconds to wait for each thread, or None to wait until it
                ends, after which the caller may update the leases.
        """
        self._stopped.set()
        for thread in self._threads:
            if thread.is_alive():
                thread.join(timeout)

    def _watch_notice(self) -> None:
        while not self._stopped.wait(self.poll_seconds):
            if check_notice(self.notice):
                self.interrupted.set()
                self.stop_requested.set()
                return
            if time.time() >= self.claim.body["expires"]:
                # Renewals failed or stalled; another worker may own the job now.
                self.stop_requested.set()
                return

    def _renew(self) -> None:
        renew_at = time.monotonic() + self.renew_seconds
        while not self._stopped.wait(self.poll_seconds):
            # Renewals come first, so that a slow upload does not delay them.
            if time.monotonic() >= renew_at:
                renew_at = time.monotonic() + self.renew_seconds
                self._renew_leases()
            try:
                with self.upload_lock:
                    self.progress()
            except Exception:
                # The next poll uploads again.
                _logger.warning("Could not upload the attempt's progress", exc_info=True)

    def _renew_leases(self) -> None:
        for lease in list(self.leases):
            try:
                if self.queue.renew(lease, self.lease_seconds):
                    continue
            except Exception:
                if time.time() < lease.body["expires"]:
                    # The lease has time left for a retry.
                    continue
            self.leases.remove(lease)
            self.lost.append(lease.key)
            if lease is self.claim:
                self.stop_requested.set()


def pending_jobs(
    queue: Queue, jobs: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Lease]]:
    """Return the jobs that are neither done nor held by a live claim.

    Args:
        queue: Queue to inspect.
        jobs: Jobs in queue order.

    Returns:
        Jobs without a ``done`` marker whose claim is absent, expired, or
        released, in queue order, and the claims of the other unfinished jobs.
    """
    done = queue.names("done")
    claimed = queue.names("claims")
    at = time.time()
    pending = []
    running: dict[str, Lease] = {}
    for job in jobs:
        if job["id"] in done:
            continue
        if job["id"] in claimed:
            lease = queue.read_lease(queue.key("claims", job["id"]))
            if lease is not None and not lease.takeable(at):
                running[job["id"]] = lease
                continue
        pending.append(job)
    return pending, running


@dataclass(frozen=True)
class Outcome:
    """Result of one job attempt.

    ``state`` is ``done`` when the attempt recorded the job's outcome,
    ``interrupted`` after a Spot interruption notice, and ``lost`` when another
    worker took over the job's claim or a slot. ``errors`` lists queries that
    could not be confirmed as stopped.
    """

    state: str
    errors: tuple[str, ...] = ()


def run_attempt(
    queue: Queue,
    job: dict[str, Any],
    claim: Lease,
    slots: list[Lease],
    workdir: Path,
    runner: Callable[..., int],
    settle: Callable[[Path], list[str]],
    lease_seconds: float,
    renew_seconds: float,
    notice: Callable[[], bool],
) -> Outcome:
    """Run one attempt of a claimed job and record its outcome.

    The attempt writes to ``results/<job>/attempt-<n>/``. A takeover (attempt
    2 or later) first downloads the earlier attempts, quiesces their trials,
    and passes them to the runner so that completed trials are not repeated.
    Trial records are uploaded within seconds of each trial's start and end. An interruption
    notice or a lost claim stops the runner; after a notice, the attempt is
    uploaded and the claim released as interrupted. Otherwise, after a
    conditional renewal confirms that the claim is still held, the job's
    report merges every attempt into ``job-summary.csv`` and
    ``job-summary.md`` of this attempt, and the first ``done`` marker of the
    job records the exit code and names that attempt.

    Args:
        queue: Queue of the job.
        job: Job dictionary.
        claim: Claim lease of the job.
        slots: API slot leases held for the job.
        workdir: Local working directory.
        runner: Function that runs the attempt; see :func:`run_job`.
        settle: Function that quiesces an output directory and returns errors.
        lease_seconds: Seconds a lease lasts without renewal.
        renew_seconds: Seconds between renewals.
        notice: Function that returns whether an interruption notice exists.

    Returns:
        The outcome of the attempt.
    """
    job_dir = workdir / "results" / job["id"]
    attempt = claim.body["attempt"]
    output = job_dir / f"attempt-{attempt}"
    remote = ("results", job["id"], output.name)
    log = workdir / "logs" / f"{job['id']}-{output.name}.log"
    checked: dict[str, int] = {}

    def progress() -> None:
        # Upload at trial boundaries: when a trial starts (its output prefix is
        # needed to quiesce it) and when it is recorded (needed to resume). A
        # boundary stays pending until its upload succeeds.
        events, trials = output / "events.jsonl", output / "trials.jsonl"
        if events.exists():
            with events.open("rb") as stream:
                stream.seek(checked.get("events", 0))
                if b'"trial_start"' in stream.read():
                    checked["pending"] = 1
                checked["events"] = stream.tell()
        if trials.exists() and trials.stat().st_size != checked.get("trials"):
            checked["trials"] = trials.stat().st_size
            checked["pending"] = 1
        if checked.get("pending"):
            for path in (events, trials):
                if path.exists():
                    queue.s3.upload_file(str(path), queue.bucket, queue.key(*remote, path.name))
            checked["pending"] = 0

    with JobWatch(queue, [claim, *slots], lease_seconds, renew_seconds, progress, notice) as watch:

        def upload_attempt(timeout: float = -1) -> None:
            # A checkpoint stuck in S3 holds the lock; it uploads the same files.
            if not watch.upload_lock.acquire(timeout=timeout):
                return
            try:
                queue.upload_tree(output, *remote)
                if log.exists():
                    queue.s3.upload_file(str(log), queue.bucket, queue.key("logs", log.name))
            finally:
                watch.upload_lock.release()

        start = now()
        resume = [
            job_dir / f"attempt-{earlier}"
            for earlier in range(1, attempt)
            if queue.download_tree(
                job_dir / f"attempt-{earlier}", "results", job["id"], f"attempt-{earlier}"
            )
        ]
        errors: list[str] = []
        for earlier_output in resume:
            try:
                errors += settle(earlier_output)
            except Exception as exc:
                errors.append(
                    f"Could not quiesce {earlier_output.name}: {type(exc).__name__}: {exc}"
                )
        code = 1
        if not errors and not watch.stop_requested.is_set():
            code = runner(job, workdir, output, resume, watch.stop_requested)
        if watch.stop_requested.is_set():
            watch.stop(timeout=10)
            # Upload before releasing the claim, so that the next attempt resumes
            # from every recorded trial; both are small within a notice's two minutes.
            upload_attempt(timeout=10)
            if not watch.interrupted.is_set():
                # The claim was lost; another worker continues the job. This
                # attempt's queries are stopped before this worker claims another.
                try:
                    errors += settle(output)
                except Exception as exc:
                    errors.append(f"Could not quiesce the job: {type(exc).__name__}: {exc}")
                return Outcome("lost", tuple(errors))
            queue.end_lease(claim, "interrupted")
            return Outcome("interrupted")
        if not errors:
            try:
                errors = settle(output) if code else []
            except Exception as exc:
                errors = [f"Could not quiesce the job: {type(exc).__name__}: {exc}"]
        upload_attempt()
        report_error = None
        try:
            report(job_dir)
        except Exception as exc:
            # A job without trials, or with unreadable records, still gets its done marker.
            report_error = f"{type(exc).__name__}: {exc}"
        watch.stop()
        # Renewing the claim conditionally proves that this worker still owns the
        # job and leaves at least --lease-seconds for the few seconds of publication.
        if (
            claim.key in watch.lost
            or (watch.stop_requested.is_set() and not watch.interrupted.is_set())
            or not queue.renew(claim, lease_seconds)
        ):
            # The claim was taken over or expired; its new owner records the job.
            return Outcome("lost", tuple(errors))
        if report_error is None:
            # The job's summary goes below this attempt, which the done marker names,
            # so a former owner's late upload cannot replace the recorded summary.
            for name in ("summary.csv", "summary.md"):
                queue.s3.upload_file(
                    str(job_dir / name), queue.bucket, queue.key(*remote, f"job-{name}")
                )
        if time.time() >= claim.body["expires"]:
            # Publication outlasted the renewed claim, which another worker may own now.
            return Outcome("lost", tuple(errors))
        # The first done marker of a job is never replaced.
        recorded = queue.put_lease(
            queue.key("done", job["id"]),
            {
                "exit_code": code,
                "host": socket.gethostname(),
                "attempt": attempt,
                "start": start,
                "end": now(),
                "quiesce_errors": errors,
                "report_error": report_error,
                "summary": None if report_error else "/".join(remote[1:]),
                "lost_slots": [key for key in watch.lost if key != claim.key],
            },
            IfNoneMatch="*",
        )
        if recorded is None:
            return Outcome("lost", tuple(errors))
    return Outcome("done", tuple(errors))


def work(
    queue: Queue,
    workdir: Path,
    api_slots: int,
    poll_seconds: float = 30.0,
    runner: Callable[..., int] = run_job,
    settle: Callable[[Path], list[str]] | None = None,
    lease_seconds: float = 600.0,
    renew_seconds: float = 60.0,
    notice: Callable[[], bool] = lambda: False,
    wait: bool = False,
) -> list[str]:
    """Claim and run queued jobs until no job is left to claim.

    Each host runs one worker, so jobs on a host never overlap. A claim and
    each API slot are leases that the worker renews while the job runs; a job
    or slot whose lease expired, because its worker died, returns to the
    fleet. Before exiting, the worker waits until every other unfinished
    job's claim has been renewed, so a job whose worker died is taken over.
    An API-heavy job starts only after the worker takes one shared slot per
    simultaneous paging query; while it waits, no later API-heavy job takes
    slots. After a failed job, the worker
    cancels the job's observed queries and waits until they stop before
    claiming another job.

    Args:
        queue: Queue to consume.
        workdir: Local directory for queue files, results, and logs.
        api_slots: Fleet-wide number of simultaneous GetQueryResults paging queries.
        poll_seconds: Wait before rescanning when only slot-limited jobs remain,
            and between checks for an unpublished queue.
        runner: Function that runs one attempt; see :func:`run_job`.
        settle: Function that quiesces a failed job, given its output
            directory, and returns errors; defaults to :func:`quiesce` with the
            queued configuration.
        lease_seconds: Seconds a claim or slot lasts without renewal.
        renew_seconds: Seconds between renewals.
        notice: Function that returns whether a Spot interruption notice exists.
        wait: Whether to wait for the queue to be published instead of failing.

    Returns:
        IDs of the jobs this worker ran to completion.

    Raises:
        ValueError: If the queue has not been completely published and
            ``wait`` is false, or a job needs more API slots than ``api_slots``.
        SpotInterruptionError: After an interruption notice, once the running attempt's
            records are uploaded and its claim is released as interrupted.
        RuntimeError: If an attempt stopped by a lost claim could not be quiesced;
            the worker's host is then kept.
    """
    while not queue.exists("queue.json"):
        if not wait:
            raise ValueError("Queue is missing or incompletely published")
        time.sleep(poll_seconds)
    workdir.mkdir(parents=True, exist_ok=True)
    for name in ("config.toml", "manifest.json", "jobs.json"):
        (workdir / name).write_bytes(queue.read(name))
    jobs = json.loads((workdir / "jobs.json").read_text())
    too_wide = [j["id"] for j in jobs if j["api_heavy"] and j.get("api_weight", 1) > api_slots]
    if too_wide:
        raise ValueError(f"Jobs need more than {api_slots} API slots: {', '.join(too_wide)}")
    if settle is None:
        settings = Settings.load(workdir / "config.toml")

        def settle(output: Path) -> list[str]:
            return quiesce(settings, output)

    host = socket.gethostname()
    completed: list[str] = []
    # Renewal times of other workers' claims when this worker found nothing to claim.
    waiting: dict[str, float | None] = {}
    while True:
        if check_notice(notice):
            raise SpotInterruptionError(completed)
        pending, running = pending_jobs(queue, jobs)
        if pending:
            waiting = {}
        else:
            at = time.time()
            for job_id, lease in running.items():
                waiting.setdefault(job_id, lease.body.get("renewed"))
            # Exit once every other claim has been renewed since and is still
            # fresh; a claim whose worker died stops being renewed and is taken
            # over when it expires.
            if all(
                lease.body.get("renewed") is None
                or (
                    lease.body["renewed"] != waiting[job_id]
                    and not lease.stale(at)
                    and not lease.takeable(at)
                )
                for job_id, lease in running.items()
            ):
                return completed
        started = False
        reserved = False
        for job in pending:
            owner = {"host": host, "job": job["id"], "at": now()}
            slots: list[Lease] = []
            if job["api_heavy"]:
                if reserved:
                    continue
                acquired = queue.acquire_slots(
                    api_slots, job.get("api_weight", 1), owner, lease_seconds
                )
                if acquired is None:
                    # Let slots drain for this job instead of starving it with smaller ones.
                    reserved = True
                    continue
                slots = acquired
            try:
                claim = queue.acquire(queue.key("claims", job["id"]), owner, lease_seconds)
                if claim is None:
                    continue
                outcome = run_attempt(
                    queue,
                    job,
                    claim,
                    slots,
                    workdir,
                    runner,
                    settle,
                    lease_seconds,
                    renew_seconds,
                    notice,
                )
                if outcome.state == "interrupted":
                    raise SpotInterruptionError(completed)
                if outcome.state == "lost" and outcome.errors:
                    # The job is unfinished, so the host is kept rather than retired.
                    raise RuntimeError(
                        f"Could not quiesce a stopped attempt of {job['id']}: "
                        + "; ".join(outcome.errors)
                    )
                if outcome.state == "done":
                    completed.append(job["id"])
                if outcome.errors:
                    # Unconfirmed queries could overlap later trials on this host.
                    return completed
                started = True
                break
            finally:
                for slot in slots:
                    queue.release_lease(slot)
        if not started:
            time.sleep(poll_seconds)


def instance_metadata(path: str, timeout: float = 2.0) -> str:
    """Read a value from the EC2 instance metadata service (IMDSv2).

    Args:
        path: Path below ``meta-data/``, such as ``instance-id``.
        timeout: Seconds to wait for each metadata request.

    Returns:
        Metadata value.
    """
    token = urllib.request.Request(
        f"{METADATA_URL}/api/token",
        method="PUT",
        headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
    )
    with urllib.request.urlopen(token, timeout=timeout) as response:
        value = response.read().decode()
    request = urllib.request.Request(
        f"{METADATA_URL}/meta-data/{path}", headers={"X-aws-ec2-metadata-token": value}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode()


def check_notice(notice: Callable[[], bool]) -> bool:
    """Call a notice check, treating an unexpected error as no notice.

    Args:
        notice: Function that returns whether an interruption notice exists.

    Returns:
        The result, or False if the check raised.
    """
    try:
        return notice()
    except Exception:
        _logger.warning("Could not check for an interruption notice", exc_info=True)
        return False


def interruption_notice(
    metadata: Callable[[str], str] | None = None, simulated: Path = SIMULATED_NOTICE
) -> bool:
    """Return whether this host has a Spot interruption notice.

    EC2 publishes ``spot/instance-action`` in instance metadata about two
    minutes before it stops or terminates a Spot Instance; the path does not
    exist otherwise. A file at ``simulated`` counts as a notice, so that a
    pilot can rehearse an interruption.

    Args:
        metadata: Function that reads an instance metadata path; defaults to
            :func:`instance_metadata`.
        simulated: File that counts as a notice.

    Returns:
        True if a notice exists; False also when instance metadata is unavailable.
    """
    if simulated.exists():
        return True
    try:
        (metadata or instance_metadata)("spot/instance-action")
    except (urllib.error.URLError, OSError, http.client.HTTPException):
        return False
    return True


def keep_host(metadata: Callable[[str], str]) -> bool:
    """Return whether the host's ``KeepHost`` instance tag is ``true``.

    The launch template exposes instance tags in instance metadata.

    Args:
        metadata: Function that reads an instance metadata path.

    Returns:
        True if the tag exists with the value ``true``.
    """
    try:
        return metadata(f"tags/instance/{KEEP_HOST_TAG}").strip().lower() == "true"
    except (urllib.error.URLError, OSError):
        return False


def suspend_rebalance(autoscaling: Any, metadata: Callable[[str], str] = instance_metadata) -> str:
    """Suspend zone rebalancing in the host's Auto Scaling group.

    With ``AZRebalance`` active, the group may launch a host in one zone and
    terminate a busy host in another when Spot capacity or retired hosts leave
    the zones uneven. The process stays suspended until the stack is deleted.

    Args:
        autoscaling: Auto Scaling client.
        metadata: Function that reads an instance metadata path.

    Returns:
        Name of the Auto Scaling group.
    """
    host = metadata("instance-id")
    group = autoscaling.describe_auto_scaling_instances(InstanceIds=[host])["AutoScalingInstances"][
        0
    ]["AutoScalingGroupName"]
    autoscaling.suspend_processes(AutoScalingGroupName=group, ScalingProcesses=["AZRebalance"])
    return str(group)


def retire_host(
    queue: Queue,
    autoscaling: Any,
    ran: list[str],
    metadata: Callable[[str], str] = instance_metadata,
) -> bool:
    """Record a finished worker and terminate its host unless the host is kept.

    The record under ``hosts/`` in the queue names the host's instance type,
    Availability Zone, and purchase option. Terminating with a decremented
    desired capacity keeps the group from launching a replacement. A host
    whose ``KeepHost`` tag is ``true`` keeps running.

    Args:
        queue: Queue the worker consumed.
        autoscaling: Auto Scaling client.
        ran: IDs of the jobs the worker ran.
        metadata: Function that reads an instance metadata path.

    Returns:
        True if termination was requested.
    """
    host = metadata("instance-id")
    kept = keep_host(metadata)
    queue.s3.put_object(
        Bucket=queue.bucket,
        Key=queue.key("hosts", f"{host}.json"),
        Body=json.dumps(
            {
                "host": socket.gethostname(),
                "instance_type": metadata("instance-type"),
                "availability_zone": metadata("placement/availability-zone"),
                "lifecycle": metadata("instance-life-cycle"),
                "ran": ran,
                "end": now(),
                "kept": kept,
            }
        ).encode(),
    )
    if kept:
        return False
    autoscaling.terminate_instance_in_auto_scaling_group(
        InstanceId=host, ShouldDecrementDesiredCapacity=True
    )
    return True


def status(queue: Queue) -> dict[str, Any]:
    """Summarize queue progress, failed jobs, and leases.

    Args:
        queue: Queue to inspect.

    Returns:
        Counts of jobs, claims, and completions, with failed job details, the
        claims of unfinished jobs, and the API slots in use, each lease with
        its host, attempt, state, and whether it has expired.
    """
    if not queue.exists("queue.json"):
        raise ValueError("Queue is missing or incompletely published")
    jobs = json.loads(queue.read("jobs.json"))
    done = queue.names("done")
    failed = {}
    for name in sorted(done):
        record = json.loads(queue.read("done", name))
        if record["exit_code"] != 0:
            failed[name] = record
    at = time.time()

    def describe(folder: str, names: Iterable[str]) -> dict[str, Any]:
        leases = {}
        for name in sorted(names):
            lease = queue.read_lease(queue.key(folder, name))
            if lease is not None and lease.body.get("state") != "released":
                leases[name] = {
                    **{k: lease.body.get(k) for k in ("host", "job", "attempt", "state")},
                    "expired": lease.takeable(at) and lease.body.get("state") == "running",
                    "stale": lease.stale(at),
                }
        return leases

    claims = queue.names("claims")
    return {
        "jobs": len(jobs),
        "claimed": len(claims),
        "done": len(done),
        "running": describe("claims", claims - done),
        "slots_in_use": describe("slots", queue.names("slots")),
        "failed": failed,
    }
