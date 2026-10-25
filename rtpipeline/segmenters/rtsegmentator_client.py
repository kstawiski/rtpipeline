# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Command-line and Python client for the RTsegmentator automation API.

The client uses only the Python standard library so that other tools, such as
RTpipeline, can vendor this single file. It uploads DICOM files through
resumable upload sessions, selects series from the portal's own assessment,
submits one job per series, waits for completion, downloads result packages and
verifies them against their packaged ``rtsegmentator_manifest.json``.

Typical use::

    python rtseg_client.py --url https://portal.example models
    python rtseg_client.py --url https://portal.example submit ./CT_SERIES \
        --model total --model lung_nodules --out ./results

Authentication is not part of RTsegmentator itself. When the portal sits behind
an authenticating reverse proxy, pass headers with ``--header`` or the
``RTSEG_CLIENT_HEADERS`` environment variable (``Name: value`` pairs separated by
newlines).
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

CLIENT_VERSION = "1.0.0"
API_PREFIX = "/api/v1"
MANIFEST_NAME = "rtsegmentator_manifest.json"
MANIFEST_SCHEMA = "rtsegmentator.result-manifest"
SUPPORTED_MANIFEST_MAJOR = "1"
TERMINAL_STATES = {"complete", "failed"}
JOB_ID_PATTERN = re.compile(r"^[A-Za-z0-9-]+$")
DEFAULT_CHUNK_BYTES = 8 * 1024 * 1024
STREAM_BLOCK = 1024 * 1024
MAX_MANIFEST_BYTES = 64 * 1024 * 1024


class ClientError(RuntimeError):
    """An API request failed or returned an unusable response."""

    def __init__(self, message: str, status: int | None = None, payload: object = None):
        super().__init__(message)
        self.status = status
        self.payload = payload


class JobFailed(ClientError):
    """A job reached the ``failed`` state."""


class ManifestError(ClientError):
    """A result package is missing its manifest or fails verification."""


def parse_headers(values: Iterable[str]) -> dict[str, str]:
    headers = {}
    for value in values:
        for line in str(value).splitlines():
            if not line.strip():
                continue
            name, separator, content = line.partition(":")
            if not separator or not name.strip():
                raise ValueError(f"Header must look like 'Name: value': {line!r}")
            headers[name.strip()] = content.strip()
    return headers


def checked_job_id(job_id: object) -> str:
    """A portal job ID, which may only contain letters, digits and hyphens."""
    if not isinstance(job_id, str) or not JOB_ID_PATTERN.match(job_id):
        raise ClientError(f"Invalid job ID: {job_id!r}")
    return job_id


class _SameOriginRedirects(urllib.request.HTTPRedirectHandler):
    """Follow redirects only within the configured origin, so proxy credentials stay there."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        old, new = urllib.parse.urlsplit(req.full_url), urllib.parse.urlsplit(newurl)
        if (old.scheme, old.hostname, old.port) != (new.scheme, new.hostname, new.port):
            raise ClientError(f"Refusing HTTP {code} redirect from {old.scheme}://{old.netloc} to another "
                              f"origin ({new.scheme}://{new.netloc})", status=code)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _member_digest(archive: zipfile.ZipFile, name: str) -> tuple[int, str]:
    """Size and SHA-256 of one member, read in blocks so large maps never sit in memory."""
    digest, size = hashlib.sha256(), 0
    with archive.open(name) as handle:
        for block in iter(lambda: handle.read(STREAM_BLOCK), b""):
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest()


def unsafe_member_name(name: str) -> bool:
    """True unless ``name`` is a canonical relative POSIX path (no '.', '..', empty or absolute parts)."""
    parts = name.split("/")
    return (not name or name.startswith("/") or "\\" in name or ":" in parts[0]
            or any(part in {"", ".", ".."} for part in parts))


@dataclass
class RTsegClient:
    """Thin wrapper around the versioned ``/api/v1`` routes."""

    base_url: str
    headers: Mapping[str, str] = field(default_factory=dict)
    timeout: float = 120.0
    ca_bundle: str | None = None
    retries: int = 4
    sleep: Callable[[float], None] = time.sleep

    def __post_init__(self):
        parsed = urllib.parse.urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"RTsegmentator URL must be http(s)://host[:port]: {self.base_url!r}")
        self.base_url = self.base_url.rstrip("/")
        handlers: list = [_SameOriginRedirects()]
        if parsed.scheme == "https":
            handlers.append(urllib.request.HTTPSHandler(context=ssl.create_default_context(cafile=self.ca_bundle)))
        self._opener = urllib.request.build_opener(*handlers)

    def _open(self, request: urllib.request.Request):
        return self._opener.open(request, timeout=self.timeout)

    # -- transport ---------------------------------------------------------
    def _request(self, method: str, path: str, *, query: Mapping | None = None, body: bytes | None = None,
                 content_type: str | None = None, expect_json: bool = True, ok: Sequence[int] = (200, 202),
                 idempotent: bool = True, sink: Path | None = None):
        """Send one request; ``sink`` streams a successful body to that file instead of returning it.

        Transient failures are retried only for idempotent requests. A request that may
        have taken effect on the server (job creation) is retried only after HTTP 429,
        which the portal returns before doing any work.
        """
        url = self.base_url + path
        if query:
            url += "?" + urllib.parse.urlencode(query, doseq=True)
        headers = {"Accept": "application/json", "User-Agent": f"rtseg-client/{CLIENT_VERSION}", **self.headers}
        if content_type:
            headers["Content-Type"] = content_type
        attempt = 0
        while True:
            request = urllib.request.Request(url, data=body, method=method, headers=headers)
            try:
                with self._open(request) as response:
                    status = response.status
                    if sink is not None and status in ok:
                        return self._stream_to(response, Path(sink))
                    data = response.read()
            except urllib.error.HTTPError as exc:
                status, data = exc.code, exc.read()
            except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException) as exc:
                if not idempotent:
                    raise ClientError(f"{method} {path} failed and was not retried because it may have taken "
                                      f"effect; check the portal's job list before resubmitting: {exc}") from exc
                if attempt >= self.retries:
                    raise ClientError(f"{method} {path} failed: {exc}") from exc
                attempt += 1
                self.sleep(min(2 ** attempt, 30))
                continue
            # 429: a session lock is briefly held; 502-504: a proxy or restart.
            retryable = {429, 502, 503, 504} if idempotent else {429}
            if status in retryable and attempt < self.retries:
                attempt += 1
                self.sleep(min(2 ** attempt, 30))
                continue
            payload = None
            if expect_json or status not in ok:
                try:
                    payload = json.loads(data.decode("utf-8")) if data else None
                except (UnicodeDecodeError, json.JSONDecodeError):
                    payload = data[:500].decode("utf-8", "replace")
            if status not in ok:
                detail = payload.get("detail") if isinstance(payload, dict) else payload
                raise ClientError(f"{method} {path} returned HTTP {status}: {detail}", status, payload)
            if expect_json and payload is not None and not isinstance(payload, (dict, list)):
                # For example an HTML login page from an authenticating proxy.
                raise ClientError(f"{method} {path} returned a non-JSON response (HTTP {status}): "
                                  f"{str(payload)[:200]}", status, payload)
            return payload if expect_json else data

    @staticmethod
    def _stream_to(response, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + ".partial")
        try:
            with partial.open("wb") as handle:
                for block in iter(lambda: response.read(STREAM_BLOCK), b""):
                    handle.write(block)
            os.replace(partial, destination)
        finally:
            partial.unlink(missing_ok=True)
        return destination

    def _form(self, path: str, fields: Sequence[tuple[str, str]], *, idempotent: bool = True):
        return self._request("POST", path, body=urllib.parse.urlencode(list(fields)).encode(),
                             content_type="application/x-www-form-urlencoded", idempotent=idempotent)

    # -- read-only routes --------------------------------------------------
    def health(self) -> dict:
        return self._request("GET", f"{API_PREFIX}/health")

    def config(self) -> dict:
        return self._request("GET", f"{API_PREFIX}/config")

    def models(self, *, enabled_only: bool = False) -> list[dict]:
        tasks = (self._request("GET", f"{API_PREFIX}/models") or {}).get("tasks", [])
        return [task for task in tasks if task.get("enabled") or not enabled_only]

    def job(self, job_id: str) -> dict:
        return self._request("GET", f"{API_PREFIX}/jobs/{checked_job_id(job_id)}")

    def manifest(self, job_id: str) -> dict:
        return self._request("GET", f"{API_PREFIX}/jobs/{checked_job_id(job_id)}/manifest")

    # -- upload and submission --------------------------------------------
    def upload_files(self, files: Sequence[Path], *, progress: Callable[[int, int], None] | None = None) -> str:
        """Send files through one resumable upload session and return its id."""
        files = [Path(path) for path in files]
        if not files:
            raise ClientError("No files to upload")
        session = self._request("POST", f"{API_PREFIX}/upload-sessions")
        session_id = session["session_id"]
        chunk_limit = max(1, min(int(session.get("max_request_bytes") or DEFAULT_CHUNK_BYTES), DEFAULT_CHUNK_BYTES))
        total = sum(path.stat().st_size for path in files)
        if session.get("max_total_bytes") and total > int(session["max_total_bytes"]):
            raise ClientError(f"Upload of {total} bytes exceeds the portal limit of {session['max_total_bytes']}")
        sent = 0
        for index, path in enumerate(files):
            size = path.stat().st_size
            offset = 0
            with path.open("rb") as handle:
                while offset < size or (size == 0 and offset == 0):
                    handle.seek(offset)
                    chunk = handle.read(chunk_limit)
                    if not chunk and offset < size:
                        raise ClientError(f"{path} changed size during upload")
                    try:
                        reply = self._request(
                            "PUT", f"{API_PREFIX}/upload-sessions/{session_id}/files/{index}",
                            query={"name": path.name, "offset": offset, "size": size}, body=chunk,
                            content_type="application/octet-stream")
                    except ClientError as exc:
                        expected = exc.payload.get("expected_offset") if isinstance(exc.payload, dict) else None
                        if exc.status == 409 and isinstance(expected, int) and 0 <= expected <= size:
                            sent += expected - offset
                            offset = expected
                            continue
                        raise
                    received = int(reply["received"])
                    sent += received - offset
                    offset = received
                    if progress:
                        progress(sent, total)
                    if size == 0:
                        break
        return session_id

    def assess(self, session_id: str, *, poll: float = 2.0, timeout: float | None = 3600) -> dict:
        """Start assessment and return the assessed upload (``upload_id`` and ``series``)."""
        self._request("POST", f"{API_PREFIX}/upload-sessions/{session_id}/assess")
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            state = self._request("GET", f"{API_PREFIX}/upload-sessions/{session_id}")
            if state.get("status") == "ready":
                return state["result"]
            if state.get("status") == "failed":
                raise ClientError(f"Upload assessment failed: {state.get('error')}", payload=state)
            if deadline is not None and time.monotonic() > deadline:
                raise ClientError("Timed out waiting for upload assessment")
            self.sleep(poll)

    def submit(self, upload_id: str, series_keys: Sequence[str], models: Sequence[str], *,
               prompt_json: Mapping | None = None, roles: Mapping | None = None,
               task_options: Mapping | None = None, postprocess: str = "") -> dict:
        """Create one job per selected series from an assessed upload."""
        if not series_keys or not models:
            raise ClientError("Select at least one series and one model")
        fields = [("series_keys", key) for key in series_keys] + [("models", name) for name in models]
        for name, value in (("prompt_json", prompt_json), ("roles", roles), ("task_options", task_options)):
            if value:
                fields.append((name, json.dumps(value)))
        if postprocess:
            fields.append(("postprocess", postprocess))
        # Job creation consumes the assessed upload, so it is never replayed after an ambiguous failure.
        return self._form(f"{API_PREFIX}/uploads/{urllib.parse.quote(upload_id, safe='')}/jobs", fields,
                          idempotent=False)

    def wait(self, job_id: str, *, poll: float = 10.0, timeout: float | None = None,
             progress: Callable[[dict], None] | None = None) -> dict:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            record = self.job(job_id)
            if progress:
                progress(record)
            if record.get("status") == "complete":
                return record
            if record.get("status") == "failed":
                raise JobFailed(f"Job {job_id} failed: {record.get('error') or record.get('detail')}",
                                payload=record)
            if deadline is not None and time.monotonic() > deadline:
                raise ClientError(f"Timed out waiting for job {job_id} (status {record.get('status')!r})")
            self.sleep(poll)

    def download(self, job_id: str, destination: Path) -> Path:
        """Stream a completed job's result ZIP to ``destination`` (written atomically)."""
        return self._request("GET", f"{API_PREFIX}/jobs/{checked_job_id(job_id)}/result",
                             expect_json=False, sink=Path(destination))


# -- result packages ---------------------------------------------------------
def read_manifest(archive: zipfile.ZipFile) -> dict:
    try:
        info = archive.getinfo(MANIFEST_NAME)
    except KeyError as exc:
        raise ManifestError(f"Result package has no {MANIFEST_NAME}") from exc
    if info.file_size > MAX_MANIFEST_BYTES:
        raise ManifestError(f"{MANIFEST_NAME} exceeds {MAX_MANIFEST_BYTES} bytes")
    try:
        manifest = json.loads(archive.read(info).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"{MANIFEST_NAME} is not valid JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ManifestError(f"{MANIFEST_NAME} must contain a JSON object")
    return manifest


def verify_result_zip(path: Path) -> dict:
    """Return the manifest of a result ZIP after checking every listed member's size and SHA-256.

    Members not listed in the manifest, missing members, unsafe or noncanonical names,
    names that collide when compared case-insensitively, and checksum mismatches are
    all errors.
    """
    with zipfile.ZipFile(path) as archive:
        manifest = read_manifest(archive)
        problems = []
        if manifest.get("schema") != MANIFEST_SCHEMA:
            problems.append(f"unexpected schema {manifest.get('schema')!r}")
        if str(manifest.get("schema_version", "")).split(".")[0] != SUPPORTED_MANIFEST_MAJOR:
            problems.append(f"unsupported schema version {manifest.get('schema_version')!r}")
        listed = {}
        for record in manifest.get("files") or []:
            name = record.get("path") if isinstance(record, dict) else None
            if not isinstance(name, str) or not name or name in listed:
                problems.append(f"invalid or duplicate file record {record!r}")
                continue
            listed[name] = record
        for info in archive.infolist():
            # A directory entry carries no bytes; one that does would hide an unchecked payload.
            if info.is_dir() and info.file_size:
                problems.append(f"directory entry with content {info.filename!r}")
        members = [info for info in archive.infolist() if not info.is_dir()]
        names = [info.filename for info in members]
        for name in names:
            if unsafe_member_name(name):
                problems.append(f"unsafe member name {name!r}")
        # Two names that one filesystem could map to the same file would let one
        # verified member silently replace another during extraction.
        if len({name.casefold() for name in names}) != len(names):
            problems.append("duplicate member names")
        unlisted = sorted(set(names) - set(listed) - {MANIFEST_NAME})
        if unlisted:
            problems.append("members missing from the manifest: " + ", ".join(unlisted))
        for name, record in listed.items():
            if name not in names:
                problems.append(f"missing member {name}")
                continue
            size, digest = _member_digest(archive, name)
            if size != record.get("size") or digest != record.get("sha256"):
                problems.append(f"checksum mismatch for {name}")
        # Every RTSTRUCT a model entry points to must be a verified member.
        for model in manifest.get("models") or []:
            path = (model.get("rtstruct") or {}).get("path") if isinstance(model, dict) else None
            if path is not None and path not in listed:
                problems.append(f"model RTSTRUCT {path!r} is not a listed member")
        if problems:
            raise ManifestError("Result package failed verification: " + "; ".join(problems), payload=problems)
        return manifest


def extract_result_zip(path: Path, destination: Path) -> dict:
    """Verify a result ZIP, then extract it below ``destination``; returns the manifest."""
    manifest = verify_result_zip(path)
    destination = Path(destination)
    root = destination.resolve()
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            target = (destination / info.filename).resolve()
            if root != target and root not in target.parents:
                raise ManifestError(f"Member escapes the destination: {info.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise ManifestError(f"Refusing to overwrite an existing file: {target}")
            with archive.open(info) as source, target.open("xb") as handle:
                while chunk := source.read(STREAM_BLOCK):
                    handle.write(chunk)
    return manifest


def collect_files(paths: Iterable[Path]) -> list[Path]:
    """Expand directories recursively; hidden files and directories are skipped."""
    found = []
    for path in map(Path, paths):
        if path.is_dir():
            for candidate in sorted(path.rglob("*")):
                relative = candidate.relative_to(path).parts
                if candidate.is_file() and not any(part.startswith(".") for part in relative):
                    found.append(candidate)
        elif path.is_file():
            found.append(path)
        else:
            raise FileNotFoundError(path)
    return found


def model_modalities(catalog: Iterable[Mapping], models: Iterable[str]) -> set[str]:
    """Modalities the named catalog tasks accept (``modality`` and ``modalities`` fields)."""
    wanted, found = set(models), set()
    for task in catalog:
        if task.get("name") in wanted:
            found.update(str(m).upper() for m in (task.get("modalities") or [task.get("modality")]) if m)
    return found


def verify_against_portal(client: "RTsegClient", job_id: str, package: Path) -> dict:
    """Verify a downloaded package and check that its manifest is the one the portal serves.

    The manifest is not signed: verification alone detects corruption and incomplete
    transfers, and comparing it with the portal's copy detects a package that does not
    belong to this job.
    """
    manifest = verify_result_zip(package)
    if client.manifest(job_id) != manifest:
        raise ManifestError(f"The packaged manifest of {package} differs from the portal's manifest for job {job_id}")
    return manifest


def select_series(series: Sequence[Mapping], *, modalities: Iterable[str] = (),
                  series_uids: Iterable[str] = ()) -> list[Mapping]:
    """Series chosen by exact SeriesInstanceUID, else by modality; unavailable series are never chosen."""
    wanted_uids, wanted_modalities = set(series_uids), {m.upper() for m in modalities}
    chosen = []
    for item in series:
        if item.get("unavailable_reason"):
            continue
        if wanted_uids and item.get("series_uid") not in wanted_uids:
            continue
        if wanted_modalities and str(item.get("modality", "")).upper() not in wanted_modalities:
            continue
        chosen.append(item)
    missing = wanted_uids - {item.get("series_uid") for item in chosen}
    if missing:
        raise ClientError("Requested series were not found or are unavailable: " + ", ".join(sorted(missing)))
    return chosen


# -- command line --------------------------------------------------------------
def _client(args) -> RTsegClient:
    headers = parse_headers([os.environ.get("RTSEG_CLIENT_HEADERS", ""), *args.header])
    url = args.url or os.environ.get("RTSEG_URL")
    if not url:
        raise SystemExit("Pass --url or set RTSEG_URL")
    return RTsegClient(url, headers=headers, timeout=args.timeout, ca_bundle=args.ca_bundle)


def _print(value) -> None:
    print(json.dumps(value, indent=2))


def _command_submit(args) -> int:
    client = _client(args)
    files = collect_files(args.paths)
    stderr = sys.stderr
    print(f"Uploading {len(files)} files", file=stderr)
    session = client.upload_files(files, progress=lambda sent, total: print(
        f"\r  {sent / max(total, 1):6.1%} of {total / 1e6:.1f} MB", end="", file=stderr))
    print(file=stderr)
    upload = client.assess(session)
    modalities = args.modality
    if not modalities and not args.series_uid:
        # Default to the series the selected models can segment, so a scout or a
        # companion series of another modality does not make the portal reject the batch.
        modalities = sorted(model_modalities(client.models(), args.model))
    chosen = select_series(upload.get("series", []), modalities=modalities, series_uids=args.series_uid)
    if not chosen:
        raise ClientError("No available series match the selection")
    batch = client.submit(upload["upload_id"], [item["key"] for item in chosen], args.model,
                          prompt_json=json.loads(args.prompt_json) if args.prompt_json else None,
                          roles=json.loads(args.roles) if args.roles else None,
                          postprocess=args.postprocess or "")
    if not args.wait and not args.out:
        _print(batch)
        return 0
    results, failed = [], False
    for job in batch.get("jobs", []):
        checked_job_id(job.get("id"))
        entry = {"id": job["id"], "series_key": job.get("series_key")}
        try:
            record = client.wait(job["id"], poll=args.poll, timeout=args.job_timeout,
                                 progress=lambda r: print(f"\r  {r['id']}: {r.get('status')} {r.get('detail', '')}"[:160]
                                                          .ljust(160), end="", file=stderr))
            entry["status"] = record.get("status")
            if args.out:
                target = Path(args.out) / f"{job['id']}.zip"
                client.download(job["id"], target)
                manifest = verify_against_portal(client, job["id"], target)
                entry.update(result=str(target), models=[
                    {"name": m["name"], "outcome": m["outcome"], "rois": len((m.get("rtstruct") or {}).get("roi_names", []))}
                    for m in manifest["models"]])
        except ClientError as exc:
            # Report every job of the batch; one failure does not hide the others.
            failed = True
            entry.update(status=entry.get("status") or "failed", error=str(exc))
        print(file=stderr)
        results.append(entry)
    _print({"batch_id": batch.get("batch_id"), "jobs": results})
    return 1 if failed else 0


def _command_download(args) -> int:
    client = _client(args)
    checked_job_id(args.job_id)
    target = client.download(args.job_id, Path(args.out) / f"{args.job_id}.zip")
    manifest = verify_against_portal(client, args.job_id, target)
    if args.extract:
        extract_result_zip(target, Path(args.out) / args.job_id)
    _print({"result": str(target), "generator": manifest.get("generator")})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rtseg_client.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", help="Portal base URL (default: RTSEG_URL)")
    parser.add_argument("--header", action="append", default=[], help="Extra request header 'Name: value'")
    parser.add_argument("--ca-bundle", help="CA bundle for HTTPS verification")
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-request timeout in seconds")
    parser.add_argument("--version", action="version", version=f"rtseg-client {CLIENT_VERSION}")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("health", help="Portal and worker health")
    models = commands.add_parser("models", help="List models")
    models.add_argument("--enabled", action="store_true", help="Only models enabled on this portal")
    models.add_argument("--json", action="store_true", help="Print full catalog entries")
    submit = commands.add_parser("submit", help="Upload DICOM files and submit one job per selected series")
    submit.add_argument("paths", nargs="+", help="DICOM files, directories or ZIP archives")
    submit.add_argument("--model", action="append", required=True, help="Model name; repeat for several")
    submit.add_argument("--modality", action="append", default=[], help="Restrict to CT, MR or PT series")
    submit.add_argument("--series-uid", action="append", default=[], help="Restrict to exact SeriesInstanceUIDs")
    submit.add_argument("--prompt-json", help="Prompt JSON object keyed by series key (prompted models)")
    submit.add_argument("--roles", help="MRI input-role JSON object keyed by primary series key")
    submit.add_argument("--postprocess", help="Optional postprocess value, e.g. prostate_sbrt_crop")
    submit.add_argument("--wait", action="store_true", help="Wait for every job to finish")
    submit.add_argument("--out", help="Download and verify each result ZIP into this directory (implies --wait)")
    submit.add_argument("--poll", type=float, default=10.0, help="Seconds between status checks")
    submit.add_argument("--job-timeout", type=float, default=None, help="Seconds to wait for each job")
    status = commands.add_parser("status", help="Show one job")
    status.add_argument("job_id")
    download = commands.add_parser("download", help="Download and verify a completed job's result")
    download.add_argument("job_id")
    download.add_argument("--out", default=".", help="Destination directory")
    download.add_argument("--extract", action="store_true", help="Also extract into OUT/JOB_ID")
    verify = commands.add_parser("verify", help="Verify a downloaded result ZIP against its manifest")
    verify.add_argument("zip")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "verify":
            manifest = verify_result_zip(Path(args.zip))
            _print({"verified": True, "job": manifest.get("job", {}).get("id"),
                    "files": len(manifest.get("files", [])), "generator": manifest.get("generator")})
            return 0
        if args.command == "health":
            _print(_client(args).health())
            return 0
        if args.command == "models":
            tasks = _client(args).models(enabled_only=args.enabled)
            if args.json:
                _print(tasks)
            else:
                for task in tasks:
                    flag = "enabled " if task.get("enabled") else "disabled"
                    print(f"{flag}  {task.get('modality', ''):<3}  {task['name']:<40} {task.get('status', '')}")
            return 0
        if args.command == "status":
            _print(_client(args).job(args.job_id))
            return 0
        if args.command == "submit":
            return _command_submit(args)
        if args.command == "download":
            return _command_download(args)
    except (ClientError, OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
