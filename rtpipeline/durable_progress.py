"""Durable synthetic/aggregate progress primitives; standard library, Python >=3.9.

Use one writer per journal. Start GuardSafeSink before installing an audit guard.
Successful sink calls acknowledge fsync, not merely delivery to a pipe. Cache
transactions are immutable; invalid and interrupted transactions remain on disk.
This module does not discover inputs, select scientific gates, or decrypt files.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
from typing import Any, Callable, Iterable, Mapping, Optional

SCHEMA_VERSION = 1
WRITE_FAILED = "DURABLE_WRITE_FAILED"
READ_FAILED = "DURABLE_READ_FAILED"
VERIFY_FAILED = "DURABLE_VERIFY_FAILED"
_HEX = re.compile(r"[0-9a-f]{64}\Z")


class DurableProgressError(RuntimeError):
    """A fixed, machine-readable code; underlying I/O errors remain chained."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _loads(raw: bytes) -> Any:
    def nonfinite(_):
        raise ValueError("nonfinite JSON")
    return json.loads(raw, object_pairs_hook=_pairs, parse_constant=nonfinite)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path) -> str:
    """Hash every byte, with no source filtering or newline normalization."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Binding:
    code_sha256: str
    config_sha256: str
    inputs_sha256: str

    def __post_init__(self):
        for value in self.as_dict().values():
            if not isinstance(value, str) or _HEX.fullmatch(value) is None:
                raise DurableProgressError("DURABLE_INVALID_BINDING")

    def as_dict(self) -> dict:
        return {"code_sha256": self.code_sha256,
                "config_sha256": self.config_sha256,
                "inputs_sha256": self.inputs_sha256}


def _binding(value) -> Binding:
    if isinstance(value, Binding):
        return value
    if not isinstance(value, Mapping) or set(value) != {
            "code_sha256", "config_sha256", "inputs_sha256"}:
        raise DurableProgressError("DURABLE_INVALID_BINDING")
    return Binding(**value)


def make_binding(code_files: Mapping[str, Any], config: Any, inputs: Any) -> Binding:
    """Freeze an explicit code manifest, configuration, and input fingerprint.

    code_files maps stable logical names to the exact files used, including
    helpers. config/inputs are JSON values, or bytes hashed without conversion.
    The caller must keep these files and inputs unchanged throughout the run.
    """
    if not code_files or any(not isinstance(name, str) for name in code_files):
        raise DurableProgressError("DURABLE_INVALID_BINDING")
    manifest = [{"path": name, "sha256": file_sha256(code_files[name])}
                for name in sorted(code_files)]
    def fingerprint(value):
        return _sha(value if isinstance(value, bytes) else _json(value))
    return Binding(_sha(_json(manifest)), fingerprint(config), fingerprint(inputs))


def _write_all(stream, raw: bytes):
    view = memoryview(raw)
    while view:
        written = stream.write(view)
        if not written:
            raise OSError("short write")
        view = view[written:]


def _fsync_dir(path: Path):
    fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write(path: Path, raw: bytes) -> dict:
    """Publish in an existing directory and verify actual destination bytes."""
    temporary = None
    try:
        fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=str(path.parent))
        with os.fdopen(fd, "wb") as stream:
            _write_all(stream, raw)
            stream.flush()
            os.fsync(stream.fileno())
        # Verify before replacement as well, preserving an old destination on a
        # short/corrupt write even when the filesystem did not raise an error.
        if Path(temporary).read_bytes() != raw:
            raise DurableProgressError(VERIFY_FAILED)
        os.replace(temporary, path)
        temporary = None
        _fsync_dir(path.parent)
        if path.read_bytes() != raw:
            raise DurableProgressError(VERIFY_FAILED)
        return {"bytes": len(raw), "sha256": _sha(raw), "verified": True}
    except OSError as error:
        raise DurableProgressError(WRITE_FAILED) from error
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            except OSError as error:
                raise DurableProgressError(WRITE_FAILED) from error


@dataclass
class JournalRead:
    records: list
    report: dict


class ProgressJournal:
    """Append-only JSONL, with strict sequence and envelope SHA-256 validation.

    read() excludes and reports a last line lacking its newline. Reopening that
    file for append refuses DURABLE_JOURNAL_TRUNCATED: archive the damaged
    journal explicitly and use a new one; no repair silently deletes evidence.
    """

    def __init__(self, path):
        self.path = Path(path)
        self._stream = None
        self._failed = False
        self._lock = threading.Lock()
        previous = self.read(self.path)
        if previous.report["truncated_last_line"]:
            raise DurableProgressError("DURABLE_JOURNAL_TRUNCATED")
        self._sequence = len(previous.records)
        try:
            self._stream = open(self.path, "ab", buffering=0)
            os.fsync(self._stream.fileno())
            _fsync_dir(self.path.parent)
        except OSError as error:
            if self._stream is not None:
                self._stream.close()
            raise DurableProgressError(WRITE_FAILED) from error

    @staticmethod
    def read(path) -> JournalRead:
        records = []
        truncated = 0
        try:
            stream = open(path, "rb")
        except FileNotFoundError:
            return JournalRead([], {"records": 0, "truncated_last_line": 0})
        except OSError as error:
            raise DurableProgressError(READ_FAILED) from error
        try:
            with stream:
                for line in stream:
                    if not line.endswith(b"\n"):
                        truncated = 1
                        break
                    try:
                        record = _loads(line)
                        if not isinstance(record, dict) or set(record) != {
                                "schema_version", "sequence", "utc", "body", "sha256"}:
                            raise ValueError("envelope")
                        body = {k: v for k, v in record.items() if k != "sha256"}
                        if (type(record["schema_version"]) is not int
                                or record["schema_version"] != SCHEMA_VERSION
                                or type(record["sequence"]) is not int
                                or record["sequence"] != len(records) + 1
                                or not isinstance(record["body"], dict)
                                or record["sha256"] != _sha(_json(body))):
                            raise ValueError("integrity")
                        utc = record["utc"]
                        if not isinstance(utc, str) or not utc.endswith("Z"):
                            raise ValueError("UTC required")
                        datetime.fromisoformat(utc[:-1] + "+00:00")
                    except (ValueError, TypeError, KeyError, UnicodeError, RecursionError) as error:
                        raise DurableProgressError("DURABLE_JOURNAL_CORRUPT") from error
                    records.append(record)
        except OSError as error:
            raise DurableProgressError(READ_FAILED) from error
        return JournalRead(records, {"records": len(records), "truncated_last_line": truncated})

    def append(self, body: dict) -> dict:
        if not isinstance(body, dict):
            raise TypeError("journal body must be a JSON object")
        with self._lock:
            if self._stream is None or self._failed:
                raise DurableProgressError("DURABLE_JOURNAL_CLOSED")
            record = {"schema_version": SCHEMA_VERSION, "sequence": self._sequence + 1,
                      "utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                      "body": body}
            record["sha256"] = _sha(_json(record))
            raw = _json(record) + b"\n"
            try:
                _write_all(self._stream, raw)
                self._stream.flush()
                os.fsync(self._stream.fileno())
            except OSError as error:
                self._failed = True  # A partial append must never be followed by another.
                raise DurableProgressError(WRITE_FAILED) from error
            self._sequence += 1
            return record

    def close(self):
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


@dataclass
class CacheRead:
    entries: dict
    report: dict


class ResultCache:
    """Immutable two-phase transactions, addressed by item key and exact binding.

    seal(bytes)->bytes runs before any payload is written. Sealed bytes have an
    .age filename and are returned as bytes unless unseal(bytes)->bytes is given.
    Metadata is plaintext: item keys MUST be nonidentifying opaque identifiers.
    One writer per logical item/binding is required. Conflicting committed
    results are excluded, rather than choosing a result by timestamp.
    """

    def __init__(self, root, *, seal: Optional[Callable] = None,
                 unseal: Optional[Callable] = None):
        self.root = Path(root)
        self.seal = seal
        self.unseal = unseal

    @staticmethod
    def _key(item_key: str, binding: Binding) -> str:
        if not isinstance(item_key, str):
            raise TypeError("item_key must be a string")
        return _sha(_json({"item_key": item_key, "binding": binding.as_dict()}))

    def _prepare(self, item_key, result, binding) -> tuple:
        binding = _binding(binding)
        self._key(item_key, binding)
        payload = _json(result)
        if self.seal is not None:
            payload = self.seal(payload)
            if not isinstance(payload, bytes):
                raise TypeError("seal must return bytes")
        return binding, payload

    def put(self, item_key: str, result: Any, binding) -> dict:
        binding, payload = self._prepare(item_key, result, binding)
        return self._put_encoded(item_key, payload, binding, self.seal is not None)

    def _put_encoded(self, item_key, payload: bytes, binding, sealed: bool) -> dict:
        binding = _binding(binding)
        key = self._key(item_key, binding)
        # New transactions never overwrite an earlier committed payload/marker.
        try:
            existed = self.root.exists()
            self.root.mkdir(exist_ok=True)
            if not existed:
                _fsync_dir(self.root.parent)
            directory = Path(tempfile.mkdtemp(prefix=key + "-", dir=str(self.root)))
            _fsync_dir(self.root)
            payload_name = "payload.age" if sealed else "payload.json"
            receipt = _atomic_write(directory / payload_name, payload)
            marker = {"schema_version": SCHEMA_VERSION, "item_key": item_key,
                      "key_sha256": key, "binding": binding.as_dict(),
                      "payload_sha256": receipt["sha256"], "sealed": sealed}
            _atomic_write(directory / "commit.json", _json(marker) + b"\n")
            return marker
        except OSError as error:
            raise DurableProgressError(WRITE_FAILED) from error

    def load_valid(self, binding) -> CacheRead:
        binding = _binding(binding)
        counts = dict.fromkeys(("total", "valid", "stale", "incomplete", "corrupt", "conflict"), 0)
        entries, hashes, accepted_counts, conflicts = {}, {}, {}, set()
        try:
            if not self.root.exists():
                return CacheRead(entries, counts)
            directories = sorted(self.root.iterdir())
            for directory in directories:
                counts["total"] += 1
                if not directory.is_dir():
                    counts["corrupt"] += 1
                    continue
                marker_path = directory / "commit.json"
                if not marker_path.exists():
                    counts["incomplete"] += 1
                    continue
                try:
                    marker = _loads(marker_path.read_bytes())
                    if not isinstance(marker, dict) or set(marker) != {
                            "schema_version", "item_key", "key_sha256", "binding", "payload_sha256", "sealed"}:
                        raise ValueError("marker")
                    saved = _binding(marker["binding"])
                    key = self._key(marker["item_key"], saved)
                    if (type(marker["schema_version"]) is not int
                            or marker["schema_version"] != SCHEMA_VERSION
                            or marker["key_sha256"] != key
                            or not directory.name.startswith(key + "-")
                            or type(marker["sealed"]) is not bool
                            or not isinstance(marker["payload_sha256"], str)
                            or _HEX.fullmatch(marker["payload_sha256"]) is None):
                        raise ValueError("marker identity")
                    if saved != binding:
                        counts["stale"] += 1
                        continue
                    payload_path = directory / ("payload.age" if marker["sealed"] else "payload.json")
                    payload = payload_path.read_bytes()
                    if _sha(payload) != marker["payload_sha256"]:
                        raise ValueError("payload hash")
                    if marker["sealed"] and self.unseal is None:
                        result = payload
                    else:
                        result = _loads(self.unseal(payload) if marker["sealed"] else payload)
                    item = marker["item_key"]
                    # Compare decoded JSON bytes too: randomized encryption may
                    # legitimately yield different ciphertext for the same result.
                    result_hash = _sha(result if isinstance(result, bytes) else _json(result))
                    if item in hashes and hashes[item] != result_hash:
                        conflicts.add(item)
                    hashes[item] = result_hash
                    entries[item] = result
                    accepted_counts[item] = accepted_counts.get(item, 0) + 1
                except (ValueError, TypeError, KeyError, UnicodeError, RecursionError, DurableProgressError):
                    counts["corrupt"] += 1
                except FileNotFoundError:
                    counts["incomplete"] += 1
            for item in conflicts:
                entries.pop(item)
                counts["conflict"] += accepted_counts.pop(item)
            counts["valid"] = sum(accepted_counts.values())
            return CacheRead(entries, counts)
        except OSError as error:
            # An unreadable cache cannot safely be interpreted as an empty cache.
            raise DurableProgressError(READ_FAILED) from error


def resume_plan(items: Iterable[str], cache: ResultCache, binding) -> tuple:
    """Return (reuse mapping, compute list, counts-only report), preserving order."""
    items = list(items)
    if any(not isinstance(item, str) for item in items) or len(set(items)) != len(items):
        raise DurableProgressError("DURABLE_INVALID_ITEMS")
    loaded = cache.load_valid(binding)
    reuse = {item: loaded.entries[item] for item in items if item in loaded.entries}
    compute = [item for item in items if item not in reuse]
    report = {"items": len(items), "reuse": len(reuse), "compute": len(compute),
              **{"cache_" + key: value for key, value in loaded.report.items()}}
    return reuse, compute, report


def final_result_writer(path, result: Any, *, sink=None) -> dict:
    """Atomically publish and read back bytes (or canonical JSON plus newline).

    The destination directory must already exist. Under a guard pass a sink
    started before that guard. To record computation failure, explicitly pass a
    failure aggregate; no scientific success/failure is inferred by this helper.
    """
    raw = result if isinstance(result, bytes) else _json(result) + b"\n"
    if sink is not None:
        return sink._request({"op": "final", "path": str(Path(path).absolute()),
                              "payload": base64.b64encode(raw).decode("ascii")})
    return _atomic_write(Path(path), raw)


class GuardSafeSink:
    """An independent writer process with acknowledged, fsynced pipe requests.

    Construct before read_only_guard(); no writable regular file is opened by
    the computing process after construction. The worker drains complete pipe
    requests and closes on EOF, including after the computing process is killed.
    Calls are serialized across threads. Do not share this object after fork.
    A kill during a call can leave either a complete or an incomplete transaction;
    resume_plan decides from committed, verified cache contents, never a count.
    """

    def __init__(self, journal_path, *, cache: Optional[ResultCache] = None):
        self.cache = cache
        self._lock = threading.Lock()
        self._closed = False
        self._owner_pid = os.getpid()
        command = [sys.executable, "-I", "-S", "-B", str(Path(__file__).resolve()),
                   "--writer", str(Path(journal_path).absolute()),
                   str(cache.root.absolute()) if cache is not None else ""]
        try:
            self._process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                             stderr=subprocess.DEVNULL, bufsize=0,
                                             close_fds=True, start_new_session=True)
        except OSError as error:
            raise DurableProgressError(WRITE_FAILED) from error
        try:
            self._receive()
        except BaseException:
            self.close(check=False)
            raise

    def _receive(self):
        try:
            raw = self._process.stdout.readline()
            if not raw:
                raise DurableProgressError(WRITE_FAILED)
            reply = _loads(raw)
            if not reply["ok"]:
                raise DurableProgressError(reply["code"])
            return reply["result"]
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise DurableProgressError(WRITE_FAILED) from error

    def _request(self, request):
        with self._lock:
            if self._closed or os.getpid() != self._owner_pid:
                raise DurableProgressError("DURABLE_SINK_CLOSED")
            raw = _json(request) + b"\n"
            try:
                _write_all(self._process.stdin, raw)
                self._process.stdin.flush()
            except OSError as error:
                raise DurableProgressError(WRITE_FAILED) from error
            return self._receive()

    def send(self, record: dict) -> dict:
        if not isinstance(record, dict):
            raise TypeError("journal record must be a JSON object")
        return self._request({"op": "journal", "record": record})

    def store(self, item_key: str, result: Any, binding) -> dict:
        if self.cache is None:
            raise DurableProgressError("DURABLE_CACHE_REQUIRED")
        bound, payload = self.cache._prepare(item_key, result, binding)
        return self._request({"op": "cache", "item_key": item_key, "binding": bound.as_dict(),
                              "sealed": self.cache.seal is not None,
                              "payload": base64.b64encode(payload).decode("ascii")})

    def close(self, *, check=True):
        if os.getpid() != self._owner_pid:
            raise DurableProgressError("DURABLE_SINK_CLOSED")
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._process.stdin.close()  # EOF: worker drains and closes journal.
                status = self._process.wait()
            finally:
                self._process.stdout.close()
            if check and status:
                raise DurableProgressError(WRITE_FAILED)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is None:
            self.close()
        else:
            # Preserve the original computation exception if storage also fails.
            try:
                self.send({"event": "exception", "exception_type": exc_type.__name__})
            except Exception as error:
                # Make secondary persistence failure inspectable without
                # replacing the computation exception that triggered cleanup.
                exc.durable_progress_error = error
            self.close(check=False)


def _writer_main(journal_path: str, cache_root: str) -> int:
    """Private pipe protocol: stdout contains acknowledgements only."""
    output_alive = True
    def reply(value):
        nonlocal output_alive
        if output_alive:
            try:
                _write_all(sys.stdout.buffer, _json(value) + b"\n")
                sys.stdout.buffer.flush()
            except BrokenPipeError:
                # Computing process died. Keep draining requests already in pipe.
                output_alive = False
    try:
        journal = ProgressJournal(journal_path)
    except DurableProgressError as error:
        reply({"ok": False, "code": error.code})
        return 1
    cache = ResultCache(cache_root) if cache_root else None
    reply({"ok": True, "result": {"ready": True}})
    failed = False
    try:
        for line in sys.stdin.buffer:
            if not line.endswith(b"\n"):
                break  # Interrupted request has no acknowledgement or commit.
            try:
                request = _loads(line)
                if request["op"] == "journal":
                    result = journal.append(request["record"])
                elif request["op"] == "cache" and cache is not None:
                    result = cache._put_encoded(request["item_key"],
                                                base64.b64decode(request["payload"], validate=True),
                                                request["binding"], request["sealed"])
                    journal.append({"event": "cache_commit", "item_key": request["item_key"],
                                    "binding": request["binding"], "payload_sha256": result["payload_sha256"]})
                elif request["op"] == "final":
                    result = _atomic_write(Path(request["path"]),
                                           base64.b64decode(request["payload"], validate=True))
                else:
                    raise DurableProgressError("DURABLE_PROTOCOL_ERROR")
                reply({"ok": True, "result": result})
            except Exception as error:
                failed = True
                code = error.code if isinstance(error, DurableProgressError) else WRITE_FAILED
                reply({"ok": False, "code": code})
    finally:
        journal.close()
    return int(failed)


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--writer":
        raise SystemExit("private durable progress writer")
    raise SystemExit(_writer_main(sys.argv[2], sys.argv[3]))
