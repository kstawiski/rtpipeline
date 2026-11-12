# Durable progress and guard-safe output

`rtpipeline.durable_progress` provides reusable persistence for long checkers.
It uses only the Python standard library and supports Python 3.9 or later.
Importing it does not import scientific packages or change pipeline stages.

## Bind results before computing

`make_binding(code_files, config, inputs)` returns an immutable `Binding` with
`code_sha256`, `config_sha256` and `inputs_sha256`. The code digest covers a sorted
manifest of stable logical names and SHA-256 hashes of **every byte** in each
explicitly supplied file. Include the checker and every helper that determines
its results. A revision label, file size or modification time is insufficient.
Configuration and input fingerprints are canonical JSON, or exact bytes when
bytes are passed. `file_sha256(path)` hashes a file without text conversion.

Use a pinned code snapshot and immutable inputs throughout the run. Hash the
actual configuration used and an input manifest containing content hashes, not
just filenames. The module cannot discover undeclared dependencies, freeze live
inputs or establish that imported code matches a subsequently changed file.

## Adopt in a checker with per-course items

This sketch uses opaque synthetic course keys. Create the output directory and
start the sink **before** installing the checker's read-only audit hook.

```python
from pathlib import Path
from rtpipeline.durable_progress import (
    GuardSafeSink, ResultCache, final_result_writer, make_binding, resume_plan,
)

out = Path("checker-output")
out.mkdir(exist_ok=True)
binding = make_binding(
    {"checker.py": Path(__file__), "checker_helpers.py": helper_source},
    config=effective_config,
    inputs=frozen_content_hash_manifest,
)
cache = ResultCache(out / "cache")
reuse, compute, counts = resume_plan(course_keys, cache, binding)
results = dict(reuse)

with GuardSafeSink(out / "progress.jsonl", cache=cache) as sink:
    read_only_guard()  # supplied by the caller
    sink.send({"event": "resume", "counts": counts})
    try:
        for key in compute:
            result = check_course(key)  # retain failed gates as per-course results
            sink.store(key, result, binding)
            results[key] = result
        final_result_writer(
            out / "result.json",
            {"status": "completed", "results": results}, sink=sink,
        )
    except Exception as error:
        final_result_writer(
            out / "result.json",
            {"status": "failed", "exception_type": type(error).__name__,
             "completed_items": len(results)}, sink=sink,
        )
        raise
```

The checker determines gate outcomes and aggregate status. Persist a failed gate
as an item result and continue only when the checker's existing rules allow it.
Persistence does not approve scientific or clinical readiness. If storage fails,
stop and handle `DurableProgressError.code`; do not claim the aggregate was saved.

## API and recovery contract

- `ProgressJournal(path)` is a context manager with `append(dict)` and `close()`.
  It holds an exclusive, nonblocking POSIX advisory lock until close. A second
  writer fails with `DURABLE_JOURNAL_LOCKED`. Sequence recovery happens under
  that lock; readers do not need the writer lock.
  Each JSON line contains schema version 1, a sequence starting at 1, UTC time,
  the body and a SHA-256 covering the complete envelope except the hash itself.
  Each append flushes and fsyncs. `ProgressJournal.read(path)` returns
  `JournalRead(records, report)`. Records include envelopes. The report counts
  accepted records and a truncated last line. A line without its final newline
  is excluded, even if its JSON is parseable. A complete malformed line, hash
  mismatch or sequence gap raises `DURABLE_JOURNAL_CORRUPT`.
- `GuardSafeSink(journal_path, cache=cache, timeout=60.0)` starts an independent
  Python writer. Existing calls without `timeout` use the 60-second default.
  `send(dict)` persists a journal record. `store(item_key, result, binding)`
  commits a cache entry, then journals its commit. Both wait for acknowledgement
  after fsync. The sink supports a context manager and idempotent `close()`;
  close drains the pipe, waits for the writer and checks its exit status.
  The positive, finite `timeout` in seconds bounds the startup handshake, each
  complete request and acknowledgement, and the wait on close. Increase it for
  storage whose expected transaction latency exceeds the default.
- `ResultCache(root, seal=None, unseal=None).put(key, result, binding)` stores a
  JSON result. Each transaction has an address derived from the item key and
  exact binding, plus a unique transaction suffix. Phase 1 writes and verifies
  the payload. Phase 2 atomically writes a `commit.json` marker containing its
  SHA-256. Both phases fsync the file and directory. A failed new transaction
  cannot overwrite a previous valid one.
- `cache.load_valid(binding)` returns `CacheRead(entries, report)`. It accepts
  only exact three-field bindings and verified committed payloads. Counts are
  **transactions**, classified as `valid`, `stale`, `incomplete`, `corrupt` or
  `conflict`, with `total` equal to their sum. A valid marker with a different
  binding is stale without reading its payload. Missing markers or payloads are
  incomplete. Malformed markers and hash mismatches are corrupt. Conflicting
  valid results for one item/binding are all excluded. Identical duplicates are
  counted as valid transactions but yield one reusable item. Unreadable storage
  raises `DURABLE_READ_FAILED`; it is never treated as an empty cache.
- `resume_plan(items, cache, binding)` returns `(reuse, compute, report)`.
  `reuse` maps requested item keys to results; `compute` preserves request order.
  Duplicate requested keys are refused. The report contains only integer counts:
  requested/reused/computed **items** and the prefixed cache transaction counts.
- `final_result_writer(path, result, sink=None)` writes canonical JSON plus a
  newline, or exact bytes when given bytes. It writes a temporary file in the
  existing destination directory, flushes, fsyncs, reads it back, replaces the
  destination, fsyncs the directory, and reads the destination back. Its receipt
  contains byte count, SHA-256 and `verified=True`. Under a guard, pass the sink;
  the independent writer performs the same operations and verification.

The computing process performs no new writable regular-file opens, directory
creation, renames or removals after the guard. Existing anonymous pipes carry
requests and acknowledgements. The writer must start before the guard; starting
a new sink or using `cache.put()` or direct final output after the guard violates
this contract. The guard remains active in the computing process. This is an
explicit output channel, not a security sandbox for an untrusted computing
process. Paths sent to the writer must be trusted output paths.

After SIGKILL, every successful sink call is already durable. The writer drains
complete queued requests and finalizes on EOF, even when its acknowledgement pipe
has closed. A final request without a newline is ignored, including when its
JSON is otherwise complete; earlier committed requests remain durable and the
writer exits cleanly. A kill during a call may commit that item or leave an
incomplete transaction. Resume uses verified cache contents, never the last
printed count.
A cache commit can survive even if its subsequent journal append fails. A final
aggregate is not automatically produced after SIGKILL; recompute it from the
verified cache on the next run. An ordinary exception inside the sink context
also appends an exception type; explicitly use the final writer for a failure
aggregate, as in the example. If exception journaling also fails, the original
exception is preserved and its `durable_progress_error` attribute holds the
secondary storage error.

Nothing silently deletes invalid entries. If the journal has a truncated tail,
archive it explicitly and choose a new journal before resuming from the cache.
Do not append to a damaged journal. For conflicting results, investigate the
binding and nondeterminism; use a new cache after resolving the cause. Use one
writer per journal and per logical cache item/binding. Sink calls from threads
are serialized; do not share a sink across a fork.

Timeouts have fixed error codes: `DURABLE_WRITER_HANDSHAKE_TIMEOUT`,
`DURABLE_WRITER_SEND_TIMEOUT`, `DURABLE_WRITER_ACK_TIMEOUT` and
`DURABLE_WRITER_CLOSE_TIMEOUT`. A send or acknowledgement timeout disables
further requests on that sink so a late reply cannot acknowledge another item.
Close the sink and inspect the verified cache before resuming; an operation
that timed out may already have committed. Close allows the writer to drain
within the timeout, then sends SIGKILL if it remains alive and waits for at most
one further timeout to reap it. Constructor failure performs the same bounded
cleanup. `close(check=False)` suppresses shutdown errors while retaining those
bounds; exception cleanup uses it to preserve the computation exception.

## Storage and confidentiality limits

File and directory fsync durability requires a filesystem that implements those
operations correctly. Journal locking requires POSIX `fcntl.flock`; it is
advisory, so every writer must use this module's locking contract. The audit-hook
and SIGKILL contracts are tested on POSIX. An OS task stuck in uninterruptible
I/O may outlive the bounded shutdown wait even after SIGKILL.
If a failure occurs after replacement, the new file may already be published;
an error never certifies rollback. `DURABLE_WRITE_FAILED` identifies I/O failures;
`DURABLE_VERIFY_FAILED` identifies a byte read-back mismatch. Other fixed codes
identify invalid bindings, invalid item lists, corrupt/truncated journals and
closed sinks/journals. Serialization and sealing callback errors propagate.
Value/type errors during unsealing count as corruption; unsealing I/O failures
raise `DURABLE_READ_FAILED`.

`seal(plaintext_bytes) -> ciphertext_bytes` runs before any payload is written;
the stored `.age` payload's ciphertext hash is committed and verified. Provide
`unseal(ciphertext_bytes) -> plaintext_bytes` only where decryption is authorized.
Without it, `load_valid` returns verified ciphertext bytes. This callback does
not encrypt journals, item keys, markers or final aggregates. Keep those free of
patient content; use opaque nonidentifying item keys. If a final aggregate
contains patient data, supply already encrypted bytes and an appropriately
sealed destination. Callers remain responsible for the project's encryption
policy and for authorized storage. Synthetic tests do not establish clinical or
manuscript readiness.

## Production DVH generations

The course DVH writer uses an exact code/configuration/input binding before
reuse. It hashes the deciding modules, CLI and workflow publishers, effective
DVH settings (including crop selection), DICOM bytes, CT/NIfTI inputs, every
source mask and upstream content/configuration/code identities. Upstream receipt
creation times and host names do not define a scientific generation. A legacy
completion without binding is unverified and is rebuilt with
`DVH_LEGACY_UNBOUND` recorded in the course journal.

Completed ROI results are immutable `ResultCache` transactions under
`.dvh/roi-cache`. They retain dictionary order, numerical values and explicit
nonfinite values. Recovery reads and verifies each transaction before reuse;
course-level QC and quarantine decisions still run when the final tables are
assembled. The journal records cache reuse, ROI commits and attempt lineage.

The writer stages the workbook, Parquet table, curves, QC, optional RS_custom
and its metadata, completion receipt and ledger evidence in one directory.
After read-back validation and fsync, a hashed commit marker certifies the full
inventory. An atomic `.dvh/current` symlink selects the complete immutable
generation. Fixed public artifact names resolve through that pointer. Readers
must validate the completion/generation before consuming artifacts; file
presence is insufficient. A reader needing a stable view across concurrent
publication can resolve the current pointer once and verify that directory.

An incomplete staging or generation directory is discarded only after its fixed
reason is durably journaled. Previously committed generations remain intact,
including evidence of a corrupted generation. Missing public aliases can be
restored from a verified commit without repeating a sampler call. A crash after
commit is repaired by projecting that generation's ledger evidence and
reconciling attempt state, rather than rerunning its measurements.

Attempt state is distinct from DAG closure. `started` without a matching commit
is interrupted and can resume its verified ROI checkpoints. An unchanged failed
binding requires an explicit retry: `dvh_for_course(..., retry_failed=True)` or
`dvh: {retry_failed: true}` in the effective workflow configuration. This retry
switch is operational and is excluded from the scientific binding. Changed
inputs/configuration/code establish a new binding. Ledger persistence failure
propagates instead of certifying success. Configured or contract-declared
nonapplicability retains its existing disposition and receives a current
code/configuration/input binding.

The DVH calculation and published table schema are unchanged. Workbook creation
metadata uses a fixed packaging epoch so identical tables have reproducible
bytes. SIGKILL tests cover process interruption; they do not establish behavior
under host power loss, filesystem failure recovery or every storage backend.
These DVH guarantees do not establish durable per-item resume for other stages.
