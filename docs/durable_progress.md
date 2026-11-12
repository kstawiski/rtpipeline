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
  Each JSON line contains schema version 1, a sequence starting at 1, UTC time,
  the body and a SHA-256 covering the complete envelope except the hash itself.
  Each append flushes and fsyncs. `ProgressJournal.read(path)` returns
  `JournalRead(records, report)`. Records include envelopes. The report counts
  accepted records and a truncated last line. A line without its final newline
  is excluded, even if its JSON is parseable. A complete malformed line, hash
  mismatch or sequence gap raises `DURABLE_JOURNAL_CORRUPT`.
- `GuardSafeSink(journal_path, cache=cache)` starts an independent Python writer.
  `send(dict)` persists a journal record. `store(item_key, result, binding)`
  commits a cache entry, then journals its commit. Both wait for acknowledgement
  after fsync. The sink supports a context manager and idempotent `close()`;
  close drains the pipe, waits for the writer and checks its exit status.
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
has closed. A kill during a call may commit that item or leave an incomplete
transaction. Resume uses verified cache contents, never the last printed count.
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

## Storage and confidentiality limits

File and directory fsync durability requires a filesystem that implements those
operations correctly. The audit-hook and SIGKILL contracts are tested on POSIX.
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
