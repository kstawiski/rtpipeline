# WS513: durable progress and verified resume

Implemented on `fix/durable-progress-cache` in the authorized worktree. Local
commit only; no push. No scientific code paths, numerical definitions or gates
were changed. No patient data, pipeline, cohort, Docker, GPU, network or service
execution was used. This work does not establish scientific, clinical or
manuscript readiness.

## Deliverables and API

- `rtpipeline/durable_progress.py`: standard library only, Python 3.9-compatible
  syntax, with no scientific imports.
- `docs/durable_progress.md`: API, adoption example, guard contract and recovery.
  No repository CHANGELOG was found.
- `tests/test_durable_progress.py`: generated synthetic fixtures only.
- `_ws513/pytest-durable-progress.xml`: final machine-readable test evidence.
  `_ws513/` is outside package discovery and is not packaged.

| API | Contract |
| --- | --- |
| `Binding(code_sha256, config_sha256, inputs_sha256)` | Immutable, exactly three validated SHA-256 strings. |
| `make_binding(code_files, config, inputs)` | Hashes every byte of explicitly declared code files, including helpers, and binds a sorted named manifest. Config/input fingerprints are canonical JSON or exact supplied bytes. |
| `file_sha256(path)` | Streaming raw-byte SHA-256; no source filtering or text conversion. |
| `ProgressJournal(path).append(record)` | Schema, monotonic sequence, UTC time and envelope SHA-256; each append flushes and fsyncs. Context manager and explicit close. |
| `ProgressJournal.read(path)` | Returns envelopes plus counts. Excludes and reports a last line without a newline. Complete corruption raises a fixed code; append to a truncated journal is refused. |
| `GuardSafeSink(journal_path, cache=cache)` | Independent writer started before the guard. `send(record)` journals; `store(key, result, binding)` commits cache then journals its commit. Calls acknowledge fsync. Explicit/context close drains and waits. |
| `ResultCache(root, seal=None, unseal=None).put(key, result, binding)` | Immutable content-addressed transaction: payload first, then marker containing its SHA-256. Both phases verify and fsync file/directory. An interrupted new attempt preserves the previous valid entry for the same key/binding. |
| `cache.load_valid(binding)` | Returns `CacheRead(entries, report)`. Exact binding and verified payload required. Stale/incomplete/corrupt/conflicting transactions are excluded, counted and retained. Unreadable storage raises an error. |
| `resume_plan(items, cache, binding)` | Returns `(reuse, compute, report)`: reusable requested mapping, ordered compute list and integer counts only. Duplicate requested keys refused. Item and transaction counts are named separately. |
| `final_result_writer(path, result, sink=None)` | Exact bytes or canonical JSON; same-directory temporary file, fsync, read-back, rename, directory fsync, destination read-back. Returns verified byte-count/SHA-256 receipt only after equality. Pass the prestarted sink under the guard. |

The computing process opens no writable regular files and performs no directory
creation or rename after the guard. Anonymous pipes carry requests/replies. EOF
after SIGKILL drains complete queued requests even if acknowledgements cannot be
delivered. Successful sink calls are already durable; an interrupted call may
leave a complete or incomplete transaction. Resume trusts verified cache contents,
not printed counts. A cache commit can survive failure of its journal append.

Optional sealing runs before writing any payload. Hashes cover ciphertext as
stored. Identical duplicate commits yield one reusable item; conflicting results
for one key/binding are all excluded. Invalid entries are never silently deleted.
Fixed I/O/verification codes include `DURABLE_WRITE_FAILED`, `DURABLE_READ_FAILED`
and `DURABLE_VERIFY_FAILED`. Computation exceptions retain their identity; a
secondary exception-journaling failure is available as `durable_progress_error`.

## Exact verification counts

Final frozen suite: **48 passed, 0 failed, 0 skipped, 0 deselected** in
**114.09 seconds**. The JUnit receipt records all 48 test cases.

| Run | Passed | Failed | Skipped | Deselected | Seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial suite | 43 | 1 | 0 | 0 | 116.51 |
| Expanded suite | 44 | 2 | 0 | 0 | 69.55 |
| Focused SIGKILL checks | 4 | 0 | 0 | 42 | 60.90 |
| Guarded regression suite | 46 | 0 | 0 | 0 | 69.65 |
| Final frozen suite | 48 | 0 | 0 | 0 | 114.09 |

The initial failure distinguished journal read failure at startup from output
write failure. The expanded run hit a ten-second wait for a whole committed
prefix on the shared disk; SIGKILL tests now wait for each acknowledged commit.
The exact reuse and final-result assertions remain. Static inspection added
normalization of excessive JSON nesting through the existing corruption paths.

Coverage includes actual journal/cache read-back; hash/sequence/schema and JSON
corruption; truncated tails; code/config/input staleness; missing marker/payload;
conflicting commits; sealing; resume partitions; SIGKILL after **0, 7 and 19** of
**20** committed items and equality with uninterrupted results; queued-request
drainage; the inlined checker's audit semantics, including regular-fd refusal
and pipe/`/dev/null` permission; guarded journal/cache/final output; computation
exceptions and persisted failure aggregates; injected ENOSPC in payload/marker
writes for an existing key; final write/fsync/rename failures; temporary and
destination read-back mismatches; startup/pipe failure; threads and close.

Pytest uses disabled plugin autoload, `-p no:cacheprovider`, `-B`, task-owned
`--basetemp`, isolated `TMPDIR`/`XDG_CACHE_HOME` and JUnit output under `_ws513/`.
The guarded runs use a parent audit hook refusing socket operations, restricting
writable opens/mutations to task outputs, and allowing only explicit isolated
Python test children and the durable writer. Test children also refuse audited
socket operations. Fixtures remove synthetic outputs after use. These boundaries
are Python-level checks, not an OS sandbox.

Additional checks: **1 Python 3.10.14 standard-library smoke check passed**
(journal, cache, writer subprocess and final read-back); **1 Python 3.9 grammar
check passed**. Pytest runs on Python 3.14.3. Python 3.9 runtime execution is
unverified because that interpreter is absent. Existing modules, `__init__.py`
and imports are unchanged, so no existing fast subset is affected by an import
change. No full pipeline or external-runtime check was run.

## Adoption for a checker with per-course items

1. Pin the checker and all result-determining helpers. Freeze effective config
   and an input manifest containing content hashes. Build a binding before the
   long run; keep code and inputs immutable throughout it.
2. Create an output directory and use opaque nonidentifying course keys. Call
   `reuse, compute, counts = resume_plan(keys, cache, binding)` and seed the
   result mapping with `reuse`.
3. Start `GuardSafeSink(output / 'progress.jsonl', cache=cache)` before installing
   `read_only_guard()`. Compute only `compute`, then call
   `sink.store(key, result, binding)` for each result. Persist failed gates when
   the checker's existing rules permit continued evaluation.
4. Pass that sink to `final_result_writer(output / 'result.json', aggregate,
   sink=sink)`. In an exception handler, explicitly publish a failure aggregate
   before reraising. Use a context manager and stop on storage failure.
5. After interruption, recompute the binding and repeat resume planning. Archive
   a truncated journal explicitly and choose a new journal path; valid cache
   entries remain reusable. Investigate conflicts and use a fresh cache after
   resolving their cause. A full code example is in the documentation page.

## Limits

Use one writer per journal and per logical cache item/binding; do not share a
sink across fork. POSIX audit/SIGKILL behavior and local file/directory fsync are
tested. Other filesystem durability semantics are unverified. Failure after
rename may leave the new file published and never certifies rollback. SIGKILL
cannot automatically produce a final aggregate; the next run rebuilds it from
the verified cache. The helper cannot discover undeclared dependencies or freeze
mutable inputs; bind the actual pinned code/config/input snapshot.

Sealing protects payloads only. Journals, keys, markers and ordinary final JSON
remain plaintext and must contain no patient content. Without an authorized
unseal callback, sealed cache values remain verified ciphertext bytes. Any
patient-bearing final output adopted elsewhere must already be encrypted and
use an appropriately sealed destination. This task used synthetic content only.
