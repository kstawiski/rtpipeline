"""Bound item transactions shared by extraction and reduction resume paths.

Only complete validated outcomes are reusable. Technical failures are retried;
their evidence remains the producer's responsibility. Keys in cache metadata
are opaque. Payloads contain the same non-identifying measurement rows as the
existing tables. One writer per item is required, as for ResultCache itself.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from contextlib import contextmanager
import fcntl
import hashlib
import json
import threading
from pathlib import Path

from .durable_progress import (DurableProgressError, ResultCache, file_sha256,
                               make_binding, resume_plan, ProgressJournal, _fsync_dir)
from .dvh_generation import _encode_row, _decode_row


OPERATIONAL = frozenset({"max_workers", "workers", "threads", "thread_limit",
    "max_workers_override", "radiomics_thread_limit", "max_parallel_courses",
    "env_probe_timeout", "radiomics_env_probe_timeout", "sequential"})
VOLATILE = frozenset({"run_identifier", "code_revision", "__task_index__"})


def scientific(value):
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {str(k): scientific(v) for k, v in value.items()
                if k not in OPERATIONAL | VOLATILE and not str(k).startswith("__progress_")}
    if isinstance(value, (set, frozenset)):
        return sorted((scientific(v) for v in value), key=lambda v: json.dumps(v, sort_keys=True))
    if isinstance(value, (tuple, list)):
        return [scientific(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    return value


def digest(value):
    return hashlib.sha256(json.dumps(scientific(value), sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def image_identity(image):
    """Bind every source byte for files; geometry and voxels for memory images."""
    if isinstance(image, (str, Path)):
        return {"sha256": file_sha256(image)}
    import SimpleITK as sitk
    array = sitk.GetArrayViewFromImage(image)
    return {"sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            "dtype": str(array.dtype), "size": image.GetSize(),
            "spacing": image.GetSpacing(), "origin": image.GetOrigin(),
            "direction": image.GetDirection()}


def code_manifest(extra=()):
    package = Path(__file__).resolve().parent
    names = {"extraction_progress.py", "durable_progress.py", "radiomics.py",
        "radiomics_parallel.py", "radiomics_conda.py", "radiomics_ct_contract.py",
        "radiomics_schema.py", "radiomics_rasterizer.py", "rtstruct_geometry.py",
        "radiomics_resource_guard.py", "radiomics_memory.py", "roi_requiredness.py", "config_dependencies.py",
        "auto_rtstruct.py", "rtstruct_identity.py", "original_segmentation.py",
        "robustness_mcc.py", "radiomics_robustness.py", "radiomics_robustness_outcomes.py", *extra}
    return {name: package / name for name in names}


def runtime_environment():
    from .stage_completion import environment_record
    return environment_record(("numpy", "pandas", "SimpleITK", "pyradiomics",
        "scipy", "pydicom", "rt-utils", "opencv-python", "opencv-python-headless",
        "PyWavelets", "pykwalify", "six"))


def radiomics_settings(config):
    settings = {k: v for k, v in vars(config).items()
                if k.startswith("radiomics_") and k not in OPERATIONAL}
    for key in ("radiomics_params_file", "radiomics_params_file_mr", "custom_structures_config"):
        path = getattr(config, key, None)
        if key == "custom_structures_config" and path:
            from .config_dependencies import scientific_custom_config_sha256
            settings[key] = scientific_custom_config_sha256(path)
        else:
            settings[key] = file_sha256(path) if path and Path(path).is_file() else None
    return scientific(settings)


def course_binding(course, dependency=None):
    """Current nominal CT/MR producer binding for a per-course DAG target."""
    import os
    from .dvh_generation import input_manifest
    from .radiomics_conda import _extraction_environment
    course = Path(course)
    settings = {}
    dependency = dependency or os.environ.get("RTPIPELINE_RADIOMICS_CONFIG_DEPENDENCY")
    if dependency and Path(dependency).is_file():
        from .config_dependencies import read_stage_dependency
        settings = read_stage_dependency(Path(dependency))["payload"]
    inputs = input_manifest(course)
    for name in ("RS_custom.dcm", "metadata/rs_custom_meta.json", "metadata/course_metadata.json"):
        path = course / name
        if path.is_file():
            inputs.append({"path": name, "sha256": file_sha256(path)})
    for name in ("MR",):
        inputs.extend({"path": p.relative_to(course).as_posix(), "sha256": file_sha256(p)}
                      for p in sorted((course / name).rglob("*")) if p.is_file() and p.suffix != ".age")
    return make_binding(code_manifest(), {"configuration": scientific(settings),
                        "environment": _extraction_environment()}, inputs)


def commit_nominal_publication(workbook):
    from .durable_progress import final_result_writer
    workbook = Path(workbook)
    course = workbook.parent
    if workbook.name != "radiomics_ct.xlsx" or not (course / "metadata/case_metadata.json").is_file():
        return
    directory = course / ".radiomics-progress"
    directory.mkdir(exist_ok=True)
    _fsync_dir(course)
    final_result_writer(directory / "publication.json", {
        "binding": course_binding(course).as_dict(),
        "artifacts": {p.name: file_sha256(p) for p in (workbook, workbook.with_suffix(".parquet"))}})


def validate_nominal_publication(course):
    course = Path(course)
    record = json.loads((course / ".radiomics-progress/publication.json").read_text())
    if record.get("binding") != course_binding(course).as_dict():
        raise ValueError("nominal publication binding changed")
    expected = {"radiomics_ct.xlsx", "radiomics_ct.parquet"}
    if set(record.get("artifacts", {})) != expected or any(
            file_sha256(course / name) != record["artifacts"][name] for name in expected):
        raise ValueError("nominal publication artifact changed")
    return record


def preflight(output, dependency, configfile=None):
    """Validate each course separately before planning; preserve every item cache."""
    from .radiomics_ct_contract import validate_completion_sentinel
    result = []
    for sentinel in sorted(Path(output).glob("*/*/.radiomics_done")):
        try:
            text = sentinel.read_text().strip()
            if text.lower() in {"failed", "disabled"}:
                continue
            observed = json.loads(text)
            if observed.get("status") == "not_applicable":
                continue  # The existing governed not-applicable preflight owns this outcome.
            validate_completion_sentinel(sentinel.parent, sentinel,
                                          configuration_dependency=Path(dependency))
        except (OSError, ValueError, KeyError, TypeError, DurableProgressError):
            sentinel.unlink(missing_ok=True)
            result.append({"course": str(sentinel.parent), "reusable": False})
        else:
            result.append({"course": str(sentinel.parent), "reusable": True})
    if configfile:
        import yaml
        from .radiomics_robustness import RobustnessConfig, admit_robustness_cohort_course
        config = yaml.safe_load(Path(configfile).read_text()) or {}
        rob_config = RobustnessConfig.from_dict(config.get("radiomics_robustness") or {})
        for sentinel in sorted(Path(output).glob("*/*/.radiomics_robustness_done")):
            try:
                admit_robustness_cohort_course(sentinel.parent,
                    patient_id=sentinel.parent.parent.name, course_id=sentinel.parent.name,
                    rob_config=rob_config, campaign_mode=bool(config.get("campaign_mode", False)),
                    custom_structures_config=config.get("custom_structures"))
            except (OSError, ValueError, KeyError, TypeError, RuntimeError):
                sentinel.unlink(missing_ok=True)
                result.append({"course": str(sentinel.parent), "stage": "robustness", "reusable": False})
    return {"courses": result}


def restamp(value, current):
    """Attempt identity is publication metadata, never a scientific cache key."""
    if isinstance(value, dict):
        return {k: (current[k] if k in VOLATILE and k in current else restamp(v, current))
                for k, v in value.items()}
    if isinstance(value, list):
        return [restamp(v, current) for v in value]
    return value


class BoundItem:
    def __init__(self, root, identity, settings, inputs, *, environment=None, code=None):
        self.key = digest(identity)
        root = Path(root)
        missing = []
        ancestor = root
        while not ancestor.exists():
            missing.append(ancestor)
            ancestor = ancestor.parent
        try:
            root.mkdir(parents=True, exist_ok=True)
            for directory in reversed(missing):
                _fsync_dir(directory.parent)
                _fsync_dir(directory)
        except OSError as exc:
            raise DurableProgressError("EXTRACTION_PROGRESS_DIRECTORY_WRITE_FAILED") from exc
        self.cache = ResultCache(root / self.key)
        self._mutex = threading.RLock()
        self._depth = 0
        self._lock_path = root / (".lock-" + self.key)
        self.binding = make_binding(code or code_manifest(),
            {"settings": scientific(settings),
             "environment": runtime_environment() if environment is None else environment},
            scientific(inputs))

    def load(self):
        reused, _, report = resume_plan([self.key], self.cache, self.binding)
        entry = reused.get(self.key)
        if entry is None and report['cache_stale']:
            # Keep the actual invalidation causes next to the affected item.
            # Read markers only; sealed payloads are never opened here.
            reasons = set()
            for path in self.cache.root.glob('*/commit.json'):
                try:
                    marker = json.loads(path.read_text())
                except (OSError, ValueError):
                    continue
                if not isinstance(marker, dict) or not isinstance(marker.get('binding'), dict):
                    continue
                previous = marker.get('binding', {})
                for field, reason in (
                    ('inputs_sha256', 'EXTRACTION_INPUTS_CHANGED'),
                    ('config_sha256', 'EXTRACTION_CONFIGURATION_CHANGED'),
                    ('code_sha256', 'EXTRACTION_CODE_CHANGED')):
                    if previous.get(field) != self.binding.as_dict()[field]:
                        reasons.add(reason)
            if reasons:
                with ProgressJournal(self._lock_path.with_suffix('.jsonl')) as journal:
                    journal.append({'event': 'item_invalidated', 'item_key': self.key,
                        'reason_codes': sorted(reasons), 'binding': self.binding.as_dict()})
        return _decode_row(entry["result"]) if entry is not None else None

    def commit(self, value):
        with self.locked():
            previous = self.load()
            if previous is not None:
                if _encode_row(scientific(previous)) != _encode_row(scientific(value)):
                    raise DurableProgressError("EXTRACTION_PROGRESS_CONFLICT")
                return value
            self.cache.put(self.key, {"result": _encode_row(value)}, self.binding)
            # Completion and child acknowledgement follow a verified read-back.
            loaded = self.load()
            if _encode_row(loaded) != _encode_row(value):
                raise DurableProgressError("EXTRACTION_PROGRESS_READBACK_FAILED")
            with ProgressJournal(self._lock_path.with_suffix(".jsonl")) as journal:
                journal.append({"event": "item_committed", "item_key": self.key,
                                "binding": self.binding.as_dict()})
        _failpoint("item_committed", self.key)
        return value

    @contextmanager
    def locked(self):
        """Serialize a cache miss with an orphaned child or a concurrent retry."""
        with self._mutex:
            if self._depth:
                yield
                return
            with self._lock_path.open("a+b") as stream:
                fcntl.flock(stream, fcntl.LOCK_EX)
                self._depth += 1
                try:
                    yield
                finally:
                    self._depth -= 1
                    fcntl.flock(stream, fcntl.LOCK_UN)


def _failpoint(event, key):
    """Inert seam for task-owned synthetic interruption tests."""


def reusable(value):
    if not value:
        return False
    if isinstance(value, list):
        from .radiomics_outcomes import extraction_status_is_nonfatal_for_required
        return all(row.get("extraction_status") in (None, "success") or
                   extraction_status_is_nonfatal_for_required(row.get("extraction_status"))
                   for row in value)
    return (value.get("__status__", "success") in {"success", "skipped"}
            and "__technical_failure__" not in value
            and ("__records__" not in value or reusable(value["__records__"])))


def task_item(task, root=None, *, environment=None, extra_settings=None):
    root = root or task.get("__progress_root__")
    if not root:
        return None
    settings = {k: v for k, v in task.items() if k not in
                {"image_path", "mask_path", "params_file", "cleanup", "ct_info"}}
    params = task.get("params_file")
    if params:
        settings["parameters_sha256"] = file_sha256(params)
    if extra_settings or task.get("__progress_context__"):
        settings["context"] = extra_settings or task["__progress_context__"]
    metadata = task.get("metadata") or task
    identity = {k: metadata.get(k) for k in ("patient_id", "course_id", "series_uid",
        "segmentation_source", "mask_identity", "roi_original_name", "stable_roi_identifier")}
    identity.update(roi_name=task.get("roi_name"),
                    condition=task.get("robustness_perturbation_id") or task.get("extra_metadata"),
                    arms=sorted(task.get("configured_parameter_hashes") or {}), label=task.get("label"))
    context = extra_settings or task.get("__progress_context__") or {}
    if isinstance(context, dict):
        perturbation = context.get("segmentation_perturbation", context)
        identity["seed"] = perturbation.get("random_seed", perturbation.get("seed"))
    return BoundItem(root, identity, settings,
        {"image": image_identity(task["image_path"]),
         "mask": image_identity(task["mask_path"]),
         'sources': task.get('__progress_sources__', {})}, environment=environment)


def pack_frame(frame):
    import base64
    from io import BytesIO
    stream = BytesIO()
    frame.to_parquet(stream, index=False)
    return {"parquet": base64.b64encode(stream.getvalue()).decode(), "attrs": frame.attrs}


def unpack_frame(value):
    import base64
    from io import BytesIO
    import pandas as pd
    frame = pd.read_parquet(BytesIO(base64.b64decode(value["parquet"])))
    frame.attrs.update(value.get("attrs", {}))
    return frame


class ResumableResults:
    """Supervised completion-order results with original stable task indices.

    The producer explicitly calls commit only after its condition validation.
    Reused items never enter the worker queue. Technical evidence is returned
    normally and remains outside the successful result cache.
    """
    def __init__(self, context, function, tasks, workers, **budgets):
        from collections import deque
        from .robustness_watchdog import SupervisedResults
        self.tasks, self.items, self.missing, self.ready = tasks, [], [], deque()
        for index, (_, task) in enumerate(tasks):
            item = task_item(task)
            self.items.append(item)
            cached = item.load() if item else None
            if cached is None:
                self.missing.append(index)
            else:
                self.ready.append(dict(restamp(cached, task), __task_index__=index))
        self.results = SupervisedResults(context, function,
            [tasks[i] for i in self.missing], workers, **budgets)

    def __enter__(self):
        self.results.__enter__()
        return self

    def __exit__(self, *args):
        return self.results.__exit__(*args)

    def next(self, timeout):
        if self.ready:
            return self.ready.popleft()
        result = self.results.next(timeout=timeout)
        result["__task_index__"] = self.missing[result["__task_index__"]]
        return result

    def commit(self, index, result):
        item = self.items[index]
        if item and reusable(result) and item.load() is None:
            item.commit({k: v for k, v in result.items() if k != "__task_index__"})
