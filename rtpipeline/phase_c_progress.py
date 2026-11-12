"""Fresh main-runtime validation before Phase C DAG resume decisions."""
import json
from pathlib import Path
from types import SimpleNamespace


def preflight(output, configfile):
    import yaml
    from .durable_progress import DurableProgressError
    from .stage_completion import (validate_stage_completion_sentinel, stage_definition,
        _code_identity, environment_record)
    config = yaml.safe_load(Path(configfile).read_text()) or {}
    output = Path(output)
    invalidated = []
    for sentinel_name, stage in ((".segmentation_done", "segmentation"),
            (".custom_models_done", "segmentation_custom"), (".crop_ct_done", "crop_ct"), (".qc_done", "qc")):
        for sentinel in sorted(output.glob("*/*/" + sentinel_name)):
            try:
                text = sentinel.read_text().strip()
                if text in {"failed", "disabled"}:
                    continue
                observed = json.loads(text)
                if observed.get("status") != "ok":
                    continue
                receipt = validate_stage_completion_sentinel(sentinel, expected_stage=stage)
                definition = stage_definition(stage)
                if receipt["code_identity"]["sources"] != _code_identity(definition)["sources"]:
                    raise ValueError("Phase C deciding code changed")
                if receipt["environment"] != environment_record(definition.environment_packages):
                    raise ValueError("Phase C production runtime changed")
                course = sentinel.parent
                if stage == "segmentation":
                    from .segmentation import _model_progress
                    manifests = list(course.rglob("Segmentation_TotalSegmentator/**/manifest.json"))
                    if not manifests:
                        raise ValueError("Segmentation has no model manifest")
                    for manifest_path in manifests:
                        base = manifest_path.parent
                        saved = json.loads((base / ".model-progress/settings.json").read_text())
                        cfg = SimpleNamespace(**saved["configuration"])
                        for entry in json.loads(manifest_path.read_text())["models"]:
                            if _model_progress(cfg, base, entry["model"], Path(saved["source"]),
                                    saved.get("extra_args_by_model", {}).get(entry["model"], [])).load() is None:
                                raise ValueError("Segmentation model binding changed")
                elif stage == "segmentation_custom":
                    from .custom_models import _parse_custom_model, _custom_progress
                    from .course_contract import load_course_contract
                    contract = load_course_contract(course)
                    for path in (course / ".custom-model-progress/configuration").glob("*.json"):
                        saved = json.loads(path.read_text())
                        directory = Path(saved["model_directory"])
                        model = _parse_custom_model(directory,
                            yaml.safe_load((directory / "custom_model.yaml").read_text()) or {}, "nnUNetv2_predict")
                        if _custom_progress(SimpleNamespace(**saved["configuration"]), course, model, contract.planning_ct_nifti).load() is None:
                            raise ValueError("Custom model binding changed")
                    from .rtsegmentator import RTsegmentatorSettings, _current_models, planning_ct_identity
                    remote = RTsegmentatorSettings.from_mapping(config.get("rtsegmentator"))
                    if remote.enabled and contract.planning_ct_dir is not None:
                        _, uid, digest = planning_ct_identity(course)
                        if _current_models(course, remote, uid, digest) != set(remote.models):
                            raise ValueError("Portal import binding changed")
                elif stage == "crop_ct":
                    from .anatomical_cropping import validate_crop_publication
                    cropping = config.get("ct_cropping") or {}
                    validate_crop_publication(course, {"region": cropping.get("region", "pelvis"),
                        "output_suffix": "_cropped", "superior_margin_cm": cropping.get("superior_margin_cm"),
                        "inferior_margin_cm": cropping.get("inferior_margin_cm")})
                elif stage == "qc":
                    from .quality_control import validate_qc_publication
                    validate_qc_publication(course)
            except (OSError, ValueError, KeyError, TypeError, RuntimeError, DurableProgressError):
                sentinel.unlink(missing_ok=True)
                invalidated.append(str(sentinel.relative_to(output)))
    return {"invalidated": invalidated}
