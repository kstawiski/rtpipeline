from __future__ import annotations

import datetime
import json
import logging
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import SimpleITK as sitk
import yaml

from .config import PipelineConfig
from .segmentation import _ensure_ct_nifti, _run as _run_shell, _sanitize_token
from .auto_rtstruct import _load_ct_image, _resample_to_reference
from .utils import sanitize_rtstruct
from .roi_fixer import fix_rtstruct_rois

if TYPE_CHECKING:
    from .organize import CourseOutput

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class NetworkDefinition:
    network_id: str
    archive_path: Path
    dataset_dir: str
    label_names: List[str]
    env: Dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class CustomModelDefinition:
    name: str
    directory: Path
    command: str
    model: str
    folds: str
    env: Dict[str, str]
    networks: List[NetworkDefinition]
    combine_order: List[str]
    description: str = ""

    def get_network(self, network_id: str) -> Optional[NetworkDefinition]:
        for net in self.networks:
            if net.network_id == network_id:
                return net
        return None

    def expected_structures(self) -> List[str]:
        seen: List[str] = []
        for network_id in self.combine_order:
            net = self.get_network(network_id)
            if net is None:
                continue
            for name in net.label_names:
                if name not in seen:
                    seen.append(name)
        return seen

    def resolved_env(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for key, value in self.env.items():
            if key in {"nnUNet_results", "nnUNet_raw", "nnUNet_preprocessed"}:
                path = Path(value)
                if not path.is_absolute():
                    path = (self.directory / path).resolve()
                out[key] = str(path)
            else:
                out[key] = value
        for key in ("nnUNet_results", "nnUNet_raw", "nnUNet_preprocessed"):
            if key not in out:
                path = (self.directory / key).resolve()
                out[key] = str(path)
        return out

    def results_root(self) -> Path:
        env = self.resolved_env()
        return Path(env["nnUNet_results"])


def _structure_filename(name: str) -> str:
    token = _sanitize_token(name)
    return token or "structure"


def discover_custom_models(
    root: Path,
    selected_names: Optional[Iterable[str]] = None,
    default_command: str = "nnUNetv2_predict",
) -> List[CustomModelDefinition]:
    models: List[CustomModelDefinition] = []
    if not root.exists() or not root.is_dir():
        logger.info("Custom models root %s does not exist or is not a directory", root)
        return models

    selected = {name.lower() for name in (selected_names or []) if name}

    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        config_path = model_dir / "custom_model.yaml"
        if not config_path.exists():
            continue
        try:
            data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            logger.warning("Failed to read custom model config %s: %s", config_path, exc)
            continue
        try:
            definition = _parse_custom_model(model_dir, data, default_command)
        except Exception as exc:
            logger.warning("Skipping custom model %s due to configuration error: %s", config_path, exc)
            continue
        if selected and definition.name.lower() not in selected:
            continue
        models.append(definition)

    if selected:
        missing = selected - {model.name.lower() for model in models}
        for name in sorted(missing):
            logger.warning("Requested custom segmentation model '%s' not found under %s", name, root)
    return models


def _parse_custom_model(model_dir: Path, data: dict, default_command: str) -> CustomModelDefinition:
    if not isinstance(data, dict):
        raise ValueError("Configuration root must be a mapping")

    model_name = str(data.get("name") or model_dir.name)
    nnunet_cfg = data.get("nnunet")
    if not isinstance(nnunet_cfg, dict):
        raise ValueError("Missing 'nnunet' section")

    command = str(nnunet_cfg.get("command") or default_command or "nnUNetv2_predict")
    model_type = str(nnunet_cfg.get("model") or "3d_fullres")
    folds = str(nnunet_cfg.get("folds") or "all")

    env_cfg = nnunet_cfg.get("env") or {}
    if not isinstance(env_cfg, dict):
        raise ValueError("'nnunet.env' must be a mapping if provided")
    env = {str(k): str(v) for k, v in env_cfg.items() if v is not None}

    networks_cfg = nnunet_cfg.get("networks")
    if not isinstance(networks_cfg, list) or not networks_cfg:
        raise ValueError("'nnunet.networks' must be a non-empty list")

    networks: List[NetworkDefinition] = []
    for entry in networks_cfg:
        if not isinstance(entry, dict):
            raise ValueError("Network entry must be a mapping")
        raw_id = entry.get("id") or entry.get("dataset")
        if raw_id is None:
            raise ValueError("Network entry missing 'id'")
        network_id = str(raw_id).strip()
        archive = entry.get("archive") or entry.get("weights")
        if not archive:
            raise ValueError(f"Network {network_id}: missing 'archive'")
        archive_path = (model_dir / str(archive)).resolve()

        dataset_dir_raw = entry.get("dataset_directory") or entry.get("dataset_dir") or entry.get("dataset")
        if dataset_dir_raw:
            dataset_dir = str(dataset_dir_raw).strip()
        else:
            dataset_dir = Path(str(archive)).stem

        labels = entry.get("label_order") or entry.get("structures") or entry.get("labels")
        if not isinstance(labels, list) or not labels:
            raise ValueError(f"Network {network_id}: missing label_order/structures")
        label_names = [str(label).strip() for label in labels]

        net_env_cfg = entry.get("env") or {}
        if not isinstance(net_env_cfg, dict):
            raise ValueError(f"Network {network_id}: env must be a mapping when provided")
        net_env = {str(k): str(v) for k, v in net_env_cfg.items() if v is not None}

        networks.append(
            NetworkDefinition(
                network_id=network_id,
                archive_path=archive_path,
                dataset_dir=dataset_dir,
                label_names=label_names,
                env=net_env,
            )
        )

    combine_cfg = nnunet_cfg.get("combine") or {}
    order_raw = combine_cfg.get("ordered_networks") or combine_cfg.get("order") or combine_cfg.get("networks")
    if isinstance(order_raw, list) and order_raw:
        combine_order = [str(item).strip() for item in order_raw if str(item).strip()]
    else:
        combine_order = [net.network_id for net in networks]

    # Keep only known networks in the final order (preserving duplicates for precedence)
    known_ids = {net.network_id for net in networks}
    combine_order = [nid for nid in combine_order if nid in known_ids]
    if not combine_order:
        combine_order = [net.network_id for net in networks]

    description = str(data.get("description") or data.get("summary") or "")

    return CustomModelDefinition(
        name=model_name,
        directory=model_dir,
        command=command,
        model=model_type,
        folds=folds,
        env=env,
        networks=networks,
        combine_order=combine_order,
        description=description,
    )


def run_custom_models_for_course(
    cfg: PipelineConfig,
    course: "CourseOutput",
    models: List[CustomModelDefinition],
    force: bool = False,
) -> None:
    if not models:
        logger.info("No custom segmentation models supplied; nothing to run")
        return

    ct_dir = course.dirs.dicom_ct
    if not ct_dir.exists():
        logger.warning("Custom segmentation skipped for %s/%s: no CT DICOM found", course.patient_id, course.course_id)
        return

    course.dirs.ensure()
    sentinel_path = course.dirs.root / ".custom_models_done"
    models_to_run = []
    for model in models:
        if force or not _model_outputs_ready(course.dirs.root, model):
            models_to_run.append(model)

    if not models_to_run and sentinel_path.exists():
        logger.info(
            "Custom segmentation outputs already present for %s/%s; skipping",
            course.patient_id,
            course.course_id,
        )
        return

    nifti_path = _ensure_ct_nifti(cfg, ct_dir, course.dirs.nifti, force=False)
    if nifti_path is None:
        raise RuntimeError(f"Unable to obtain NIfTI for course {course.patient_id}/{course.course_id}")

    for model in models_to_run or models:
        _run_single_model(cfg, course, nifti_path, ct_dir, model)

    sentinel_path.write_text("ok\n", encoding="utf-8")


def _model_outputs_ready(course_root: Path, model: CustomModelDefinition) -> bool:
    output_dir = course_root / "Segmentation_CustomModels" / model.name
    if not output_dir.exists():
        return False
    rtstruct_path = output_dir / "rtstruct.dcm"
    if not rtstruct_path.exists():
        return False
    expected_structures = model.expected_structures()
    for struct_name in expected_structures:
        fname = f"{_structure_filename(struct_name)}.nii.gz"
        if not (output_dir / fname).exists():
            return False
    return True


def _combined_env(base_env: Dict[str, str], extra: Dict[str, str]) -> Dict[str, str]:
    env = base_env.copy()
    env.update(extra)
    return env


def _prepare_network_weights(model: CustomModelDefinition, network: NetworkDefinition) -> Path:
    results_root = model.results_root()
    dataset_dir = results_root / network.dataset_dir
    if dataset_dir.exists():
        return dataset_dir

    archive_path = network.archive_path
    if not archive_path.exists():
        raise FileNotFoundError(f"Required archive not found for network {network.network_id}: {archive_path}")

    results_root.mkdir(parents=True, exist_ok=True)
    logger.info("Extracting weights for model %s network %s", model.name, network.network_id)
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(results_root)

    if not dataset_dir.exists():
        available = sorted(p.name for p in results_root.iterdir())
        raise FileNotFoundError(
            f"After extracting {archive_path.name}, expected directory '{network.dataset_dir}' "
            f"under {results_root} but found {available}"
        )
    return dataset_dir


def _command_prefix(cfg: PipelineConfig) -> str:
    activate = cfg.custom_models_conda_activate or cfg.conda_activate
    return f"{activate} && " if activate else ""


def _run_nnunet_prediction(
    cfg: PipelineConfig,
    model: CustomModelDefinition,
    network: NetworkDefinition,
    input_dir: Path,
    output_dir: Path,
    env: Dict[str, str],
) -> None:
    import shlex

    cmd_parts = [
        model.command or cfg.nnunet_predict_cmd,
        "-i",
        str(input_dir),
        "-o",
        str(output_dir),
        "-d",
        str(network.network_id),
        "-c",
        model.model,
        "-f",
        model.folds,
    ]

    cmd = f"{_command_prefix(cfg)}{' '.join(shlex.quote(part) for part in cmd_parts)}"
    logger.info("Running nnUNet prediction: %s", cmd)
    success = _run_shell(cmd, env=env)
    if not success:
        raise RuntimeError(f"nnUNet prediction failed for network {network.network_id}")


def _load_segmentation_output(output_dir: Path) -> sitk.Image:
    candidates = sorted(output_dir.glob("*.nii.gz"))
    if not candidates:
        candidates = sorted(output_dir.glob("*.nii"))
    if not candidates:
        raise FileNotFoundError(f"No nnUNet output found in {output_dir}")
    return sitk.ReadImage(str(candidates[0]))


def _make_binary_masks(seg_img: sitk.Image, label_names: List[str]) -> Dict[str, sitk.Image]:
    seg_arr = sitk.GetArrayFromImage(seg_img)
    masks: Dict[str, sitk.Image] = {}
    for label_idx, structure_name in enumerate(label_names, start=1):
        mask_arr = (seg_arr == label_idx).astype(np.uint8)
        mask_img = sitk.GetImageFromArray(mask_arr)
        mask_img.CopyInformation(seg_img)
        masks[structure_name] = mask_img
    return masks


def _write_structure_masks(
    combined_structures: Dict[str, sitk.Image],
    structure_source: Dict[str, str],
    output_dir: Path,
) -> Tuple[List[Dict[str, str]], Dict[str, Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries: List[Dict[str, str]] = []
    paths: Dict[str, Path] = {}

    # Clean previous structures for this course
    for existing in output_dir.glob("*.nii*"):
        try:
            existing.unlink()
        except Exception:
            logger.debug("Unable to remove previous mask %s", existing)

    for name, img in combined_structures.items():
        filename = f"{_structure_filename(name)}.nii.gz"
        out_path = output_dir / filename
        sitk.WriteImage(img, str(out_path))

        manifest_entries.append(
            {
                "name": name,
                "file": filename,
                "network": structure_source.get(name, ""),
            }
        )
        paths[name] = out_path

    return manifest_entries, paths


def _build_rtstruct(
    ct_dir: Path,
    structure_masks: Dict[str, sitk.Image],
    output_dir: Path,
    structure_paths: Dict[str, Path],
) -> Path:
    try:
        from rt_utils import RTStructBuilder
    except Exception as exc:
        raise RuntimeError("rt-utils is required for custom segmentation RTSTRUCT generation") from exc

    ct_img = _load_ct_image(ct_dir)
    if ct_img is None:
        raise RuntimeError(f"Failed to load CT series at {ct_dir}")

    rtstruct_path = output_dir / "rtstruct.dcm"

    if rtstruct_path.exists():
        try:
            rtstruct_path.unlink()
        except Exception:
            logger.debug("Unable to remove previous RTSTRUCT %s", rtstruct_path)

    rtstruct = RTStructBuilder.create_new(dicom_series_path=str(ct_dir))
    added_any = False
    for name, mask_img in structure_masks.items():
        resampled = _resample_to_reference(mask_img, ct_img)
        mask_arr = sitk.GetArrayFromImage(resampled)
        mask_arr = np.moveaxis(mask_arr, 0, -1)
        mask_bin = mask_arr > 0
        if not np.any(mask_bin):
            continue
        rtstruct.add_roi(mask=mask_bin, name=name)
        added_any = True

    if not added_any:
        raise RuntimeError("Custom segmentation produced no non-empty structures")

    rtstruct.save(str(rtstruct_path))
    sanitize_rtstruct(rtstruct_path)
    try:
        summary = fix_rtstruct_rois(ct_dir, rtstruct_path)
        if summary and summary.changed:
            logger.info(
                "RTSTRUCT ROI normalization for %s: %d fixed, %d pending",
                rtstruct_path,
                len(summary.fixed),
                len(summary.failed),
            )
    except Exception as exc:
        logger.debug("RTSTRUCT ROI fix failed for %s: %s", rtstruct_path, exc)

    try:
        if alias_path.exists():
            alias_path.unlink()
        shutil.copy2(rtstruct_path, alias_path)
    except Exception as exc:
        logger.debug("Unable to copy RTSTRUCT to %s: %s", alias_path, exc)

    manifest = {
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "rtstruct": "rtstruct.dcm",
        "structures": [
            {"name": name, "file": str(structure_paths[name].name)}
            for name in sorted(structure_paths)
        ],
    }
    (output_dir / "rtstruct_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return rtstruct_path


def _run_single_model(
    cfg: PipelineConfig,
    course: "CourseOutput",
    nifti_path: Path,
    ct_dir: Path,
    model: CustomModelDefinition,
) -> None:
    output_root = course.dirs.segmentation_custom_models / model.name
    output_root.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Running custom model '%s' for patient %s course %s",
        model.name,
        course.patient_id,
        course.course_id,
    )

    env = os.environ.copy()
    resolved_env = model.resolved_env()
    env.update(resolved_env)
    for key in ("nnUNet_results", "nnUNet_raw", "nnUNet_preprocessed"):
        try:
            Path(resolved_env[key]).mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.debug("Unable to ensure directory for %s (%s): %s", key, resolved_env.get(key), exc)
    if cfg.segmentation_thread_limit and cfg.segmentation_thread_limit > 0:
        thread_str = str(cfg.segmentation_thread_limit)
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
            env[var] = thread_str

    with tempfile.TemporaryDirectory(prefix=f"custom_seg_{model.name}_", dir=str(course.dirs.root)) as tmp_root_str:
        tmp_root = Path(tmp_root_str)
        input_dir = tmp_root / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        input_case = input_dir / "case_0000.nii.gz"
        shutil.copy2(nifti_path, input_case)

        structures_by_network: Dict[str, Dict[str, sitk.Image]] = {}

        for network in model.networks:
            network_env = _combined_env(env, network.env)
            _prepare_network_weights(model, network)
            output_dir = tmp_root / f"output_{network.network_id}"
            output_dir.mkdir(parents=True, exist_ok=True)
            _run_nnunet_prediction(cfg, model, network, input_dir, output_dir, network_env)
            seg_img = _load_segmentation_output(output_dir)
            structures_by_network[network.network_id] = _make_binary_masks(seg_img, network.label_names)

    combined_structures: Dict[str, sitk.Image] = {}
    structure_source: Dict[str, str] = {}
    for network_id in model.combine_order:
        for name, img in structures_by_network.get(network_id, {}).items():
            combined_structures[name] = img
            structure_source[name] = network_id

    manifest_entries, structure_paths = _write_structure_masks(
        combined_structures,
        structure_source,
        output_root,
    )

    if not manifest_entries:
        raise RuntimeError(f"No structures were produced for model {model.name}")

    rtstruct_path = _build_rtstruct(ct_dir, combined_structures, output_root, structure_paths)

    manifest = {
        "model": model.name,
        "description": model.description,
        "course_id": course.course_id,
        "patient_id": course.patient_id,
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "nnunet": {
            "command": model.command,
            "model": model.model,
            "folds": model.folds,
            "combine_order": model.combine_order,
        },
        "structures": manifest_entries,
        "rtstruct": str(rtstruct_path.name),
        "source_nifti": str(nifti_path),
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not cfg.custom_models_retain_weights:
        _cleanup_model_cache(model)


def _cleanup_model_cache(model: CustomModelDefinition) -> None:
    model_root = model.directory.resolve()
    env = model.resolved_env()
    for key in ("nnUNet_results", "nnUNet_raw", "nnUNet_preprocessed"):
        location = env.get(key)
        if not location:
            continue
        try:
            path = Path(location).resolve()
        except Exception:
            continue
        if not path.exists():
            continue
        try:
            if not path.is_dir():
                continue
        except OSError:
            continue
        try:
            if not path.is_relative_to(model_root):
                logger.debug(
                    "Skipping cleanup of %s for model %s (outside model directory)",
                    path,
                    model.name,
                )
                continue
        except Exception:
            continue
        try:
            shutil.rmtree(path)
            logger.info("Removed cached %s for model %s at %s", key, model.name, path)
        except Exception as exc:
            logger.warning("Failed to remove cached %s for model %s: %s", key, model.name, exc)


def list_custom_model_outputs(course_dir: Path) -> List[Tuple[str, Path]]:
    """Return (model_name, model_course_dir) for all custom model outputs available for a course."""
    outputs: List[Tuple[str, Path]] = []
    custom_root = course_dir / "Segmentation_CustomModels"
    if not custom_root.exists():
        return outputs
    for seg_dir in sorted(custom_root.iterdir()):
        if not seg_dir.is_dir():
            continue
        if not (seg_dir / "rtstruct.dcm").exists():
            continue
        outputs.append((seg_dir.name, seg_dir))
    return outputs
