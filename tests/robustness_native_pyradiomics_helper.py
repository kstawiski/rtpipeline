"""Pytest-free execution of the two real-PyRadiomics robustness cases.

Why this module exists
----------------------
``tests/test_robustness_nonmeasurements.py`` runs under the main campaign
interpreter (``rtpipeline``, Python 3.11, NumPy 2.x), which deliberately has no
native PyRadiomics. Two cases in that file need a *real* extractor, so they
re-launch themselves in the separate radiomics interpreter
(``rtpipeline-radiomics``, Python 3.10, NumPy 1.26.4, PyRadiomics 3.0.1).

That interpreter has the scientific stack but **no ``pytest``**, so the previous
``python -m pytest ...`` re-launch could never start. This module is the
pytest-free entry point it launches instead: it performs the identical real work
with the identical production imports and prints the measured result as JSON on
its last stdout line. Every assertion stays on the manager host, which rebuilds
the frame from the returned rows and runs the original checks unchanged.

Nothing here is a stand-in for the measurement: no mock extractor, no simplified
geometry, no alternative feature algorithm. Only the *transport* changed.

Usage (as launched by the test module)::

    <radiomics-python> -B -E -s <this file> <case> <scratch-dir>

``case`` is ``resampled_minimum`` or ``identical_condition_retry``. Every byte
written goes under ``scratch-dir``, which the caller owns.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Import the production modules from this working tree, not from any installed
# copy, so the helper certifies the same bytes the host test file does.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import SimpleITK as sitk

from radiomics import featureextractor

from rtpipeline import radiomics_parallel as rp
from rtpipeline import radiomics_robustness as rr
from rtpipeline.config import PipelineConfig
from rtpipeline.radiomics_robustness_outcomes import extraction_nonmeasurement

PARAMS_YAML = (
    'imageType:\n  Original: {}\nfeatureClass:\n  firstorder: []\n  shape: []\n'
    'setting:\n  minimumROISize: 10\n  minimumROIDimensions: 2\n'
)

RETRY_IDENTITY = dict(
    patient_id='P', course_id='C', series_uid='1.2.3',
    segmentation_source='AutoRTS_total', roi_original_name='urinary_bladder',
    mask_identity='source-mask', stable_roi_identifier='roi-1',
)


def _json_ready(value):
    """Transport measured values without changing them.

    NumPy scalars become their exact Python counterparts; NaN and infinities
    survive the round trip, so the host's finiteness assertion still tests the
    real extraction rather than a laundered copy. An unexpected type raises
    instead of being coerced to a string.
    """
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"helper cannot transport {type(value).__name__} without altering it")


def case_resampled_minimum(scratch: Path) -> dict:
    """Real extractor typing of an 8-voxel ROI against minimumROISize=64."""
    a = np.zeros((8, 8, 8), np.uint8)
    a[2:4, 2:4, 2:4] = 1
    mask = sitk.GetImageFromArray(a)
    image = sitk.GetImageFromArray(a.astype(np.float32))
    factory = lambda: featureextractor.RadiomicsFeatureExtractor(
        minimumROISize=64, minimumROIDimensions=2
    )
    outcome = extraction_nonmeasurement(image, mask, factory)
    if outcome is None:
        raise RuntimeError("real PyRadiomics returned no geometric non-measurement")
    return {
        'reason_code': outcome.reason_code,
        'evidence': _json_ready(dict(outcome.evidence)),
        'native_voxels_written': int(a.sum()),
    }


def case_identical_condition_retry(scratch: Path) -> dict:
    """Real extraction after one injected transient failure on the same task."""
    course = scratch / 'P' / 'C'
    course.mkdir(parents=True)
    temp = scratch / 'inputs'
    temp.mkdir()
    params = scratch / 'params.yaml'
    params.write_text(PARAMS_YAML)
    array = np.zeros((12, 12, 12), np.uint8)
    array[2:10, 2:10, 2:10] = 1
    mask = sitk.GetImageFromArray(array)
    image = sitk.GetImageFromArray(
        np.arange(array.size, dtype=np.float32).reshape(array.shape) % 100
    )
    cfg = PipelineConfig(scratch, scratch, scratch)
    cfg.radiomics_params_file = params
    mask_path, task = rp._prepare_radiomics_task(
        image, mask, cfg, 'AutoRTS_total', 'urinary_bladder', course, temp, False,
        source_identity=RETRY_IDENTITY,
    )
    task['extra_metadata'] = {'perturbation_id': 'ntcv_v0'}

    original = rp._isolated_radiomics_extraction
    seen = []

    def transient_once(same_task):
        seen.append(same_task)
        if len(seen) == 1:
            raise OSError('injected transient read failure')
        return original(same_task)

    # monkeypatch is a pytest fixture and pytest is absent here; restore by hand.
    rp._isolated_radiomics_extraction = transient_once
    try:
        result = rp._isolated_radiomics_extraction_with_retry((mask_path, task))
    finally:
        rp._isolated_radiomics_extraction = original

    return {
        'attempt_count': len(seen),
        'same_task_object': len(seen) > 1 and seen[0] is seen[1],
        'robustness_attempts': result.get('robustness_attempts'),
        # The real production flattener runs here because it needs the worker
        # result object; the host re-validates and asserts on these rows.
        'rows': _json_ready(rr._feature_rows_from_worker_result(result)),
        'source_identity': dict(RETRY_IDENTITY),
    }


CASES = {
    'resampled_minimum': case_resampled_minimum,
    'identical_condition_retry': case_identical_condition_retry,
}


def main(argv) -> int:
    if len(argv) != 3 or argv[1] not in CASES:
        print(f"usage: {argv[0]} {{{'|'.join(CASES)}}} <scratch-dir>", file=sys.stderr)
        return 2
    scratch = Path(argv[2])
    scratch.mkdir(parents=True, exist_ok=True)
    payload = CASES[argv[1]](scratch)
    payload['interpreter'] = sys.executable
    payload['versions'] = {
        'python': sys.version.split()[0],
        'numpy': np.__version__,
        'SimpleITK': sitk.__version__,
        'pyradiomics': __import__('radiomics').__version__,
    }
    print(json.dumps(payload))
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
