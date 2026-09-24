"""Plain-script real-feature equivalence check in the PyRadiomics environment."""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import robustness_performance_fixture as h

h.sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
(ROOT/'.robustness-perf').mkdir(exist_ok=True)
old = h.baseline()
image, masks = h.synthetic()
config_pert = h.rr.PerturbationConfig()
conditions = rows_compared = 0
with tempfile.TemporaryDirectory(prefix='native-equivalence-', dir=ROOT/'.robustness-perf') as scratch:
    tmp = Path(scratch)
    cfg, course, identity = h.task_context(tmp)
    for name, mask in masks.items():
        before_masks, before_images = old.generate_ntcv_perturbations(mask, image, config_pert, name)
        after_masks, after_images = h.rr.generate_ntcv_perturbations(mask, image, config_pert, name)
        cache = {}
        for key in before_masks:
            h.assert_image(before_images[key], after_images[key])
            if isinstance(before_masks[key], h.rr.GeometricNonmeasurement):
                assert before_masks[key] == after_masks[key]
                continue
            h.assert_image(before_masks[key], after_masks[key])
            results = []
            for prepare, pm, pi, kwargs in (
                (old._prepare_radiomics_task, before_masks, before_images, {}),
                (h.rp._prepare_radiomics_task, after_masks, after_images, {'_cache': cache}),
            ):
                task = prepare(pi[key], pm[key], cfg, 'AutoRTS_total', 'urinary_bladder', course,
                               tmp, False, 'synthetic-run', identity, **kwargs)
                task[1]['extra_metadata'] = {'perturbation_id': key}
                result = h.rp._isolated_radiomics_extraction(task)
                results.append(h.rr._feature_rows_from_worker_result(result))
            # JSON preserves scalar floating-point round trips and also compares
            # all identifiers, missingness and contract fields in the rows.
            encoded = [json.dumps(r, sort_keys=True, default=lambda x: x.item()) for r in results]
            assert encoded[0] == encoded[1], (name, key)
            rows_compared += len(results[0])
            conditions += 1
        print(json.dumps({'roi': name, 'conditions_so_far': conditions, 'rows_so_far': rows_compared}), flush=True)
print(json.dumps({'exact_conditions': conditions, 'exact_feature_rows': rows_compared}), flush=True)
