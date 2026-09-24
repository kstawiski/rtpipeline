"""Synthetic parent-only benchmark; run separately for each implementation."""
import cProfile
import gc
import importlib.util
import json
import pstats
import resource
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('equivalence', ROOT/'tests/robustness_performance_fixture.py')
h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
h.sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
mode = sys.argv[1]
if mode not in {"before", "after"}:
    raise SystemExit("usage: benchmark_robustness_parent.py before|after")
(ROOT/'.robustness-perf').mkdir(exist_ok=True)
impl = h.baseline() if mode == 'before' else h.rr
prepare = impl._prepare_radiomics_task if mode == 'before' else h.rp._prepare_radiomics_task
image, masks = h.synthetic((200, 512, 512))
config_pert = h.rr.PerturbationConfig()
print('config', config_pert, flush=True)
profile = cProfile.Profile()
rows = []
noise_cache = {}
image_digest_cache = {}
with tempfile.TemporaryDirectory(prefix='profile-', dir=ROOT/'.robustness-perf') as scratch:
    tmp = Path(scratch)
    config, course, identity = h.task_context(tmp)
    for name, mask in masks.items():
        gc.collect()
        start = time.perf_counter(); profile.enable()
        kwargs = {'_noise_cache': noise_cache} if mode == 'after' else {}
        pm, pi = impl.generate_ntcv_perturbations(mask, image, config_pert, name, **kwargs)
        generation_peak_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        prepared = 0
        cache = {'images': image_digest_cache} if mode == 'after' else None
        for key, perturbed in pm.items():
            if isinstance(perturbed, h.rr.GeometricNonmeasurement): continue
            extra = {'_cache': cache} if mode == 'after' else {}
            prepare(pi[key], perturbed, config, 'AutoRTS_total', 'urinary_bladder', course, tmp, False,
                    'synthetic-run', identity, **extra)
            prepared += 1
        profile.disable()
        rows.append(dict(roi=name, seconds=time.perf_counter()-start, conditions=len(pm), tasks=prepared,
                         generation_peak_rss_kib=generation_peak_rss_kib,
                         peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                         volume_ml=int(h.np.count_nonzero(h.sitk.GetArrayViewFromImage(mask)))*.98*.98*3/1000))
        print(rows[-1], flush=True)
        if cache is not None:
            cache.clear()  # Match the course loop: release mask references before the next ROI.
        del pm, pi, cache, perturbed, extra
profile.dump_stats(str(ROOT/'.robustness-perf'/f'{mode}.prof'))
stats = pstats.Stats(profile)
functions = []
for (file,line,name), (cc,nc,tt,ct,callers) in stats.stats.items():
    if name in {'volume_adapt_mask','SignedMaurerDistanceMap','_select_largest_scores_deterministically',
                '_perturbed_mask_identity','_prepare_radiomics_task','generate_ntcv_perturbations',
                'randomize_contour','translate_mask','add_noise_to_image','classify_ct_roi',
                'configured_parameter_hash','load_custom_structure_provenance','WriteImage'} or 'openssl_sha1' in name or 'tobytes' in name:
        functions.append(dict(function=name,calls=nc,self_seconds=tt,cumulative_seconds=ct))
result = dict(mode=mode, rois=rows, functions=functions, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
(ROOT/'.robustness-perf'/f'{mode}.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result, indent=2), flush=True)
