"""Exact comparisons with the committed pre-optimization implementation."""
import ast
import builtins
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import SimpleITK as sitk

from rtpipeline import radiomics_robustness as rr
from rtpipeline import radiomics_parallel as rp

ROOT = Path(__file__).resolve().parents[1]
BASE = '5de543a'


def baseline():
    """Compile verbatim function definitions from the pinned git object."""
    namespace = dict(vars(rr))
    names = {'_select_largest_scores_deterministically', 'volume_adapt_mask',
             'randomize_contour', 'translate_mask', 'add_noise_to_image',
             'generate_ntcv_perturbations', '_perturbed_mask_identity'}
    for module, selected, target in [('radiomics_robustness', names, namespace),
                                    ('radiomics_parallel', {'_prepare_radiomics_task'}, dict(vars(rp)))]:
        source = subprocess.check_output(['git', 'show', f'{BASE}:rtpipeline/{module}.py'], cwd=ROOT, text=True)
        tree = ast.parse(source)
        definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in selected]
        assert len(definitions) == len(selected)
        if module == 'radiomics_parallel':
            def reference_import(name, globals=None, locals=None, fromlist=(), level=0):
                if name == 'radiomics_robustness' and level == 1:
                    return SimpleNamespace(**namespace)
                return builtins.__import__(name, globals, locals, fromlist, level)
            target['__builtins__'] = dict(vars(builtins), __import__=reference_import)
        exec(compile(ast.Module(body=definitions, type_ignores=[]), f'{BASE}/{module}.py', 'exec'), target)
        if module == 'radiomics_parallel':
            namespace['_prepare_radiomics_task'] = target['_prepare_radiomics_task']
    return SimpleNamespace(**namespace)


def synthetic(shape=(48, 80, 80)):
    z, y, x = np.ogrid[tuple(slice(0, n) for n in shape)]
    image = sitk.GetImageFromArray(np.broadcast_to(((x + 3*y + 7*z) % 1800 - 900).astype(np.int16), shape).copy())
    image.SetSpacing((0.98, 0.98, 3.0))
    image.SetOrigin((-40., 21., -80.))
    image.SetDirection((0., -1., 0., 1., 0., 0., 0., 0., 1.))
    scale = np.array(shape) / np.array((200, 512, 512))
    # Approximately 5 ml, 1500 ml, a border ROI, and a thin irregular ROI.
    specs = [('small', (95, 250, 250), (4, 10, 10)),
             ('large', (100, 260, 260), (35, 62, 62)),
             ('border', (100, 245, 0), (15, 32, 32)),
             ('thin', (90, 250, 250), (20, 2, 40))]
    masks = {}
    for name, center, radii in specs:
        center = np.array(center)*scale
        radii = np.maximum(np.array(radii)*scale, (2, 3, 3))
        a = sum(((grid-c)/r)**2 for grid,c,r in zip((z,y,x),center,radii)) <= 1
        if name == 'thin':
            a |= ((abs(z-center[0]) < radii[0]) & (abs(y-center[1] - 2*np.sin(x/4)) < 1.1) & (abs(x-center[2]) < radii[2]))
        mask = sitk.GetImageFromArray(a.astype(np.uint8)); mask.CopyInformation(image)
        masks[name] = mask
    return image, masks


def assert_image(a, b):
    assert a.GetSize() == b.GetSize()
    assert a.GetSpacing() == b.GetSpacing()
    assert a.GetOrigin() == b.GetOrigin()
    assert a.GetDirection() == b.GetDirection()
    assert a.GetPixelID() == b.GetPixelID()
    assert np.array_equal(sitk.GetArrayViewFromImage(a), sitk.GetArrayViewFromImage(b))


def task_context(tmp_path, roi='urinary_bladder'):
    from rtpipeline.config import PipelineConfig
    course = tmp_path / 'synthetic' / 'course'; course.mkdir(parents=True, exist_ok=True)
    params = tmp_path / 'parameters.yaml'
    params.write_text('imageType:\n  Original: {}\nfeatureClass:\n  firstorder: []\n  shape: []\nsetting:\n  minimumROISize: 10\n  minimumROIDimensions: 2\n')
    config = PipelineConfig(tmp_path, tmp_path, tmp_path); config.radiomics_params_file = params
    identity = dict(patient_id='synthetic', course_id='course', series_uid='1.2.3',
                    segmentation_source='AutoRTS_total', roi_original_name=roi,
                    mask_identity='synthetic-mask', stable_roi_identifier='synthetic-roi')
    return config, course, identity
