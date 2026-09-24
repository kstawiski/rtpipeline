"""Synthetic organize inputs and verbatim readers from the pinned baseline."""
import builtins
import io
import json
import re
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import pydicom
from pydicom.uid import generate_uid

from synthetic_rt_fixtures import make_plan, make_dose, make_struct, make_record, _new_dataset

ROOT = Path(__file__).resolve().parents[1]
BASE = '3af9f51'


def baseline():
    modules = {}

    def imports(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 1 and name in modules:
            return modules[name]
        return builtins.__import__(name, globals, locals, fromlist, level)

    for name in ('utils', 'dicom_copy', 'meta', 'rt_details', 'course_contract', 'plan_disposition', 'organize'):
        source = subprocess.check_output(
            ['git', 'show', f'{BASE}:rtpipeline/{name}.py'], cwd=ROOT, text=True
        )
        module = types.ModuleType(f'rtpipeline._organize_io_before_{name}')
        module.__package__ = 'rtpipeline'
        module.__file__ = str(ROOT / 'rtpipeline' / f'{name}.py')
        module.__dict__['__builtins__'] = dict(vars(builtins), __import__=imports)
        sys.modules[module.__name__] = module
        exec(compile(source, f'{BASE}/rtpipeline/{name}.py', 'exec'), module.__dict__)
        modules[name] = module
    return modules


def synthetic(root, *, records_per_course=3, ct_slices=2, ct_size=8, roi_names=("BODY", "PTV1"), all_slices=False):
    records = []
    for patient in ('SYNTH_A', 'SYNTH_B'):
        for course in range(2):
            folder = root / patient / str(course)
            study, frame, struct, plan, dose = [generate_uid() for _ in range(5)]
            common = dict(study_uid=study, frame_uid=frame, patient_id=patient)
            make_struct(folder / 'struct.dcm', struct, roi_names=roi_names, **common)
            make_plan(folder / 'plan.dcm', plan, struct_uid=struct,
                      date=f'20240{course + 1}01', fractions=records_per_course,
                      rx_gy=6.0, **common)
            make_dose(folder / 'dose.dcm', dose, plan_uid=plan, with_pixels=True,
                      dose_gy=6.0, **common)
            for index in range(records_per_course):
                records.append(make_record(
                    folder / f'record{index}.dcm', plan, patient_id=patient,
                    fraction_number=index + 1, date=f'20240{course + 1}02',
                ))
            series = generate_uid()
            images = []
            for index in range(ct_slices):
                image = _new_dataset('1.2.840.10008.5.1.4.1.1.2', generate_uid(), patient_id=patient)
                image.Modality = 'CT'
                image.StudyInstanceUID, image.SeriesInstanceUID = study, series
                image.FrameOfReferenceUID = frame
                image.InstanceNumber = index + 1
                image.ImagePositionPatient = [-10., -10., float(index * 5)]
                image.ImageOrientationPatient = [1., 0., 0., 0., 1., 0.]
                image.PixelSpacing = [5., 5.]
                image.SliceThickness = 5.
                image.Rows = image.Columns = ct_size
                image.SamplesPerPixel = 1
                image.PhotometricInterpretation = 'MONOCHROME2'
                image.BitsAllocated = image.BitsStored = 16
                image.HighBit = 15
                image.PixelRepresentation = 0
                image.PixelData = b'\0' * (2 * ct_size * ct_size)
                image.save_as(folder / f'ct{index}.dcm', enforce_file_format=True)
                ref = pydicom.Dataset()
                ref.ReferencedSOPClassUID = image.SOPClassUID
                ref.ReferencedSOPInstanceUID = image.SOPInstanceUID
                images.append(ref)
            if images:
                structure = pydicom.dcmread(folder / 'struct.dcm')
                ref = pydicom.Dataset()
                ref.SeriesInstanceUID = series
                ref.ContourImageSequence = images
                structure.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence = [ref]
                for roi in structure.ROIContourSequence:
                    if all_slices:
                        import copy
                        template = roi.ContourSequence[0]
                        roi.ContourSequence = [copy.deepcopy(template) for _ in images]
                    for i, contour in enumerate(roi.ContourSequence):
                        contour.ContourImageSequence = [images[i]]
                        if all_slices:
                            points = list(contour.ContourData)
                            points[2::3] = [float(i * 5)] * (len(points) // 3)
                            contour.ContourData = points
                structure.save_as(folder / 'struct.dcm', enforce_file_format=True)
            # A separate path with exactly the same SOP and source bytes.
            duplicate = folder / 'duplicate' / 'plan.dcm'
            duplicate.parent.mkdir()
            duplicate.write_bytes((folder / 'plan.dcm').read_bytes())
    records.append(make_record(root / 'SYNTH_A' / 'unresolved.dcm', generate_uid(),
                               patient_id='SYNTH_A'))
    return records


def tree_bytes(root):
    """Compare every file; normalize only explicitly run-dependent fields.

    Same output path is reused in sequential runs, so no path substitution is
    needed. Ledger generated_at is a wall-clock timestamp. XLSX ZIP timestamps
    and core created/modified properties are workbook packaging timestamps;
    all member names and all remaining uncompressed XML bytes are compared.
    """
    result = {}
    for path in sorted(root.rglob('*')):
        relative = str(path.relative_to(root))
        if path.is_dir():
            result[relative + '/'] = b''
            continue
        data = path.read_bytes()
        if path.suffix == '.xlsx':
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                members = {}
                for name in archive.namelist():
                    content = archive.read(name)
                    if name == 'docProps/core.xml':
                        content = re.sub(rb'(<dcterms:(?:created|modified)[^>]*>)[^<]*(</dcterms:(?:created|modified)>)',
                                         rb'\1TIMESTAMP\2', content)
                    members[name] = content
                result[relative] = members
        elif path.name == 'organize_ledger.json':
            # Preserve byte formatting and every other field.
            result[relative] = re.sub(rb'("generated_at": ")[^"]+"', rb'\1TIMESTAMP"', data)
        else:
            result[relative] = data
    return result


def publish_manifest(courses, output):
    """Exercise the workflow's actual manifest serializer without launching it."""
    import runpy
    helpers = runpy.run_path(str(ROOT / 'workflow/scripts/organize_courses.py'))
    ledger = json.loads((output / '_COURSES/organize_ledger.json').read_text())
    entries = sorted((dict(patient=course.patient_id, course=course.course_id,
                           path=str(course.dirs.root),
                           complexity=helpers['_estimate_course_complexity'](course.dirs.root))
                      for course in courses), key=lambda row: (row['patient'], row['course']))
    payload = helpers['_manifest_payload'](ledger, entries)
    path = output / 'manifests/courses.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    helpers['_write_text_atomic'](path, json.dumps(payload, indent=2, sort_keys=True) + '\n')


def require_process_pool():
    """Do not mistake the infrastructure fallback for process-path coverage."""
    import multiprocessing
    import os
    from concurrent.futures import ProcessPoolExecutor
    import pytest
    from rtpipeline.organize_scale import _ready
    try:
        with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context('spawn')) as pool:
            assert pool.submit(_ready).result() != os.getpid()
    except (OSError, RuntimeError) as exc:
        pytest.skip(f'process pool unavailable: {exc}')
