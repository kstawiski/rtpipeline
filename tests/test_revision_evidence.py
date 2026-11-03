"""Equal prescriptions without delivery records need positive revision evidence."""
from dataclasses import asdict

import pydicom
import pytest
from pydicom.uid import generate_uid

from rtpipeline import organize as org
from synthetic_rt_fixtures import make_plan, make_dose


def pair(root, *, distinct_isocentres=True, target='different'):
    study, frame, struct = [generate_uid() for _ in range(3)]
    plans, doses = [], []
    for i in range(2):
        uid = generate_uid()
        plan = make_plan(root / f'p{i}.dcm', uid, study_uid=study, frame_uid=frame,
                         struct_uid=struct, date='20240101', rx_gy=30, fractions=10)
        ds = pydicom.dcmread(plan)
        ds.BeamSequence[0].ControlPointSequence[0].IsocenterPosition = [100 * i if distinct_isocentres else 0, 0, 0]
        if target == 'absent' or (target == 'partial' and i == 1):
            del ds.DoseReferenceSequence[0].DoseReferenceUID
        ds.save_as(plan, enforce_file_format=True)
        plans.append(plan)
        doses.append(make_dose(root / f'd{i}.dcm', generate_uid(), plan_uid=uid,
                               study_uid=study, frame_uid=frame))
    return plans, doses


@pytest.mark.parametrize('distinct_isocentres', [False, True])
@pytest.mark.parametrize('target', ['different', 'absent', 'partial'])
@pytest.mark.parametrize('reverse', [False, True])
def test_unproven_revisions_held_then_positive_evidence_allows_collapse(
        tmp_path, monkeypatch, distinct_isocentres, target, reverse):
    monkeypatch.delenv('RTPIPELINE_CONCURRENT_COMPONENTS', raising=False)
    plans, doses = pair(tmp_path, distinct_isocentres=distinct_isocentres, target=target)
    if reverse:
        plans.reverse()
        doses.reverse()
    held = org._classify_doses(plans, doses)
    assert held.classification == org.INDEPENDENT_DELIVERY_UNRECONCILED
    assert set(held.selected_plans) == set(plans)
    assert held.selected_doses == [] and not held.should_sum
    assert set(held.excluded_doses) == set(doses)
    assert asdict(org._classify_doses(plans[::-1], doses[::-1]))['classification'] == held.classification

    if target == 'different':
        # Matching TARGET UIDs plus a later calendar day permit a moved isocentre.
        first, second = [pydicom.dcmread(p) for p in plans]
        second.DoseReferenceSequence[0].DoseReferenceUID = first.DoseReferenceSequence[0].DoseReferenceUID
        second.save_as(plans[1], enforce_file_format=True)
        if distinct_isocentres:
            same_day = org._classify_doses(plans, doses)
            assert same_day.classification == org.INDEPENDENT_DELIVERY_UNRECONCILED
            second.RTPlanDate = '20240102'
            second.save_as(plans[1], enforce_file_format=True)
    else:
        # With no complete target identity, only strict equivalence suffices.
        # Byte-identical copies of one SOP are the strongest strict proof.
        plans[1].write_bytes(plans[0].read_bytes())
        doses = doses[:1]
        same = pydicom.dcmread(doses[0])
        same.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID = pydicom.dcmread(plans[0]).SOPInstanceUID
        same.save_as(doses[0], enforce_file_format=True)
        assert org._strict_plan_equivalence(org._plan_evidence(plans[0]), org._plan_evidence(plans[1]))[0]
    revised = org._classify_doses(plans, doses)
    assert revised.classification == 'plan_revisions_deduplicated'
    assert len(revised.selected_plans) == len(revised.selected_doses) == 1
    assert not revised.should_sum
