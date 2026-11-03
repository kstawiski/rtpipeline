"""Synthetic adaptation of Lucjan's resume-twice warning regression."""
import hashlib
import json

import pytest

from test_review_repairs import declaration

from organize_io_fixture import synthetic, synthetic_ct_conversion
from review_organize_fixture import build
from rtpipeline import organize as org
from rtpipeline.config import PipelineConfig


@pytest.mark.parametrize("declared,fixture", [(False, "review"), (True, "review"), (False, "original")],
                         ids=["sequential-and-single", "declared-and-sequential", "original-single"])
def test_summed_course_contract_digest_survives_first_resume(tmp_path, monkeypatch, declared, fixture):
    """F3, extended from WS270: whole metadata and summed-course digest."""
    root, output = tmp_path / 'input', tmp_path / 'output'
    if fixture == "original":
        for record in synthetic(root):
            record.unlink()
        expected_courses = 4
    else:
        uids = build(root)
        expected_courses = 5
    monkeypatch.delenv('RTPIPELINE_CONCURRENT_COMPONENTS', raising=False)
    if declared:
        path = declaration(tmp_path / 'decl.json', uids['D'].values())
        monkeypatch.setenv('RTPIPELINE_CONCURRENT_COMPONENTS', str(path))
    monkeypatch.setattr(org, 'run_dcm2niix', synthetic_ct_conversion)
    config = PipelineConfig(root, output, tmp_path / 'logs', max_workers_override=1,
                            dicom_copy_use_hardlinks=False, dicom_copy_dedup_by_sop_uid=False)

    def snapshot():
        result = {}
        for path in output.glob('*/*/metadata/case_metadata.json'):
            metadata = json.loads(path.read_text())
            contract = metadata['course_contract']
            result[str(path.relative_to(output))] = (
                contract['delivery']['warnings'], metadata['delivery_warnings'],
                hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest(),
                metadata,
            )
        assert len(result) == expected_courses
        assert sum(any(w.startswith('Dose-response eligibility withheld:') for w in item[0])
                   for item in result.values()) == 4
        if fixture == "review":
            assert any(item[3]['course_contract']['dose_classification']['classification']
                       == 'sequential_phases_summed' for item in result.values())
            if declared:
                assert any(item[3]['course_contract']['dose_classification']['classification']
                           == org.DECLARED_CONCURRENT_COMPONENTS for item in result.values())
        return result

    first_courses = org.organize_and_merge(config)
    assert len(first_courses) == expected_courses
    first = snapshot()
    # A rebuild could hide the bug; prove the subsequent runs hydrate the courses.
    def no_rebuild(*args, **kwargs):
        raise AssertionError('resume rebuilt a synthetic course')
    monkeypatch.setattr(org, 'run_dcm2niix', no_rebuild)
    config.resume = True
    for _ in range(2):
        assert len(org.organize_and_merge(config)) == expected_courses
        current = snapshot()
        def differences(before, after, path=""):
            if isinstance(before, dict) and isinstance(after, dict):
                return [diff for key in before.keys() | after.keys()
                        for diff in differences(before.get(key), after.get(key), f"{path}/{key}")]
            return [] if before == after else [(path, before, after)]
        assert current == first, differences(
            {key: item[3] for key, item in first.items()},
            {key: item[3] for key, item in current.items()},
        )
