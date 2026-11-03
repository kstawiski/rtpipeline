"""Synthetic adaptation of Lucjan's resume-twice warning regression."""
import hashlib
import json

from organize_io_fixture import synthetic, synthetic_ct_conversion
from rtpipeline import organize as org
from rtpipeline.config import PipelineConfig


def test_resume_twice_preserves_warnings_and_contract(tmp_path, monkeypatch):
    root, output = tmp_path / 'input', tmp_path / 'output'
    for record in synthetic(root):
        record.unlink()
    monkeypatch.delenv('RTPIPELINE_CONCURRENT_COMPONENTS', raising=False)
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
            )
        assert len(result) == 4
        assert all(any(w.startswith('Dose-response eligibility withheld:') for w in item[0])
                   for item in result.values())
        return result

    first_courses = org.organize_and_merge(config)
    assert len(first_courses) == 4
    first = snapshot()
    # A rebuild could hide the bug; prove the subsequent runs hydrate the courses.
    def no_rebuild(*args, **kwargs):
        raise AssertionError('resume rebuilt a synthetic course')
    monkeypatch.setattr(org, 'run_dcm2niix', no_rebuild)
    config.resume = True
    for _ in range(2):
        assert len(org.organize_and_merge(config)) == 4
        assert snapshot() == first
