"""A/B correctness proof for the parallelized organize/pre-segmentation discovery loops.

``index_ct_series`` (rtpipeline/ct.py), ``extract_rt`` (rtpipeline/rt_details.py), and
``_index_series_and_registrations`` (rtpipeline/organize.py) each used to read every
DICOM file under the (cohort-scoped) tree in a single-threaded loop. NFS-backed trees
are latency- not CPU-bound, so those loops were parallelized with a shared thread-pool
helper (``rtpipeline.utils.parallel_map_files``). Only the per-file *read* is
parallelized; the classification/assembly logic still runs single-threaded over the
results in the SAME order the files were discovered in, so the output cannot depend on
thread completion order.

Each test below builds a small synthetic DICOM tree (CT + RTPLAN + RTDOSE + RTSTRUCT +
one non-CT/RT series [MR] + a REG, under a STUDY-level directory so the CT indexer's
TCIA fast-path bails to the generic per-file loop under test) and asserts:

1. ``func(..., max_workers=1) == func(..., max_workers=8)`` -- deep equality, proving
   the parallel path reproduces the single-threaded path exactly.
2. The result matches a hand-specified expected index -- so the test is not merely
   self-consistent, it pins down the actual expected content.
"""
from __future__ import annotations

import dataclasses
import os
import threading
import time
from pathlib import Path

import pytest
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import (
    CTImageStorage,
    ExplicitVRLittleEndian,
    RTDoseStorage,
    RTPlanStorage,
    RTStructureSetStorage,
    generate_uid,
)

from rtpipeline.ct import CTInstance, index_ct_series
from rtpipeline.organize import _index_series_and_registrations
from rtpipeline.rt_details import extract_rt
from rtpipeline.utils import (
    DEFAULT_INDEX_WORKERS,
    _default_index_workers,
    _scoped_walk,
    parallel_map_files,
    read_dicom,
)


# --------------------------------------------------------------------------- helpers
def _base_ds(path: Path, sop_class_uid: str, sop_instance_uid: str, *, patient: str,
             study: str, series: str, modality: str) -> FileDataset:
    fm = FileMetaDataset()
    fm.MediaStorageSOPClassUID = sop_class_uid
    fm.MediaStorageSOPInstanceUID = sop_instance_uid
    fm.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=fm, preamble=b"\0" * 128)
    ds.SOPClassUID = sop_class_uid
    ds.SOPInstanceUID = sop_instance_uid
    ds.PatientID = patient
    ds.Modality = modality
    ds.StudyInstanceUID = study
    ds.SeriesInstanceUID = series
    return ds


def _save(ds: FileDataset, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(path), write_like_original=False)
    return path


def _mk_ct(path: Path, *, patient: str, study: str, series: str, instance: int | None) -> str:
    sop = generate_uid()
    ds = _base_ds(path, CTImageStorage, sop, patient=patient, study=study, series=series, modality="CT")
    # instance=None -> omit InstanceNumber entirely, so get((0020,0013)) returns None
    # (exercises the "instance_number is None" tail of the per-series sort key).
    if instance is not None:
        ds.InstanceNumber = instance
    _save(ds, path)
    return sop


def _mk_plan(path: Path, *, patient: str, study: str, series: str, sop: str,
             frame_of_reference_uid: str, label: str = "PlanA") -> None:
    ds = _base_ds(path, RTPlanStorage, sop, patient=patient, study=study, series=series, modality="RTPLAN")
    ds.RTPlanLabel = label
    ds.RTPlanName = label
    ds.RTPlanDate = "20240101"
    # rt_details.extract_rt reads frame_of_reference_uid via tag (3006,0024)
    # ReferencedFrameOfReferenceUID (not the (0020,0052) FrameOfReferenceUID attribute)
    # -- set the exact tag the code under test looks at.
    ds.ReferencedFrameOfReferenceUID = frame_of_reference_uid
    _save(ds, path)


def _mk_dose(path: Path, *, patient: str, study: str, series: str, sop: str,
             plan_sop: str, frame_of_reference_uid: str) -> None:
    ds = _base_ds(path, RTDoseStorage, sop, patient=patient, study=study, series=series, modality="RTDOSE")
    ds.ReferencedFrameOfReferenceUID = frame_of_reference_uid
    ref = Dataset()
    ref.ReferencedSOPClassUID = RTPlanStorage
    ref.ReferencedSOPInstanceUID = plan_sop
    ds.ReferencedRTPlanSequence = Sequence([ref])
    _save(ds, path)


def _mk_struct(path: Path, *, patient: str, study: str, series: str, sop: str,
               frame_of_reference_uid: str, roi_names: list[str]) -> None:
    ds = _base_ds(path, RTStructureSetStorage, sop, patient=patient, study=study, series=series, modality="RTSTRUCT")
    ds.ReferencedFrameOfReferenceUID = frame_of_reference_uid
    roi_items = []
    for i, name in enumerate(roi_names, start=1):
        roi = Dataset()
        roi.ROINumber = i
        roi.ROIName = name
        roi_items.append(roi)
    ds.StructureSetROISequence = Sequence(roi_items)
    _save(ds, path)


def _mk_mr(path: Path, *, patient: str, study: str, series: str) -> None:
    ds = _base_ds(path, "1.2.840.10008.5.1.4.1.1.4", generate_uid(), patient=patient,
                  study=study, series=series, modality="MR")
    _save(ds, path)


def _mk_reg(path: Path, *, patient: str, study: str, series: str,
            frame_of_reference_uid: str, referenced_series: str) -> None:
    ds = _base_ds(path, "1.2.840.10008.5.1.4.1.1.66.1", generate_uid(), patient=patient,
                  study=study, series=series, modality="REG")
    rt_series = Dataset()
    rt_series.SeriesInstanceUID = referenced_series
    rt_study = Dataset()
    rt_study.RTReferencedSeriesSequence = Sequence([rt_series])
    ref_for = Dataset()
    ref_for.FrameOfReferenceUID = frame_of_reference_uid
    ref_for.RTReferencedStudySequence = Sequence([rt_study])
    ds.ReferencedFrameOfReferenceSequence = Sequence([ref_for])
    _save(ds, path)


@dataclasses.dataclass
class Fixture:
    root: Path
    patient: str
    study: str
    ct_series: str
    rp_series: str
    rd_series: str
    rs_series: str
    mr_series: str
    reg_series: str
    plan_sop: str
    dose_sop: str
    struct_sop: str
    frame_of_reference_uid: str
    ct_instance_numbers: dict  # {filename: instance_number}


def _build_fixture(tmp_path: Path) -> Fixture:
    """Build a study-level directory (CT fast-path bails) with CT + RTPLAN + RTDOSE +
    RTSTRUCT + MR (non-CT/RT) + REG, all sharing one patient/study."""
    patient = "PT1"
    study = generate_uid()
    ct_series = generate_uid()
    rp_series, rd_series, rs_series, mr_series, reg_series = (generate_uid() for _ in range(5))
    plan_sop, dose_sop, struct_sop = generate_uid(), generate_uid(), generate_uid()
    frame_of_reference_uid = generate_uid()

    # Study-level directory (named by StudyInstanceUID, NOT a per-series dir) so the
    # TCIA fast-path (_index_tcia_patient_series_layout) bails to the generic per-file
    # CT indexing loop under test.
    d = tmp_path / patient / study

    # 5 CT slices with SHUFFLED instance numbers, to prove the existing final
    # instance-number sort still runs correctly after parallelizing the reads.
    instance_by_name = {"CT_0.dcm": 3, "CT_1.dcm": 1, "CT_2.dcm": 5, "CT_3.dcm": 2, "CT_4.dcm": 4}
    for name, inst in instance_by_name.items():
        _mk_ct(d / name, patient=patient, study=study, series=ct_series, instance=inst)

    _mk_plan(d / "RP.dcm", patient=patient, study=study, series=rp_series, sop=plan_sop,
              frame_of_reference_uid=frame_of_reference_uid)
    _mk_dose(d / "RD.dcm", patient=patient, study=study, series=rd_series, sop=dose_sop,
             plan_sop=plan_sop, frame_of_reference_uid=frame_of_reference_uid)
    _mk_struct(d / "RS.dcm", patient=patient, study=study, series=rs_series, sop=struct_sop,
               frame_of_reference_uid=frame_of_reference_uid, roi_names=["PTV"])
    _mk_mr(d / "MR_0.dcm", patient=patient, study=study, series=mr_series)
    _mk_reg(d / "REG.dcm", patient=patient, study=study, series=reg_series,
            frame_of_reference_uid=frame_of_reference_uid, referenced_series=ct_series)

    return Fixture(
        root=tmp_path, patient=patient, study=study, ct_series=ct_series,
        rp_series=rp_series, rd_series=rd_series, rs_series=rs_series, mr_series=mr_series,
        reg_series=reg_series, plan_sop=plan_sop, dose_sop=dose_sop, struct_sop=struct_sop,
        frame_of_reference_uid=frame_of_reference_uid, ct_instance_numbers=instance_by_name,
    )


# --------------------------------------------------------------------------- 1) index_ct_series
def test_index_ct_series_parallel_matches_serial(tmp_path):
    fx = _build_fixture(tmp_path)

    idx1 = index_ct_series(fx.root, max_workers=1)
    idx8 = index_ct_series(fx.root, max_workers=8)

    assert idx1 == idx8, "parallel (max_workers=8) must reproduce the serial (max_workers=1) result exactly"

    # Hand-specified expected content (not just self-consistency).
    assert set(idx1.keys()) == {fx.patient}
    assert set(idx1[fx.patient].keys()) == {fx.study}
    series_map = idx1[fx.patient][fx.study]
    assert set(series_map.keys()) == {fx.ct_series}, "only the CT series must be indexed (RP/RD/RS/MR/REG excluded)"

    series = series_map[fx.ct_series]
    assert len(series) == 5
    # The existing end-sort must still order strictly by instance_number, regardless
    # of the (parallel, out-of-order-completing) read order.
    assert [inst.instance_number for inst in series] == [1, 2, 3, 4, 5]
    expected_name_by_instance = {v: k for k, v in fx.ct_instance_numbers.items()}
    assert [inst.path.name for inst in series] == [expected_name_by_instance[i] for i in range(1, 6)]
    for inst in series:
        assert inst.patient_id == fx.patient
        assert inst.study_uid == fx.study
        assert inst.series_uid == fx.ct_series


def test_index_ct_series_max_workers_one_is_plain_serial_loop(tmp_path):
    """max_workers=1 must take the no-executor branch (parallel_map_files yields
    fn(item) one at a time with no thread pool), so it is byte-for-byte the historical
    single-threaded behaviour with zero concurrency involved."""
    fx = _build_fixture(tmp_path)
    idx_default = index_ct_series(fx.root)  # max_workers=None -> env default
    idx_serial = index_ct_series(fx.root, max_workers=1)
    assert idx_default == idx_serial


# --------------------------------------------------------------------------- 2) extract_rt
def test_extract_rt_parallel_matches_serial(tmp_path):
    fx = _build_fixture(tmp_path)

    plans1, doses1, structs1 = extract_rt(fx.root, max_workers=1)
    plans8, doses8, structs8 = extract_rt(fx.root, max_workers=8)

    assert (plans1, doses1, structs1) == (plans8, doses8, structs8)

    assert len(plans1) == 1 and len(doses1) == 1 and len(structs1) == 1
    plan, dose, struct = plans1[0], doses1[0], structs1[0]

    assert plan.path.name == "RP.dcm"
    assert plan.patient_id == fx.patient
    assert plan.sop_instance_uid == fx.plan_sop
    assert plan.study_uid == fx.study
    assert plan.plan_label == "PlanA"
    assert plan.plan_name == "PlanA"
    assert plan.plan_date == "20240101"
    assert plan.frame_of_reference_uid == fx.frame_of_reference_uid

    assert dose.path.name == "RD.dcm"
    assert dose.patient_id == fx.patient
    assert dose.sop_instance_uid == fx.dose_sop
    assert dose.study_uid == fx.study
    assert dose.frame_of_reference_uid == fx.frame_of_reference_uid
    assert dose.referenced_plan_sop == fx.plan_sop

    assert struct.path.name == "RS.dcm"
    assert struct.patient_id == fx.patient
    assert struct.sop_instance_uid == fx.struct_sop
    assert struct.study_uid == fx.study
    assert struct.frame_of_reference_uid == fx.frame_of_reference_uid
    assert struct.roi_names == ["PTV"]


# --------------------------------------------------------------------------- 3) _index_series_and_registrations
def test_index_series_and_registrations_parallel_matches_serial(tmp_path):
    fx = _build_fixture(tmp_path)

    series_index1, registrations1, series_meta1 = _index_series_and_registrations(fx.root, max_workers=1)
    series_index8, registrations8, series_meta8 = _index_series_and_registrations(fx.root, max_workers=8)

    assert series_index1 == series_index8
    assert registrations1 == registrations8
    assert series_meta1 == series_meta8

    expected_keys = {
        (fx.patient, fx.ct_series),
        (fx.patient, fx.rp_series),
        (fx.patient, fx.rd_series),
        (fx.patient, fx.rs_series),
        (fx.patient, fx.mr_series),
        (fx.patient, fx.reg_series),
    }
    assert set(series_index1.keys()) == expected_keys

    assert {p.name for p in series_index1[(fx.patient, fx.ct_series)]} == set(fx.ct_instance_numbers.keys())
    assert {p.name for p in series_index1[(fx.patient, fx.rp_series)]} == {"RP.dcm"}
    assert {p.name for p in series_index1[(fx.patient, fx.rd_series)]} == {"RD.dcm"}
    assert {p.name for p in series_index1[(fx.patient, fx.rs_series)]} == {"RS.dcm"}
    assert {p.name for p in series_index1[(fx.patient, fx.mr_series)]} == {"MR_0.dcm"}
    assert {p.name for p in series_index1[(fx.patient, fx.reg_series)]} == {"REG.dcm"}

    assert series_meta1[(fx.patient, fx.ct_series)] == {"modality": "CT"}
    assert series_meta1[(fx.patient, fx.rp_series)] == {"modality": "RTPLAN"}
    assert series_meta1[(fx.patient, fx.rd_series)] == {"modality": "RTDOSE"}
    assert series_meta1[(fx.patient, fx.rs_series)] == {"modality": "RTSTRUCT"}
    assert series_meta1[(fx.patient, fx.mr_series)] == {"modality": "MR"}
    assert series_meta1[(fx.patient, fx.reg_series)] == {"modality": "REG"}

    assert set(registrations1.keys()) == {fx.patient}
    regs = registrations1[fx.patient]
    assert len(regs) == 1
    reg = regs[0]
    assert Path(reg["path"]).name == "REG.dcm"
    assert reg["for_uids"] == {fx.frame_of_reference_uid}
    assert reg["referenced_series"] == {fx.ct_series}
    assert reg["series_by_for"] == {fx.frame_of_reference_uid: {fx.ct_series}}


# --------------------------------------------------------------------------- remediation: stable sort tie-break
def _expected_ct_order_from_walk(root: Path, patient: str, study: str, series: str) -> list[str]:
    """Reconstruct the EXPECTED per-series ordering directly from the raw walk order.

    ``index_ct_series`` builds its candidate ``paths`` list in ``_scoped_walk`` order,
    keeps CT files in that order, then applies a STABLE sort by the key
    ``(instance_number is None, instance_number or 0)``. We replay exactly that here
    against the real on-disk walk, so the expected order is derived from the actual
    filesystem order (not an assumed alphabetical one) -- and both the serial and the
    parallel results must equal it. This is the property most at risk from
    parallelization: if reads completed out of order and results were assembled in
    completion order, duplicate/None instance numbers would reshuffle.
    """
    ct_files: list[tuple[Path, int | None]] = []
    for base, _dirs, files in _scoped_walk(root, [patient]):
        for name in files:
            p = Path(base) / name
            ds = read_dicom(p)
            if ds is None or getattr(ds, "Modality", None) != "CT":
                continue
            if str(getattr(ds, "SeriesInstanceUID", "")) != series:
                continue
            inst = getattr(ds, "InstanceNumber", None)
            ct_files.append((p, int(inst) if inst is not None else None))
    ct_files.sort(key=lambda t: (t[1] is None, t[1] or 0))  # stable, same key as the code
    return [p.name for p, _ in ct_files]


def test_index_ct_series_duplicate_and_none_instance_numbers_stable(tmp_path):
    patient, study, series = "PT1", generate_uid(), generate_uid()
    d = tmp_path / patient / study
    # Duplicate instance numbers (two 3s, two 1s) AND two missing (None) instance numbers.
    # Distinct filenames so ordering differences are observable in the CTInstance.path.
    specs = {
        "a_i3.dcm": 3, "b_iNone.dcm": None, "c_i1.dcm": 1, "d_i3.dcm": 3,
        "e_iNone.dcm": None, "f_i1.dcm": 1, "g_i2.dcm": 2,
    }
    for name, inst in specs.items():
        _mk_ct(d / name, patient=patient, study=study, series=series, instance=inst)

    idx1 = index_ct_series(tmp_path, max_workers=1)
    idx8 = index_ct_series(tmp_path, max_workers=8)

    # 1) parallel byte-identical to serial (the core equivalence claim under stress).
    assert idx1 == idx8

    series1 = idx1[patient][study][series]
    inst_order = [ci.instance_number for ci in series1]
    # 2) sort invariant: real numbers ascending, all None values last.
    numbers = [n for n in inst_order if n is not None]
    assert numbers == sorted(numbers)
    assert inst_order.count(None) == 2
    assert inst_order[-2:] == [None, None]
    # 3) STABILITY vs the actual walk order: both serial and parallel must equal the
    #    stable-sorted-from-walk expectation, including the relative order of the
    #    duplicate-3s / duplicate-1s / None pairs.
    expected_names = _expected_ct_order_from_walk(tmp_path, patient, study, series)
    assert [ci.path.name for ci in series1] == expected_names
    assert [ci.path.name for ci in idx8[patient][study][series]] == expected_names


# --------------------------------------------------------------------------- remediation: non-.dcm files present
def test_non_dcm_files_present_parallel_equals_serial(tmp_path):
    """A junk .txt AND a valid CT DICOM WITHOUT a .dcm extension are present.
    index_ct_series/extract_rt scan ALL files; _index_series_and_registrations filters
    to .dcm. Parallel must handle every case identically to serial, and the semantic
    difference (extension filter vs scan-all) must be preserved under parallelism."""
    fx = _build_fixture(tmp_path)
    d = fx.root / fx.patient / fx.study
    # (a) non-DICOM junk file that read_dicom will parse into a Modality-less dataset.
    (d / "notes.txt").write_text("this is not dicom")
    # (b) a genuine CT slice for the SAME series but named without a .dcm suffix.
    _mk_ct(d / "extra_ct_no_ext", patient=fx.patient, study=fx.study, series=fx.ct_series, instance=99)

    # index_ct_series (scans all files): the no-extension CT IS indexed -> 6 CT slices.
    ct1 = index_ct_series(fx.root, max_workers=1)
    ct8 = index_ct_series(fx.root, max_workers=8)
    assert ct1 == ct8
    ct_series = ct1[fx.patient][fx.study][fx.ct_series]
    assert len(ct_series) == 6
    assert "extra_ct_no_ext" in {ci.path.name for ci in ct_series}
    assert "notes.txt" not in {ci.path.name for ci in ct_series}

    # extract_rt (scans all files): junk/no-ext-CT are not RT -> RT sets unchanged.
    rt1 = extract_rt(fx.root, max_workers=1)
    rt8 = extract_rt(fx.root, max_workers=8)
    assert rt1 == rt8
    assert (len(rt1[0]), len(rt1[1]), len(rt1[2])) == (1, 1, 1)

    # _index_series_and_registrations (.dcm filter): the no-extension CT is EXCLUDED,
    # so the CT series still has exactly its 5 .dcm slices; junk .txt never appears.
    si1, rg1, sm1 = _index_series_and_registrations(fx.root, max_workers=1)
    si8, rg8, sm8 = _index_series_and_registrations(fx.root, max_workers=8)
    assert (si1, rg1, sm1) == (si8, rg8, sm8)
    ct_paths = {p.name for p in si1[(fx.patient, fx.ct_series)]}
    assert ct_paths == set(fx.ct_instance_numbers.keys())  # 5 .dcm only
    assert "extra_ct_no_ext" not in ct_paths and "notes.txt" not in ct_paths


# --------------------------------------------------------------------------- remediation: corrupt/unreadable -> None
def test_unreadable_file_returns_none_and_is_skipped(tmp_path):
    """A genuinely unreadable file makes read_dicom return None. Under the parallel
    path this must be skipped with NO exception, identically to serial (error parity)."""
    fx = _build_fixture(tmp_path)
    d = fx.root / fx.patient / fx.study
    bad = d / "unreadable.dcm"
    bad.write_bytes(b"\x00" * 128 + b"DICM" + b"\x00" * 64)
    os.chmod(bad, 0o000)
    try:
        if read_dicom(bad) is not None:
            # Some environments (root / permissive fs) can still read it; the None
            # branch is what we are proving here, so skip if we cannot induce None.
            pytest.skip("environment can read a 0-perm file; cannot induce read_dicom->None")

        ct1 = index_ct_series(fx.root, max_workers=1)
        ct8 = index_ct_series(fx.root, max_workers=8)
        assert ct1 == ct8
        assert "unreadable.dcm" not in {
            ci.path.name for ci in ct1[fx.patient][fx.study][fx.ct_series]
        }

        rt1 = extract_rt(fx.root, max_workers=1)
        rt8 = extract_rt(fx.root, max_workers=8)
        assert rt1 == rt8
        assert (len(rt1[0]), len(rt1[1]), len(rt1[2])) == (1, 1, 1)

        si1, rg1, sm1 = _index_series_and_registrations(fx.root, max_workers=1)
        si8, rg8, sm8 = _index_series_and_registrations(fx.root, max_workers=8)
        assert (si1, rg1, sm1) == (si8, rg8, sm8)
        # the unreadable .dcm has no readable tags -> never becomes a series key
        assert (fx.patient, "") not in si1
        for key, paths in si1.items():
            assert "unreadable.dcm" not in {p.name for p in paths}
    finally:
        os.chmod(bad, 0o644)  # let tmp cleanup remove it


# --------------------------------------------------------------------------- remediation: env parsing + no-thread serial
def test_default_index_workers_env_parsing(monkeypatch):
    def _with_env(value):
        if value is None:
            monkeypatch.delenv("RTPIPELINE_INDEX_WORKERS", raising=False)
        else:
            monkeypatch.setenv("RTPIPELINE_INDEX_WORKERS", value)
        return _default_index_workers()

    assert _with_env(None) == 32          # unset -> default 32
    assert _with_env("") == 32            # empty -> default 32
    assert _with_env("garbage") == 32     # non-int -> default 32
    assert _with_env("8") == 8            # honoured
    assert _with_env("100") == 64         # clamped to max 64
    assert _with_env("0") == 1            # <=0 -> min 1
    assert _with_env("-5") == 1           # negative -> min 1
    assert _with_env("64") == 64          # boundary
    assert _with_env("1") == 1            # boundary


def test_max_workers_one_spawns_no_threads():
    """max_workers=1 must take the plain-serial branch of parallel_map_files with no
    ThreadPoolExecutor at all -- observe that the active thread count never rises."""
    baseline = threading.active_count()
    peak = [0]

    def observe(x):
        peak[0] = max(peak[0], threading.active_count())
        return x

    out = list(parallel_map_files(range(64), observe, max_workers=1))
    assert out == list(range(64))
    assert peak[0] == baseline, "max_workers=1 must not spawn worker threads"

    # DEFAULT_INDEX_WORKERS is the module-level default surfaced when callers pass None.
    assert 1 <= DEFAULT_INDEX_WORKERS <= 64


def test_parallel_map_files_multi_chunk_boundary_order_preserved():
    """Exercise the MULTI-CHUNK streaming path — the heart of the bounded-memory fix.

    The committed DICOM fixtures are all far smaller than the default chunk_size, so
    without this test the ``while``/``islice`` chunk loop only ever runs once. Here a
    small chunk_size over many items forces many chunk boundaries; results must still
    be yielded in input (submission) order with no drop, duplicate, or reorder at the
    edges, and must equal the plain serial result even when later items complete first.
    """
    items = list(range(50))  # chunk_size=7 -> ~8 chunks

    # Identity across ~8 chunk boundaries: order and contents preserved exactly.
    out = list(parallel_map_files(items, lambda x: x, max_workers=4, chunk_size=7))
    assert out == items

    # Out-of-order completion: earlier items sleep LONGER so later items finish first;
    # ThreadPoolExecutor.map must still surface results in submission order per chunk.
    def slow_early(x):
        time.sleep((5 - (x % 5)) * 0.001)
        return x * x

    got = list(parallel_map_files(items, slow_early, max_workers=8, chunk_size=5))
    assert got == [x * x for x in items]

    # chunk_size larger than the input still yields the full, in-order result.
    assert list(parallel_map_files(items, lambda x: x, max_workers=4, chunk_size=999)) == items
