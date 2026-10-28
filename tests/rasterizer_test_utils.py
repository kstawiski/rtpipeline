"""Declared runtime identities for synthetic publication fixtures only."""
PINNED_RASTERIZER_ROW = {
    "rasterizer_backend": "rt-utils/cv2.fillPoly",
    "rasterizer_opencv_version": "4.8.1",
    "rasterizer_rt_utils_version": "1.2.7",
    "rasterizer_providers": '{"opencv-python-headless": "4.8.1.78"}',
    "rasterizer_opencv_path": "/synthetic-radiomics/cv2/__init__.py",
    "rasterizer_probe_error": "",
}


def stub_synthetic_rasterizer_columns(monkeypatch):
    from rtpipeline import radiomics_ct_contract
    monkeypatch.setattr(radiomics_ct_contract, "_rasterizer_columns", lambda backend=None: {
        **PINNED_RASTERIZER_ROW,
        "rasterizer_backend": backend or "rt-utils/cv2.fillPoly",
    })


def stub_course_runtime(monkeypatch):
    """Keep synthetic extractor tests in-process; dispatch has separate tests."""
    from rtpipeline import radiomics_runtime
    monkeypatch.setattr(radiomics_runtime, "needs_radiomics_process", lambda: False)
