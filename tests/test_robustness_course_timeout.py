"""Robustness course deadline scales with the course's task waves."""
from rtpipeline.radiomics_robustness import _robustness_course_timeout


def test_small_course_keeps_historical_floor(monkeypatch):
    monkeypatch.delenv("RTPIPELINE_ROBUSTNESS_COURSE_TIMEOUT", raising=False)
    monkeypatch.delenv("RTPIPELINE_ROBUSTNESS_TASK_BUDGET", raising=False)
    assert _robustness_course_timeout(81, 9) == 14400
    assert _robustness_course_timeout(0, 9) == 14400


def test_large_course_scales_with_waves(monkeypatch):
    monkeypatch.delenv("RTPIPELINE_ROBUSTNESS_COURSE_TIMEOUT", raising=False)
    monkeypatch.delenv("RTPIPELINE_ROBUSTNESS_TASK_BUDGET", raising=False)
    # 1,215 tasks on 9 workers = 135 waves x 900 s
    assert _robustness_course_timeout(1215, 9) == 135 * 900
    assert _robustness_course_timeout(1216, 9) == 136 * 900


def test_explicit_timeout_and_wave_budget_override(monkeypatch):
    monkeypatch.setenv("RTPIPELINE_ROBUSTNESS_COURSE_TIMEOUT", "86400")
    assert _robustness_course_timeout(10000, 1) == 86400
    monkeypatch.delenv("RTPIPELINE_ROBUSTNESS_COURSE_TIMEOUT")
    monkeypatch.setenv("RTPIPELINE_ROBUSTNESS_TASK_BUDGET", "1200")
    assert _robustness_course_timeout(1215, 9) == 135 * 1200
    assert _robustness_course_timeout(9, 0) == 14400
