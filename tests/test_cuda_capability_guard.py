"""A CUDA device that cannot run this build's kernels must not be selected.

``torch.cuda.is_available()`` returns True on a Pascal card even when the
installed PyTorch wheel compiles no Pascal SASS, so segmentation would be
dispatched to GPU and then fail at the first convolution on every course. The
guard compares device capability against the compiled architecture list.

Reproduced on an NVIDIA Quadro P6000 (sm_61) inside the shipped container:
arch list ['sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120'],
torch.cuda.is_available() True, first Conv3d raising
"CUDA error: no kernel image is available for execution on the device".
"""

from __future__ import annotations

import pytest

from rtpipeline.cli import _cuda_capability_is_executable

PASCAL_P6000 = (6, 1)
MODERN_ARCH_LIST = ["sm_70", "sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"]


def test_pascal_is_rejected_by_a_modern_arch_list():
    assert not _cuda_capability_is_executable(PASCAL_P6000, MODERN_ARCH_LIST)


def test_pascal_is_accepted_when_the_build_targets_it():
    assert _cuda_capability_is_executable(PASCAL_P6000, ["sm_60", "sm_61", "sm_70"])


def test_exact_match_is_accepted():
    assert _cuda_capability_is_executable((8, 6), MODERN_ARCH_LIST)


def test_lower_minor_inside_the_same_major_generation_is_accepted():
    # An sm_80 binary runs on an sm_86 device; the reverse must not be assumed.
    assert _cuda_capability_is_executable((8, 6), ["sm_80"])
    assert not _cuda_capability_is_executable((8, 0), ["sm_86"])


def test_embedded_ptx_allows_forward_jit_only():
    assert _cuda_capability_is_executable((9, 0), ["compute_80"])
    assert not _cuda_capability_is_executable((6, 1), ["compute_80"])


def test_a_different_major_generation_is_never_assumed_compatible():
    assert not _cuda_capability_is_executable((6, 1), ["sm_75"])
    assert not _cuda_capability_is_executable((7, 5), ["sm_61"])


@pytest.mark.parametrize("arch_list", [[], ["nonsense"], ["sm_"], ["compute_"]])
def test_unparseable_arch_entries_do_not_raise(arch_list):
    assert _cuda_capability_is_executable(PASCAL_P6000, arch_list) is False


def test_detect_gpu_count_prefers_usable_devices(monkeypatch):
    import rtpipeline.cli as cli

    monkeypatch.setattr(cli, "_usable_cuda_device_count", lambda: 2)
    assert cli._detect_gpu_count() == 2


def test_detect_gpu_count_does_not_claim_an_unusable_device(monkeypatch):
    """A present-but-unrunnable GPU must not be counted from torch alone.

    Without this, a Pascal host reports one CUDA device, segmentation is
    dispatched to GPU, and every course dies at the first kernel launch.
    """
    import rtpipeline.cli as cli

    # A definitive zero from torch must short-circuit the environment and
    # nvidia-smi fallbacks, which count physical cards without knowing whether
    # this build can run on them. On the host that exposed this bug, nvidia-smi
    # reports one Quadro P6000 and would otherwise re-assert a GPU.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(cli, "_usable_cuda_device_count", lambda: 0)
    assert cli._detect_gpu_count() == 0


def test_undeterminable_torch_state_still_allows_the_fallbacks(monkeypatch):
    """None means "could not tell", and must not suppress the other probes."""
    import rtpipeline.cli as cli

    monkeypatch.setattr(cli, "_usable_cuda_device_count", lambda: None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    assert cli._detect_gpu_count() == 2
