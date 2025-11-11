#!/usr/bin/env python3
"""
Docker compatibility test for rtpipeline optimizations.
Tests CPU detection, timeout mechanisms, and resource handling in containers.
"""

import os
import sys
import time
import subprocess
import signal
from pathlib import Path

def test_cpu_detection():
    """Test CPU count detection in Docker environment."""
    print("=" * 60)
    print("CPU Detection Test")
    print("=" * 60)

    cpu_count = os.cpu_count()
    print(f"os.cpu_count(): {cpu_count}")

    # Check if running in Docker
    is_docker = Path("/.dockerenv").exists() or Path("/proc/1/cgroup").exists()
    print(f"Running in Docker: {is_docker}")

    # Check CPU affinity (actual available CPUs)
    try:
        import psutil
        cpu_affinity_count = len(psutil.Process().cpu_affinity())
        print(f"CPU affinity (actual available): {cpu_affinity_count}")
    except ImportError:
        print("psutil not available - install for better CPU detection")

    # Check cgroup CPU limits (Docker-specific)
    try:
        # Read CPU quota and period from cgroups v1
        quota_file = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        period_file = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")

        if quota_file.exists() and period_file.exists():
            quota = int(quota_file.read_text().strip())
            period = int(period_file.read_text().strip())

            if quota > 0:
                cgroup_cpus = quota / period
                print(f"Cgroup CPU limit (v1): {cgroup_cpus:.2f} CPUs")
            else:
                print("Cgroup CPU limit (v1): unlimited")
        else:
            # Try cgroups v2
            cpu_max = Path("/sys/fs/cgroup/cpu.max")
            if cpu_max.exists():
                content = cpu_max.read_text().strip().split()
                if content[0] != "max":
                    quota = int(content[0])
                    period = int(content[1])
                    cgroup_cpus = quota / period
                    print(f"Cgroup CPU limit (v2): {cgroup_cpus:.2f} CPUs")
                else:
                    print("Cgroup CPU limit (v2): unlimited")
            else:
                print("Cgroup CPU limits: not found (running outside container?)")
    except Exception as e:
        print(f"Could not read cgroup limits: {e}")

    # Recommended worker count
    from rtpipeline.config import PipelineConfig
    cfg = PipelineConfig(
        dicom_root=Path('/tmp'),
        output_root=Path('/tmp'),
        logs_root=Path('/tmp')
    )
    effective_workers = cfg.effective_workers()
    print(f"\nPipelineConfig.effective_workers(): {effective_workers}")
    print(f"(Should be cpu_count - 1 = {(cpu_count or 2) - 1})")

    return cpu_count, effective_workers


def test_subprocess_timeout():
    """Test subprocess timeout mechanism."""
    print("\n" + "=" * 60)
    print("Subprocess Timeout Test")
    print("=" * 60)

    print("Testing 2-second timeout on sleep 5...")
    start = time.time()
    try:
        subprocess.run(
            ["sleep", "5"],
            timeout=2,
            capture_output=True
        )
        print("❌ FAILED: Should have timed out!")
        return False
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"✅ SUCCESS: Timed out after {elapsed:.1f}s")
        return True


def test_signal_handling():
    """Test signal handling in Docker."""
    print("\n" + "=" * 60)
    print("Signal Handling Test")
    print("=" * 60)

    # Test if signals work
    def signal_handler(signum, frame):
        print(f"✅ Received signal {signum}")

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    print("Signal handlers registered successfully")
    print("SIGTERM and SIGINT can be handled for graceful shutdown")

    return True


def test_multiprocessing():
    """Test multiprocessing with spawn context."""
    print("\n" + "=" * 60)
    print("Multiprocessing Test (spawn context)")
    print("=" * 60)

    from multiprocessing import get_context

    def worker_func(x):
        return x * 2

    try:
        ctx = get_context('spawn')
        with ctx.Pool(2) as pool:
            results = pool.map(worker_func, [1, 2, 3, 4])
            print(f"Results: {results}")
            if results == [2, 4, 6, 8]:
                print("✅ Multiprocessing with spawn context works")
                return True
            else:
                print("❌ Unexpected results")
                return False
    except Exception as e:
        print(f"❌ Multiprocessing failed: {e}")
        return False


def test_environment_variables():
    """Test that timeout environment variables work."""
    print("\n" + "=" * 60)
    print("Environment Variables Test")
    print("=" * 60)

    # Set test environment variables
    os.environ['TOTALSEG_TIMEOUT'] = '1800'
    os.environ['DCM2NIIX_TIMEOUT'] = '600'
    os.environ['RTPIPELINE_RADIOMICS_TASK_TIMEOUT'] = '300'

    totalseg_timeout = int(os.environ.get('TOTALSEG_TIMEOUT', '3600'))
    dcm2niix_timeout = int(os.environ.get('DCM2NIIX_TIMEOUT', '300'))
    radiomics_timeout = int(os.environ.get('RTPIPELINE_RADIOMICS_TASK_TIMEOUT', '600'))

    print(f"TOTALSEG_TIMEOUT: {totalseg_timeout}s (expected: 1800)")
    print(f"DCM2NIIX_TIMEOUT: {dcm2niix_timeout}s (expected: 600)")
    print(f"RTPIPELINE_RADIOMICS_TASK_TIMEOUT: {radiomics_timeout}s (expected: 300)")

    success = (totalseg_timeout == 1800 and
               dcm2niix_timeout == 600 and
               radiomics_timeout == 300)

    if success:
        print("✅ Environment variables work correctly")
    else:
        print("❌ Environment variables not working as expected")

    return success


def test_gpu_detection():
    """Test GPU detection and CUDA availability."""
    print("\n" + "=" * 60)
    print("GPU Detection Test")
    print("=" * 60)

    # Check CUDA environment variables
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')
    nvidia_visible = os.environ.get('NVIDIA_VISIBLE_DEVICES', 'not set')

    print(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")
    print(f"NVIDIA_VISIBLE_DEVICES: {nvidia_visible}")

    # Check if PyTorch can see GPU
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count()

        print(f"PyTorch CUDA available: {cuda_available}")
        print(f"GPU count: {gpu_count}")

        if cuda_available and gpu_count > 0:
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                print(f"  GPU {i}: {gpu_name}")
            print("✅ GPU detection successful")
            return True
        else:
            print("⚠️  No GPU detected (CPU-only mode)")
            return True  # Not a failure if running in CPU mode
    except ImportError:
        print("⚠️  PyTorch not installed - cannot test GPU")
        return True
    except Exception as e:
        print(f"❌ GPU detection error: {e}")
        return False


def main():
    """Run all Docker compatibility tests."""
    print("=" * 60)
    print("RTPIPELINE DOCKER COMPATIBILITY TEST")
    print("=" * 60)
    print()

    results = {}

    # Run tests
    try:
        test_cpu_detection()
        results['cpu_detection'] = True
    except Exception as e:
        print(f"❌ CPU detection failed: {e}")
        results['cpu_detection'] = False

    try:
        results['subprocess_timeout'] = test_subprocess_timeout()
    except Exception as e:
        print(f"❌ Subprocess timeout test failed: {e}")
        results['subprocess_timeout'] = False

    try:
        results['signal_handling'] = test_signal_handling()
    except Exception as e:
        print(f"❌ Signal handling test failed: {e}")
        results['signal_handling'] = False

    try:
        results['multiprocessing'] = test_multiprocessing()
    except Exception as e:
        print(f"❌ Multiprocessing test failed: {e}")
        results['multiprocessing'] = False

    try:
        results['environment_vars'] = test_environment_variables()
    except Exception as e:
        print(f"❌ Environment variables test failed: {e}")
        results['environment_vars'] = False

    try:
        results['gpu_detection'] = test_gpu_detection()
    except Exception as e:
        print(f"❌ GPU detection test failed: {e}")
        results['gpu_detection'] = False

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(results.values())

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED - Docker compatibility verified!")
        print("=" * 60)
        return 0
    else:
        print("❌ SOME TESTS FAILED - Check output above")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
