from __future__ import annotations

import docker_test


def test_docker_spawn_context_smoke():
    assert docker_test.test_multiprocessing() is True
