import pytest
import ray


@pytest.fixture(scope="module")
def ray_cluster():
    """Lightweight Ray setup for actor-based execute() calls used in tests."""
    ray.init(ignore_reinit_error=True, include_dashboard=False, num_cpus=2)
    try:
        yield
    finally:
        ray.shutdown()
