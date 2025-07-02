import pytest
import numpy as np

from .particles import ParticleGroup


@pytest.fixture
def test_beam() -> ParticleGroup:
    """Create a test beam with random particles"""
    # Create a simple beam with position and momentum spread
    n_particle = 1000
    np.random.seed(42)  # For reproducibility
    return ParticleGroup.from_random_normal(n_particle=n_particle)
