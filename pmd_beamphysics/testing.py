import pytest
import numpy as np

from .particles import ParticleGroup


@pytest.fixture
def test_beam() -> ParticleGroup:
    """Create a test beam with random particles"""
    # Create a simple beam with position and momentum spread
    n_particles = 1000
    np.random.seed(42)  # For reproducibility

    # Initialize with data dictionary
    data = {
        "x": np.random.normal(0, 1e-3, n_particles),  # m
        "y": np.random.normal(0, 2e-3, n_particles),  # m
        "z": np.random.normal(0, 3e-3, n_particles),  # m
        "px": np.random.normal(0, 4e-3, n_particles) * 1e6,
        "py": np.random.normal(0, 5e-3, n_particles) * 1e6,  # eV/c
        "pz": np.random.normal(10, 6, n_particles) * 1e6,  # eV/c, main momentum
        "t": np.zeros(n_particles),  # s
        "status": np.ones(n_particles),
        "weight": np.ones(n_particles) * 1e-12,  # C, equal weights
        "species": "electron",
    }

    return ParticleGroup(data=data)
