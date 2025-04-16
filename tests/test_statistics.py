import pmd_beamphysics.statistics as stats
import numpy as np


def test_twiss_roundtrip():
    """
    Test the roundtrip conversion: sigma -> Twiss -> sigma.

    Ensures numerical consistency of the transformations.

    Raises
    ------
    AssertionError
        If the original and reconstructed sigma matrices are not close within tolerance.
    """
    emit = 3.0
    alpha = 1.2
    beta = 2.5

    sigma_original = np.array(
        [[beta * emit, -alpha * emit], [-alpha * emit, ((1 + alpha**2) / beta) * emit]]
    )

    # Test with Twiss object
    twiss = stats.twiss_from_sigma(sigma_original)
    sigma_reconstructed = stats.sigma_from_twiss(twiss)

    np.testing.assert_allclose(
        sigma_original, sigma_reconstructed, rtol=1e-6, atol=1e-12
    )

    # Test with explicit parameters
    sigma_reconstructed_explicit = stats.sigma_from_twiss(
        alpha=alpha, beta=beta, emit=emit
    )

    np.testing.assert_allclose(
        sigma_original, sigma_reconstructed_explicit, rtol=1e-6, atol=1e-12
    )
