"""
Tests for the parametric cavity wake of Zagorodnov et al. (2021) and the
bin-averaged tabulation that a wake singular at the source particle requires.
"""

import pathlib

import numpy as np
import pytest
from scipy.integrate import quad

from beamphysics import ParticleGroup
from beamphysics.interfaces.impact import (
    create_impact_t_wakefield_data,
    create_impact_z_wakefield_rfdata,
)
from beamphysics.wakefields import (
    ZAGORODNOV_2021_TABLE_IV,
    ParametricWakefield,
    Pseudomode,
    PseudomodeWakefield,
    ResistiveWallWakefield,
    TabularWakefield,
)

STRUCTURES = ["A1", "AH1", "TDS"]

# ECHO curves of Fig. 7 of the paper, digitized by scripts/digitize_zagorodnov_fig7.py.
FIG7_DATA = (
    pathlib.Path(__file__).resolve().parents[1]
    / "docs/examples/data/zagorodnov_2021_fig7_echo.csv"
)

# Lengths over which the module wakes are spread (m). The TDS length is from
# Table 1 of the paper; the cryomodule lengths are nominal.
LENGTHS = {"A1": 12.2, "AH1": 10.5, "TDS": 0.7}


@pytest.fixture(scope="module", params=STRUCTURES)
def published(request) -> ParametricWakefield:
    """One of the three published coefficient sets."""
    return ParametricWakefield(
        length=LENGTHS[request.param], **ZAGORODNOV_2021_TABLE_IV[request.param]
    )


@pytest.fixture(scope="module")
def ah1() -> ParametricWakefield:
    """The 3.9 GHz module, the only set with a delta-function term."""
    return ParametricWakefield(length=LENGTHS["AH1"], **ZAGORODNOV_2021_TABLE_IV["AH1"])


def gaussian(z: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * (z / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)


# -----------------------------------------------------------------------------
# Construction and units
# -----------------------------------------------------------------------------


def test_from_mm_pC_reproduces_the_formula_in_mm_units():
    """The SI wake at s must equal Eq. 18 evaluated with s in mm and w in V/pC."""
    c = dict(A=711, k0=1.94, B=47.4, k1=106, alpha=0.661, k2=6.36, C=49.2, k3=0.154)
    wake = ParametricWakefield.from_mm_pC(length=1.0, **c)

    s_mm = 0.37
    expected_V_per_pC = (
        c["A"] * np.exp(-np.sqrt(c["k0"] * s_mm))
        + c["B"]
        * np.cos((c["k1"] * s_mm) ** c["alpha"])
        / (np.sqrt(s_mm) + c["k2"] * s_mm)
        + c["C"] * np.cos(c["k3"] * s_mm)
    )

    assert wake.wake(-s_mm * 1e-3) == pytest.approx(expected_V_per_pC * 1e12)


def test_from_mm_pC_converts_each_coefficient():
    wake = ParametricWakefield.from_mm_pC(
        length=2.0, A=1, k0=1, B=1, k1=1, alpha=0.5, k2=1, C=1, k3=1, D=1
    )

    assert wake.A == 1e12
    assert wake.k0 == 1e3
    assert wake.B == pytest.approx(1e12 / np.sqrt(1e3))
    assert wake.k1 == 1e3
    assert wake.k2 == pytest.approx(np.sqrt(1e3))
    assert wake.C == 1e12
    assert wake.k3 == 1e3
    assert wake.D == pytest.approx(1e9)
    assert wake.length == 2.0


def test_length_scales_the_wake_per_metre():
    tds = ZAGORODNOV_2021_TABLE_IV["TDS"]
    one = ParametricWakefield(length=1.0, **tds)
    two = ParametricWakefield(length=2.0, **tds)
    z = -np.array([1e-5, 1e-4, 1e-3])

    np.testing.assert_allclose(two.wake(z), one.wake(z) / 2)
    np.testing.assert_allclose(two.antiderivative(z), one.antiderivative(z) / 2)


def test_table_iv_reproduces_the_digitized_fig7():
    """
    The AH1 wake potentials must match the ECHO curves of Fig. 7 to the
    resolution of the figure, about 6 V/pC in W for a peak of 1120 V/pC.
    """
    data = np.loadtxt(FIG7_DATA, delimiter=",")
    wake = ParametricWakefield(length=1.0, **ZAGORODNOV_2021_TABLE_IV["AH1"])

    for sigma_um in np.unique(data[:, 0]):
        rows = data[data[:, 0] == sigma_um]
        sigma = sigma_um * 1e-6
        V = wake.wake_potential(-rows[:, 1] * sigma, sigma) * 1e-12
        residual = V - rows[:, 2]

        assert np.sqrt(np.mean(residual**2)) < 10
        assert np.abs(residual).max() < 30


@pytest.mark.parametrize(
    "kwargs",
    [dict(length=0), dict(length=1, k0=-1), dict(length=1, alpha=0)],
)
def test_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        ParametricWakefield(**kwargs)


def test_repr_lists_the_coefficients(ah1):
    s = repr(ah1)
    assert s.startswith("ParametricWakefield(A=7.11e+14")
    assert "length=10.5" in s


# -----------------------------------------------------------------------------
# Wake
# -----------------------------------------------------------------------------


def test_wake_is_causal_and_singular_at_the_source(published):
    z = np.array([1e-3, 1e-9, 0.0, -1e-9, -1e-3])
    W = published.wake(z)

    assert W[0] == 0 and W[1] == 0
    assert np.isinf(W[2])
    assert np.all(np.isfinite(W[3:]))
    assert published.is_singular


def test_wake_without_the_diffraction_term_is_finite_at_the_source():
    wake = ParametricWakefield(length=2.0, A=3.0, k0=10.0, C=1.0, k3=5.0)

    assert not wake.is_singular
    assert wake.wake(0.0) == pytest.approx((3.0 + 1.0) / 2.0)
    assert wake.wake(-1e-12) == pytest.approx((3.0 + 1.0) / 2.0, rel=1e-5)


def test_wake_scalar_and_array_agree(published):
    z = -2.5e-4
    assert published.wake(z) == published.wake(np.array([z]))[0]
    assert isinstance(published.wake(z), float)


# -----------------------------------------------------------------------------
# Antiderivative
# -----------------------------------------------------------------------------

S_VALUES = [1e-7, 1e-5, 1e-4, 1e-3, 1e-2, 3e-2]


def test_antiderivative_vanishes_ahead_of_and_at_the_source(ah1):
    np.testing.assert_array_equal(ah1.antiderivative(np.array([0.0, 1e-9, 1.0])), 0)
    assert ah1.antiderivative(0.0) == 0.0


def test_antiderivative_of_the_A_term():
    A, k0, L = 2.0, 300.0, 1.5
    wake = ParametricWakefield(length=L, A=A, k0=k0)

    for s in S_VALUES:
        # Integrate in u = sqrt(t), where the integrand is smooth.
        ref, _ = quad(lambda u: 2 * u * A * np.exp(-np.sqrt(k0) * u), 0, np.sqrt(s))
        assert wake.antiderivative(-s) == pytest.approx(ref / L, rel=1e-10)


def test_antiderivative_of_the_C_term():
    C, k3, L = 5.0, 200.0, 2.0
    wake = ParametricWakefield(length=L, C=C, k3=k3)

    for s in S_VALUES:
        ref, _ = quad(lambda t: C * np.cos(k3 * t), 0, s)
        assert wake.antiderivative(-s) == pytest.approx(ref / L, rel=1e-10)


def test_antiderivative_of_the_delta_term_is_a_step():
    wake = ParametricWakefield(length=4.0, D=2.0)

    assert wake.antiderivative(-1e-9) == pytest.approx(0.5)
    assert wake.antiderivative(-1.0) == pytest.approx(0.5)
    assert wake.antiderivative(0.0) == 0.0


def test_antiderivative_of_the_diffraction_term_without_oscillation():
    """With k1 = 0 the B term integrates to 2B/k2 ln(1 + k2 sqrt(s))."""
    B, k2, L = 3.0, 40.0, 1.0
    wake = ParametricWakefield(length=L, B=B, k2=k2)

    for s in S_VALUES:
        expected = 2 * B / k2 * np.log1p(k2 * np.sqrt(s))
        assert wake.antiderivative(-s) == pytest.approx(expected, rel=1e-12)

    # Pure diffraction model.
    pure = ParametricWakefield(length=L, B=B)
    for s in S_VALUES:
        assert pure.antiderivative(-s) == pytest.approx(2 * B * np.sqrt(s), rel=1e-12)


def test_antiderivative_of_the_general_diffraction_term_by_substitution(published):
    """Check against quadrature of the bounded integrand after s = u^2."""
    B, k1, alpha, k2, L = (
        published.B,
        published.k1,
        published.alpha,
        published.k2,
        published.length,
    )
    wake = ParametricWakefield(length=L, B=B, k1=k1, alpha=alpha, k2=k2)

    for s in S_VALUES:
        ref, err = quad(
            lambda u: 2 * B * np.cos((k1 * u**2) ** alpha) / (1 + k2 * u),
            0,
            np.sqrt(s),
            limit=500,
        )
        assert wake.antiderivative(-s) == pytest.approx(ref / L, rel=1e-9, abs=err)


def test_antiderivative_of_the_general_diffraction_term_by_the_paper_split(
    published,
):
    """
    Check against the split of Eqs. 24 and 25: the closed form with the cosine
    replaced by 1, plus the integral of B [cos((k1 s)^alpha) - 1] / (sqrt(s) + k2 s),
    which is regular at s = 0 for alpha > 1/4.
    """
    B, k1, alpha, k2, L = (
        published.B,
        published.k1,
        published.alpha,
        published.k2,
        published.length,
    )
    assert alpha > 0.25
    wake = ParametricWakefield(length=L, B=B, k1=k1, alpha=alpha, k2=k2)

    for s in S_VALUES:
        smooth = 2 * B / k2 * np.log1p(k2 * np.sqrt(s))
        rest, err = quad(
            lambda t: B * (np.cos((k1 * t) ** alpha) - 1) / (np.sqrt(t) + k2 * t),
            0,
            s,
            limit=500,
        )
        assert wake.antiderivative(-s) == pytest.approx(
            (smooth + rest) / L, rel=1e-7, abs=err
        )


def test_antiderivative_sums_the_terms(published):
    """The full antiderivative is the sum over the four terms."""
    s = np.array(S_VALUES)
    z = -s
    L = published.length
    parts = [
        ParametricWakefield(length=L, A=published.A, k0=published.k0),
        ParametricWakefield(
            length=L,
            B=published.B,
            k1=published.k1,
            alpha=published.alpha,
            k2=published.k2,
        ),
        ParametricWakefield(length=L, C=published.C, k3=published.k3),
        ParametricWakefield(length=L, D=published.D),
    ]
    total = sum(part.antiderivative(z) for part in parts)

    np.testing.assert_allclose(published.antiderivative(z), total, rtol=1e-12)


def test_antiderivative_derivative_recovers_the_wake(published):
    """Away from the source, dF/ds must equal the wake."""
    s = np.geomspace(1e-5, 1e-2, 50)
    h = 1e-7 * s
    dF = (published.antiderivative(-(s + h)) - published.antiderivative(-(s - h))) / (
        2 * h
    )

    np.testing.assert_allclose(dF, published.wake(-s), rtol=1e-5)


# -----------------------------------------------------------------------------
# Bin averages
# -----------------------------------------------------------------------------


def test_bin_averaged_wake_follows_the_table_convention(ah1):
    dz, n = 2e-5, 50
    K = ah1.bin_averaged_wake(dz, n)
    k = np.arange(n)

    assert K[0] == pytest.approx(2 * ah1.antiderivative(-dz / 2) / dz)
    upper = ah1.antiderivative(-(k[1:] + 0.5) * dz)
    lower = ah1.antiderivative(-(k[1:] - 0.5) * dz)
    np.testing.assert_allclose(K[1:], (upper - lower) / dz)


def test_bin_averaged_wake_telescopes_to_the_antiderivative(ah1):
    """Halving the first row and summing the bins recovers F at the last edge."""
    dz, n = 1e-5, 200
    K = ah1.bin_averaged_wake(dz, n)
    K[0] /= 2

    assert K.sum() * dz == pytest.approx(ah1.antiderivative(-(n - 0.5) * dz))


def test_bin_averaged_wake_includes_the_delta_term_in_the_first_row():
    D, L, dz = 3.0, 2.0, 1e-4
    wake = ParametricWakefield(length=L, D=D)
    K = wake.bin_averaged_wake(dz, 5)

    assert K[0] == pytest.approx(2 * D / L / dz)
    np.testing.assert_array_equal(K[1:], 0)


def test_bin_averaged_wake_requires_an_antiderivative():
    wake = ResistiveWallWakefield.from_material("copper-slac-pub-10707", radius=2.5e-3)
    with pytest.raises(NotImplementedError, match="antiderivative"):
        wake.bin_averaged_wake(1e-6, 10)


def test_pseudomode_antiderivative_matches_quadrature():
    modes = [
        Pseudomode(A=1e15, d=1e4, k=1e5, phi=np.pi / 3),
        Pseudomode(A=-4e14, d=3e4, k=0.0, phi=np.pi / 2),
        Pseudomode(A=2e14, d=0.0, k=0.0, phi=1.0),
    ]
    wake = PseudomodeWakefield(modes)

    for s in [1e-6, 1e-5, 1e-4, 5e-4]:
        ref, _ = quad(lambda z: wake.wake(z), -s, 0, limit=200)
        assert wake.antiderivative(-s) == pytest.approx(ref, rel=1e-9)

    assert wake.antiderivative(0.0) == 0.0
    assert wake.antiderivative(1e-6) == 0.0


def test_pseudomode_bin_averages_approach_the_samples():
    """Interior bins agree to second order in dz; the self term to first order."""
    wake = PseudomodeWakefield([Pseudomode(A=1e15, d=1e4, k=1e5, phi=np.pi / 2)])

    def errors(dz):
        n = 40
        K = wake.bin_averaged_wake(dz, n)
        W = wake.wake(-np.arange(n) * dz)
        return np.abs(K[1:] - W[1:]).max(), abs(K[0] - W[0])

    interior1, self1 = errors(2e-6)
    interior2, self2 = errors(1e-6)
    assert interior2 < interior1 / 3.5
    assert self2 < self1 / 1.8


# -----------------------------------------------------------------------------
# Gaussian wake potential
# -----------------------------------------------------------------------------


def test_wake_potential_matches_quadrature(published):
    sigma = 50e-6
    L = published.length

    for z in [-3 * sigma, -0.5 * sigma, 0.0, 1.5 * sigma]:

        def integrand(u):
            t = u**2
            return (
                2
                * u
                * published._wake_regular(np.array([t]))[0]
                * gaussian(z + t, sigma)
            )

        ref, err = quad(integrand, 0, np.sqrt(8 * sigma - z), limit=1000)
        ref = (ref + published.D * gaussian(z, sigma)) / L

        assert published.wake_potential(z, sigma) == pytest.approx(
            ref, rel=1e-8, abs=err
        )


def test_wake_potential_vanishes_far_ahead_of_the_bunch(ah1):
    sigma = 100e-6
    V0 = ah1.wake_potential(0.0, sigma)
    assert abs(ah1.wake_potential(12 * sigma, sigma)) < 1e-12 * V0
    assert abs(ah1.wake_potential(7 * sigma, sigma)) < 1e-8 * V0


def test_wake_potential_carries_the_delta_term():
    D, L, sigma = 3.0, 2.0, 1e-4
    wake = ParametricWakefield(length=L, D=D)
    z = np.linspace(-3 * sigma, 3 * sigma, 7)

    np.testing.assert_allclose(
        wake.wake_potential(z, sigma), D / L * gaussian(z, sigma)
    )


def test_wake_potential_rejects_a_nonpositive_sigma(ah1):
    with pytest.raises(ValueError, match="sigma"):
        ah1.wake_potential(0.0, 0.0)


def test_loss_factor_and_spread_are_moments_of_the_wake_potential(ah1):
    sigma = 200e-6
    z = np.linspace(-8 * sigma, 8 * sigma, 4001)
    lam = gaussian(z, sigma)
    V = ah1.wake_potential(z, sigma)
    kappa = np.trapezoid(V * lam, z)
    spread = np.sqrt(np.trapezoid((V - kappa) ** 2 * lam, z))

    assert ah1.loss_factor(sigma) == pytest.approx(kappa, rel=1e-6)
    assert ah1.rms_energy_spread(sigma) == pytest.approx(spread, rel=1e-6)
    assert kappa > 0


def test_loss_factor_decreases_with_bunch_length(published):
    kappas = [published.loss_factor(s) for s in (10e-6, 100e-6, 1e-3)]
    assert kappas[0] > kappas[1] > kappas[2] > 0


# -----------------------------------------------------------------------------
# Bin-averaged tabulation and convolution
# -----------------------------------------------------------------------------


def test_from_wakefield_bin_averages_a_singular_wake(ah1):
    zmax, n = 2e-3, 401
    table = TabularWakefield.from_wakefield(ah1, zmax=zmax, n=n)

    dz = zmax / (n - 1)
    np.testing.assert_allclose(table.W_data[::-1], ah1.bin_averaged_wake(dz, n))
    assert np.all(np.isfinite(table.W_data))
    # Linear interpolation, so the midpoint is the mean of its neighbours.
    z = table.z_data
    mid = (z[10] + z[11]) / 2
    assert table.wake(mid) == pytest.approx((table.W_data[10] + table.W_data[11]) / 2)


def test_from_wakefield_rejects_point_samples_of_a_singular_wake(ah1):
    with pytest.raises(ValueError, match="singular"):
        TabularWakefield.from_wakefield(ah1, zmax=1e-3, n=11, method="sample")


def test_from_wakefield_rejects_an_unknown_method(ah1):
    with pytest.raises(ValueError, match="method"):
        TabularWakefield.from_wakefield(ah1, zmax=1e-3, n=11, method="midpoint")


def test_from_wakefield_bin_average_needs_an_antiderivative():
    wake = ResistiveWallWakefield.from_material("copper-slac-pub-10707", radius=2.5e-3)
    with pytest.raises(NotImplementedError, match="antiderivative"):
        TabularWakefield.from_wakefield(wake, zmax=1e-4, n=11, method="bin_average")


def test_from_wakefield_bin_average_of_a_smooth_wake_is_close_to_the_samples():
    wake = PseudomodeWakefield([Pseudomode(A=1e15, d=1e4, k=1e5, phi=np.pi / 2)])
    sampled = TabularWakefield.from_wakefield(wake, zmax=1e-4, n=1001)
    averaged = TabularWakefield.from_wakefield(
        wake, zmax=1e-4, n=1001, method="bin_average"
    )

    # Interior rows agree to second order. The first row holds twice the
    # self-kick average, which is W(0) to first order.
    np.testing.assert_allclose(
        averaged.W_data[:-1], sampled.W_data[:-1], rtol=1e-4, atol=1e-4 * 1e15
    )
    assert averaged.W_data[-1] == pytest.approx(sampled.W_data[-1], rel=1e-3)


@pytest.mark.parametrize("sigma", [20e-6, 200e-6, 2e-3])
def test_tabulated_convolution_converges_to_the_wake_potential(published, sigma):
    """
    A bin-averaged table convolved with a Gaussian bunch must approach the exact
    wake potential as the bin width shrinks, for each published coefficient set.
    """
    Q = 250e-12
    z_ref = np.linspace(-6 * sigma, 6 * sigma, 25)
    V_ref = published.wake_potential(z_ref, sigma)
    scale = np.abs(V_ref).max()

    def error(bins_per_sigma):
        dz = sigma / bins_per_sigma
        half = int(np.ceil(8 * sigma / dz))
        z = dz * np.arange(-half, half + 1)
        density = Q * gaussian(z, sigma)
        table = TabularWakefield.from_wakefield(
            published, zmax=2 * half * dz, n=2 * half + 1
        )
        V = table.convolve_density(density, dz) / Q
        return np.abs(np.interp(z_ref, z, V) - V_ref).max() / scale

    coarse, fine = error(5), error(20)
    assert fine < coarse
    assert fine < 2e-3


def test_convolve_density_uses_bin_averages_for_a_singular_wake(ah1):
    """The model and its table at the same spacing give the same potential."""
    sigma, dz = 100e-6, 5e-6
    half = int(8 * sigma / dz)
    z = dz * np.arange(-half, half + 1)
    density = 1e-9 * gaussian(z, sigma)

    direct = ah1.convolve_density(density, dz)
    table = TabularWakefield.from_wakefield(ah1, zmax=2 * half * dz, n=2 * half + 1)
    via_table = table.convolve_density(density, dz)

    np.testing.assert_allclose(
        direct, via_table, rtol=1e-10, atol=1e-10 * np.abs(direct).max()
    )
    assert np.all(np.isfinite(direct))


def test_convolve_density_rejects_point_samples_of_a_singular_wake(ah1):
    with pytest.raises(ValueError, match="singular"):
        ah1.convolve_density(np.ones(8), 1e-6, method="sample")


def test_convolve_density_rejects_bin_averages_with_an_offset(ah1):
    with pytest.raises(ValueError, match="offset"):
        ah1.convolve_density(np.ones(8), 1e-6, offset=1e-7, method="bin_average")


def test_convolve_density_delta_term_gives_D_times_the_density():
    """With the delta inside the first bin, the self term is D lambda(z) / L."""
    D, L, dz = 3.0, 2.0, 1e-5
    wake = ParametricWakefield(length=L, D=D)
    density = np.array([0.0, 1.0, 2.0, 5.0, 2.0, 1.0, 0.0]) * 1e-6

    V = wake.convolve_density(density, dz)

    np.testing.assert_allclose(V, D / L * density)


def test_convolve_density_bin_average_matches_samples_for_a_smooth_wake():
    wake = PseudomodeWakefield([Pseudomode(A=1e15, d=1e4, k=1e5, phi=np.pi / 2)])
    dz = 1e-7
    z = dz * np.arange(-400, 401)
    density = 1e-9 * gaussian(z, 10e-6)

    averaged = wake.convolve_density(density, dz)
    sampled = wake.convolve_density(density, dz, method="sample")

    np.testing.assert_allclose(
        averaged, sampled, rtol=0, atol=1e-5 * np.abs(sampled).max()
    )


# -----------------------------------------------------------------------------
# First-row convention through the code interfaces
# -----------------------------------------------------------------------------


def _bmad_z_long_rows(s: str) -> np.ndarray:
    body = s.split("w = {", 1)[1].rsplit("}}}", 1)[0]
    rows = [line.rstrip(",").split() for line in body.splitlines() if line.strip()]
    return np.array(rows, dtype=float)


def test_bmad_round_trip_keeps_the_doubled_first_row(ah1):
    table = TabularWakefield.from_wakefield(ah1, zmax=1e-3, n=101)
    dz = 1e-3 / 100

    rows = _bmad_z_long_rows(table.to_bmad())
    assert rows[-1, 0] == 0.0
    assert rows[-1, 1] == pytest.approx(2 * ah1.antiderivative(-dz / 2) / dz)

    back = TabularWakefield.from_bmad(rows)
    np.testing.assert_allclose(back.W_data, table.W_data)
    np.testing.assert_allclose(back.z_data, table.z_data, atol=1e-18)


def test_impact_tables_keep_the_doubled_first_row(ah1):
    zmax, n = 1e-3, 101
    dz = zmax / (n - 1)
    K = ah1.bin_averaged_wake(dz, n)

    data, length = create_impact_t_wakefield_data(ah1, zmax=zmax, n=n)
    assert length == pytest.approx(zmax)
    np.testing.assert_allclose(data[:, 0], K)

    rfdata = create_impact_z_wakefield_rfdata(ah1, zmax=zmax, n=n)
    np.testing.assert_allclose(rfdata[:, 1], K)
    np.testing.assert_allclose(rfdata[:, 0], dz * np.arange(n))

    # The table methods write the same rows.
    table = TabularWakefield.from_wakefield(ah1, zmax=zmax, n=n)
    np.testing.assert_allclose(table.to_impact_t(n=n)[0][:, 0], K)


# -----------------------------------------------------------------------------
# Rebin
# -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fine_table(ah1) -> TabularWakefield:
    """Bin-averaged table at 1 µm spacing over 2 mm."""
    return TabularWakefield.from_wakefield(ah1, zmax=2e-3, n=2001)


def _integral(table: TabularWakefield) -> float:
    """Integral of the bin averages, with the first row halved."""
    W = table.W_data[::-1].copy()
    W[0] /= 2
    dz = -table.z_data[-2]
    return W.sum() * dz


def test_rebin_is_exact_for_an_odd_multiple(ah1, fine_table):
    dz = 3e-6
    rebinned = fine_table.rebin(dz)
    n = len(rebinned.z_data)

    # The last bin reaches beyond the source table, so it is partial.
    np.testing.assert_allclose(
        rebinned.W_data[::-1][:-1], ah1.bin_averaged_wake(dz, n)[:-1], rtol=1e-9
    )


def test_rebin_conserves_the_integral(fine_table):
    for dz in [2e-6, 3e-6, 7.3e-6]:
        rebinned = fine_table.rebin(dz)
        assert _integral(rebinned) == pytest.approx(_integral(fine_table), rel=1e-3)


def test_rebin_passes_a_table_through_at_its_own_spacing(fine_table):
    same = fine_table.rebin(1e-6)
    np.testing.assert_array_equal(same.W_data, fine_table.W_data)
    np.testing.assert_allclose(same.z_data, fine_table.z_data, rtol=0, atol=1e-18)


def test_rebin_is_close_for_a_non_multiple(ah1, fine_table):
    """A non-multiple spacing is accurate well below 0.1% in the bunch moments."""
    dz = 7.3e-6
    rebinned = fine_table.rebin(dz)
    exact = TabularWakefield.from_wakefield(
        ah1, zmax=rebinned.default_zmax, n=len(rebinned.z_data)
    )

    sigma = 100e-6
    half = int(8 * sigma / dz)
    z = dz * np.arange(-half, half + 1)
    density = gaussian(z, sigma)
    V_rebinned = rebinned.convolve_density(density, dz)
    V_exact = exact.convolve_density(density, dz)
    lam = density * dz
    loss_rebinned = np.sum(V_rebinned * lam)
    loss_exact = np.sum(V_exact * lam)

    assert loss_rebinned == pytest.approx(loss_exact, rel=1e-3)


def test_rebin_extends_the_range_with_zeros(fine_table):
    rebinned = fine_table.rebin(1e-6, zmax=3e-3)

    assert rebinned.default_zmax == pytest.approx(3e-3)
    assert np.all(rebinned.W_data[: 1000 - 1] == 0)


def test_rebin_limits_the_rows_with_a_warning(fine_table):
    with pytest.warns(UserWarning, match="limited to 500 rows"):
        short = fine_table.rebin(1e-6, zmax=2e-3, max_rows=500)

    assert len(short.z_data) == 500


def test_rebin_rejects_a_nonuniform_table_and_a_bad_spacing(fine_table):
    z = np.r_[-np.geomspace(1e-4, 1e-7, 50), 0.0]
    nonuniform = TabularWakefield(z, np.exp(z / 1e-5))
    with pytest.raises(ValueError, match="uniform"):
        nonuniform.rebin(1e-6)
    with pytest.raises(ValueError, match="dz"):
        fine_table.rebin(0.0)


# -----------------------------------------------------------------------------
# Particle kicks
# -----------------------------------------------------------------------------


def test_particle_kicks_refuses_a_singular_wake(ah1):
    with pytest.raises(ValueError, match="binned_particle_kicks"):
        ah1.particle_kicks(np.array([0.0, -1e-6]), np.array([1e-12, 1e-12]))


def test_binned_particle_kicks_agree_with_the_exact_pseudomode_sum():
    rng = np.random.default_rng(1)
    sigma, n = 50e-6, 20000
    z = rng.normal(0, sigma, n)
    weight = np.full(n, 250e-12 / n)
    wake = PseudomodeWakefield([Pseudomode(A=1e14, d=3e3, k=2e4, phi=np.pi / 2)])

    exact = wake.particle_kicks(z, weight)
    binned = wake.binned_particle_kicks(z, weight, dz=sigma / 20)

    scale = np.abs(exact).max()
    assert np.abs(binned - exact).max() < 0.02 * scale


def test_binned_particle_kicks_follow_the_wake_potential(ah1):
    rng = np.random.default_rng(2)
    sigma, n, Q = 100e-6, 50000, 250e-12
    z = rng.normal(0, sigma, n)
    weight = np.full(n, Q / n)

    kicks = ah1.binned_particle_kicks(z, weight, dz=sigma / 10)
    V = Q * ah1.wake_potential(z, sigma)

    core = np.abs(z) < 2 * sigma
    assert np.abs(kicks[core] + V[core]).max() < 0.05 * np.abs(V).max()


def test_binned_particle_kicks_default_spacing(ah1):
    z = np.linspace(-1e-4, 1e-4, 500)
    weight = np.full(500, 1e-15)
    kicks = ah1.binned_particle_kicks(z, weight)

    assert kicks.shape == z.shape
    assert np.all(np.isfinite(kicks))
    assert kicks[0] < 0  # the tail loses energy


def test_apply_wakefield_uses_the_binned_path_for_a_singular_wake(ah1):
    rng = np.random.default_rng(3)
    sigma, n, Q = 100e-6, 20000, 250e-12
    z = rng.normal(0, sigma, n)
    P = ParticleGroup(
        data=dict(
            x=np.zeros(n),
            px=np.zeros(n),
            y=np.zeros(n),
            py=np.zeros(n),
            z=z,
            pz=np.full(n, 130e6),
            t=np.zeros(n),
            status=np.ones(n, dtype=int),
            weight=np.full(n, Q / n),
            species="electron",
        )
    )

    P2 = P.apply_wakefield(ah1, length=ah1.length, dz=sigma / 10)
    dpz = P2.pz - P.pz
    expected = -ah1.length * Q * ah1.wake_potential(z, sigma)

    core = np.abs(z) < 2 * sigma
    assert np.abs(dpz[core] - expected[core]).max() < 0.05 * np.abs(expected).max()

    # Without dz the singular wake still takes the binned path.
    P3 = P.apply_wakefield(ah1, length=ah1.length)
    assert np.all(np.isfinite(P3.pz))
    assert np.mean(P3.pz) < np.mean(P.pz)


# -----------------------------------------------------------------------------
# Defaults and impedance
# -----------------------------------------------------------------------------


def test_default_zmax_covers_the_long_range_term(published):
    expected = max(25 / published.k0, 2 * np.pi / published.k3)
    assert published.default_zmax == pytest.approx(expected)


def test_min_wavelength_is_the_shortest_scale(published):
    expected = min(
        1 / published.k0,
        (2 * np.pi) ** (1 / published.alpha) / published.k1,
        2 * np.pi / published.k3,
    )
    assert published.min_wavelength == pytest.approx(expected)


def test_defaults_are_not_offered_without_a_scale():
    wake = ParametricWakefield(length=1.0, D=1.0)
    with pytest.raises(NotImplementedError):
        wake.default_zmax
    with pytest.raises(NotImplementedError):
        wake.min_wavelength


def test_impedance_is_not_available(ah1):
    with pytest.raises(NotImplementedError, match="impedance"):
        ah1.impedance(1e3)
