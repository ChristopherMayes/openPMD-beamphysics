"""Tests for the Bokeh plotting backend."""

from __future__ import annotations


import numpy as np
import pytest

try:
    from bokeh.io import save
    from bokeh.models.layouts import LayoutDOM
    from bokeh.resources import Resources
except ImportError:
    # <-- I like sorted imports without 'noqa' everywhere; repeat import then
    # skip here
    bokeh = pytest.importorskip("bokeh")
    raise


from beamphysics import ParticleGroup, set_default_backend
from beamphysics.particles import single_particle
from beamphysics.plot_dispatch import get_backend, get_default_backend, _default_backend
from beamphysics.wavefront.wavefront import Wavefront

from conftest import test_artifacts

import beamphysics.plot_bokeh as _plot_bokeh_mod

P = ParticleGroup("docs/examples/data/bmad_particles.h5")

_bokeh_artifacts = test_artifacts / "bokeh"
_bokeh_artifacts.mkdir(exist_ok=True)
_resources = Resources()


@pytest.fixture(autouse=True)
def _bokeh_show_to_save(request, monkeypatch):
    """Intercept bokeh show() calls and save to HTML artifacts instead."""
    index = 0
    node_name = request.node.name.replace("/", "_")

    def _save_instead(layout):
        nonlocal index
        filename = _bokeh_artifacts / f"{node_name}_{index}.html"
        save(layout, filename=str(filename), resources=_resources, title=node_name)
        print(f"Saved bokeh artifact to {filename}")
        index += 1

    monkeypatch.setattr(_plot_bokeh_mod, "_bokeh_show", _save_instead)


# ---------------------------------------------------------------------------
# Dispatch tests
# ---------------------------------------------------------------------------


def test_dispatch_default():
    assert get_default_backend() == _default_backend
    be = get_backend()
    assert be.name == _default_backend


def test_dispatch_explicit_bokeh():
    be = get_backend("bokeh")
    assert be.name == "bokeh"


def test_dispatch_set_default():
    old = get_default_backend()
    try:
        set_default_backend("bokeh")
        assert get_default_backend() == "bokeh"
        be = get_backend()
        assert be.name == "bokeh"
    finally:
        set_default_backend(old)


def test_dispatch_invalid():
    with pytest.raises(ValueError, match="Unknown backend"):
        set_default_backend("plotly")
    with pytest.raises(ValueError, match="Unknown backend"):
        get_backend("plotly")


# ---------------------------------------------------------------------------
# ParticleGroup density_plot (1D)
# ---------------------------------------------------------------------------


def test_density_plot():
    result = P.plot("x", backend="bokeh", return_figure=True)
    assert isinstance(result, LayoutDOM)
    # Artifact saved automatically by _bokeh_show_to_save fixture
    # _save_bokeh(result,"density_plot_x")


def test_density_plot_with_options():
    result = P.plot("t", backend="bokeh", return_figure=True, bins=50)
    assert isinstance(result, LayoutDOM)
    # Artifact saved automatically by _bokeh_show_to_save fixture
    # _save_bokeh(result,"density_plot_t_bins50")


# ---------------------------------------------------------------------------
# ParticleGroup marginal_plot (2D)
# ---------------------------------------------------------------------------

MARGINAL_PAIRS = [
    ("x", "px"),
    ("y", "py"),
    ("t", "energy"),
    ("x", "y"),
]


@pytest.mark.parametrize("key1,key2", MARGINAL_PAIRS, ids=lambda p: str(p))
def test_marginal_plot(key1, key2):
    result = P.plot(key1, key2, backend="bokeh", return_figure=True)
    assert isinstance(result, LayoutDOM)
    # Artifact saved automatically by _bokeh_show_to_save fixture
    # _save_bokeh(result,f"marginal_plot_{key1}_{key2}")


def test_marginal_plot_with_ellipse():
    result = P.plot("x", "px", backend="bokeh", return_figure=True, ellipse=True)
    assert isinstance(result, LayoutDOM)
    # Artifact saved automatically by _bokeh_show_to_save fixture
    # _save_bokeh(result,"marginal_plot_x_px_ellipse")


# ---------------------------------------------------------------------------
# ParticleGroup slice_plot
# ---------------------------------------------------------------------------

SLICE_KEYS = ["sigma_x", "norm_emit_x"]


@pytest.mark.parametrize("stat_key", SLICE_KEYS)
def test_slice_plot(stat_key):
    result = P.slice_plot(stat_key, backend="bokeh", return_figure=True)
    assert isinstance(result, LayoutDOM)
    # Artifact saved automatically by _bokeh_show_to_save fixture
    # _save_bokeh(result,f"slice_plot_{stat_key}")


def test_slice_plot_multi_keys():
    result = P.slice_plot("sigma_x", "sigma_y", backend="bokeh", return_figure=True)
    assert isinstance(result, LayoutDOM)


# ---------------------------------------------------------------------------
# ParticleGroup density_and_slice_plot
# ---------------------------------------------------------------------------


def test_density_and_slice_plot():
    be = get_backend("bokeh")
    result = be.density_and_slice_plot(P, key1="t", key2="p")
    assert isinstance(result, LayoutDOM)


def test_density_and_slice_plot_custom_keys():
    be = get_backend("bokeh")
    result = be.density_and_slice_plot(
        P, key1="t", key2="energy", stat_keys=["sigma_x", "sigma_y"]
    )
    assert isinstance(result, LayoutDOM)


# ---------------------------------------------------------------------------
# Single-particle edge case
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore:.*invalid value encountered in.*")
@pytest.mark.filterwarnings("ignore:.*divide by zero.*")
@pytest.mark.filterwarnings("ignore:.*Degrees of freedom.*")
@pytest.mark.filterwarnings("ignore:.*The fit may be poorly conditioned.*")
def test_single_particle_density():
    Ps = single_particle(pz=10e6)
    result = Ps.plot("x", backend="bokeh", return_figure=True)
    assert isinstance(result, LayoutDOM)
    # Artifact saved automatically by _bokeh_show_to_save fixture
    # _save_bokeh(result,"single_particle_density")


@pytest.mark.filterwarnings("ignore:.*invalid value encountered in.*")
@pytest.mark.filterwarnings("ignore:.*divide by zero.*")
@pytest.mark.filterwarnings("ignore:.*Degrees of freedom.*")
@pytest.mark.filterwarnings("ignore:.*The fit may be poorly conditioned.*")
def test_single_particle_marginal():
    Ps = single_particle(pz=10e6)
    result = Ps.plot("x", "px", backend="bokeh", return_figure=True)
    assert isinstance(result, LayoutDOM)
    # Artifact saved automatically by _bokeh_show_to_save fixture
    # _save_bokeh(result,"single_particle_marginal")


# ---------------------------------------------------------------------------
# Generic plot_1d_density (used by Wavefront)
# ---------------------------------------------------------------------------


def test_plot_1d_density_bar():
    be = get_backend("bokeh")
    x = np.linspace(0, 10, 50)
    y = np.exp(-x)
    result = be.plot_1d_density(x, y, x_name="x", y_name="f(x)", kind="bar")
    assert isinstance(result, LayoutDOM)
    # Artifact saved automatically by _bokeh_show_to_save fixture
    # _save_bokeh(result,"plot_1d_density_bar")


def test_plot_1d_density_line():
    be = get_backend("bokeh")
    x = np.linspace(0, 10, 50)
    y = np.exp(-x)
    result = be.plot_1d_density(x, y, x_name="x", y_name="f(x)", kind="line")
    assert isinstance(result, LayoutDOM)
    # Artifact saved automatically by _bokeh_show_to_save fixture
    # _save_bokeh(result,"plot_1d_density_line")


def test_plot_1d_density_with_data_dict():
    be = get_backend("bokeh")
    data = {"time": np.linspace(0, 1, 100), "signal": np.random.randn(100)}
    result = be.plot_1d_density("time", "signal", data=data, kind="line")
    assert isinstance(result, LayoutDOM)
    # Artifact saved automatically by _bokeh_show_to_save fixture
    # _save_bokeh(result,"plot_1d_density_data_dict")


def test_plot_1d_density_with_cdf():
    be = get_backend("bokeh")
    x = np.linspace(0, 10, 50)
    y = np.exp(-x)
    result = be.plot_1d_density(x, y, show_cdf=True)
    assert isinstance(result, LayoutDOM)
    # Artifact saved automatically by _bokeh_show_to_save fixture
    # _save_bokeh(result,"plot_1d_density_cdf")


# ---------------------------------------------------------------------------
# Generic plot_2d_density_with_marginals (used by Wavefront)
# ---------------------------------------------------------------------------


def test_plot_2d_density_with_marginals():
    be = get_backend("bokeh")
    data = np.random.rand(50, 50)
    result = be.plot_2d_density_with_marginals(
        data, dx=0.1, dy=0.1, x_name="x", y_name="y"
    )
    assert isinstance(result, LayoutDOM)
    # Artifact saved automatically by _bokeh_show_to_save fixture
    # _save_bokeh(result,"plot_2d_density_with_marginals")


def test_plot_2d_density_with_log_scale():
    be = get_backend("bokeh")
    data = np.random.rand(50, 50) + 0.01
    result = be.plot_2d_density_with_marginals(
        data, dx=0.1, dy=0.1, log_scale_z=True, log_scale_marginals=True
    )
    assert isinstance(result, LayoutDOM)
    # Artifact saved automatically by _bokeh_show_to_save fixture
    # _save_bokeh(result,"plot_2d_density_log_scale")


# ---------------------------------------------------------------------------
# Wavefront plots
# ---------------------------------------------------------------------------


def _make_wavefront():
    return Wavefront.from_gaussian(
        shape=(51, 51, 21),
        dx=10e-6,
        dy=10e-6,
        dz=10e-6,
        wavelength=1e-9,
        sigma0=50e-6,
        energy=1.0,
    )


@pytest.mark.filterwarnings("ignore:.*identical low and high.*:UserWarning")
def test_wavefront_plot_power():
    W = _make_wavefront()
    result = W.plot_power(backend="bokeh")
    assert isinstance(result, LayoutDOM)
    # Artifact saved automatically by _bokeh_show_to_save fixture
    # _save_bokeh(result,"wavefront_plot_power")


def test_wavefront_plot_fluence():
    W = _make_wavefront()
    result = W.plot_fluence(backend="bokeh")
    assert isinstance(result, LayoutDOM)
    # Artifact saved automatically by _bokeh_show_to_save fixture
    # _save_bokeh(result,"wavefront_plot_fluence")


def test_wavefront_plot2():
    W = _make_wavefront()
    result = W.plot2(backend="bokeh")
    assert isinstance(result, LayoutDOM)
    # Artifact saved automatically by _bokeh_show_to_save fixture
    # _save_bokeh(result,"wavefront_plot2")


def test_wavefront_plot_spectral_intensity():
    W = _make_wavefront()
    Wk = W.to_kspace()
    result = Wk.plot_spectral_intensity(backend="bokeh")
    assert isinstance(result, LayoutDOM)
    # Artifact saved automatically by _bokeh_show_to_save fixture
    # _save_bokeh(result,"wavefront_plot_spectral_intensity")


def test_wavefront_plot_photon_energy_spectrum():
    W = _make_wavefront()
    Wk = W.to_kspace()
    result = Wk.plot_photon_energy_spectrum(backend="bokeh")
    assert isinstance(result, LayoutDOM)
    # Artifact saved automatically by _bokeh_show_to_save fixture
    # _save_bokeh(result,"wavefront_plot_photon_energy_spectrum")
