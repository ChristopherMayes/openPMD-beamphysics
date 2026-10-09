"""Tests for the Bokeh plotting backend."""

from __future__ import annotations

import numpy as np
import pytest

try:
    from bokeh.io import save
    from bokeh.models import (
        DataRange1d,
        Div,
        GridBox,
        LinearColorMapper,
        LogColorMapper,
        Range1d,
    )
    from bokeh.models.layouts import LayoutDOM
    from bokeh.palettes import Inferno256
    from bokeh.resources import Resources
except ImportError:
    # <-- I like sorted imports without 'noqa' everywhere; repeat import then
    # skip here
    bokeh = pytest.importorskip("bokeh")
    raise


from conftest import test_artifacts

import beamphysics.plot_bokeh as _plot_bokeh_mod
from beamphysics import ParticleGroup, set_default_backend
from beamphysics.particles import single_particle
from beamphysics.plot_dispatch import get_backend, get_default_backend
from beamphysics.wavefront.wavefront import Wavefront

_bokeh_artifacts = test_artifacts / "bokeh"
_bokeh_artifacts.mkdir(exist_ok=True)
_resources = Resources()


@pytest.fixture(scope="module")
def P() -> ParticleGroup:
    return ParticleGroup("docs/examples/data/bmad_particles.h5")


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
    assert get_backend().name == get_default_backend()


def test_dispatch_explicit_bokeh():
    be = get_backend("bokeh")
    assert be.name == "bokeh"
    assert be.marginal_plot is _plot_bokeh_mod.marginal_plot


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


def test_unused_kwargs_warn(P):
    with pytest.warns(UserWarning, match="not used by the 'bokeh'"):
        P.plot("x", backend="bokeh", return_figure=True, figsize=(4, 4))


# ---------------------------------------------------------------------------
# ParticleGroup density_plot (1D)
# ---------------------------------------------------------------------------


def test_density_plot(P):
    result = P.plot("x", backend="bokeh", return_figure=True)
    assert isinstance(result, LayoutDOM)


def test_density_plot_with_options(P):
    result = P.plot("t", backend="bokeh", return_figure=True, bins=50)
    assert isinstance(result, LayoutDOM)


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
def test_marginal_plot(P, key1, key2):
    result = P.plot(key1, key2, backend="bokeh", return_figure=True)
    assert isinstance(result, LayoutDOM)


def test_marginal_plot_with_ellipse(P):
    result = P.plot("x", "px", backend="bokeh", return_figure=True, ellipse=True)
    assert isinstance(result, LayoutDOM)


def _joint_figure(layout):
    grid = layout if isinstance(layout, GridBox) else layout.children[0]
    return next(child for child, row, col in grid.children if (row, col) == (1, 0))


@pytest.mark.parametrize("tex", [False, True])
def test_marginal_plot_tex_labels(P, tex):
    layout = get_backend("bokeh").marginal_plot(P, "x", "px", tex=tex, show=False)
    joint = _joint_figure(layout)
    assert ("$$" in joint.xaxis.axis_label) == tex
    top = next(c for c, r, col in _grid_of(layout).children if (r, col) == (0, 0))
    assert ("$$" in top.yaxis.axis_label) == tex


def test_marginal_plot_limits(P):
    be = get_backend("bokeh")
    joint = _joint_figure(be.marginal_plot(P, "x", "px", show=False))
    assert isinstance(joint.x_range, DataRange1d)
    joint = _joint_figure(
        be.marginal_plot(P, "x", "px", xlim=(-1e-3, 1e-3), show=False)
    )
    assert isinstance(joint.x_range, Range1d)
    assert joint.x_range.start == pytest.approx(-1)
    assert joint.x_range.end == pytest.approx(1)


def test_marginal_plot_all_lost_renders_message(P):
    P_dead = ParticleGroup(data=dict(P.data))
    P_dead.status = np.zeros_like(P_dead.status)
    result = get_backend("bokeh").marginal_plot(P_dead, "x", "y", show=False)
    assert isinstance(result, Div)
    assert "lost" in result.text


# ---------------------------------------------------------------------------
# ParticleGroup slice_plot
# ---------------------------------------------------------------------------

SLICE_KEYS = ["sigma_x", "norm_emit_x"]


@pytest.mark.parametrize("stat_key", SLICE_KEYS)
def test_slice_plot(P, stat_key):
    result = P.slice_plot(stat_key, backend="bokeh", return_figure=True)
    assert isinstance(result, LayoutDOM)


def test_slice_plot_multi_keys(P):
    result = P.slice_plot("sigma_x", "sigma_y", backend="bokeh", return_figure=True)
    assert isinstance(result, LayoutDOM)


@pytest.mark.parametrize("tex", [False, True])
def test_slice_plot_legend_is_unicode(P, tex):
    fig = P.slice_plot(
        "sigma_x",
        "norm_emit_y",
        backend="bokeh",
        tex=tex,
        show=False,
        return_figure=True,
    )
    labels = [item.label.value for item in fig.legend[0].items]
    assert labels[0].startswith("σ_x (")
    assert labels[1].startswith("ε_n,y (")
    assert not any("$" in label or "\\" in label for label in labels)


# ---------------------------------------------------------------------------
# ParticleGroup density_and_slice_plot
# ---------------------------------------------------------------------------


def test_density_and_slice_plot(P):
    be = get_backend("bokeh")
    result = be.density_and_slice_plot(P, key1="t", key2="p")
    assert isinstance(result, LayoutDOM)


def test_density_and_slice_plot_custom_keys(P):
    be = get_backend("bokeh")
    result = be.density_and_slice_plot(
        P, key1="t", key2="energy", stat_keys=["sigma_x", "sigma_y"]
    )
    assert isinstance(result, LayoutDOM)


# ---------------------------------------------------------------------------
# Single-particle edge case
# ---------------------------------------------------------------------------

single_particle_warnings = pytest.mark.filterwarnings(
    "ignore:.*invalid value encountered in.*",
    "ignore:.*divide by zero.*",
    "ignore:.*Degrees of freedom.*",
    "ignore:.*The fit may be poorly conditioned.*",
)


@single_particle_warnings
def test_single_particle_density():
    Ps = single_particle(pz=10e6)
    result = Ps.plot("x", backend="bokeh", return_figure=True)
    assert isinstance(result, LayoutDOM)


@single_particle_warnings
def test_single_particle_marginal():
    Ps = single_particle(pz=10e6)
    result = Ps.plot("x", "px", backend="bokeh", return_figure=True)
    assert isinstance(result, LayoutDOM)
    joint = _joint_figure(result)
    assert (joint.x_range.start, joint.x_range.end) == (-1, 1)


# ---------------------------------------------------------------------------
# Generic plot_1d_density (used by Wavefront)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["bar", "line"])
def test_plot_1d_density(kind):
    be = get_backend("bokeh")
    x = np.linspace(0, 10, 50)
    y = np.exp(-x)
    result = be.plot_1d_density(
        x, y, x_name="x", y_name="f(x)", kind=kind, return_figure=True
    )
    assert isinstance(result, LayoutDOM)


def test_plot_1d_density_with_data_dict():
    be = get_backend("bokeh")
    data = {"time": np.linspace(0, 1, 100), "signal": np.random.randn(100)}
    result = be.plot_1d_density(
        "time", "signal", data=data, kind="line", return_figure=True
    )
    assert isinstance(result, LayoutDOM)


def test_plot_1d_density_with_cdf():
    be = get_backend("bokeh")
    x = np.linspace(0, 10, 50)
    y = np.exp(-x)
    result = be.plot_1d_density(x, y, show_cdf=True, return_figure=True)
    assert isinstance(result, LayoutDOM)


# ---------------------------------------------------------------------------
# Generic plot_2d_density_with_marginals (used by Wavefront)
# ---------------------------------------------------------------------------


def test_plot_2d_density_with_marginals():
    be = get_backend("bokeh")
    data = np.random.rand(50, 50)
    result = be.plot_2d_density_with_marginals(
        data, dx=0.1, dy=0.1, x_name="x", y_name="y", return_figure=True
    )
    assert isinstance(result, LayoutDOM)


def _color_mapper(layout):
    main = _joint_figure(layout)
    return next(
        r.glyph.color_mapper for r in main.renderers if hasattr(r.glyph, "color_mapper")
    )


def test_plot_2d_density_with_log_scale():
    be = get_backend("bokeh")
    data = np.random.rand(50, 50)
    data[0, :] = 0  # a zero row: must not break the log marginals
    result = be.plot_2d_density_with_marginals(
        data,
        dx=0.1,
        dy=0.1,
        log_scale_z=True,
        log_scale_marginals=True,
        return_figure=True,
    )
    assert isinstance(_color_mapper(result), LogColorMapper)
    top = next(c for c, r, col in result.children if (r, col) == (0, 0))
    source = top.renderers[0].data_source.data
    assert np.all(np.asarray(source["top"]) > 0)
    assert top.renderers[0].glyph.bottom > 0


def test_plot_2d_density_cmap():
    be = get_backend("bokeh")
    layout = be.plot_2d_density_with_marginals(
        np.random.rand(10, 10), cmap="inferno", show=False
    )
    mapper = _color_mapper(layout)
    assert isinstance(mapper, LinearColorMapper)
    assert tuple(mapper.palette) == tuple(Inferno256)
    with pytest.raises(ValueError, match="No 256-color"):
        be.plot_2d_density_with_marginals(
            np.random.rand(10, 10), cmap="nope", show=False
        )


# ---------------------------------------------------------------------------
# Wavefront plots
# ---------------------------------------------------------------------------


@pytest.fixture
def W() -> Wavefront:
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
def test_wavefront_plot_power(W):
    result = W.plot_power(backend="bokeh", return_figure=True)
    assert isinstance(result, LayoutDOM)


def test_wavefront_plot_fluence(W):
    result = W.plot_fluence(backend="bokeh", return_figure=True)
    assert isinstance(result, LayoutDOM)


def test_wavefront_plot2_deprecated(W):
    with pytest.deprecated_call():
        result = W.plot2(backend="bokeh", return_figure=True)
    assert isinstance(result, LayoutDOM)


def test_wavefront_plot_spectral_intensity(W):
    Wk = W.to_kspace()
    result = Wk.plot_spectral_intensity(backend="bokeh", return_figure=True)
    assert isinstance(result, LayoutDOM)


def test_wavefront_plot_photon_energy_spectrum(W):
    Wk = W.to_kspace()
    result = Wk.plot_photon_energy_spectrum(backend="bokeh", return_figure=True)
    assert isinstance(result, LayoutDOM)


# ---------------------------------------------------------------------------
# Marginal grid sizing
# ---------------------------------------------------------------------------


def _grid_of(layout):
    if isinstance(layout, GridBox):
        return layout
    return next(child for child in layout.children if isinstance(child, GridBox))


@pytest.mark.parametrize("stats_location", ["top-right", "bottom"])
def test_marginal_plot_responsive_grid(P, stats_location):
    be = get_backend("bokeh")
    layout = be.marginal_plot(
        P,
        "x",
        "y",
        sizing_mode="stretch_both",
        marginal_fraction=0.25,
        stats_location=stats_location,
        show=False,
    )
    grid = _grid_of(layout)
    assert grid.sizing_mode == "stretch_both"
    assert grid.cols == ["minmax(0, 75fr)", "minmax(0, 25fr)"]
    assert grid.rows == ["minmax(0, 25fr)", "minmax(0, 75fr)"]
    assert grid.aspect_ratio is None
    assert grid.styles["aspect-ratio"] == "600 / 600"
    for child, _row, _col in grid.children:
        assert child.sizing_mode == "stretch_both"
        assert child.aspect_ratio is None


def test_marginal_plot_scale_both_keeps_aspect(P):
    be = get_backend("bokeh")
    layout = be.marginal_plot(
        P, "x", "y", sizing_mode="scale_both", width=800, height=400, show=False
    )
    assert _grid_of(layout).aspect_ratio == 2.0


def test_marginal_plot_fixed_sizes(P):
    be = get_backend("bokeh")
    layout = be.marginal_plot(
        P, "x", "y", width=600, height=300, marginal_fraction=0.25, show=False
    )
    grid = _grid_of(layout)
    assert grid.sizing_mode is None
    assert grid.rows is None and grid.cols is None
    sizes = {
        (row, col): (child.width, child.height) for child, row, col in grid.children
    }
    assert sizes == {
        (0, 0): (450, 75),
        (0, 1): (150, 75),
        (1, 0): (450, 225),
        (1, 1): (150, 225),
    }
    assert all(child.sizing_mode is None for child, _, _ in grid.children)


def test_marginal_plot_stats_div_scrolls(P):
    be = get_backend("bokeh")
    layout = be.marginal_plot(P, "x", "y", text="a<br>" * 50, show=False)
    stats = next(
        child for child, _, _ in _grid_of(layout).children if isinstance(child, Div)
    )
    assert 'class="stats-cell"' in stats.text
    assert 'class="stats-popover"' in stats.text
    assert any("@container" in css for css in stats.stylesheets)


def test_plot_2d_density_with_marginals_responsive_grid():
    be = get_backend("bokeh")
    layout = be.plot_2d_density_with_marginals(
        np.random.rand(20, 20),
        sizing_mode="stretch_both",
        marginal_fraction=0.2,
        show=False,
    )
    assert layout.cols == ["minmax(0, 80fr)", "minmax(0, 20fr)"]
    assert layout.rows == ["minmax(0, 20fr)", "minmax(0, 80fr)"]
    assert all(child.sizing_mode == "stretch_both" for child, _, _ in layout.children)
