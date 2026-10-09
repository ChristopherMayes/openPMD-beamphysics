"""Backend-agnostic marginal plot behaviour, checked on the matplotlib backend."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from beamphysics import ParticleGroup
from beamphysics.particles import single_particle
from beamphysics.plot import marginal_plot, plot_2d_density_with_marginals
from beamphysics.plot_base import prepare_marginal_plot

matplotlib.use("Agg")


@pytest.fixture(scope="module")
def P() -> ParticleGroup:
    return ParticleGroup("docs/examples/data/bmad_particles.h5")


@pytest.fixture
def close_figures():
    yield
    plt.close("all")


def _axes(fig):
    """(joint, top marginal, right marginal) axes of a marginal plot figure."""
    return fig.axes[0], fig.axes[1], fig.axes[2]


@pytest.mark.parametrize("tex", [False, True])
def test_marginal_labels_follow_tex(P, tex, close_figures):
    fig = marginal_plot(P, "x", "px", tex=tex)
    joint, top, right = _axes(fig)
    for label in (
        joint.get_xlabel(),
        joint.get_ylabel(),
        top.get_ylabel(),
        right.get_xlabel(),
    ):
        assert ("$" in label) == tex, label


def test_prepare_marginal_limits_only_when_requested(P):
    pdata = prepare_marginal_plot(P, "x", "px")
    assert pdata.x.lim is None and pdata.y.lim is None
    pdata = prepare_marginal_plot(P, "x", "px", xlim=(-1e-3, 1e-3))
    assert pdata.x.lim == pytest.approx((-1, 1))
    assert pdata.y.lim is None


def test_marginal_explicit_limits_applied(P, close_figures):
    fig = marginal_plot(P, "x", "px", xlim=(-1e-3, 1e-3), ylim=(-2e3, 2e3))
    joint, top, right = _axes(fig)
    assert joint.get_xlim() == pytest.approx((-1, 1))
    assert top.get_xlim() == pytest.approx((-1, 1))
    assert joint.get_ylim() == pytest.approx((-2, 2))
    assert right.get_ylim() == pytest.approx((-2, 2))


@pytest.mark.filterwarnings("ignore:.*invalid value encountered in.*")
@pytest.mark.filterwarnings("ignore:.*divide by zero.*")
@pytest.mark.filterwarnings("ignore:.*Degrees of freedom.*")
def test_single_particle_marginal_limits(close_figures):
    fig = marginal_plot(single_particle(pz=10e6), "x", "px")
    joint, *_ = _axes(fig)
    assert joint.get_xlim() == (-1, 1)
    assert joint.get_ylim() == (-1, 1)


def test_all_lost_renders_message(P, close_figures):
    P_dead = ParticleGroup(data=dict(P.data))
    P_dead.status = np.zeros_like(P_dead.status)
    fig = marginal_plot(P_dead, "x", "y")
    assert any("lost" in t.get_text() for t in fig.texts)


def test_unused_kwargs_warn(close_figures):
    with pytest.warns(UserWarning, match="not used by the 'mpl'"):
        plot_2d_density_with_marginals(np.random.rand(10, 10), width=300)
