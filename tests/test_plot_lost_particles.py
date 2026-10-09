"""Lost-particle filtering and the ``n_dead`` annotation on both plot backends."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest
from bokeh.models import Div, GridBox
from matplotlib.text import Text

from beamphysics import ParticleGroup
from beamphysics.plot_base import PlotPreparationError, drop_lost_particles
from beamphysics.plot_dispatch import get_backend

matplotlib.use("Agg")

N_DEAD = 25


@pytest.fixture(scope="module")
def alive() -> ParticleGroup:
    return ParticleGroup("docs/examples/data/bmad_particles.h5")


@pytest.fixture
def with_dead(alive: ParticleGroup) -> ParticleGroup:
    P = ParticleGroup(data=dict(alive.data))
    P.status = np.array(P.status, copy=True)
    P.status[:N_DEAD] = 0
    return P


def test_drop_lost_particles_filters_and_counts(with_dead: ParticleGroup):
    kept, n_dead = drop_lost_particles(with_dead)
    assert n_dead == N_DEAD
    assert kept.n_particle == with_dead.n_particle - N_DEAD
    assert kept.n_dead == 0


def test_drop_lost_particles_passthrough(
    alive: ParticleGroup, with_dead: ParticleGroup
):
    kept, n_dead = drop_lost_particles(alive)
    assert kept is alive and n_dead == 0
    kept, n_dead = drop_lost_particles(with_dead, filter_lost_particles=False)
    assert kept is with_dead and n_dead == N_DEAD
    _, n_dead = drop_lost_particles(alive, n_dead=7)
    assert n_dead == 7


def test_drop_lost_particles_all_lost(with_dead: ParticleGroup):
    with_dead.status[:] = 0
    with pytest.raises(PlotPreparationError, match="lost"):
        drop_lost_particles(with_dead)


def _bokeh_stats_text(layout) -> str:
    grid = layout if isinstance(layout, GridBox) else layout.children[0]
    return "".join(c.text for c, _, _ in grid.children if isinstance(c, Div))


def _bokeh_top_hist(layout) -> np.ndarray:
    grid = layout if isinstance(layout, GridBox) else layout.children[0]
    top = next(c for c, r, col in grid.children if (r, col) == (0, 0))
    return np.asarray(top.renderers[0].data_source.data["top"])


def test_bokeh_marginal_filters_and_annotates(with_dead: ParticleGroup):
    be = get_backend("bokeh")
    filtered = be.marginal_plot(with_dead, "x", "y", bins=50, show=False)
    unfiltered = be.marginal_plot(
        with_dead, "x", "y", bins=50, filter_lost_particles=False, show=False
    )
    assert not np.allclose(_bokeh_top_hist(filtered), _bokeh_top_hist(unfiltered))
    for layout in (filtered, unfiltered):
        text = _bokeh_stats_text(layout)
        assert "n_dead" in text and f"{N_DEAD:,}" in text and "color:red" in text


def test_bokeh_marginal_custom_text_keeps_n_dead(with_dead: ParticleGroup):
    be = get_backend("bokeh")
    layout = be.marginal_plot(with_dead, "x", "y", text="hello", show=False)
    text = _bokeh_stats_text(layout)
    assert "hello" in text and "n_dead" in text and "⟨x⟩" not in text
    layout = be.marginal_plot(with_dead, "x", "y", text="hello", stats=True, show=False)
    text = _bokeh_stats_text(layout)
    assert "hello" in text and "n_dead" in text and "⟨x⟩" in text


def test_bokeh_marginal_no_dead_no_line(alive: ParticleGroup):
    be = get_backend("bokeh")
    assert "n_dead" not in _bokeh_stats_text(
        be.marginal_plot(alive, "x", "y", show=False)
    )


def _mpl_texts(fig) -> list[Text]:
    return [t for t in fig.findobj(Text) if t.get_text()]


def test_mpl_marginal_filters_and_annotates(with_dead: ParticleGroup):
    be = get_backend("mpl")
    fig = be.marginal_plot(with_dead, "x", "y", title="spot", stats=True)
    texts = {t.get_text(): t for t in _mpl_texts(fig)}
    dead = next(t for s, t in texts.items() if s.startswith("n_dead"))
    assert dead.get_text() == f"n_dead = {N_DEAD:,}"
    assert dead.get_color() == "red"
    assert any(s.startswith("⟨x⟩") for s in texts)
    assert "spot" in texts
    matplotlib.pyplot.close(fig)


def test_mpl_marginal_stats_opt_in_keeps_n_dead(with_dead: ParticleGroup):
    be = get_backend("mpl")
    fig = be.marginal_plot(with_dead, "x", "y")
    labels = [t.get_text() for t in _mpl_texts(fig)]
    assert any(s.startswith("n_dead") for s in labels)
    assert not any(s.startswith("⟨x⟩") for s in labels)
    matplotlib.pyplot.close(fig)


@pytest.mark.parametrize("backend", ["mpl", "bokeh"])
def test_particlegroup_plot_forwards_filter(with_dead: ParticleGroup, backend: str):
    kwargs = {"show": False} if backend == "bokeh" else {}
    result = with_dead.plot(
        "x",
        "px",
        backend=backend,
        return_figure=True,
        filter_lost_particles=False,
        **kwargs,
    )
    assert result is not None
    if backend == "mpl":
        matplotlib.pyplot.close(result)
