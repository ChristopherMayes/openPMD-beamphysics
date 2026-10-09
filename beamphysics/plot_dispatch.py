from __future__ import annotations

import functools
import logging
import os
import sys
from dataclasses import dataclass, fields
from types import ModuleType
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from .plot_base import Limit

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .particles import ParticleGroup


# ---------------------------------------------------------------------------
# Protocols – common parameter signatures for each plot function
# ---------------------------------------------------------------------------


class DensityPlotFn(Protocol):
    """1D density histogram of a single particle key."""

    def __call__(
        self,
        particle_group: ParticleGroup,
        key: str = ...,
        bins: int | str | None = ...,
        *,
        xlim: Limit | None = ...,
        nice: bool = ...,
        **kwargs: Any,
    ) -> Any: ...


class MarginalPlotFn(Protocol):
    """2D density with marginal histograms."""

    def __call__(
        self,
        particle_group: ParticleGroup,
        key1: str = ...,
        key2: str = ...,
        bins: int | None = ...,
        *,
        xlim: Limit | None = ...,
        ylim: Limit | None = ...,
        nice: bool = ...,
        ellipse: bool = ...,
        stats: bool = ...,
        text: str | None = ...,
        title: str | None = ...,
        filter_lost_particles: bool = ...,
        n_dead: int | None = ...,
        **kwargs: Any,
    ) -> Any: ...


class SlicePlotFn(Protocol):
    """Slice statistics with density overlay."""

    def __call__(
        self,
        particle_group: ParticleGroup,
        *keys: str,
        n_slice: int = ...,
        slice_key: str | None = ...,
        xlim: Limit | None = ...,
        ylim: Limit | None = ...,
        nice: bool = ...,
        **kwargs: Any,
    ) -> Any: ...


class WakefieldPlotFn(Protocol):
    """Wakefield kicks scatter with density overlay."""

    def __call__(
        self,
        particle_group: ParticleGroup,
        wake: Any,
        key: str | None = ...,
        nice: bool = ...,
        xlim: Limit | None = ...,
        ylim: Limit | None = ...,
        **kwargs: Any,
    ) -> Any: ...


class Plot1dDensityFn(Protocol):
    """Generic 1D density distribution plot."""

    def __call__(
        self,
        x: str | np.ndarray,
        y: str | np.ndarray,
        *,
        nice: bool = ...,
        xlim: Limit | None = ...,
        ylim: Limit | None = ...,
        **kwargs: Any,
    ) -> Any: ...


class Plot2dDensityWithMarginalsFn(Protocol):
    """Generic 2D density map with marginal histograms."""

    def __call__(
        self,
        data: np.ndarray,
        dx: float = ...,
        dy: float = ...,
        **kwargs: Any,
    ) -> Any: ...


class DensityAndSlicePlotFn(Protocol):
    """2D density with overlaid slice statistics."""

    def __call__(
        self,
        particle_group: ParticleGroup,
        key1: str = ...,
        key2: str = ...,
        stat_keys: list[str] = ...,
        bins: int = ...,
        n_slice: int = ...,
        **kwargs: Any,
    ) -> Any: ...


# ---------------------------------------------------------------------------
# PlotBackend – holds one callable per plot type
# ---------------------------------------------------------------------------


@dataclass
class PlotBackend:
    """
    Container for a set of plot functions from a single backend.

    Instantiated lazily on first access via `get_backend`.
    """

    name: str
    density_plot: DensityPlotFn
    marginal_plot: MarginalPlotFn
    slice_plot: SlicePlotFn
    wakefield_plot: WakefieldPlotFn
    density_and_slice_plot: DensityAndSlicePlotFn
    plot_1d_density: Plot1dDensityFn
    plot_2d_density_with_marginals: Plot2dDensityWithMarginalsFn


# ---------------------------------------------------------------------------
# Module-level default and resolution
# ---------------------------------------------------------------------------

_default_backend: str = os.environ.get("BEAMPHYSICS_PLOT", "mpl")
_backend_cache: dict[str, PlotBackend] = {}


def _check_backend_name(name: str) -> None:
    if name not in _backend_loaders:
        choices = ", ".join(repr(key) for key in _backend_loaders)
        raise ValueError(f"Unknown backend {name!r}. Choose one of: {choices}")


def set_default_backend(name: str) -> None:
    """
    Set the module-level default plot backend.

    Parameters
    ----------
    name : str
        ``"mpl"`` for Matplotlib or ``"bokeh"`` for Bokeh.
    """
    global _default_backend
    _check_backend_name(name)
    _default_backend = name


def get_default_backend() -> str:
    """Return the current module-level default backend name."""
    return _default_backend


@functools.cache
def is_jupyter() -> bool:
    """
    Determine if we're in a Jupyter notebook session.

    This works by way of interacting with IPython display and seeing what
    choice it makes regarding reprs.

    Returns
    -------
    bool
    """
    if "IPython" not in sys.modules or "IPython.display" not in sys.modules:
        return False

    from IPython.display import display

    class ReprCheck:
        def _repr_html_(self) -> str:
            self.mode = "jupyter"
            logger.info("Detected Jupyter. Using the notebook graph backend.")
            return "<!-- Detected Jupyter. -->"

        def __repr__(self) -> str:
            self.mode = "console"
            return ""

    check = ReprCheck()
    display(check)
    return check.mode == "jupyter"


def _backend_from_module(name: str, mod: ModuleType) -> PlotBackend:
    """Collect the plot functions of a backend module by their shared names."""
    funcs = {
        field.name: getattr(mod, field.name)
        for field in fields(PlotBackend)
        if field.name != "name"
    }
    return PlotBackend(name=name, **funcs)


def _load_mpl_backend() -> PlotBackend:
    from . import plot

    return _backend_from_module("mpl", plot)


def _load_bokeh_backend() -> PlotBackend:
    from . import plot_bokeh

    if is_jupyter():
        plot_bokeh.initialize_jupyter()
    return _backend_from_module("bokeh", plot_bokeh)


_backend_loaders = {
    "mpl": _load_mpl_backend,
    "bokeh": _load_bokeh_backend,
}


def get_backend(backend: str | None = None) -> PlotBackend:
    """
    Resolve the backend name and return a :class:`PlotBackend`.

    The backend module is lazy-loaded on first access and cached.

    Parameters
    ----------
    backend : str or None
        Explicit backend name (``"mpl"`` or ``"bokeh"``).
        If ``None``, the module default is used (see `set_default_backend`).
    """
    name = _default_backend if backend is None else backend
    _check_backend_name(name)
    if name not in _backend_cache:
        _backend_cache[name] = _backend_loaders[name]()
    return _backend_cache[name]
