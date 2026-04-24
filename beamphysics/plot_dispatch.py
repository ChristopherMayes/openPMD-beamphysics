"""
Plot backend dispatch for openPMD-beamphysics.

Provides a protocol-typed PlotBackend that holds callables for each plot type.
Each backend (mpl, bokeh) provides functions conforming to these protocols.
Backend-specific extra keyword arguments are allowed via ``**kwargs``.

Resolution priority: explicit ``backend=`` arg > ``obj.plot_backend`` > module default.
"""

from __future__ import annotations

import os

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

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
        xlim: tuple[float, float] | None = ...,
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
        xlim: tuple[float, float] | None = ...,
        ylim: tuple[float, float] | None = ...,
        nice: bool = ...,
        ellipse: bool = ...,
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
        xlim: tuple[float, float] | None = ...,
        ylim: tuple[float, float] | None = ...,
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
        xlim: tuple[float, float] | None = ...,
        ylim: tuple[float, float] | None = ...,
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
        xlim: tuple[float, float] | None = ...,
        ylim: tuple[float, float] | None = ...,
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
    plot_1d_density: Plot1dDensityFn
    plot_2d_density_with_marginals: Plot2dDensityWithMarginalsFn


# ---------------------------------------------------------------------------
# Module-level default and resolution
# ---------------------------------------------------------------------------

_default_backend: str = os.environ.get("BEAMPHYSICS_PLOT", "mpl")
_backend_cache: dict[str, PlotBackend] = {}


def set_default_backend(name: str) -> None:
    """
    Set the module-level default plot backend.

    Parameters
    ----------
    name : str
        ``"mpl"`` for Matplotlib or ``"bokeh"`` for Bokeh.
    """
    global _default_backend
    if name not in ("mpl", "bokeh"):
        raise ValueError(f"Unknown backend {name!r}. Choose 'mpl' or 'bokeh'.")
    _default_backend = name


def get_default_backend() -> str:
    """Return the current module-level default backend name."""
    return _default_backend


def resolve_backend(backend: str | None = None, obj: Any = None) -> str:
    """
    Determine which backend to use.

    Priority: explicit *backend* argument > ``obj.plot_backend`` > module default.
    """
    if backend is not None:
        return backend
    if obj is not None:
        obj_backend = getattr(obj, "plot_backend", None)
        if obj_backend is not None:
            return obj_backend
    return _default_backend


def _load_mpl_backend() -> PlotBackend:
    from . import plot as mod

    return PlotBackend(
        name="mpl",
        density_plot=mod.density_plot,
        marginal_plot=mod.marginal_plot,
        slice_plot=mod.slice_plot,
        wakefield_plot=mod.wakefield_plot,
        plot_1d_density=mod.plot_1d_density,
        plot_2d_density_with_marginals=mod.plot_2d_density_with_marginals,
    )


def _load_bokeh_backend() -> PlotBackend:
    from . import plot_bokeh as mod

    return PlotBackend(
        name="bokeh",
        density_plot=mod.density_plot,
        marginal_plot=mod.marginal_plot,
        slice_plot=mod.slice_plot,
        wakefield_plot=mod.wakefield_plot,
        plot_1d_density=mod.plot_1d_density,
        plot_2d_density_with_marginals=mod.plot_2d_density_with_marginals,
    )


_backend_loaders = {
    "mpl": _load_mpl_backend,
    "bokeh": _load_bokeh_backend,
}


def get_backend(backend: str | None = None, obj: Any = None) -> PlotBackend:
    """
    Resolve the backend name and return a :class:`PlotBackend`.

    The backend module is lazy-loaded on first access and cached.

    Parameters
    ----------
    backend : str or None
        Explicit backend name (``"mpl"`` or ``"bokeh"``).
        If ``None``, falls through to *obj* and then the module default.
    obj : object, optional
        An object with an optional ``plot_backend`` attribute
        (e.g. a :class:`ParticleGroup`).
    """
    name = resolve_backend(backend, obj)
    if name not in _backend_cache:
        loader = _backend_loaders.get(name)
        if loader is None:
            raise ValueError(f"Unknown backend {name!r}. Choose 'mpl' or 'bokeh'.")
        _backend_cache[name] = loader()
    return _backend_cache[name]
