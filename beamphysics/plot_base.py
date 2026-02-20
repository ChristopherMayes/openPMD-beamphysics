from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np

from .labels import mathlabel
from .statistics import twiss_ellipse_points
from .units import nice_array, nice_scale_prefix, plottable_array

if TYPE_CHECKING:
    from .particles import ParticleGroup


class PlotPreparationError(Exception): ...


class NanDataError(PlotPreparationError): ...


@dataclass
class MarginalAxisData:
    """
    Single axis of a marginal plot.
    """

    key: str
    data: np.ndarray
    lim: tuple[float, float]
    unit_factor: float
    unit_prefix: str
    unit_symbol: str

    # Histogram/Profile data
    hist_centers: np.ndarray
    hist_values: np.ndarray
    hist_width: np.ndarray
    hist_unit_factor: float
    hist_label_prefix: str

    @property
    def full_unit(self) -> str:
        """Returns the combined scale prefix and root symbol, e.g. 'mm'"""
        return f"{self.unit_prefix}{self.unit_symbol}"

    @property
    def axis_label(self) -> str:
        if self.unit_symbol == "s":
            _, hist_prefix = nice_scale_prefix(self.hist_unit_factor / self.unit_factor)
            return f"{hist_prefix}A"
        return self.hist_label_prefix + mathlabel(f"C/{self.full_unit}")


@dataclass
class MarginalPlotData:
    """
    Marginal plot data.
    """

    x: MarginalAxisData
    y: MarginalAxisData
    weights: np.ndarray
    bins: int
    ellipse_x: np.ndarray | None = None  # Scaled ellipse coords
    ellipse_y: np.ndarray | None = None  # Scaled ellipse coords


def calculate_marginal(
    data_scaled: np.ndarray,
    weights: np.ndarray,
    bins_count: int,
):
    hist, bin_edges = np.histogram(data_scaled, bins=bins_count, weights=weights)
    h_width = np.diff(bin_edges)
    h_centers = bin_edges[:-1] + h_width / 2

    profile = hist / h_width

    profile_scaled, profile_f, profile_prefix = nice_array(profile)
    return profile_scaled, h_centers, h_width, profile_f, profile_prefix


def prepare_marginal_plot(
    particle_group: ParticleGroup,
    key1: str = "t",
    key2: str = "p",
    bins: int | None = None,
    *,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    nice: bool = True,
    ellipse: bool = False,
) -> MarginalPlotData:
    """
    Prepares the data structures required for marginal plotting.

    Calculates units, scaling, limits, histograms, and ellipses.
    """
    if not bins:
        n = len(particle_group)
        bins = int(np.sqrt(n / 4))

    x_raw = cast(np.ndarray, particle_group[key1])
    y_raw = cast(np.ndarray, particle_group[key2])

    if np.all(np.isnan(x_raw)):
        raise NanDataError(f"{key1} is all NaN")

    if np.all(np.isnan(y_raw)):
        raise NanDataError(f"{key2} is all NaN")

    if len(x_raw) == 1:
        bins = 100
        if xlim is None:
            (x0,) = x_raw
            if np.isclose(x0, 0.0):
                xlim = (-1.0, 1.0)
            else:
                params = sorted((0.9 * x0, 1.1 * x0))
                xlim = (params[0], params[1])
        if ylim is None:
            (y0,) = y_raw
            if np.isclose(y0, 0.0):
                ylim = (-1.0, 1.0)
            else:
                params = sorted((0.9 * y0, 1.1 * y0))
                ylim = (params[0], params[1])

    x_data, f1, p1, xmin_raw, xmax_raw = plottable_array(x_raw, nice=nice, lim=xlim)
    y_data, f2, p2, ymin_raw, ymax_raw = plottable_array(y_raw, nice=nice, lim=ylim)

    u1 = particle_group.units(key1).unitSymbol
    u2 = particle_group.units(key2).unitSymbol

    weights = cast(np.ndarray, particle_group["weight"])

    # X marginal (top)
    x_prof, x_cents, x_width, x_prof_factor, x_prof_prefix = calculate_marginal(
        x_data, weights, bins
    )

    # Y marginal (right)
    y_prof, y_cents, y_width, y_prof_factor, y_prof_prefix = calculate_marginal(
        y_data, weights, bins
    )

    ell_x, ell_y = None, None
    if ellipse and len(x_data) > 1:
        sigma_mat2 = particle_group.cov(key1, key2)
        x_ellipse, y_ellipse = twiss_ellipse_points(sigma_mat2)
        x_ellipse += particle_group.avg(key1)
        y_ellipse += particle_group.avg(key2)
        ell_x = x_ellipse / f1
        ell_y = y_ellipse / f2

    return MarginalPlotData(
        x=MarginalAxisData(
            key=key1,
            data=x_data,
            lim=(xmin_raw / f1, xmax_raw / f1),
            unit_factor=f1,
            unit_prefix=p1,
            unit_symbol=u1,
            hist_centers=x_cents,
            hist_values=x_prof,
            hist_width=x_width,
            hist_unit_factor=x_prof_factor,
            hist_label_prefix=x_prof_prefix,
        ),
        y=MarginalAxisData(
            key=key2,
            data=y_data,
            lim=(ymin_raw / f2, ymax_raw / f2),
            unit_factor=f2,
            unit_prefix=p2,
            unit_symbol=u2,
            hist_centers=y_cents,
            hist_values=y_prof,
            hist_width=y_width,
            hist_unit_factor=y_prof_factor,
            hist_label_prefix=y_prof_prefix,
        ),
        weights=weights,
        bins=bins,
        ellipse_x=ell_x,
        ellipse_y=ell_y,
    )
