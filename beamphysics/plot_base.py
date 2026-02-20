from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np

from .labels import mathlabel
from .statistics import twiss_ellipse_points
from .units import nice_array, plottable_array_and_units, pmd_unit

if TYPE_CHECKING:
    from .particles import ParticleGroup


def _charge_density_units_str(
    x_unit, axis_units, hist_f: float, axis_f: float = 1.0
) -> str:
    """
    Units string for a histogram density: charge per displayed axis unit.

    The denominator is the axis unit exactly as the axis label shows it,
    prefix included, and the histogram's own scale becomes a prefix on the
    C: 'nC/µm', 'pC/(keV/c)'. A time axis is special-cased to amps
    (C/s = A), which folds the histogram and axis scales into one prefix.

    ``x_unit`` is the coordinate's pmd_unit, ``axis_units`` its displayed
    label string, ``hist_f`` the factor the density values were divided by,
    and ``axis_f`` the factor the coordinate was divided by.
    """
    if (pmd_unit("C") / x_unit).simplify() == pmd_unit("A"):
        return pmd_unit("A").scaled_symbol(hist_f / axis_f)

    numerator = pmd_unit("C").scaled_symbol(hist_f)
    try:
        return (pmd_unit(numerator) / pmd_unit(str(axis_units))).unitSymbol
    except (ValueError, KeyError):
        # Axis units in the explicit power-of-ten form ('1e-3 sqrt(m)').
        return f"{numerator}/({axis_units})"


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
    unit: pmd_unit
    display_unit: str

    # Histogram/Profile data
    hist_centers: np.ndarray
    hist_values: np.ndarray
    hist_width: np.ndarray
    hist_unit_factor: float

    @property
    def full_unit(self) -> str:
        """Returns the scaled display unit, e.g. 'mm'"""
        return self.display_unit

    @property
    def axis_label(self) -> str:
        density_units = _charge_density_units_str(
            self.unit, self.display_unit, self.hist_unit_factor, self.unit_factor
        )
        return mathlabel(units=density_units)


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

    profile_scaled, profile_f, _ = nice_array(profile)
    return profile_scaled, h_centers, h_width, profile_f


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

    u1 = particle_group.units(key1)
    u2 = particle_group.units(key2)

    x_data, f1, ux, xmin_raw, xmax_raw = plottable_array_and_units(
        x_raw, u1, nice=nice, lim=xlim
    )
    y_data, f2, uy, ymin_raw, ymax_raw = plottable_array_and_units(
        y_raw, u2, nice=nice, lim=ylim
    )

    weights = cast(np.ndarray, particle_group["weight"])

    # X marginal (top)
    x_prof, x_cents, x_width, x_prof_factor = calculate_marginal(x_data, weights, bins)

    # Y marginal (right)
    y_prof, y_cents, y_width, y_prof_factor = calculate_marginal(y_data, weights, bins)

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
            unit=u1,
            display_unit=ux,
            hist_centers=x_cents,
            hist_values=x_prof,
            hist_width=x_width,
            hist_unit_factor=x_prof_factor,
        ),
        y=MarginalAxisData(
            key=key2,
            data=y_data,
            lim=(ymin_raw / f2, ymax_raw / f2),
            unit_factor=f2,
            unit=u2,
            display_unit=uy,
            hist_centers=y_cents,
            hist_values=y_prof,
            hist_width=y_width,
            hist_unit_factor=y_prof_factor,
        ),
        weights=weights,
        bins=bins,
        ellipse_x=ell_x,
        ellipse_y=ell_y,
    )
