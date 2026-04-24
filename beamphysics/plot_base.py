from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np

from .labels import mathlabel
from .statistics import twiss_ellipse_points
from .units import c_light, nice_array, nice_scale_prefix, plottable_array

if TYPE_CHECKING:
    from .particles import ParticleGroup

Limit = tuple[float | None, float | None]


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


@dataclass
class DensityPlotData:
    """Prepared data for a 1D density histogram."""

    key: str
    hist_centers: np.ndarray
    hist_values: np.ndarray
    hist_width: np.ndarray
    x_label: str
    y_label: str
    xlim: tuple[float, float] | None
    x_factor: float


def prepare_density_plot(
    particle_group: ParticleGroup,
    key: str = "x",
    bins: int | str | None = None,
    *,
    xlim: tuple[float, float] | None = None,
    nice: bool = True,
    tex: bool = True,
) -> DensityPlotData:
    """
    Prepare data for a 1D density plot.

    Computes the histogram, scales units, and formats labels.
    """
    if bins is None:
        n = len(particle_group)
        bins = int(n / 100)

    x, f1, p1, xmin, xmax = plottable_array(particle_group[key], nice=nice, lim=xlim)
    w = particle_group["weight"]
    u1 = particle_group.units(key).unitSymbol
    ux = p1 + u1

    x_label = mathlabel(key, units=ux, tex=tex)

    hist, bin_edges = np.histogram(x, bins=bins, weights=w)
    hist_x = bin_edges[:-1] + np.diff(bin_edges) / 2
    hist_width = np.diff(bin_edges)
    hist_y, hist_f, hist_prefix, _hist_xmin, _hist_xmax = plottable_array(
        hist / hist_width, nice=nice
    )

    if u1 == "s":
        _, hist_prefix = nice_scale_prefix(hist_f / f1)
        y_label = f"density ({hist_prefix}A)"
    else:
        y_label = f"{hist_prefix}C/{ux}"

    scaled_xlim = (xmin / f1, xmax / f1) if xlim else None

    return DensityPlotData(
        key=key,
        hist_centers=hist_x,
        hist_values=hist_y,
        hist_width=hist_width,
        x_label=x_label,
        y_label=y_label,
        xlim=scaled_xlim,
        x_factor=f1,
    )


@dataclass
class SliceCurve:
    """A single curve in a slice plot."""

    key: str
    label: str
    values: np.ndarray  # scaled


@dataclass
class SlicePlotData:
    """Prepared data for a slice statistics plot."""

    x: np.ndarray  # scaled slice positions
    x_label: str
    curves: list[SliceCurve]
    y_label: str
    density_values: np.ndarray  # scaled density for rhs overlay
    density_label: str
    xlim: tuple[float, float] | None  # scaled
    ylim: tuple[float, float] | None  # scaled
    x_factor: float
    y_factor: float


def prepare_slice_plot(
    particle_group: ParticleGroup,
    *keys: str,
    n_slice: int = 40,
    slice_key: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    nice: bool = True,
    tex: bool = True,
) -> SlicePlotData:
    """
    Prepare data for a slice statistics plot.

    Computes slice statistics, scales units, and formats labels.
    """
    if slice_key is None:
        if particle_group.in_t_coordinates:
            slice_key = "z"
        else:
            slice_key = "t"

    # Special case for delta_
    if slice_key.startswith("delta_"):
        slice_key = slice_key[6:]
        has_delta_prefix = True
    else:
        has_delta_prefix = False

    # Get all data
    x_key = "mean_" + slice_key
    slice_dat = particle_group.slice_statistics(
        *keys, n_slice=n_slice, slice_key=slice_key
    )
    slice_dat["density"] = slice_dat["charge"] / slice_dat["ptp_" + slice_key]

    # X-axis
    x = slice_dat["mean_" + slice_key]
    if has_delta_prefix:
        x -= particle_group["mean_" + slice_key]
        slice_key = "delta_" + slice_key  # restore

    x, f1, p1, xmin, xmax = plottable_array(x, nice=nice, lim=xlim)
    ux = p1 + str(particle_group.units(slice_key))

    # Y-axis - units check
    ulist = [particle_group.units(k).unitSymbol for k in keys]
    uy = ulist[0]
    if not all(u == uy for u in ulist):
        raise ValueError(f"Incompatible units: {ulist}")

    ymin = max(slice_dat[k].min() for k in keys)
    ymax = max(slice_dat[k].max() for k in keys)

    _, f2, p2, ymin, ymax = plottable_array(np.array([ymin, ymax]), nice=nice, lim=ylim)
    uy = p2 + uy

    # Curves
    curves = []
    for k in keys:
        label = mathlabel(k, units=uy, tex=tex)
        curves.append(SliceCurve(key=k, label=label, values=slice_dat[k] / f2))

    # Density on r.h.s
    y2, _, prey2, _, _ = plottable_array(slice_dat["density"], nice=nice, lim=None)

    y2_units = f"C/{particle_group.units(x_key)}"
    if y2_units == "C/s":
        y2_units = "A"
    y2_units = prey2 + y2_units

    x_label = mathlabel(slice_key, units=ux, tex=tex)
    y_label = mathlabel(*keys, units=uy, tex=tex)
    density_label = mathlabel("density", units=y2_units, tex=tex)

    scaled_xlim = (xmin / f1, xmax / f1) if xlim else None
    scaled_ylim = (ymin / f2, ymax / f2) if ylim else None

    return SlicePlotData(
        x=x,
        x_label=x_label,
        curves=curves,
        y_label=y_label,
        density_values=y2,
        density_label=density_label,
        xlim=scaled_xlim,
        ylim=scaled_ylim,
        x_factor=f1,
        y_factor=f2,
    )


@dataclass
class WakefieldPlotData:
    """Prepared data for a wakefield plot."""

    scatter_x: np.ndarray  # scaled x positions
    scatter_y: np.ndarray  # scaled kick values
    x_label: str
    y_label: str
    density: DensityPlotData  # for overlay
    xlim: tuple[float, float] | None
    ylim: tuple[float, float] | None


def prepare_wakefield_plot(
    particle_group: ParticleGroup,
    wake,
    key: str | None = None,
    nice: bool = True,
    tex: bool = True,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    bins: int | str | None = None,
) -> WakefieldPlotData:
    """
    Prepare data for a wakefield plot.

    Computes wake kicks, density histogram, and formats labels.
    """
    if key is None:
        if particle_group.in_t_coordinates:
            key = "delta_z/c"
        else:
            key = "delta_t"

    # Density data for overlay
    density_data = prepare_density_plot(
        particle_group, key=key, bins=bins, xlim=xlim, nice=nice, tex=tex
    )

    # Wake kicks
    if particle_group.in_t_coordinates:
        z = np.asarray(particle_group.z)
    else:
        z = -c_light * np.asarray(particle_group.t)
    kicks = wake.particle_kicks(z=z, weight=particle_group.weight)

    x_raw = particle_group[key]
    x, f1, p1, xmin, xmax = plottable_array(x_raw, nice=nice, lim=xlim)
    y, f2, p2, ymin, ymax = plottable_array(kicks, nice=nice, lim=ylim)

    ux = p1 + particle_group.units(key).unitSymbol
    x_label = mathlabel(key, units=ux, tex=tex)

    uy = p2 + "eV/m"
    y_label = mathlabel("W_z", units=uy, tex=tex)

    scaled_xlim = (xmin / f1, xmax / f1) if xlim else None
    scaled_ylim = (ymin / f2, ymax / f2) if ylim else None

    return WakefieldPlotData(
        scatter_x=x,
        scatter_y=y,
        x_label=x_label,
        y_label=y_label,
        density=density_data,
        xlim=scaled_xlim,
        ylim=scaled_ylim,
    )


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
