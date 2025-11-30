""" """

from copy import copy
from typing import Optional, Tuple, Dict, List, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm

# For field legends
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .labels import mathlabel
from .statistics import slice_statistics, twiss_ellipse_points
from .units import nice_array, nice_scale_prefix, plottable_array, pg_units

CMAP0 = copy(plt.get_cmap("viridis"))
CMAP0.set_under("white")

CMAP1 = copy(plt.get_cmap("plasma"))


def plt_histogram(a, weights=None, bins=40):
    """
    This produces the same plot as plt.hist

    For reference only

    Note that bins='auto', etc cannot be used if there are weights.
    """
    cnts, bins = np.histogram(a, weights=weights, bins=bins)
    plt.bar(bins[:-1] + np.diff(bins) / 2, cnts, np.diff(bins))


def slice_plot(
    particle_group,
    *keys,
    n_slice=40,
    slice_key=None,
    xlim=None,
    ylim=None,
    tex=True,
    nice=True,
    **kwargs,
):
    """
    Complete slice plotting routine. Will plot the density of the slice key on the right axis.

    Parameters
    ----------
    particle_group: ParticleGroup
        The object to plot

    keys: iterable of str
        Keys to calculate the statistics, e.g. `sigma_x`.

    n_slice: int, default = 40
        Number of slices

    slice_key: str, default = None
         The dimension to slice in. This is typically `t` or `z`.
         `delta_t`, etc. are also allowed.
         If None, `t` or `z` will automatically be determined.

    ylim: tuple, default = None
        Manual setting of the y-axis limits.

    tex: bool, default = True
        Use TEX for labels

    Returns
    -------
    fig: matplotlib.figure.Figure

    """

    # Allow a single key
    # if isinstance(keys, str):
    #
    #     keys = (keys, )

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
    y2_key = "density"

    # X-axis
    x = slice_dat["mean_" + slice_key]
    if has_delta_prefix:
        x -= particle_group["mean_" + slice_key]
        slice_key = "delta_" + slice_key  # restore

    x, f1, p1, xmin, xmax = plottable_array(x, nice=nice, lim=xlim)
    ux = p1 + str(particle_group.units(slice_key))

    # Y-axis

    # Units check
    ulist = [particle_group.units(k).unitSymbol for k in keys]
    uy = ulist[0]
    if not all([u == uy for u in ulist]):
        raise ValueError(f"Incompatible units: {ulist}")

    ymin = max([slice_dat[k].min() for k in keys])
    ymax = max([slice_dat[k].max() for k in keys])

    _, f2, p2, ymin, ymax = plottable_array(np.array([ymin, ymax]), nice=nice, lim=ylim)
    uy = p2 + uy

    # Form Figure
    fig, ax = plt.subplots(**kwargs)

    # Main curves
    if len(keys) == 1:
        color = "black"
    else:
        color = None

    for k in keys:
        label = mathlabel(k, units=uy, tex=tex)
        ax.plot(x, slice_dat[k] / f2, label=label, color=color)
    if len(keys) > 1:
        ax.legend()

    # Density on r.h.s
    y2, _, prey2, _, _ = plottable_array(slice_dat[y2_key], nice=nice, lim=None)

    # Convert to Amps if possible
    y2_units = f"C/{particle_group.units(x_key)}"
    if y2_units == "C/s":
        y2_units = "A"
    y2_units = prey2 + y2_units

    # Labels
    labelx = mathlabel(slice_key, units=ux, tex=tex)
    labely = mathlabel(*keys, units=uy, tex=tex)
    labely2 = mathlabel(y2_key, units=y2_units, tex=tex)

    ax.set_xlabel(labelx)
    ax.set_ylabel(labely)

    # rhs plot
    ax2 = ax.twinx()
    ax2.set_ylabel(labely2)
    ax2.fill_between(x, 0, y2, color="black", alpha=0.2)
    ax2.set_ylim(0, None)

    # Actual plot limits, considering scaling
    if xlim:
        ax.set_xlim(xmin / f1, xmax / f1)
    if ylim:
        ax.set_ylim(ymin / f2, ymax / f2)

    return fig


def density_plot(
    particle_group, key="x", bins=None, *, xlim=None, tex=True, nice=True, **kwargs
):
    """
    1D density plot. Also see: marginal_plot

    Example:

        density_plot(P, 'x', bins=100)

    """

    if not bins:
        n = len(particle_group)
        bins = int(n / 100)

    # Scale to nice units and get the factor, unit prefix
    x, f1, p1, xmin, xmax = plottable_array(particle_group[key], nice=nice, lim=xlim)
    w = particle_group["weight"]
    u1 = particle_group.units(key).unitSymbol
    ux = p1 + u1

    # mathtext label
    labelx = mathlabel(key, units=ux, tex=tex)

    fig, ax = plt.subplots(**kwargs)

    hist, bin_edges = np.histogram(x, bins=bins, weights=w)
    hist_x = bin_edges[:-1] + np.diff(bin_edges) / 2
    hist_width = np.diff(bin_edges)
    hist_y, hist_f, hist_prefix = nice_array(hist / hist_width)
    ax.bar(hist_x, hist_y, hist_width, color="grey")
    # Special label for C/s = A
    if u1 == "s":
        _, hist_prefix = nice_scale_prefix(hist_f / f1)
        ax.set_ylabel(f"{hist_prefix}A")
    else:
        ax.set_ylabel(f"{hist_prefix}C/{ux}")

    ax.set_xlabel(labelx)

    # Limits
    if xlim:
        ax.set_xlim(xmin / f1, xmax / f1)

    return fig


def marginal_plot(
    particle_group,
    key1="t",
    key2="p",
    bins=None,
    *,
    xlim=None,
    ylim=None,
    tex=True,
    nice=True,
    ellipse=False,
    **kwargs,
):
    """
    Density plot and projections

    Example:

        marginal_plot(P, 't', 'energy', bins=200)


    Parameters
    ----------
    particle_group: ParticleGroup
        The object to plot

    key1: str, default = 't'
        Key to bin on the x-axis

    key2: str, default = 'p'
        Key to bin on the y-axis

    bins: int, default = None
       Number of bins. If None, this will use a heuristic: bins = sqrt(n_particle/4)

    xlim: tuple, default = None
        Manual setting of the x-axis limits.

    ylim: tuple, default = None
        Manual setting of the y-axis limits.

    tex: bool, default = True
        Use TEX for labels

    nice: bool, default = True

    ellipse: bool, default = True
        If True, plot an ellipse representing the
        2x2 sigma matrix

    Returns
    -------
    fig: matplotlib.figure.Figure


    """
    if not bins:
        n = len(particle_group)
        bins = int(np.sqrt(n / 4))

    # Scale to nice units and get the factor, unit prefix
    x = particle_group[key1]
    y = particle_group[key2]

    if len(x) == 1:
        bins = 100

        if xlim is None:
            (x0,) = x
            if np.isclose(x0, 0.0):
                xlim = (-1, 1)
            else:
                xlim = tuple(sorted((0.9 * x0, 1.1 * x0)))
        if ylim is None:
            (y0,) = y
            if np.isclose(y0, 0.0):
                ylim = (-1, 1)
            else:
                ylim = tuple(sorted((0.9 * y0, 1.1 * y0)))

    # Form nice arrays
    x, f1, p1, xmin, xmax = plottable_array(x, nice=nice, lim=xlim)
    y, f2, p2, ymin, ymax = plottable_array(y, nice=nice, lim=ylim)

    w = particle_group["weight"]

    u1 = particle_group.units(key1).unitSymbol
    u2 = particle_group.units(key2).unitSymbol
    ux = p1 + u1
    uy = p2 + u2

    # Handle labels.
    labelx = mathlabel(key1, units=ux, tex=tex)
    labely = mathlabel(key2, units=uy, tex=tex)

    fig = plt.figure(**kwargs)
    if np.all(np.isnan(x)):
        fig.text(0.5, 0.5, f"{key1} is all NaN", ha="center", va="center")
        return fig
    if np.all(np.isnan(y)):
        fig.text(0.5, 0.5, f"{key2} is all NaN", ha="center", va="center")
        return fig

    gs = GridSpec(4, 4)

    ax_joint = fig.add_subplot(gs[1:4, 0:3])
    ax_marg_x = fig.add_subplot(gs[0, 0:3])
    ax_marg_y = fig.add_subplot(gs[1:4, 3])
    # ax_info = fig.add_subplot(gs[0, 3:4])
    # ax_info.table(cellText=['a'])

    # Main plot
    # Proper weighting
    if len(x) == 1:
        ax_joint.scatter(x, y)
    else:
        ax_joint.hexbin(
            x,
            y,
            C=w,
            reduce_C_function=np.sum,
            gridsize=bins,
            cmap=CMAP0,
            vmin=1e-20,
        )

    if ellipse:
        sigma_mat2 = particle_group.cov(key1, key2)
        x_ellipse, y_ellipse = twiss_ellipse_points(sigma_mat2)
        x_ellipse += particle_group.avg(key1)
        y_ellipse += particle_group.avg(key2)
        ax_joint.plot(x_ellipse / f1, y_ellipse / f2, color="red")

    # Manual histogramming version
    # H, xedges, yedges = np.histogram2d(x, y, weights=w, bins=bins)
    # extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    # ax_joint.imshow(H.T, cmap=cmap, vmin=1e-16, origin='lower', extent=extent, aspect='auto')

    # Top histogram
    # Old method:
    # dx = x.ptp()/bins
    # ax_marg_x.hist(x, weights=w/dx/f1, bins=bins, color='gray')
    hist, bin_edges = np.histogram(x, bins=bins, weights=w)
    hist_x = bin_edges[:-1] + np.diff(bin_edges) / 2
    hist_width = np.diff(bin_edges)
    hist_y, hist_f, hist_prefix = nice_array(hist / hist_width)
    ax_marg_x.bar(hist_x, hist_y, hist_width, color="gray")
    # Special label for C/s = A
    if u1 == "s":
        _, hist_prefix = nice_scale_prefix(hist_f / f1)
        ax_marg_x.set_ylabel(f"{hist_prefix}A")
    else:
        ax_marg_x.set_ylabel(f"{hist_prefix}" + mathlabel(f"C/{ux}"))  # Always use tex

    # Side histogram
    # Old method:
    # dy = y.ptp()/bins
    # ax_marg_y.hist(y, orientation="horizontal", weights=w/dy, bins=bins, color='gray')
    hist, bin_edges = np.histogram(y, bins=bins, weights=w)
    hist_x = bin_edges[:-1] + np.diff(bin_edges) / 2
    hist_width = np.diff(bin_edges)
    hist_y, hist_f, hist_prefix = nice_array(hist / hist_width)
    ax_marg_y.barh(hist_x, hist_y, hist_width, color="gray")
    ax_marg_y.set_xlabel(f"{hist_prefix}" + mathlabel(f"C/{uy}"))  # Always use tex

    # Turn off tick labels on marginals
    plt.setp(ax_marg_x.get_xticklabels(), visible=False)
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)

    # Set labels on joint
    ax_joint.set_xlabel(labelx)
    ax_joint.set_ylabel(labely)

    # Actual plot limits, considering scaling
    if xlim:
        ax_joint.set_xlim(xmin / f1, xmax / f1)
        ax_marg_x.set_xlim(xmin / f1, xmax / f1)

    if ylim:
        ax_joint.set_ylim(ymin / f2, ymax / f2)
        ax_marg_y.set_ylim(ymin / f2, ymax / f2)

    return fig


def density_and_slice_plot(
    particle_group,
    key1="t",
    key2="p",
    stat_keys=["norm_emit_x", "norm_emit_y"],
    bins=100,
    n_slice=30,
    tex=True,
):
    """
    Density plot and projections

    Example:

        marginal_plot(P, 't', 'energy', bins=200)

    """

    # Scale to nice units and get the factor, unit prefix
    x, f1, p1, xmin, xmax = plottable_array(particle_group[key1])
    y, f2, p2, ymin, ymax = plottable_array(particle_group[key2])
    w = particle_group["weight"]

    u1 = particle_group.units(key1).unitSymbol
    u2 = particle_group.units(key2).unitSymbol
    ux = p1 + u1
    uy = p2 + u2

    labelx = mathlabel(key1, units=ux, tex=tex)
    labely = mathlabel(key2, units=uy, tex=tex)

    fig, ax = plt.subplots()

    ax.set_xlabel(labelx)
    ax.set_ylabel(labely)

    # Proper weighting
    # ax_joint.hexbin(x, y, C=w, reduce_C_function=np.sum, gridsize=bins, cmap=cmap, vmin=1e-15)

    # Manual histogramming version
    H, xedges, yedges = np.histogram2d(x, y, weights=w, bins=bins)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    ax.imshow(H.T, cmap=CMAP0, vmin=1e-16, origin="lower", extent=extent, aspect="auto")

    # Slice data
    slice_dat = slice_statistics(
        particle_group,
        n_slice=n_slice,
        slice_key=key1,
        keys=stat_keys + ["ptp_" + key1, "mean_" + key1, "charge"],
    )

    slice_dat["density"] = slice_dat["charge"] / slice_dat["ptp_" + key1]

    #
    ax2 = ax.twinx()
    # ax2.set_ylim(0, 1e-6)
    x2 = slice_dat["mean_" + key1] / f1
    ulist = [particle_group.units(k).unitSymbol for k in stat_keys]

    max2 = max([np.ptp(slice_dat[k]) for k in stat_keys])

    f3, p3 = nice_scale_prefix(max2)

    u2 = ulist[0]
    assert all([u == u2 for u in ulist])
    u2 = p3 + u2
    labely2 = mathlabel(*stat_keys, units=u2, tex=tex)
    for k in stat_keys:
        label = mathlabel(k, units=u2, tex=tex)
        ax2.plot(x2, slice_dat[k] / f3, label=label)
    ax2.legend()
    ax2.set_ylabel(labely2)
    ax2.set_ylim(bottom=0)

    # Add density
    y2 = slice_dat["density"]
    y2 = y2 * max2 / y2.max() / f3 / 2
    ax2.fill_between(x2, 0, y2, color="black", alpha=0.1)


# -------------------------------------
# -------------------------------------
# Fields


def _symmetrize_data(data, sign=1, axis=0):
    """
    Symmetrizes a given array along a specified axis.

    This function creates a symmetric version of the input data by flipping it along
    the specified axis and concatenating the flipped data with the original data.
    The flipped data is multiplied by the specified `sign` value. The axis value
    corresponding to the center (e.g., r=0) is included only once.
    """
    flipped_data = np.flip(data, axis=axis)[:-1] * sign
    return np.concatenate((flipped_data, data), axis=axis)


def plot_fieldmesh_cylindrical_2d(
    fm,
    component=None,
    time=None,
    axes=None,
    aspect="auto",
    cmap=None,
    return_figure=False,
    stream=False,
    mirror=None,
    density=1,
    linewidth=1,
    arrowsize=1,
    **kwargs,
):
    """
    Plots a fieldmesh component


    """

    assert fm.geometry == "cylindrical"

    if mirror not in (None, "r"):
        raise ValueError("mirror must be None or 'r'")

    if component is None:
        if fm.is_pure_magnetic:
            component = "B"
        else:
            component = "E"

    if not axes:
        fig, ax = plt.subplots(**kwargs)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    if not cmap:
        cmap = CMAP1

    unit = fm.units(component)

    xmin, _, zmin = fm.mins
    xmax, _, zmax = fm.maxs

    # plt.imshow on [r, z] will put z on the horizontal axis.

    xlabel = r"$z$ (m)"
    ylabel = r"$r$ (m)"

    dat = fm[component][:, 0, :]
    dat = np.real_if_close(dat)

    if mirror == "r":
        if xmin != 0:
            raise ValueError("mirror='r' only available when r=0 is in the data")
        if component in ("Br", "Er"):
            sign = -1
        else:
            sign = 1

        xmin = -xmax
        dat = _symmetrize_data(dat, sign)

    dmin = dat.min()
    dmax = dat.max()
    extent = [zmin, zmax, xmin, xmax]

    ax.set_aspect(aspect)

    if stream:
        if component not in ("E", "B"):
            raise ValueError(f"{component} for stream plot must be 'E' or 'B' ")

        fx = np.real(fm[component + "z"][:, 0, :])
        fy = np.real(fm[component + "r"][:, 0, :])
        x = fm.coord_vec("z")
        y = fm.coord_vec("r")

        if mirror == "r":
            y = _symmetrize_data(y, sign=-1)
            fx = _symmetrize_data(fx, sign=1)
            fy = _symmetrize_data(fy, sign=-1)

        ax.streamplot(
            x,
            y,
            fx,
            fy,
            color=dat,
            cmap=cmap,
            density=density,
            linewidth=linewidth,
            arrowsize=arrowsize,
        )

        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())

    else:
        # Need to flip for image
        ax.imshow(dat, extent=extent, cmap=cmap, aspect=aspect, origin="lower")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Add legend
    llabel = mathlabel(component, units=unit)

    norm = matplotlib.colors.Normalize(vmin=dmin, vmax=dmax)
    fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax,
        orientation="vertical",
        label=llabel,
    )

    if return_figure:
        return fig


# Intended to be used as a method in FieldMesh
def plot_fieldmesh_cylindrical_1d(fm, axes=None, return_figure=False, **kwargs):
    """

    Plots the on-axis Ez and/or Bz from a FieldMesh
    with cylindrical geometry.

    Parameters
    ----------
    axes: matplotlib axes object, default None

    return_figure: bool, default False

    Returns
    -------
    fig, optional
        if return_figure, returns matplotlib Figure instance
        for further modifications.

    """

    if not axes:
        fig, ax = plt.subplots(**kwargs)

    has_Ez = "electricField/z" in fm.components
    has_Bz = "magneticField/z" in fm.components

    Bzlabel = r"$B_z$ (T)"
    z0 = fm.coord_vec("z")

    ylabel = None
    if has_Ez:
        Ez0 = fm["Ez"][0, 0, :]
        ax.plot(z0, Ez0, color="black", label=r"E_{z0}")
        ylabel = r"$E_z$ (V/m)"

    if has_Bz:
        Bz0 = fm["Bz"][0, 0, :]
        if has_Ez:
            ax2 = ax.twinx()
            ax2.plot(z0, Bz0, color="blue", label=r"$B_{z0}$")
            ax2.set_ylabel(Bzlabel)
        else:
            ax.plot(z0, Bz0, color="blue", label=r"$B_{z0}$")
            ylabel = Bzlabel

    if has_Ez and has_Bz:
        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")

    ax.set_xlabel(r"$z$ (m)")
    ax.set_ylabel(ylabel)

    if return_figure:
        return fig


def plot_fieldmesh_rectangular_1d(
    fm, field_component, axes=None, return_figure=False, **kwargs
):
    """

    Plots the on-axis field components from a FieldMesh
    with rectangular geometry.

    Parameters
    ----------
    axes: matplotlib axes object, default None

    return_figure: bool, default False

    Returns
    -------
    fig, optional
        if return_figure, returns matplotlib Figure instance
        for further modifications.

    """

    # Here only to plot a single field component
    if "color" in kwargs:
        color = kwargs["color"]
        del kwargs["color"]
    else:
        color = None

    if not axes:
        fig, axes = plt.subplots(**kwargs)

    # Use recursion to plot multiple field components
    if isinstance(field_component, list):
        for ii, fc in enumerate(field_component):
            plot_fieldmesh_rectangular_1d(fm, fc, axes=axes, **kwargs)

        if return_figure:
            return fig
        else:
            return

    assert field_component in [
        "Ex",
        "Ey",
        "Ez",
        "Bx",
        "By",
        "Bz",
    ], f"Unknown field component: {field_component}"

    fieldmesh_component = field_component.replace("B", "magneticField/").replace(
        "E", "electricField/"
    )

    assert (
        fieldmesh_component in fm.components
    ), f"FieldMesh was missing field component: {field_component}"

    field_unit = fm.units(fieldmesh_component)

    ylabel = r"$" + field_component[0] + "_" + field_component[1] + rf"$ ({field_unit})"
    label = r"$" + field_component[0] + "_" + field_component[1] + "(x=y=0, z)$"

    z, field0 = fm.axis_values("z", field_component)

    if np.all(np.isclose(field0.imag, 0)):  # Close to real
        if not fm.is_static:
            label = r"$\Re$[" + label + "]"

        axes.plot(z, np.real(field0), label=label, color=color)

    elif np.all(np.isclose(field0.real, 0)):  # Close to imag
        if not fm.is_static:
            label = r"$\Im$[" + label + "]"

        axes.plot(z, np.imag(field0), label=label, color=color)

    else:  # Complex
        axes.plot(z, np.real(field0), label="Re[" + label + "]", color=color)
        axes.plot(z, np.imag(field0), label="Im[" + label + "]")

    axes.set_xlabel("z (m)")
    axes.set_ylabel(ylabel)
    axes.legend()

    if return_figure:
        return fig


def plot_fieldmesh_rectangular_2d(
    fm,
    component=None,
    time=None,
    axes=None,
    aspect="auto",
    cmap=None,
    return_figure=False,
    nice=True,
    **kwargs,
):
    """

    Plots a field component evaluated on a plane defined by coordinate [x, y, or z] = coordinate_value
    from a FieldMesh with rectangular geometry.

    Parameters
    ----------
    axes: matplotlib axes object, default None

    return_figure: bool, default False

    Returns
    -------
    fig, optional
        if return_figure, returns matplotlib Figure instance
        for further modifications.

    """

    assert fm.geometry == "rectangular"

    valid_coordinates = set(fm.axis_labels)
    coordinate_value = None

    # Identify which coordinate is provided in kwargs
    for key in valid_coordinates:
        if key in kwargs:
            coordinate = key
            coordinate_value = kwargs.pop(key)
            break
    else:
        # just use y=0 plane
        coordinate = "y"
        coordinate_value = 0

    if not axes:
        fig, axes = plt.subplots(**kwargs)

    if not cmap:
        cmap = CMAP1

    divider = make_axes_locatable(axes)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    assert component in (
        "Ex",
        "Ey",
        "Ez",
        "Bx",
        "By",
        "Bz",
    ), f"Unknown field component: {component}"

    x, y, z = fm.coord_vec("x"), fm.coord_vec("y"), fm.coord_vec("z")

    interpolator = fm.interpolator(component)

    unit = fm.units(component)

    xmin, ymin, zmin = fm.mins
    xmax, ymax, zmax = fm.maxs

    # Prefer z to be on x-axis for accelerators
    if coordinate == "x":
        extent = [zmin, zmax, ymin, ymax]
        xlabel = r"$z$ (m)"
        ylabel = r"$y$ (m)"

        # points0 = np.array([[coordinate_value, y_val, z_val] for y_val in y for z_val in z])
        a, b = np.meshgrid(y, z, indexing="ij")
        points = np.column_stack(
            [np.full(a.size, coordinate_value), a.ravel(), b.ravel()]
        )
        # assert np.allclose(points0, points)
        interpolated_values = interpolator(points)
        field_2d = interpolated_values.reshape(len(y), len(z))

    elif coordinate == "y":
        extent = [zmin, zmax, xmin, xmax]
        xlabel = r"$z$ (m)"
        ylabel = r"$x$ (m)"

        # points0 = np.array([[x_val, coordinate_value, z_val] for x_val in x for z_val in z])
        a, b = np.meshgrid(x, z, indexing="ij")
        points = np.column_stack(
            [a.ravel(), np.full(a.size, coordinate_value), b.ravel()]
        )
        # assert np.allclose(points0, points)
        interpolated_values = interpolator(points)
        field_2d = interpolated_values.reshape(len(x), len(z))

    elif coordinate == "z":
        extent = [xmin, xmax, ymin, ymax]
        xlabel = r"$x$ (m)"
        ylabel = r"$y$ (m)"

        # Leave here to check
        # points0 = np.array([[x_val, y_val, coordinate_value] for x_val in x for y_val in y])
        a, b = np.meshgrid(x, y, indexing="ij")
        points = np.column_stack(
            [a.ravel(), b.ravel(), np.full(a.size, coordinate_value)]
        )
        # assert np.allclose(points0, points)
        interpolated_values = interpolator(points)
        field_2d = interpolated_values.reshape(len(x), len(y))

    if nice:
        field_2d, _, prefix = nice_array(field_2d)
    else:
        prefix = ""

    dmin = field_2d.min()
    dmax = field_2d.max()

    axes.set_aspect(aspect)

    plane = f"{coordinate} = {coordinate_value}"

    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)

    # Add legend
    llabel = (
        r"$"
        + f"{component[0]}_{component[1]}({plane})"
        + rf"$ ({prefix}{unit.unitSymbol})"
    )

    if np.all(np.isclose(np.imag(field_2d), 0)):  # Close to real
        axes.imshow(
            np.real(field_2d), extent=extent, origin="lower", aspect="auto", cmap=cmap
        )
        dmin = np.real(field_2d.min())
        dmax = np.real(field_2d.max())

        if not fm.is_static:
            llabel = "Re" + llabel + "]"

    elif np.all(np.isclose(np.real(field_2d), 0)):  # Close to imag
        axes.imshow(
            np.imag(field_2d), extent=extent, origin="lower", aspect="auto", cmap=cmap
        )
        dmin = np.imag(field_2d.min())
        dmax = np.imag(field_2d.max())

        if not fm.is_static:
            llabel = "Im" + llabel

    else:
        raise ValueError("Complex components not supported")

    norm = matplotlib.colors.Normalize(vmin=dmin, vmax=dmax)
    fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax,
        orientation="vertical",
        label=llabel,
    )


def plot_1d_density(
    x: Union[str, np.ndarray],
    y: Union[str, np.ndarray],
    x_name: str = "",
    y_name: Optional[str] = None,
    x_units: Optional[str] = None,
    y_units: Optional[str] = None,
    figsize: Tuple[float, float] = (6, 4),
    log_scale_y: bool = False,
    show_cdf: bool = False,
    cdf_label: str = "CDF",
    cdf_style: Optional[Dict[str, Union[str, float]]] = None,
    kind: str = "bar",
    plot_style: Optional[Dict[str, Union[str, float]]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = (0, None),
    ax: Optional[plt.Axes] = None,
    nice: bool = True,
    auto_label: bool = False,
    tex: bool = True,
    data: Optional[Dict[str, np.ndarray]] = None,
    return_axes: bool = False,
) -> Optional[Tuple[plt.Figure, Dict[str, plt.Axes]]]:
    """
    Plot a 1D density distribution with optional cumulative distribution function (CDF).

    Parameters
    ----------
    x : np.ndarray or str
        1D array representing the x-axis positions (bin centers), or a string key
        to index into the `data` dict.
    y : np.ndarray or str
        1D array representing the density data (y-axis values), or a string key
        to index into the `data` dict.
    x_name : str, default=""
        Label for the X-axis. If not provided and `x` is a string, will use `x` as the label.
    y_name : str, optional
        Label for the Y-axis (density). If None (default), will use "Density" as the label,
        or the key name if `y` is a string key.
    x_units : str, optional
        Units for the X-axis, displayed in parentheses next to the label.
    y_units : str, optional
        Units for the Y-axis (density), displayed in parentheses next to the label.
    figsize : tuple of float, default=(6, 4)
        Figure size for the plot.
    log_scale_y : bool, default=False
        If True, use a log scale for the density Y-axis.
    show_cdf : bool, default=False
        If True, plot the cumulative distribution function on a secondary Y-axis.
    cdf_label : str, default="CDF"
        Label for the CDF axis.
    cdf_style : dict, optional
        Style options for the CDF line plot (e.g., {"color": "red", "linewidth": 2}).
        Default is {"color": "blue", "linewidth": 2}.
    kind : str, default="bar"
        Type of plot for the density: "bar" for bar chart or "line" for line plot.
    plot_style : dict, optional
        Style options passed to the plot function. For bar plots, options like
        {"color": "gray", "alpha": 0.7}. For line plots, options like
        {"color": "blue", "linewidth": 2}.
    xlim : tuple of float, optional
        Limits for the X-axis in the form (xmin, xmax).
    ylim : tuple of float, optional, default=(0, None)
        Limits for the Y-axis (density) in the form (ymin, ymax).
        Default starts at 0, which is typical for density distributions.
        Set to None for automatic limits based on data.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes object to plot on. If None, a new figure and axes will be created.
    nice : bool, default=True
        If True, scale the arrays to nice units with appropriate SI prefixes.
    auto_label : bool, default=False
        If True, automatically generate labels and units from data keys using
        `texlabel` and `pg_units`. Only works when `x` and `y` are strings.
    tex : bool, default=True
        If True, use TeX formatting for labels (via `mathlabel`). Only applies when
        `auto_label=True`.
    data : dict, optional
        Dictionary containing data arrays. If provided, `x` and `y` should be string
        keys to index into this dict. This follows matplotlib's convention.
    return_axes : bool, default=False
        If True, return the figure and axis objects for further customization.

    Returns
    -------
    None or (plt.Figure, dict)
        Returns None if `return_axes` is False. Otherwise, returns the figure and
        a dictionary of axis objects.

    Examples
    --------
    Direct array input:

        >>> plot_1d_density(x_array, y_array, x_name="Position", y_name="Density")

    Using data dict (matplotlib style):

        >>> data = {"position": x_array, "density": y_array}
        >>> plot_1d_density("position", "density", data=data)

    Using auto_label with ParticleGroup-style keys:

        >>> data = {"t": time_array, "norm_emit_x": emittance_array}
        >>> plot_1d_density("t", "norm_emit_x", data=data, auto_label=True)
        # Will automatically use TeX labels and proper units
    """
    # Handle data dict indexing (matplotlib pattern)
    # Handle data dict indexing (matplotlib pattern)
    x_key = None
    y_key = None

    if isinstance(x, str):
        if data is None:
            raise ValueError("If `x` is a string, `data` dict must be provided")
        x_key = x
        x = np.asarray(data[x_key])
    else:
        x = np.asarray(x)

    if isinstance(y, str):
        if data is None:
            raise ValueError("If `y` is a string, `data` dict must be provided")
        y_key = y
        y = np.asarray(data[y_key])
    else:
        y = np.asarray(y)

    # Use the key as the label if not explicitly provided
    # Note: x_name and y_name are function parameters with defaults
    if x_key is not None:
        if x_name == "":  # noqa: F821
            x_name = x_key

    if y_key is not None:
        if y_name is None:  # noqa: F821
            y_name = y_key

    # Set default y_name if still None
    if y_name is None:
        y_name = "Density"

    # Validate array lengths match
    if len(x) != len(y):
        raise ValueError(
            f"Length mismatch: x has {len(x)} elements, y has {len(y)} elements"
        )

    # Warn if auto_label is True but we don't have string keys
    if auto_label and not (x_key or y_key):
        import warnings

        warnings.warn(
            "auto_label=True but x and y are not string keys. "
            "auto_label only works when x/y are passed as strings with a data dict.",
            UserWarning,
            stacklevel=2,
        )

    # Auto-generate labels and units from keys if requested
    if auto_label:
        if x_key and x_units is None:
            try:
                x_units_obj = pg_units(x_key)
                x_units = x_units_obj.unitSymbol
            except (ValueError, KeyError):
                pass  # Keep x_units as None if lookup fails

        if y_key and y_units is None:
            try:
                y_units_obj = pg_units(y_key)
                y_units = y_units_obj.unitSymbol
            except (ValueError, KeyError):
                pass  # Keep y_units as None if lookup fails

    # Store base units before adding prefixes (needed for CDF)
    x_units_base = str(x_units) if x_units else None
    y_units_base = str(y_units) if y_units else None

    # Form nice arrays
    x, f1, p1, x_min, x_max = plottable_array(x, nice=nice, lim=xlim)
    y, f2, p2, y_min, y_max = plottable_array(y, nice=nice, lim=ylim)

    # Update units with prefixes
    if x_units:
        x_units = p1 + str(x_units)
    else:
        x_units = p1 if p1 else None

    if y_units:
        y_units = p2 + str(y_units)
    else:
        y_units = p2 if p2 else None

    # Compute bar widths from x spacing for continuous bars
    if len(x) > 1:
        # Width is distance to next point
        widths = np.diff(x)
        # For the last bar, use the same width as the second-to-last
        widths = np.append(widths, widths[-1])
    else:
        widths = 1.0

    # Create figure and axis
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Default plot style
    if plot_style is None:
        if kind == "bar":
            plot_style = {"color": "gray", "alpha": 0.7}
        else:  # line
            plot_style = {"color": "blue", "linewidth": 2}

    # Plot density
    if kind == "bar":
        ax.bar(x, y, width=widths, align="edge", **plot_style)
    elif kind == "line":
        ax.plot(x, y, **plot_style)
    else:
        raise ValueError(f"kind must be 'bar' or 'line', got '{kind}'")

    # Set labels
    if auto_label and x_key:
        ax.set_xlabel(mathlabel(x_key, units=x_units, tex=tex))
    else:
        ax.set_xlabel(f"{x_name} ({x_units})" if x_units else x_name)

    if auto_label and y_key:
        ax.set_ylabel(mathlabel(y_key, units=y_units, tex=tex))
    else:
        ax.set_ylabel(f"{y_name} ({y_units})" if y_units else y_name)

    # Apply log scale if requested
    if log_scale_y:
        ax.set_yscale("log")

    # Set limits using values from plottable_array
    if xlim is not None:
        ax.set_xlim(x_min / f1, x_max / f1)
    if ylim is not None:
        ax.set_ylim(y_min / f2, y_max / f2)

    # Dictionary to hold axes
    axes = {"main": ax}

    # Add CDF on secondary axis if requested
    if show_cdf:
        ax_cdf = ax.twinx()

        # Compute cumulative sum (CDF)
        # Note: cdf has units of y * widths (density * position)
        cdf = np.cumsum(y * widths) * f1 * f2

        cdf_scaled, _, cdf_prefix, _, _ = plottable_array(cdf, nice=nice)

        # Default CDF style
        if cdf_style is None:
            cdf_style = {"color": "blue", "linewidth": 2}

        ax_cdf.plot(x, cdf_scaled, label=cdf_label, **cdf_style)

        # Label with appropriate units
        # CDF has units of y_base * x_base with combined prefix
        if x_units_base and y_units_base:
            try:
                from pmd_beamphysics.units import pmd_unit

                cdf_units_base = (
                    pmd_unit(y_units_base) * pmd_unit(x_units_base)
                ).simplify()
                cdf_units_str = cdf_prefix + str(cdf_units_base)
                ax_cdf.set_ylabel(f"{cdf_label} ({cdf_units_str})")
            except (ValueError, KeyError, TypeError) as e:
                # If unit parsing/multiplication fails, fall back to simpler label
                import warnings

                warnings.warn(
                    f"Could not compute CDF units: {e}. Using label without units.",
                    UserWarning,
                    stacklevel=2,
                )
                ax_cdf.set_ylabel(cdf_label)
        elif cdf_prefix:
            ax_cdf.set_ylabel(f"{cdf_label} ({cdf_prefix})")
        else:
            ax_cdf.set_ylabel(cdf_label)

        ax_cdf.set_ylim(0, cdf_scaled.max())
        ax_cdf.legend(loc="upper left")

        axes["cdf"] = ax_cdf

    # Return axes if requested
    if return_axes:
        return fig, axes


def plot_2d_density_with_marginals(
    data: np.ndarray,
    dx: Optional[float] = 1,
    dy: Optional[float] = 1,
    xmin: Optional[float] = None,
    ymin: Optional[float] = None,
    x_name: str = "",
    y_name: str = "",
    z_name: str = "",
    x_units: Optional[str] = None,
    y_units: Optional[str] = None,
    z_units: Optional[str] = None,
    cmap: str = "inferno",
    figsize: Tuple[float, float] = (5, 5),
    log_scale_z: bool = False,
    log_scale_marginals: bool = False,
    marginal_titles: Tuple[Optional[str], Optional[str]] = (None, None),
    highlight_regions: Optional[
        List[Dict[str, Union[float, Tuple[float, float]]]]
    ] = None,
    marginal_style: Optional[Dict[str, Union[str, float]]] = None,
    show_stats: bool = False,
    show_colorbar: bool = True,
    xlim: Tuple[float, float] = None,
    ylim: Tuple[float, float] = None,
    vmin: Optional[float] = None,
    vcenter: Optional[float] = None,
    vmax: Optional[float] = None,
    aspect: Optional[str] = "auto",
    return_axes: bool = False,
) -> Optional[Tuple[plt.Figure, Dict[str, plt.Axes]]]:
    """
    Basic plot for a 2D density map with marginal histograms.

    Parameters
    ----------
    data : np.ndarray
        2D array representing the binned density data.
    extent : tuple of float
        The spatial range of the data in the form (xmin, xmax, ymin, ymax).
    x_name : str
        Label for the X-axis.
    y_name : str
        Label for the Y-axis.
    z_name : str
        Label for the Z-axis (density).
    x_units : str, optional
        Units for the X-axis, displayed in parentheses next to the label.
    y_units : str, optional
        Units for the Y-axis, displayed in parentheses next to the label.
    z_units : str, optional
        Units for the Z-axis (density), displayed in parentheses next to the label.
    cmap : str, default="viridis"
        Colormap to use for the density plot.
    figsize : tuple of float, default=(8, 8)
        Figure size for the plot.
    log_scale_z : bool, default=False
        If True, apply a log scale to the density data (Z-axis) using a LogNorm.
    log_scale_marginals : bool, default=False
        If True, use a log scale for the marginal histograms.
    marginal_titles : tuple of str, optional
        Titles for the X and Y marginal plots in the form (x_title, y_title).
    annotations : list of dict, optional
        List of annotations for the main density plot. Each dict should include:
        {"x": float, "y": float, "text": str}.
    marginal_style : dict, optional
        Style for the marginal bars (e.g., {"color": "gray", "alpha": 0.7}).
    show_stats : bool, default=False
        If True, display basic statistics (mean, median, std) on the plot.
    return_axes : bool, default=False
        If True, return the figure and axis objects for further customization.

    Returns
    -------
    None or (plt.Figure, dict)
        Returns None if `return_axes` is False. Otherwise, returns the figure and
        a dictionary of axis objects.
    """

    # Compute bin edges from extent and data shape
    nx, ny = data.shape

    # Coordinates represent centers of pixels
    if xmin is None:
        # Put 0 in the center
        xmin = -((nx - 1) * dx) / 2
    if ymin is None:
        # Put 0 in the center
        ymin = -((ny - 1) * dy) / 2

    xmax = xmin + (nx - 1) * dx
    ymax = ymin + (ny - 1) * dy

    xvec = np.linspace(xmin, xmax, nx)
    yvec = np.linspace(ymin, ymax, ny)

    # Compute marginal histograms
    x_marginal = np.sum(data, axis=1) * dy  # sum over y
    y_marginal = np.sum(data, axis=0) * dx

    # Define normalization for the density plot
    # Set defaults for vmin and vmax based on the data
    vmin = vmin if vmin is not None else np.min(data)
    vmax = vmax if vmax is not None else np.max(data)

    # Choose normalization
    if log_scale_z:
        norm = LogNorm(vmin=vmin, vmax=vmax)
    elif vcenter is None:
        norm = Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

    # Create figure and GridSpec
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(6, 6, figure=fig, wspace=0.05, hspace=0.05)

    # Main density plot
    ax_main = fig.add_subplot(gs[1:, :-1])
    extent = (xmin - dx / 2, xmax + dx / 2, ymin - dy / 2, ymax + dy / 2)
    density_plot = ax_main.imshow(
        data.T,
        origin="lower",
        # Extent is the full extent of all pixels, so add half widths
        extent=extent,
        cmap=cmap,
        aspect=aspect,
    )
    ax_main.set_xlabel(f"{x_name} ({x_units})" if x_units else x_name)
    ax_main.set_ylabel(f"{y_name} ({y_units})" if y_units else y_name)
    density_plot.set_norm(norm)

    # Top marginal
    ax_top = fig.add_subplot(gs[0, :-1], sharex=ax_main)
    bar_style = marginal_style if marginal_style else {"color": "gray"}
    ax_top.bar(xvec, x_marginal, width=dx, align="center", **bar_style)
    # ax_top.set_ylabel(marginal_titles[0] if marginal_titles[0] else "Density")
    if z_units and y_units:
        ax_top.set_ylabel(f"{z_units}" + r"$\cdot$" + f"{y_units}")

    if log_scale_marginals:
        ax_top.set_yscale("log")

    # Right marginal
    ax_right = fig.add_subplot(gs[1:, -1], sharey=ax_main)
    ax_right.barh(yvec, y_marginal, height=dy, align="edge", **bar_style)
    # ax_right.set_xlabel(marginal_titles[1] if marginal_titles[1] else "Density")
    if z_units and x_units:
        ax_right.set_xlabel(f"{z_units}" + r"$\cdot$" + f"{x_units}")
    if log_scale_marginals:
        ax_right.set_xscale("log")

    # Return axes if requested
    axes = {"main": ax_main, "top": ax_top, "right": ax_right}

    # Add color bar
    if show_colorbar:
        cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
        fig.colorbar(density_plot, cax=cbar_ax, orientation="vertical")
        cbar_ax.set_ylabel(f"{z_name} ({z_units})" if z_units else z_name)
        axes["cbar"] = cbar_ax

    # Turn off tick labels on marginals
    plt.setp(ax_top.get_xticklabels(), visible=False)
    plt.setp(ax_right.get_yticklabels(), visible=False)

    if xlim is None:
        xlim = (extent[0], extent[1])
    ax_main.set_xlim(xlim)
    ax_top.set_xlim(xlim)

    if ylim is None:
        ylim = (extent[2], extent[3])
    ax_main.set_ylim(ylim)
    ax_right.set_ylim(ylim)

    # Return axes if requested
    if return_axes:
        return fig, axes
