""" """

from copy import copy

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# For field legends
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .labels import mathlabel
from .statistics import slice_statistics, twiss_ellipse_points
from .units import nice_array, nice_scale_prefix, plottable_array

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
