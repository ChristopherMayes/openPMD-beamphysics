from __future__ import annotations


from bokeh.core.enums import SizingModeType
import numpy as np
from bokeh.layouts import column, gridplot, row
from bokeh.models import (
    ColumnDataSource,
    LayoutDOM,
    LinearColorMapper,
    ColorBar,
    Spacer,
)
from bokeh.palettes import Palette, Viridis256

from bokeh.plotting import figure

from .labels import mathlabel
from .plot_base import prepare_marginal_plot


def mathjax_fix(label: str) -> str:
    """
    Adjust the Matplotlib-style LaTeX label for bokeh/MathJax.
    """
    label = label.replace("µ", r" \mu ")
    label = label.replace("$", "$$")
    return label


def marginal_plot(
    particle_group,
    key1: str = "t",
    key2: str = "p",
    bins: int | None = None,
    *,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    nice: bool = True,
    ellipse: bool = False,
    width: int = 600,
    height: int = 600,
    colorbar: bool = False,
    sizing_mode: SizingModeType | None = None,
    x_label_orientation: float | None = np.pi / 4,
    marginal_fraction: float = 0.33,
    palette: Palette = Viridis256,
    low_color: str = "#ffffff00",
) -> LayoutDOM:
    """
    Density plot and projections with bokeh.

    Parameters
    ----------
    particle_group: ParticleGroup
        The object to plot
    key1: str, default = 't'
        Key to bin on the x-axis
    key2: str, default = 'p'
        Key to bin on the y-axis
    bins: int, default = None
       Number of bins. If None, this will use a heuristic:
       `bins = sqrt(n_particle/4)`
    xlim: tuple, default = None
        Manual setting of the x-axis limits.
    ylim: tuple, default = None
        Manual setting of the y-axis limits.
    tex: bool, default = True
        Use TEX for labels
    nice: bool, default = True
        Use "nice" prefixes.
    ellipse: bool, default = True
        If True, plot an ellipse representing the 2x2 sigma matrix.
    sizing_mode: str, default = None
        Bokeh sizing mode for responsive layout.  When set (e.g.
        ``"stretch_width"``), the layout uses ``row``/``column`` instead of
        ``gridplot`` so that the marginal histograms scale correctly with the
        main figure.
        By default (None), a fixed-size ``GridPlot`` is returned.
    marginal_fraction : float, default = 0.2
        Fraction of the plot to use for the marginal plots.
    palette : bokeh.palettes.Palette, default=Viridis256
        Color map.

    Returns
    -------
    LayoutDOM

    Examples
    --------

    >>> P = ParticleGroup("particles.h5")
    >>> obj = marginal_plot(P, 't', 'energy', bins=200)
    >>> bokeh.io.save(obj, "t_vs_energy.html")
    """
    pdata = prepare_marginal_plot(
        particle_group,
        key1=key1,
        key2=key2,
        bins=bins,
        xlim=xlim,
        ylim=ylim,
        nice=nice,
        ellipse=ellipse,
    )

    labelx = mathlabel(key1, units=pdata.x.full_unit, tex=False)
    labely = mathlabel(key2, units=pdata.y.full_unit, tex=False)

    # Layout Sizes
    main_w = int(width * (1.0 - marginal_fraction))
    main_h = int(height * (1.0 - marginal_fraction))
    marg_w = int(width * marginal_fraction)
    marg_h = int(height * marginal_fraction)

    # Main Joint Figure
    fig_joint = figure(
        width=main_w,
        height=main_h,
        x_axis_label=labelx,
        y_axis_label=labely,
        x_range=pdata.x.lim,
        y_range=pdata.y.lim,
        tools="pan,wheel_zoom,box_zoom,save,reset,hover",
        toolbar_location="left",
    )

    if len(pdata.x.data) == 1:
        fig_joint.scatter(pdata.x.data, pdata.y.data, size=10, color="navy")
    else:
        H, xedges, yedges = np.histogram2d(
            pdata.x.data, pdata.y.data, bins=pdata.bins, weights=pdata.weights
        )
        H = H.T

        h_min = np.min(H[H > 0]) if np.any(H > 0) else 0
        h_max = np.max(H) if np.any(H) else 1

        mapper = LinearColorMapper(
            palette=palette,
            low=h_min,
            high=h_max,
            low_color=low_color,
        )

        if colorbar:
            color_bar = ColorBar(color_mapper=mapper, location=(0, 0))
            fig_joint.add_layout(color_bar, "left")

        source_img = ColumnDataSource(
            {
                "image": [H],
                "x": [xedges[0]],
                "y": [yedges[0]],
                "dw": [xedges[-1] - xedges[0]],
                "dh": [yedges[-1] - yedges[0]],
            }
        )

        fig_joint.image(
            image="image",
            x="x",
            y="y",
            dw="dw",
            dh="dh",
            source=source_img,
            color_mapper=mapper,
        )

    if pdata.ellipse_x is not None and pdata.ellipse_y is not None:
        fig_joint.line(
            pdata.ellipse_x,
            pdata.ellipse_y,
            color="red",
            line_width=2,
            alpha=0.8,
        )

    # Marginal Plots

    # Top (X projection)
    p_top = figure(
        width=main_w,
        height=marg_h,
        x_range=fig_joint.x_range,
        y_axis_location="left",
        min_border=0,
        outline_line_color=None,
        tools="",
    )
    p_top.vbar(
        x=pdata.x.hist_centers,
        top=pdata.x.hist_values,
        width=pdata.x.hist_width,
        bottom=0,
        fill_color="gray",
        line_color="gray",
    )
    p_top.yaxis.axis_label = mathjax_fix(pdata.x.axis_label)
    # p_top.yaxis.axis_label_orientation = ...
    p_top.xaxis.visible = False

    # Right (Y projection)
    p_right = figure(
        width=marg_w,
        height=main_h,
        y_range=fig_joint.y_range,
        x_axis_location="below",
        min_border=0,
        outline_line_color=None,
        tools="",
    )
    p_right.hbar(
        y=pdata.y.hist_centers,
        right=pdata.y.hist_values,
        height=pdata.y.hist_width,
        left=0,
        fill_color="gray",
        line_color="gray",
    )
    p_right.xaxis.axis_label = mathjax_fix(pdata.y.axis_label)
    # p_right.xaxis.axis_label_orientation = ...
    p_right.yaxis.visible = False

    plots = [p_right, p_top, fig_joint]
    if x_label_orientation is not None:
        for plot in plots:
            plot.xaxis.major_label_orientation = x_label_orientation
    for plot in plots:
        plot.toolbar.logo = None

    if sizing_mode is not None:
        fig_joint.sizing_mode = "scale_width"
        fig_joint.aspect_ratio = main_h / main_w
        p_top.sizing_mode = "stretch_width"
        p_right.sizing_mode = "stretch_height"

        left_col = column(p_top, fig_joint, sizing_mode=sizing_mode)
        right_col = column(
            Spacer(width=marg_w, height=marg_h),
            p_right,
            sizing_mode="stretch_height",
            width=marg_w,
        )
        return row(left_col, right_col, sizing_mode=sizing_mode)

    return gridplot(
        [
            [p_top, None],
            [fig_joint, p_right],
        ],
        merge_tools=True,
        toolbar_location="left",
    )
