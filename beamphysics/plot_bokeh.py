from __future__ import annotations

import dataclasses
import logging

import numpy as np
from bokeh.core.enums import SizingModeType
from bokeh.layouts import column, gridplot, row
from bokeh.models import (
    ColorBar,  # pyright: ignore[reportPrivateImportUsage]
    ColumnDataSource,  # pyright: ignore[reportPrivateImportUsage]
    Div,  # pyright: ignore[reportPrivateImportUsage]
    HoverTool,  # pyright: ignore[reportPrivateImportUsage]
    LayoutDOM,  # pyright: ignore[reportPrivateImportUsage]
    LinearAxis,  # pyright: ignore[reportPrivateImportUsage]
    LinearColorMapper,  # pyright: ignore[reportPrivateImportUsage]
    Range1d,  # pyright: ignore[reportPrivateImportUsage]
    Spacer,  # pyright: ignore[reportPrivateImportUsage]
)
from bokeh.io import show as _bokeh_show
from bokeh.palettes import Palette, Viridis256
from bokeh.plotting import figure

from .labels import mathlabel
from .plot_base import (
    Limit,
    prepare_density_and_slice_plot,
    prepare_density_plot,
    prepare_marginal_plot,
    prepare_slice_plot,
    prepare_wakefield_plot,
)
from .units import c_light

logger = logging.getLogger(__name__)


def initialize_jupyter():
    # Is this public bokeh API? An attempt at forward-compatibility
    try:
        from bokeh.io.state import curstate
    except ImportError:
        pass
    else:
        state = curstate()
        if getattr(state, "notebook", False):
            # Jupyter already initialized
            logger.debug("Bokeh output_notebook already called; not re-initializing")
            return

    from bokeh.plotting import output_notebook

    output_notebook()


def _maybe_show(layout: LayoutDOM, show: bool = True) -> LayoutDOM:
    """Call ``bokeh.io.show`` on *layout* if *show* is truthy."""
    if show:
        _bokeh_show(layout)
    return layout


@dataclasses.dataclass
class FontPlotSettings:
    """Font settings for a single Bokeh figure's axes."""

    axis_label_text_font_size: str = "14px"
    axis_label_text_font_style: str = "italic"
    major_label_text_font_size: str = "12px"
    major_label_text_font_style: str = "normal"

    def apply(self, *axes) -> None:
        """Apply these font settings to one or more Bokeh axis objects."""
        for axis in axes:
            axis.axis_label_text_font_size = self.axis_label_text_font_size
            axis.axis_label_text_font_style = self.axis_label_text_font_style
            axis.major_label_text_font_size = self.major_label_text_font_size
            axis.major_label_text_font_style = self.major_label_text_font_style


@dataclasses.dataclass
class MarginalFontSettings:
    """Font settings for Bokeh marginal plots."""

    text_font: str | None = None
    annotation_text_font_size: str | None = None
    main: FontPlotSettings = dataclasses.field(default_factory=FontPlotSettings)
    top: FontPlotSettings = dataclasses.field(
        default_factory=lambda: FontPlotSettings(
            axis_label_text_font_size="10px",
            major_label_text_font_size="8px",
        )
    )
    right: FontPlotSettings = dataclasses.field(
        default_factory=lambda: FontPlotSettings(
            axis_label_text_font_size="10px",
            major_label_text_font_size="8px",
        )
    )


def mathjax_fix(label: str) -> str:
    """
    Adjust the Matplotlib-style LaTeX label for bokeh/MathJax.
    """
    label = label.replace("µ", r" \mu ")
    label = label.replace("$", "$$")
    return label


@dataclasses.dataclass
class StatsAnnotation:
    """A single beam statistic annotation row."""

    label: str
    sub_label: str
    value: str
    units: str


def get_annotations(particle_group, key1: str, key2: str) -> list[StatsAnnotation]:
    """
    Return beam-statistic annotations for a given key combination.

    Parameters
    ----------
    particle_group : ParticleGroup
        The particle group to compute statistics from.
    key1 : str
        The x-axis key.
    key2 : str
        The y-axis key.

    Returns
    -------
    list[StatsAnnotation]
    """

    # Longitudinal phase space: delta_z/c or z/c vs energy
    if key1 in ("delta_z/c", "z/c") and key2 == "energy":
        sigma_z = particle_group["sigma_z"]
        sigma_p = particle_group["sigma_p"]
        p0 = particle_group["mean_p"]
        return [
            StatsAnnotation("σ", "z", f"{sigma_z / c_light * 1e15:.0f}", "fs"),
            StatsAnnotation("σ", "δ", f"{sigma_p / p0 * 1e4:.1f} × 10⁻⁴", ""),
            StatsAnnotation(
                "⟨E⟩", "", f"{particle_group['mean_energy'] / 1e6:.1f}", "MeV"
            ),
        ]

    # Transverse spot: x vs y
    if key1 == "x" and key2 == "y":
        return [
            StatsAnnotation("⟨x⟩", "", f"{particle_group['mean_x'] * 1e6:.1f}", "µm"),
            StatsAnnotation("⟨y⟩", "", f"{particle_group['mean_y'] * 1e6:.1f}", "µm"),
            StatsAnnotation("σ", "x", f"{particle_group['sigma_x'] * 1e6:.1f}", "µm"),
            StatsAnnotation("σ", "y", f"{particle_group['sigma_y'] * 1e6:.1f}", "µm"),
        ]

    # Horizontal phase space: x vs xp or px
    if key1 == "x" and key2 in ("xp", "px"):
        return [
            StatsAnnotation(
                "ε", "n,x", f"{particle_group['norm_emit_x'] * 1e6:.2f}", "mm-mrad"
            ),
        ]

    # Vertical phase space: y vs yp or py
    if key1 == "y" and key2 in ("yp", "py"):
        return [
            StatsAnnotation(
                "ε", "n,y", f"{particle_group['norm_emit_y'] * 1e6:.2f}", "mm-mrad"
            ),
        ]

    return []


def _annotations_to_html(annotations: list[StatsAnnotation]) -> str | None:
    """Convert a list of Annotation objects to an HTML table."""
    if not annotations:
        return None

    rows = []
    for a in annotations:
        label = f"{a.label}<sub>{a.sub_label}</sub>" if a.sub_label else a.label
        rows.append(
            f"<tr><td style='text-align:right;padding-right:4px'>{label}</td>"
            f"<td>{a.value} {a.units}</td></tr>"
        )
    return "<table style='border-collapse:collapse'>" + "".join(rows) + "</table>"


def density_plot(
    particle_group,
    key: str = "x",
    bins: int | str | None = None,
    *,
    xlim: Limit | None = None,
    tex: bool = False,
    nice: bool = True,
    width: int = 600,
    height: int = 400,
    color: str = "gray",
    alpha: float = 0.7,
    sizing_mode: SizingModeType | None = None,
    title: str | None = None,
    show: bool = True,
    **kwargs,
) -> LayoutDOM:
    """
    1D density histogram with Bokeh.

    Parameters
    ----------
    particle_group : ParticleGroup
        The object to plot.
    key : str, default = 'x'
        Which quantity to plot.
    bins : int or str, optional
        Number of bins.
    xlim : tuple of float, optional
        Manual x-axis limits.
    nice : bool, default = True
        Use nice unit scaling.
    width, height : int
        Figure dimensions in pixels.
    show : bool, default = True
        Display the plot.

    Returns
    -------
    LayoutDOM
    """
    pdata = prepare_density_plot(
        particle_group, key=key, bins=bins, xlim=xlim, nice=nice, tex=tex
    )

    fig = figure(
        width=width,
        height=height,
        x_axis_label=mathjax_fix(pdata.x_label),
        y_axis_label=mathjax_fix(pdata.y_label),
        tools="pan,wheel_zoom,box_zoom,save,reset",
        toolbar_location="right",
    )

    fig.vbar(
        x=pdata.hist_centers,
        top=pdata.hist_values,
        width=pdata.hist_width,
        bottom=0,
        fill_color=color,
        line_color=color,
        fill_alpha=alpha,
    )

    if pdata.xlim:
        fig.x_range.start, fig.x_range.end = pdata.xlim

    if title:
        fig.title.text = title

    if sizing_mode is not None:
        fig.sizing_mode = sizing_mode

    fig.toolbar.logo = None

    return _maybe_show(fig, show)


def marginal_plot(
    particle_group,
    key1: str = "t",
    key2: str = "p",
    bins: int | None = None,
    *,
    xlim: Limit | None = None,
    ylim: Limit | None = None,
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
    text: str | None = None,
    title: str | None = None,
    font_settings: MarginalFontSettings | None = None,
    show: bool = True,
    **kwargs,
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
    text : str or None, optional
        Custom HTML text to display in the top-right corner.  If ``None``
        (the default), automatic beam-statistics text is generated for
        recognized key combinations (e.g. ``x``/``y``, ``x``/``px``,
        ``delta_z/c``/``energy``).  Pass an empty string ``""`` to suppress
        automatic text.
    title : str or None, optional
        Title to set on the main density plot.

    Returns
    -------
    LayoutDOM

    Examples
    --------

    >>> P = ParticleGroup("particles.h5")
    >>> obj = marginal_plot(P, 't', 'energy', bins=200)
    >>> bokeh.io.save(obj, "t_vs_energy.html")
    """
    if kwargs:
        logger.warning(f"Unsupported kwargs: {kwargs}")
    if font_settings is None:
        font_settings = MarginalFontSettings()

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
        tools="pan,wheel_zoom,box_zoom,save,reset",
        toolbar_location="left",
    )

    font_settings.main.apply(fig_joint.xaxis, fig_joint.yaxis)

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

        image_renderer = fig_joint.image(
            image="image",
            x="x",
            y="y",
            dw="dw",
            dh="dh",
            source=source_img,
            color_mapper=mapper,
        )

        hover = HoverTool(
            renderers=[image_renderer],
            tooltips=[
                (labelx, "$x"),
                (labely, "$y"),
                ("density", "@image"),
            ],
        )
        fig_joint.add_tools(hover)

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

    font_settings.top.apply(p_top.yaxis)
    font_settings.right.apply(p_right.xaxis)

    if font_settings.text_font is not None:
        for plot in plots:
            for axis in (plot.xaxis, plot.yaxis):
                axis.axis_label_text_font = font_settings.text_font
                axis.major_label_text_font = font_settings.text_font

    annotation_html: str | None = None
    if text is None:
        annotation_html = _annotations_to_html(
            get_annotations(particle_group, key1, key2)
        )
    elif text:
        annotation_html = text.replace("\n", "<br>")

    text_div: Div | None = None
    if annotation_html:
        popup_font_size = font_settings.annotation_text_font_size or "12px"
        popup_css = ""
        if font_settings.text_font is not None:
            popup_css += f"font-family: {font_settings.text_font}; "
        popup_html = f"""
            <style>
            .bk-stats-trigger {{
              cursor: pointer;
              position: relative;
              text-align: center;
            }}
            .bk-stats-popup {{
              display: none;
              position: absolute;
              top: 100%;
              right: 0;
              background: white;
              color: black;
              border: 1px solid #ccc;
              border-radius: 4px;
              padding: 8px 12px;
              box-shadow: 0 2px 8px rgba(0,0,0,0.15);
              z-index: 100;
              white-space: nowrap;
              line-height: 1.6;
              font-size: {popup_font_size};
              {popup_css}
            }}
            .bk-stats-trigger:hover .bk-stats-popup {{
              display: block;
            }}
            </style>
              <div class="bk-stats-trigger">Stats
              <div class="bk-stats-popup">{annotation_html}</div>
            </div>
           """
        text_div = Div(
            text=popup_html,
            width=marg_w,
            height=marg_h,
            styles={"overflow": "visible"},
        )

    if title:
        fig_joint.title.text = title

    top_right = (
        text_div if text_div is not None else Spacer(width=marg_w, height=marg_h)
    )

    if sizing_mode is not None:
        fig_joint.sizing_mode = "scale_both"
        fig_joint.aspect_ratio = main_h / main_w
        p_top.sizing_mode = "stretch_width"
        p_right.sizing_mode = "stretch_height"

        left_col = column(p_top, fig_joint, sizing_mode=sizing_mode)
        right_col = column(
            top_right,
            p_right,
            sizing_mode="stretch_height",
            width=marg_w,
        )
        layout = row(left_col, right_col, sizing_mode=sizing_mode)
    elif text_div is not None:
        left_col = column(p_top, fig_joint)
        right_col = column(text_div, p_right)
        layout = row(left_col, right_col)
    else:
        layout = gridplot(
            [
                [p_top, None],
                [fig_joint, p_right],
            ],
            merge_tools=True,
            toolbar_location="left",
        )

    return _maybe_show(layout, show)


# Default Bokeh color cycle for multi-curve plots
_BOKEH_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def slice_plot(
    particle_group,
    *keys: str,
    n_slice: int = 40,
    slice_key: str | None = None,
    xlim: Limit | None = None,
    ylim: Limit | None = None,
    nice: bool = True,
    tex: bool = False,
    width: int = 700,
    height: int = 400,
    sizing_mode: SizingModeType | None = None,
    title: str | None = None,
    density_alpha: float = 0.2,
    show: bool = True,
    **kwargs,
) -> LayoutDOM:
    """
    Slice statistics plot with Bokeh.

    Plots slice statistics as lines on the primary y-axis and
    the bunch density as a filled area on a secondary y-axis.

    Parameters
    ----------
    particle_group : ParticleGroup
        The object to plot.
    keys : str
        Statistical quantities to plot (e.g. ``'sigma_x'``, ``'norm_emit_x'``).
    n_slice : int, default = 40
        Number of slices.
    slice_key : str, optional
        Dimension to slice in (``'t'``, ``'z'``, ``'delta_t'``, etc.).
    xlim, ylim : tuple of float, optional
        Manual axis limits.
    nice : bool, default = True
        Use nice unit scaling.
    width, height : int
        Figure dimensions in pixels.

    Returns
    -------
    LayoutDOM
    """
    if kwargs:
        logger.warning(f"Unsupported kwargs: {kwargs}")

    pdata = prepare_slice_plot(
        particle_group,
        *keys,
        n_slice=n_slice,
        slice_key=slice_key,
        xlim=xlim,
        ylim=ylim,
        nice=nice,
        tex=tex,
    )

    fig = figure(
        width=width,
        height=height,
        x_axis_label=mathjax_fix(pdata.x_label),
        y_axis_label=mathjax_fix(pdata.y_label),
        tools="pan,wheel_zoom,box_zoom,save,reset",
        toolbar_location="right",
    )

    # Main curves
    for i, curve in enumerate(pdata.curves):
        color = (
            "black" if len(pdata.curves) == 1 else _BOKEH_COLORS[i % len(_BOKEH_COLORS)]
        )
        fig.line(
            pdata.x,
            curve.values,
            legend_label=curve.label,
            color=color,
            line_width=2,
        )

    if len(pdata.curves) > 1:
        fig.legend.click_policy = "hide"

    # Density on secondary y-axis
    density_max = (
        float(np.max(pdata.density_values)) if len(pdata.density_values) > 0 else 1.0
    )
    fig.extra_y_ranges["density"] = Range1d(start=0, end=density_max * 1.1)
    fig.add_layout(
        LinearAxis(
            y_range_name="density",
            axis_label=mathjax_fix(pdata.density_label),
        ),
        "right",
    )

    fig.varea(
        x=pdata.x,
        y1=0,
        y2=pdata.density_values,
        y_range_name="density",
        fill_color="black",
        fill_alpha=density_alpha,
    )

    if pdata.xlim:
        fig.x_range.start, fig.x_range.end = pdata.xlim
    if pdata.ylim:
        fig.y_range.start, fig.y_range.end = pdata.ylim

    if title:
        fig.title.text = title

    if sizing_mode is not None:
        fig.sizing_mode = sizing_mode

    fig.toolbar.logo = None

    return _maybe_show(fig, show)


def wakefield_plot(
    particle_group,
    wake,
    key: str | None = None,
    nice: bool = True,
    xlim: Limit | None = None,
    ylim: Limit | None = None,
    tex: bool = False,
    bins: int | str | None = None,
    width: int = 700,
    height: int = 400,
    sizing_mode: SizingModeType | None = None,
    title: str | None = None,
    density_alpha: float = 0.3,
    scatter_size: float = 2,
    show: bool = True,
    **kwargs,
) -> LayoutDOM:
    """
    Wakefield kicks scatter plot with density overlay using Bokeh.

    Parameters
    ----------
    particle_group : ParticleGroup
        The particle distribution.
    wake : WakefieldBase
        Wakefield object providing ``particle_kicks(z, weight)``.
    key : str, optional
        Independent variable key. Auto-detected if None.
    nice : bool, default = True
        Use nice unit scaling.
    width, height : int
        Figure dimensions in pixels.
    density_alpha : float
        Alpha for the density overlay bars.
    scatter_size : float
        Size of scatter markers.

    Returns
    -------
    LayoutDOM
    """

    if kwargs:
        logger.warning(f"Unsupported kwargs: {kwargs}")
    pdata = prepare_wakefield_plot(
        particle_group,
        wake,
        key=key,
        nice=nice,
        tex=tex,
        xlim=xlim,
        ylim=ylim,
        bins=bins,
    )

    fig = figure(
        width=width,
        height=height,
        x_axis_label=mathjax_fix(pdata.x_label),
        y_axis_label=mathjax_fix(pdata.y_label),
        tools="pan,wheel_zoom,box_zoom,save,reset",
        toolbar_location="right",
    )

    # Density overlay on secondary y-axis
    density_max = (
        float(np.max(pdata.density.hist_values))
        if len(pdata.density.hist_values) > 0
        else 1.0
    )
    fig.extra_y_ranges["density"] = Range1d(start=0, end=density_max * 1.1)
    fig.add_layout(
        LinearAxis(
            y_range_name="density",
            axis_label=mathjax_fix(pdata.density.y_label),
        ),
        "right",
    )

    fig.vbar(
        x=pdata.density.hist_centers,
        top=pdata.density.hist_values,
        width=pdata.density.hist_width,
        bottom=0,
        y_range_name="density",
        fill_color="gray",
        line_color="gray",
        fill_alpha=density_alpha,
    )

    # Wake kicks scatter
    fig.scatter(
        pdata.scatter_x,
        pdata.scatter_y,
        size=scatter_size,
        color="black",
    )

    if pdata.xlim:
        fig.x_range.start, fig.x_range.end = pdata.xlim
    if pdata.ylim:
        fig.y_range.start, fig.y_range.end = pdata.ylim

    if title:
        fig.title.text = title

    if sizing_mode is not None:
        fig.sizing_mode = sizing_mode

    fig.toolbar.logo = None

    return _maybe_show(fig, show)


def density_and_slice_plot(
    particle_group,
    key1: str = "t",
    key2: str = "p",
    stat_keys: list[str] | None = None,
    bins: int = 100,
    n_slice: int = 30,
    tex: bool = False,
    width: int = 700,
    height: int = 450,
    sizing_mode: SizingModeType | None = None,
    title: str | None = None,
    density_alpha: float = 0.1,
    palette: Palette = Viridis256,
    show: bool = True,
    **kwargs,
) -> LayoutDOM:
    """
    2D density plot with overlaid slice statistics using Bokeh.

    Parameters
    ----------
    particle_group : ParticleGroup
        The object to plot.
    key1 : str, default = 't'
        Key for x-axis (also used as slice key).
    key2 : str, default = 'p'
        Key for y-axis (density).
    stat_keys : list of str, optional
        Slice statistics to overlay.
    bins : int, default = 100
        Number of bins for the 2D histogram.
    n_slice : int, default = 30
        Number of slices.
    width, height : int
        Figure dimensions in pixels.

    Returns
    -------
    LayoutDOM
    """
    pdata = prepare_density_and_slice_plot(
        particle_group,
        key1=key1,
        key2=key2,
        stat_keys=stat_keys,
        bins=bins,
        n_slice=n_slice,
        tex=tex,
    )

    ext = pdata.extent  # [xmin, xmax, ymin, ymax]

    # Color mapper for the 2D histogram
    H = pdata.hist2d
    h_min = float(np.min(H[H > 0])) if np.any(H > 0) else 0
    h_max = float(np.max(H)) if np.any(H) else 1

    mapper = LinearColorMapper(
        palette=palette,
        low=h_min,
        high=h_max,
        low_color="#ffffff00",
    )

    fig = figure(
        width=width,
        height=height,
        x_axis_label=mathjax_fix(pdata.x_label),
        y_axis_label=mathjax_fix(pdata.y_label),
        x_range=(ext[0], ext[1]),
        y_range=(ext[2], ext[3]),
        tools="pan,wheel_zoom,box_zoom,save,reset",
        toolbar_location="right",
    )

    fig.image(
        image=[H.T],
        x=ext[0],
        y=ext[2],
        dw=ext[1] - ext[0],
        dh=ext[3] - ext[2],
        color_mapper=mapper,
    )

    # Slice statistics on secondary y-axis
    stat_max = (
        max(float(np.max(c.values)) for c in pdata.slice_curves)
        if pdata.slice_curves
        else 1.0
    )
    fig.extra_y_ranges["stats"] = Range1d(start=0, end=stat_max * 1.1)
    fig.add_layout(
        LinearAxis(
            y_range_name="stats",
            axis_label=mathjax_fix(pdata.slice_y_label),
        ),
        "right",
    )

    for i, curve in enumerate(pdata.slice_curves):
        color = _BOKEH_COLORS[i % len(_BOKEH_COLORS)]
        fig.line(
            pdata.slice_x,
            curve.values,
            y_range_name="stats",
            legend_label=curve.label,
            color=color,
            line_width=2,
        )

    if len(pdata.slice_curves) > 1:
        fig.legend.click_policy = "hide"

    # Density overlay
    fig.varea(
        x=pdata.slice_x,
        y1=0,
        y2=pdata.slice_density,
        y_range_name="stats",
        fill_color="black",
        fill_alpha=density_alpha,
    )

    if title:
        fig.title.text = title
    if sizing_mode is not None:
        fig.sizing_mode = sizing_mode
    fig.toolbar.logo = None

    return _maybe_show(fig, show)


# ---------------------------------------------------------------------------
# Generic plotting functions (used by Wavefront, etc.)
# ---------------------------------------------------------------------------


def plot_1d_density(
    x,
    y,
    x_name: str = "",
    y_name: str | None = None,
    x_units: str | None = None,
    y_units: str | None = None,
    log_scale_y: bool = False,
    show_cdf: bool = False,
    cdf_label: str = "CDF",
    kind: str = "bar",
    plot_style: dict | None = None,
    xlim: Limit | None = None,
    ylim: Limit | None = (0, None),
    nice: bool = True,
    auto_label: bool = False,
    tex: bool = False,
    data: dict | None = None,
    width: int = 600,
    height: int = 400,
    sizing_mode: SizingModeType | None = None,
    title: str | None = None,
    show: bool = True,
    **kwargs,
) -> LayoutDOM:
    """
    Generic 1D density distribution plot with Bokeh.

    Mirrors the API of the matplotlib ``plot_1d_density``.

    Parameters
    ----------
    x, y : array or str
        Data arrays or string keys into *data* dict.
    x_name, y_name : str
        Axis labels.
    x_units, y_units : str, optional
        Units appended to labels.
    log_scale_y : bool
        Log scale on y-axis.
    show_cdf : bool
        Show cumulative distribution on secondary y-axis.
    kind : str
        ``'bar'`` or ``'line'``.
    nice : bool
        Use nice unit scaling.
    data : dict, optional
        Dict mapping string keys to arrays.
    width, height : int
        Figure dimensions.

    Returns
    -------
    LayoutDOM
    """

    if kwargs:
        logger.warning(f"Unsupported kwargs: {kwargs}")
    from .units import pg_units, plottable_array

    # Resolve data dict
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

    if x_key is not None and x_name == "":
        x_name = x_key
    if y_key is not None and y_name is None:
        y_name = y_key
    if y_name is None:
        y_name = "Density"

    # Auto-label
    if auto_label:
        if x_key and x_units is None:
            try:
                x_units = pg_units(x_key).unitSymbol
            except (ValueError, KeyError):
                pass
        if y_key and y_units is None:
            try:
                y_units = pg_units(y_key).unitSymbol
            except (ValueError, KeyError):
                pass

    # Nice scaling
    x, f1, p1, x_min, x_max = plottable_array(x, nice=nice, lim=xlim)
    y, f2, p2, y_min, y_max = plottable_array(y, nice=nice, lim=ylim)

    if x_units:
        x_units = p1 + str(x_units)
    elif p1:
        x_units = p1

    if y_units:
        y_units = p2 + str(y_units)
    elif p2:
        y_units = p2

    # Labels
    if auto_label and x_key:
        x_label = mathjax_fix(mathlabel(x_key, units=x_units, tex=tex))
    else:
        x_label = f"{x_name} ({x_units})" if x_units else x_name

    if auto_label and y_key:
        y_label = mathjax_fix(mathlabel(y_key, units=y_units, tex=tex))
    else:
        y_label = f"{y_name} ({y_units})" if y_units else y_name

    # Bar widths
    if len(x) > 1:
        widths = np.diff(x)
        widths = np.append(widths, widths[-1])
    else:
        widths = np.ones_like(x)

    fig = figure(
        width=width,
        height=height,
        x_axis_label=x_label,
        y_axis_label=y_label,
        tools="pan,wheel_zoom,box_zoom,save,reset",
        toolbar_location="right",
        y_axis_type="log" if log_scale_y else "auto",
    )

    if plot_style is None:
        plot_style = {}

    if kind == "bar":
        color = plot_style.get("color", "gray")
        alpha = plot_style.get("alpha", 0.7)
        fig.vbar(
            x=x,
            top=y,
            width=widths,
            bottom=0,
            fill_color=color,
            line_color=color,
            fill_alpha=alpha,
        )
    elif kind == "line":
        color = plot_style.get("color", "blue")
        line_width = plot_style.get("linewidth", plot_style.get("line_width", 2))
        fig.line(x, y, color=color, line_width=line_width)
    else:
        raise ValueError(f"kind must be 'bar' or 'line', got '{kind}'")

    if xlim is not None:
        fig.x_range.start, fig.x_range.end = x_min / f1, x_max / f1
    if ylim is not None:
        if ylim[0] is not None:
            fig.y_range.start = y_min / f2
        if ylim[1] is not None:
            fig.y_range.end = y_max / f2

    # CDF on secondary y-axis
    if show_cdf:
        cdf = np.cumsum(y * widths) * f1 * f2
        cdf_scaled, _, cdf_prefix, _, _ = plottable_array(cdf, nice=nice)

        cdf_max = float(np.max(cdf_scaled)) if len(cdf_scaled) > 0 else 1.0
        fig.extra_y_ranges["cdf"] = Range1d(start=0, end=cdf_max)
        cdf_axis_label = f"{cdf_label} ({cdf_prefix})" if cdf_prefix else cdf_label
        fig.add_layout(
            LinearAxis(y_range_name="cdf", axis_label=cdf_axis_label),
            "right",
        )
        fig.line(x, cdf_scaled, y_range_name="cdf", color="blue", line_width=2)

    if title:
        fig.title.text = title
    if sizing_mode is not None:
        fig.sizing_mode = sizing_mode
    fig.toolbar.logo = None

    return _maybe_show(fig, show)


def plot_2d_density_with_marginals(
    data: np.ndarray,
    dx: float = 1,
    dy: float = 1,
    xmin: float | None = None,
    ymin: float | None = None,
    x_name: str = "",
    y_name: str = "",
    z_name: str = "",
    x_units: str | None = None,
    y_units: str | None = None,
    z_units: str | None = None,
    log_scale_z: bool = False,
    log_scale_marginals: bool = False,
    show_colorbar: bool = True,
    xlim: Limit | None = None,
    ylim: Limit | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    width: int = 600,
    height: int = 600,
    marginal_fraction: float = 0.25,
    palette: Palette = Viridis256,
    sizing_mode: SizingModeType | None = None,
    title: str | None = None,
    show: bool = True,
    **kwargs,
) -> LayoutDOM:
    """
    2D density map with marginal histograms using Bokeh.

    Mirrors the API of the matplotlib ``plot_2d_density_with_marginals``.

    Parameters
    ----------
    data : np.ndarray
        2D array of density values, shape ``(nx, ny)``.
    dx, dy : float
        Grid spacing.
    xmin, ymin : float, optional
        Origin of the grid. Default centers at 0.
    x_name, y_name, z_name : str
        Axis labels.
    x_units, y_units, z_units : str, optional
        Units appended to labels.
    log_scale_z : bool
        Log color mapping.
    palette : Palette
        Bokeh color palette.
    width, height : int
        Figure dimensions.

    Returns
    -------
    LayoutDOM
    """
    if kwargs:
        logger.warning(f"Unsupported kwargs: {kwargs}")
    nx, ny = data.shape

    if xmin is None:
        xmin = -((nx - 1) * dx) / 2
    if ymin is None:
        ymin = -((ny - 1) * dy) / 2

    xmax = xmin + (nx - 1) * dx
    ymax = ymin + (ny - 1) * dy

    xvec = np.linspace(xmin, xmax, nx)
    yvec = np.linspace(ymin, ymax, ny)

    x_marginal = np.sum(data, axis=1) * dy
    y_marginal = np.sum(data, axis=0) * dx

    vmin = vmin if vmin is not None else float(np.min(data))
    vmax = vmax if vmax is not None else float(np.max(data))

    x_label = f"{x_name} ({x_units})" if x_units else x_name
    y_label = f"{y_name} ({y_units})" if y_units else y_name

    # Layout sizes
    main_w = int(width * (1.0 - marginal_fraction))
    main_h = int(height * (1.0 - marginal_fraction))
    marg_w = int(width * marginal_fraction)
    marg_h = int(height * marginal_fraction)

    # Color mapper
    if log_scale_z:
        low = max(vmin, vmax * 1e-6)
        mapper = LinearColorMapper(palette=palette, low=low, high=vmax)
    else:
        mapper = LinearColorMapper(palette=palette, low=vmin, high=vmax)

    # Main density figure
    x_range = xlim or (xmin - dx / 2, xmax + dx / 2)
    y_range = ylim or (ymin - dy / 2, ymax + dy / 2)

    fig_main = figure(
        width=main_w,
        height=main_h,
        x_axis_label=x_label,
        y_axis_label=y_label,
        x_range=x_range,
        y_range=y_range,
        tools="pan,wheel_zoom,box_zoom,save,reset",
        toolbar_location="left",
    )

    fig_main.image(
        image=[data.T],
        x=xmin - dx / 2,
        y=ymin - dy / 2,
        dw=xmax - xmin + dx,
        dh=ymax - ymin + dy,
        color_mapper=mapper,
    )

    if show_colorbar:
        cbar_label = f"{z_name} ({z_units})" if z_units else z_name
        color_bar = ColorBar(color_mapper=mapper, title=cbar_label, location=(0, 0))
        fig_main.add_layout(color_bar, "left")

    if title:
        fig_main.title.text = title

    # Top marginal (X projection)
    p_top = figure(
        width=main_w,
        height=marg_h,
        x_range=fig_main.x_range,
        y_axis_type="log" if log_scale_marginals else "auto",
        min_border=0,
        outline_line_color=None,
        tools="",
    )
    p_top.vbar(
        x=xvec,
        top=x_marginal,
        width=dx,
        bottom=0,
        fill_color="gray",
        line_color="gray",
    )
    if z_units and y_units:
        p_top.yaxis.axis_label = f"{z_units} {y_units}"
    p_top.xaxis.visible = False

    # Right marginal (Y projection)
    p_right = figure(
        width=marg_w,
        height=main_h,
        y_range=fig_main.y_range,
        x_axis_type="log" if log_scale_marginals else "auto",
        min_border=0,
        outline_line_color=None,
        tools="",
    )
    p_right.hbar(
        y=yvec,
        right=y_marginal,
        height=dy,
        left=0,
        fill_color="gray",
        line_color="gray",
    )
    if z_units and x_units:
        p_right.xaxis.axis_label = f"{z_units} {x_units}"
    p_right.yaxis.visible = False

    for p in (fig_main, p_top, p_right):
        p.toolbar.logo = None

    top_right = Spacer(width=marg_w, height=marg_h)

    if sizing_mode is not None:
        fig_main.sizing_mode = "scale_both"
        # fig_main.aspect_ratio = main_h / main_w
        p_top.sizing_mode = "stretch_width"
        p_right.sizing_mode = "stretch_height"
        left_col = column(p_top, fig_main, sizing_mode=sizing_mode)
        right_col = column(
            top_right, p_right, sizing_mode="stretch_height", width=marg_w
        )
        layout = row(left_col, right_col, sizing_mode=sizing_mode)
    else:
        layout = gridplot(
            [
                [p_top, top_right],
                [fig_main, p_right],
            ],
            merge_tools=True,
            toolbar_location="left",
        )

    return _maybe_show(layout, show)
