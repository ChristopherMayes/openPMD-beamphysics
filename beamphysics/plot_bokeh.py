from __future__ import annotations

import dataclasses

import numpy as np
from bokeh.core.enums import SizingModeType
from bokeh.layouts import column, gridplot, row
from bokeh.models import (
    ColorBar,
    ColumnDataSource,
    Div,
    HoverTool,
    LayoutDOM,
    LinearColorMapper,
    Spacer,
)
from bokeh.palettes import Palette, Viridis256
from bokeh.plotting import figure

from .labels import mathlabel
from .plot_base import prepare_marginal_plot
from .units import c_light


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
    text: str | None = None,
    title: str | None = None,
    font_settings: MarginalFontSettings | None = None,
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
        return row(left_col, right_col, sizing_mode=sizing_mode)

    if text_div is not None:
        left_col = column(p_top, fig_joint)
        right_col = column(text_div, p_right)
        return row(left_col, right_col)

    return gridplot(
        [
            [p_top, None],
            [fig_joint, p_right],
        ],
        merge_tools=True,
        toolbar_location="left",
    )
