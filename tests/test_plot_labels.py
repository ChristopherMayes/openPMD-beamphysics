from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pytest

from beamphysics import single_particle
from beamphysics.labels import mathlabel
from beamphysics.plot import (
    _charge_density_units_str,
    marginal_plot,
)
from beamphysics.units import plottable_array_and_units, pmd_unit


@pytest.mark.parametrize(
    ("factor", "symbol", "expected"),
    [
        (1, "m", "m"),
        (1e-3, "m", "mm"),
        (1e-3, "eV/c", "meV/c"),  # spelling kept: prefix on the leading token
        (1e6, "V/m", "MV/m"),
        # Gluing would change the meaning -> explicit power of ten.
        (1e-3, "m^2", "1e-3 m^2"),  # 'mm^2' would read as (mm)^2 = 1e-6
        (1e-3, "sqrt(m)", "1e-3 sqrt(m)"),  # 'msqrt(m)' is unparseable
        (1e-3, "1/m", "1e-3 1/m"),
        # Re-spelling rescues cases the spelling can't express.
        (1e-6, "mC", "nC"),  # never a double prefix
        (1e3, "1/s", "kHz"),
        (1e-3, "", "1e-3"),  # dimensionless: pure number
        (2.0, "m", "2 m"),  # non-power-of-ten factor written plainly
    ],
)
def test_scaled_symbol(factor: float, symbol: str, expected: str) -> None:
    assert pmd_unit(symbol).scaled_symbol(factor) == expected


def test_scaled_symbol_output_never_misparses() -> None:
    """Whatever scaled_symbol returns either re-parses to the scaled unit or
    is not a parseable unit symbol at all — never a different unit."""
    for factor in (1e-3, 1e3, 1e-6, 1e9):
        for symbol in ("m", "eV/c", "m^2", "sqrt(m)", "1/m", "T*m", "kg*m/s"):
            base = pmd_unit(symbol)
            out = base.scaled_symbol(factor)
            try:
                parsed = pmd_unit(out)
            except (ValueError, KeyError):
                continue  # explicit 1eN form: not a unit symbol, fine
            assert parsed.compatible_with(base)
            assert parsed.unitSI == pytest.approx(factor * base.unitSI)


def test_scaled_symbol_finds_named_units() -> None:
    # Exact named/prefixed match through simplify.
    assert pmd_unit("C").scaled_symbol(1e-6) == "µC"
    assert (pmd_unit("mA") * pmd_unit("s")).scaled_symbol(1e-6) == "nC"
    assert pmd_unit("m").scaled_symbol(1) == "m"


def test_charge_density_units_str() -> None:
    """Histogram density labels keep the displayed axis unit (with its own
    prefix) as the denominator, with the histogram prefix on the C. A time
    axis instead displays in amps, with the two scales folded into one
    prefix."""
    s, m = pmd_unit("s"), pmd_unit("m")
    # Time axes: C/s = A.
    assert _charge_density_units_str(s, "s", 1.0) == "A"
    assert _charge_density_units_str(s, "s", 1e-3) == "mA"
    assert _charge_density_units_str(s, "µs", 1.0, 1e-6) == "MA"  # 1 C/µs
    # Other axes: prefixed C per displayed axis unit.
    assert _charge_density_units_str(m, "µm", 1e-9) == "nC/µm"
    assert _charge_density_units_str(m, "m", 1.0) == "C/m"
    assert _charge_density_units_str(pmd_unit("eV/c"), "keV/c", 1e-12) == "pC/(keV/c)"
    # Unparseable axis units (explicit power-of-ten form) stay grouped.
    assert (
        _charge_density_units_str(pmd_unit("sqrt(m)"), "1e-3 sqrt(m)", 1e-9)
        == "nC/(1e-3 sqrt(m))"
    )


def test_marginal_density_labels_keep_axis_prefixes() -> None:
    """The marginal histogram labels read as charge per *displayed* axis
    unit — nC/µm against an x axis in µm, pC/(keV/c) against keV/c — rather
    than consolidating everything into one prefix against the SI unit. A
    time axis is the exception: C/s displays as amps."""
    import matplotlib.pyplot as plt

    from beamphysics import ParticleGroup

    P = ParticleGroup("docs/examples/data/bmad_particles.h5")

    fig = marginal_plot(P, "x", "px")
    xlabel = fig.axes[0].get_xlabel()
    x_marg = fig.axes[1].get_ylabel()
    y_marg = fig.axes[2].get_xlabel()
    # The denominator of each marginal label is the axis unit as displayed.
    assert r"\mathrm{µm}" in xlabel
    assert x_marg.endswith(r"/\mathrm{µm}$")
    assert y_marg.endswith(r"/(\mathrm{keV}/\mathrm{c})$")
    plt.close("all")

    fig = marginal_plot(P, "t", "p")
    assert fig.axes[1].get_ylabel().endswith(r"\mathrm{A}$")
    plt.close("all")


def test_plottable_array_and_units_tolerates_unparseable_strings() -> None:
    """User-supplied unit strings that are not unit symbols still get a
    correct power-of-ten label instead of raising."""
    import numpy as np

    arr = np.array([1e3, 2e3])
    _, _, units_str, _, _ = plottable_array_and_units(arr, "counts")
    assert units_str == "1e3 counts"
    _, _, units_str, _, _ = plottable_array_and_units(arr, "counts", nice=False)
    assert units_str == "counts"


def test_mathlabel_units_only() -> None:
    """mathlabel with no keys is a units-only label, as used by the
    marginal histograms."""
    assert mathlabel(units="") == ""
    assert mathlabel(units="A") == r"$\mathrm{A}$"
    assert mathlabel(units="C/(eV/c)") == r"$\mathrm{C}/(\mathrm{eV}/\mathrm{c})$"
    # Unparseable falls back to mathrm with * -> cdot.
    assert r"{\cdot}" in mathlabel(units="foo*bar")
    assert mathlabel(units="eV/c", tex=False) == "eV/c"


def test_single_particle_plots_draw() -> None:
    """Plots for a single particle (dimensionless densities, degenerate
    histograms) must render without mathtext errors."""
    import matplotlib.pyplot as plt

    P = single_particle(pz=10e6)
    for key in ("weight", "x", "p"):
        P.plot("z", key)
        plt.gcf().canvas.draw()  # force label rendering
        plt.close("all")
