from __future__ import annotations

from beamphysics.labels import mathlabel
from beamphysics.units import pmd_unit, sqrt_unit


def test_mathlabel_renders_pmd_unit_via_to_tex() -> None:
    assert mathlabel("x", units=pmd_unit("eV/c")) == r"$x~(\mathrm{eV}/\mathrm{c})$"
    assert (
        mathlabel("x_bar", units=sqrt_unit(pmd_unit("m")))
        == r"$\overline{x}~(\sqrt{\mathrm{m}})$"
    )


def test_mathlabel_parses_unit_strings() -> None:
    # A str that parses as a unit symbol renders through to_tex.
    assert mathlabel("x", units="kg*m/s") == (
        r"$x~(\mathrm{kg}{\cdot}\mathrm{m}/\mathrm{s})$"
    )


def test_mathlabel_unparseable_string_fallback_keeps_cdot() -> None:
    # Regression: the fallback for strings that are not valid unit symbols
    # must still replace "*" with {\cdot} (as the pre-to_tex code did).
    label = mathlabel("x", units="foo*bar")
    assert r"{\cdot}" in label
    assert "*" not in label


def test_mathlabel_dimensionless_unit_adds_no_parens() -> None:
    # Regression: a pmd_unit is always truthy, so the non-TeX branch used to
    # emit "x ()" for a dimensionless unit (empty symbol). Both branches must
    # omit the unit annotation entirely.
    dimensionless = pmd_unit("1")
    assert mathlabel("x", units=pmd_unit(""), tex=False) == "x"
    assert mathlabel("x", units=pmd_unit(""), tex=True) == "$x$"
    # pmd_unit("1") stores the symbol "1" and is annotated as such.
    assert mathlabel("x", units=dimensionless, tex=False) == "x (1)"


def test_mathlabel_no_units() -> None:
    assert mathlabel("x", tex=False) == "x"
    assert mathlabel("x", units=None, tex=True) == "$x$"
    assert mathlabel("x", units="", tex=True) == "$x$"


def test_mathlabel_non_tex_uses_symbol() -> None:
    assert mathlabel("x", units=pmd_unit("eV/c"), tex=False) == "x (eV/c)"
    assert mathlabel("x", "y", units="µC", tex=False) == "x, y (µC)"
