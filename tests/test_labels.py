from __future__ import annotations

import pytest

from beamphysics.labels import mathlabel
from beamphysics.units import pmd_unit, sqrt_unit


@pytest.mark.parametrize(
    ("keys", "units", "tex", "expected"),
    [
        # A pmd_unit renders through to_tex.
        pytest.param(
            ("x",),
            pmd_unit("eV/c"),
            True,
            r"$x~(\mathrm{eV}/\mathrm{c})$",
            id="pmd_unit-compound",
        ),
        pytest.param(
            ("x_bar",),
            sqrt_unit(pmd_unit("m")),
            True,
            r"$\overline{x}~(\sqrt{\mathrm{m}})$",
            id="pmd_unit-sqrt",
        ),
        # A str that parses as a unit symbol renders the same way.
        pytest.param(
            ("x",),
            "kg*m/s",
            True,
            r"$x~(\mathrm{kg}{\cdot}\mathrm{m}/\mathrm{s})$",
            id="str-parses",
        ),
        # A str that is not a unit symbol falls back to mathrm, * -> cdot.
        pytest.param(
            ("x",),
            "foo*bar",
            True,
            r"$x~(\mathrm{ foo{\cdot}bar })$",
            id="str-fallback-cdot",
        ),
        # Multiple keys (the docstring example).
        pytest.param(
            ("x_bar", "sigma_x"),
            "µC",
            True,
            r"$\overline{x}, \sigma_{ x }~(\mathrm{µC})$",
            id="multiple-keys",
        ),
        # A dimensionless unit (empty symbol) adds no "()" in either branch;
        # pmd_unit("1") stores the symbol "1" and is annotated as such.
        pytest.param(("x",), pmd_unit(""), True, "$x$", id="dimensionless-tex"),
        pytest.param(("x",), pmd_unit(""), False, "x", id="dimensionless-plain"),
        pytest.param(("x",), pmd_unit("1"), False, "x (1)", id="unity-plain"),
        # No units.
        pytest.param(("x",), None, True, "$x$", id="no-units-tex"),
        pytest.param(("x",), None, False, "x", id="no-units-plain"),
        pytest.param(("x",), "", True, "$x$", id="empty-units-tex"),
        # tex=False uses the plain symbol.
        pytest.param(("x",), pmd_unit("eV/c"), False, "x (eV/c)", id="plain-pmd_unit"),
        pytest.param(("x", "y"), "µC", False, "x, y (µC)", id="plain-two-keys"),
        # No keys: a units-only label, as used by the marginal histograms.
        pytest.param((), "A", True, r"$\mathrm{A}$", id="units-only"),
        pytest.param(
            (),
            "C/(eV/c)",
            True,
            r"$\mathrm{C}/(\mathrm{eV}/\mathrm{c})$",
            id="units-only-grouped",
        ),
        pytest.param(
            (),
            "foo*bar",
            True,
            r"$\mathrm{ foo{\cdot}bar }$",
            id="units-only-fallback",
        ),
        pytest.param((), "eV/c", False, "eV/c", id="units-only-plain"),
        pytest.param((), "", True, "", id="units-only-empty"),
        pytest.param((), None, False, "", id="units-only-none-plain"),
    ],
)
def test_mathlabel(keys, units, tex, expected) -> None:
    assert mathlabel(*keys, units=units, tex=tex) == expected
