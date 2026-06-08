from __future__ import annotations
import numpy as np
import pytest
import itertools
import math

from beamphysics.units import (
    NAMED_UNITS,
    SHORT_PREFIX_FACTOR,
    dimension,
    nice_array,
    nice_scale_prefix,
    pg_units,
    plottable_array,
    pmd_unit,
    unit as deprecated_unit,
)


def test_deprecated_unit():
    with pytest.deprecated_call():
        assert deprecated_unit("T") == pmd_unit("T")


def assert_round_trip(u: pmd_unit) -> None:
    """Re-parsing the unit's symbol must reproduce its dimension and SI value.

    Parse / dimension / SI failures all surface as pytest errors with the
    offending symbol in the message — no try/except swallowing them.
    """
    reparsed = pmd_unit(u.unitSymbol)
    assert (
        reparsed.unitDimension == u.unitDimension
    ), f"{u.unitSymbol!r}: dimension {reparsed.unitDimension} != {u.unitDimension}"
    assert reparsed.unitSI == u.unitSI or math.isclose(
        reparsed.unitSI, u.unitSI, rel_tol=1e-9
    ), f"{u.unitSymbol!r}: SI {reparsed.unitSI} != {u.unitSI}"


def _round_trips(u: pmd_unit) -> bool:
    """Bool predicate wrapping ``assert_round_trip`` for use as a test guard."""
    try:
        assert_round_trip(u)
    except (AssertionError, ValueError, KeyError):
        return False
    return True


@pytest.mark.parametrize(
    ("build", "expected_symbol"),
    [
        # Atomic matches (the float-tolerant path).
        pytest.param(lambda: pmd_unit("eV/c") * pmd_unit("c"), "eV", id="eV/c*c->eV"),
        pytest.param(lambda: pmd_unit("J") / pmd_unit("C"), "V", id="J/C->V"),
        pytest.param(lambda: pmd_unit("V") * pmd_unit("A"), "W", id="V*A->W"),
        # Compound matches.
        pytest.param(lambda: pmd_unit("T") * pmd_unit("m"), "T*m", id="T*m"),
        pytest.param(
            lambda: pmd_unit("T") / pmd_unit("m") / pmd_unit("m") * pmd_unit("m"),
            "T/m",
            id="T/m",
        ),
    ],
)
def test_simplify_examples(build, expected_symbol: str) -> None:
    assert build().simplify().unitSymbol == expected_symbol


def test_simplify_electric_potential_volt() -> None:
    # V = J/C = kg*m^2*s^-3*A^-1
    assert pmd_unit("V").unitDimension == (2, 1, -3, -1, 0, 0, 0)


def test_simplify_division_by_compound_round_trips() -> None:
    # Regression: a compound divisor (e.g. "W") must not be emitted as a
    # bare "a/b/c" string that the flat, left-associative parser regroups.
    simplified = (pmd_unit("V/m") / pmd_unit("W")).simplify()
    assert_round_trip(simplified)


@pytest.mark.parametrize("si", [1e-3, 1e3, 1e-6, 2.0])
def test_simplify_dimensionless_never_bare_prefix(si: float) -> None:
    # Regression: an SI prefix on a dimensionless unit produced bogus symbols
    # like "m" (re-parses as metres) or "k" (unparseable). The result must
    # stay dimensionless and never be a bare SI prefix.
    result = pmd_unit("ratio", si, (0, 0, 0, 0, 0, 0, 0)).simplify()
    assert result.unitDimension == (0, 0, 0, 0, 0, 0, 0)
    assert result.unitSymbol not in SHORT_PREFIX_FACTOR
    try:
        reparsed = pmd_unit(result.unitSymbol)
    except (ValueError, KeyError):
        # No simplification found: self returned with its placeholder symbol.
        return
    # If the symbol is parseable, it must not re-parse to a dimensioned unit.
    assert reparsed.unitDimension == (0, 0, 0, 0, 0, 0, 0)


def test_simplify_never_breaks_round_trip() -> None:
    # Across every product/quotient of named units, simplify() must never
    # turn a round-tripping unit into a non-round-tripping one.
    named = [u for u in NAMED_UNITS if u.unitSymbol]
    for a, b in itertools.product(named, named):
        for u in (a * b, a / b):
            if not _round_trips(u):
                continue  # construction already mangled the symbol; not our concern
            assert_round_trip(u.simplify())


@pytest.mark.parametrize(
    ("symbol", "expected_si", "base_symbol"),
    [
        ("keV", 1e3, "eV"),
        ("MeV", 1e6, "eV"),
        ("meV", 1e-3, "eV"),
        ("mm", 1e-3, "m"),
        ("cm", 1e-2, "m"),
        ("kV", 1e3, "V"),
        ("mg", 1e-3, "g"),  # milli-gram: 1e-3 * gram(1e-3 kg) = 1e-6 kg
        ("GHz", 1e9, "Hz"),
        ("mrad", 1e-3, "rad"),
        ("mA", 1e-3, "A"),
    ],
)
def test_si_prefix_parsing(symbol: str, expected_si: float, base_symbol: str) -> None:
    base = pmd_unit(base_symbol)
    u = pmd_unit(symbol)
    assert u.unitDimension == base.unitDimension
    assert math.isclose(u.unitSI, expected_si * base.unitSI, rel_tol=1e-12)


@pytest.mark.parametrize("symbol", ["k", "M", "da", "k1", "min", "Pa", "xyz"])
def test_bare_prefix_and_junk_rejected(symbol: str) -> None:
    # A bare SI prefix is not a unit, and the prefix fallback must not parse
    # arbitrary strings (the remainder has to be a known unit).
    with pytest.raises(ValueError):
        pmd_unit(symbol)


def test_prefix_does_not_shadow_base_units() -> None:
    # Prefix letters that collide with unit symbols must still resolve to the
    # unit (lookup happens before the prefix fallback): c=speed of light (not
    # centi), T=tesla (not tera), g=gram, cd=candela.
    assert math.isclose(pmd_unit("c").unitSI, 299792458.0)
    assert pmd_unit("c").unitDimension == dimension("velocity")
    assert pmd_unit("T").unitDimension == dimension("magnetic_field")
    assert pmd_unit("g").unitDimension == dimension("mass")
    assert pmd_unit("cd").unitDimension == (0, 0, 0, 0, 0, 0, 1)


def test_simplify_prefixed_output_round_trips() -> None:
    # simplify() emits prefixed atomic symbols (e.g. "mg", "kV"); the parser
    # must read them back to the same unit.
    for u in (
        pmd_unit("g") * pmd_unit("g") / pmd_unit("kg"),  # 1e-6 kg -> "mg"
        pmd_unit("ratio_v", 1e3, dimension("electric_potential")),  # -> "kV"
        pmd_unit("ratio_m", 1e-6, dimension("mass")),  # -> "mg"
    ):
        assert_round_trip(u.simplify())


def test_simplify_dimensionless_ratio_not_compound() -> None:
    # Regression: a pure-number ratio must not be "simplified" into a same-
    # dimension compound like "kg/g" (mass/mass == 1000). It has no meaningful
    # named form, so simplify returns it unchanged.
    u = pmd_unit("mrad/µrad")  # 1000, dimensionless
    s = u.simplify()
    assert s.unitDimension == (0, 0, 0, 0, 0, 0, 0)
    assert s.unitSymbol == u.unitSymbol  # unchanged
    assert s.unitSI == u.unitSI  # value preserved exactly
    # An exact-1 dimensionless still collapses to the empty symbol.
    assert pmd_unit("r", 1.0, (0,) * 7).simplify().unitSymbol == ""


def test_ohm_unit_and_alias() -> None:
    ohm = pmd_unit("Ω")
    assert ohm.unitDimension == (2, 1, -3, -2, 0, 0, 0)  # V/A
    assert math.isclose(ohm.unitSI, 1.0)
    # ASCII alias resolves to the same dimension/value.
    alias = pmd_unit("Ohm")
    assert alias.unitDimension == ohm.unitDimension
    assert math.isclose(alias.unitSI, ohm.unitSI)
    # simplify recognizes V/A as the ohm, and prefixes/compounds work.
    assert (pmd_unit("V") / pmd_unit("A")).simplify().unitSymbol == "Ω"
    assert math.isclose(pmd_unit("kΩ").unitSI, 1e3)
    assert pmd_unit("Ω/m").unitDimension == (1, 1, -3, -2, 0, 0, 0)


@pytest.mark.parametrize(
    "symbol",
    # Unit strings documented/plotted in the wakefields module must all parse.
    ["V/C/m", "V/C", "Ohm/m", "Ω/m", "V/pC/m", "MV/m", "1/mm"],
)
def test_wakefield_unit_strings_parse(symbol: str) -> None:
    assert_round_trip(pmd_unit(symbol))


def test_sqrt_unit_halves_dimension() -> None:
    # Regression: sqrt_unit used integer floor division (x // 2), so sqrt(m)
    # collapsed to dimensionless instead of m^0.5.
    from beamphysics.units import sqrt_unit

    root_m = sqrt_unit(pmd_unit("m"))
    assert root_m.unitDimension == (0.5, 0, 0, 0, 0, 0, 0)
    assert math.isclose(root_m.unitSI, 1.0)
    # x_bar/px_bar are defined via sqrt_unit and document units of sqrt(m).
    assert pg_units("x_bar").unitDimension == (0.5, 0, 0, 0, 0, 0, 0)


def test_cov_units_with_prefixed_subkey() -> None:
    # Regression: pg_units used key.strip("cov_"), which strips the char set
    # {c,o,v,_} and mangled subkeys like "charge" -> "harge" (KeyError).
    u = pg_units("cov_charge__x")
    assert u.unitDimension == (1, 0, 1, 1, 0, 0, 0)  # charge * length
    # Order independence and the common case still work.
    assert pg_units("cov_x__charge").unitDimension == (1, 0, 1, 1, 0, 0, 0)
    assert pg_units("cov_x__px").unitDimension == (2, 1, -1, 0, 0, 0, 0)


def test_parsing_compound_does_not_mutate_named_units() -> None:
    # Regression: parsing a compound whose first token is a known unit used to
    # mutate the shared NAMED_UNITS entry in place (e.g. parsing "m*rad" rewrote
    # the global "m" symbol to "m*rad"), corrupting all later simplify() calls.
    before = [(u.unitSymbol, u.unitSI, u.unitDimension) for u in NAMED_UNITS]
    for symbol in ("m*rad", "rad*s", "rad*eV", "m/rad", "T*m", "kg*m/s"):
        pmd_unit(symbol)
    after = [(u.unitSymbol, u.unitSI, u.unitDimension) for u in NAMED_UNITS]
    assert before == after


def test_smoke_properties() -> None:
    u = pmd_unit("K")
    u.unitSymbol
    u.unitSI
    u.unitDimension


def test_equality() -> None:
    assert pmd_unit("T") == pmd_unit("T")
    assert pmd_unit("T") != pmd_unit("eV")
    assert pmd_unit("T") != "foo"


def test_hashability() -> None:
    assert len({pmd_unit("T"), pmd_unit("eV"), pmd_unit("T")}) == 2


def test_divide() -> None:
    kg_over_g = pmd_unit("kg") / pmd_unit("g")
    assert kg_over_g.unitDimension == (0,) * 7
    assert kg_over_g.unitSI == 1000.0


def test_multiply() -> None:
    momentum = pmd_unit("kg") * pmd_unit("m") / pmd_unit("s")
    assert momentum.unitDimension == dimension("momentum")


def test_multiply_same_symbol_round_trips() -> None:
    """``m * m`` (and longer chains) must produce a re-parseable symbol.

    Previously the s1 == s2 branch emitted ``"(m)^2"``, which the parser
    rejected because bare parentheses are not a supported grouping.
    """
    squared = pmd_unit("m") * pmd_unit("m")
    assert squared.unitDimension == (2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert_round_trip(squared)

    cubed = pmd_unit("m") * pmd_unit("m") * pmd_unit("m")
    assert cubed.unitDimension == (3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert_round_trip(cubed)


def test_negative_power_normalizes_negative_zero() -> None:
    """Negative exponents must not produce ``-0.0`` in the dimension tuple.

    ``-1.0 * 0.0 == -0.0`` in IEEE float; while equality and hashing still
    work, the repr is noisy and dict-key behavior across construction paths
    is less robust if some tuples carry ``-0.0`` and others ``0.0``.
    """
    inv_hz = pmd_unit("1") / pmd_unit("Hz")
    for i, d in enumerate(inv_hz.unitDimension):
        assert (
            math.copysign(1.0, d) == 1.0
        ), f"dimension index {i} is signed-negative: {d!r}"


@pytest.mark.parametrize(
    "symbol",
    ["eV / c", " eV/c", "eV/c ", "kg * m / s", "kg*m / s", " V/m "],
)
def test_whitespace_around_operators_tolerated(symbol: str) -> None:
    """Whitespace around operators (and leading/trailing) must parse the same
    as the unspaced form. Internal whitespace inside a token is not allowed —
    multi-token named units (e.g. ``"charge #"``) are looked up verbatim."""
    canonical = pmd_unit(symbol.replace(" ", ""))
    assert pmd_unit(symbol).unitDimension == canonical.unitDimension
    assert pmd_unit(symbol).unitSI == canonical.unitSI


def test_legacy_multitoken_name_still_parses() -> None:
    """The ``"charge #"`` legacy entry must still resolve via direct lookup,
    even though it contains an internal space.
    """
    u = pmd_unit("charge #")
    assert u.unitDimension == dimension("charge")
    assert u.unitSI == 1


@pytest.mark.parametrize(
    ("symbol", "expected_unit"),
    [
        pytest.param("A*s", pmd_unit("A") * pmd_unit("s")),
        pytest.param("A/s", pmd_unit("A") / pmd_unit("s")),
        pytest.param("A/s/s", pmd_unit("A") / pmd_unit("s") / pmd_unit("s")),
    ],
)
def test_unit(symbol: str, expected_unit: pmd_unit) -> None:
    assert pmd_unit(symbol) == expected_unit


@pytest.mark.parametrize(
    ("prefix", "factor"),
    [
        pytest.param(prefix, factor, id=prefix)
        for prefix, factor in SHORT_PREFIX_FACTOR.items()
    ],
)
def test_smoke_nice_scale_prefix(prefix: str, factor: float) -> None:
    res = nice_scale_prefix(factor)
    print(f"{factor=} raw {prefix=} ->", res)


@pytest.mark.parametrize(
    ("value", "prefix", "factor"),
    [
        (1e-3, "m", 1e-3),
        (1000.0, "k", 1000.0),
        (1e6, "M", 1e6),
        (1e9, "G", 1e9),
        (1e25, "Y", 1e24),
        (1e-25, "y", 1e-24),
        (0.0, "", 1),
    ],
)
def test_nice_scale_prefix(value: float, prefix: str, factor: float) -> None:
    assert nice_scale_prefix(value) == (factor, prefix)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(1.0, (1.0, 1, "")),
        pytest.param([1.0], (1.0, 1, "")),
        pytest.param([1.0, 1.0, 1.0], (1.0, 1, "")),
    ],
)
def test_nice_array(
    value: float | list[float],
    expected: tuple[np.ndarray | float, float, str],
) -> None:
    expected_scaled, expected_scaling, expected_prefix = expected
    res_scaled, res_scaling, res_prefix = nice_array(value)
    np.testing.assert_allclose(actual=res_scaled, desired=expected_scaled)
    assert expected_scaling == res_scaling
    assert res_prefix == expected_prefix


@pytest.mark.parametrize(
    ("value",),
    [
        pytest.param(1.0),
        pytest.param([1.0]),
        pytest.param([1.0, 1.0, 1.0]),
    ],
)
def test_plottable_array_smoke(value: float | list[float]) -> None:
    for lim, nice in [
        (None, True),
        (None, False),
        ((-10, 10), False),
        ((None, 10), False),
        ((None, None), False),
        ((-10, None), False),
    ]:
        res, *_ = plottable_array(value, lim=lim, nice=nice)
        np.testing.assert_allclose(actual=res, desired=value)
