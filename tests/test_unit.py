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
    # simplify() output must re-parse to the same unit, including when the
    # input was built by dividing by a compound unit.
    simplified = (pmd_unit("V/m") / pmd_unit("W")).simplify()
    assert_round_trip(simplified)


@pytest.mark.parametrize("si", [1e-3, 1e3, 1e-6, 2.0])
def test_simplify_dimensionless_never_bare_prefix(si: float) -> None:
    # A simplified dimensionless unit must stay dimensionless and must never
    # be a bare SI prefix (which would be unparseable, or collide with a
    # named unit like "m").
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
    # A pure-number ratio has no meaningful named form: simplify must return
    # it unchanged, never a same-dimension compound like "kg/g" (== 1000).
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
    # sqrt(m) has dimension m^0.5, carried as a fractional exponent.
    from beamphysics.units import sqrt_unit

    root_m = sqrt_unit(pmd_unit("m"))
    assert root_m.unitDimension == (0.5, 0, 0, 0, 0, 0, 0)
    assert math.isclose(root_m.unitSI, 1.0)
    # x_bar/px_bar are defined via sqrt_unit and document units of sqrt(m).
    assert pg_units("x_bar").unitDimension == (0.5, 0, 0, 0, 0, 0, 0)


def test_norm_emit_4d_stored_as_m_squared() -> None:
    """``norm_emit_4d`` is area-like (m^2). The stored symbol should be
    ``m^2`` rather than the equivalent-but-clumsy ``m*m`` so plot labels
    render cleanly."""
    u = pg_units("norm_emit_4d")
    assert u.unitSymbol == "m^2"
    assert u.unitDimension == (2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert u.unitSI == 1


def test_cov_units_with_prefixed_subkey() -> None:
    # The "cov_" prefix must be removed exactly (removeprefix, not
    # strip("cov_"), which strips the char set {c,o,v,_} and would mangle
    # subkeys like "charge").
    u = pg_units("cov_charge__x")
    assert u.unitDimension == (1, 0, 1, 1, 0, 0, 0)  # charge * length
    # Order independence and the common case still work.
    assert pg_units("cov_x__charge").unitDimension == (1, 0, 1, 1, 0, 0, 0)
    assert pg_units("cov_x__px").unitDimension == (2, 1, -1, 0, 0, 0, 0)


def test_parsing_compound_does_not_mutate_named_units() -> None:
    # Parsing must be side-effect-free: the shared NAMED_UNITS entries stay
    # untouched no matter what symbols are parsed.
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


def test_equality_is_value_based_per_standard() -> None:
    """Per the openPMD standard, a unit is defined by (unitSI, unitDimension);
    the symbol is presentational. Two units denoting the same physical
    quantity must therefore compare equal regardless of how they were
    spelled, and survive floating-point drift introduced by arithmetic."""
    # Cross-spelling equivalence — same SI and dimension via different syntax.
    assert pmd_unit("1/s") == pmd_unit("Hz")
    assert pmd_unit("s^-1") == pmd_unit("Hz")
    # Arithmetic round-trip: (eV/c) * c must equal eV even though the
    # intermediate float math doesn't reproduce e_charge bit-exactly.
    assert pmd_unit("eV/c") * pmd_unit("c") == pmd_unit("eV")
    # Different physical scale must remain unequal.
    assert pmd_unit("eV") != pmd_unit("J")  # same dim, different unitSI
    # Hash invariant: equal objects must share a hash.
    assert hash(pmd_unit("1/s")) == hash(pmd_unit("Hz"))


def test_hash_distinguishes_same_dimension_units() -> None:
    """The hash includes the rounded unitSI, not just the dimension, so
    units that share a dimension but differ in scale (eV, keV, J) get
    different hashes instead of all colliding in one bucket. Equal units
    still hash equal."""
    energy = [pmd_unit("eV"), pmd_unit("J"), pmd_unit("keV"), pmd_unit("MeV")]
    # All share the energy dimension but are pairwise unequal (different SI).
    assert len({hash(u) for u in energy}) == len(energy)
    # Invariant still holds for an arithmetic-drifted equal pair.
    rebuilt = pmd_unit("eV/c") * pmd_unit("c")
    assert rebuilt == pmd_unit("eV")
    assert hash(rebuilt) == hash(pmd_unit("eV"))


def test_hashability() -> None:
    assert len({pmd_unit("T"), pmd_unit("eV"), pmd_unit("T")}) == 2


def test_pow_operator_matches_power_unit() -> None:
    """``u ** n`` is sugar for ``power_unit(u, n)`` and must agree on the
    stored symbol, SI factor, and dimension for both integer and fractional
    exponents (including compound bases that need distribution)."""
    from beamphysics.units import power_unit

    for base, exp in [
        (pmd_unit("m"), 2),
        (pmd_unit("m"), 0.5),
        (pmd_unit("eV*s"), 2),
        (pmd_unit("m^2*s^-1"), 3),
    ]:
        via_op = base**exp
        via_fn = power_unit(base, exp)
        assert via_op.unitSymbol == via_fn.unitSymbol
        assert via_op.unitSI == via_fn.unitSI
        assert via_op.unitDimension == via_fn.unitDimension


def test_compatible_with_is_dimension_only() -> None:
    """``compatible_with`` is a pure-dimension check: equal-dimension units
    are compatible regardless of SI scale, and non-pmd_unit operands raise."""
    # Equal dimension, different SI scale -> still compatible.
    assert pmd_unit("eV").compatible_with(pmd_unit("J"))
    assert pmd_unit("J").compatible_with(pmd_unit("eV"))
    # Different syntax, same physical unit.
    assert pmd_unit("Hz").compatible_with(pmd_unit("1/s"))
    # Different dimensions -> incompatible.
    assert not pmd_unit("m").compatible_with(pmd_unit("s"))
    # Reflexive and symmetric.
    u = pmd_unit("T*m")
    assert u.compatible_with(u)
    # Type guard.
    with pytest.raises(TypeError):
        pmd_unit("m").compatible_with("m")


def test_conversion_factor_to_round_trips_and_rejects_mismatch() -> None:
    """``conversion_factor_to`` returns ``self.unitSI / other.unitSI`` after
    a dimension check; multiplying a value by it must move it from ``self``
    to ``other``. Incompatible dimensions raise ``ValueError``."""
    # Same-dimension conversion: 1 eV in joules == e_charge.
    fac = pmd_unit("eV").conversion_factor_to(pmd_unit("J"))
    assert fac == pmd_unit("eV").unitSI / pmd_unit("J").unitSI
    # Round-trip: km -> m -> km.
    km, m = pmd_unit("km"), pmd_unit("m")
    assert km.conversion_factor_to(m) == 1000.0
    assert m.conversion_factor_to(km) == 0.001
    # Equal units yield factor 1.
    assert pmd_unit("Hz").conversion_factor_to(pmd_unit("1/s")) == 1.0
    # Incompatible dimensions raise.
    with pytest.raises(ValueError, match="Incompatible units"):
        pmd_unit("m").conversion_factor_to(pmd_unit("s"))


def test_divide() -> None:
    kg_over_g = pmd_unit("kg") / pmd_unit("g")
    assert kg_over_g.unitDimension == (0,) * 7
    assert kg_over_g.unitSI == 1000.0


def test_multiply() -> None:
    momentum = pmd_unit("kg") * pmd_unit("m") / pmd_unit("s")
    assert momentum.unitDimension == dimension("momentum")


def test_multiply_same_symbol_round_trips() -> None:
    """``m * m`` (and longer chains) must produce a re-parseable symbol."""
    squared = pmd_unit("m") * pmd_unit("m")
    assert squared.unitDimension == (2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert_round_trip(squared)

    cubed = pmd_unit("m") * pmd_unit("m") * pmd_unit("m")
    assert cubed.unitDimension == (3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert_round_trip(cubed)


@pytest.mark.parametrize(
    ("numerator", "divisor"),
    [
        ("V/m", "eV/c"),
        ("V", "m*s"),
        ("W", "m^2*s^-1"),
        ("T", "kg*m/s"),
        ("eV/c", "eV/c/s"),
    ],
)
def test_divide_by_compound_divisor_round_trips(numerator: str, divisor: str) -> None:
    """Dividing by a compound unit parenthesizes the divisor, so the symbol
    re-parses to the same unit (``"a/b/c"`` means ``(a/b)/c``, a different
    grouping)."""
    assert_round_trip(pmd_unit(numerator) / pmd_unit(divisor))


def test_parenthesized_grouping_parses() -> None:
    """``(expr)`` groups: ``V/(eV/c)`` divides by the whole group, while
    ``V/eV/c`` means ``(V/eV)/c``."""
    u = pmd_unit("V/(eV/c)")
    assert u == pmd_unit("V") / pmd_unit("eV/c")
    assert u.unitSymbol == "V/(eV/c)"
    assert_round_trip(u)
    assert u != pmd_unit("V/eV/c")  # grouping matters

    # Nested groups.
    nested = pmd_unit("W/(V/(eV/c))")
    assert nested == pmd_unit("W") / (pmd_unit("V") / pmd_unit("eV/c"))
    assert_round_trip(nested)

    # Redundant parens around an atomic token are dropped.
    assert pmd_unit("V/(m)").unitSymbol == "V/m"

    # Whitespace inside a group canonicalizes.
    assert pmd_unit("V / ( eV / c )").unitSymbol == "V/(eV/c)"


def test_parenthesized_group_with_power() -> None:
    """A group can carry an exponent: ``(eV*s)^2`` is ``(eV*s)`` squared,
    not ``eV*(s^2)``. Fractional exponents work through the same branch."""
    assert pmd_unit("(eV*s)^2") == pmd_unit("eV*s") ** 2
    assert_round_trip(pmd_unit("(eV*s)^2"))
    assert pmd_unit("(eV/c)^2") == pmd_unit("eV/c") ** 2
    assert pmd_unit("(m^2*s^2)^(1/2)") == pmd_unit("m*s")
    # power_unit distribution emits paren-power tokens that re-parse.
    assert_round_trip(pmd_unit("V/(eV/c)") ** 2)


@pytest.mark.parametrize("symbol", ["()", "V/()", "(m", "m)", "(a)(b)", "(m*)"])
def test_malformed_parens_rejected(symbol: str) -> None:
    with pytest.raises(ValueError):
        pmd_unit(symbol)


def test_divide_units_emits_parenthesized_divisor() -> None:
    """Dividing by a compound unit wraps the divisor in parens — readable and
    round-trips now that the parser understands grouping."""
    u = pmd_unit("V/m") / pmd_unit("eV/c")
    assert u.unitSymbol == "V/m/(eV/c)"
    assert_round_trip(u)


def test_divide_with_empty_numerator_symbol() -> None:
    """The dimensionless identity (empty symbol) as numerator must emit
    ``"1/..."``, not a malformed symbol starting with a bare operator."""
    inv_m = pmd_unit("") / pmd_unit("m")
    assert inv_m.unitSymbol == "1/m"
    assert_round_trip(inv_m)


def test_unicode_homoglyphs_normalized() -> None:
    """A pasted Ω or µ must parse no matter which of its visually identical
    Unicode twins it is: OHM SIGN (U+2126) is converted to GREEK CAPITAL
    OMEGA (U+03A9, the character in NAMED_UNITS), and GREEK SMALL MU
    (U+03BC) to MICRO SIGN (U+00B5, the character in SHORT_PREFIX_FACTOR).
    Explicit escapes are used here because the twins look the same in
    source code.
    """
    assert pmd_unit("\u2126") == pmd_unit("\u03a9")  # OHM SIGN == GREEK OMEGA
    assert pmd_unit("\u2126").unitSymbol == "\u03a9"
    assert math.isclose(pmd_unit("k\u2126").unitSI, 1e3)
    assert pmd_unit("\u03bcm") == pmd_unit("\u00b5m")  # Greek mu == micro sign
    assert pmd_unit("\u03bcm").unitSymbol == "\u00b5m"
    assert math.isclose(pmd_unit("\u03bcm").unitSI, 1e-6)
    assert math.isclose(pmd_unit("\u03bcrad").unitSI, 1e-6)


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


def test_power_unit_distributes_over_compound() -> None:
    """``power_unit`` distributes the exponent across compound symbols:
    ``(eV*s)^2`` emits ``eV^2*s^2``, matching the dimension and round-tripping
    through the parser."""
    from beamphysics.units import power_unit

    squared = power_unit(pmd_unit("eV*s"), 2)
    assert squared.unitDimension == (4.0, 2.0, -2.0, 0.0, 0.0, 0.0, 0.0)
    assert_round_trip(squared)

    # Existing exponents inside the compound must compose, not be replaced.
    compound = pmd_unit("m^2*s^-1")
    cubed = power_unit(compound, 3)
    assert cubed.unitDimension == tuple(d * 3 for d in compound.unitDimension)
    assert_round_trip(cubed)

    # Halving a squared compound returns the original dimension.
    halved = power_unit(pmd_unit("m^2*s^2"), 0.5)
    assert halved.unitDimension == (1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    assert_round_trip(halved)


def test_sqrt_unit_symbol_round_trips() -> None:
    """``sqrt_unit`` emits the canonical ``sqrt(X)`` form (no LaTeX, no
    whitespace) so the result re-parses cleanly to itself.
    """
    from beamphysics.units import sqrt_unit

    root_m = sqrt_unit(pmd_unit("m"))
    assert root_m.unitSymbol == "sqrt(m)"
    assert_round_trip(root_m)

    # Compound interior also round-trips.
    root_compound = sqrt_unit(pmd_unit("kg*m"))
    assert root_compound.unitSymbol == "sqrt(kg*m)"
    assert_round_trip(root_compound)


def test_sqrt_symbol_whitespace_normalized() -> None:
    """Whitespace inside ``sqrt(...)`` is stripped on parse, so all spellings
    canonicalize to the same stored symbol (and therefore compare equal).
    """
    canonical = pmd_unit("sqrt(m)")
    for spelling in ("sqrt( m )", "  sqrt(m)  ", "sqrt(m )", "sqrt( m)"):
        u = pmd_unit(spelling)
        assert u.unitSymbol == "sqrt(m)"
        assert u == canonical
        assert hash(u) == hash(canonical)


def test_latex_sqrt_form_not_accepted() -> None:
    """The LaTeX-style ``\\sqrt{...}`` form is not part of the supported unit
    grammar and must be rejected by the parser.
    """
    with pytest.raises(ValueError):
        pmd_unit(r"\sqrt{m}")
    with pytest.raises(ValueError):
        pmd_unit(r"\sqrt{ m }")


def test_simplify_skips_compound_numerator() -> None:
    """``simplify`` emits only ``atomic*atomic`` and ``atomic/atomic``
    compounds — a compound operand is not a simplification."""
    # Any simplify output must not be of the form "X/Y*Z" or "X*Y/Z" where
    # one of the operands is itself compound; the constraint we test is the
    # round-trip property (which would fail under ambiguous emission).
    for a, b in itertools.product(NAMED_UNITS, NAMED_UNITS):
        for u in (a * b, a / b):
            if not _round_trips(u):
                continue
            simplified = u.simplify()
            sym = simplified.unitSymbol
            # Symbol must be either atomic, or pure compound of atomic
            # operands joined by * and /. We assert specifically: no operand
            # within the top-level split is itself compound.
            from beamphysics.units import _tokenize_compound

            parts, _ops = _tokenize_compound(sym)
            for p in parts:
                # Each part must itself contain no top-level * or /.
                p_parts, _ = _tokenize_compound(p)
                assert len(p_parts) == 1, f"{sym!r}: token {p!r} is itself compound"


@pytest.mark.parametrize("symbol", ["m*", "/m", "m//s", "m**s", "*", "/", "m*/s"])
def test_parser_rejects_empty_operands(symbol: str) -> None:
    """Operators with no operand on one side must be flagged, not silently
    parsed via the empty-string identity.
    """
    with pytest.raises(ValueError):
        pmd_unit(symbol)


@pytest.mark.parametrize(
    "symbol",
    ["m ^ 2", "m^ 2", "m ^2", "Hz ^ -1", "s^ -2"],
)
def test_whitespace_around_caret_tolerated(symbol: str) -> None:
    """Whitespace around ``^`` must parse the same as the unspaced form."""
    canonical = pmd_unit(symbol.replace(" ", ""))
    assert pmd_unit(symbol).unitDimension == canonical.unitDimension
    assert pmd_unit(symbol).unitSI == canonical.unitSI


def test_symbol_whitespace_normalized_in_stored_form() -> None:
    """Compound symbols with whitespace around operators must canonicalize.

    Without normalization, ``pmd_unit("eV / c") != pmd_unit("eV/c")`` even
    though they denote the same unit, and they hash differently.
    """
    spaced = pmd_unit("eV / c")
    tight = pmd_unit("eV/c")
    assert spaced.unitSymbol == tight.unitSymbol == "eV/c"
    assert spaced == tight
    assert hash(spaced) == hash(tight)

    # Three-token compound also normalizes.
    assert pmd_unit("kg * m / s").unitSymbol == "kg*m/s"


@pytest.mark.parametrize("bad_si", [-1.0, -1e-30, -0.0001, -1e30])
def test_negative_unitSI_rejected(bad_si: float) -> None:
    """The openPMD standard requires unitSI to be a positive conversion
    factor to SI. A negative value would invert every value written through
    this unit, so the constructor must reject it.
    """
    with pytest.raises(ValueError, match="unitSI must be non-negative"):
        pmd_unit("bogus", unitSI=bad_si, unitDimension="1")


@pytest.mark.parametrize(
    ("symbol", "expected_tex"),
    [
        ("", ""),
        ("1", "1"),
        ("m", r"\mathrm{m}"),
        ("eV", r"\mathrm{eV}"),
        ("eV/c", r"\mathrm{eV}/\mathrm{c}"),
        ("kg*m/s", r"\mathrm{kg}{\cdot}\mathrm{m}/\mathrm{s}"),
        ("m^2", r"\mathrm{m}^{2}"),
        ("s^-1", r"\mathrm{s}^{-1}"),
        ("m^0.5", r"\mathrm{m}^{0.5}"),
        ("m^(1/2)", r"\mathrm{m}^{1/2}"),
        ("sqrt(m)", r"\sqrt{\mathrm{m}}"),
        ("sqrt(kg*m)", r"\sqrt{\mathrm{kg}{\cdot}\mathrm{m}}"),
        ("charge #", r"\mathrm{charge #}"),
        ("V/(eV/c)", r"\mathrm{V}/(\mathrm{eV}/\mathrm{c})"),
        ("(eV/c)^2", r"(\mathrm{eV}/\mathrm{c})^{2}"),
    ],
)
def test_to_tex_translation(symbol: str, expected_tex: str) -> None:
    """``to_tex`` is a structural translation only: each atomic name wraps
    in ``\\mathrm{...}``, ``*`` becomes ``{\\cdot}``, ``/`` stays, ``^N``
    becomes ``^{N}``, and ``sqrt(X)`` becomes ``\\sqrt{...}``. No
    simplification, no reordering.
    """
    if symbol in ("", "1"):
        # pmd_unit("") and pmd_unit("1") with the default unitSI=0 would go
        # through symbol lookup; construct directly to control the symbol.
        u = pmd_unit(symbol, unitSI=1, unitDimension="1")
    else:
        u = pmd_unit(symbol)
    assert u.to_tex() == expected_tex


def test_to_tex_preserves_input_structure() -> None:
    """``to_tex`` must reflect the *stored* symbol, not the simplified form.
    A redundant compound stays compound in the TeX output.
    """
    u = pmd_unit("m*s/m")  # dimension reduces, but the symbol is preserved
    assert u.to_tex() == r"\mathrm{m}{\cdot}\mathrm{s}/\mathrm{m}"


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


def test_named_units_are_float_typed() -> None:
    """Every construction path yields float dimension tuples and a float
    unitSI, per the openPMD standard (float64 attributes). The string-name
    dimension path must coerce the same way the tuple path does."""
    for u in NAMED_UNITS:
        assert all(isinstance(d, float) for d in u.unitDimension), u
        assert isinstance(u.unitSI, float), u
    assert all(isinstance(d, float) for d in pmd_unit("T").unitDimension)
    assert isinstance(pmd_unit("m").unitSI, float)


def test_write_unit_h5_emits_float64_and_reads_back_bytes() -> None:
    """write_unit_h5 writes float64 unitSI/unitDimension; read_unit_h5
    decodes bytes unitSymbol attrs (as written by older fixed-length-string
    writers) so unit arithmetic works on the result."""
    h5py = pytest.importorskip("h5py")
    import io

    from beamphysics.units import read_unit_h5, write_unit_h5

    bio = io.BytesIO()
    with h5py.File(bio, "w") as f:
        g = f.create_group("g")
        write_unit_h5(g, pmd_unit("T"))
        assert f["g"].attrs["unitDimension"].dtype == np.float64
        assert isinstance(f["g"].attrs["unitSI"], float)

        gb = f.create_group("gb")
        gb.attrs["unitSI"] = 1.0
        gb.attrs["unitDimension"] = (1, 0, 0, 0, 0, 0, 0)
        gb.attrs["unitSymbol"] = np.bytes_("m")  # legacy bytes encoding
        u = read_unit_h5(gb)
        assert u.unitSymbol == "m"
        assert (u * pmd_unit("s")).unitSymbol == "m*s"  # arithmetic works


def test_sqrt_product_parses() -> None:
    """The whole-string sqrt() match must not swallow a product of sqrts:
    the final ')' of 'sqrt(m)*sqrt(m)' is not the partner of the first '('."""
    u = pmd_unit("sqrt(m)*sqrt(m)")
    assert u.unitDimension == (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert u.simplify().unitSymbol == "m"
    assert_round_trip(u)
    # Nested sqrt still parses.
    assert pmd_unit("sqrt(sqrt(m))").unitDimension == (0.25, 0, 0, 0, 0, 0, 0)


def test_slice_statistic_keys_have_units() -> None:
    """The keys slice_statistics emits ('current', 'density') resolve."""
    assert pg_units("current") == pmd_unit("A")
    assert pg_units("density") == pmd_unit("C") / pmd_unit("m")


def test_nice_scale_prefix_returns_python_float() -> None:
    """No numpy scalar leakage in the public API."""
    f, p = nice_scale_prefix(2e-10)
    assert type(f) is float and (f, p) == (1e-12, "p")
    f, p = nice_scale_prefix(0)
    assert type(f) is float and p == ""


def test_units_module_doctests() -> None:
    """Keep the docstring examples in beamphysics.units honest."""
    import doctest

    import beamphysics.units as units_module

    results = doctest.testmod(units_module)
    assert results.failed == 0, f"{results.failed} doctest failure(s)"


def test_sqrt_of_compound_inside_expression() -> None:
    """All notation must work uniformly at any recursion depth: the parser
    has a single recursive entry point (from_symbol), so a sqrt-of-compound
    token behaves the same alone and inside a larger expression."""
    alone = pmd_unit("sqrt(kg*m)")
    for s in ("sqrt(kg*m)*s", "s*sqrt(kg*m)", "V/sqrt(kg*m)", "sqrt(kg*m)^2"):
        u = pmd_unit(s)
        assert_round_trip(u)
    assert pmd_unit("sqrt(kg*m)*s").unitDimension == tuple(
        a + b for a, b in zip(alone.unitDimension, pmd_unit("s").unitDimension)
    )
    assert pmd_unit("sqrt(kg*m)^2") == pmd_unit("kg*m")


def test_single_token_spelling_preserved() -> None:
    """A lone token keeps the user's spelling as the stored symbol; the
    parsed value is canonical regardless."""
    assert pmd_unit("m^(1/2)").unitSymbol == "m^(1/2)"
    assert pmd_unit("(eV/c)^2").unitSymbol == "(eV/c)^2"
    assert pmd_unit("(eV/c)").unitSymbol == "(eV/c)"
    assert pmd_unit("(m)").unitSymbol == "m"  # redundant parens drop
    assert pmd_unit("m^(1/2)") == pmd_unit("m^0.5")
