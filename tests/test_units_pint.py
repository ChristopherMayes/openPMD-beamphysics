"""
Independent verification of ``beamphysics.units`` against `pint`.

`pint` is a mature, widely-used units library. These tests treat it as an
external oracle: for every named unit, named dimension, and a sampling of
compound symbols, we check that beamphysics agrees with pint on both the SI
dimensionality (the 7-tuple of base-unit exponents) and the SI scale factor.

`pint` is a *test-only* dependency. The package itself must not import it, so
this module skips entirely when pint is unavailable.
"""

from __future__ import annotations

import math
import re

import pytest

pint = pytest.importorskip("pint")

from beamphysics.units import (  # noqa: E402  (after importorskip)
    DIMENSION,
    NAMED_UNITS,
    PARTICLEGROUP_UNITS,
    SI_symbol,
    pg_units,
    pmd_unit,
)

# beamphysics stores dimensions as a 7-tuple in this order:
#   (length, mass, time, current, temperature, mol, luminous)
# Map each pint base-dimension name to its index in that tuple.
PINT_DIM_INDEX = {
    "[length]": 0,
    "[mass]": 1,
    "[time]": 2,
    "[current]": 3,
    "[temperature]": 4,
    "[substance]": 5,
    "[luminosity]": 6,
}

# beamphysics uses a handful of non-standard, openPMD-specific symbols that pint
# cannot parse directly. Map them to the pint expression matching the unit's
# *declared* physical meaning (dimension + SI scale).
SYMBOL_TO_PINT = {
    "": "dimensionless",
    "1": "dimensionless",
    "rad": "radian",
    "degree": "degree",
    "c": "speed_of_light",  # declared: dim velocity, unitSI = c_light
    "charge #": "coulomb",  # declared: dim charge, unitSI = 1
    "W/rad^2": "W / radian**2",
    "W/m^2": "W / m**2",
}


def to_pint_expr(symbol: str) -> str:
    """Translate a beamphysics unit symbol into a pint-parseable expression."""
    if symbol in SYMBOL_TO_PINT:
        return SYMBOL_TO_PINT[symbol]
    # beamphysics writes 'sqrt(X)'; pint has no sqrt function, wants '(X)**0.5'.
    expr = re.sub(r"sqrt\(([^()]+)\)", r"(\1)**0.5", symbol)
    # beamphysics writes powers with '^'; pint wants '**'.
    return expr.replace("^", "**")


@pytest.fixture(scope="module")
def ureg() -> "pint.UnitRegistry":
    return pint.UnitRegistry()


def pint_dimension_tuple(quantity: "pint.Quantity") -> tuple[int, ...]:
    """Convert a pint quantity's dimensionality to a beamphysics 7-tuple."""
    dims = [0] * 7
    for name, exponent in quantity.dimensionality.items():
        assert name in PINT_DIM_INDEX, f"unexpected pint dimension {name!r}"
        rounded = round(exponent)
        assert math.isclose(rounded, exponent), f"non-integer exponent {exponent}"
        dims[PINT_DIM_INDEX[name]] = rounded
    return tuple(dims)


def pint_dimension_floats(quantity: "pint.Quantity") -> tuple[float, ...]:
    """Like pint_dimension_tuple but keeps fractional exponents (e.g. m**0.5)."""
    dims = [0.0] * 7
    for name, exponent in quantity.dimensionality.items():
        assert name in PINT_DIM_INDEX, f"unexpected pint dimension {name!r}"
        dims[PINT_DIM_INDEX[name]] = float(exponent)
    return tuple(dims)


def dimensions_close(a, b, abs_tol: float = 1e-9) -> bool:
    """Elementwise comparison of two dimension tuples (handles fractions)."""
    return len(a) == len(b) and all(
        math.isclose(x, y, abs_tol=abs_tol) for x, y in zip(a, b)
    )


# --------------------------------------------------------------------------- #
# 1. Named dimensions (the DIMENSION dict) vs pint.
#    This is the check that would have caught the electric_potential bug.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", sorted(DIMENSION))
def test_dimension_matches_pint(name: str, ureg) -> None:
    symbol = SI_symbol[name]
    quantity = ureg.Quantity(1, to_pint_expr(symbol)).to_base_units()
    assert pint_dimension_tuple(quantity) == DIMENSION[name], (
        f"{name} ({symbol}): beamphysics {DIMENSION[name]} != pint "
        f"{pint_dimension_tuple(quantity)}"
    )


# --------------------------------------------------------------------------- #
# 2. Every named unit vs pint: dimension AND SI scale factor.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "unit", NAMED_UNITS, ids=[u.unitSymbol or "<dimensionless>" for u in NAMED_UNITS]
)
def test_named_unit_matches_pint(unit: pmd_unit, ureg) -> None:
    quantity = ureg.Quantity(1, to_pint_expr(unit.unitSymbol)).to_base_units()
    assert pint_dimension_tuple(quantity) == unit.unitDimension
    assert math.isclose(quantity.magnitude, unit.unitSI, rel_tol=1e-9)


# --------------------------------------------------------------------------- #
# 3. Compound-symbol parsing vs pint: independently validates pmd_unit's
#    operator/power parser and unit arithmetic.
# --------------------------------------------------------------------------- #
COMPOUND_SYMBOLS = [
    "A*s",  # = C
    "A/s",
    "kg*m/s",  # momentum
    "kg*m^2/s^2",  # energy
    "m/s^2",  # acceleration
    "J/s",  # = W
    "J/C",  # = V  (exercises the electric_potential dimension)
    "V*A",  # = W
    "V/m",
    "T*m",
    "T/m",
    "C/s",  # = A
    "W/m^2",
    "eV/c",
    "kg/m^3",
    # Parenthesized grouping (pint parses parens natively).
    "V/(eV/c)",
    "J/(A*s)",  # = V
    "(eV*s)^2",
    "W/(V/(eV/c))",  # nested group
]


@pytest.mark.parametrize("symbol", COMPOUND_SYMBOLS)
def test_compound_symbol_matches_pint(symbol: str, ureg) -> None:
    bp = pmd_unit(symbol)
    quantity = ureg.Quantity(1, to_pint_expr(symbol)).to_base_units()
    assert pint_dimension_tuple(quantity) == bp.unitDimension, (
        f"{symbol}: beamphysics {bp.unitDimension} != pint "
        f"{pint_dimension_tuple(quantity)}"
    )
    assert math.isclose(quantity.magnitude, bp.unitSI, rel_tol=1e-9)


# --------------------------------------------------------------------------- #
# 4. simplify() preserves physical meaning: the simplified unit must still
#    agree with pint on dimension and SI scale (independent of the symbol).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("symbol", COMPOUND_SYMBOLS)
def test_simplify_preserves_physics_vs_pint(symbol: str, ureg) -> None:
    simplified = pmd_unit(symbol).simplify()
    quantity = ureg.Quantity(1, to_pint_expr(symbol)).to_base_units()
    assert pint_dimension_tuple(quantity) == simplified.unitDimension
    assert math.isclose(quantity.magnitude, simplified.unitSI, rel_tol=1e-9)


# --------------------------------------------------------------------------- #
# 5. SI-prefixed symbols (keV, mm, mrad, GHz, ...) parse to the same dimension
#    and SI scale that pint computes.
# --------------------------------------------------------------------------- #
PREFIXED_SYMBOLS = [
    "keV",
    "MeV",
    "GeV",
    "meV",
    "mm",
    "cm",
    "µm",
    "mrad",
    "GHz",
    "kHz",
    "MHz",
    "mA",
    "kV",
    "mg",
    "ns",
    "ps",
    "fs",
]


@pytest.mark.parametrize("symbol", PREFIXED_SYMBOLS)
def test_prefixed_symbol_matches_pint(symbol: str, ureg) -> None:
    bp = pmd_unit(symbol)
    quantity = ureg.Quantity(1, to_pint_expr(symbol)).to_base_units()
    assert pint_dimension_tuple(quantity) == bp.unitDimension
    assert math.isclose(quantity.magnitude, bp.unitSI, rel_tol=1e-9)


# --------------------------------------------------------------------------- #
# 6. Fractional powers (sqrt and X^(p/q)) — these arise in practice (e.g.
#    spectral densities like V/m/Hz^0.5). pint represents them with fractional
#    dimension exponents, so compare elementwise.
# --------------------------------------------------------------------------- #
FRACTIONAL_SYMBOLS = [
    "m^0.5",
    "sqrt(m)",
    "m^(-3/2)",
    "T*m^0.5",
    "eV^0.5",
    "V/m/Hz^0.5",
    "s^-1",
]


@pytest.mark.parametrize("symbol", ["Ω", "Ω/m", "V/A", "V/C", "V/C/m"])
def test_impedance_and_wake_units_match_pint(symbol: str, ureg) -> None:
    # Impedance (ohm = V/A) and wakefield units, cross-checked against pint.
    bp = pmd_unit(symbol)
    quantity = ureg.Quantity(1, to_pint_expr(symbol)).to_base_units()
    assert pint_dimension_tuple(quantity) == bp.unitDimension
    assert math.isclose(quantity.magnitude, bp.unitSI, rel_tol=1e-9)


@pytest.mark.parametrize("symbol", FRACTIONAL_SYMBOLS)
def test_fractional_power_matches_pint(symbol: str, ureg) -> None:
    bp = pmd_unit(symbol)
    quantity = ureg.Quantity(1, to_pint_expr(symbol)).to_base_units()
    assert dimensions_close(pint_dimension_floats(quantity), bp.unitDimension), (
        f"{symbol}: beamphysics {bp.unitDimension} != pint "
        f"{pint_dimension_floats(quantity)}"
    )
    assert math.isclose(quantity.magnitude, bp.unitSI, rel_tol=1e-9)


# --------------------------------------------------------------------------- #
# 7. pg_units(): the ParticleGroup-key -> unit lookup table. The expected
#    SI *dimension* per key is specified independently here (from the physical
#    meaning of the key, via canonical SI symbols pint parses), so this cross-
#    check catches a key mapped to the wrong dimension. We compare dimension
#    only: beamphysics stores energy/momentum in eV / eV-c, not J / kg-m-s.
# --------------------------------------------------------------------------- #
_PG_EXPECTED_GROUPS = {
    "dimensionless": [
        "n_particle",
        "status",
        "id",
        "n_alive",
        "n_dead",
        "beta",
        "beta_x",
        "beta_y",
        "beta_z",
        "gamma",
        "bunching",
        "theta",
        "bunching_phase",
        "xp",
        "yp",
        "twiss_alpha_x",
        "twiss_alpha_y",
        "twiss_etap_x",
        "twiss_etap_y",
    ],
    "s": ["t", "z/c"],
    "J": [
        "energy",
        "kinetic_energy",
        "mass",  # mass is stored as rest energy
        "higher_order_energy",
        "higher_order_energy_spread",
    ],
    "kg*m/s": ["px", "py", "pz", "p", "pr", "ptheta"],
    "m": [
        "x",
        "y",
        "z",
        "r",
        "Jx",
        "Jy",
        "norm_emit_x",
        "norm_emit_y",
        "twiss_beta_x",
        "twiss_beta_y",
        "twiss_eta_x",
        "twiss_eta_y",
        "twiss_emit_x",
        "twiss_emit_y",
        "twiss_norm_emit_x",
        "twiss_norm_emit_y",
    ],
    "C": ["charge", "species_charge", "weight"],
    "A": ["average_current"],
    "W": ["power"],
    "m**2": ["norm_emit_4d"],
    "m*kg*m/s": ["Lz"],  # angular momentum = length * momentum
    "m**0.5": ["x_bar", "px_bar", "y_bar", "py_bar"],  # normal-form coords
    "V/m": ["E", "Ex", "Ey", "Ez", "Etheta", "Er"],
    "T": ["B", "Bx", "By", "Bz", "Btheta", "Br"],
    "1/m": ["twiss_gamma_x", "twiss_gamma_y"],
}
PG_EXPECTED = {key: expr for expr, keys in _PG_EXPECTED_GROUPS.items() for key in keys}


def test_pg_units_expectation_covers_all_keys() -> None:
    # Completeness guard: a newly added PARTICLEGROUP_UNITS key must also get an
    # independent expected dimension here, or this test fails and forces review.
    assert set(PG_EXPECTED) == set(PARTICLEGROUP_UNITS)


@pytest.mark.parametrize("key", sorted(PG_EXPECTED))
def test_pg_units_dimension_matches_pint(key: str, ureg) -> None:
    bp = pg_units(key)
    expected = ureg.Quantity(1, PG_EXPECTED[key]).to_base_units()
    assert dimensions_close(pint_dimension_floats(expected), bp.unitDimension), (
        f"pg_units({key!r}) dim = {bp.unitDimension}, expected dim of "
        f"{PG_EXPECTED[key]!r} = {pint_dimension_floats(expected)}"
    )


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        # Operator prefixes preserve the base key's dimension.
        ("sigma_x", "m"),
        ("mean_energy", "J"),
        ("min_px", "kg*m/s"),
        ("max_z", "m"),
        ("ptp_t", "s"),
        ("delta_p", "kg*m/s"),
        # Covariance multiplies the two sub-key dimensions.
        ("cov_x__px", "m*kg*m/s"),
        ("cov_t__t", "s**2"),
        # startswith-based field keys.
        ("electricField", "V/m"),
        ("magneticField", "T"),
        ("bunching_5", "dimensionless"),
    ],
)
def test_pg_units_derived_keys_match_pint(key: str, expected: str, ureg) -> None:
    bp = pg_units(key)
    quantity = ureg.Quantity(1, expected).to_base_units()
    assert dimensions_close(
        pint_dimension_floats(quantity), bp.unitDimension
    ), f"pg_units({key!r}) dim = {bp.unitDimension}, expected {expected!r}"
