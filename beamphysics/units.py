"""
Simple units functionality for the openPMD beamphysics records.

For more advanced units, use a package like Pint:
    https://pint.readthedocs.io/
"""

from __future__ import annotations

import math
import re
import warnings
from typing import Optional, Sequence

import numpy as np
import scipy.constants

mec2 = scipy.constants.value("electron mass energy equivalent in MeV") * 1e6
mpc2 = scipy.constants.value("proton mass energy equivalent in MeV") * 1e6
c_light = scipy.constants.c
e_charge = scipy.constants.e
mu_0 = scipy.constants.mu_0  # Note that this is not 4pi*10^-7 !
# Derived values. Note that scipy.constants.epsilon_0 is not consistent!
epsilon_0 = 1 / (mu_0 * c_light**2)  # F/m
Z0 = mu_0 * c_light  # Omh = V^2/W

Limit = tuple[Optional[float], Optional[float]]
Dimension = tuple[int, int, int, int, int, int, int]

# Module-level dict that will be populated after NAMED_UNITS is created
# This avoids the globals() check chicken-and-egg problem
known_unit: dict[str, "pmd_unit"] = {}


class pmd_unit:
    """
    OpenPMD representation of a unit.

    Parameters
    ----------
    unitSymbol : str
        Native units name. Can be a simple symbol (e.g., 'eV', 'm') or a
        compound expression using * and / operators (e.g., 'eV/c', 'kg*m/s').
    unitSI : float, optional
        Conversion factor to the corresponding SI unit.  Defaults to 0.
        If unspecified, `unitSymbol` must be a recognized symbol name.
    unitDimension : str, or list of int, optional
        Common name of dimensions or list of 7 SI Base Exponents.
        Valid names include:
        * "1"
        * "length"
        * "mass"
        * "time"
        * "current"
        * "temperature"
        * "mol"
        * "luminous"
        * "charge"
        * "electric_field"
        * "electric_potential"
        * "magnetic_field"
        * "velocity"
        * "energy"
        * "momentum"

        For a full list, see `beamphysics.units.DIMENSION`.

    Notes
    -----

    Base unit dimensions are defined as:

       Base dimension  | exponents.       | SI unit
       ---------------- -----------------   -------
       length          : (1,0,0,0,0,0,0)     m
       mass            : (0,1,0,0,0,0,0)     kg
       time            : (0,0,1,0,0,0,0)     s
       current         : (0,0,0,1,0,0,0)     A
       temperature     : (0,0,0,0,1,0,0)     K
       mol             : (0,0,0,0,0,1,0)     mol
       luminous        : (0,0,0,0,0,0,1)     cd

    Examples
    --------

    Define that an eV is 1.602176634e-19 of base units m^2 kg/s^2, which is a Joule (J):

    >>> pmd_unit('eV', 1.602176634e-19, (2, 1, -2, 0, 0, 0, 0))

    If unitSI=0 (default), `pmd_unit` may be initialized with a known symbol:

    >>> pmd_unit('T')
    pmd_unit('T', 1, (0, 1, -2, -1, 0, 0, 0))

    Compound units can be created using * and / operators in the symbol string:

    >>> pmd_unit('eV/c')
    pmd_unit('eV/c', 5.344286295439521e-28, (1, 1, -1, 0, 0, 0, 0))

    >>> pmd_unit('kg*m/s')
    pmd_unit('kg*m/s', 1, (1, 1, -1, 0, 0, 0, 0))

    Alternatively, use the `from_symbol` class method for explicit symbol lookup:

    >>> pmd_unit.from_symbol('A*s')
    pmd_unit('A*s', 1, (0, 0, 1, 1, 0, 0, 0))

    Simple equalities are possible:

    >>> pmd_unit("T") == pmd_unit("T")
    True
    >>> pmd_unit("eV") == pmd_unit("T")
    False
    """

    def __init__(
        self,
        unitSymbol: str = "",
        unitSI: int | float = 0,
        unitDimension: str | Dimension | tuple[int, ...] = (0, 0, 0, 0, 0, 0, 0),
    ):
        # If unitSI is provided explicitly, use it directly
        if unitSI != 0:
            self._unitSymbol = unitSymbol
            self._unitSI = unitSI
            if isinstance(unitDimension, str):
                self._unitDimension = dimension(unitDimension)
            else:
                self._unitDimension = make_dimension(unitDimension)
        else:
            # unitSI == 0: try to lookup/parse the symbol
            # Check if known_unit dict has been populated (it won't during NAMED_UNITS construction)
            if not known_unit:
                # During NAMED_UNITS construction - just use what's provided
                self._unitSymbol = unitSymbol
                self._unitSI = unitSI
                if isinstance(unitDimension, str):
                    self._unitDimension = dimension(unitDimension)
                else:
                    self._unitDimension = make_dimension(unitDimension)
            else:
                # known_unit exists - delegate to from_symbol for lookup/parsing
                u = self.from_symbol(unitSymbol)
                self._unitSymbol = u._unitSymbol
                self._unitSI = u._unitSI
                self._unitDimension = u._unitDimension

    @classmethod
    def from_symbol(cls, unitSymbol: str) -> pmd_unit:
        """
        Create a pmd_unit from a symbol string, with automatic lookup and parsing.

        This method handles:
        - Simple known units (e.g., 'eV', 'm', 'T')
        - Compound expressions with operators (e.g., 'eV/c', 'kg*m/s')
        - Power notation (e.g., 'm^2', 'eV^2', 's^-1')
        - Square root notation (e.g., 'sqrt(m)')

        Parameters
        ----------
        unitSymbol : str
            The unit symbol to look up or parse.

        Returns
        -------
        pmd_unit
            The resulting unit object.

        Raises
        ------
        ValueError
            If the symbol is not found in known_unit dict or cannot be parsed.
        """
        # Check if known_unit dict has been populated
        if not known_unit:
            raise ValueError(
                f"Cannot lookup unitSymbol '{unitSymbol}': known_unit dict not yet initialized. "
                "Use pmd_unit(unitSymbol, unitSI, unitDimension) with explicit parameters instead."
            )

        # known_unit exists - check if it's already a known unit (priority)
        if unitSymbol in known_unit:
            # Copy internals from known unit
            u = known_unit[unitSymbol]
            return cls(unitSymbol, unitSI=u.unitSI, unitDimension=u.unitDimension)

        # Handle sqrt() notation: sqrt(X) = X^0.5
        sqrt_match = re.match(r"^sqrt\((.+)\)$", unitSymbol)
        if sqrt_match:
            inner = sqrt_match.group(1)
            base_unit = cls.from_symbol(inner)
            result = power_unit(base_unit, 0.5)
            # Preserve the original sqrt notation in the symbol
            result._unitSymbol = unitSymbol
            return result

        # Not in known_unit - try operator parsing (e.g., "A*s", "kg*m/s", "m^2")
        if "*" in unitSymbol or "/" in unitSymbol or "^" in unitSymbol:
            # Tokenize: split by * and / but not inside parentheses
            # This allows m^(-3/2) to remain intact
            parts = []
            operators = []
            current = ""
            paren_depth = 0

            for char in unitSymbol:
                if char == "(":
                    paren_depth += 1
                    current += char
                elif char == ")":
                    paren_depth -= 1
                    current += char
                elif char in "*/" and paren_depth == 0:
                    parts.append(current)
                    operators.append(char)
                    current = ""
                else:
                    current += char

            parts.append(current)  # Don't forget the last part

            # Parse the first part (may include ^power)
            result = cls._parse_single_unit(parts[0])

            # Process remaining parts
            for op, part in zip(operators, parts[1:]):
                unit_obj = cls._parse_single_unit(part)
                if op == "*":
                    result = result * unit_obj
                elif op == "/":
                    result = result / unit_obj

            # Preserve the original symbol but keep computed dimension
            result._unitSymbol = unitSymbol
            return result

        # Single token, not directly known: delegate to _parse_single_unit,
        # which handles power/sqrt notation and the SI-prefix fallback (and
        # raises "Unknown unitSymbol" if it is genuinely unparseable).
        return cls._parse_single_unit(unitSymbol)

    @classmethod
    def _parse_single_unit(cls, part: str) -> pmd_unit:
        """
        Parse a single unit that may include a power (e.g., 'm^2', 'eV', 's^-1')
        or sqrt notation (e.g., 'sqrt(m)').

        Parameters
        ----------
        part : str
            A single unit token, optionally with power or sqrt notation.

        Returns
        -------
        pmd_unit
            The parsed unit object.
        """
        # Check if it's a known unit first. Return a copy, never the shared
        # cached object: callers (e.g. from_symbol) reassign ``_unitSymbol`` on
        # the result, which would otherwise mutate the global NAMED_UNITS entry.
        if part in known_unit:
            u = known_unit[part]
            return cls(part, unitSI=u.unitSI, unitDimension=u.unitDimension)

        # Check for sqrt() notation
        sqrt_match = re.match(r"^sqrt\((.+)\)$", part)
        if sqrt_match:
            inner = sqrt_match.group(1)
            base_unit = cls._parse_single_unit(inner)
            result = power_unit(base_unit, 0.5)
            result._unitSymbol = part
            return result

        # Check for power notation with parenthesized fraction: m^(-3/2)
        paren_power_match = re.match(r"^(.+)\^\((-?\d+)/(\d+)\)$", part)
        if paren_power_match:
            base_symbol = paren_power_match.group(1)
            numerator = int(paren_power_match.group(2))
            denominator = int(paren_power_match.group(3))
            power = numerator / denominator
            base_unit = cls._parse_single_unit(base_symbol)
            return power_unit(base_unit, power)

        # Check for power notation: m^2, m^-1, m^0.5
        power_match = re.match(r"^(.+)\^(-?\d+(?:\.\d+)?)$", part)
        if power_match:
            base_symbol = power_match.group(1)
            power = float(power_match.group(2))
            # Recursively parse base (could be sqrt(m)^2)
            base_unit = cls._parse_single_unit(base_symbol)
            return power_unit(base_unit, power)

        # SI-prefix fallback: e.g. "keV" -> kilo + "eV", "mm" -> milli + "m",
        # "mrad" -> milli + "rad". Only matches when the remainder is itself a
        # known unit; a bare prefix ("k", "M") or a prefixed dimensionless
        # identity ("k1") is not a valid unit. Longest prefixes first so "da"
        # (deca) wins over "d" (deci). This is the inverse of the prefixed
        # symbols simplify() emits, so those always round-trip.
        for prefix in sorted(SHORT_PREFIX_FACTOR, key=len, reverse=True):
            if prefix == "" or not part.startswith(prefix):
                continue
            remainder = part[len(prefix) :]
            if remainder in ("", "1") or remainder not in known_unit:
                continue
            base = known_unit[remainder]
            factor = SHORT_PREFIX_FACTOR[prefix]
            return cls(
                part, unitSI=factor * base.unitSI, unitDimension=base.unitDimension
            )

        # Unknown unit
        raise ValueError(f"Unknown unitSymbol: {part}")

    def __hash__(self) -> int:
        return hash((self.unitSymbol, self.unitSI, self.unitDimension))

    @property
    def unitSymbol(self) -> str:
        return self._unitSymbol

    @property
    def unitSI(self) -> float:
        return self._unitSI

    @property
    def unitDimension(self) -> Dimension:
        return self._unitDimension

    def __mul__(self, other) -> pmd_unit:
        return multiply_units(self, other)

    def __truediv__(self, other) -> pmd_unit:
        return divide_units(self, other)

    def __eq__(self, other) -> bool:
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return False

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def __str__(self) -> str:
        return self.unitSymbol

    def __repr__(self) -> str:
        return f"pmd_unit('{self.unitSymbol}', {self.unitSI}, {self.unitDimension})"

    def simplify(self, named_units=None) -> pmd_unit:
        """
        Search for a simpler equivalent symbol with the same dimension and
        SI value as this unit.

        The search proceeds in two passes:

        1. **Atomic match.** Look for a single entry in ``named_units`` whose
           dimension matches ``self.unitDimension`` and whose ``unitSI``
           equals ``self.unitSI`` (optionally up to a single SI prefix
           factor from :data:`SHORT_PREFIX_FACTOR`). Comparisons use
           ``math.isclose`` so factors built from arithmetic are matched
           reliably.
        2. **Compound match.** If no atomic match is found, look for an
           ``a*b`` or ``a/b`` combination of two named units (both with
           nonzero dimension) whose product/quotient matches ``self`` exactly.

        Among candidates, the one with the simplest symbol is returned
        (no operators preferred; then shortest; then lexicographic).
        If nothing matches, ``self`` is returned unchanged.

        Parameters
        ----------
        named_units : sequence of pmd_unit, optional
            Pool of named units to search. Defaults to :data:`NAMED_UNITS`.

        Returns
        -------
        pmd_unit
            A simplified unit, or ``self`` if no simplification was found.
        """
        if named_units is None:
            named_units = NAMED_UNITS

        target_dim = self.unitDimension
        target_si = self.unitSI

        # Pass 1: single named-unit match (optionally with an SI prefix).
        atomic = _best_atomic_match(target_dim, target_si, named_units)
        if atomic is not None:
            return atomic

        # Pass 2: a*b or a/b compound match (exact SI, no extra prefix).
        compound = _best_compound_match(target_dim, target_si, named_units)
        if compound is not None:
            return compound

        return self


def is_dimensionless(u: pmd_unit) -> bool:
    """Checks if the unit is dimensionless"""
    if not isinstance(u, pmd_unit):
        raise ValueError("`u` is not a pmd_unit instance")
    return u.unitDimension == (0, 0, 0, 0, 0, 0, 0)


def is_identity(u: pmd_unit) -> bool:
    """Checks if the unit is equivalent to 1"""
    if not isinstance(u, pmd_unit):
        raise ValueError("`u` is not a pmd_unit instance")
    return u.unitSI == 1 and u.unitDimension == (0, 0, 0, 0, 0, 0, 0)


_SIMPLIFY_REL_TOL = 1e-9
_DIMENSIONLESS: Dimension = (0, 0, 0, 0, 0, 0, 0)


def _symbol_complexity(symbol: str) -> tuple[int, int, str]:
    """Sort key: prefer no operators, then shorter, then lexicographic."""
    has_ops = any(c in symbol for c in "*/^")
    return (int(has_ops), len(symbol), symbol)


def _closest_prefix(factor: float) -> tuple[str, float] | None:
    """
    Return ``(prefix, prefix_factor)`` from :data:`SHORT_PREFIX_FACTOR` if
    ``factor`` matches a known SI prefix within :data:`_SIMPLIFY_REL_TOL`,
    else ``None``.
    """
    if math.isclose(factor, 1.0, rel_tol=_SIMPLIFY_REL_TOL):
        return ("", 1.0)
    for prefix, pf in SHORT_PREFIX_FACTOR.items():
        if math.isclose(factor, pf, rel_tol=_SIMPLIFY_REL_TOL):
            return (prefix, pf)
    return None


def _best_atomic_match(
    target_dim: Dimension,
    target_si: float,
    named_units: Sequence[pmd_unit],
) -> pmd_unit | None:
    """
    Find the simplest named unit (optionally with an SI prefix) whose
    dimension and SI value match the target.
    """
    candidates: list[tuple[int, tuple[int, int, str], pmd_unit, str, float]] = []
    for u in named_units:
        if u.unitDimension != target_dim or u.unitSI == 0:
            continue
        prefix_match = _closest_prefix(target_si / u.unitSI)
        if prefix_match is None:
            continue
        prefix, pf = prefix_match
        # A prefix on a dimensionless unit is meaningless: the empty symbol
        # becomes a bare prefix ("" -> "m", which re-parses as metres), and
        # "mrad"/"krad" would imply an angle for a pure-number ratio. Only an
        # exact (unprefixed) dimensionless match is valid.
        if prefix != "" and u.unitDimension == _DIMENSIONLESS:
            continue
        out_symbol = prefix + u.unitSymbol
        # Priority: exact match (priority 0) beats prefixed match (priority 1).
        priority = 0 if prefix == "" else 1
        candidates.append((priority, _symbol_complexity(out_symbol), u, prefix, pf))

    if not candidates:
        return None

    candidates.sort(key=lambda c: (c[0], c[1]))
    _, _, u, prefix, pf = candidates[0]
    if prefix == "":
        return u
    return pmd_unit(prefix + u.unitSymbol, pf * u.unitSI, u.unitDimension)


def _best_compound_match(
    target_dim: Dimension,
    target_si: float,
    named_units: Sequence[pmd_unit],
) -> pmd_unit | None:
    """
    Find the simplest ``a*b`` or ``a/b`` of two named units that matches
    the target dimension and SI value exactly. Dimensionless factors are
    skipped to avoid trivial decorations like ``"m*1"``.

    The symbol parser is flat and left-associative (``a/b/c`` means
    ``(a/b)/c``), so a divisor that itself contains ``*`` or ``/`` would be
    regrouped when the symbol is re-parsed (``a/(b/c)`` written as
    ``"a/b/c"`` reads back as ``a/(b*c)``). The ``a/b`` form therefore only
    accepts an operator-free divisor; the ``a*b`` form is unaffected because
    a product chain is associative under the parser's grouping.

    A dimensionless target is never matched: it is a pure number, and any
    ``a*b`` / ``a/b`` reaching it would pair two same-dimension units (e.g.
    ``kg/g`` for a ratio of 1000), which is meaningless as a unit symbol.
    """
    if target_dim == _DIMENSIONLESS:
        return None

    dim_index: dict[Dimension, list[pmd_unit]] = {}
    for u in named_units:
        if u.unitDimension == _DIMENSIONLESS or u.unitSI == 0:
            continue
        dim_index.setdefault(u.unitDimension, []).append(u)

    if not dim_index or target_si == 0:
        return None

    best: pmd_unit | None = None
    best_key: tuple[int, int, str] | None = None

    for a in (u for group in dim_index.values() for u in group):
        a_si = a.unitSI

        # self = a * b  =>  b_dim = target - a_dim, b_si = target / a
        mul_dim = tuple(td - ad for td, ad in zip(target_dim, a.unitDimension))
        mul_target_si = target_si / a_si
        for b in dim_index.get(mul_dim, ()):
            if not math.isclose(b.unitSI, mul_target_si, rel_tol=_SIMPLIFY_REL_TOL):
                continue
            sym = f"{a.unitSymbol}*{b.unitSymbol}"
            key = _symbol_complexity(sym)
            if best_key is None or key < best_key:
                best = pmd_unit(sym, a_si * b.unitSI, target_dim)
                best_key = key

        # self = a / b  =>  b_dim = a_dim - target, b_si = a / target
        div_dim = tuple(ad - td for ad, td in zip(a.unitDimension, target_dim))
        div_target_si = a_si / target_si
        for b in dim_index.get(div_dim, ()):
            # An operator-containing divisor would be regrouped on re-parse.
            if any(op in b.unitSymbol for op in "*/"):
                continue
            if not math.isclose(b.unitSI, div_target_si, rel_tol=_SIMPLIFY_REL_TOL):
                continue
            sym = f"{a.unitSymbol}/{b.unitSymbol}"
            key = _symbol_complexity(sym)
            if best_key is None or key < best_key:
                best = pmd_unit(sym, a_si / b.unitSI, target_dim)
                best_key = key

    return best


def multiply_units(u1: pmd_unit, u2: pmd_unit) -> pmd_unit:
    """
    Multiplies two pmd_unit symbols.

    Parameters
    ----------
    u1 : pmd_unit
    u2 : pmd_unit

    Returns
    -------
    pmd_unit
        The resulting unit after multiplication.
    """

    if not isinstance(u1, pmd_unit):
        raise ValueError("u1 is not a pmd_unit instance")
    if not isinstance(u2, pmd_unit):
        raise ValueError("u2 is not a pmd_unit instance")

    if is_identity(u1):
        return u2
    if is_identity(u2):
        return u1

    s1 = u1.unitSymbol
    s2 = u2.unitSymbol
    if s1 == s2:
        symbol = f"({s1})^2"
    else:
        symbol = s1 + "*" + s2
    d1 = u1.unitDimension
    d2 = u2.unitDimension
    dim = tuple(sum(x) for x in zip(d1, d2))
    unitSI = u1.unitSI * u2.unitSI

    # Bypass make_dimension to preserve fractional dimensions
    result = object.__new__(pmd_unit)
    result._unitSymbol = symbol
    result._unitSI = unitSI
    result._unitDimension = dim

    return result


def divide_units(u1: pmd_unit, u2: pmd_unit) -> pmd_unit:
    """
    Divides two pmd_unit symbols: u1 / u2

    Parameters
    ----------
    u1 : pmd_unit
        The numerator unit.
    u2 : pmd_unit
        The denominator unit.

    Returns
    -------
    pmd_unit
    """
    if not isinstance(u1, pmd_unit):
        raise ValueError("u1 is not a pmd_unit instance")
    if not isinstance(u2, pmd_unit):
        raise ValueError("u2 is not a pmd_unit instance")

    if is_identity(u2):
        return u1

    s1 = u1.unitSymbol
    s2 = u2.unitSymbol
    if s1 == s2:
        symbol = "1"
    else:
        symbol = s1 + "/" + s2
    d1 = u1.unitDimension
    d2 = u2.unitDimension
    dim = tuple(a - b for a, b in zip(d1, d2))
    unitSI = u1.unitSI / u2.unitSI

    # Bypass make_dimension to preserve fractional dimensions
    result = object.__new__(pmd_unit)
    result._unitSymbol = symbol
    result._unitSI = unitSI
    result._unitDimension = dim

    return result


def sqrt_unit(u: pmd_unit) -> pmd_unit:
    """
    Returns the square root of a unit.

    Parameters
    ----------
    u : pmd_unit
        The unit to take the square root of.

    Returns
    -------
    pmd_unit
    """
    if not isinstance(u, pmd_unit):
        raise ValueError("`u` is not a pmd_unit instance")

    # Delegate to power_unit(0.5): it halves the dimension exponents with true
    # division (e.g. m -> m^0.5) and bypasses make_dimension, so fractional
    # dimensions survive. (Building via pmd_unit(...) instead would route
    # through make_dimension, which int()-truncates 0.5 back to 0.)
    result = power_unit(u, 0.5)
    symbol = u.unitSymbol
    if symbol not in ["", "1"]:
        result._unitSymbol = rf"\sqrt{{ {symbol} }}"
    return result


def power_unit(u: pmd_unit, power: float) -> pmd_unit:
    """
    Raises a unit to an arbitrary power.

    Parameters
    ----------
    u : pmd_unit
        The base unit.
    power : float
        The exponent to raise the unit to.

    Returns
    -------
    pmd_unit
        The unit raised to the given power.

    Examples
    --------
    >>> power_unit(pmd_unit("m"), 2)
    pmd_unit('m^2', 1, (2, 0, 0, 0, 0, 0, 0))

    >>> power_unit(pmd_unit("m"), 0.5)
    pmd_unit('m^0.5', 1, (0.5, 0, 0, 0, 0, 0, 0))
    """
    if not isinstance(u, pmd_unit):
        raise ValueError("`u` is not a pmd_unit instance")

    symbol = u.unitSymbol
    if symbol in ["", "1"]:
        new_symbol = symbol
    elif power == int(power):
        new_symbol = f"{symbol}^{int(power)}"
    else:
        new_symbol = f"{symbol}^{power}"

    unitSI = u.unitSI**power
    # Scale each dimension by the power (preserves fractional dimensions)
    dim = tuple(d * power for d in u.unitDimension)

    # Create unit object without going through make_dimension
    # to preserve fractional dimensions
    result = object.__new__(pmd_unit)
    result._unitSymbol = new_symbol
    result._unitSI = unitSI
    result._unitDimension = dim

    return result


# length mass time current temperature mol luminous
DIMENSION: dict[str, Dimension] = {
    "1": (0, 0, 0, 0, 0, 0, 0),
    # Base units
    "length": (1, 0, 0, 0, 0, 0, 0),
    "mass": (0, 1, 0, 0, 0, 0, 0),
    "time": (0, 0, 1, 0, 0, 0, 0),
    "current": (0, 0, 0, 1, 0, 0, 0),
    "temperature": (0, 0, 0, 0, 1, 0, 0),
    "mol": (0, 0, 0, 0, 0, 1, 0),
    "luminous": (0, 0, 0, 0, 0, 0, 1),
    #
    "charge": (0, 0, 1, 1, 0, 0, 0),
    "electric_field": (1, 1, -3, -1, 0, 0, 0),
    "electric_potential": (2, 1, -3, -1, 0, 0, 0),
    "magnetic_field": (0, 1, -2, -1, 0, 0, 0),
    "velocity": (1, 0, -1, 0, 0, 0, 0),
    "energy": (2, 1, -2, 0, 0, 0, 0),
    "momentum": (1, 1, -1, 0, 0, 0, 0),
}
# Inverse
DIMENSION_NAME: dict[Dimension, str] = {v: k for k, v in DIMENSION.items()}


def make_dimension(dim: Sequence[int]) -> Dimension:
    dim = tuple(int(d) for d in dim)
    if len(dim) != 7:
        raise ValueError("Dimensions must be 7 elements.")
    return dim


def dimension(name: str) -> Dimension | None:
    try:
        return DIMENSION[name]
    except KeyError:
        options = ", ".join(DIMENSION)
        raise ValueError(
            f"Invalid unit dimension string: {name}. Valid options are: {options}"
        )


def dimension_name(dim_array: Dimension) -> str:
    return DIMENSION_NAME[make_dimension(dim_array)]


SI_symbol: dict[str, str] = {
    "1": "1",
    "length": "m",
    "mass": "kg",
    "time": "s",
    "current": "A",
    "temperature": "K",
    "mol": "mol",
    "luminous": "cd",
    "charge": "C",
    "electric_field": "V/m",
    "electric_potential": "V",
    "velocity": "m/s",
    "energy": "J",
    "power": "W",
    "momentum": "kg*m/s",
    "magnetic_field": "T",
}
# Inverse
SI_name: dict[str, str] = {v: k for k, v in SI_symbol.items()}

# List of named units
NAMED_UNITS = [
    pmd_unit("", 1, "1"),
    pmd_unit("degree", np.pi / 180, "1"),
    pmd_unit("rad", 1, "1"),
    pmd_unit("m", 1, "length"),
    pmd_unit("kg", 1, "mass"),
    pmd_unit("g", 0.001, "mass"),
    pmd_unit("s", 1, "time"),
    pmd_unit("A", 1, "current"),
    pmd_unit("K", 1, "temperature"),
    pmd_unit("mol", 1, "mol"),
    pmd_unit("cd", 1, "luminous"),
    pmd_unit("C", 1, "charge"),
    pmd_unit("charge #", 1, "charge"),
    pmd_unit("V/m", 1, "electric_field"),
    pmd_unit("V", 1, "electric_potential"),
    pmd_unit("c", c_light, "velocity"),  # Speed of light
    pmd_unit("vel/c", c_light, "velocity"),
    pmd_unit("m/s", 1, "velocity"),
    pmd_unit("eV", e_charge, "energy"),
    pmd_unit("J", 1, "energy"),
    pmd_unit("eV/c", e_charge / c_light, "momentum"),
    pmd_unit("eV/m", e_charge, (1, 1, -2, 0, 0, 0, 0)),
    pmd_unit("J/m", 1, (1, 1, -2, 0, 0, 0, 0)),
    pmd_unit("W", 1, (2, 1, -3, 0, 0, 0, 0)),
    pmd_unit("W/rad^2", 1, (2, 1, -3, 0, 0, 0, 0)),
    pmd_unit("W/m^2", 1, (0, 1, -3, 0, 0, 0, 0)),
    pmd_unit("T", 1, "magnetic_field"),
    pmd_unit("Hz", 1, (0, 0, -1, 0, 0, 0, 0)),
    pmd_unit("Ω", 1, (2, 1, -3, -2, 0, 0, 0)),  # ohm = V/A; used for impedance
]

# Populate the module-level known_unit dict
known_unit.update({u.unitSymbol: u for u in NAMED_UNITS})
# Add inconsistent legacy keys
known_unit.update(
    {
        "1": pmd_unit("", 1, "1"),
        "charge_num": pmd_unit("charge #", 1, "charge"),
        "c_light": pmd_unit("vel/c", c_light, "velocity"),
        "deg": known_unit["degree"],
        "Ohm": known_unit["Ω"],  # ASCII alias for the ohm
    }
)


def unit(symbol: str) -> pmd_unit:
    """
    Returns a pmd_unit from a known symbol.

    * and / are allowed between two known symbols.

    .. deprecated::
        Use `pmd_unit(symbol)` directly instead. The `unit()` function is
        deprecated and will be removed in a future version.

    Examples
    --------
    Old style (deprecated):

        >>> unit("eV/c")

    New style (preferred):

        >>> pmd_unit("eV/c")
    """
    warnings.warn(
        "unit() is deprecated. Use pmd_unit() directly instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return pmd_unit(symbol)


# Dicts for prefixes
PREFIX_FACTOR: dict[str, float] = {
    "yocto-": 1e-24,
    "zepto-": 1e-21,
    "atto-": 1e-18,
    "femto-": 1e-15,
    "pico-": 1e-12,
    "nano-": 1e-9,
    "micro-": 1e-6,
    "milli-": 1e-3,
    "centi-": 1e-2,
    "deci-": 1e-1,
    "deca-": 1e1,
    "hecto-": 1e2,
    "kilo-": 1e3,
    "mega-": 1e6,
    "giga-": 1e9,
    "tera-": 1e12,
    "peta-": 1e15,
    "exa-": 1e18,
    "zetta-": 1e21,
    "yotta-": 1e24,
}
# Inverse
PREFIX: dict[float, str] = dict((v, k) for k, v in PREFIX_FACTOR.items())

SHORT_PREFIX_FACTOR: dict[str, float] = {
    "y": 1e-24,
    "z": 1e-21,
    "a": 1e-18,
    "f": 1e-15,
    "p": 1e-12,
    "n": 1e-9,
    "µ": 1e-6,
    "m": 1e-3,
    "c": 1e-2,
    "d": 1e-1,
    "": 1,
    "da": 1e1,
    "h": 1e2,
    "k": 1e3,
    "M": 1e6,
    "G": 1e9,
    "T": 1e12,
    "P": 1e15,
    "E": 1e18,
    "Z": 1e21,
    "Y": 1e24,
}
# Inverse
SHORT_PREFIX: dict[float, str] = dict((v, k) for k, v in SHORT_PREFIX_FACTOR.items())


# Nice scaling


def nice_scale_prefix(scale: float) -> tuple[float, str]:
    """
    Returns a nice factor and an SI prefix string.

    Parameters
    ----------
    scale : float
        The scale to be converted into a nice factor and SI prefix.

    Returns
    -------
    f : float
        The nice factor corresponding to the scale.
    u : str
        The SI prefix string corresponding to the scale.

    Examples
    --------
    >>> nice_scale_prefix(scale=2e-10)
    (1e-12, 'p')
    """

    if scale == 0:
        return 1, ""

    p10 = np.log10(abs(scale))

    if p10 < -24:  # Limits of SI prefixes
        f = 1e-24
    elif p10 > 24:
        f = 1e24
    elif p10 < -1.5 or p10 > 2:
        f = 10 ** (p10 // 3 * 3)
    else:
        f = 1

    return f, SHORT_PREFIX[f]


def nice_array(a: np.ndarray) -> tuple[np.ndarray, float, str]:
    """
    Scale an input array and return the scaled array, the scaling factor, and the
    corresponding unit prefix.

    Parameters
    ----------
    a : array-like, or float
        Input array to be scaled.

    Returns
    -------
    scaled_array : np.ndarray
        The scaled array, of the same shape as `a`.
    scaling : float
        The scale factor applied to the input array.
    prefix : str
        The unit prefix corresponding to the scale factor (e.g., 'p' for pico).

    Examples
    --------
    >>> nice_array(np.array([2e-10, 3e-10]))
    (array([200., 300.]), 1e-12, 'p')
    """
    if np.isscalar(a):
        x = a
    elif len(a) == 1:
        a = np.asarray(a)
        x = a[0]
    else:
        a = np.asarray(a)
        x = max(np.ptp(a), abs(np.mean(a)))  # Account for tiny spread

    fac, prefix = nice_scale_prefix(x)
    return a / fac, fac, prefix


def plottable_array(x: np.ndarray, nice: bool = True, lim: Limit | None = None):
    """
    Similar to nice_array, but also considers limits for plotting

    Parameters
    ----------
    x: array-like
    nice: bool, default = True
        Scale array by some nice factor.
    xlim: tuple, default = None

    Returns
    -------
    scaled_array: np.ndarray
    factor: float
    prefix: str
    xmin: float
    xmax : float

    """
    x = np.asarray(x)
    if lim is not None:
        if lim[0] is None:
            xmin = x.min()
        else:
            xmin = lim[0]
        if lim[1] is None:
            xmax = x.max()
        else:
            xmax = lim[1]

    else:
        xmin = x.min()
        xmax = x.max()

    if nice:
        _, factor, p1 = nice_array([xmin, xmax])
    else:
        factor, p1 = 1, ""

    return x / factor, factor, p1, xmin, xmax


# -------------------------
# Units for ParticleGroup

PARTICLEGROUP_UNITS = {}
for k in ["n_particle", "status", "id", "n_alive", "n_dead"]:
    PARTICLEGROUP_UNITS[k] = pmd_unit("1")
for k in ["t", "z/c"]:
    PARTICLEGROUP_UNITS[k] = pmd_unit("s")
for k in [
    "energy",
    "kinetic_energy",
    "mass",
    "higher_order_energy_spread",
    "higher_order_energy",
]:
    PARTICLEGROUP_UNITS[k] = pmd_unit("eV")
for k in ["px", "py", "pz", "p", "pr", "ptheta"]:
    PARTICLEGROUP_UNITS[k] = pmd_unit("eV/c")
for k in ["x", "y", "z", "r", "Jx", "Jy"]:
    PARTICLEGROUP_UNITS[k] = pmd_unit("m")
for k in ["beta", "beta_x", "beta_y", "beta_z", "gamma", "bunching"]:
    PARTICLEGROUP_UNITS[k] = pmd_unit("1")
for k in ["theta", "bunching_phase"]:
    PARTICLEGROUP_UNITS[k] = pmd_unit("rad")
for k in ["charge", "species_charge", "weight"]:
    PARTICLEGROUP_UNITS[k] = pmd_unit("C")
for k in ["average_current"]:
    PARTICLEGROUP_UNITS[k] = pmd_unit("A")
for k in ["power"]:
    PARTICLEGROUP_UNITS[k] = pmd_unit("W")
for k in ["norm_emit_x", "norm_emit_y"]:
    PARTICLEGROUP_UNITS[k] = pmd_unit("m")
for k in ["norm_emit_4d"]:
    PARTICLEGROUP_UNITS[k] = multiply_units(pmd_unit("m"), pmd_unit("m"))
for k in ["Lz"]:
    PARTICLEGROUP_UNITS[k] = multiply_units(pmd_unit("m"), pmd_unit("eV/c"))
for k in ["xp", "yp"]:
    PARTICLEGROUP_UNITS[k] = pmd_unit("rad")
for k in ["x_bar", "px_bar", "y_bar", "py_bar"]:
    PARTICLEGROUP_UNITS[k] = sqrt_unit(pmd_unit("m"))
for component in ["", "x", "y", "z", "theta", "r"]:
    PARTICLEGROUP_UNITS[f"E{component}"] = pmd_unit("V/m")
    PARTICLEGROUP_UNITS[f"B{component}"] = pmd_unit("T")

# Twiss
for plane in ("x", "y"):
    for k in ("alpha", "etap"):
        PARTICLEGROUP_UNITS[f"twiss_{k}_{plane}"] = pmd_unit("1")
    for k in ("beta", "eta", "emit", "norm_emit"):
        PARTICLEGROUP_UNITS[f"twiss_{k}_{plane}"] = pmd_unit("m")
    for k in ("gamma",):
        PARTICLEGROUP_UNITS[f"twiss_{k}_{plane}"] = divide_units(
            pmd_unit("1"), pmd_unit("m")
        )


def pg_units(key: str) -> pmd_unit:
    """
    Returns a str representing the units of any attribute
    """

    # Basic cases
    if key in PARTICLEGROUP_UNITS:
        return PARTICLEGROUP_UNITS[key]

    # Operators
    for prefix in ["sigma_", "mean_", "min_", "max_", "ptp_", "delta_"]:
        if key.startswith(prefix):
            nkey = key[len(prefix) :]
            return pg_units(nkey)

    if key.startswith("cov_"):
        # NB: removeprefix, not strip("cov_") -- str.strip removes any leading/
        # trailing chars in the set {c,o,v,_}, which mangles subkeys like
        # "charge" (-> "harge"). removeprefix drops exactly the "cov_" prefix.
        subkeys = key.removeprefix("cov_").split("__")
        unit0 = PARTICLEGROUP_UNITS[subkeys[0]]

        unit1 = PARTICLEGROUP_UNITS[subkeys[1]]

        return multiply_units(unit0, unit1)

    # Fields
    if key.startswith("electricField"):
        return pmd_unit("V/m")
    if key.startswith("magneticField"):
        return pmd_unit("T")
    if key.startswith("bunching_phase"):
        return pmd_unit("rad")
    if key.startswith("bunching"):
        return pmd_unit("1")

    raise ValueError(f"No known unit for: {key}")


# -------------------------
# Special parsers


def parse_bunching_str(s):
    """
    Parse a string of the on of the forms to extract the wavelength:
        'bunching_1.23e-4'
        'bunching_1.23e-4_nm'
        'bunching_phase_1.23e-4'

    Returns
    -------
    wavelength: float

    """
    assert s.startswith("bunching_")

    # Remove bunching and phase prefixes
    s = s.replace("bunching_", "")
    s = s.replace("phase_", "")

    x = s.split("_")

    wavelength = float(x[0])

    if len(x) == 1:
        factor = 1
    elif len(x) == 2:
        unit = x[1]
        if unit == "m":
            factor = 1
        elif unit == "mm":
            factor = 1e-3
        elif unit == "µm" or unit == "um":
            factor = 1e-6
        elif unit == "nm":
            factor = 1e-9
        else:
            raise ValueError(f"Unparsable unit: {unit}")
    else:
        raise ValueError(f"Cannot parse {s}")

    return wavelength * factor


# -------------------------
# h5 tools


def write_unit_h5(h5, u):
    """
    Writes an pmd_unit to an h5 handle
    """

    h5.attrs["unitSI"] = u.unitSI
    h5.attrs["unitDimension"] = u.unitDimension
    h5.attrs["unitSymbol"] = u.unitSymbol


def read_unit_h5(h5):
    """
    Reads unit data from an h5 handle and returns a pmd_unit object
    """
    a = h5.attrs

    unitSI = a["unitSI"]
    unitDimension = tuple(a["unitDimension"])
    if "unitSymbol" not in a:
        unitSymbol = "unknown"
    else:
        unitSymbol = a["unitSymbol"]

    return pmd_unit(unitSymbol=unitSymbol, unitSI=unitSI, unitDimension=unitDimension)


def read_dataset_and_unit_h5(h5, expected_unit=None, convert=True):
    """
    Reads a dataset that has openPMD unit attributes.

    expected_unit can be a pmd_unit object, or a known unit str. Examples: 'kg', 'J', 'eV'

    If expected_unit is given, will check that the units are compatible.

    If convert, the data will be returned with the expected_units.


    Returns a tuple:
        np.array, pmd_unit

    """

    # Read the unit that is there.
    u = read_unit_h5(h5)

    # Simple case
    if not expected_unit:
        return np.array(h5), u

    if isinstance(expected_unit, str):
        # Try to get unit
        expected_unit = pmd_unit(expected_unit)

    # Check dimensions
    du = divide_units(u, expected_unit)

    assert du.unitDimension == (0, 0, 0, 0, 0, 0, 0), "incompatible units"

    if convert:
        fac = du.unitSI
        return fac * np.array(h5), expected_unit
    else:
        return np.array(h5), u


def write_dataset_and_unit_h5(h5, name, data, unit=None):
    """
    Writes data and pmd_unit to h5[name]

    See: read_dataset_and_unit_h5
    """
    h5[name] = data

    if unit:
        write_unit_h5(h5[name], unit)
