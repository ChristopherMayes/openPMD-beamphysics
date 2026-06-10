"""
Simple units functionality for the openPMD beamphysics records.

For more advanced units, use a package like Pint:
    https://pint.readthedocs.io/
"""

from __future__ import annotations

import copy
import math
import re
import unicodedata
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
# 7-tuple of base-SI exponents (length, mass, time, current, temperature,
# mol, luminous). The openPMD standard specifies ``unitDimension`` as an
# array of 7 float64 values, so we store floats uniformly — this also lets
# fractional powers (e.g. from sqrt_unit) round-trip through the same code
# path as integer ones. Equality with int tuples still works (``1.0 == 1``).
Dimension = tuple[float, float, float, float, float, float, float]

# Some characters have look-alike twins in Unicode: the Greek letter µ and
# the "micro sign", the Greek letter Ω and the "ohm sign". A pasted symbol
# may contain either twin, and they are visually indistinguishable. The
# parser converts each pair to the single character the unit tables use.
# NFC normalization already converts OHM SIGN (U+2126) to GREEK CAPITAL
# OMEGA (U+03A9); this table handles the mu pair, which NFC leaves alone:
# GREEK SMALL MU (U+03BC) becomes MICRO SIGN (U+00B5), the character in
# SHORT_PREFIX_FACTOR (and what Option-m types on a Mac).
_HOMOGLYPH_TRANSLATION = str.maketrans({"μ": "µ"})

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
    pmd_unit('eV', 1.602176634e-19, (2.0, 1.0, -2.0, 0.0, 0.0, 0.0, 0.0))

    If unitSI=0 (default), `pmd_unit` may be initialized with a known symbol:

    >>> pmd_unit('T')
    pmd_unit('T', 1.0, (0.0, 1.0, -2.0, -1.0, 0.0, 0.0, 0.0))

    Compound units can be created using * and / operators in the symbol string:

    >>> pmd_unit('eV/c')
    pmd_unit('eV/c', 5.344285992678308e-28, (1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0))

    >>> pmd_unit('kg*m/s')
    pmd_unit('kg*m/s', 1.0, (1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0))

    Parsing is left-associative ('V/eV/c' means '(V/eV)/c'); parentheses
    group a sub-expression into a single operand:

    >>> pmd_unit('V/(eV/c)')
    pmd_unit('V/(eV/c)', 1.8711573470618972e+27, (1.0, 0.0, -2.0, -1.0, 0.0, 0.0, 0.0))

    Alternatively, use the `from_symbol` class method for explicit symbol lookup:

    >>> pmd_unit.from_symbol('A*s')
    pmd_unit('A*s', 1.0, (0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0))

    Equality follows the openPMD standard: the symbol is only a display
    label, and two units are equal when they share the same ``unitDimension``
    and ``unitSI`` (compared with a small relative tolerance):

    >>> pmd_unit("T") == pmd_unit("T")
    True
    >>> pmd_unit("eV") == pmd_unit("T")
    False
    >>> pmd_unit("1/s") == pmd_unit("Hz")
    True
    >>> pmd_unit("J/m") == pmd_unit("eV/m")  # different SI scales, unequal
    False
    """

    def __init__(
        self,
        unitSymbol: str = "",
        unitSI: int | float = 0,
        unitDimension: str | Sequence[int | float] = (0, 0, 0, 0, 0, 0, 0),
    ):
        # The openPMD standard defines unitSI as a positive conversion factor
        # to the corresponding SI unit. Negative scales would invert the sign
        # of every value an instrument writes through this unit, so they are
        # rejected here. (unitSI=0 is permitted: it means "not given", and
        # the symbol is looked up instead, below.)
        if unitSI < 0:
            raise ValueError(
                f"unitSI must be non-negative (got {unitSI!r}). "
                "The openPMD standard defines unitSI as a conversion factor to SI."
            )

        # Use the provided values directly when unitSI is given explicitly, or
        # while NAMED_UNITS is still being built (known_unit not yet populated).
        # Otherwise look the symbol up / parse it.
        if unitSI != 0 or not known_unit:
            self._unitSymbol = unitSymbol
            # float, per the openPMD standard (unitSI is a float64 attribute).
            self._unitSI = float(unitSI)
            if isinstance(unitDimension, str):
                self._unitDimension = dimension(unitDimension)
            else:
                self._unitDimension = make_dimension(unitDimension)
        else:
            u = self.from_symbol(unitSymbol)
            self._unitSymbol = u._unitSymbol
            self._unitSI = u._unitSI
            self._unitDimension = u._unitDimension

    @classmethod
    def from_symbol(cls, unitSymbol: str) -> pmd_unit:
        """
        Create a pmd_unit from a symbol string, with automatic lookup and parsing.

        This is the single, recursive parser. It handles:

        - Known units ('eV', 'm', 'T') and aliases ('Ohm', 'deg')
        - Compound expressions ('eV/c', 'kg*m/s') — each token recurses here
        - Parenthesized grouping ('V/(eV/c)', '(eV*s)^2')
        - Power notation ('m^2', 's^-1', 'm^(-3/2)')
        - Square root notation ('sqrt(m)', 'sqrt(kg*m)')
        - SI-prefixed known units ('keV', 'mm', 'mrad')

        Every recursive call parses a strictly shorter string (a token, a
        sqrt/group inner expression, or a power base) — never the unchanged
        input — so parsing always terminates.

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
        # Canonicalize the input:
        #
        # - Look-alike Unicode characters are converted to the ones the
        #   unit tables use, so a pasted Ω or µ parses no matter which of
        #   its visually identical twins it actually is. NFC normalization
        #   handles the ohm/omega pair; _HOMOGLYPH_TRANSLATION handles
        #   Greek mu vs the micro sign (see its definition for details).
        # - Outer whitespace and whitespace around "^" are stripped, so
        #   "  eV/c", "kg*m / s", and "m ^ 2" parse like their tight forms.
        #   We do not split on whitespace (no implicit space-as-*), which
        #   would conflict with multi-token named units like "charge #" and
        #   with the SI-prefix fallback ("k eV" vs "keV").
        unitSymbol = re.sub(
            r"\s*\^\s*",
            "^",
            unicodedata.normalize("NFC", unitSymbol)
            .translate(_HOMOGLYPH_TRANSLATION)
            .strip(),
        )

        # Check if known_unit dict has been populated
        if not known_unit:
            raise ValueError(
                f"Cannot lookup unitSymbol '{unitSymbol}': known_unit dict not yet initialized. "
                "Use pmd_unit(unitSymbol, unitSI, unitDimension) with explicit parameters instead."
            )

        # Known unit (or alias) lookup has priority. Return a shallow copy,
        # never the shared cached object: callers reassign ``_unitSymbol`` on
        # the result, which would otherwise mutate the global NAMED_UNITS
        # entry. (All three slots are immutable: str, float, tuple.)
        if unitSymbol in known_unit:
            result = copy.copy(known_unit[unitSymbol])
            result._unitSymbol = unitSymbol
            return result

        # Compound expression: split on top-level "*" and "/" (parentheses
        # protect their contents) and recurse on each token.
        parts, operators = _tokenize_compound(unitSymbol)
        if len(parts) > 1:
            # Reject empty operands around an operator ("m*", "/m", "m**s").
            if any(p == "" for p in parts):
                raise ValueError(
                    f"Malformed unit symbol {unitSymbol!r}: empty operand around an operator."
                )

            token_units = [cls.from_symbol(p) for p in parts]
            result = token_units[0]
            for op, unit_obj in zip(operators, token_units[1:]):
                if op == "*":
                    result = result * unit_obj
                elif op == "/":
                    result = result / unit_obj

            # Canonical symbol: stripped tokens joined by the original
            # operators with no whitespace. So "eV / c" and "eV/c" share a
            # symbol (and therefore compare equal / hash identically). A
            # parenthesized token is re-emitted from its parsed unit, so
            # "( eV / c )" canonicalizes to "(eV/c)" — and the parens are
            # dropped entirely when the group is atomic ("(m)" -> "m").
            canonical_parts = []
            for p, u in zip(parts, token_units):
                if p.startswith("(") and p.endswith(")") and _parens_balanced(p[1:-1]):
                    s = u.unitSymbol
                    canonical_parts.append(s if _is_atomic_symbol(s) else f"({s})")
                else:
                    canonical_parts.append(p)
            result._unitSymbol = _join_compound(canonical_parts, operators)
            return result

        # --- Single token from here on. ---

        # sqrt() notation: sqrt(X) = X^0.5. The stored symbol is always the
        # canonical, whitespace-free "sqrt(<inner>)" form so that "sqrt( m )"
        # and "sqrt(m)" compare equal and hash identically. The
        # balanced-parens guard keeps a product like "sqrt(m)*sqrt(m)" (whose
        # final ")" is not the partner of the first "(") from being consumed
        # as one sqrt.
        sqrt_match = re.match(r"^sqrt\(\s*(.+?)\s*\)$", unitSymbol)
        if sqrt_match and _parens_balanced(sqrt_match.group(1)):
            base_unit = cls.from_symbol(sqrt_match.group(1))
            result = power_unit(base_unit, 0.5)
            result._unitSymbol = f"sqrt({base_unit.unitSymbol})"
            return result

        # Parenthesized grouping: "(eV/c)" parses the inner expression as a
        # unit of its own, so "V/(eV/c)" divides by the whole group and
        # "(eV*s)^2" exponentiates it (via the power branches below, whose
        # base recursion lands back here). The balanced-parens guard keeps
        # adjacent groups with no operator, like "(a)(b)", from matching as
        # one group with inner "a)(b".
        paren_group_match = re.match(r"^\((.+)\)$", unitSymbol)
        if paren_group_match and _parens_balanced(paren_group_match.group(1)):
            result = cls.from_symbol(paren_group_match.group(1))
            # Preserve the grouping in the stored symbol unless it is
            # redundant: "(eV/c)" stays, "(m)" canonicalizes to "m".
            s = result.unitSymbol
            if not _is_atomic_symbol(s):
                result._unitSymbol = f"({s})"
            return result

        # Power notation with a parenthesized fraction: m^(-3/2)
        paren_power_match = re.match(r"^(.+)\^\((-?\d+)/(\d+)\)$", unitSymbol)
        if paren_power_match:
            power = int(paren_power_match.group(2)) / int(paren_power_match.group(3))
            base_unit = cls.from_symbol(paren_power_match.group(1))
            result = power_unit(base_unit, power)
            # Preserve the user's spelling ("m^(1/2)", not "m^0.5").
            result._unitSymbol = unitSymbol
            return result

        # Power notation: m^2, m^-1, m^0.5 (the base recurses, so sqrt(m)^2
        # and (eV*s)^2 work)
        power_match = re.match(r"^(.+)\^(-?\d+(?:\.\d+)?)$", unitSymbol)
        if power_match:
            base_unit = cls.from_symbol(power_match.group(1))
            result = power_unit(base_unit, float(power_match.group(2)))
            # Preserve the user's spelling ("(eV*s)^2", not the distributed
            # "eV^2*s^2" that power_unit emits).
            result._unitSymbol = unitSymbol
            return result

        # SI-prefix fallback: e.g. "keV" -> kilo + "eV", "mm" -> milli + "m",
        # "mrad" -> milli + "rad". Only matches when the remainder is itself a
        # known unit; a bare prefix ("k", "M") or a prefixed dimensionless
        # identity ("k1") is not a valid unit. Longest prefixes first so "da"
        # (deca) wins over "d" (deci). This is the inverse of the prefixed
        # symbols simplify() emits, so those always round-trip.
        for prefix in sorted(SHORT_PREFIX_FACTOR, key=len, reverse=True):
            if prefix == "" or not unitSymbol.startswith(prefix):
                continue
            remainder = unitSymbol[len(prefix) :]
            if remainder in ("", "1") or remainder not in known_unit:
                continue
            base = known_unit[remainder]
            factor = SHORT_PREFIX_FACTOR[prefix]
            return cls(
                unitSymbol,
                unitSI=factor * base.unitSI,
                unitDimension=base.unitDimension,
            )

        # Unknown unit
        raise ValueError(f"Unknown unitSymbol: {unitSymbol}")

    def __hash__(self) -> int:
        # The openPMD standard defines a unit by its dimension and conversion
        # factor; the symbol is only a display label and is excluded here.
        # The hash uses the same rounded conversion factor that __eq__
        # compares, so equal units always hash alike, while units that merely
        # share a dimension (such as eV and J) hash differently.
        return hash((self.unitDimension, _canonical_unitSI(self.unitSI)))

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

    def __pow__(self, power: float) -> pmd_unit:
        return power_unit(self, power)

    def compatible_with(self, other: pmd_unit) -> bool:
        """Return ``True`` when ``other`` has the same dimension as ``self``.

        Compatibility is a pure-dimension check: two units are compatible
        iff a value expressed in one can be converted to the other by a
        single multiplicative SI factor (see :meth:`conversion_factor_to`).
        It does *not* require equal :attr:`unitSI`, so e.g. ``eV`` and
        ``J`` are compatible.

        Parameters
        ----------
        other : pmd_unit
            The unit to compare against.

        Returns
        -------
        bool
            ``True`` iff ``self.unitDimension == other.unitDimension``.

        Examples
        --------
        >>> pmd_unit("eV").compatible_with(pmd_unit("J"))
        True
        >>> pmd_unit("Hz").compatible_with(pmd_unit("1/s"))
        True
        >>> pmd_unit("m").compatible_with(pmd_unit("s"))
        False
        """
        if not isinstance(other, pmd_unit):
            raise TypeError(
                f"compatible_with expects a pmd_unit, got {type(other).__name__}"
            )
        return self.unitDimension == other.unitDimension

    def conversion_factor_to(self, other: pmd_unit) -> float:
        """Return the factor that converts a value in ``self`` to ``other``.

        For a value ``x`` measured in ``self``, the value in ``other`` is
        ``x * self.conversion_factor_to(other)``. Equivalent to
        ``self.unitSI / other.unitSI`` after a dimension check.

        Parameters
        ----------
        other : pmd_unit
            The target unit. Must have the same :attr:`unitDimension` as
            ``self``.

        Returns
        -------
        float
            Multiplicative factor mapping values in ``self`` to ``other``.

        Raises
        ------
        ValueError
            If ``other`` is not dimensionally :meth:`compatible_with` ``self``.

        Examples
        --------
        >>> pmd_unit("eV").conversion_factor_to(pmd_unit("J"))
        1.602176634e-19
        >>> pmd_unit("km").conversion_factor_to(pmd_unit("m"))
        1000.0
        """
        if not self.compatible_with(other):
            raise ValueError(
                f"Incompatible units: {self.unitSymbol!r} (dimension "
                f"{self.unitDimension}) cannot be converted to "
                f"{other.unitSymbol!r} (dimension {other.unitDimension})."
            )
        return self.unitSI / other.unitSI

    def __eq__(self, other) -> bool:
        # Per the openPMD standard, a unit is defined by its conversion
        # factor and dimension; the symbol is only a display label. Two units
        # are therefore equal when their dimensions match and their conversion
        # factors agree after rounding (_canonical_unitSI, shared with
        # __hash__). The rounding lets results of arithmetic like
        # (eV/c) * c -> eV compare equal to eV despite tiny floating-point
        # differences.
        if not isinstance(other, self.__class__):
            return False
        if self.unitDimension != other.unitDimension:
            return False
        return _canonical_unitSI(self.unitSI) == _canonical_unitSI(other.unitSI)

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

    def to_tex(self) -> str:
        """Render the stored unit symbol as a matplotlib-mathtext string.

        The translation is purely structural — no algebraic simplification
        or reordering. Each atomic unit name is wrapped in ``\\mathrm{...}``
        (so it renders upright), ``*`` becomes ``{\\cdot}``, ``/`` is kept
        as ``/``, ``^N`` becomes ``^{N}``, ``sqrt(X)`` becomes
        ``\\sqrt{...}``, and a parenthesized group ``(X)`` keeps its parens
        (recursing on the inner expression in both cases).

        The result is intended to be embedded inside an ``$...$`` mathtext
        block by the caller. The identity / empty symbol returns ``""``.

        Returns
        -------
        str
            A TeX fragment equivalent to :attr:`unitSymbol`.

        Examples
        --------
        >>> pmd_unit("eV/c").to_tex()
        '\\\\mathrm{eV}/\\\\mathrm{c}'

        >>> pmd_unit("kg*m/s").to_tex()
        '\\\\mathrm{kg}{\\\\cdot}\\\\mathrm{m}/\\\\mathrm{s}'

        >>> pmd_unit("m^2").to_tex()
        '\\\\mathrm{m}^{2}'

        >>> pmd_unit("sqrt(m)").to_tex()
        '\\\\sqrt{\\\\mathrm{m}}'
        """
        return _symbol_to_tex(self._unitSymbol)


def _symbol_to_tex(symbol: str) -> str:
    """Recursive worker for :meth:`pmd_unit.to_tex`."""
    if symbol == "":
        return ""
    if symbol == "1":
        return "1"

    # Top-level compound (``a*b``, ``a/b``, ``sqrt(a)*b``). Tokenize first so
    # the recursion handles nested ``sqrt(...)`` and per-token powers.
    if "*" in symbol or "/" in symbol:
        parts, operators = _tokenize_compound(symbol)
        if len(parts) > 1:
            tex_parts = [_symbol_to_tex(p) for p in parts]
            out = tex_parts[0]
            for op, p in zip(operators, tex_parts[1:]):
                out += r"{\cdot}" + p if op == "*" else "/" + p
            return out

    # sqrt(X) -> \sqrt{ <recursed X> }
    sqrt_match = re.match(r"^sqrt\((.+)\)$", symbol)
    if sqrt_match:
        inner = _symbol_to_tex(sqrt_match.group(1))
        return rf"\sqrt{{{inner}}}"

    # Parenthesized group: "(X)" -> ( <recursed X> ). The balanced guard
    # mirrors the parser's, so "(a)(b)" falls through to the atomic fallback
    # rather than matching as one group.
    paren_group_match = re.match(r"^\((.+)\)$", symbol)
    if paren_group_match and _parens_balanced(paren_group_match.group(1)):
        return "(" + _symbol_to_tex(paren_group_match.group(1)) + ")"

    # base^(n/d) -> \mathrm{base}^{n/d}
    paren_power_match = _TOKEN_PAREN_POWER_RE.match(symbol)
    if paren_power_match:
        base = _symbol_to_tex(paren_power_match.group(1))
        num = paren_power_match.group(2)
        den = paren_power_match.group(3)
        return rf"{base}^{{{num}/{den}}}"

    # base^N -> \mathrm{base}^{N}
    power_match = _TOKEN_POWER_RE.match(symbol)
    if power_match:
        base = _symbol_to_tex(power_match.group(1))
        return f"{base}^{{{power_match.group(2)}}}"

    # Atomic name (possibly with internal whitespace, e.g. "charge #").
    return rf"\mathrm{{{symbol}}}"


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

# Number of significant figures to which unitSI is rounded before it is
# compared or hashed. Twelve figures is precise enough to keep genuinely
# different conversion factors apart (for example eV versus J, or meters
# versus kilometers) while remaining coarse enough to ignore the tiny
# rounding error that ordinary unit arithmetic introduces -- reconstructing
# eV via (eV/c) * c, for instance, lands within roughly one part in 1e16 of
# the original value.
_UNITSI_SIG_FIGS = 12


def _canonical_unitSI(unitSI: float) -> float:
    """Round ``unitSI`` to a canonical value used by both ``__eq__`` and ``__hash__``.

    Two units are considered equal when they share a dimension and their
    rounded conversion factors are identical. Rounding to a fixed precision is
    deliberate: it makes equality transitive (if a == b and b == c then
    a == c), which a tolerance comparison like ``math.isclose`` cannot
    guarantee — two values can each be within tolerance of a middle value but
    not of each other. Transitivity is what lets ``__hash__`` include the
    conversion factor: were the hash to use the raw factor, two units that
    compare equal but differ in the last few bits could hash differently,
    breaking the rule that equal objects must hash equal. Including the
    factor in the hash (rather than hashing on the dimension alone) keeps
    units of the same dimension but different scale, such as ``eV`` and
    ``J``, from all colliding.

    Zero and non-finite factors are returned unchanged, since they have no
    meaningful significant-figure representation.
    """
    if unitSI == 0 or not math.isfinite(unitSI):
        return unitSI
    return float(f"{unitSI:.{_UNITSI_SIG_FIGS - 1}e}")


def _tokenize_compound(symbol: str) -> tuple[list[str], list[str]]:
    """Split ``symbol`` on top-level ``*`` and ``/`` (respecting parentheses).

    Returns ``(parts, operators)`` where each part is whitespace-stripped and
    ``operators[i]`` joins ``parts[i]`` to ``parts[i+1]``. Used by both the
    parser and :func:`power_unit` (for distributing an exponent across a
    compound symbol).
    """
    parts: list[str] = []
    operators: list[str] = []
    current = ""
    paren_depth = 0
    for char in symbol:
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
    parts.append(current)
    return [p.strip() for p in parts], operators


def _parens_balanced(symbol: str) -> bool:
    """True if parentheses in ``symbol`` are balanced and never close early."""
    depth = 0
    for char in symbol:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _join_compound(parts: Sequence[str], operators: Sequence[str]) -> str:
    """Inverse of :func:`_tokenize_compound`: re-emit a compound symbol."""
    out = parts[0]
    for op, p in zip(operators, parts[1:]):
        out += op + p
    return out


def _is_atomic_symbol(symbol: str) -> bool:
    """True if the symbol has no top-level ``*`` or ``/`` (parens are kept)."""
    parts, _ = _tokenize_compound(symbol)
    return len(parts) == 1


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

    Sort key: exact match beats prefixed, then no operators preferred, then
    shorter, then lexicographic.
    """
    best: pmd_unit | None = None
    best_key: tuple[int, int, int, str] | None = None

    for u in named_units:
        if u.unitDimension != target_dim or u.unitSI == 0:
            continue
        prefix_match = _closest_prefix(target_si / u.unitSI)
        if prefix_match is None:
            continue
        prefix, pf = prefix_match
        # A prefix on a dimensionless unit is meaningless: the empty symbol
        # becomes a bare prefix ("" -> "m", which re-parses as meters), and
        # "mrad"/"krad" would imply an angle for a pure-number ratio. Only an
        # exact (unprefixed) dimensionless match is valid.
        if prefix != "" and u.unitDimension == _DIMENSIONLESS:
            continue

        out_symbol = prefix + u.unitSymbol
        priority = 0 if prefix == "" else 1
        key = (priority, *_symbol_complexity(out_symbol))
        if best_key is not None and key >= best_key:
            continue

        best = (
            u if prefix == "" else pmd_unit(out_symbol, pf * u.unitSI, u.unitDimension)
        )
        best_key = key

    return best


def _best_compound_match(
    target_dim: Dimension,
    target_si: float,
    named_units: Sequence[pmd_unit],
) -> pmd_unit | None:
    """
    Find the simplest ``a*b`` or ``a/b`` of two named units that matches
    the target dimension and SI value exactly. Both ``a`` and ``b`` must be
    atomic (no ``*`` or ``/`` in the symbol) — a readability rule for
    simplify's output: ``T*m`` is a simplification, ``(J/m)*s`` is not.

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
        if not _is_atomic_symbol(u.unitSymbol):
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
    # Products never need parentheses: an operator appended on the right
    # acts on everything to its left, so "s1*s2" reads back correctly no
    # matter what s1 and s2 contain.
    symbol = s1 + "*" + s2
    d1 = u1.unitDimension
    d2 = u2.unitDimension
    dim = tuple(sum(x) for x in zip(d1, d2))
    unitSI = u1.unitSI * u2.unitSI

    return pmd_unit(symbol, unitSI=unitSI, unitDimension=dim)


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
        # A divisor that itself contains operators must be parenthesized,
        # because "a/b/c" reads back as (a/b)/c, not a/(b/c). The numerator
        # never needs parens: an operator appended on the right acts on
        # everything to its left. A numerator with an empty symbol (the
        # dimensionless unit) is written as "1".
        numerator = s1 if s1 else "1"
        divisor = s2 if _is_atomic_symbol(s2) else f"({s2})"
        symbol = f"{numerator}/{divisor}"
    d1 = u1.unitDimension
    d2 = u2.unitDimension
    dim = tuple(a - b for a, b in zip(d1, d2))
    unitSI = u1.unitSI / u2.unitSI

    return pmd_unit(symbol, unitSI=unitSI, unitDimension=dim)


def sqrt_unit(u: pmd_unit) -> pmd_unit:
    """
    Returns the square root of a unit.

    The emitted symbol uses the canonical ``sqrt(<inner>)`` form (no
    whitespace, no LaTeX wrapping) so it round-trips through the parser.

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

    result = power_unit(u, 0.5)
    symbol = u.unitSymbol
    if symbol not in ["", "1"]:
        result._unitSymbol = f"sqrt({symbol})"
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
    pmd_unit('m^2', 1.0, (2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    >>> power_unit(pmd_unit("m"), 0.5)
    pmd_unit('m^0.5', 1.0, (0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    """
    if not isinstance(u, pmd_unit):
        raise ValueError("`u` is not a pmd_unit instance")

    symbol = u.unitSymbol
    if symbol in ["", "1"]:
        new_symbol = symbol
    elif "*" in symbol or "/" in symbol:
        # Distribute the exponent across the tokens of a compound symbol:
        # (eV*s)^2 -> "eV^2*s^2". The distributed form is the conventional
        # spelling and composes with existing per-token exponents.
        new_symbol = _distribute_power_in_symbol(symbol, power)
    else:
        new_symbol = _format_power(symbol, power)

    unitSI = u.unitSI**power
    dim = tuple(d * power for d in u.unitDimension)

    return pmd_unit(new_symbol, unitSI=unitSI, unitDimension=dim)


_TOKEN_PAREN_POWER_RE = re.compile(r"^(.+)\^\((-?\d+)/(\d+)\)$")
_TOKEN_POWER_RE = re.compile(r"^(.+)\^(-?\d+(?:\.\d+)?)$")


def _format_power(base: str, power: float) -> str:
    """Render ``base^power`` with an integer exponent where possible."""
    if power == 1:
        return base
    if power == int(power):
        return f"{base}^{int(power)}"
    return f"{base}^{power}"


def _combine_token_power(token: str, factor: float) -> str:
    """Multiply a token's existing exponent by ``factor``.

    ``m`` with factor 2 becomes ``m^2``; ``s^2`` with factor 2 becomes
    ``s^4``; ``m^(1/2)`` with factor 2 becomes ``m^1`` (i.e. ``m``).
    Unknown / nested forms (e.g. ``sqrt(m)``) are wrapped as
    ``sqrt(m)^factor`` and resolved by the parser's recursion.
    """
    paren_match = _TOKEN_PAREN_POWER_RE.match(token)
    if paren_match:
        base = paren_match.group(1)
        existing = int(paren_match.group(2)) / int(paren_match.group(3))
    else:
        power_match = _TOKEN_POWER_RE.match(token)
        if power_match:
            base = power_match.group(1)
            existing = float(power_match.group(2))
        else:
            base = token
            existing = 1.0
    return _format_power(base, existing * factor)


def _distribute_power_in_symbol(symbol: str, power: float) -> str:
    """Distribute an exponent across a compound symbol (e.g. ``(eV*s)^2``)."""
    parts, operators = _tokenize_compound(symbol)
    new_parts = [_combine_token_power(p, power) for p in parts]
    return _join_compound(new_parts, operators)


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


def make_dimension(dim: Sequence[int | float]) -> Dimension:
    """
    Coerce a 7-element sequence of base-SI exponents into the canonical
    ``Dimension`` form: a 7-tuple of floats.

    The openPMD standard defines ``unitDimension`` as an array of 7
    ``float64`` values, so floats are used uniformly here. This also lets
    fractional exponents (e.g. from :func:`sqrt_unit`) share a single
    representation with integer ones. Negative zeros (from e.g. ``-1.0 * 0.0``
    inside :func:`power_unit`) are normalized to positive ``0.0`` so the
    ``repr`` is stable and exponent tuples compare cleanly as dict keys.
    """
    dim = tuple(0.0 if d == 0 else float(d) for d in dim)
    if len(dim) != 7:
        raise ValueError("Dimensions must be 7 elements.")
    return dim


def dimension(name: str) -> Dimension | None:
    try:
        # Coerce to the canonical float form so every construction path
        # (string name or explicit tuple) yields the same Dimension type.
        return make_dimension(DIMENSION[name])
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
    pmd_unit("m/s", 1, "velocity"),
    pmd_unit("eV", e_charge, "energy"),
    pmd_unit("J", 1, "energy"),
    pmd_unit("eV/c", e_charge / c_light, "momentum"),
    # The Newton (dimension L M T^-2, unitSI=1) is intentionally omitted.
    # It does *not* conflict with eV/m / MeV/m (those have unitSI = n *
    # e_charge, which is not within rel_tol=1e-9 of any SI-prefix factor
    # times N=1, so the atomic-match prefix fallback never reaches N). It
    # *does* however generate an "N/A" compound for the magnetic-rigidity
    # dimension (T*m), which would beat the conventional "T*m" output of
    # simplify() under the lexicographic tiebreaker. Until we have a way
    # to prefer accelerator-physics-familiar spellings in the compound
    # matcher, we leave SI-magnitude forces as "J/m".
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
        "c_light": known_unit["c"],
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
    Old style (deprecated)::

        unit("eV/c")

    New style (preferred)::

        pmd_unit("eV/c")
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
        return 1.0, ""

    p10 = np.log10(abs(scale))

    if p10 < -24:  # Limits of SI prefixes
        f = 1e-24
    elif p10 > 24:
        f = 1e24
    elif p10 < -1.5 or p10 > 2:
        f = 10 ** (p10 // 3 * 3)
    else:
        f = 1

    # Plain Python float (np.log10 produces np.float64 otherwise).
    f = float(f)
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
for k in ["average_current", "current"]:  # 'current' appears in slice statistics
    PARTICLEGROUP_UNITS[k] = pmd_unit("A")
for k in ["density"]:  # slice statistics: charge per slice length
    PARTICLEGROUP_UNITS[k] = pmd_unit("C") / pmd_unit("m")
for k in ["power"]:
    PARTICLEGROUP_UNITS[k] = pmd_unit("W")
for k in ["norm_emit_x", "norm_emit_y"]:
    PARTICLEGROUP_UNITS[k] = pmd_unit("m")
for k in ["norm_emit_4d"]:
    PARTICLEGROUP_UNITS[k] = pmd_unit("m") ** 2
for k in ["Lz"]:
    PARTICLEGROUP_UNITS[k] = pmd_unit("m") * pmd_unit("eV/c")
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
        PARTICLEGROUP_UNITS[f"twiss_{k}_{plane}"] = pmd_unit("1") / pmd_unit("m")


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
        return PARTICLEGROUP_UNITS[subkeys[0]] * PARTICLEGROUP_UNITS[subkeys[1]]

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
    if not s.startswith("bunching_"):
        raise ValueError(f"Invalid bunching key: {s!r} (must start with 'bunching_')")

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
    Writes a pmd_unit to an h5 handle.

    ``unitSI`` and ``unitDimension`` are written as float64 per the openPMD
    standard; ``unitSymbol`` as a UTF-8 string (it may contain e.g. 'Ω', 'µ').
    """

    h5.attrs["unitSI"] = float(u.unitSI)
    h5.attrs["unitDimension"] = np.asarray(u.unitDimension, dtype=np.float64)
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
        # Files written with fixed-length/bytes string attrs (e.g. via
        # np.bytes_) yield bytes here; a str symbol is required for unit
        # arithmetic and to_tex.
        if isinstance(unitSymbol, bytes):
            unitSymbol = unitSymbol.decode("utf-8")

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

    if not u.compatible_with(expected_unit):
        raise ValueError(
            f"Incompatible units: file unit {u.unitSymbol!r} (dimension "
            f"{u.unitDimension}) does not match expected "
            f"{expected_unit.unitSymbol!r} (dimension "
            f"{expected_unit.unitDimension})."
        )

    if convert:
        fac = u.conversion_factor_to(expected_unit)
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
