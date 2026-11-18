import re

from .statistics import KeyExpression, normalize_key
from .units import (
    deprecated_field_record_names,
    nice_array,
    parse_bunching_str,
    pmd_unit,
)

TEXLABEL = {
    # 'status'
    "t": "t",
    "energy": "E",
    "kinetic_energy": r"E_\text{kinetic}",
    # 'mass',
    "higher_order_energy_spread": r"\sigma_{E_{(2)}}",
    "higher_order_energy": r"E_{(2)}",
    "E": "E",
    "Ex": "E_x",
    "Ey": "E_y",
    "Ez": "E_z",
    "Er": "E_r",
    "B": "B",
    "Bx": "B_x",
    "By": "B_y",
    "Bz": "B_z",
    "Br": "B_r",
    "Etheta": r"E_{\theta}",
    "Btheta": r"B_{\theta}",
    "px": "p_x",
    "py": "p_y",
    "pz": "p_z",
    "p": "p",
    "pr": "p_r",
    "ptheta": r"p_{\theta}",
    "x": "x",
    "y": "y",
    "z": "z",
    "r": "r",
    "s": "s",
    "p0c": "p_0 c",
    "p0": "p_0",
    "dp": "p - p_0",
    "px_over_p0": "p_x/p_0",
    "py_over_p0": "p_y/p_0",
    "dp_over_p0": r"\delta",
    "dp/p0": r"\delta",
    "spin_x": "S_x",
    "spin_y": "S_y",
    "spin_z": "S_z",
    "ix_branch": r"\mathrm{ix\_branch}",
    "ix_ele": r"\mathrm{ix\_ele}",
    "location": r"\mathrm{location}",
    "charge_state": r"\mathrm{charge\_state}",
    "Jx": "J_x",
    "Jy": "J_y",
    "beta": r"\beta",
    "beta_x": r"\beta_x",
    "beta_y": r"\beta_y",
    "beta_z": r"\beta_z",
    "gamma": r"\gamma",
    "theta": r"\theta",
    "charge": "Q",
    "twiss_alpha_x": r"\text{Twiss}\ \alpha_x",
    "twiss_beta_x": r"\text{Twiss}\ \beta_x",
    "twiss_gamma_x": r"\text{Twiss}\ \gamma_x",
    "twiss_eta_x": r"\text{Twiss}\ \eta_x",
    "twiss_etap_x": r"\text{Twiss}\ \eta'_x",
    "twiss_emit_x": r"\text{Twiss}\ \epsilon_{x}",
    "twiss_norm_emit_x": r"\text{Twiss}\ \epsilon_{n, x}",
    "twiss_alpha_y": r"\text{Twiss}\ \alpha_y",
    "twiss_beta_y": r"\text{Twiss}\ \beta_y",
    "twiss_gamma_y": r"\text{Twiss}\ \gamma_y",
    "twiss_eta_y": r"\text{Twiss}\ \eta_y",
    "twiss_etap_y": r"\text{Twiss}\ \eta'_y",
    "twiss_emit_y": r"\text{Twiss}\ \epsilon_{y}",
    "twiss_norm_emit_y": r"\text{Twiss}\ \epsilon_{n, y}",
    # 'species_charge',
    # 'weight',
    "average_current": r"I_{av}",
    "norm_emit_x": r"\epsilon_{n, x}",
    "norm_emit_y": r"\epsilon_{n, y}",
    "norm_emit_4d": r"\epsilon_{4D}",
    "Lz": "L_z",
    "xp": "x'",
    "yp": "y'",
    "x_bar": r"\overline{x}",
    "px_bar": r"\overline{p_x}",
    "y_bar": r"\overline{y}",
    "py_bar": r"\overline{p_y}",
}


def texlabel(key: str):
    """
    Returns a tex label from a proper attribute name.

    Parameters
    ----------
    key : str
        any beamphysics attribute

    Returns
    -------
    tex: str
        A TeX fragment: a curated label when the key is known, otherwise
        the key itself wrapped in ``\\mathrm{...}``.


    Examples
    --------
        texlabel('cov_x__px')
        returns: '\\left<x, p_x\\right>'



    Notes:
    -----
        See matplotlib:
            https://matplotlib.org/stable/tutorials/text/mathtext.html

    """
    return _texlabel(deprecated_field_record_names(key))


def _texlabel(key: str) -> str:
    key = normalize_key(key)
    if key in TEXLABEL:
        return TEXLABEL[key]

    # Operators
    for prefix in ("sigma_", "mean_", "min_", "max_", "ptp_", "delta_", "abs_"):
        if key.startswith(prefix):
            key0 = key.removeprefix(prefix)
            tex0 = _texlabel(key0)
            prefix = prefix.removesuffix("_")

            if prefix == "abs":
                return rf"\left|{tex0}\right|"
            if prefix in ("min", "max"):
                return f"\\{prefix}({tex0})"
            if prefix == "ptp":
                return rf"\mathrm{{ptp}}({tex0})"
            if prefix == "sigma":
                return rf"\sigma_{{ {tex0} }}"
            if prefix == "delta":
                return rf"{tex0} - \left<{tex0}\right>"
            if prefix == "mean":
                return rf"\left<{tex0}\right>"

    if key.startswith("cov_"):
        subkeys = key.removeprefix("cov_").split("__")
        tex0 = _texlabel(subkeys[0])
        tex1 = _texlabel(subkeys[1])
        return rf"\left<{tex0}, {tex1}\right>"

    expression = KeyExpression.from_string(key)
    if expression is not None:
        return _expression_tex(expression)

    if key.startswith("bunching"):
        wavelength = parse_bunching_str(key)
        x, _, prefix = nice_array(wavelength)
        return rf"\mathrm{{bunching~at}}~{x:.1f}~\mathrm{{ {prefix}m }}"

    return rf"\mathrm{{ {key} }}"


def _group(tex: str) -> str:
    """Parenthesize a label that is not a single factor, such as `p - p_0`."""
    depth = 0
    for token in re.findall(r"\\left|\\right|[{} /-]", tex):
        if token in ("{", r"\left"):
            depth += 1
        elif token in ("}", r"\right"):
            depth -= 1
        elif depth == 0:
            return f"({tex})"
    return tex


def _number_tex(x: float) -> str:
    mantissa, _, exponent = repr(x).partition("e")
    mantissa = mantissa.removesuffix(".0")
    if not exponent:
        return mantissa
    power = f"10^{{{int(exponent)}}}"
    if mantissa in ("1", "-1"):
        return mantissa.removesuffix("1") + power
    return rf"{mantissa} \times {power}"


def _expression_tex(expression: KeyExpression) -> str:
    r"""
    TeX fragment for an arithmetic key, as a single quotient: `x/y*z` is
    $x\,z/y$.
    """
    numerator, denominator = (
        [_group("c" if name == "c" else _texlabel(name)) for name in names]
        for names in (expression.numerator, expression.denominator)
    )
    if expression.scale == -1 and numerator:
        numerator[0] = "-" + numerator[0]
    elif expression.scale != 1 or not numerator:
        numerator.insert(0, _number_tex(expression.scale))
    if expression.divisor != 1:
        denominator.insert(0, _group(_number_tex(expression.divisor)))
    tex = r"\,".join(numerator)
    if len(denominator) > 1:
        tex += "/(" + r"\,".join(denominator) + ")"
    elif denominator:
        tex += "/" + denominator[0]
    return tex


def _units_str_to_tex(units: str) -> str:
    """
    TeX fragment for a display units string.

    Explicit power-of-ten scales from `pmd_unit.scaled_symbol`, as in `'1e-3'`,
    `'1e-3 m^2'` and `'pC/(1e-3)'`, are written as $10^{-3}$.
    """
    try:
        return pmd_unit(units).to_tex()
    except (ValueError, KeyError):
        pass

    scaled = re.fullmatch(r"1e([+-]?\d+)(?: (.+))?", units)
    if scaled:
        tex = f"10^{{{int(scaled.group(1))}}}"
        if scaled.group(2):
            tex += rf"\,{_units_str_to_tex(scaled.group(2))}"
        return tex

    per_group = re.fullmatch(r"([^/]+)/\((.+)\)", units)
    if per_group:
        numerator = _units_str_to_tex(per_group.group(1))
        return f"{numerator}/({_units_str_to_tex(per_group.group(2))})"

    fallback = units.replace("*", r"{\cdot}")
    return rf"\mathrm{{ {fallback} }}"


def mathlabel(*keys, units=None, tex=True):
    """
    Axis/legend label for one or more attribute keys, with optional units.

    With no keys, the label is the units alone — useful for axes that show
    a bare quantity of known units, like the marginal histograms.

    Parameters
    ----------
    *keys : str
        any beamphysics attributes

    units : pmd_unit or str or None
        Units to append in parentheses. A ``pmd_unit``, or a string that
        parses as a unit symbol, renders via ``pmd_unit.to_tex()`` (so
        compound symbols like eV/c, sqrt(m), m^2, and parenthesized groups
        come out right); explicit scales such as `1e-3` render as $10^{-3}$;
        any other string is shown as-is, with ``*`` as a dot.

    tex : bool, default=True
        if True, a mathtext (TeX) string wrapped in $$ will be returned.
        Uses `beamphysics.labels.texlabel` to get a proper label

    Returns
    -------
    label: str
        A mathtext string when ``tex`` is True, otherwise a plain-text label.


    Examples
    --------
        mathlabel('x_bar', 'sigma_x', units='µC')
        returns:
        '$\\overline{x}, \\sigma_{ x }~(\\mathrm{µC})$'

        mathlabel(units='fC/µm')
        returns:
        '$\\mathrm{fC}/\\mathrm{µm}$'

    """
    if not tex:
        # Compare via str(): a pmd_unit object is always truthy, but a
        # dimensionless unit has an empty symbol and should add no "()".
        units_str = "" if units is None else str(units)
        if not keys:
            return units_str
        label = ", ".join(keys)
        if units_str:
            label = rf"{label} ({units_str})"
        return label

    # Units -> TeX fragment ('' when there are none).
    if units is None or str(units) == "":
        units_frag = ""
    elif isinstance(units, pmd_unit):
        units_frag = units.to_tex()
    else:
        units_frag = _units_str_to_tex(str(units))

    if not keys:
        return rf"${units_frag}$" if units_frag else ""

    label = ", ".join(texlabel(key) for key in keys)
    if units_frag:
        label = rf"{label}~({units_frag})"
    return rf"${label}$"
