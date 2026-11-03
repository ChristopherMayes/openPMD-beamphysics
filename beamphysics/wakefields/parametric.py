"""
Parametric short-range longitudinal wake of a chain of RF cavities.

This module provides the analytical wake model of Zagorodnov et al. (2021),
which fits the point-charge wake of a whole RF module, cavities together with
their bellows and transitions, as a sum of four terms. The model is singular at
the source particle, so it is tabulated and applied through its antiderivative.

Classes
-------
ParametricWakefield
    Parametric wake model, Eq. 18 of Zagorodnov et al. (2021)

Data
----
ZAGORODNOV_2021_TABLE_IV
    Published coefficient sets for the European XFEL injector modules
"""

from __future__ import annotations

import numpy as np

from .base import WakefieldBase

__all__ = ["ParametricWakefield", "ZAGORODNOV_2021_TABLE_IV"]

# Table IV of Zagorodnov, Tomin, Chen and Brinker, NIM A (2021),
# doi:10.1016/j.nima.2021.165111 (arXiv:2010.10204). Units are those of the paper:
# A, C in V/pC, B in V mm^(1/2)/pC, D in V mm/pC, k0, k1, k3 in 1/mm, k2 in 1/mm^(1/2).
ZAGORODNOV_2021_TABLE_IV: dict[str, dict[str, float]] = {
    # TESLA cryomodule: 8 nine-cell 1.3 GHz cavities with bellows
    "A1": dict(
        A=287, k0=0.413, B=4.75, k1=35.0, alpha=0.746, k2=1.52, C=12.1, k3=0.181, D=0
    ),
    # Third-harmonic 3.9 GHz cryomodule: 8 cavities with bellows and an input taper
    "AH1": dict(
        A=711,
        k0=1.94,
        B=1.50,
        k1=106,
        alpha=0.661,
        k2=0.201,
        C=49.2,
        k3=0.154,
        D=0.0114,
    ),
    # Transverse deflecting structure: 16 cells
    "TDS": dict(
        A=40.7, k0=0.239, B=0.524, k1=6.54, alpha=0.655, k2=0.140, C=1.57, k3=0.159, D=0
    ),
}

# Gauss-Legendre nodes per panel in the quadratures below.
_GL_NODES = 12

# Largest node array built at once in `wake_potential`.
_MAX_NODES = 2_000_000

# Bunch density beyond this many sigma is neglected (exp(-32) ~ 1e-14).
_GAUSSIAN_CUTOFF = 8.0


def _gauss_legendre_panels(
    upper: np.ndarray, n_panels: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Composite Gauss-Legendre nodes and weights on [0, upper] for each upper limit.

    Returns arrays of shape (len(upper), n_panels * _GL_NODES).
    """
    x, w = np.polynomial.legendre.leggauss(_GL_NODES)
    edges = np.linspace(0.0, 1.0, n_panels + 1)
    mid = (edges[:-1] + edges[1:]) / 2
    half = (edges[1] - edges[0]) / 2
    # Unit-interval nodes and weights, then scaled by each upper limit.
    nodes = (mid[:, None] + half * x[None, :]).ravel()
    weights = np.tile(half * w, n_panels)
    upper = np.asarray(upper, dtype=float)[:, None]
    return upper * nodes[None, :], upper * weights[None, :]


class ParametricWakefield(WakefieldBase):
    """
    Parametric short-range longitudinal wake of a chain of RF cavities.

    The point-charge wake of a whole RF module, with s = -z >= 0 the distance
    behind the source particle, is Eq. 18 of Zagorodnov et al. (2021):

    $$w(s) = \\theta(s) \\left( A e^{-\\sqrt{k_0 s}}
    + B \\frac{\\cos\\big((k_1 s)^\\alpha\\big)}{\\sqrt{s} + k_2 s}
    + C \\cos(k_3 s) + D \\, \\delta(s) \\right)$$

    The A term is the short-range wake of an infinite chain of cavities (Bane
    2003), the B term that of an isolated cavity, the diffraction model (Bane and
    Sands 1987) with a cosine for the oscillation seen with short bunches, the C
    term is the long-range behaviour at the tail of a long bunch, and the D term
    stands for step-out or collimator-like features. The fit holds for Gaussian
    bunches with rms length from 10 µm to 2 mm.

    The wake per unit length, as returned by `wake`, is w(s) / length. Because the
    B term diverges as $s^{-1/2}$, `wake` returns inf at z = 0 and the model must
    be tabulated and applied through `antiderivative`, which also carries the D
    term: `TabularWakefield.from_wakefield` and `convolve_density` do this
    automatically.

    Parameters
    ----------
    length : float
        Length of the structure (m), which converts the wake of the whole module
        to V/C/m.
    A : float, optional
        Amplitude of the periodic-chain term (V/C). Default is 0.
    k0 : float, optional
        Wavenumber of the periodic-chain term (1/m). Default is 0.
    B : float, optional
        Amplitude of the diffraction term (V m^(1/2)/C). Default is 0.
    k1 : float, optional
        Wavenumber of the diffraction-term oscillation (1/m). Default is 0.
    alpha : float, optional
        Exponent of the diffraction-term oscillation. Default is 1.
    k2 : float, optional
        Crossover of the diffraction term from $s^{-1/2}$ to $s^{-1}$
        (1/m^(1/2)). Default is 0.
    C : float, optional
        Amplitude of the long-range term (V/C). Default is 0.
    k3 : float, optional
        Wavenumber of the long-range term (1/m). Default is 0.
    D : float, optional
        Strength of the delta-function term (V m/C). Default is 0.

    Notes
    -----
    The paper lists its coefficients in mm and V/pC; use `from_mm_pC` or
    `from_zagorodnov_2021` for those.

    The C term does not decay and the B term falls only as 1/s, so there is no
    distance beyond which the wake is negligible. `default_zmax` covers one period
    of the C term; pass zmax explicitly to cover the whole bunch.

    References
    ----------
    I. Zagorodnov, S. Tomin, Y. Chen, F. Brinker, "Experimental validation of
    collective effects modeling at injector section of x-ray free-electron laser",
    Nucl. Instrum. Methods A 995, 165111 (2021), arXiv:2010.10204.

    Examples
    --------
    >>> wake = ParametricWakefield.from_zagorodnov_2021("AH1", length=10.5)
    >>> table = TabularWakefield.from_wakefield(wake, zmax=20e-3, n=2001)
    >>> V = wake.wake_potential(z, sigma=100e-6)  # Gaussian bunch, V/C/m
    """

    def __init__(
        self,
        *,
        length: float,
        A: float = 0.0,
        k0: float = 0.0,
        B: float = 0.0,
        k1: float = 0.0,
        alpha: float = 1.0,
        k2: float = 0.0,
        C: float = 0.0,
        k3: float = 0.0,
        D: float = 0.0,
    ) -> None:
        if length <= 0:
            raise ValueError(f"length must be positive, got {length}")
        for name, k in (("k0", k0), ("k1", k1), ("k2", k2), ("k3", k3)):
            if k < 0:
                raise ValueError(f"{name} must be non-negative, got {k}")
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")

        self.length = float(length)
        self.A = float(A)
        self.k0 = float(k0)
        self.B = float(B)
        self.k1 = float(k1)
        self.alpha = float(alpha)
        self.k2 = float(k2)
        self.C = float(C)
        self.k3 = float(k3)
        self.D = float(D)

    @classmethod
    def from_mm_pC(
        cls,
        *,
        length: float,
        A: float = 0.0,
        k0: float = 0.0,
        B: float = 0.0,
        k1: float = 0.0,
        alpha: float = 1.0,
        k2: float = 0.0,
        C: float = 0.0,
        k3: float = 0.0,
        D: float = 0.0,
    ) -> ParametricWakefield:
        """
        Construct from coefficients in the units of Zagorodnov et al. (2021).

        The paper writes s in mm and w in V/pC for the whole module. For Eq. 18 to
        be dimensionally consistent the coefficients are then A, C in V/pC, B in
        V mm^(1/2)/pC, D in V mm/pC, k0, k1, k3 in 1/mm and k2 in 1/mm^(1/2).

        Parameters
        ----------
        length : float
            Length of the structure (m).
        A, k0, B, k1, alpha, k2, C, k3, D : float, optional
            Coefficients of Eq. 18 in the units above. Default is 0, except alpha
            which defaults to 1.

        Returns
        -------
        ParametricWakefield

        Examples
        --------
        The TESLA cryomodule row of Table IV:

        >>> wake = ParametricWakefield.from_mm_pC(
        ...     length=12.2, A=287, k0=0.413, B=4.75, k1=35.0, alpha=0.746,
        ...     k2=1.52, C=12.1, k3=0.181, D=0,
        ... )
        """
        mm = 1e-3
        pC = 1e-12
        return cls(
            length=length,
            A=A / pC,
            k0=k0 / mm,
            B=B * np.sqrt(mm) / pC,
            k1=k1 / mm,
            alpha=alpha,
            k2=k2 / np.sqrt(mm),
            C=C / pC,
            k3=k3 / mm,
            D=D * mm / pC,
        )

    @classmethod
    def from_zagorodnov_2021(cls, structure: str, length: float) -> ParametricWakefield:
        """
        Construct from a row of Table IV of Zagorodnov et al. (2021).

        Parameters
        ----------
        structure : str
            'A1' for the TESLA 1.3 GHz cryomodule, 'AH1' for the 3.9 GHz
            third-harmonic cryomodule, or 'TDS' for the transverse deflecting
            structure of the European XFEL injector.
        length : float
            Length of the structure (m), over which the wake of the whole module
            is spread.

        Returns
        -------
        ParametricWakefield

        Raises
        ------
        KeyError
            If structure is not one of the published sets.

        Examples
        --------
        >>> wake = ParametricWakefield.from_zagorodnov_2021("A1", length=12.2)
        """
        try:
            coefficients = ZAGORODNOV_2021_TABLE_IV[structure]
        except KeyError:
            raise KeyError(
                f"Unknown structure {structure!r}. Choose from "
                f"{sorted(ZAGORODNOV_2021_TABLE_IV)}."
            ) from None

        return cls.from_mm_pC(length=length, **coefficients)

    def __repr__(self) -> str:
        fields = ", ".join(
            f"{name}={getattr(self, name):.6g}"
            for name in ("A", "k0", "B", "k1", "alpha", "k2", "C", "k3", "D")
        )
        return f"ParametricWakefield({fields}, length={self.length:.6g})"

    @property
    def is_singular(self) -> bool:
        """Whether the diffraction term is present, which diverges at z = 0."""
        return self.B != 0

    # -------------------------------------------------------------------------
    # Wake and antiderivative
    # -------------------------------------------------------------------------

    def _wake_regular(self, s: np.ndarray) -> np.ndarray:
        """The A, B and C terms of the whole-module wake at s > 0 (V/C)."""
        w = self.A * np.exp(-np.sqrt(self.k0 * s)) + self.C * np.cos(self.k3 * s)
        if self.B != 0:
            w = w + (
                self.B
                * np.cos((self.k1 * s) ** self.alpha)
                / (np.sqrt(s) + self.k2 * s)
            )
        return w

    def wake(self, z: np.ndarray | float) -> np.ndarray | float:
        """
        Evaluate the wake per unit length at position z.

        The delta-function term is excluded; it enters through `antiderivative`.

        Parameters
        ----------
        z : float or np.ndarray
            Longitudinal position (m). Negative z is behind the source.

        Returns
        -------
        W : float or np.ndarray
            Wakefield value (V/C/m). Returns 0 for z > 0, and inf at z = 0 when
            B is nonzero.
        """
        z = np.asarray(z, dtype=float)
        scalar_input = z.ndim == 0
        z = np.atleast_1d(z)

        out = np.zeros_like(z)
        behind = z < 0
        out[behind] = self._wake_regular(-z[behind])

        at_source = z == 0
        if np.any(at_source):
            out[at_source] = np.inf if self.B != 0 else self.A + self.C

        out /= self.length

        if scalar_input:
            return float(out[0])
        return out

    def _antiderivative_A(self, s: np.ndarray) -> np.ndarray:
        """Integral of the A term from 0 to s > 0."""
        if self.k0 == 0:
            return self.A * s
        r = np.sqrt(self.k0 * s)
        return 2 * self.A / self.k0 * (1 - (1 + r) * np.exp(-r))

    def _antiderivative_B_smooth(self, s: np.ndarray) -> np.ndarray:
        """Integral of the B term from 0 to s > 0 with the cosine replaced by 1."""
        if self.k2 == 0:
            return 2 * self.B * np.sqrt(s)
        return 2 * self.B / self.k2 * np.log1p(self.k2 * np.sqrt(s))

    def _antiderivative_B(self, s: np.ndarray) -> np.ndarray:
        """
        Integral of the B term from 0 to s > 0.

        With s = u^2 the integrand becomes the bounded function
        2 B cos((k1 u^2)^alpha) / (1 + k2 u), which is integrated by composite
        Gauss-Legendre quadrature. The panels are shared by all s: the integral
        is accumulated at the panel edges once, and each s adds its partial panel.
        """
        if self.k1 == 0:
            return self._antiderivative_B_smooth(s)

        def g(u):
            return (
                2 * self.B * np.cos((self.k1 * u**2) ** self.alpha) / (1 + self.k2 * u)
            )

        u = np.sqrt(s)
        U = u.max()
        # The phase (k1 u^2)^alpha advances fastest at U; 4 panels per radian
        # there keeps every panel well below one oscillation.
        phase = (self.k1 * U**2) ** self.alpha
        n_panels = max(16, int(np.ceil(4 * phase)))
        h = U / n_panels
        edges = h * np.arange(n_panels + 1)

        x, w = np.polynomial.legendre.leggauss(_GL_NODES)
        mid = (edges[:-1] + edges[1:]) / 2
        nodes = mid[:, None] + (h / 2) * x[None, :]
        F_edges = np.concatenate([[0.0], np.cumsum((h / 2) * (g(nodes) @ w))])

        j = np.minimum((u / h).astype(int), n_panels - 1)
        a = edges[j]
        half = (u - a) / 2
        nodes = (a + half)[:, None] + half[:, None] * x[None, :]

        return F_edges[j] + half * (g(nodes) @ w)

    def _antiderivative_C(self, s: np.ndarray) -> np.ndarray:
        """Integral of the C term from 0 to s > 0."""
        if self.k3 == 0:
            return self.C * s
        return self.C / self.k3 * np.sin(self.k3 * s)

    def antiderivative(self, z: np.ndarray | float) -> np.ndarray | float:
        """
        Integral of the wake per unit length from z up to the source particle.

        $$F(z) = \\int_z^0 W(z') \\, dz'$$

        With s = -z, the terms integrate to

        $$\\frac{2A}{k_0}\\left[1 - \\left(1 + \\sqrt{k_0 s}\\right)
        e^{-\\sqrt{k_0 s}}\\right]
        + \\frac{C}{k_3} \\sin(k_3 s) + D$$

        plus the B term, which is $\\frac{2B}{k_2} \\ln(1 + k_2 \\sqrt{s})$ for
        $k_1 = 0$ and is otherwise integrated numerically after the substitution
        s = u^2, which removes the singularity. The delta function is counted in
        full for every z < 0, so it lands in the first bin of a bin-averaged table.

        Parameters
        ----------
        z : float or np.ndarray
            Longitudinal position (m). Negative z is behind the source.

        Returns
        -------
        F : float or np.ndarray
            Integrated wake (V/C). Returns 0 for z >= 0.
        """
        z = np.asarray(z, dtype=float)
        scalar_input = z.ndim == 0
        z = np.atleast_1d(z)

        out = np.zeros_like(z)
        behind = z < 0
        if np.any(behind):
            s = -z[behind]
            F = self._antiderivative_A(s) + self._antiderivative_C(s) + self.D
            if self.B != 0:
                F = F + self._antiderivative_B(s)
            out[behind] = F / self.length

        if scalar_input:
            return float(out[0])
        return out

    # -------------------------------------------------------------------------
    # Gaussian bunch
    # -------------------------------------------------------------------------

    def wake_potential(self, z: np.ndarray | float, sigma: float) -> np.ndarray | float:
        """
        Wake potential of a Gaussian bunch per unit bunch charge.

        $$V(z) = \\int_0^\\infty W(-t) \\, \\lambda(z + t) \\, dt
        + \\frac{D}{L} \\lambda(z)$$

        Here lambda is the normalized Gaussian density of rms length sigma, z is
        the position relative to the bunch centre with z < 0 towards the tail, and
        the integral runs over the charge ahead of z. The substitution t = u^2
        makes the integrand bounded, and it is integrated by composite
        Gauss-Legendre quadrature with enough panels to resolve the Gaussian and
        the oscillation of the B term. This is how the paper fits and checks the
        model (its Eqs. 20 and 21), and it is the reference for a tabulated or
        binned convolution.

        Parameters
        ----------
        z : float or np.ndarray
            Position relative to the bunch centre (m).
        sigma : float
            Rms bunch length (m).

        Returns
        -------
        V : float or np.ndarray
            Wake potential per unit charge (V/C/m). Multiply by the bunch charge
            for V/m.
        """
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")

        z = np.asarray(z, dtype=float)
        scalar_input = z.ndim == 0
        z = np.atleast_1d(z)

        def density(x):
            return np.exp(-0.5 * (x / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)

        out = self.D * density(z)

        # Charge ahead of z lies within the Gaussian cutoff.
        t_max = _GAUSSIAN_CUTOFF * sigma - z
        active = t_max > 0
        if np.any(active):
            za = z[active]
            ta = t_max[active]
            T = ta.max()

            # Panels resolve the Gaussian, whose width in u is sigma / (2 u), the
            # phase of the B term and the period of the C term.
            n_panels = max(
                64,
                int(np.ceil(16 * T / sigma)),
                int(np.ceil(4 * (self.k1 * T) ** self.alpha)) if self.B else 0,
                int(np.ceil(4 * self.k3 * T)),
            )

            def integrate(za, ta):
                u, w = _gauss_legendre_panels(np.sqrt(ta), n_panels)
                t = u**2
                # 2 u w(u^2): the B term becomes 2 B cos((k1 t)^alpha) / (1 + k2 u).
                f = (
                    2
                    * u
                    * (
                        self.A * np.exp(-np.sqrt(self.k0 * t))
                        + self.C * np.cos(self.k3 * t)
                    )
                )
                if self.B != 0:
                    f = f + (
                        2
                        * self.B
                        * np.cos((self.k1 * t) ** self.alpha)
                        / (1 + self.k2 * u)
                    )
                return np.sum(f * density(za[:, None] + t) * w, axis=1)

            # Bound the size of the node arrays.
            chunk = max(1, _MAX_NODES // (n_panels * _GL_NODES))
            out[active] += np.concatenate(
                [
                    integrate(za[i : i + chunk], ta[i : i + chunk])
                    for i in range(0, len(za), chunk)
                ]
            )

        out /= self.length

        if scalar_input:
            return float(out[0])
        return out

    def _gaussian_grid(self, sigma: float) -> tuple[np.ndarray, np.ndarray]:
        """Uniform grid over the bunch and the normalized density on it."""
        z = np.linspace(-_GAUSSIAN_CUTOFF * sigma, _GAUSSIAN_CUTOFF * sigma, 1601)
        lam = np.exp(-0.5 * (z / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)
        return z, lam

    def loss_factor(self, sigma: float) -> float:
        """
        Loss factor of a Gaussian bunch per unit length.

        $$\\kappa = \\int V(z) \\, \\lambda(z) \\, dz$$

        Parameters
        ----------
        sigma : float
            Rms bunch length (m).

        Returns
        -------
        float
            Loss factor (V/C/m). Positive for energy loss.
        """
        z, lam = self._gaussian_grid(sigma)
        return float(np.trapezoid(self.wake_potential(z, sigma) * lam, z))

    def rms_energy_spread(self, sigma: float) -> float:
        """
        Rms spread of the wake potential over a Gaussian bunch per unit length.

        $$\\sigma_V = \\sqrt{\\int (V(z) - \\kappa)^2 \\, \\lambda(z) \\, dz}$$

        Parameters
        ----------
        sigma : float
            Rms bunch length (m).

        Returns
        -------
        float
            Rms spread of the wake potential per unit charge (V/C/m).
        """
        z, lam = self._gaussian_grid(sigma)
        V = self.wake_potential(z, sigma)
        kappa = np.trapezoid(V * lam, z)
        return float(np.sqrt(np.trapezoid((V - kappa) ** 2 * lam, z)))

    # -------------------------------------------------------------------------
    # Tabulation defaults and impedance
    # -------------------------------------------------------------------------

    @property
    def default_zmax(self) -> float:
        """
        Default tabulation range (m).

        $$z_{\\max} = \\max\\left(\\frac{25}{k_0}, \\frac{2\\pi}{k_3}\\right)$$

        over the terms present: the A term has fallen to exp(-5), and the C term
        has completed one period. The C term does not decay, so this is a
        convenience for the bunch lengths the fit is meant for, not a distance
        beyond which the wake is negligible. Pass zmax to cover the bunch.

        Raises
        ------
        NotImplementedError
            If neither the A nor the C term sets a scale.
        """
        scales = []
        if self.A != 0 and self.k0 > 0:
            scales.append(25 / self.k0)
        if self.C != 0 and self.k3 > 0:
            scales.append(2 * np.pi / self.k3)
        if not scales:
            return super().default_zmax
        return max(scales)

    @property
    def min_wavelength(self) -> float:
        """
        Shortest length scale of the wake (m).

        $$\\lambda_{\\min} = \\min\\left(\\frac{1}{k_0}, \\frac{(2\\pi)^{1/\\alpha}}{k_1},
        \\frac{2\\pi}{k_3}\\right)$$

        over the terms present: the decay scale of the A term, the distance over
        which the B term completes its first oscillation, and the period of the C
        term.

        Raises
        ------
        NotImplementedError
            If no term sets a scale.
        """
        scales = []
        if self.A != 0 and self.k0 > 0:
            scales.append(1 / self.k0)
        if self.B != 0 and self.k1 > 0:
            scales.append((2 * np.pi) ** (1 / self.alpha) / self.k1)
        if self.C != 0 and self.k3 > 0:
            scales.append(2 * np.pi / self.k3)
        if not scales:
            return super().min_wavelength
        return min(scales)

    def impedance(self, k: np.ndarray | float) -> np.ndarray | complex:
        """
        Not available for this model.

        The C term does not decay, so its impedance is a pair of delta functions,
        and the B term has no closed-form transform. Tabulate the wake with
        `TabularWakefield.from_wakefield` for a numerical impedance.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError(
            "ParametricWakefield has no closed-form impedance: the C term is "
            "undamped and the B term has no analytic transform. Tabulate the wake "
            "with TabularWakefield.from_wakefield for a numerical impedance."
        )
