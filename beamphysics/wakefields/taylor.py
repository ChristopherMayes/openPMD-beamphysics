"""
Taylor-expanded 3D wakefield model.

This module implements the second-order Taylor expansion of the
longitudinal point-charge wake function near the reference axis,
following I. Zagorodnov, K. Bane, and G. Stupakov,
"Calculation of wakefields in 2D rectangular structures"
(Phys. Rev. ST Accel. Beams 18, 104401, 2015), as implemented in the
ocelot ``Wake`` physics process (``ocelot.cpbd.wake3D``).

The longitudinal wake for a source particle at transverse position
(x_s, y_s) and a witness particle at (x_w, y_w) is expanded as

$$w(x_s, y_s, x_w, y_w, s) = \\sum_{a \\le b} h_{ab}(s) \\, u_a u_b$$

with $u = (1, x_s, y_s, x_w, y_w)$, so each component $h_{ab}(s)$ is a
one-dimensional function of the source-witness distance s >= 0.
Transverse wakes follow from the Panofsky-Wenzel theorem by
integrating the transverse gradient of the longitudinal wake.

Component index meaning:

- 0 : constant
- 1 : x of the source particle
- 2 : y of the source particle
- 3 : x of the witness particle
- 4 : y of the witness particle

For example ``(0, 0)`` is the monopole longitudinal wake, ``(0, 4)`` the
vertical dipole wake, and ``(3, 3)``/``(2, 4)`` quadrupole-like terms.

Unlike the 1D wakefields in this package, the wake amplitudes here are in
[V/C] for the *whole structure* (the structure length is baked into the
table), and kicks are returned in [eV/c] rather than [eV/m].

Classes
-------
TaylorWakeComponent
    A single one-dimensional wake component h_ab(s)
TaylorWakefield
    Second-order Taylor-expanded 3D wakefield built from components

References
----------
- I. Zagorodnov, K.L.F. Bane, G. Stupakov, Phys. Rev. ST Accel. Beams 18,
  104401 (2015). https://doi.org/10.1103/PhysRevSTAB.18.104401
- K. Bane, G. Stupakov, I. Zagorodnov, "Analytical formulas for short bunch
  wakes in a flat dechirper", Phys. Rev. Accel. Beams 19, 084401 (2016).
- K. Bane, G. Stupakov, I. Zagorodnov, SLAC-PUB-16881 (2016).
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants

from ..units import c_light

__all__ = ["TaylorWakeComponent", "TaylorWakefield"]

# Free-space impedance [Ohm]
Z0 = scipy.constants.value("characteristic impedance of vacuum")

# Meaning of the Taylor indices
INDEX_LABELS = {0: "1", 1: "x_s", 2: "y_s", 3: "x_w", 4: "y_w"}


# -----------------------------------------------------------------------------
# Low-level numerical helpers (ported from ocelot.cpbd.wake3D)
# -----------------------------------------------------------------------------


def _triangle_filter(x: np.ndarray, order: int) -> np.ndarray:
    """Apply a triangular smoothing filter of the given order, in place."""
    n = x.shape[0]
    for _ in range(order):
        x[1:n] = (x[1:n] + x[0 : n - 1]) * 0.5
        x[0 : n - 1] = (x[1:n] + x[0 : n - 1]) * 0.5
    return x


def _derivative(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Numerical derivative dy/dx using central differences."""
    n = x.shape[0]
    dy = np.zeros(n)
    dy[1 : n - 1] = (y[2:n] - y[0 : n - 2]) / (x[2:n] - x[0 : n - 2])
    dy[0] = (y[1] - y[0]) / (x[1] - x[0])
    dy[n - 1] = (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2])
    return dy


def _cumtrapz(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Cumulative trapezoidal integral of y(x), starting at 0."""
    out = np.zeros(y.shape[0])
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return out


def _cumtrapz_uniform(h: float, y: np.ndarray) -> np.ndarray:
    """Cumulative trapezoidal integral of y on a uniform grid of spacing h."""
    out = np.zeros(y.shape[0])
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1])) * h
    return out


def _convolution(
    xu: np.ndarray, u: np.ndarray, xw: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convolution of two equally spaced functions."""
    hx = xu[1] - xu[0]
    wc = np.convolve(u, w) * hx
    x0 = xu[0] + xw[0]
    xc = x0 + np.arange(len(w) + len(u) - 1) * hx
    return xc, wc


def _wake_convolution(
    xb: np.ndarray, bunch: np.ndarray, xw: np.ndarray, wake: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convolve a bunch profile with a point wake sampled at arbitrary points.

    The wake is first interpolated onto the (uniform) bunch grid, with the
    self-term at zero distance counted at half weight.
    """
    nb = xb.shape[0]
    xwi = xb - xb[0]
    wake1 = np.interp(xwi, xw, wake, 0, 0)
    wake1[0] = wake1[0] * 0.5
    xc, wc = _convolution(xb, bunch, xwi, wake1)
    return xc[0:nb], wc[0:nb]


def _project_current(
    tau: np.ndarray, charge: np.ndarray, n_points: int, filter_order: int
) -> np.ndarray:
    """
    Project particle charges onto a uniform grid and form a current profile.

    Parameters
    ----------
    tau : np.ndarray
        Longitudinal coordinate of each particle [m], increasing toward
        the *tail* of the bunch (ocelot convention).
    charge : np.ndarray
        Charge of each particle [C].
    n_points : int
        Number of sampling points (before filter padding).
    filter_order : int
        Triangular smoothing filter order.

    Returns
    -------
    current : np.ndarray
        Array of shape (n, 2): column 0 is tau [m], column 1 is current [A].
    """
    s0 = np.min(tau)
    s1 = np.max(tau)
    if s1 <= s0:
        raise ValueError("Zero-length bunch: all particles have the same z")
    nf2 = int(np.floor(filter_order / 2.0))
    n_total = n_points + 2 * nf2

    ds = (s1 - s0) / (n_points - 2)
    s = s0 + np.arange(-nf2, n_total - nf2) * ds

    ip = (tau - s0) / ds
    i0 = np.floor(ip).astype(np.int64)
    frac = ip - i0
    i0 = i0 + nf2
    rho = np.bincount(i0, weights=(1 - frac) * charge, minlength=n_total)
    rho += np.bincount(i0 + 1, weights=frac * charge, minlength=n_total)

    if filter_order > 0:
        _triangle_filter(rho, filter_order)

    current = np.empty((n_total, 2))
    current[:, 0] = s
    current[:, 1] = rho * c_light / ds
    return current


# -----------------------------------------------------------------------------
# Wake components
# -----------------------------------------------------------------------------


@dataclass
class TaylorWakeComponent:
    """
    One component h_ab(s) of a Taylor-expanded wake.

    Each component describes a one-dimensional longitudinal wake function
    multiplying the transverse monomial ``u_a * u_b`` with
    u = (1, x_s, y_s, x_w, y_w). The wake is the sum of a tabulated part,
    a tabulated derivative-coupled part, and lumped R, L, 1/C circuit terms.

    Parameters
    ----------
    a, b : int
        Taylor indices in 0..4 (0: constant, 1: x_source, 2: y_source,
        3: x_witness, 4: y_witness). Order does not matter; they are
        stored with a <= b.
    s0, w0 : np.ndarray, optional
        Tabulated wake: distance behind the source s >= 0 [m] and wake
        amplitude [V/C] (times [1/m] per transverse index > 0).
        Positive w0 at (0, 0) means energy loss.
    s1, w1 : np.ndarray, optional
        Tabulated wake convolved with the derivative of the bunch profile
        (inductive-like term) [V*s/C].
    R : float, optional
        Lumped resistive term [Ohm]. Default 0.
    L : float, optional
        Lumped inductive term [H]. Default 0.
    Cinv : float, optional
        Lumped inverse capacitance [1/F]. Default 0.
    """

    a: int
    b: int
    s0: np.ndarray | None = None
    w0: np.ndarray | None = None
    s1: np.ndarray | None = None
    w1: np.ndarray | None = None
    R: float = 0.0
    L: float = 0.0
    Cinv: float = 0.0

    def __post_init__(self):
        if not (0 <= self.a <= 4 and 0 <= self.b <= 4):
            raise ValueError(
                f"Taylor indices must be in 0..4, got ({self.a}, {self.b})"
            )
        if self.a > self.b:
            self.a, self.b = self.b, self.a
        for attr in ("s0", "w0", "s1", "w1"):
            val = getattr(self, attr)
            if val is not None:
                setattr(self, attr, np.asarray(val, dtype=float))
        if (self.s0 is None) != (self.w0 is None):
            raise ValueError("s0 and w0 must be given together")
        if (self.s1 is None) != (self.w1 is None):
            raise ValueError("s1 and w1 must be given together")
        if self.s0 is not None and self.s0.shape != self.w0.shape:
            raise ValueError("s0 and w0 must have the same shape")
        if self.s1 is not None and self.s1.shape != self.w1.shape:
            raise ValueError("s1 and w1 must have the same shape")

    @property
    def key(self) -> tuple[int, int]:
        """Component key (a, b) with a <= b."""
        return (self.a, self.b)

    @property
    def label(self) -> str:
        """Human-readable label, e.g. 'h04: y_w'."""
        factors = [INDEX_LABELS[i] for i in (self.a, self.b) if i != 0]
        monomial = " * ".join(factors) if factors else "1"
        return f"h{self.a}{self.b}: {monomial}"

    def convolve(self, current: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Convolve this component with a (generalized) current profile.

        Parameters
        ----------
        current : np.ndarray
            Array of shape (n, 2): column 0 is the longitudinal grid tau [m]
            (increasing toward the tail), column 1 the current [A].

        Returns
        -------
        tau : np.ndarray
            The grid, unchanged.
        W : np.ndarray
            Wake potential on the grid [V]. Negative at (0, 0) means
            energy loss.
        """
        x = current[:, 0]
        bunch = current[:, 1]
        nb = x.shape[0]
        if self.L != 0 or self.w1 is not None:
            d1_bunch = _derivative(x, bunch)
        W = np.zeros(nb)
        if self.w0 is not None:
            _, ww = _wake_convolution(x, bunch, self.s0, self.w0)
            W = W - ww[0:nb] / c_light
        if self.w1 is not None:
            _, ww = _wake_convolution(x, d1_bunch, self.s1, self.w1)
            W = W + ww[0:nb]
        if self.R != 0:
            W = W - bunch * self.R
        if self.L != 0:
            W = W + d1_bunch * self.L * c_light
        if self.Cinv != 0:
            W = W - _cumtrapz(x, bunch) * self.Cinv / c_light
        return x, W


# -----------------------------------------------------------------------------
# TaylorWakefield
# -----------------------------------------------------------------------------


class TaylorWakefield:
    """
    Second-order Taylor-expanded 3D wakefield.

    Computes longitudinal and transverse wakefield kicks for a particle
    distribution from a table of one-dimensional wake components
    (see module docstring for the formalism). This is a port of the
    ocelot ``Wake`` physics process and reads/writes the same wake table
    file format.

    Parameters
    ----------
    components : list of TaylorWakeComponent or dict
        The wake components. At most one component per index pair (a, b).

    Examples
    --------
    ::

        # From an ocelot-format wake table file
        wake = TaylorWakefield.from_file("wake_table.dat")

        # Analytic corrugated parallel-plate (dechirper) wake
        wake = TaylorWakefield.parallel_plate(
            plate_distance=250e-6, half_gap=500e-6, length=1.0, sigma=10e-6
        )

        # Apply to a ParticleGroup
        P_out = P.apply_wakefield(wake)
    """

    def __init__(self, components):
        if isinstance(components, dict):
            components = list(components.values())
        self.components: dict[tuple[int, int], TaylorWakeComponent] = {}
        for comp in components:
            if comp.key in self.components:
                raise ValueError(f"Duplicate wake component for indices {comp.key}")
            self.components[comp.key] = comp

    def __repr__(self) -> str:
        keys = ", ".join(f"h{a}{b}" for (a, b) in sorted(self.components))
        return f"<{type(self).__name__} with components: {keys}>"

    def __contains__(self, key: tuple[int, int]) -> bool:
        return tuple(sorted(key)) in self.components

    def __getitem__(self, key: tuple[int, int]) -> TaylorWakeComponent:
        return self.components[tuple(sorted(key))]

    # -- file I/O -------------------------------------------------------------

    @classmethod
    def from_file(cls, filename) -> TaylorWakefield:
        """
        Load a wake table in the ocelot/Zagorodnov format.

        The file is a plain whitespace-separated numeric table. The first
        row gives the number of components Nt. Each component block is:
        ``[N0 N1]``, ``[R L]``, ``[Cinv ab]`` (ab encodes the Taylor index
        pair as a two-digit number), followed by N0 rows of (s, w0) and
        N1 rows of (s, w1).

        Parameters
        ----------
        filename : str or path-like
            Path to the wake table file.

        Returns
        -------
        TaylorWakefield
        """
        table = np.loadtxt(filename)
        return cls(cls._parse_table(table))

    @staticmethod
    def _parse_table(table: np.ndarray) -> list[TaylorWakeComponent]:
        """Parse a numeric wake table array into components."""
        n_components = int(table[0, 0])
        components = []
        ind = 0
        for _ in range(n_components):
            ind = ind + 1
            n0 = int(table[ind, 0])
            n1 = int(table[ind, 1])
            R = table[ind + 1, 0]
            L = table[ind + 1, 1]
            Cinv = table[ind + 2, 0]
            ab = int(table[ind + 2, 1])
            a = ab // 10
            b = ab % 10
            ind = ind + 2
            s0 = w0 = s1 = w1 = None
            if n0 > 0:
                s0 = table[ind + 1 : ind + n0 + 1, 0].copy()
                w0 = table[ind + 1 : ind + n0 + 1, 1].copy()
                ind = ind + n0
            if n1 > 0:
                s1 = table[ind + 1 : ind + n1 + 1, 0].copy()
                w1 = table[ind + 1 : ind + n1 + 1, 1].copy()
                ind = ind + n1
            components.append(
                TaylorWakeComponent(
                    a=a, b=b, s0=s0, w0=w0, s1=s1, w1=w1, R=R, L=L, Cinv=Cinv
                )
            )
        return components

    def to_file(self, filename) -> None:
        """
        Write this wakefield as an ocelot-format wake table file.

        Parameters
        ----------
        filename : str or path-like
            Output path.
        """
        blocks = [np.array([[len(self.components), 0.0]])]
        for key in sorted(self.components):
            comp = self.components[key]
            n0 = 0 if comp.w0 is None else len(comp.w0)
            n1 = 0 if comp.w1 is None else len(comp.w1)
            blocks.append(
                np.array(
                    [
                        [n0, n1],
                        [comp.R, comp.L],
                        [comp.Cinv, comp.a * 10 + comp.b],
                    ]
                )
            )
            if n0 > 0:
                blocks.append(np.column_stack([comp.s0, comp.w0]))
            if n1 > 0:
                blocks.append(np.column_stack([comp.s1, comp.w1]))
        np.savetxt(filename, np.vstack(blocks))

    # -- kick calculation -----------------------------------------------------

    def particle_kicks_3d(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        weight: np.ndarray,
        n_points: int = 500,
        filter_order: int = 20,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute 3D wakefield momentum kicks for a particle distribution.

        The particle charges are projected onto a smoothed longitudinal
        grid (together with the transverse-moment weighted "generalized
        currents"), each wake component is convolved with the appropriate
        current, and the transverse wakes are obtained through the
        Panofsky-Wenzel theorem.

        Parameters
        ----------
        x, y : np.ndarray
            Transverse particle positions [m], measured from the axis the
            wake table was computed for.
        z : np.ndarray
            Longitudinal particle positions [m]. Larger z is the bunch
            head (beamphysics convention).
        weight : np.ndarray
            Particle charges [C].
        n_points : int, optional
            Number of longitudinal grid points. Default 500.
        filter_order : int, optional
            Triangular smoothing filter order. Default 20.

        Returns
        -------
        dpx, dpy, dpz : np.ndarray
            Momentum kicks [eV/c] for each particle, for the whole
            structure represented by the wake table. dpz is negative
            for energy loss.
        """
        X = np.asarray(x, dtype=float)
        Y = np.asarray(y, dtype=float)
        q = np.asarray(weight, dtype=float)
        # Internal longitudinal coordinate: increases toward the tail
        tau = -np.asarray(z, dtype=float)

        has = self.__contains__

        X2 = X**2
        Y2 = Y**2
        XY = X * Y

        # Generalized currents
        I00 = _project_current(tau, q, n_points, filter_order)
        grid = I00[:, 0]
        n_grid = len(grid)
        I01 = I10 = I11 = I20_02 = None
        if has((0, 2)) or has((2, 3)) or has((2, 4)):
            I01 = _project_current(tau, q * Y, n_points, filter_order)
        if has((0, 1)) or has((1, 3)) or has((1, 4)):
            I10 = _project_current(tau, q * X, n_points, filter_order)
        if has((1, 2)):
            I11 = _project_current(tau, q * XY, n_points, filter_order)
        if has((1, 1)):
            I20_02 = _project_current(tau, q * (X2 - Y2), n_points, filter_order)

        def wake(key, current):
            return self[key].convolve(current)[1]

        # Longitudinal wake, monomials independent of witness position
        Wz = np.zeros(n_grid)
        if has((0, 0)):
            Wz = Wz + wake((0, 0), I00)
        if has((0, 1)):
            Wz = Wz + wake((0, 1), I10)
        if has((0, 2)):
            Wz = Wz + wake((0, 2), I01)
        if has((1, 1)):
            Wz = Wz + wake((1, 1), I20_02)
        if has((1, 2)):
            Wz = Wz + 2 * wake((1, 2), I11)
        Pz = np.interp(tau, grid, Wz, 0, 0)
        Px = np.zeros(len(X))
        Py = np.zeros(len(Y))

        h = grid[1] - grid[0]

        # Terms linear in witness y
        Wz = np.zeros(n_grid)
        Wy = np.zeros(n_grid)
        if has((0, 4)):
            w = wake((0, 4), I00)
            Wz = Wz + w
            Wy = Wy + w
        if has((1, 4)):
            w = wake((1, 4), I10)
            Wz = Wz + 2 * w
            Wy = Wy + 2 * w
        if has((2, 4)):
            w = wake((2, 4), I01)
            Wz = Wz + 2 * w
            Wy = Wy + 2 * w
        Pz = Pz + np.interp(tau, grid, Wz, 0, 0) * Y
        Wy = -_cumtrapz_uniform(h, Wy)
        Py = Py + np.interp(tau, grid, Wy, 0, 0)

        # Terms linear in witness x
        Wz = np.zeros(n_grid)
        Wx = np.zeros(n_grid)
        if has((0, 3)):
            w = wake((0, 3), I00)
            Wz = Wz + w
            Wx = Wx + w
        if has((1, 3)):
            w = wake((1, 3), I10)
            Wz = Wz + 2 * w
            Wx = Wx + 2 * w
        if has((2, 3)):
            w = wake((2, 3), I01)
            Wz = Wz + 2 * w
            Wx = Wx + 2 * w
        Wx = -_cumtrapz_uniform(h, Wx)
        Pz = Pz + np.interp(tau, grid, Wz, 0, 0) * X
        Px = Px + np.interp(tau, grid, Wx, 0, 0)

        # Witness x*y term
        if has((3, 4)):
            w = wake((3, 4), I00)
            Wx = -2 * _cumtrapz_uniform(h, w)
            p = np.interp(tau, grid, Wx, 0, 0)
            Px = Px + p * Y
            Py = Py + p * X
            Pz = Pz + 2 * np.interp(tau, grid, w, 0, 0) * XY

        # Witness x^2 - y^2 (quadrupole) term
        if has((3, 3)):
            w = wake((3, 3), I00)
            Pz = Pz + np.interp(tau, grid, w, 0, 0) * (X2 - Y2)
            Wx = -2 * _cumtrapz_uniform(h, w)
            p = np.interp(tau, grid, Wx, 0, 0)
            Px = Px + p * X
            Py = Py - p * Y

        return Px, Py, Pz

    # -- wake potentials for a current profile --------------------------------

    def wake_potential(
        self, current_profile: np.ndarray, key: tuple[int, int] = (0, 0)
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convolve a single wake component with a current profile.

        For transverse witness components ((0, 3) horizontal, (0, 4)
        vertical dipole), the Panofsky-Wenzel integral is applied so the
        returned potential is the transverse kick per unit offset [V/m].

        Parameters
        ----------
        current_profile : np.ndarray
            Array of shape (n, 2): column 0 is z [m] on a uniform grid
            (larger z is the bunch head), column 1 the current [A].
        key : tuple of int, optional
            Component indices (a, b). Default (0, 0), the longitudinal
            monopole wake.

        Returns
        -------
        z : np.ndarray
            Longitudinal positions [m], same convention as the input.
        W : np.ndarray
            Wake potential [V] (longitudinal; negative means energy loss)
            or [V/m] (transverse witness components).
        """
        profile = np.asarray(current_profile, dtype=float)
        # Convert to internal tail-positive coordinate, ascending
        tau = -profile[::-1, 0]
        current = np.column_stack([tau, profile[::-1, 1]])
        grid, W = self[key].convolve(current)
        a, b = tuple(sorted(key))
        if a == 0 and b in (3, 4):
            h = grid[1] - grid[0]
            W = -_cumtrapz_uniform(h, W)
        return -grid[::-1], W[::-1]

    # -- analytic wake tables --------------------------------------------------

    @classmethod
    def parallel_plate(
        cls,
        plate_distance: float = 500e-6,
        half_gap: float = 500e-6,
        corrugation_gap: float = 250e-6,
        corrugation_period: float = 500e-6,
        length: float = 1.0,
        sigma: float = 30e-6,
        orientation: str = "horizontal",
        decay: bool = True,
    ) -> TaylorWakefield:
        """
        Analytic wake table for a corrugated parallel-plate structure.

        Surface-impedance model of Bane, Stupakov, and Zagorodnov for a
        flat corrugated dechirper with the beam offset from the center.
        Port of ocelot's ``WakeTableParallelPlate`` (``decay=True``, first
        order: components decay as exp(-sqrt(s/s0))) and
        ``WakeTableParallelPlate_origin`` (``decay=False``, zeroth order:
        constant components).

        Parameters
        ----------
        plate_distance : float, optional
            Distance b from the beam to the nearest (+) plate [m]. The
            beam offset from the center is ``half_gap - plate_distance``.
        half_gap : float, optional
            Half gap a between the plates [m]. Requires 0 < b < 2a.
        corrugation_gap : float, optional
            Longitudinal gap t of the corrugations [m].
        corrugation_period : float, optional
            Period p of the corrugations [m].
        length : float, optional
            Length of the structure [m]. Default 1.
        sigma : float, optional
            Characteristic rms bunch length [m], used to set the tabulated
            s range (0 to 50 sigma). Default 30e-6.
        orientation : str, optional
            'horizontal' for horizontal plates (offset and kick in y) or
            'vertical' for vertical plates (offset and kick in x).
            Default 'horizontal'.
        decay : bool, optional
            Include the first-order exponential decay of the wake
            components. Default True.

        Returns
        -------
        TaylorWakefield

        References
        ----------
        K. Bane, G. Stupakov, I. Zagorodnov, Phys. Rev. Accel. Beams 19,
        084401 (2016); SLAC-PUB-16881.
        """
        a = half_gap
        b = plate_distance
        t = corrugation_gap
        p = corrugation_period
        offset = a - b
        if np.abs(offset) >= a:
            raise ValueError("plate_distance must satisfy 0 < b < 2 * half_gap")

        s = np.arange(0, 50 + 0.01, 0.01) * sigma

        t2p = t / p
        alpha = 1 - 0.465 * np.sqrt(t2p) - 0.07 * t2p
        s0r = a * a * t / (2 * np.pi * alpha * alpha * p * p)  # Bane s0r

        # cgs -> mks conversion, scaled by structure length
        mks = Z0 * c_light / (4 * np.pi) * length

        Y = np.pi * offset / (2 * a)

        def decay_term(s_scale):
            return np.exp(-np.sqrt(s / s_scale)) if decay else np.ones(s.shape)

        h02 = None
        if Y == 0:
            sl = 9 / 4 * s0r
            sd = (15 / 14) ** 2 * s0r
            sq = (15 / 16) ** 2 * s0r

            h00 = mks * np.pi**2 / (4 * a**2) * decay_term(sl)
            h11 = mks * (-1) * np.pi**4 / (64 * a**4) * decay_term(sq)
            h24 = mks * np.pi**4 / (64 * a**4) * decay_term(sd)
        else:
            sec_Y = 1 / np.cos(Y)
            csc_Y = 1 / np.sin(Y)
            cot_2Y = 1 / np.tan(2 * Y)

            sl = 4 * s0r * (1 + np.cos(Y) ** 2 / 3 + Y * np.tan(Y)) ** (-2)
            sm = 4 * s0r * (1.5 - Y * cot_2Y + Y * csc_Y * sec_Y) ** (-2)
            sd = (
                4
                * s0r
                * (
                    (64 + np.cos(2 * Y)) / 30
                    + 2 * Y * np.tan(Y)
                    + (0.3 - Y * np.sin(2 * Y)) / (np.cos(2 * Y) - 2)
                )
                ** (-2)
            )
            sq = (
                4
                * s0r
                * (
                    (56 - np.cos(2 * Y)) / 30
                    + 2 * Y * np.tan(Y)
                    - (0.3 + Y * np.sin(2 * Y)) / (np.cos(2 * Y) - 2)
                )
                ** (-2)
            )

            h00 = mks * np.pi**2 / (4 * a**2) * sec_Y**2 * decay_term(sl)
            h02 = (
                mks * np.pi**3 / (16 * a**3) * np.sin(2 * Y) * sec_Y**4 * decay_term(sm)
            )
            h11 = (
                mks
                * np.pi**4
                / (64 * a**4)
                * (np.cos(2 * Y) - 2)
                * sec_Y**4
                * decay_term(sq)
            )
            h24 = (
                mks
                * np.pi**4
                / (64 * a**4)
                * (2 - np.cos(2 * Y))
                * sec_Y**4
                * decay_term(sd)
            )
        h13 = -h11
        h33 = h11

        return cls._from_h_arrays(
            s,
            h00=h00,
            h02=h02,
            h04=h02,
            h11=h11,
            h13=h13,
            h24=h24,
            h33=h33,
            orientation=orientation,
        )

    @classmethod
    def dechirper_off_axis(
        cls,
        plate_distance: float = 500e-6,
        half_gap: float = 0.01,
        width: float = 0.02,
        corrugation_gap: float = 250e-6,
        corrugation_period: float = 500e-6,
        length: float = 1.0,
        sigma: float = 30e-6,
        orientation: str = "horizontal",
        n_modes: int = 300,
    ) -> TaylorWakefield:
        """
        Mode-sum wake table for a corrugated plate of finite width.

        Intended for a beam close to a single plate of a dechirper (large
        half gap, small plate distance). Port of ocelot's
        ``WakeTableDechirperOffAxis``.

        Parameters
        ----------
        plate_distance : float, optional
            Distance b from the beam to the plate [m]. Default 500e-6.
        half_gap : float, optional
            Half gap a between the plates [m]. Default 0.01.
        width : float, optional
            Width of the corrugated structure [m]. Default 0.02.
        corrugation_gap : float, optional
            Longitudinal gap t of the corrugations [m]. Default 250e-6.
        corrugation_period : float, optional
            Period p of the corrugations [m]. Default 500e-6.
        length : float, optional
            Length of the structure [m]. Default 1.
        sigma : float, optional
            Characteristic rms bunch length [m], used to set the tabulated
            s range (0 to 50 sigma). Default 30e-6.
        orientation : str, optional
            'horizontal' or 'vertical' plate orientation.
            Default 'horizontal'.
        n_modes : int, optional
            Number of transverse modes in the sum. Default 300.

        Returns
        -------
        TaylorWakefield

        References
        ----------
        https://doi.org/10.1016/j.nima.2016.09.001 and SLAC-PUB-16881.
        """
        # Work in mm as in the original implementation
        p = corrugation_period * 1e3
        t = corrugation_gap * 1e3
        L = length
        D = width * 1e3
        a = half_gap * 1e3
        b = plate_distance * 1e3
        sig = sigma * 1e3
        y0 = a - b  # position of the charge
        y = y0
        x0 = D / 2.0
        x = x0
        s = np.arange(0, 50 + 0.01, 0.01) * sig
        ns = len(s)

        t2p = t / p
        alpha = 1 - 0.465 * np.sqrt(t2p) - 0.07 * t2p
        s0r_bane = a * a * t / (2 * np.pi * alpha**2 * p**2)
        s0 = 4 * s0r_bane * np.pi / 4

        W = np.zeros(ns)
        dWdx0 = np.zeros(ns)
        dWdy0 = np.zeros(ns)
        dWdx = np.zeros(ns)
        dWdy = np.zeros(ns)
        ddWdx0dx0 = np.zeros(ns)
        ddWdxdx0 = np.zeros(ns)
        ddWdydy0 = np.zeros(ns)

        A = Z0 * c_light / (2 * a) * L
        for i in range(n_modes):
            m = i + 1
            M = np.pi / D * m
            X = M * a
            dx = np.sin(M * x0) * np.sin(M * x)
            # avoid overflow of cosh/sinh
            coeff = X / (np.cosh(X) * np.sinh(X)) if X < 350.0 else 0.0
            Wcc = A * coeff * np.exp(-((s / s0) ** 0.5) * (X / np.tanh(X)))
            Wss = A * coeff * np.exp(-((s / s0) ** 0.5) * (X * np.tanh(X)))

            Fz = Wcc * np.cosh(M * y) * np.cosh(M * y0) + Wss * np.sinh(
                M * y
            ) * np.sinh(M * y0)

            W = W + Fz * dx
            ddx0 = np.cos(M * x0) * np.sin(M * x)
            dWdx0 = dWdx0 + M * Fz * ddx0
            ddy0 = Wcc * np.cosh(M * y) * np.sinh(M * y0) + Wss * np.sinh(
                M * y
            ) * np.cosh(M * y0)
            dWdy0 = dWdy0 + M * ddy0 * dx
            ddx = np.sin(M * x0) * np.cos(M * x)
            dWdx = dWdx + M * Fz * ddx
            ddy = Wcc * np.sinh(M * y) * np.cosh(M * y0) + Wss * np.cosh(
                M * y
            ) * np.sinh(M * y0)
            dWdy = dWdy + M * ddy * dx
            ddWdx0dx0 = ddWdx0dx0 - M**2 * Fz * dx
            ddWdxdx0 = ddWdxdx0 + M**2 * Fz * np.cos(M * x0) * np.cos(M * x)
            ddWdydy0 = (
                ddWdydy0
                + M**2
                * (
                    Wcc * np.sinh(M * y) * np.sinh(M * y0)
                    + Wss * np.cosh(M * y) * np.cosh(M * y0)
                )
                * dx
            )

        # mm -> m unit restoration; factors 1e6, 1e9, 1e12 restore V/C/m^k
        h00 = W * 2 / D * 1e6
        h02 = dWdy0 * 2 / D * 1e9
        h04 = dWdy * 2 / D * 1e9
        h11 = ddWdx0dx0 * 2 / D * 1e12 * 0.5
        h13 = ddWdxdx0 * 2 / D * 1e12 * 0.5
        h24 = ddWdydy0 * 2 / D * 1e12 * 0.5
        h33 = h11

        s = s * 1e-3
        return cls._from_h_arrays(
            s,
            h00=h00,
            h02=h02,
            h04=h04,
            h11=h11,
            h13=h13,
            h24=h24,
            h33=h33,
            orientation=orientation,
        )

    @classmethod
    def _from_h_arrays(
        cls, s, *, h00, h02, h04, h11, h13, h24, h33, orientation
    ) -> TaylorWakefield:
        """
        Build the component set for a flat structure from its h arrays.

        The h arrays are for horizontal plates (offset and kick in y).
        For vertical plates, x and y swap roles: source/witness indices
        map 1 <-> 2 and 3 <-> 4, and the sign of the quadrupole terms
        h11/h33 flips.
        """

        def comp(a, b, w):
            return TaylorWakeComponent(a=a, b=b, s0=s, w0=w)

        if orientation in ("horizontal", "horz"):
            components = [
                comp(0, 0, h00),
                comp(1, 1, h11),
                comp(1, 3, h13),
                comp(2, 4, h24),
                comp(3, 3, h33),
            ]
            if h02 is not None:
                components += [comp(0, 2, h02), comp(0, 4, h04)]
        elif orientation in ("vertical", "vert"):
            components = [
                comp(0, 0, h00),
                comp(1, 1, -h11),
                comp(1, 3, h24),
                comp(2, 4, h13),
                comp(3, 3, -h33),
            ]
            if h02 is not None:
                components += [comp(0, 1, h02), comp(0, 3, h04)]
        else:
            raise ValueError(
                f"orientation must be 'horizontal' or 'vertical', got {orientation!r}"
            )
        return cls(components)

    # -- plotting --------------------------------------------------------------

    def plot(self, ax=None):
        """
        Plot the tabulated wake components.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates a new figure.
        """
        if ax is None:
            _, ax = plt.subplots()
        for key in sorted(self.components):
            comp = self.components[key]
            if comp.w0 is None:
                continue
            ax.plot(comp.s0 * 1e6, comp.w0, label=comp.label)
        ax.set_xlabel(r"Distance behind source $s$ (µm)")
        ax.set_ylabel(r"$h_{ab}(s)$ (V/C $\cdot$ m$^{-k}$)")
        ax.legend()
        return ax
