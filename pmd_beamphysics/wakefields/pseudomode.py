"""
Pseudomode wakefield representation.

This module provides the pseudomode wakefield model, which represents
wakefields as a sum of damped sinusoidal modes.

Classes
-------
Pseudomode
    Single pseudomode parameters (amplitude, decay, wavenumber, phase)
PseudomodeWakefield
    Wakefield represented as a sum of pseudomodes
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from ..units import c_light
from .base import WakefieldBase

__all__ = ["Pseudomode", "PseudomodeWakefield"]


@dataclass
class Pseudomode:
    """
    Single pseudomode parameters for a damped sinusoidal wakefield component.

    Represents one term in the sum: W(z) = A·exp(d·z)·sin(k·z + φ)

    Parameters
    ----------
    A : float
        Amplitude coefficient [V/C/m].
    d : float
        Exponential decay rate [1/m].
    k : float
        Oscillation wavenumber [1/m].
    phi : float
        Phase offset [rad].
    """

    A: float
    d: float
    k: float
    phi: float

    def __call__(self, z: np.ndarray) -> np.ndarray:
        """Evaluate this mode at position z (array input assumed)."""
        return self.A * np.exp(self.d * z) * np.sin(self.k * z + self.phi)

    def impedance(self, k: np.ndarray | float) -> np.ndarray | complex:
        """
        Compute the impedance Z(k) for this pseudomode analytically.

        For a pseudomode wakefield defined for z ≤ 0:

        $$W(z) = A \\cdot e^{dz} \\cdot \\sin(k_0 z + \\phi)$$

        The impedance is defined as:

        $$Z(k) = \\frac{1}{c} \\int_{-\\infty}^{0} W(z) \\cdot e^{-ikz} \\, dz$$

        **Derivation:**

        Substituting the wakefield:

        $$Z(k) = \\frac{A}{c} \\int_{-\\infty}^{0} e^{dz} \\sin(k_0 z + \\phi) e^{-ikz} \\, dz$$

        Using $\\sin(\\theta) = \\frac{e^{i\\theta} - e^{-i\\theta}}{2i}$:

        $$Z(k) = \\frac{A}{2ic} \\left[
            e^{i\\phi} \\int_{-\\infty}^{0} e^{(d + i(k_0 - k))z} dz
            - e^{-i\\phi} \\int_{-\\infty}^{0} e^{(d - i(k_0 + k))z} dz
        \\right]$$

        Since $d > 0$, both integrals converge:

        $$\\int_{-\\infty}^{0} e^{az} dz = \\frac{1}{a} \\quad \\text{for } \\text{Re}(a) > 0$$

        Evaluating:

        $$Z(k) = \\frac{A}{2ic} \\left[
            \\frac{e^{i\\phi}}{d + i(k_0 - k)}
            - \\frac{e^{-i\\phi}}{d - i(k_0 + k)}
        \\right]$$

        **Mathematica verification:**

        ```mathematica
        (* Define the wakefield and compute impedance numerically *)
        W[z_, A_, d_, k0_, phi_] := A Exp[d z] Sin[k0 z + phi]

        (* Analytical result *)
        Zanalytic[k_, A_, d_, k0_, phi_] :=
            A/(2 I c) (Exp[I phi]/(d + I (k0 - k)) - Exp[-I phi]/(d - I (k0 + k)))

        (* Numerical integration (should match) *)
        Znumeric[k_, A_, d_, k0_, phi_] :=
            1/c NIntegrate[W[z, A, d, k0, phi] Exp[-I k z], {z, -Infinity, 0}]

        (* Test with sample values *)
        c = 299792458;
        {A, d, k0, phi} = {1*^15, 1*^4, 1*^5, Pi/4};
        ktest = 5*^4;
        {Zanalytic[ktest, A, d, k0, phi], Znumeric[ktest, A, d, k0, phi]}
        (* Both should give the same complex number *)
        ```

        Parameters
        ----------
        k : float or np.ndarray
            Wavenumber [1/m].

        Returns
        -------
        Z : complex or np.ndarray
            Impedance [Ohm/m].
        """
        k = np.asarray(k)
        scalar_input = k.ndim == 0
        k = np.atleast_1d(k)

        d = self.d
        k0 = self.k
        A = self.A
        phi = self.phi

        # Z(k) = A/(2ic) * [e^{iφ}/(d + i(k₀-k)) - e^{-iφ}/(d - i(k₀+k))]
        denom1 = d + 1j * (k0 - k)
        denom2 = d - 1j * (k0 + k)

        Z = (A / (2j * c_light)) * (
            np.exp(1j * phi) / denom1 - np.exp(-1j * phi) / denom2
        )

        if scalar_input:
            return complex(Z[0])
        return Z

    def to_bmad(
        self,
        type: str = "longitudinal",
        transverse_dependence: str = "none",
    ) -> str:
        """Format as Bmad-compatible string."""
        return (
            f"{type} = {{{self.A}, {self.d}, {self.k}, "
            f"{self.phi / (2 * np.pi)}, {transverse_dependence}}}"
        )


class PseudomodeWakefield(WakefieldBase):
    """
    Wakefield represented as a sum of damped sinusoidal pseudomodes.

    Models the longitudinal wakefield as:

    $$W(z) = \\sum_i A_i \\cdot e^{d_i \\cdot z} \\cdot \\sin(k_i \\cdot z + \\phi_i)$$

    This form is used to approximate short-range wakefields such as the
    resistive wall wake.

    Parameters
    ----------
    modes : list of Pseudomode or list of dict
        List of pseudomodes. Each can be a Pseudomode object or a dict
        with keys 'A', 'd', 'k', 'phi'.

    Notes
    -----
    - Wakefields are defined for z ≤ 0 (trailing the source particle).
    - For z > 0, returns 0 (causality).

    Examples
    --------
    Single mode::

        pm = PseudomodeWakefield([Pseudomode(A=1e15, d=1e4, k=1e5, phi=np.pi/2)])
        pm.wake(-10e-6)  # Wake at 10 µm behind source

    Multiple modes::

        modes = [
            Pseudomode(A=1e15, d=1e4, k=1e5, phi=np.pi/2),
            Pseudomode(A=5e14, d=2e4, k=2e5, phi=np.pi/4),
        ]
        pm = PseudomodeWakefield(modes)
    """

    def __init__(self, modes: list) -> None:
        self._modes = []
        for mode in modes:
            if isinstance(mode, Pseudomode):
                self._modes.append(mode)
            elif isinstance(mode, dict):
                self._modes.append(Pseudomode(**mode))
            else:
                raise TypeError(f"Mode must be Pseudomode or dict, got {type(mode)}")

    @property
    def modes(self) -> list:
        """List of Pseudomode objects."""
        return self._modes

    @property
    def n_modes(self) -> int:
        """Number of pseudomodes."""
        return len(self._modes)

    def wake(self, z: np.ndarray | float) -> np.ndarray | float:
        """
        Evaluate the pseudomode wakefield at position z.

        Parameters
        ----------
        z : float or np.ndarray
            Longitudinal position [m]. Negative z is behind the source.

        Returns
        -------
        W : float or np.ndarray
            Wakefield value [V/C/m]. Returns 0 for z > 0.
        """
        z = np.asarray(z)
        scalar_input = z.ndim == 0
        z = np.atleast_1d(z)

        out = np.zeros_like(z, dtype=float)
        mask = z <= 0
        z_masked = z[mask]

        # Sum contributions from all modes
        for mode in self._modes:
            out[mask] += mode(z_masked)

        if scalar_input:
            return float(out[0])
        return out

    def impedance(self, k: np.ndarray | float) -> np.ndarray | complex:
        """
        Compute the impedance Z(k) analytically.

        Sums the contribution from each pseudomode using the analytic
        Fourier transform.

        Parameters
        ----------
        k : float or np.ndarray
            Wavenumber [1/m].

        Returns
        -------
        Z : complex or np.ndarray
            Impedance [Ohm/m].
        """
        k = np.asarray(k)
        scalar_input = k.ndim == 0
        k = np.atleast_1d(k)

        Z = np.zeros(len(k), dtype=complex)
        for mode in self._modes:
            Z += mode.impedance(k)

        if scalar_input:
            return complex(Z[0])
        return Z

    def _self_kick_value(self) -> float:
        """Return W(0⁻) = sum of A_i * sin(phi_i)."""
        return sum(mode.A * np.sin(mode.phi) for mode in self._modes)

    def particle_kicks(
        self,
        z: np.ndarray,
        weight: np.ndarray,
        include_self_kick: bool = True,
    ) -> np.ndarray:
        """
        Compute short-range wakefield energy kicks per unit length.

        Uses an O(N) single-pass algorithm for each mode by exploiting
        the exponential form of the pseudomode wakefield.

        Parameters
        ----------
        z : np.ndarray
            Particle positions [m].
        weight : np.ndarray
            Particle charges [C].
        include_self_kick : bool, optional
            If True, applies the self-kick. Default is True.

        Returns
        -------
        kicks : np.ndarray
            Wake-induced energy kick per unit length [eV/m].
        """
        z = np.asarray(z)
        weight = np.asarray(weight)

        if z.shape != weight.shape:
            raise ValueError(
                f"Mismatched shapes: z.shape={z.shape}, weight.shape={weight.shape}"
            )
        if z.ndim != 1:
            raise ValueError("z and weight must be 1D arrays")

        # Sort particles from tail to head
        ix = z.argsort()
        z_sorted = z[ix].copy()
        z_sorted -= z_sorted.max()  # Offset for numerical stability
        weight_sorted = weight[ix]

        n = len(z)
        delta_E = np.zeros(n)

        # Process each mode with O(N) algorithm
        for mode in self._modes:
            s = mode.d + 1j * mode.k
            c = mode.A * np.exp(1j * mode.phi)

            # O(N) accumulator for this mode
            b = 0.0 + 0.0j

            # Iterate from tail to head
            for i in range(n - 1, -1, -1):
                zi = z_sorted[i]
                qi = weight_sorted[i]

                # Kick from trailing particles
                delta_E[i] -= np.imag(c * np.exp(s * zi) * b)

                # Add this particle to accumulator
                b += qi * np.exp(-s * zi)

            if include_self_kick:
                delta_E -= 0.5 * mode.A * weight_sorted * np.sin(mode.phi)

        # Restore original order
        kicks = np.empty_like(delta_E)
        kicks[ix] = delta_E

        return kicks

    def to_bmad(
        self,
        type: str = "longitudinal",
        transverse_dependence: str = "none",
    ) -> str:
        """
        Format pseudomode parameters as Bmad-compatible strings.

        Parameters
        ----------
        type : str, optional
            Wake type. Default is "longitudinal".
        transverse_dependence : str, optional
            Transverse dependence. Default is "none".

        Returns
        -------
        str
            Bmad-formatted string (one line per mode).
        """
        lines = [mode.to_bmad(type, transverse_dependence) for mode in self._modes]
        return "\n".join(lines)

    def plot(self, zmax: float = 0.001, zmin: float = 0, n: int = 200, ax=None):
        """Plot the pseudomode wakefield."""
        zlist = np.linspace(zmin, zmax, n)
        Wz = self.wake(-zlist)

        if ax is None:
            _, ax = plt.subplots()
        ax.plot(zlist * 1e6, Wz * 1e-12)
        ax.set_xlabel(r"Distance behind source $|z|$ (µm)")
        ax.set_ylabel(r"$W(z)$ (V/pC/m)")
        title = f"Pseudomode Wakefield ({self.n_modes} mode"
        if self.n_modes > 1:
            title += "s"
        title += ")"
        ax.set_title(title)

    def __repr__(self) -> str:
        if self.n_modes == 1:
            m = self._modes[0]
            return f"PseudomodeWakefield(A={m.A}, d={m.d}, k={m.k}, phi={m.phi})"
        return f"PseudomodeWakefield(modes=[{self.n_modes} modes])"
