"""
Abstract base classes and general wakefield implementations.

This module provides:
- WakefieldBase: Abstract base class defining the wakefield interface
- PseudomodeWakefield: Damped sinusoidal wakefield model
- TabularWakefield: Interpolation-based wakefield from user-supplied data
- ImpedanceWakefield: FFT-based convolution using impedance Z(k)

Classes
-------
WakefieldBase
    Abstract base class for all wakefield models
PseudomodeWakefield
    Single-mode analytic representation (damped sinusoid)
TabularWakefield
    User-supplied tabular wakefield with interpolation
ImpedanceWakefield
    Wakefield defined through its impedance Z(k)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.interpolate import interp1d

from ..units import c_light

__all__ = [
    "WakefieldBase",
    "Pseudomode",
    "PseudomodeWakefield",
    "TabularWakefield",
    "ImpedanceWakefield",
]


class WakefieldBase(ABC):
    """
    Abstract base class for longitudinal wakefield models.

    All wakefield implementations should inherit from this class and
    implement the required abstract methods.

    Sign Convention
    ---------------
    The wakefield W(z) is defined for a trailing particle at position z
    relative to the source particle at z=0:

    - z < 0: Behind the source (trailing particle feels the wake)
    - z > 0: Ahead of the source (causality requires W=0)

    The wake is positive for an energy-losing interaction (the trailing
    particle loses energy).

    Methods
    -------
    wake(z)
        Evaluate the wakefield at position z [V/C/m]
    impedance(k)
        Evaluate the impedance at wavenumber k [Ohm/m]
    convolve_density(density, dz, offset=0)
        Compute integrated wake from a charge density distribution [V/m]
    particle_kicks(z, weight, include_self_kick=True)
        Compute momentum kicks for a distribution of particles [eV/m]
    apply_to_particles(particle_group, length, inplace=False, include_self_kick=True)
        Apply wakefield kicks to a ParticleGroup
    plot(zmax=None, zmin=0, n=200)
        Plot the wakefield over a range of z values
    """

    @abstractmethod
    def wake(self, z: np.ndarray | float) -> np.ndarray | float:
        """
        Evaluate the wakefield at longitudinal position z.

        Parameters
        ----------
        z : float or np.ndarray
            Longitudinal position [m]. Negative z is behind the source.

        Returns
        -------
        W : float or np.ndarray
            Wakefield value [V/C/m]. Returns 0 for z > 0 (causality).
        """
        pass

    @abstractmethod
    def impedance(self, k: np.ndarray | float) -> np.ndarray | complex:
        """
        Evaluate the impedance at wavenumber k.

        Parameters
        ----------
        k : float or np.ndarray
            Wavenumber [1/m].

        Returns
        -------
        Z : complex or np.ndarray
            Impedance [Ohm/m].
        """
        pass

    def __call__(self, z: np.ndarray | float) -> np.ndarray | float:
        """
        Evaluate the wakefield at position z (convenience method).

        This is equivalent to calling wake(z).
        """
        return self.wake(z)

    def convolve_density(
        self,
        density: np.ndarray,
        dz: float,
        offset: float = 0,
        include_self_kick: bool = True,
    ) -> np.ndarray:
        """
        Compute integrated wakefield from a charge density distribution.

        This computes the causal wake potential:

        $$V(z_i) = \\sum_{j>i} Q_j \\cdot W(z_j - z_i) + \\frac{1}{2} Q_i \\cdot W(0)$$

        where only particles ahead (larger z index) contribute to the wake
        felt by each particle. This is mathematically a correlation, not
        a convolution, and is computed efficiently using FFT.

        Parameters
        ----------
        density : np.ndarray
            Charge density array [C/m]. Index 0 is the tail (back),
            index n-1 is the head (front).
        dz : float
            Grid spacing [m].
        offset : float, optional
            Offset for the z coordinate [m]. Default is 0.
        include_self_kick : bool, optional
            Whether to include the half self-kick term. Default is True.

        Returns
        -------
        integrated_wake : np.ndarray
            Integrated longitudinal wakefield [V].
        """
        from scipy.signal import correlate

        n = len(density)
        charge = density * dz  # Convert to charge per bin [C]

        # Build causal wake array: W[k] = wake at lag k*dz
        # W[0] = W(0), W[1] = W(-dz), W[2] = W(-2*dz), ...
        z_wake = -np.arange(n) * dz + offset
        W_causal = self.wake(z_wake)

        # Correlation: result[i] = sum_k charge[i+k] * W[k]
        # This gives the wake from particles ahead (larger indices)
        corr_result = correlate(charge, W_causal, mode="full")

        # Extract the portion corresponding to non-negative lags
        # correlate output has length 2*n-1, with zero lag at index n-1
        integrated_wake = corr_result[n - 1 : 2 * n - 1]

        # Correlation includes full W[0] contribution from self
        # Remove it, then add half if self-kick is requested
        W0 = W_causal[0]
        integrated_wake = integrated_wake - charge * W0
        if include_self_kick:
            integrated_wake = integrated_wake + 0.5 * W0 * charge

        return integrated_wake

    def particle_kicks(
        self,
        z: np.ndarray,
        weight: np.ndarray,
        include_self_kick: bool = True,
    ) -> np.ndarray:
        """
        Compute wakefield-induced energy kicks per unit length.

        This is a default O(N²) implementation. Subclasses may override
        with more efficient algorithms.

        Parameters
        ----------
        z : np.ndarray
            Particle positions [m].
        weight : np.ndarray
            Particle charges [C].
        include_self_kick : bool, optional
            Whether to include the self-kick term. Default is True.

        Returns
        -------
        kicks : np.ndarray
            Energy kick per unit length at each particle [eV/m].
        """
        z = np.asarray(z)
        weight = np.asarray(weight)
        n = len(z)
        kicks = np.zeros(n)

        # O(N²) algorithm: sum contribution from all particles
        for i in range(n):
            for j in range(n):
                if i == j:
                    if include_self_kick:
                        # Self-kick uses W(0⁻) / 2
                        kicks[i] -= 0.5 * weight[j] * self._self_kick_value()
                else:
                    dz = z[i] - z[j]
                    if dz < 0:  # j is ahead of i, so i feels wake from j
                        kicks[i] -= weight[j] * self.wake(dz)

        return kicks

    def _self_kick_value(self) -> float:
        """
        Return the self-kick value W(0⁻).

        Subclasses should override this for efficiency.
        """
        return self.wake(-1e-15)

    def apply_to_particles(
        self,
        particle_group,
        length: float,
        inplace: bool = False,
        include_self_kick: bool = True,
    ):
        """
        Apply wakefield momentum kicks to a ParticleGroup.

        Parameters
        ----------
        particle_group : ParticleGroup
            The particle group to apply the wakefield to.
        length : float
            Length over which the wakefield acts [m].
        inplace : bool, optional
            If True, modifies in place. If False, returns a modified copy.
        include_self_kick : bool, optional
            Whether to include the self-kick term. Default is True.

        Returns
        -------
        ParticleGroup or None
            Modified ParticleGroup if inplace=False, otherwise None.
        """
        if not inplace:
            particle_group = particle_group.copy()

        if particle_group.in_t_coordinates:
            z = np.asarray(particle_group.z)
        else:
            z = -c_light * np.asarray(particle_group.t)

        weight = np.asarray(particle_group.weight)
        kicks = self.particle_kicks(z, weight, include_self_kick=include_self_kick)
        particle_group.pz += kicks * length

        if not inplace:
            return particle_group

    def plot(self, zmax: float = None, zmin: float = 0, n: int = 200):
        """
        Plot the wakefield over a range of z values.

        Parameters
        ----------
        zmax : float, optional
            Maximum trailing distance [m]. If None, uses a sensible default.
        zmin : float, optional
            Minimum trailing distance [m]. Default is 0.
        n : int, optional
            Number of points. Default is 200.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib figure.
        """
        import matplotlib.pyplot as plt

        if zmax is None:
            zmax = 1e-3  # 1 mm default

        zlist = np.linspace(zmin, zmax, n)
        Wz = self.wake(-zlist)  # Negative z for trailing particles

        fig, ax = plt.subplots()
        ax.plot(zlist * 1e6, Wz * 1e-12)
        ax.set_xlabel(r"Distance behind source $|z|$ (µm)")
        ax.set_ylabel(r"$W(z)$ (V/pC/m)")
        ax.set_title(f"{self.__class__.__name__}")

        return fig


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
    Single mode:

    >>> pm = PseudomodeWakefield([Pseudomode(A=1e15, d=1e4, k=1e5, phi=np.pi/2)])
    >>> pm.wake(-10e-6)  # Wake at 10 µm behind source

    Multiple modes:

    >>> modes = [
    ...     Pseudomode(A=1e15, d=1e4, k=1e5, phi=np.pi/2),
    ...     Pseudomode(A=5e14, d=2e4, k=2e5, phi=np.pi/4),
    ... ]
    >>> pm = PseudomodeWakefield(modes)
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

    def plot(self, zmax: float = 0.001, zmin: float = 0, n: int = 200):
        """Plot the pseudomode wakefield."""
        import matplotlib.pyplot as plt

        zlist = np.linspace(zmin, zmax, n)
        Wz = self.wake(-zlist)

        fig, ax = plt.subplots()
        ax.plot(zlist * 1e6, Wz * 1e-12)
        ax.set_xlabel(r"Distance behind source $|z|$ (µm)")
        ax.set_ylabel(r"$W(z)$ (V/pC/m)")
        title = f"Pseudomode Wakefield ({self.n_modes} mode"
        if self.n_modes > 1:
            title += "s"
        title += ")"
        ax.set_title(title)

        return fig

    def __repr__(self) -> str:
        if self.n_modes == 1:
            m = self._modes[0]
            return f"PseudomodeWakefield(A={m.A}, d={m.d}, k={m.k}, phi={m.phi})"
        return f"PseudomodeWakefield(modes=[{self.n_modes} modes])"


class TabularWakefield(WakefieldBase):
    """
    Wakefield defined by user-supplied tabular data with interpolation.

    Uses cubic spline interpolation to evaluate the wakefield at
    arbitrary positions between the supplied data points.

    Parameters
    ----------
    z : np.ndarray
        Longitudinal positions [m]. Should be negative (behind source)
        and sorted in ascending order.
    W : np.ndarray
        Wakefield values [V/C/m] at each z position.
    fill_value : float, optional
        Value to return outside the interpolation range. Default is 0.
    kind : str, optional
        Interpolation method. Default is 'cubic'.

    Examples
    --------
    >>> z_data = -np.linspace(1e-6, 1e-3, 100)
    >>> W_data = 1e15 * np.exp(z_data / 100e-6) * np.sin(1e5 * z_data)
    >>> wake = TabularWakefield(z_data, W_data)
    >>> wake.wake(-50e-6)  # Interpolated wake at 50 µm behind source
    """

    def __init__(
        self,
        z: np.ndarray,
        W: np.ndarray,
        fill_value: float = 0.0,
        kind: str = "cubic",
    ) -> None:
        z = np.asarray(z)
        W = np.asarray(W)

        if z.shape != W.shape:
            raise ValueError(f"Shape mismatch: z.shape={z.shape}, W.shape={W.shape}")
        if len(z) < 4:
            raise ValueError("Need at least 4 points for cubic interpolation")

        # Store data
        self._z = z
        self._W = W
        self._fill_value = fill_value

        # Create interpolator
        self._interp = interp1d(
            z,
            W,
            kind=kind,
            bounds_error=False,
            fill_value=fill_value,
        )

    def wake(self, z: np.ndarray | float) -> np.ndarray | float:
        """
        Evaluate the wakefield at position z using interpolation.

        Parameters
        ----------
        z : float or np.ndarray
            Longitudinal position [m].

        Returns
        -------
        W : float or np.ndarray
            Interpolated wakefield value [V/C/m]. Returns fill_value
            outside the data range, and 0 for z > 0 (causality).
        """
        z = np.asarray(z)
        scalar_input = z.ndim == 0
        z = np.atleast_1d(z)

        # Apply causality
        result = np.where(z > 0, 0.0, self._interp(z))

        if scalar_input:
            return float(result[0])
        return result

    def impedance(self, k: np.ndarray | float) -> np.ndarray | complex:
        """
        Compute the impedance Z(k) via numerical FFT.

        Uses FFT to compute the Fourier transform of the tabular wake data.

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

        # Use the stored data for FFT
        z_data = self._z
        W_data = self._W

        # Sort by z (ascending)
        sort_idx = np.argsort(z_data)
        z_sorted = z_data[sort_idx]
        W_sorted = W_data[sort_idx]

        # Z(k) = (1/c) * integral of W(z) * exp(-ikz) dz
        # For each k, compute numerical integral
        Z = np.zeros(len(k), dtype=complex)
        for i, ki in enumerate(k):
            integrand = W_sorted * np.exp(-1j * ki * z_sorted)
            Z[i] = np.trapezoid(integrand, z_sorted) / c_light

        if scalar_input:
            return complex(Z[0])
        return Z

    def _self_kick_value(self) -> float:
        """Return W(0⁻) by extrapolation."""
        # Use the closest point to z=0
        idx = np.argmax(self._z)
        return float(self._W[idx])

    @property
    def z_data(self) -> np.ndarray:
        """Return the z data points."""
        return self._z.copy()

    @property
    def W_data(self) -> np.ndarray:
        """Return the W data points."""
        return self._W.copy()


class ImpedanceWakefield(WakefieldBase):
    """
    Wakefield defined through its longitudinal impedance Z(k).

    Uses FFT-based convolution in frequency domain for efficient
    wake potential calculations.

    Parameters
    ----------
    impedance_func : callable
        Function that returns complex impedance Z(k) [Ohm/m] for
        wave number k [1/m]. Should handle arrays.
    wakefield_func : callable, optional
        Function that returns wakefield W(z) [V/C/m] for position z [m].
        If not provided, uses numerical cosine transform of impedance.
    k_max : float, optional
        Maximum wavenumber for integration [1/m]. Default is 1e7.

    Notes
    -----
    The wakefield W(z) is defined for z ≤ 0 (trailing particles), with W(z) = 0
    for z > 0 (causality). The impedance Z(k) is related to the wakefield by:

    $$Z(k) = \\frac{1}{c} \\int_{-\\infty}^{0} W(z) e^{-ikz} dz$$

    And the inverse:

    $$W(z) = \\frac{2c}{\\pi} \\int_0^{\\infty} \\text{Re}[Z(k)] \\cos(kz) dk \\quad (z \\le 0)$$

    Examples
    --------
    >>> def my_impedance(k):
    ...     return 100 / (1 + 1j * k * 1e-6)  # Simple resonator
    >>> wake = ImpedanceWakefield(my_impedance)
    """

    def __init__(
        self,
        impedance_func: Callable[[np.ndarray], np.ndarray],
        wakefield_func: Callable[[np.ndarray], np.ndarray] = None,
        k_max: float = 1e7,
    ) -> None:
        self._impedance_func = impedance_func
        self._wakefield_func = wakefield_func
        self._k_max = k_max

    def impedance(self, k: np.ndarray | float) -> np.ndarray | complex:
        """
        Evaluate the impedance at wavenumber k.

        Parameters
        ----------
        k : float or np.ndarray
            Wavenumber [1/m].

        Returns
        -------
        Z : complex or np.ndarray
            Impedance [Ohm/m].
        """
        return self._impedance_func(k)

    def wake(self, z: np.ndarray | float) -> np.ndarray | float:
        """
        Evaluate the wakefield at position z.

        If a wakefield function was provided, uses it directly.
        Otherwise, computes via numerical cosine transform of Re[Z(k)].

        Parameters
        ----------
        z : float or np.ndarray
            Longitudinal position [m].

        Returns
        -------
        W : float or np.ndarray
            Wakefield value [V/C/m]. Returns 0 for z > 0.
        """
        if self._wakefield_func is not None:
            return self._wakefield_func(z)

        # Numerical cosine transform
        from scipy.integrate import quad

        z = np.asarray(z)
        scalar_input = z.ndim == 0
        z = np.atleast_1d(z)

        def _compute_single(z_val):
            if z_val > 0:
                return 0.0

            def integrand(k):
                return np.real(self._impedance_func(k)) * np.cos(k * z_val)

            result, _ = quad(integrand, 0, self._k_max, limit=200)
            return (2 * c_light / np.pi) * result

        result = np.array([_compute_single(zi) for zi in z])

        if scalar_input:
            return float(result[0])
        return result

    # convolve_density is inherited from WakefieldBase

    def particle_kicks(
        self,
        z: np.ndarray,
        weight: np.ndarray,
        include_self_kick: bool = True,
        n_bins: int = None,
    ) -> np.ndarray:
        """
        Compute wakefield-induced energy kicks per unit length.

        Uses FFT-based density convolution with interpolation back to
        particle positions. This is O(N + M log M) where N is the number
        of particles and M is the number of grid points.

        Parameters
        ----------
        z : np.ndarray
            Particle positions [m].
        weight : np.ndarray
            Particle charges [C].
        include_self_kick : bool, optional
            Whether to include the self-kick term. Default is True.
        n_bins : int, optional
            Number of bins for the density grid. Default is max(100, N//10).

        Returns
        -------
        kicks : np.ndarray
            Energy kick per unit length at each particle [eV/m].
        """
        from scipy.interpolate import interp1d

        z = np.asarray(z)
        weight = np.asarray(weight)
        n = len(z)

        if n_bins is None:
            n_bins = max(100, n // 10)

        # Bin particles into density distribution
        z_min, z_max = z.min(), z.max()
        z_range = z_max - z_min
        if z_range == 0:
            z_range = 1e-6  # Avoid division by zero for single particle

        # Add padding to avoid edge effects
        padding = 0.1 * z_range
        z_min_padded = z_min - padding
        z_max_padded = z_max + padding

        dz = (z_max_padded - z_min_padded) / n_bins
        z_grid = np.linspace(z_min_padded + dz / 2, z_max_padded - dz / 2, n_bins)

        # Create density histogram (charge per bin)
        density, bin_edges = np.histogram(
            z, bins=n_bins, range=(z_min_padded, z_max_padded), weights=weight
        )
        # Convert to charge density [C/m]
        density = density / dz

        # Compute wake potential via FFT convolution
        wake_potential = self.convolve_density(
            density, dz, include_self_kick=include_self_kick
        )

        # Interpolate wake potential to particle positions
        interp = interp1d(
            z_grid, wake_potential, kind="linear", bounds_error=False, fill_value=0.0
        )
        kicks = -interp(z)  # Negative sign: energy loss for trailing particles

        return kicks

    @property
    def k_max(self) -> float:
        """Maximum wavenumber for integration."""
        return self._k_max

    @k_max.setter
    def k_max(self, value: float) -> None:
        self._k_max = value
