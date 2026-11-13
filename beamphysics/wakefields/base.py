"""
Abstract base class for wakefield models.

This module provides the abstract base class that defines the interface
for all wakefield implementations.

Classes
-------
WakefieldBase
    Abstract base class for all wakefield models
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np

from ._convolution import _WakefieldConvolution

__all__ = ["SAMPLES_PER_WAVELENGTH", "WakefieldBase"]

# Linear interpolation, as in IMPACT-Z, errs by about (2*pi/m)^2/8 at m samples per
# wavelength; 128 keeps the resistive wall wakes within 1e-03 of W0 and 5000 rows.
SAMPLES_PER_WAVELENGTH = 128

# Fewest samples accepted by the cubic interpolant of TabularWakefield.
MIN_SAMPLES = 4


class WakefieldBase(_WakefieldConvolution, ABC):
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
        Evaluate the wakefield at position z (V/C/m)
    impedance(k)
        Evaluate the impedance at wavenumber k (Ohm/m)
    integrated_wake(z)
        Integral of the wake from z up to the source particle (V/C), if the
        model defines one
    bin_averaged_wake(dz, n)
        Bin averages of the wake, as tabulated for tracking codes (V/C/m)
    convolve_density(density, dz, offset=0)
        Compute integrated wake from a charge density distribution (V/m)
    particle_kicks(z, weight, include_self_kick=True)
        Compute momentum kicks for a distribution of particles (eV/m)
    binned_particle_kicks(z, weight, dz=None, include_self_kick=True)
        The same through a charge deposit and the bin-averaged wake (eV/m)
    plot(zmax=None, zmin=0, n=200)
        Plot the wakefield over a range of z values
    plot_impedance(kmax=None, kmin=0, n=200)
        Plot the impedance over a range of wavenumbers

    See Also
    --------
    ParticleGroup.apply_wakefield : Apply wakefield kicks to a ParticleGroup
    """

    @abstractmethod
    def wake(self, z: np.ndarray | float) -> np.ndarray | float:
        """
        Evaluate the wakefield at longitudinal position z.

        Parameters
        ----------
        z : float or np.ndarray
            Longitudinal position (m). Negative z is behind the source.

        Returns
        -------
        W : float or np.ndarray
            Wakefield value (V/C/m). Returns 0 for z > 0 (causality).
        """
        pass

    @abstractmethod
    def impedance(self, k: np.ndarray | float) -> np.ndarray | complex:
        """
        Evaluate the impedance at wavenumber k.

        Parameters
        ----------
        k : float or np.ndarray
            Wavenumber (1/m).

        Returns
        -------
        Z : complex or np.ndarray
            Impedance (Ohm/m).
        """
        pass

    def __call__(self, z: np.ndarray | float) -> np.ndarray | float:
        """
        Evaluate the wakefield at position z (convenience method).

        This is equivalent to calling wake(z).
        """
        return self.wake(z)

    def integrated_wake(self, z: np.ndarray | float) -> np.ndarray | float:
        """
        Integral of the wake from z up to the source particle.

        $$F(z) = \\int_z^0 W(z') \\, dz'$$

        This is the integral of the wake over the distance s = -z behind the
        source. A delta function at the source, as in `ParametricWakefield`, is
        counted in full for every z < 0. Models that define this can be tabulated
        as bin averages, which is the only correct tabulation of a wake that is
        singular at z = 0.

        Parameters
        ----------
        z : float or np.ndarray
            Longitudinal position (m). Negative z is behind the source.

        Returns
        -------
        F : float or np.ndarray
            Integrated wake (V/C). Returns 0 for z >= 0.

        Raises
        ------
        NotImplementedError
            If the model does not implement `integrated_wake`.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement integrated_wake."
        )

    def _has_integrated_wake(self) -> bool:
        """Whether the model overrides `integrated_wake`."""
        return type(self).integrated_wake is not WakefieldBase.integrated_wake

    @property
    def is_singular(self) -> bool:
        """
        Whether the wake has no finite point value at z = 0.

        A singular wake, such as the diffraction-model or delta-function term of
        `ParametricWakefield`, cannot be sampled at points. It is tabulated and
        applied through `bin_averaged_wake` instead.
        """
        return False

    def bin_averaged_wake(self, dz: float, n: int) -> np.ndarray:
        """
        Bin averages of the wake at n uniform lags, in the convention of tracking codes.

        With F(s) = `integrated_wake(-s)` and s = k dz behind the source,

        $$K_0 = \\frac{F(\\Delta/2)}{\\Delta}, \\qquad
        K_k = \\frac{F((k + \\tfrac12)\\Delta) - F((k - \\tfrac12)\\Delta)}{\\Delta},
        \\quad k \\ge 1$$

        where Delta is dz. This is the integrated Green function of Qiang et al.
        (2006): $K_k$ is the wake at the centre of a bin from a unit charge spread
        uniformly over the bin k bins ahead, and $K_0$ is the self term from the
        half of its own bin that lies ahead. Bmad, IMPACT-T, IMPACT-Z and
        `convolve_density` all halve the first entry of a wake table for the
        self-kick, so the first entry returned is $2 K_0$. For a smooth wake it
        agrees with the point sample to second order in dz.

        Parameters
        ----------
        dz : float
            Bin width (m).
        n : int
            Number of lags, from k = 0 to k = n - 1.

        Returns
        -------
        np.ndarray
            Array of length n holding $2 K_0, K_1, \\ldots, K_{n-1}$ (V/C/m).

        Raises
        ------
        NotImplementedError
            If the model does not implement `integrated_wake`.
        """
        k = np.arange(n)
        # F(z) = 0 for z >= 0, so the lower edge of the k = 0 bin contributes nothing.
        F_upper = self.integrated_wake(-(k + 0.5) * dz)
        F_lower = self.integrated_wake(-(k - 0.5) * dz)
        K = (np.asarray(F_upper, dtype=float) - np.asarray(F_lower, dtype=float)) / dz
        K[0] *= 2

        return K

    @property
    def default_zmax(self) -> float:
        """
        Distance behind the source particle beyond which the wake is negligible (m).

        Used as the default range when tabulating.

        Raises
        ------
        NotImplementedError
            If the model defines no default; pass zmax explicitly.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not define a natural tabulation range. "
            "Pass zmax explicitly."
        )

    @property
    def min_wavelength(self) -> float:
        """
        Shortest length scale of the wake (m), which sets the default sampling.

        Raises
        ------
        NotImplementedError
            If the model defines no length scale; pass n explicitly.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not define a shortest length scale. "
            "Pass n explicitly."
        )

    def default_n_samples(self, zmax: float) -> int:
        """
        Default number of uniform samples out to zmax.

        $$n = \\max\\left( \\left\\lceil \\frac{128 \\, z_{\\max}}{\\lambda_{\\min}}
        \\right\\rceil + 1, \\; 4 \\right)$$

        Here lambda_min is `min_wavelength`, and 4 is the minimum for cubic
        interpolation.

        Parameters
        ----------
        zmax : float
            Largest distance behind the source particle (m).

        Returns
        -------
        int
        """
        n = int(np.ceil(SAMPLES_PER_WAVELENGTH * zmax / self.min_wavelength)) + 1

        return max(n, MIN_SAMPLES)

    def _wake_kernel(self, dz: float, n: int, offset: float, method: str) -> np.ndarray:
        """Wake at lags k dz for k = 0 to n - 1, as used by `convolve_density`."""
        if method == "auto":
            if offset == 0 and self._has_integrated_wake():
                method = "bin_average"
            else:
                method = "sample"

        if method == "bin_average":
            if offset != 0:
                raise ValueError("method='bin_average' does not support an offset")
            return self.bin_averaged_wake(dz, n)
        if method == "sample":
            if self.is_singular:
                raise ValueError(
                    f"{type(self).__name__} is singular at z = 0 and cannot be "
                    "sampled at points. Use method='bin_average'."
                )
            return np.asarray(self.wake(-np.arange(n) * dz + offset), dtype=float)
        raise ValueError(
            f"method must be 'auto', 'sample' or 'bin_average', got {method!r}"
        )

    def particle_kicks(
        self,
        z: np.ndarray,
        weight: np.ndarray,
        include_self_kick: bool = True,
    ) -> np.ndarray:
        """
        Compute wakefield-induced energy kicks per unit length.

        Smooth wakes use a default O(N²) point-to-point sum. Singular wakes use
        `binned_particle_kicks`; call that method to choose the bin width.
        Subclasses may override with more efficient algorithms.

        Parameters
        ----------
        z : np.ndarray
            Particle positions (m).
        weight : np.ndarray
            Particle charges (C).
        include_self_kick : bool, optional
            Whether to include the self-kick term. Default is True.

        Returns
        -------
        kicks : np.ndarray
            Energy kick per unit length at each particle (eV/m).

        """
        if self.is_singular:
            return self.binned_particle_kicks(
                z, weight, include_self_kick=include_self_kick
            )

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

    def plot(self, zmax: float = None, zmin: float = 0, n: int = 200, ax=None):
        """
        Plot the wakefield over a range of z values.

        Parameters
        ----------
        zmax : float, optional
            Maximum trailing distance (m). If None, uses a sensible default.
        zmin : float, optional
            Minimum trailing distance (m). Default is 0.
        n : int, optional
            Number of points. Default is 200.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates a new figure.
        """
        if zmax is None:
            zmax = 1e-3  # 1 mm default

        zlist = np.linspace(zmin, zmax, n)
        Wz = self.wake(-zlist)  # Negative z for trailing particles

        if ax is None:
            _, ax = plt.subplots()
        ax.plot(zlist * 1e6, Wz * 1e-12)
        ax.set_xlabel(r"Distance behind source $|z|$ (µm)")
        ax.set_ylabel(r"$W(z)$ (V/pC/m)")

    def plot_impedance(
        self, kmax: float = None, kmin: float = 0, n: int = 200, ax=None
    ):
        """
        Plot the impedance over a range of wavenumbers.

        Parameters
        ----------
        kmax : float, optional
            Maximum wavenumber (1/m). If None, uses a sensible default.
        kmin : float, optional
            Minimum wavenumber (1/m). Default is 0.
        n : int, optional
            Number of points. Default is 200.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates a new figure.
        """
        if kmax is None:
            kmax = 1e6  # 1/µm default

        k = np.linspace(kmin, kmax, n)
        Z = self.impedance(k)

        if ax is None:
            _, ax = plt.subplots()
        ax.plot(k * 1e-3, np.real(Z), label=r"Re[$Z$]")
        ax.plot(k * 1e-3, np.imag(Z), label=r"Im[$Z$]")
        ax.set_xlabel(r"$k$ (1/mm)")
        ax.set_ylabel(r"$Z(k)$ (Ω/m)")
        ax.legend()
