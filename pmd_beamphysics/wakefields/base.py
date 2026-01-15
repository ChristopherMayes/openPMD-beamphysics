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

from ..units import c_light

__all__ = ["WakefieldBase"]


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
        plot: bool = False,
        ax=None,
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
        plot : bool, optional
            If True, plot the density profile and wake potential. Default is False.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None and plot=True, creates a new figure.

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

        if plot:
            self._plot_convolve_density(density, dz, integrated_wake, ax=ax)

        return integrated_wake

    def _plot_convolve_density(self, density, dz, integrated_wake, ax=None):
        """Plot density profile and wake potential from convolve_density."""
        n = len(density)
        z = np.arange(n) * dz
        z0 = z[n // 2]  # Assume centered

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))

        # Wake potential on primary axis
        ax.plot((z - z0) * 1e6, integrated_wake * 1e-6, "C0")
        ax.set_xlabel(r"$z - z_0$ (µm)")
        ax.set_ylabel(r"Wake potential (MV/m)", color="C0")
        ax.tick_params(axis="y", labelcolor="C0")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title("Wake Potential and Current Profile")

        # Density as current on secondary axis (I = ρ * c)
        ax2 = ax.twinx()
        current = density * c_light  # [A]
        ax2.fill_between((z - z0) * 1e6, current, alpha=0.3, color="C1")
        ax2.set_ylabel(r"Current (A)", color="C1")
        ax2.tick_params(axis="y", labelcolor="C1")
        ax2.set_ylim(0, None)

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

    def plot(self, zmax: float = None, zmin: float = 0, n: int = 200, ax=None):
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
            Maximum wavenumber [1/m]. If None, uses a sensible default.
        kmin : float, optional
            Minimum wavenumber [1/m]. Default is 0.
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
