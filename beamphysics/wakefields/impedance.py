"""
Impedance-based wakefield model.

This module provides the ImpedanceWakefield class for defining wakefields
through their impedance function Z(k).

Classes
-------
ImpedanceWakefield
    Wakefield defined through its impedance Z(k)
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..units import c_light
from .base import WakefieldBase

__all__ = ["ImpedanceWakefield"]


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
    ::

        def my_impedance(k):
            return 100 / (1 + 1j * k * 1e-6)  # Simple resonator
        wake = ImpedanceWakefield(my_impedance)
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

    def wake(self, z: np.ndarray | float, n_fft: int = 4096) -> np.ndarray | float:
        """
        Evaluate the wakefield at position z.

        If a wakefield function was provided, uses it directly.
        Otherwise, computes via cosine transform of Re[Z(k)]:
        - Scalars: numerical quadrature (accurate)
        - Arrays: FFT with interpolation (fast)

        Parameters
        ----------
        z : float or np.ndarray
            Longitudinal position [m].
        n_fft : int, optional
            Number of FFT points for array evaluation. Default is 4096.

        Returns
        -------
        W : float or np.ndarray
            Wakefield value [V/C/m]. Returns 0 for z > 0.
        """
        if self._wakefield_func is not None:
            return self._wakefield_func(z)

        z = np.asarray(z)
        scalar_input = z.ndim == 0
        z = np.atleast_1d(z)

        if scalar_input:
            # Single point: use quadrature for accuracy
            from scipy.integrate import quad

            z_val = float(z[0])
            if z_val > 0:
                return 0.0

            def integrand(k):
                return np.real(self._impedance_func(k)) * np.cos(k * z_val)

            result, _ = quad(integrand, 0, self._k_max, limit=200)
            return (2 * c_light / np.pi) * result
        else:
            # Array: use FFT for speed
            from scipy.fft import irfft
            from scipy.interpolate import interp1d

            k_max = self._k_max
            dk = k_max / (n_fft - 1)
            k_grid = np.linspace(0, k_max, n_fft)

            Zk_grid = self._impedance_func(k_grid)
            ReZ = np.real(Zk_grid)

            n_full = 2 * (n_fft - 1)
            dz = 2 * np.pi / (n_full * dk)
            z_grid = np.arange(n_full) * dz

            W_grid = irfft(ReZ, n=n_full) * n_full * dk * (c_light / np.pi)

            interp = interp1d(
                z_grid, W_grid, kind="cubic", bounds_error=False, fill_value=0.0
            )

            result = interp(-z)
            result = np.where(z > 0, 0.0, result)

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
