"""
Tabular wakefield representation.

This module provides the TabularWakefield class for wakefields defined
by user-supplied tabular data with interpolation.

Classes
-------
TabularWakefield
    Interpolation-based wakefield from user-supplied data
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d

from ..units import c_light
from .base import WakefieldBase

__all__ = ["TabularWakefield"]


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
    ::

        z_data = -np.linspace(1e-6, 1e-3, 100)
        W_data = 1e15 * np.exp(z_data / 100e-6) * np.sin(1e5 * z_data)
        wake = TabularWakefield(z_data, W_data)
        wake.wake(-50e-6)  # Interpolated wake at 50 µm behind source
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
