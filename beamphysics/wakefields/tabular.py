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
from ._table import _WakefieldTableIO
from ._table import BMAD_MAX_AUTOMATIC_ROWS as BMAD_MAX_AUTOMATIC_ROWS
from .base import MIN_SAMPLES, WakefieldBase
from .impedance import ImpedanceWakefield

__all__ = ["TabularWakefield"]


class TabularWakefield(_WakefieldTableIO, WakefieldBase):
    """
    Wakefield defined by user-supplied tabular data with interpolation.

    Uses cubic spline interpolation to evaluate the wakefield at
    arbitrary positions between the supplied data points.

    Parameters
    ----------
    z : np.ndarray
        Longitudinal positions (m). Should be negative (behind source)
        and must be strictly increasing.
    W : np.ndarray
        Wakefield values (V/C/m) at each z position. At z = 0 this is W(0-), the
        wake immediately behind the source particle.
    fill_value : float, optional
        Value to return outside the interpolation range. Default is 0.
    kind : str, optional
        Interpolation method. Default is 'cubic'.

    Examples
    --------
    >>> z = -np.linspace(1e-3, 0, 100)
    >>> wake = TabularWakefield(z, 1e15 * np.exp(z / 100e-6) * np.sin(1e5 * z))
    >>> wake.wake(-50e-6)  # 50 µm behind the source
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
        if np.any(np.diff(z) <= 0):
            raise ValueError("z must be strictly increasing")

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

    @classmethod
    def from_wakefield(
        cls,
        wakefield: WakefieldBase,
        zmax: float | None = None,
        n: int | None = None,
        kind: str = "cubic",
    ) -> TabularWakefield:
        """
        Resample a wakefield model onto a uniform table.

        The wake is tabulated at n equally spaced points from z = -zmax to z = 0.
        This makes an expensive model, such as `ResistiveWallWakefield`, cheap to
        evaluate, and gives the uniform table that IMPACT-Z and Bmad read.

        Parameters
        ----------
        wakefield : WakefieldBase
            Model to resample.
        zmax : float, optional
            Largest distance behind the source particle (m). Defaults to
            `wakefield.default_zmax`.
        n : int, optional
            Number of samples. Defaults to `wakefield.default_n_samples(zmax)`.
        kind : str, optional
            Interpolation method. Defaults to 'cubic'.

        Returns
        -------
        TabularWakefield
            Table covering -zmax <= z <= 0.

        Raises
        ------
        ValueError
            If zmax is not positive, n < 4, the source is singular or binned,
            or a source table does not cover -zmax <= z <= 0.
        NotImplementedError
            If zmax or n is omitted and the model defines no default for it.

        Notes
        -----
        This class holds point samples only. For singular wakes or integrated
        tracking kernels, use `BinnedWakefield.from_wakefield`.

        Examples
        --------
        >>> wake = ResistiveWallPseudomode.from_material(
        ...     "copper-slac-pub-10707", radius=2.5e-3
        ... )
        >>> table = TabularWakefield.from_wakefield(wake)

        """
        if not isinstance(wakefield, WakefieldBase) or wakefield.is_singular:
            raise ValueError(
                "Only nonsingular pointwise models can be sampled. "
                "Use BinnedWakefield.from_wakefield for integrated kernels."
            )

        if zmax is None:
            zmax = wakefield.default_zmax

        if zmax <= 0:
            raise ValueError(f"zmax must be a positive trailing distance, got {zmax}")

        if isinstance(wakefield, TabularWakefield):
            zmax_source = wakefield.default_zmax
            if zmax > zmax_source:
                raise ValueError(
                    f"Requested zmax={zmax:.6e} m exceeds the {zmax_source:.6e} m "
                    "covered by the source table. Resampling beyond the tabulated "
                    "range would return the fill value, giving a silently zero "
                    "padded result."
                )
            z_end = float(np.max(wakefield.z_data))
            if z_end < 0:
                raise ValueError(
                    f"The source table ends at z={z_end:.6e} m, short of the source "
                    "particle at z = 0. Resampling there would return the fill "
                    "value, giving a silently zero padded result."
                )

        if n is None:
            n = wakefield.default_n_samples(zmax)

        if n < 4:
            raise ValueError(f"Need at least 4 points for interpolation, got n={n}")

        z = -np.linspace(zmax, 0.0, n)

        if isinstance(wakefield, ImpedanceWakefield):
            # The cosine transform has period L = 2*pi*(n_fft - 1)/k_max and is even,
            # so W(s) is aliased by W(L - s). L >= 2*zmax keeps the mirror beyond zmax.
            n_required = int(np.ceil(zmax * wakefield._k_max / np.pi)) + 2
            n_fft = max(4096, 1 << int(n_required - 1).bit_length())
            W = wakefield.wake(z, n_fft=n_fft)
        else:
            # The scalar branch of some models uses quadrature, which is
            # prohibitively slow point by point.
            W = wakefield.wake(z)

        return cls(z, np.asarray(W, dtype=float), kind=kind)

    @classmethod
    def _from_table_data(cls, z, W, kind=None):
        return cls(z, W, kind="cubic" if kind is None else kind)

    @property
    def default_zmax(self) -> float:
        """
        Range covered by the table (m).

        $$z_{\\max} = -\\min_i z_i$$
        """
        return -float(np.min(self._z))

    def default_n_samples(self, zmax: float) -> int:
        """
        Number of uniform samples that keeps the finest spacing of the table.

        $$n = \\max\\left( \\left\\lceil \\frac{z_{\\max}}{\\Delta z_{\\min}}
        \\right\\rceil + 1, \\; 4 \\right)$$

        Here Delta z_min is the finest spacing within -zmax <= z <= 0.

        Parameters
        ----------
        zmax : float
            Largest distance behind the source particle (m).

        Returns
        -------
        int
        """
        dz = np.diff(self._z)
        dz = dz[self._z[1:] > -zmax]
        # Tolerate rounding in the spacing of a nominally uniform grid.
        intervals = np.ceil(zmax / dz.min() * (1 - 1e-9))

        return max(int(intervals) + 1, MIN_SAMPLES)

    def wake(self, z: np.ndarray | float) -> np.ndarray | float:
        """
        Evaluate the wakefield at position z using interpolation.

        Parameters
        ----------
        z : float or np.ndarray
            Longitudinal position (m).

        Returns
        -------
        W : float or np.ndarray
            Interpolated wakefield value (V/C/m). Returns fill_value
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
            Wavenumber (1/m).

        Returns
        -------
        Z : complex or np.ndarray
            Impedance (Ohm/m).
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
