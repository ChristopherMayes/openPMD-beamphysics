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
from .impedance import ImpedanceWakefield

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

    @classmethod
    def from_wakefield(
        cls,
        wakefield: WakefieldBase,
        zmax: float,
        n: int = 1000,
        kind: str = "cubic",
    ) -> TabularWakefield:
        """
        Resample an arbitrary wakefield model onto a uniform table.

        The wake is evaluated on n equally spaced points spanning the trailing
        distances from zero to zmax behind the source particle. The result is stored
        in the package convention, in which the longitudinal coordinate satisfies
        z <= 0 behind the source and a positive wake is energy-losing.

        Tabulation is the practical way to reuse an expensive model. The wake of
        :class:`ResistiveWallWakefield` is obtained from its impedance by a transform
        on every call, which is far too slow to evaluate inside a tracking loop or an
        export routine, and even the analytic :class:`ResistiveWallPseudomode` must be
        tabulated before it can be handed to an external code.

        Parameters
        ----------
        wakefield : WakefieldBase
            Source wakefield model to resample.
        zmax : float
            Largest trailing distance behind the source particle to tabulate [m],
            given as a positive number.
        n : int, optional
            Number of samples. Default is 1000.
        kind : str, optional
            Interpolation method passed to the constructor. Default is 'cubic'.

        Returns
        -------
        TabularWakefield
            Table covering -zmax <= z <= 0.

        Raises
        ------
        ValueError
            If zmax is not positive, or if n is smaller than the four points required
            by the interpolator.

        Examples
        --------
        ::

            wake = ResistiveWallWakefield.from_material(
                "copper-slac-pub-10707", radius=2.5e-3
            )
            table = TabularWakefield.from_wakefield(wake, zmax=100 * wake.s0)
        """
        if zmax <= 0:
            raise ValueError(f"zmax must be a positive trailing distance, got {zmax}")
        if n < 4:
            raise ValueError(f"Need at least 4 points for interpolation, got n={n}")

        # Ascending in z, from -zmax up to the source particle at z = 0.
        z = -np.linspace(zmax, 0.0, n)

        if isinstance(wakefield, ImpedanceWakefield):
            # The array branch of ImpedanceWakefield.wake inverts the impedance on a
            # grid reaching a trailing distance of 2*pi*(n_fft - 1)/k_max. Enlarge
            # n_fft when required so that zmax lies inside that grid, because beyond
            # it the transform wraps around and the tail of the table is aliased.
            n_required = int(np.ceil(zmax * wakefield._k_max / (2 * np.pi))) + 2
            n_fft = max(4096, 1 << int(n_required - 1).bit_length())
            W = wakefield.wake(z, n_fft=n_fft)
        else:
            # A single array call. The scalar branch of some models uses quadrature
            # and is prohibitively slow point by point.
            W = wakefield.wake(z)

        return cls(z, np.asarray(W, dtype=float), kind=kind)

    @classmethod
    def from_impact_z(cls, filename, kind: str = "cubic") -> TabularWakefield:
        """
        Read the longitudinal wake from an IMPACT-Z wake table.

        IMPACT-Z tabulates the wake against the distance s = -z >= 0 behind the source
        particle, whereas this package uses z <= 0. The two share the sign convention
        for the wake itself, so only the abscissa is reversed. The transverse columns
        of the file are ignored, because :class:`WakefieldBase` is longitudinal only.

        Parameters
        ----------
        filename : str or pathlib.Path
            Path to an IMPACT-Z wake table, conventionally named rfdata{N}.in.
        kind : str, optional
            Interpolation method passed to the constructor. Default is 'cubic'.

        Returns
        -------
        TabularWakefield
            Longitudinal wake [V/C/m] as a function of z <= 0 [m].

        See Also
        --------
        beamphysics.interfaces.impact.parse_impact_z_wakefield : Underlying reader.
        beamphysics.interfaces.impact.write_impact_z_wakefield : Corresponding writer.

        Examples
        --------
        ::

            table = TabularWakefield.from_impact_z("rfdata41.in")
        """
        from ..interfaces.impact import parse_impact_z_wakefield

        data = parse_impact_z_wakefield(filename)

        # s ascends from zero, so z = -s descends. Reverse to ascend in z.
        z = -data["s"][::-1]
        W = data["Wz"][::-1]

        return cls(z, W, kind=kind)

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
