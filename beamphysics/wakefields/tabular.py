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
from .base import MIN_SAMPLES, WakefieldBase
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
        and must be strictly increasing.
    W : np.ndarray
        Wakefield values [V/C/m] at each z position.
    fill_value : float, optional
        Value to return outside the interpolation range. Default is 0.
    kind : str, optional
        Interpolation method. Default is 'cubic'.

    Examples
    --------
    ::

        z_data = -np.linspace(1e-3, 0, 100)
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
        zmax : float, optional
            Largest trailing distance behind the source particle to tabulate [m],
            given as a positive number. Defaults to the natural range of the source
            model, reported by its default_zmax.
        n : int, optional
            Number of samples. Defaults to the number the source model needs over that
            range, reported by its default_n_samples.
        kind : str, optional
            Interpolation method passed to the constructor. Default is 'cubic'.

        Returns
        -------
        TabularWakefield
            Table covering -zmax <= z <= 0.

        Raises
        ------
        ValueError
            If zmax is not positive, if n is smaller than the four points required by
            the interpolator, or if the source is itself a :class:`TabularWakefield`
            that does not cover the requested range. Resampling beyond the range of a
            table would return the fill value rather than the wake, so the result
            would be silently zero padded.
        NotImplementedError
            If zmax or n is omitted and the source model defines no natural range or
            no shortest length scale.

        Examples
        --------
        ::

            wake = ResistiveWallPseudomode.from_material(
                "copper-slac-pub-10707", radius=2.5e-3
            )
            table = TabularWakefield.from_wakefield(wake)
        """
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

        # Ascending in z, from -zmax up to the source particle at z = 0.
        z = -np.linspace(zmax, 0.0, n)

        if isinstance(wakefield, ImpedanceWakefield):
            # The array branch of ImpedanceWakefield.wake inverts the impedance on a
            # grid of period L = 2*pi*(n_fft - 1)/k_max. The cosine transform is even,
            # so the wake at s is aliased by its mirror at L - s. Enlarging n_fft until
            # L >= 2*zmax keeps every mirror beyond zmax, which bounds the aliasing by
            # the wake that truncation at zmax already discards.
            n_required = int(np.ceil(zmax * wakefield._k_max / np.pi)) + 2
            n_fft = max(4096, 1 << int(n_required - 1).bit_length())
            W = wakefield.wake(z, n_fft=n_fft)
        else:
            # A single array call. The scalar branch of some models uses quadrature
            # and is prohibitively slow point by point.
            W = wakefield.wake(z)

        return cls(z, np.asarray(W, dtype=float), kind=kind)

    @classmethod
    def from_impact_z(cls, source, kind: str = "cubic") -> TabularWakefield:
        """
        Read the longitudinal wake from an IMPACT-Z wake table.

        IMPACT-Z tabulates the wake against the distance s = -z >= 0 behind the source
        particle, whereas this package uses z <= 0. The two share the sign convention
        for the wake itself, so only the abscissa is reversed. The transverse columns
        of the table are ignored, because :class:`WakefieldBase` is longitudinal only.

        Parameters
        ----------
        source : str, pathlib.Path or array_like
            Path to an IMPACT-Z wake table, conventionally named rfdata{N}.in, or the
            table itself as an array of shape (n, 4). The array form recovers a model
            from a table carried in memory, such as an entry of the file_data mapping
            of a lume-impact ImpactZInput.
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
            table = TabularWakefield.from_impact_z(I.input.file_data["41"])
        """
        from ..interfaces.impact import parse_impact_z_wakefield

        data = parse_impact_z_wakefield(source)

        # s ascends from zero, so z = -s descends. Reverse to ascend in z.
        z = -data["s"][::-1]
        W = data["Wz"][::-1]

        return cls(z, W, kind=kind)

    @property
    def default_zmax(self) -> float:
        """
        Trailing distance beyond which the wake may be truncated [m].

        The range the table itself covers. Beyond it the interpolator returns the
        fill value rather than the wake.

        Returns
        -------
        float
            Largest trailing distance behind the source particle worth tabulating.
        """
        return -float(np.min(self._z))

    def default_n_samples(self, zmax: float) -> int:
        """
        Number of uniform samples needed to represent the wake out to zmax.

        The finest spacing of the table within the requested range, so that a
        non-uniform table loses none of its detail when resampled uniformly.
        Resampling more finely than that recovers nothing beyond the interpolant.

        Parameters
        ----------
        zmax : float
            Largest trailing distance behind the source particle to tabulate [m],
            given as a positive number.

        Returns
        -------
        int
            Number of samples.
        """
        dz = np.diff(self._z)
        # Only intervals overlapping the requested range [-zmax, 0] matter.
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
