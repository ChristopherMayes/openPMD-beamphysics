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

import warnings
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import interp1d

from ..units import c_light
from .base import MIN_SAMPLES, WakefieldBase
from .impedance import ImpedanceWakefield

if TYPE_CHECKING:
    import os
    from collections.abc import Callable

__all__ = ["TabularWakefield"]

# Bmad bins the bunch at the table spacing, so a finer table mainly adds binning noise.
BMAD_MAX_AUTOMATIC_ROWS = 10000


class TabularWakefield(WakefieldBase):
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

        The wake is evaluated at n equally spaced points from z = -zmax to z = 0.
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
            Interpolation method. Default is 'cubic'.

        Returns
        -------
        TabularWakefield
            Table covering -zmax <= z <= 0.

        Raises
        ------
        ValueError
            If zmax is not positive, if n < 4, or if a `TabularWakefield` source does
            not cover -zmax <= z <= 0, which would silently pad the result with zeros.
        NotImplementedError
            If zmax or n is omitted and the model defines no default for it.

        Examples
        --------
        >>> wake = ResistiveWallPseudomode.from_material(
        ...     "copper-slac-pub-10707", radius=2.5e-3
        ... )
        >>> table = TabularWakefield.from_wakefield(wake)
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
    def from_impact_z(
        cls, source: str | os.PathLike | np.ndarray, kind: str = "cubic"
    ) -> TabularWakefield:
        """
        Read the longitudinal wake from an IMPACT-Z wake table.

        The abscissa s = -z is mirrored into z <= 0, and the transverse columns are
        ignored. See `beamphysics.interfaces.impact.parse_impact_z_wakefield`.

        Parameters
        ----------
        source : str, pathlib.Path or np.ndarray
            Path to the table, or the table itself as an array of shape (n, 4), such
            as an entry of the `file_data` of a lume-impact `ImpactZInput`.
        kind : str, optional
            Interpolation method. Default is 'cubic'.

        Returns
        -------
        TabularWakefield

        Examples
        --------
        >>> table = TabularWakefield.from_impact_z("rfdata41.in")
        """
        from ..interfaces.impact import parse_impact_z_wakefield

        data = parse_impact_z_wakefield(source)

        # s ascends from zero, so z = -s descends. Reverse to ascend in z.
        z = -data["s"][::-1]
        W = data["Wz"][::-1]

        return cls(z, W, kind=kind)

    @classmethod
    def from_impact_t(
        cls, source: str | os.PathLike | np.ndarray, length: float, kind: str = "cubic"
    ) -> TabularWakefield:
        """
        Read the longitudinal wake from an IMPACT-T wake table.

        The table has no position column, so its length must be supplied. The
        transverse columns are ignored. See
        `beamphysics.interfaces.impact.parse_impact_t_wakefield`.

        Parameters
        ----------
        source : str, pathlib.Path or np.ndarray
            Path to the table, or the table itself as an array of shape (n, 3) or
            (n, 2).
        length : float
            Distance covered by the table (m), given as V7 on the -6 element.
        kind : str, optional
            Interpolation method. Default is 'cubic'.

        Returns
        -------
        TabularWakefield

        Examples
        --------
        >>> table = TabularWakefield.from_impact_t("fort.41", length=3.55e-4)
        """
        from ..interfaces.impact import parse_impact_t_wakefield

        data = parse_impact_t_wakefield(source, length)

        return cls(-data["s"][::-1], data["Wz"][::-1], kind=kind)

    @classmethod
    def from_bmad(
        cls,
        source: np.ndarray,
        amp_scale: float = 1,
        z_scale: float = 1,
        kind: str = "cubic",
    ) -> TabularWakefield:
        """
        Read the longitudinal wake from a Bmad z_long wake table.

        This is the inverse of `to_bmad`, applied to the array that pytao returns for
        `tao.ele_wake(ele_id, who="sr_z_long_table")`. Bmad extends the table
        symmetrically about z = 0; the rows ahead of the source particle are dropped,
        and the grid is rebuilt to end exactly at z = 0.

        $$W(z) = \\text{amp\\_scale} \\cdot W_\\text{table}(\\text{z\\_scale} \\cdot z)$$

        Parameters
        ----------
        source : np.ndarray
            Table of shape (n, 2) holding z (m) and W (V/C/m) on a uniform grid.
        amp_scale : float, optional
            Bmad `sr%amp_scale`. Default is 1.
        z_scale : float, optional
            Bmad `sr%z_scale`. Default is 1.
        kind : str, optional
            Interpolation method. Default is 'cubic'.

        Returns
        -------
        TabularWakefield

        Raises
        ------
        ValueError
            If the table is empty (the element has no z_long wake), if z_scale is not
            positive, if the grid is not uniform or has no row at z = 0, or if the
            wake is nonzero ahead of the source particle.

        Notes
        -----
        With `scale_with_length = F`, Bmad reads the table in V/C; pass amp_scale
        divided by the structure length to obtain V/C/m. The smoothing_sigma and
        position_dependence settings are ignored.

        Examples
        --------
        >>> rows = tao.ele_wake("pipe1", who="sr_z_long_table")
        >>> table = TabularWakefield.from_bmad(rows)
        """
        dat = np.asarray(source, dtype=float)
        if dat.size == 0:
            raise ValueError(
                "The table is empty. The element has no z_long wake; a wake given only "
                "by pseudomodes has no table."
            )
        if dat.ndim != 2 or dat.shape[1] != 2 or dat.shape[0] < 2:
            raise ValueError(
                f"Expected an array of shape (n, 2) holding z and W, got {dat.shape}"
            )
        if z_scale <= 0:
            raise ValueError(f"z_scale must be positive, got {z_scale}")
        z, W = dat[:, 0], dat[:, 1]

        dz = (z[-1] - z[0]) / (len(z) - 1)
        # A fraction of the spacing, so the check does not depend on Tao's print format.
        atol = 1e-2 * abs(dz)
        if dz <= 0 or not np.allclose(np.diff(z), dz, rtol=0, atol=atol):
            raise ValueError(
                "The z column must be uniformly spaced and increasing, as Bmad stores "
                "z_long tables."
            )

        ahead = z > dz / 2
        if np.any(W[ahead] != 0):
            raise ValueError(
                "The table is nonzero ahead of the source particle, up to "
                f"{np.abs(W[ahead]).max():.6e}. TabularWakefield is causal, and "
                "dropping these values would change the wake."
            )
        m = np.count_nonzero(~ahead)
        if m == 0 or abs(z[m - 1]) > atol:
            raise ValueError("The table has no row at z = 0.")

        z = -dz * np.arange(m - 1, -1, -1)

        return cls(z / z_scale, amp_scale * W[:m], kind=kind)

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

    def to_bmad(
        self,
        file: str | os.PathLike | None = None,
        amp_scale: float = 1,
        z_scale: float = 1,
        scale_with_length: bool = True,
    ) -> str:
        """
        Format this table as the value of a Bmad `sr_wake` with a z_long table.

        Bmad needs a uniform table that ends at z = 0. Any other table is first
        resampled at its finest spacing, limited to 10000 rows. Bmad uses the same
        sign conventions and halves the z = 0 entry itself, so W(0-) is written.

        Parameters
        ----------
        file : str or pathlib.Path, optional
            If given, also write the value there, for use as `sr_wake = call::<file>`.
        amp_scale : float, optional
            Bmad `amp_scale`. Default is 1.
        z_scale : float, optional
            Bmad `z_scale`. Default is 1.
        scale_with_length : bool, optional
            If True (default), Bmad reads the table in V/C/m and multiplies it by the
            element length. If False, Bmad reads it in V/C, so multiply a table in
            V/C/m by the structure length, for example through amp_scale.

        Returns
        -------
        str

        Warns
        -----
        UserWarning
            If the resample would need more than 10000 rows.

        Notes
        -----
        Bmad applies the table by a circular convolution over the table extended
        symmetrically about z = 0, so the table must be longer than the bunch.

        Examples
        --------
        >>> table.to_bmad("wake.bmad")  # sr_wake = call::wake.bmad
        """
        dz = np.diff(self._z)
        uniform = np.allclose(dz, dz[0], rtol=1e-9, atol=0) and self._z[-1] == 0
        if uniform:
            table = self
        else:
            n = self.default_n_samples(self.default_zmax)
            if n > BMAD_MAX_AUTOMATIC_ROWS:
                warnings.warn(
                    f"Preserving the finest spacing of this table needs {n} rows. The "
                    f"Bmad table is resampled at {BMAD_MAX_AUTOMATIC_ROWS} rows instead, "
                    "which is coarser than that spacing. Use "
                    "TabularWakefield.from_wakefield with an explicit n to choose the "
                    "size.",
                    UserWarning,
                    stacklevel=2,
                )
                n = BMAD_MAX_AUTOMATIC_ROWS
            table = TabularWakefield.from_wakefield(self, n=n)

        logical = "T" if scale_with_length else "F"
        lines = [
            f"{{amp_scale = {amp_scale}, z_scale = {z_scale}, "
            f"scale_with_length = {logical},",
            "z_long = {w = {",
        ]
        # Adding 0.0 turns -0.0 into 0.0.
        for z, W in zip(table._z + 0.0, table._W):
            lines.append(f"{z:.17e} {W:.17e},")
        lines.append("}}}")
        s = "\n".join(lines) + "\n"

        if file is not None:
            with open(file, "w") as f:
                f.write(s)

        return s

    def to_impact_z(
        self,
        file: str | os.PathLike | None = None,
        zmax: float | None = None,
        n: int | None = None,
        wake_x: Callable[[np.ndarray], np.ndarray] | None = None,
        wake_y: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> np.ndarray:
        """
        Build, and optionally write, the IMPACT-Z wakefield table for this wake.

        See `beamphysics.interfaces.impact.create_impact_z_wakefield_rfdata`.

        Parameters
        ----------
        file : str or pathlib.Path, optional
            If given, also write the table there, conventionally as
            `rfdata{file_id}.in`.
        zmax, n, wake_x, wake_y
            As for `create_impact_z_wakefield_rfdata`.

        Returns
        -------
        np.ndarray
            Array of shape (n, 4) with columns s (m), Wz (V/C/m), Wx and Wy (V/C/m^2).

        Examples
        --------
        >>> table.to_impact_z("rfdata41.in")
        """
        from ..interfaces.impact import (
            _WAKEFIELD_FMT,
            create_impact_z_wakefield_rfdata,
        )

        rfdata = create_impact_z_wakefield_rfdata(self, zmax, n, wake_x, wake_y)

        if file is not None:
            np.savetxt(file, rfdata, fmt=_WAKEFIELD_FMT)

        return rfdata

    def to_impact_t(
        self,
        file: str | os.PathLike | None = None,
        zmax: float | None = None,
        n: int | None = None,
        wake_x: Callable[[np.ndarray], np.ndarray] | None = None,
        wake_y: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> tuple[np.ndarray, float]:
        """
        Build, and optionally write, the IMPACT-T wakefield table for this wake.

        See `beamphysics.interfaces.impact.create_impact_t_wakefield_data`.

        Parameters
        ----------
        file : str or pathlib.Path, optional
            If given, also write the table there, conventionally as `fort.{Bmpstp}`.
        zmax, n, wake_x, wake_y
            As for `create_impact_t_wakefield_data`.

        Returns
        -------
        data : np.ndarray
            Array of shape (n, 3) with columns Wz (V/C/m), Wx and Wy (V/C/m^2).
        length : float
            Distance covered by the table (m), to be given as V7.

        Examples
        --------
        >>> data, length = table.to_impact_t("fort.41", n=1000)
        """
        from ..interfaces.impact import (
            _WAKEFIELD_FMT,
            create_impact_t_wakefield_data,
        )

        data, length = create_impact_t_wakefield_data(self, zmax, n, wake_x, wake_y)

        if file is not None:
            np.savetxt(file, data, fmt=_WAKEFIELD_FMT)

        return data, length

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
