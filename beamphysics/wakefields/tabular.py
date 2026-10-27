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

import numpy as np
from scipy.interpolate import interp1d

from ..units import c_light
from .base import MIN_SAMPLES, WakefieldBase
from .impedance import ImpedanceWakefield

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
        Longitudinal positions [m]. Should be negative (behind source)
        and must be strictly increasing.
    W : np.ndarray
        Wakefield values [V/C/m] at each z position. These are point values
        W(z_i) of the wake, not averages over a bin. At z = 0 the value is
        W(0-), the wake immediately behind the source particle.
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
        export routine. A uniform table is also the form in which some tracking codes
        read a short-range wake: IMPACT-Z requires one, and Bmad accepts one as an
        alternative to pseudomodes.

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

    @classmethod
    def from_impact_t(
        cls, source, length: float, kind: str = "cubic"
    ) -> TabularWakefield:
        """
        Read the longitudinal wake from an IMPACT-T wake table.

        An IMPACT-T table has no position column. Its rows are equally spaced in the
        distance s = -z >= 0 behind the source particle, from zero to the length V7
        given on the -6 element, which must therefore be supplied. IMPACT-T shares
        the sign convention of this package, and the transverse columns are ignored,
        because :class:`WakefieldBase` is longitudinal only.

        Parameters
        ----------
        source : str, pathlib.Path or array_like
            Path to an IMPACT-T wake table, conventionally named fort.{Bmpstp}, or the
            table itself as an array of shape (n, 3) or (n, 2).
        length : float
            Distance covered by the table [m], given as V7 on the -6 element.
        kind : str, optional
            Interpolation method passed to the constructor. Default is 'cubic'.

        Returns
        -------
        TabularWakefield
            Longitudinal wake [V/C/m] as a function of z <= 0 [m].

        See Also
        --------
        beamphysics.interfaces.impact.parse_impact_t_wakefield : Underlying reader.
        beamphysics.interfaces.impact.write_impact_t_wakefield : Corresponding writer.

        Examples
        --------
        ::

            table = TabularWakefield.from_impact_t("fort.41", length=3.55e-4)
        """
        from ..interfaces.impact import parse_impact_t_wakefield

        data = parse_impact_t_wakefield(source, length)

        return cls(-data["s"][::-1], data["Wz"][::-1], kind=kind)

    @classmethod
    def from_bmad(
        cls,
        source,
        amp_scale: float = 1,
        z_scale: float = 1,
        kind: str = "cubic",
    ) -> TabularWakefield:
        """
        Read the longitudinal wake from a Bmad z_long wake table.

        This is the inverse of :meth:`to_bmad`. The table is taken as the array
        returned by Tao for an element with a z_long short-range wake, which pytao
        provides as ``tao.ele_wake(ele_id, who="sr_z_long_table")``. Reading a Bmad
        lattice or wake file directly is not supported, because parsing Bmad syntax
        belongs to Bmad.

        Bmad uses the same conventions as this package: z <= 0 lies behind the source
        particle and a positive wake is energy-losing. When Bmad parses a table it
        extends it symmetrically about z = 0, so the rows ahead of the source particle
        are dropped here. Bmad stores the z = 0 entry as given and halves it only in
        its convolution kernel, so the value read back is W(0-) and is used unchanged.

        Parameters
        ----------
        source : array_like
            Table of shape (n, 2) holding z [m] and W [V/C/m], uniformly spaced and
            increasing in z, as returned by ``Tao.ele_wake`` with
            ``who="sr_z_long_table"``.
        amp_scale : float, optional
            Bmad wake amplitude scale factor, sr%amp_scale. Default is 1.
        z_scale : float, optional
            Bmad z-distance scale factor, sr%z_scale. Default is 1.
        kind : str, optional
            Interpolation method passed to the constructor. Default is 'cubic'.

        Returns
        -------
        TabularWakefield
            Longitudinal wake as a function of z <= 0 [m]. Bmad evaluates the table
            at z_scale * z and multiplies the kick by amp_scale, so the result is
            amp_scale * W_table(z_scale * z).

        Raises
        ------
        ValueError
            If the table is empty, as Tao returns for an element without a z_long
            wake, if z_scale is not positive, or if the table does not have two
            columns, is not uniformly spaced and increasing, has no row at z = 0, or
            is nonzero ahead of the source particle. TabularWakefield is causal, so
            dropping nonzero values ahead of the source would silently change the
            wake.

        Notes
        -----
        Tao currently prints the table with nine significant digits, which limits the
        accuracy of a round trip through Tao to about 1e-09. The positions are rebuilt
        on an exactly uniform grid ending at z = 0, so that :meth:`to_bmad` writes the
        result back without resampling. Each row must lie within one percent of the
        spacing of that grid.

        If scale_with_length is False, Bmad reads the table as a wake in V/C rather
        than V/C/m. Passing amp_scale divided by the length of the structure then
        gives the wake per unit length. The smoothing_sigma and position_dependence
        settings of the wake have no counterpart here and are ignored; they are
        reported by ``tao.ele_wake(ele_id, who="sr_z_long")``.

        Examples
        --------
        ::

            rows = tao.ele_wake("pipe1", who="sr_z_long_table")
            table = TabularWakefield.from_bmad(rows)
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
        Trailing distance beyond which the wake may be truncated [m].

        $$z_{\\max} = -\\min_i z_i$$

        This is the range the table itself covers. Beyond it the interpolator returns
        the fill value rather than the wake.

        Returns
        -------
        float
            Largest trailing distance behind the source particle worth tabulating.
        """
        return -float(np.min(self._z))

    def default_n_samples(self, zmax: float) -> int:
        """
        Number of uniform samples needed to represent the wake out to zmax.

        $$n = \\max\\left( \\left\\lceil \\frac{z_{\\max}}{\\Delta z_{\\min}}
        \\right\\rceil + 1, \\; 4 \\right)$$

        Here Delta z_min is the finest spacing of the table within the requested range,
        so that a non-uniform table loses none of its detail when resampled uniformly.
        Resampling more finely than that recovers nothing beyond the interpolant. The
        lower limit of four is the number of points required by the cubic interpolant.

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

    def to_bmad(
        self,
        file=None,
        amp_scale: float = 1,
        z_scale: float = 1,
        scale_with_length: bool = True,
    ) -> str:
        """
        Format this table as a Bmad short-range wake with a z_long wake table.

        Bmad requires the table to be equally spaced, increasing in z and to contain
        the point z = 0. A table that does not already satisfy these conditions is
        first resampled with :meth:`from_wakefield` at its finest spacing, limited to
        10000 rows. Bmad uses the same conventions as this package: z <= 0 lies behind
        the source particle, a positive wake is energy-losing, and the wake is in
        V/C/m when scale_with_length is True. Bmad sets the wake to zero ahead of the
        source particle and halves the z = 0 entry itself, so the table is written
        unchanged.

        Parameters
        ----------
        file : str or pathlib.Path, optional
            Output path. If given, the string is also written there, so that it can
            be used in a lattice as ``sr_wake = call::<file>``.
        amp_scale : float, optional
            Bmad wake amplitude scale factor. Default is 1.
        z_scale : float, optional
            Bmad z-distance scale factor. Default is 1.
        scale_with_length : bool, optional
            Whether Bmad multiplies the wake by the length of the element that
            carries it. Default is True, in which case the table is read as a wake
            per unit length in V/C/m, as it is throughout this package, and the
            sr_wake should be attached to an element whose length equals that of the
            wake-producing structure. If False, Bmad applies the table once as a wake
            in V/C, so a table in V/C/m must first be multiplied by the length of the
            structure, for example through amp_scale.

        Returns
        -------
        str
            The sr_wake value, beginning with ``{``.

        Warns
        -----
        UserWarning
            If resampling a non-uniform table at its finest spacing needs more than
            10000 rows, in which case the table is resampled at 10000 rows. Bmad bins
            the bunch at the table spacing, so a finer table mainly adds binning
            noise. A table of any size can be built explicitly with
            :meth:`from_wakefield`.

        Notes
        -----
        Bmad applies the table by a circular FFT convolution over a window equal to
        the table extended symmetrically about z = 0. The tabulated range must
        therefore exceed the full length of the bunch, or the wake of the tail wraps
        onto the head. A longer table can be built from the source model with
        :meth:`from_wakefield`.

        Examples
        --------
        ::

            table = wake.to_tabular()
            print(table.to_bmad())
            table.to_bmad("wake.bmad")  # sr_wake = call::wake.bmad
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
        self, file=None, zmax=None, n=None, wake_x=None, wake_y=None
    ) -> np.ndarray:
        """
        Build the IMPACT-Z read-in short-range wakefield table for this wake.

        This is :func:`beamphysics.interfaces.impact.create_impact_z_wakefield_rfdata`
        applied to this table, which describes the format, the defaults and the row
        limit.

        Parameters
        ----------
        file : str or pathlib.Path, optional
            Output path. If given, the table is also written there. IMPACT-Z expects
            the name rfdata{file_id}.in, where file_id is given on the -41 element.
        zmax : float, optional
            Largest trailing distance behind the source particle to tabulate [m].
            Defaults to the range of this table.
        n : int, optional
            Number of rows. Defaults to the number needed to keep the finest spacing
            of this table, clamped to the 5000 rows IMPACT-Z accepts.
        wake_x, wake_y : callable, optional
            Horizontal and vertical dipole wakes as functions of z <= 0 [m],
            returning [V/C/m^2]. Their columns are zero when they are not supplied.

        Returns
        -------
        np.ndarray
            Array of shape (n, 4) holding the columns s [m], Wz [V/C/m], Wx [V/C/m^2]
            and Wy [V/C/m^2], with s ascending from zero.

        Examples
        --------
        ::

            table = wake.to_tabular()
            table.to_impact_z("rfdata41.in")
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
        self, file=None, zmax=None, n=None, wake_x=None, wake_y=None
    ) -> tuple[np.ndarray, float]:
        """
        Build the IMPACT-T read-in short-range wakefield table for this wake.

        This is :func:`beamphysics.interfaces.impact.create_impact_t_wakefield_data`
        applied to this table, which describes the format, the element parameters,
        the defaults and the row limit.

        Parameters
        ----------
        file : str or pathlib.Path, optional
            Output path. If given, the table is also written there. IMPACT-T expects
            the name fort.{Bmpstp}, where Bmpstp is given on the -6 element.
        zmax : float, optional
            Largest trailing distance behind the source particle to tabulate [m].
            Defaults to the range of this table.
        n : int, optional
            Number of rows. Defaults to the number needed to keep the finest spacing
            of this table, clamped to the 1000 rows IMPACT-T reads.
        wake_x, wake_y : callable, optional
            Horizontal and vertical dipole wakes as functions of z <= 0 [m],
            returning [V/C/m^2]. Their columns are zero when they are not supplied.

        Returns
        -------
        data : np.ndarray
            Array of shape (n, 3) holding the columns Wz [V/C/m], Wx [V/C/m^2] and
            Wy [V/C/m^2], ordered by increasing distance behind the source particle.
        length : float
            Distance covered by the table [m], to be given as V7 on the -6 element.

        Examples
        --------
        ::

            table = wake.to_tabular()
            data, length = table.to_impact_t("fort.41", n=1000)  # V7 = length
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
