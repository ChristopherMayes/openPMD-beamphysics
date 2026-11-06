"""Shared wake table readers and writers."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, TypeVar

import numpy as np

if TYPE_CHECKING:
    import os
    from collections.abc import Callable

BMAD_MAX_AUTOMATIC_ROWS = 10000
_TableType = TypeVar("_TableType", bound="_WakefieldTableIO")


class _WakefieldTableIO:
    @classmethod
    def from_impact_z(
        cls: type[_TableType],
        source: str | os.PathLike | np.ndarray,
        kind: str | None = None,
    ) -> _TableType:
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
            Interpolation for `TabularWakefield` (default 'cubic'). Leave unset
            for `BinnedWakefield`, which requires a uniform grid ending at zero.

        Returns
        -------
        TabularWakefield or BinnedWakefield
            Table of the class used to call this reader.

        Examples
        --------
        >>> table = TabularWakefield.from_impact_z("rfdata41.in")
        """
        from ..interfaces.impact import parse_impact_z_wakefield

        data = parse_impact_z_wakefield(source)

        # s ascends from zero, so z = -s descends. Reverse to ascend in z.
        z = -data["s"][::-1]
        W = data["Wz"][::-1]

        return cls._from_table_data(z, W, kind=kind)

    @classmethod
    def from_impact_t(
        cls: type[_TableType],
        source: str | os.PathLike | np.ndarray,
        length: float,
        kind: str | None = None,
    ) -> _TableType:
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
            Interpolation for `TabularWakefield` (default 'cubic'). Leave unset
            for `BinnedWakefield`, which requires a uniform grid ending at zero.

        Returns
        -------
        TabularWakefield or BinnedWakefield
            Table of the class used to call this reader.

        Examples
        --------
        >>> table = TabularWakefield.from_impact_t("fort.41", length=3.55e-4)
        """
        from ..interfaces.impact import parse_impact_t_wakefield

        data = parse_impact_t_wakefield(source, length)

        return cls._from_table_data(-data["s"][::-1], data["Wz"][::-1], kind=kind)

    @classmethod
    def from_bmad(
        cls: type[_TableType],
        source: np.ndarray,
        amp_scale: float = 1,
        z_scale: float = 1,
        kind: str | None = None,
    ) -> _TableType:
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
            Interpolation for `TabularWakefield` (default 'cubic'). Leave unset
            for `BinnedWakefield`, which requires a uniform grid ending at zero.

        Returns
        -------
        TabularWakefield or BinnedWakefield
            Table of the class used to call this reader.

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
                f"{np.abs(W[ahead]).max():.6e}. Wake tables are causal, and "
                "dropping these values would change the wake."
            )
        m = np.count_nonzero(~ahead)
        if m == 0 or abs(z[m - 1]) > atol:
            raise ValueError("The table has no row at z = 0.")

        z = -dz * np.arange(m - 1, -1, -1)

        return cls._from_table_data(z / z_scale, amp_scale * W[:m], kind=kind)

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
        dz = np.diff(self.z_data)
        uniform = np.allclose(dz, dz[0], rtol=1e-9, atol=0) and self.z_data[-1] == 0
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
            table = type(self).from_wakefield(self, n=n)

        logical = "T" if scale_with_length else "F"
        lines = [
            f"{{amp_scale = {amp_scale}, z_scale = {z_scale}, "
            f"scale_with_length = {logical},",
            "z_long = {w = {",
        ]
        # Adding 0.0 turns -0.0 into 0.0.
        for z, W in zip(table.z_data + 0.0, table.W_data):
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
