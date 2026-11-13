"""Fixed-width integrated Green functions for wakefield tracking."""

from __future__ import annotations

import warnings

import numpy as np

from ._convolution import _WakefieldConvolution
from ._table import _WakefieldTableIO
from .base import WakefieldBase

__all__ = ["BinnedWakefield"]


class BinnedWakefield(_WakefieldConvolution, _WakefieldTableIO):
    """
    Wake bin averages on a fixed-width causal grid.

    This is a discrete tracking kernel, not a pointwise wake model. Change its
    spacing with `rebin`, not interpolation. The first entry is doubled for the
    half-self convention used by the convolution and the tracking-code writers.

    Parameters
    ----------
    dz : float
        Bin width (m).
    W : array-like
        At least two values (V/C/m) in increasing trailing distance: `2 K0, K1, ...`, where
        `K0 = integrated_wake(-dz/2) / dz`. The full delta contribution is in K0.

    Examples
    --------
    >>> bins = BinnedWakefield.from_wakefield(wake, zmax=20e-3, n=2001)
    >>> impact = bins.rebin(dz=50e-6)
    """

    def __init__(self, dz: float, W: np.ndarray) -> None:
        if not np.isfinite(dz) or dz <= 0:
            raise ValueError("dz must be finite and positive")
        values = np.asarray(W, dtype=float)
        if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
            raise ValueError("W must be a finite 1D array with at least two bins")
        self._dz = float(dz)
        self._values = values.copy()

    @property
    def dz(self) -> float:
        """Fixed bin width (m)."""
        return self._dz

    @property
    def z_data(self) -> np.ndarray:
        """Bin centres (m), increasing from the tail to zero."""
        return -self.dz * np.arange(len(self._values) - 1, -1, -1)

    @property
    def W_data(self) -> np.ndarray:
        """Stored values (V/C/m) in the same order as `z_data`."""
        return self._values[::-1].copy()

    @property
    def default_zmax(self) -> float:
        """Trailing distance of the last bin centre (m)."""
        return (len(self._values) - 1) * self.dz

    def default_n_samples(self, zmax: float) -> int:
        """Number of bins covering zmax without coarsening the spacing."""
        return max(2, int(np.ceil(zmax / self.dz * (1 - 1e-9))) + 1)

    @classmethod
    def _from_table_data(cls, z, W, kind=None) -> BinnedWakefield:
        if kind is not None:
            raise ValueError("Bin averages have no interpolation kind")
        z = np.asarray(z, dtype=float)
        values = np.asarray(W, dtype=float)
        if z.ndim != 1 or len(z) < 2 or z.shape != values.shape:
            raise ValueError(
                "Expected matching 1D z and W arrays with at least two rows"
            )
        dz = (z[-1] - z[0]) / (len(z) - 1)
        if (
            not np.isfinite(dz)
            or dz <= 0
            or z[-1] != 0
            or not np.allclose(np.diff(z), dz, rtol=1e-6, atol=0)
        ):
            raise ValueError("A binned table must be uniform and end at z = 0")
        return cls(dz, values[::-1])

    @classmethod
    def from_wakefield(
        cls,
        wakefield: WakefieldBase | BinnedWakefield,
        zmax: float | None = None,
        n: int | None = None,
    ) -> BinnedWakefield:
        """
        Integrate a model into bins, or conservatively rebin an existing kernel.

        Parameters
        ----------
        wakefield : WakefieldBase or BinnedWakefield
            Model with `integrated_wake`, or an existing binned kernel.
        zmax : float, optional
            Last bin centre's trailing distance (m). Defaults to the model range.
        n : int, optional
            Number of bins, including zero lag. Defaults to the model's sampling.

        Returns
        -------
        BinnedWakefield
        """
        if zmax is None:
            zmax = wakefield.default_zmax
        if not np.isfinite(zmax) or zmax <= 0:
            raise ValueError("zmax must be finite and positive")
        if n is None:
            n = wakefield.default_n_samples(zmax)
        if not isinstance(n, (int, np.integer)) or n < 2:
            raise ValueError("n must be an integer of at least 2")
        dz = zmax / (n - 1)
        if isinstance(wakefield, BinnedWakefield):
            return wakefield.rebin(dz, zmax=zmax, max_rows=n)
        return cls(dz, wakefield.bin_averaged_wake(dz, n))

    def _wake_kernel(self, dz: float, n: int, offset: float, method: str) -> np.ndarray:
        if offset != 0:
            raise ValueError("A binned wake does not support an offset")
        if method not in ("auto", "bin_average"):
            raise ValueError("A binned wake cannot be sampled at points")
        if not np.isfinite(dz) or not np.isclose(dz, self.dz, rtol=1e-9, atol=0):
            raise ValueError("Use rebin to match the density or particle grid spacing")
        values = np.zeros(n)
        count = min(n, len(self._values))
        values[:count] = self._values[:count]
        return values

    def convolve_density(
        self,
        density,
        dz=None,
        offset=0,
        include_self_kick=True,
        plot=False,
        ax=None,
        method="auto",
    ) -> np.ndarray:
        """
        Convolve charge density (C/m) at this kernel's spacing, returning V/m.

        An explicit dz must match `dz`; use `rebin` to change the spacing.
        """
        return super().convolve_density(
            density,
            self.dz if dz is None else dz,
            offset,
            include_self_kick,
            plot,
            ax,
            method,
        )

    def gaussian_wake_potential(
        self, z: np.ndarray | float, sigma: float, dz: float | None = None
    ) -> np.ndarray | float:
        """
        Wake potential of a Gaussian bunch per unit charge (V/C/m) at this spacing.

        As for `WakefieldBase.gaussian_wake_potential`. An explicit dz must
        match `dz`.
        """
        return super().gaussian_wake_potential(z, sigma, self.dz if dz is None else dz)

    def binned_particle_kicks(
        self,
        z,
        weight,
        dz=None,
        include_self_kick=True,
        method="auto",
    ) -> np.ndarray:
        """
        Apply the fixed-width kernel to particle positions (m) and charges (C).

        Returns energy kicks (eV/m). An explicit dz must match this kernel's
        spacing. With `include_self_kick=False`, remove each particle's complete
        response to its own cloud-in-cell deposit.
        """
        spacing = self.dz if dz is None else dz
        self._wake_kernel(spacing, 2, 0, method)
        return super().binned_particle_kicks(
            z, weight, spacing, include_self_kick, method
        )

    def particle_kicks(self, z, weight, include_self_kick=True) -> np.ndarray:
        """Energy kicks (eV/m) using this kernel's fixed-width particle grid."""
        return self.binned_particle_kicks(
            z, weight, include_self_kick=include_self_kick
        )

    def rebin(
        self,
        dz: float,
        zmax: float | None = None,
        max_rows: int | None = None,
    ) -> BinnedWakefield:
        """
        Change spacing conservatively by interpolating cumulative bin integrals.

        Odd multiples of the original spacing align the bin edges exactly.
        Refinement cannot recover structure below the original bin width.

        Parameters
        ----------
        dz : float
            Target spacing (m).
        zmax : float, optional
            Minimum trailing range (m). The original range is retained unless
            capped by max_rows, and extensions beyond its bins are zero-filled.
        max_rows : int, optional
            Limit the row count (at least two) by truncating the range, never
            by coarsening dz.

        Returns
        -------
        BinnedWakefield
        """
        if not np.isfinite(dz) or dz <= 0:
            raise ValueError("dz must be finite and positive")
        if zmax is not None and (not np.isfinite(zmax) or zmax <= 0):
            raise ValueError("zmax must be finite and positive")
        if max_rows is not None and (
            not isinstance(max_rows, (int, np.integer)) or max_rows < 2
        ):
            raise ValueError("max_rows must be an integer of at least 2")
        extent = max(self.default_zmax, zmax or 0)
        outer_edge = (len(self._values) - 0.5) * self.dz
        count = max(
            2,
            int(np.ceil(extent / dz * (1 - 1e-9))) + 1,
            int(np.ceil((outer_edge / dz + 0.5) * (1 - 1e-9))),
        )
        if max_rows is not None and count > max_rows:
            count = max_rows
            if zmax is not None and (count - 1) * dz < zmax:
                warnings.warn(
                    f"The table is limited to {max_rows} rows, shorter than zmax; "
                    "the spacing is kept to resolve the self-kick.",
                    UserWarning,
                    stacklevel=2,
                )
        values = self._values.copy()
        if np.isclose(dz, self.dz, rtol=1e-9, atol=0):
            rebinned = np.zeros(count)
            retained = min(count, len(values))
            rebinned[:retained] = values[:retained]
        else:
            values[0] /= 2
            edges = np.r_[0.0, (np.arange(len(values)) + 0.5) * self.dz]
            integral = np.r_[0.0, np.cumsum(values) * self.dz]
            new_integral = np.interp((np.arange(count) + 0.5) * dz, edges, integral)
            rebinned = np.diff(new_integral, prepend=0.0) / dz
            rebinned[0] *= 2
        return type(self)(dz, rebinned)
