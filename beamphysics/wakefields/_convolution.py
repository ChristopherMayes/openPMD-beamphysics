"""Shared grid convolution and cloud-in-cell tracking."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ..units import c_light

# Bunch density beyond this many sigma is neglected (exp(-32) ~ 1e-14).
_GAUSSIAN_CUTOFF = 8.0


class _WakefieldConvolution:
    def convolve_density(
        self,
        density: np.ndarray,
        dz: float,
        offset: float = 0,
        include_self_kick: bool = True,
        plot: bool = False,
        ax=None,
        method: str = "auto",
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
            Charge density array (C/m). Index 0 is the tail (back),
            index n-1 is the head (front).
        dz : float
            Grid spacing (m).
        offset : float, optional
            Offset for the z coordinate (m). Default is 0.
        include_self_kick : bool, optional
            Whether to include the half self-kick term. Default is True.
        plot : bool, optional
            If True, plot the density profile and wake potential. Default is False.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None and plot=True, creates a new figure.
        method : str, optional
            How the wake is evaluated at each lag. 'sample' uses the point values
            W(-k dz), with W(0) halved for the self-kick. 'bin_average' uses
            `bin_averaged_wake`, the exact kick of a bin of uniform charge, which
            is required for a wake that is singular at z = 0. 'auto' (default)
            uses 'bin_average' when the model defines an `integrated_wake` and
            offset is 0, and 'sample' otherwise.

        Returns
        -------
        integrated_wake : np.ndarray
            Integrated longitudinal wakefield (V/m).

        Raises
        ------
        ValueError
            If method is 'sample' for a singular wake, or 'bin_average' with a
            nonzero offset.
        """
        from scipy.signal import correlate

        n = len(density)
        charge = density * dz  # Convert to charge per bin (C)

        W_causal = self._wake_kernel(dz, n, offset, method)

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
        current = density * c_light  # (A)
        ax2.fill_between((z - z0) * 1e6, current, alpha=0.3, color="C1")
        ax2.set_ylabel(r"Current (A)", color="C1")
        ax2.tick_params(axis="y", labelcolor="C1")
        ax2.set_ylim(0, None)

    def wake_potential(
        self, z: np.ndarray | float, sigma: float, dz: float | None = None
    ) -> np.ndarray | float:
        """
        Wake potential of a Gaussian bunch per unit bunch charge, on a grid.

        The bunch is sampled at spacing dz over the requested positions and
        +-8 sigma, convolved with `convolve_density`, and interpolated linearly
        to z. The result includes the error of binning at dz, which is what a
        tracking code binning at dz sees. Rows beyond the end of a table count as
        zero, so a table shorter than the bunch also shows up as an error.

        Parameters
        ----------
        z : float or np.ndarray
            Position relative to the bunch centre (m), with z < 0 towards the tail.
        sigma : float
            Rms bunch length (m).
        dz : float, optional
            Grid spacing (m). Default is sigma / 20.

        Returns
        -------
        V : float or np.ndarray
            Wake potential per unit charge (V/C/m). Positive for energy loss.
        """
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError(f"sigma must be finite and positive, got {sigma}")
        if dz is None:
            dz = sigma / 20
        if not np.isfinite(dz) or dz <= 0:
            raise ValueError(f"dz must be finite and positive, got {dz}")

        z = np.asarray(z, dtype=float)
        scalar_input = z.ndim == 0
        z = np.atleast_1d(z)

        z_lo = min(-_GAUSSIAN_CUTOFF * sigma, z.min(initial=0.0))
        z_hi = max(_GAUSSIAN_CUTOFF * sigma, z.max(initial=0.0))
        grid = z_lo + dz * np.arange(int(np.ceil((z_hi - z_lo) / dz)) + 1)
        density = np.exp(-0.5 * (grid / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)
        V = np.interp(z, grid, self.convolve_density(density, dz))

        if scalar_input:
            return float(V[0])
        return V

    def plot_wake_potential(
        self, sigmas, zlim: float = 5.0, n: int = 400, ax=None
    ) -> None:
        """
        Plot the wake potential of Gaussian bunches against z / sigma.

        Parameters
        ----------
        sigmas : float or sequence of float
            Rms bunch lengths (m), one curve each.
        zlim : float, optional
            Half-width of the plotted range in units of sigma. Default is 5.
        n : int, optional
            Number of points per curve. Default is 400.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates a new figure.
        """
        if ax is None:
            _, ax = plt.subplots()
        x = np.linspace(-zlim, zlim, n)
        for sigma in np.atleast_1d(sigmas):
            V = self.wake_potential(x * sigma, sigma)
            ax.plot(x, V * 1e-12, label=f"$\\sigma_z$ = {sigma * 1e6:g} µm")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel(r"$z/\sigma_z$")
        ax.set_ylabel(r"$V(z)$ (V/pC/m)")
        ax.legend()

    def binned_particle_kicks(
        self,
        z: np.ndarray,
        weight: np.ndarray,
        dz: float | None = None,
        include_self_kick: bool = True,
        method: str = "auto",
    ) -> np.ndarray:
        """
        Compute energy kicks per unit length through a charge deposit on a grid.

        The charge is deposited on a uniform grid by cloud-in-cell weighting, the
        wake potential is computed with `convolve_density`, and the result is
        interpolated linearly back to the particles. This is O(N + M log M) for N
        particles on M bins, and it is the only path for a wake that is singular
        at z = 0, whose point-to-point sum is dominated by the closest pairs.

        Parameters
        ----------
        z : np.ndarray
            Particle positions (m).
        weight : np.ndarray
            Particle charges (C).
        dz : float, optional
            Bin width (m). Defaults to the bunch length divided by max(100, N // 10).
        include_self_kick : bool, optional
            Whether to include the self-kick term. If False, the whole response of
            each particle to its own deposited charge is removed. Default is True.
        method : str, optional
            Passed to `convolve_density`. Default is 'auto'.

        Returns
        -------
        kicks : np.ndarray
            Energy kick per unit length at each particle (eV/m).
        """
        z = np.asarray(z, dtype=float)
        weight = np.asarray(weight, dtype=float)
        if z.shape != weight.shape or z.ndim != 1:
            raise ValueError(
                f"z and weight must be 1D arrays of the same shape, got "
                f"z.shape={z.shape}, weight.shape={weight.shape}"
            )

        z_min, z_max = z.min(), z.max()
        if dz is None:
            z_range = z_max - z_min
            if z_range == 0:
                z_range = 1e-6
            dz = z_range / max(100, len(z) // 10)
        if dz <= 0:
            raise ValueError(f"dz must be positive, got {dz}")

        # One empty bin on each side so the cloud-in-cell deposit never spills.
        z0 = z_min - dz
        x = (z - z0) / dz
        i = np.floor(x).astype(int)
        f = x - i
        n = int(i.max()) + 3

        density = np.zeros(n)
        np.add.at(density, i, weight * (1 - f))
        np.add.at(density, i + 1, weight * f)
        density /= dz

        V = self.convolve_density(density, dz, method=method)
        centers = z0 + dz * np.arange(n)

        # Negative sign: an energy-losing wake is positive.
        kicks = -np.interp(z, centers, V)

        if not include_self_kick:
            # Cloud-in-cell spreads a particle over two nodes, which kick each other.
            K0, K1 = self._wake_kernel(dz, 2, 0, method)
            kicks += weight * (0.5 * K0 * ((1 - f) ** 2 + f**2) + K1 * f * (1 - f))

        return kicks
