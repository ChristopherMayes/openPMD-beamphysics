import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def fit_m2(
    z, beam_sizes, wavelength, initial_guess=None, return_covariance=False, plot=False
):
    """
    Fit M² (beam quality factor) from RMS beam size measurements along z.

    The beam size evolution follows the hyperbolic relation:

        σ²(z) = σ₀² [1 + (M² λ (z - z₀) / (4π σ₀²))²]

    where:
        σ(z)  : RMS beam size at position z (m)
        σ₀    : Minimum RMS beam size at the waist (m)
        z₀    : Longitudinal position of the beam waist (m)
        M²    : Beam quality factor (M² = 1 for ideal Gaussian beam)
        λ     : Wavelength (m)

    The Rayleigh range for a real beam is z_R = 4π σ₀² / (M² λ).

    Parameters
    ----------
    z : array-like
        Propagation distances (m)
    beam_sizes : array-like
        RMS beam sizes (σ) at each z position (m)
    wavelength : float
        Wavelength of the beam (m)
    initial_guess : tuple, optional
        Initial guess for (σ₀, z₀, M²). Default: (min(beam_sizes), 0.0, 1.0)
    return_covariance : bool, optional
        If True, also return the covariance matrix. Default: False
    plot : bool, optional
        If True, produce a diagnostic plot. Default: False

    Returns
    -------
    dict
        Dictionary with fitted parameters:
        - 'sigma0': RMS beam waist size (m)
        - 'z0': waist position (m)
        - 'M2': beam quality factor
        - 'zR': Rayleigh range z_R = 4π σ₀² / (M² λ) (m)
        - 'pcov': covariance matrix (if return_covariance=True)
    """

    sigma2 = np.array(beam_sizes) ** 2
    z = np.array(z)

    def sigma_squared(z, sigma0, z0, M2):
        k = M2 * wavelength / (4 * np.pi * sigma0**2)
        return sigma0**2 * (1 + (k * (z - z0)) ** 2)

    if initial_guess is None:
        initial_guess = [min(beam_sizes), 0.0, 1.0]

    popt, pcov = curve_fit(sigma_squared, z, sigma2, p0=initial_guess)

    sigma0_fit, z0_fit, M2_fit = popt
    zR_fit = 4 * np.pi * sigma0_fit**2 / (M2_fit * wavelength)

    result = {
        "sigma0": sigma0_fit,
        "z0": z0_fit,
        "M2": M2_fit,
        "zR": zR_fit,
    }

    if return_covariance:
        result["pcov"] = pcov

    if plot:
        z_fit = np.linspace(min(z), max(z), 500)
        sigma2_fit = sigma_squared(z_fit, *popt)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(
            z, np.array(beam_sizes) * 1e6, label="Data", color="blue", marker="o"
        )
        ax.plot(z_fit, np.sqrt(sigma2_fit) * 1e6, label="Fit", color="red")
        ax.set_xlabel("z (m)")
        ax.set_ylabel(r"$\sigma$ (µm)")
        ax.set_title(
            rf"$M^2$ Fit: $\sigma_0$ = {sigma0_fit*1e6:.2f} µm, "
            rf"$z_0$ = {z0_fit:.3f} m, $M^2$ = {M2_fit:.3f}, "
            rf"$z_R$ = {zR_fit:.2f} m"
        )
        ax.legend()
        ax.grid(True)

    return result
