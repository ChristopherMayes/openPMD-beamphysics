from functools import partial
from math import factorial

import numpy as np
from scipy.integrate import quad
from scipy.optimize import curve_fit


def synthesize_field(
    x: np.ndarray | float,
    y: np.ndarray | float,
    multipoles: list[tuple[float, float]],
) -> tuple[np.ndarray | float, np.ndarray | float]:
    r"""
    Calculate transverse magnetic field components at position (x, y)
    from multipole coefficients.

    The magnetic field is computed from the complex multipole expansion:

    .. math::
        B_y + i B_x = \sum_{n=0}^{N} \frac{C_n}{n!} z^n

    where :math:`z = x + iy`, :math:`C_n = B_n + i S_n`, and the sum runs
    over the provided multipole coefficients.

    Parameters
    ----------
    x : float or np.ndarray
        Horizontal position in meters.
    y : float or np.ndarray
        Vertical position in meters.
    multipoles : list of tuple[float, float]
        Multipole coefficients as (B_n, S_n) pairs, where
        n=0 is dipole, n=1 is quadrupole, n=2 is sextupole, etc.
        B_n is the normal component and S_n is the skew component,
        both in units of T/m^n.

    Returns
    -------
    B_x : float or np.ndarray
        Horizontal magnetic field component in Tesla.
    B_y : float or np.ndarray
        Vertical magnetic field component in Tesla.
    """
    z = x + 1j * y

    B_complex = 0j
    for n, (B_n, S_n) in enumerate(multipoles):
        C_n = B_n + 1j * S_n
        B_complex = B_complex + (C_n / factorial(n)) * z**n

    B_y = np.real(B_complex)
    B_x = np.imag(B_complex)

    return B_x, B_y


def decompose_field(
    data: list[tuple[float, float]] | np.ndarray,
    r0: float,
    nmax: int,
) -> list[tuple[float, float]]:
    r"""
    Decompose azimuthal field measurements on a circle into multipole
    coefficients using a least-squares fit.

    Given measurements of the tangential field component
    :math:`B_\phi(\phi)` at radius :math:`r_0`, this function fits
    the model:

    .. math::
        B_\phi(\phi) = \sum_{n=0}^{n_\mathrm{max}} \frac{r_0^n}{n!}
        \left[ B_n \cos\!\left((n+1)\phi\right)
        - S_n \sin\!\left((n+1)\phi\right) \right]

    Parameters
    ----------
    data : list of tuple[float, float] or np.ndarray
        Measurement data as (phi, B_phi) pairs, where phi is
        the azimuthal angle in radians and B_phi is the tangential
        field component in Tesla.
    r0 : float
        Measurement radius in meters.
    nmax : int
        Maximum multipole order to fit. 0 = dipole, 1 = quadrupole,
        2 = sextupole, etc.

    Returns
    -------
    list of tuple[float, float]
        Multipole coefficients as (B_n, S_n) pairs for n = 0 to nmax.
        B_n is the normal component and S_n is the skew component,
        both in units of T/m^n.
    """
    data_array = np.array(data)
    phi = data_array[:, 0]
    B_phi_measured = data_array[:, 1]

    def model(phi_vals, *params):
        B_phi = np.zeros_like(phi_vals)
        for n in range(nmax + 1):
            B_n = params[2 * n]
            S_n = params[2 * n + 1]
            factor = (r0**n) / factorial(n)
            B_phi += factor * (
                B_n * np.cos((n + 1) * phi_vals)
                - S_n * np.sin((n + 1) * phi_vals)
            )
        return B_phi

    initial_params = np.zeros(2 * (nmax + 1))
    popt, _ = curve_fit(model, phi, B_phi_measured, p0=initial_params)

    multipoles = []
    for n in range(nmax + 1):
        B_n = popt[2 * n]
        S_n = popt[2 * n + 1]
        multipoles.append((B_n, S_n))

    return multipoles


def _integrand(
    phi: float,
    B_actual: callable,
    B_design: callable,
    r0: float,
) -> float:
    """Squared relative field error at azimuthal angle phi."""
    x = r0 * np.cos(phi)
    y = r0 * np.sin(phi)

    B_x_actual, B_y_actual = B_actual(x, y)
    B_x_design, B_y_design = B_design(x, y)

    B_design_mag_sq = B_x_design**2 + B_y_design**2
    error_mag_sq = (B_x_actual - B_x_design) ** 2 + (B_y_actual - B_y_design) ** 2

    return error_mag_sq / B_design_mag_sq


def scalar_error(
    B_actual: callable,
    B_design: callable,
    r0: float,
) -> float:
    r"""
    Compute the normalized RMS field error between two field
    distributions on a circle.

    Evaluates:

    .. math::
        \Delta B / B = \sqrt{
            \frac{1}{2\pi} \int_0^{2\pi}
            \frac{|\mathbf{B}_\mathrm{actual} - \mathbf{B}_\mathrm{design}|^2}
            {|\mathbf{B}_\mathrm{design}|^2}
            \, d\phi
        }

    Parameters
    ----------
    B_actual : callable
        Field function with signature ``(x, y) -> (B_x, B_y)``
        returning the actual field in Tesla.
    B_design : callable
        Field function with signature ``(x, y) -> (B_x, B_y)``
        returning the design field in Tesla.
    r0 : float
        Evaluation radius in meters.

    Returns
    -------
    float
        Normalized RMS field error (dimensionless).
    """
    integrand = partial(_integrand, B_actual=B_actual, B_design=B_design, r0=r0)
    integral_result, _ = quad(integrand, 0, 2 * np.pi)
    return np.sqrt(integral_result / (2 * np.pi))
