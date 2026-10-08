from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, NamedTuple, Tuple

import numpy as np
from scipy import stats as scipy_stats

if TYPE_CHECKING:
    pass


def norm_emit_calc(particle_group, planes=["x"]):
    """

    2d, 4d, 6d normalized emittance calc

    planes = ['x', 'y'] is the 4d emittance

    planes = ['x', 'y', 'z'] is the 6d emittance

    Momenta for each plane are takes as p+plane, e.g. 'px' for plane='x'

    The normalization factor is (1/mc)^n_planes, so that the units are meters^n_planes

    """

    dim = len(planes)
    vars = []
    for k in planes:
        vars.append(k)
        vars.append("p" + k)

    S = particle_group.cov(*vars)

    mc2 = particle_group.mass

    norm_emit = np.sqrt(np.linalg.det(S)) / mc2**dim

    return norm_emit


def twiss_calc(sigma_mat2):
    """
    Calculate Twiss parameters from the 2D sigma matrix (covariance matrix):
    sigma_mat = <x,x>   <x, p>
                <p, x>  <p, p>

    This is a simple calculation. Makes no assumptions about units.

    alpha = -<x, p>/emit
    beta  =  <x, x>/emit
    gamma =  <p, p>/emit
    emit = det(sigma_mat)

    """
    assert sigma_mat2.shape == (
        2,
        2,
    ), f"Bad shape: {sigma_mat2.shape}. This should be (2,2)"  # safety check
    twiss = {}
    emit = np.sqrt(np.linalg.det(sigma_mat2))
    twiss["alpha"] = -sigma_mat2[0, 1] / emit
    twiss["beta"] = sigma_mat2[0, 0] / emit
    twiss["gamma"] = sigma_mat2[1, 1] / emit
    twiss["emit"] = emit

    return twiss


def twiss_ellipse_points(sigma_mat2, n_points=36):
    """
    Returns points that will trace a the rms ellipse
    from a 2x2 covariance matrix `sigma_mat2`.

    Returns
    -------
    vec: np.ndarray with shape (2, n_points)
        x, p representing the ellipse points.

    """
    twiss = twiss_calc(sigma_mat2)
    A = A_mat_calc(twiss["beta"], twiss["alpha"])

    theta = np.linspace(0, np.pi * 2, n_points)
    zvec0 = np.array([np.cos(theta), np.sin(theta)]) * np.sqrt(2 * twiss["emit"])

    zvec1 = np.matmul(A, zvec0)
    return zvec1


def twiss_match(x, p, beta0=1, alpha0=0, beta1=1, alpha1=0):
    """
    Simple Twiss matching.

    Takes positions x and momenta p, and transforms them according to
    initial Twiss parameters:
        beta0, alpha0
    into final  Twiss parameters:
        beta1, alpha1

    This is simply the matrix ransformation:
        xnew  = (   sqrt(beta1/beta0)                  0                 ) . ( x )
        pnew    (  (alpha0-alpha1)/sqrt(beta0*beta1)   sqrt(beta0/beta1) )   ( p )


    Returns new x, p

    """
    m11 = np.sqrt(beta1 / beta0)
    m21 = (alpha0 - alpha1) / np.sqrt(beta0 * beta1)

    xnew = x * m11
    pnew = x * m21 + p / m11

    return xnew, pnew


def matched_particles(
    particle_group, beta=None, alpha=None, plane="x", p0c=None, inplace=False
):
    """
    Performs simple Twiss 'matching' by applying a linear transformation to
        x, px if plane == 'x', or y, py if plane == 'y'

    Returns a new ParticleGroup

    If inplace, a copy will not be made, and changes will be done in place.

    """

    assert plane in ("x", "y"), f"Invalid plane: {plane}"

    if inplace:
        P = particle_group
    else:
        P = particle_group.copy()

    if not p0c:
        p0c = P["mean_p"]

    # Use Bmad-style coordinates.
    # Get plane.
    if plane == "x":
        x = P.x
        p = P.px / p0c
    else:
        x = P.y
        p = P.py / p0c

    # Get current Twiss
    tx = twiss_calc(np.cov(x, p, aweights=P.weight))

    # If not specified, just fill in the current value.
    if alpha is None:
        alpha = tx["alpha"]
    if beta is None:
        beta = tx["beta"]

    # New coordinates
    xnew, pnew = twiss_match(
        x, p, beta0=tx["beta"], alpha0=tx["alpha"], beta1=beta, alpha1=alpha
    )

    # Set
    if plane == "x":
        P.x = xnew
        P.px = pnew * p0c
    else:
        P.y = xnew
        P.py = pnew * p0c

    return P


def twiss_dispersion_calc(sigma3):
    """
    Twiss and Dispersion calculation from a 3x3 sigma (covariance) matrix from particles
    x, p, delta

    Formulas from:
        https://uspas.fnal.gov/materials/19Knoxville/g-2/creation-and-analysis-of-beam-distributions.html

    Returns a dict with:
        alpha
        beta
        gamma
        emit
        eta
        etap

    """

    # Collect terms

    delta2 = sigma3[2, 2]
    xd = sigma3[0, 2]
    pd = sigma3[1, 2]

    eb = sigma3[0, 0] - xd**2 / delta2
    eg = sigma3[1, 1] - pd**2 / delta2
    ea = -sigma3[0, 1] + xd * pd / delta2

    emit = np.sqrt(eb * eg - ea**2)

    # Form the output dict
    d = {}

    d["alpha"] = ea / emit
    d["beta"] = eb / emit
    d["gamma"] = eg / emit
    d["emit"] = emit
    d["eta"] = xd / delta2
    d["etap"] = pd / delta2

    return d


def particle_twiss_dispersion(particle_group, plane="x", fraction=1, p0c=None):
    """
    Twiss and Dispersion calc for a ParticleGroup.

    Plane muse be:
        'x' or 'y'

    p0c is the reference momentum. If not give, the mean p will be used.

    Returns the same output dict as twiss_dispersion_calc, but with keys suffixed with the plane, i.e.:

        alpha_x
        beta_x
        gamma_x
        emit_x
        eta_x
        etap_x
        norm_emit_x

    """

    assert plane in ["x", "y"]

    P = particle_group  # convenience

    if fraction < 1:
        P = P[np.argsort(P[f"J{plane}"])][0 : int(fraction * len(P))]

    if not p0c:
        p0c = P["mean_p"]

    x = P[plane]
    xp = P["p" + plane] / p0c
    delta = P["p"] / p0c  # - 1

    # Form weighted covariance matrix
    sigma = np.cov([x, xp, delta], aweights=P.weight)

    # Actual calc
    twiss = twiss_dispersion_calc(sigma)

    # Add norm
    twiss["norm_emit"] = twiss["emit"] * p0c / P.mass

    # Add suffix
    out = {}
    for k in twiss:
        out[k + f"_{plane}"] = twiss[k]

    return out


# Linear Normal Form in 1 phase space plane.
# TODO: more advanced analysis e.g. Forest or Wolski or Sagan and Rubin or Ehrlichman.


def A_mat_calc(beta, alpha, inverse=False):
    """
    Returns the 1D normal form matrix from twiss parameters beta and alpha

        A =   sqrt(beta)         0
             -alpha/sqrt(beta)   1/sqrt(beta)

    If inverse, the inverse will be returned:

        A^-1 =  1/sqrt(beta)     0
                alpha/sqrt(beta) sqrt(beta)

    This corresponds to the linear normal form decomposition:

        M = A . Rot(theta) . A^-1

    with a clockwise rotation matrix:

        Rot(theta) =  cos(theta) sin(theta)
                     -sin(theta) cos(theta)

    In the Bmad manual, G_q (Bmad) = A (here) in the Linear Optics chapter.

    A^-1 can be used to form normalized coordinates:
        x_bar, px_bar   = A^-1 . (x, px)

    """
    a11 = np.sqrt(beta)
    a22 = 1 / a11
    a21 = -alpha / a11

    if inverse:
        return np.array([[a22, 0], [-a21, a11]])
    else:
        return np.array([[a11, 0], [a21, a22]])


def amplitude_calc(x, p, beta=1, alpha=0):
    """
    Simple amplitude calculation of position and momentum coordinates
    relative to twiss beta and alpha.

    J = (gamma x^2 + 2 alpha x p + beta p^2)/2

      = (x_bar^2 + px_bar^2)/ 2

    where gamma = (1+alpha^2)/beta

    """
    return (1 + alpha**2) / beta / 2 * x**2 + alpha * x * p + beta / 2 * p**2


def particle_amplitude(particle_group, plane="x", twiss=None, mass_normalize=True):
    """
    Returns the normalized amplitude array from a ParticleGroup for a given plane.

    Plane should be:
        'x' for the x, px plane
        'y' for the y, py plane
    Other planes will work, but please check that the units make sense.

    If mass_normalize (default=True), the momentum will be divided by the mass, so that the units are sqrt(m).

    See: normalized_particle_coordinate
    """
    x = particle_group[plane]
    key2 = "p" + plane

    if mass_normalize:
        # Note: do not do /=, because this will replace the ParticleGroup's internal array!
        p = particle_group[key2] / particle_group.mass
    else:
        p = particle_group[key2]

    # User could supply twiss
    if not twiss:
        sigma_mat2 = np.cov(x, p, aweights=particle_group.weight)
        twiss = twiss_calc(sigma_mat2)

    J = amplitude_calc(x, p, beta=twiss["beta"], alpha=twiss["alpha"])

    return J


def normalized_particle_coordinate(
    particle_group, key, twiss=None, mass_normalize=True
):
    """
    Returns a single normalized coordinate array from a ParticleGroup

    Position or momentum is determined by the key.
    If the key starts with 'p', it is a momentum, else it is a position,
    and the

    Intended use is for key to be one of:
        x, px, y py

    and the corresponding normalized coordinates are named with suffix _bar, i.e.:
        x_bar, px_bar, y_bar, py_bar

    If mass_normalize (default=True), the momentum will be divided by the mass, so that the units are sqrt(m).

    These are related to action-angle coordinates
        J: amplitude
        phi: phase

        x_bar =  sqrt(2 J) cos(phi)
        px_bar = sqrt(2 J) sin(phi)

    So therefore:
        J = (x_bar^2 + px_bar^2)/2
        phi = arctan(px_bar/x_bar)
    and:
        <J> = norm_emit_x

     Note that the center may need to be subtracted in this case.

    """

    # Parse key for position or momentum coordinate
    if key.startswith("p"):
        momentum = True
        key1 = key[1:]
        key2 = key

    else:
        momentum = False
        key1 = key
        key2 = "p" + key

    x = particle_group[key1]

    if mass_normalize:
        # Note: do not do /=, because this will replace the ParticleGroup's internal array!
        p = particle_group[key2] / particle_group.mass
    else:
        p = particle_group[key2]

    # User could supply twiss
    if not twiss:
        sigma_mat2 = np.cov(x, p, aweights=particle_group.weight)
        twiss = twiss_calc(sigma_mat2)

    A_inv = A_mat_calc(twiss["beta"], twiss["alpha"], inverse=True)

    if momentum:
        return A_inv[1, 0] * x + A_inv[1, 1] * p
    else:
        return A_inv[0, 0] * x


# ---------------
# Other utilities


def slice_statistics(particle_group, keys=["mean_z"], n_slice=40, slice_key=None):
    """
    Slices a particle group into n slices and returns statistics from each sliced defined in keys.

    These statistics should be scalar floats for now.

    Any key can be used to slice on.

    """

    if slice_key is None:
        if particle_group.in_t_coordinates:
            slice_key = "z"
        else:
            slice_key = "t"

    sdat = {}
    twiss_planes = set()
    twiss = {}

    normal_keys = []

    for k in keys:
        if k.startswith("twiss"):
            if k == "twiss" or k == "twiss_xy":
                twiss_planes.add("x")
                twiss_planes.add("y")
            else:
                plane = k[-1]  #
                assert plane in ("x", "y")
                twiss_planes.add(plane)
        else:
            sdat[k] = np.empty(n_slice)
            normal_keys.append(k)

    twiss_plane = "".join(twiss_planes)  # flatten
    assert twiss_plane in ("x", "y", "xy", "yx", "")

    for i, pg in enumerate(particle_group.split(n_slice, key=slice_key)):
        if normal_keys:
            for k, value in pg.statistics(*normal_keys).items():
                sdat[k][i] = value

        # Handle twiss
        if twiss_plane:
            twiss = pg.twiss(plane=twiss_plane)
            for k in twiss:
                full_key = f"twiss_{k}"
                if full_key not in sdat:
                    sdat[full_key] = np.empty(n_slice)
                sdat[full_key][i] = twiss[k]

    return sdat


def resample_particles(particle_group, n=0, equal_weights=False):
    """
    Resamples a ParticleGroup randomly.

    If n equals particle_group.n_particle or n=0,
    particle indices will be scrambled.

    Otherwise if weights are equal, a random subset of particles will be selected.

    Otherwise if weights are not equal, particles will be sampled according to their weight using
    [scipy.stats.rv_discrete](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_discrete.html).
    Note that this latter method can result in duplicate particles, and can be very slow for a large number of particles.

    Parameters
    ----------
    n: int, default = 0
        Number to resample.
        If n = 0, this will use all particles.

    equal_weights: bool, default = False
        If True, will ensure that all particles have equal weights.

    Returns
    -------
    data: dict of ParticleGroup data

    """
    n_old = particle_group.n_particle
    if n == 0:
        n = n_old

    if n > n_old:
        raise ValueError(f"Cannot supersample {n_old} to {n}")

    weight = particle_group.weight

    # Equal weights
    if len(set(particle_group.weight)) == 1:
        ixlist = np.random.choice(n_old, n, replace=False)
        weight = np.full(n, particle_group.charge / n)

    # variable weights found
    elif equal_weights or n != n_old:
        # From SciPy example:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_discrete.html#scipy.stats.rv_discrete
        pk = weight / np.sum(weight)  # Probabilities
        xk = np.arange(len(pk))  # index
        ixsampler = scipy_stats.rv_discrete(name="ixsampler", values=(xk, pk))
        ixlist = ixsampler.rvs(size=n)
        weight = np.full(n, particle_group.charge / n)

    else:
        assert n == n_old, f"Internal error: expected n == n_old, got {n} != {n_old}"
        ixlist = np.random.choice(n_old, n, replace=False)
        weight = weight[ixlist]  # just scramble

    data = {}
    for key in particle_group._settable_array_keys:
        data[key] = particle_group[key][ixlist]
    data["species"] = particle_group["species"]
    data["weight"] = weight

    return data


def bunching(z: np.ndarray, wavelength: float, weight: np.ndarray = None) -> complex:
    r"""
    Calculate the normalized bunching parameter, which is the
    complex sum of weighted exponentials.

    The formula for bunching is given by:

    $$
    B(z, \lambda) = \frac{\sum w_i e^{i k z_i}}{\sum w_i}
    $$

    where:
    - $z$ is the position array,
    - $\lambda$ is the wavelength,
    - $k = \frac{2\pi}{\lambda}$  is the wave number,
    - $w_i$ are the weights.

    Parameters
    ----------
    z : np.ndarray
        Array of positions where the bunching parameter is calculated.
    wavelength : float
        Wavelength of the wave.
    weight : np.ndarray, optional
        Weights for each exponential term. Default is 1 for all terms.

    Returns
    -------
    complex
        The bunching parameter

    Raises
    ------
    ValueError
        If `wavelength` is not a positive number.
    """
    if wavelength <= 0:
        raise ValueError("Wavelength must be a positive number.")

    if weight is None:
        weight = np.ones(len(z))
    if len(weight) != len(z):
        raise ValueError(
            f"Weight array has length {len(weight)} != length of the z array, {len(z)}"
        )

    k = 2 * np.pi / wavelength
    f = np.exp(1j * k * z)
    return np.sum(weight * f) / np.sum(weight)


def mean_calc(x: np.ndarray, weight: np.ndarray) -> float:
    """
    Calculate the weighted mean.
    """
    P = weight / np.sum(weight)
    return np.sum(x * P)


def mean_variance_calc(x: np.ndarray, weight: np.ndarray) -> Tuple[float, float]:
    """
    Calculate the weighted mean and variance.

    """
    P = weight / np.sum(weight)
    mean = np.sum(x * P)
    variance = np.sum((x - mean) ** 2 * P)
    return mean, variance


def standard_deviation_calc(x: np.ndarray, weight: np.ndarray) -> float:
    return np.sqrt(mean_variance_calc(x, weight)[1])


class StatisticOperator(str, Enum):
    """
    An operation in a statistic key, such as `mean` in `mean_x`.

    All but `COV` take a single array and must match the operators of the
    statistics standard. Synchronization is checked in the test suite.
    """

    MEAN = "mean"
    SIGMA = "sigma"
    MIN = "min"
    MAX = "max"
    PTP = "ptp"
    DELTA = "delta"
    COV = "cov"

    @property
    def prefix(self) -> str:
        """The key prefix, e.g. `mean_`."""
        return f"{self.value}_"

    @property
    def n_arrays(self) -> int:
        """The number of arrays the operation applies to."""
        return 2 if self is StatisticOperator.COV else 1

    @property
    def preserves_dtype(self) -> bool:
        """Whether the result has the dtype of its array, rather than float."""
        return self in (
            StatisticOperator.MIN,
            StatisticOperator.MAX,
            StatisticOperator.PTP,
        )

    def _check_names(self, names: tuple[str, ...], given: str) -> None:
        if len(names) != self.n_arrays or not all(names):
            example = self.prefix + "__".join(("x", "px")[: self.n_arrays])
            raise ValueError(
                f"{self.value!r} statistic keys need exactly {self.n_arrays} "
                f"nonempty array name(s), as in {example!r}: {given!r}"
            )

    def key(self, *names: str) -> str:
        """
        Make the statistic key for this operation on arrays.

        Parameters
        ----------
        *names : str
            The array names, `n_arrays` of them.

        Returns
        -------
        str
            The key, e.g. `mean_x` or `cov_x__px`.

        Raises
        ------
        ValueError
            If the wrong number of names is given, or any is empty.
        """
        self._check_names(names, given=repr(names))
        return self.prefix + "__".join(names)

    def parse(self, key: str) -> tuple[str, ...] | None:
        """
        Get the array names of a statistic key for this operation.

        The inverse of `key`.

        Parameters
        ----------
        key : str
            A statistic key, e.g. `mean_x` or `cov_x__px`.

        Returns
        -------
        tuple of str, or None
            The array names, or None if `key` does not start with `prefix`.

        Raises
        ------
        ValueError
            If `key` starts with `prefix` but has the wrong number of array
            names, or any is empty, as in `cov_x` or `mean_`.
        """
        if not key.startswith(self.prefix):
            return None
        rest = key.removeprefix(self.prefix)
        names = tuple(rest.split("__")) if self.n_arrays > 1 else (rest,)
        self._check_names(names, given=key)
        return names


class TwissParameter(str, Enum):
    """
    A Twiss or dispersion parameter in a statistic key, such as `beta` in `twiss_beta_x`.

    These are the keys of `particle_twiss_dispersion` for a plane, in order.
    """

    alpha = "alpha"
    beta = "beta"
    gamma = "gamma"
    emit = "emit"
    norm_emit = "norm_emit"
    eta = "eta"
    etap = "etap"

    def key(self, plane: str) -> str:
        """The statistic key for this parameter in a plane, e.g. `twiss_beta_x`."""
        return f"twiss_{self.value}_{plane}"

    @classmethod
    def parse(cls, key: str) -> tuple[TwissParameter, str] | None:
        """
        Split a Twiss statistic key into its parameter and plane.

        The inverse of `key`.

        Parameters
        ----------
        key : str
            A statistic key, e.g. `twiss_beta_x`.

        Returns
        -------
        tuple of (TwissParameter, str), or None
            The parameter and plane (`x` or `y`), or None if `key` is not a
            Twiss statistic key.
        """
        name, _, plane = key.removeprefix("twiss_").rpartition("_")
        if not key.startswith("twiss_") or plane not in ("x", "y"):
            return None
        try:
            return cls(name), plane
        except ValueError:
            return None


class StatisticKey(NamedTuple):
    """
    A statistic key split into its operation and array names.

    Attributes
    ----------
    op : StatisticOperator
        The operation.
    names : tuple of str
        The array names the operation applies to: two for `cov`, one otherwise.
    """

    op: StatisticOperator
    names: tuple[str, ...]

    @classmethod
    def from_string(cls, key: str) -> StatisticKey | None:
        """
        Split a statistic key into its operation and array names.

        This is the prefix handling of `ParticleGroup.__getitem__`.

        Parameters
        ----------
        key : str
            A statistic key, e.g. `cov_x__px` or `sigma_x`.

        Returns
        -------
        StatisticKey or None
            The operation and array names, e.g.
            `StatisticKey(StatisticOperator.COV, ("x", "px"))` for `cov_x__px`
            and `StatisticKey(StatisticOperator.SIGMA, ("x",))` for `sigma_x`.
            None for keys that are not an operation on arrays.

        Raises
        ------
        ValueError
            For malformed keys with an operator prefix, such as `cov_x` or
            `mean_`.
        """
        for op in StatisticOperator:
            names = op.parse(key)
            if names is not None:
                return cls(op, names)
        return None
