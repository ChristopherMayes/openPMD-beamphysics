import numpy as np
from scipy import interpolate
from scipy.integrate import solve_ivp
from scipy.optimize import brent, brentq

from pmd_beamphysics.species import charge_state, mass_of
from pmd_beamphysics.units import c_light, mec2

# Numpy migration per https://numpy.org/doc/stable/numpy_2_0_migration_guide.html
if np.lib.NumpyVersion(np.__version__) >= "2.0.0":
    from numpy import trapezoid
else:
    # Support 'trapz' from numpy 1.0
    from numpy import trapz as trapezoid

from matplotlib import pyplot as plt


# ----------------------
# Analysis


def accelerating_voltage_and_phase(z, Ez, frequency):
    r"""
    Computes the accelerating voltage and phase for a v=c positively charged particle in an accelerating cavity field.

        Z = \int Ez * e^{-i k z} dz

        where k = omega/c = 2*pi*frequency/c

        voltage = abs(Z)
        phase = arg(Z)

    Input:
        z  (float array):   z-coordinate array (m)
        Ez (complex array): On-axis complex Ez field array (V/m), oscillating as exp(-i omega t), with omega = 2*pi*frequency

    Output:
        voltage, phase in V, radian

    """

    omega = 2 * np.pi * frequency
    k = omega / c_light
    fz = Ez * np.exp(-1j * k * z)

    # Integrate
    Z = trapezoid(fz, z)

    # Max voltage at phase
    voltage = np.abs(Z)
    phase = np.angle(Z)

    return voltage, phase


def track_field_1d(
    z,
    Ez,
    frequency=0,
    z0=0,
    pz0=0,
    t0=0,
    mc2=mec2,  # electron
    q0=-1,
    debug=False,
    max_step=None,
):
    r"""
    Tracks a particle in a 1d complex electric field Ez, oscillating as Ez * exp(-i omega t)

    Uses scipy.integrate.solve_ivp to track the particle.

    Equations of motion:

    $ \frac{dz}{dt} = \frac{pc}{\sqrt{(pc)^2 + m^2 c^4)}} c $

    $ \frac{dp}{dt} = q E_z $

    $ E_z = \Re f(z) \exp(-i \omega t) $


    Parameters
    ----------
    z : array_like
        positions of the field Ez (m)

    Ez : array_like, complex
        On-axis longitudinal electric field (V/m)

    frequency : float
        RF frequency in Hz

    z0 :  float, optional = 0
        initial particle position (m)

    pz0 : float, optional = 0
        initial particle momentum (eV/c)

    t0 : float, optional = 0
        initial particle time (s)

    mc2 : float, optional = mec2
        initial particle mass (eV)

    q0 : float, optional = -1
        initial particle charge (e) (= -1 for electron)

    max_step: float, optional = None
        Maximum timestep for solve_ivp (s)
        None => max_step = 1/(2*frequency)
        Fields typically have variation over a wavelength,
        so this should help avoid noisy solutions.

    debug, bool, optional = False
        If True, Will return the full solution to solve_ivp

    Returns
    -------
    z1 : float
        final z position in (m)

    pz1 : float:
        final particle momemtum (eV/c)

    t1 : float
        final time (s)


    """

    # Make interpolating function
    field = interpolate.interp1d(z, Ez * q0 * c_light, fill_value="extrapolate")
    zmax = z.max()
    tmax = 100 / frequency
    omega = 2 * np.pi * frequency

    # function to integrate
    def fun(t, y):
        z = y[0]
        p = y[1]
        zdot = p / np.hypot(p, mc2) * c_light
        pdot = np.real(field(z) * np.exp(-1j * omega * t))
        return np.array([zdot, pdot])

    # Events (stopping conditions)
    def went_backwards(t, y):
        return y[0]

    went_backwards.terminal = True
    went_backwards.direction = -1

    def went_max(t, y):
        return y[0] - zmax

    went_max.terminal = True
    went_max.direction = 1

    if max_step is None:
        max_step = 1 / (10 * frequency)

    # Solve
    y0 = np.array([z0, pz0])
    sol = solve_ivp(
        fun,
        (t0, tmax),
        y0,
        first_step=1 / frequency / 1000,
        events=[went_backwards, went_max],
        # vectorized=True,   # Make it slower?
        method="RK45",
        max_step=max_step,
    )
    #      max_step=1/frequency/20)

    if debug:
        return sol

    # Final z, p, t
    zf = sol.y[0][-1]
    pf = sol.y[1][-1]
    tf = sol.t[-1]

    return zf, pf, tf


def track_field_1df(
    Ez_f,
    zstop=0,
    tmax=0,
    z0=0,
    pz0=0,
    t0=0,
    mc2=mec2,  # electron
    q0=-1,
    debug=False,
    max_step=None,
    method="RK23",
):
    r"""
    Similar to track_field_1d, execpt uses a function Ez_f

    Tracks a particle in a 1d electric field Ez(z, t)

    Uses scipy.integrate.solve_ivp to track the particle.

    Equations of motion:

    $ \frac{dz}{dt} = \frac{pc}{\sqrt{(pc)^2 + m^2 c^4)}} c $

    $ \frac{dp}{dt} = q E_z $

    $ E_z = \Re f(z) \exp(-i \omega t) $


    Parameters
    ----------


    Ez_f : callable
        Ez_f(z, t) callable with two arguments z (m) and t (s)
        On-axis longitudinal electric field (V/m)

    zstop : float
        z stopping position (m)

    tmax: float
        maximum timestep (s)

    z0 :  float, optional = 0
        initial particle position (m)

    pz0 : float, optional = 0
        initial particle momentum (eV/c)

    t0 : float, optional = 0
        initial particle time (s)

    mc2 : float, optional = mec2
        initial particle mass (eV)

    q0 : float, optional = -1
        initial particle charge (e) (= -1 for electron)

    max_step: float, optional = None
        Maximum timestep for solve_ivp (s)
        None => max_step = tmax/10
        Fields typically have variation over a wavelength,
        so this should help avoid noisy solutions.

    debug, bool, optional = False
        If True, Will return the full solution to solve_ivp

    Returns
    -------
    z1 : float
        final z position in (m)

    pz1 : float:
        final particle momemtum (eV/c)

    t1 : float
        final time (s)


    """

    # function to integrate
    def fun(t, y):
        z = y[0]
        p = y[1]
        zdot = p / np.hypot(p, mc2) * c_light
        pdot = Ez_f(z, t) * q0 * c_light
        return np.array([zdot, pdot])

    # Events (stopping conditions)
    def went_backwards(t, y):
        return y[0]

    went_backwards.terminal = True
    went_backwards.direction = -1

    def went_max(t, y):
        return y[0] - zstop

    went_max.terminal = True
    went_max.direction = 1

    if max_step is None:
        max_step = tmax / 10

    # Solve
    y0 = np.array([z0, pz0])
    sol = solve_ivp(
        fun,
        (t0, tmax),
        y0,
        first_step=tmax / 1000,
        events=[went_backwards, went_max],
        # vectorized=True,   # Make it slower?
        method=method,
        max_step=max_step,
    )
    if debug:
        return sol

    # Final z, p, t
    zf = sol.y[0][-1]
    pf = sol.y[1][-1]
    tf = sol.t[-1]

    return zf, pf, tf


def autophase_field(
    field_mesh, pz0=0, scale=1, species="electron", tol=1e-9, verbose=False, debug=False
):
    """
    Finds the maximum accelerating of a FieldMesh by tracking a particle and using Brent's method from scipy.optimize.brent.

    NOTE: Assumes that field_mesh.Ez[0,0,:] is the on-axis Ez field.
    TODO: generalize

    Parameters
    ----------
    fieldmesh : FieldMesh object

    pz0 : float, optional = 0
        initial particle momentum in the z direction, in eV/c
        pz = 0 is a particle at rest.
    scale : float, optional = 1
        Additional field scale.
    species : str, optional = 'electron'
        species to track.
    tol : float, optional = 1e-9
        Tolerence for brent: Stop if between iteration change is less than tol.

    debug : bool, optional = False
        If true, will return a function that tracks the field at a given phase in deg.
    verbose : bool, optional = False
        If true, prints information about the v=c voltage and phase for the initial guess, and function call information.

    Returns
    -------
    phase : float
        Maximum accelerating phase in deg
    pz1 : float
        Final particle momentum in the z direction, in eV/c

    """

    # Get field on-axis
    z = field_mesh.coord_vec("z")
    Ez = field_mesh.Ez[0, 0, :] * scale
    frequency = field_mesh.frequency
    zmin = z.min()

    # Get mass and charge state
    mc2 = mass_of(species)
    q0 = charge_state(species)  # -1 for electrons

    # Function for use in brent
    def phase_f(phase_deg):
        zf, pf, _ = track_field_1d(
            z,
            Ez,
            frequency=frequency,
            z0=zmin,
            pz0=pz0,
            t0=phase_deg / 360 / frequency,
            mc2=mc2,
            max_step=1 / frequency / 10,
            q0=q0,
        )
        return pf

    if debug:
        return phase_f

    # Get a quick estimate, to use in the bracket
    voltage0, phase0 = accelerating_voltage_and_phase(z, q0 * Ez, frequency)
    phase0_deg = phase0 * 180 / np.pi
    if verbose:
        print(f"v=c voltage: {voltage0} V, phase: {phase0_deg} deg")

    alg_sign = -1
    phase_range = [phase0_deg - 90, phase0_deg + 90]
    phase1_deg, pz1, iter, funcalls = brent(
        lambda x: alg_sign * phase_f(x % 360),
        brack=phase_range,
        maxiter=250,
        tol=tol,
        full_output=True,
    )
    if verbose:
        print(f"    iterations: {iter}")
        print(f"    function calls: {funcalls}")

    return phase1_deg % 360, alg_sign * pz1


def autophase_and_scale_field(
    field_mesh, voltage, pz0=0, species="electron", debug=False, verbose=False
):
    """
    Finds the maximum accelerating of a FieldMesh.

    Uses two iterations of phasing, scaling.

    Parameters
    ----------
    fieldmesh : FieldMesh object

    voltage : float
        Desired on-crest voltage in V

    pz0 : float, optional = 0
        initial particle momentum in the z direction, in eV/c
        pz = 0 is a particle at rest.

    species : str, optional = 'electron'
        species to track.

    debug : bool, optional = False
        If true, will return a function that tracks the field at a given phase and scale.

    verbose : bool, optional = False
        If true, prints information about the v=c voltage and phase for the initial guess, and function call information.

    Returns
    -------
    phase : float
        Maximum accelerating phase in deg
    scale : float
        scale factor for the field

    """

    z = field_mesh.coord_vec("z")
    Ez = field_mesh.Ez[0, 0, :]
    frequency = field_mesh.frequency
    zmin = z.min()

    # Get mass and charge
    mc2 = mass_of(species)
    q0 = charge_state(species)
    energy0 = np.hypot(pz0, mc2)

    # Get and initial estimate
    voltage0, phase0 = accelerating_voltage_and_phase(z, q0 * Ez, frequency)
    # convert to deg
    phase0 = phase0 * 180 / np.pi
    scale0 = voltage / voltage0
    if verbose:
        print(f"v=c voltage: {voltage0} V, phase: {phase0} deg")

    def phase_scale_f(phase_deg, scale):
        zf, pf, _ = track_field_1d(
            z,
            Ez * scale,
            frequency=frequency,
            z0=zmin,
            pz0=pz0,
            t0=phase_deg / 360 / frequency,
            mc2=mc2,
            max_step=1 / frequency / 10,
            q0=q0,
        )

        delta_energy = np.hypot(pf, mc2) - energy0

        return delta_energy

    if debug:
        return phase_scale_f

    # Phase 1
    brack = [phase0 - 90, phase0 + 90]
    phase1 = (
        brent(
            lambda x: -phase_scale_f(x, scale0),
            brack=brack,
            maxiter=250,
            tol=1e-6,
            full_output=False,
        )
        % 360
    )

    # Scale 1
    s0 = scale0 * 0.9
    s1 = scale0 * 1.1
    scale1 = brentq(
        lambda x: phase_scale_f(phase1, x) / voltage - 1.0,
        s0,
        s1,
        maxiter=20,
        rtol=1e-6,
        full_output=False,
    )

    if verbose:
        print(
            f"    Pass 1 delta energy: {phase_scale_f(phase1, scale1)} at phase  {phase1} deg"
        )

    # Phase 2
    brack = [phase1 - 10, phase1 + 10]
    phase2 = (
        brent(
            lambda x: -phase_scale_f(x, scale1),
            brack=brack,
            maxiter=250,
            tol=1e-9,
            full_output=False,
        )
        % 360
    )

    # Scale 2
    s0 = scale1 * 0.9
    s1 = scale1 * 1.1
    scale2 = brentq(
        lambda x: phase_scale_f(phase2, x) / voltage - 1.0,
        s0,
        s1,
        maxiter=20,
        rtol=1e-9,
        full_output=False,
    )

    if verbose:
        print(
            f"    Pass 2 delta energy: {phase_scale_f(phase2, scale2)} at phase  {phase2} deg"
        )

    return phase2, scale2


# Checking Maxwell Equations:
def check_static_div_equation(FM, plot=False, rtol=1e-4, **kwargs):
    """
    Checks the static divergence equation based on the geometry of the field mesh.

    This function verifies the static divergence condition, `∇·E = 0` for electric fields or `∇·B = 0`
    for magnetic fields, in the specified geometry of the field mesh. It automatically selects the appropriate
    divergence check function for either cylindrical or rectangular geometries.

    Parameters
    ----------
    FM : FieldMesh
        Object containing the static electric or magnetic field data as well as spatial coordinates and
        geometry information. Must specify a geometry via the `geometry` attribute, which can be either
        "cylindrical" or "rectangular". Must also contain an `is_static` attribute.
    plot : bool, default=False
        If True, plots the components of the divergence equation and the divergence error (if `plot_diff`
        is also True in `kwargs`).
    rtol : float, default=1e-4
        The relative tolerance for verifying that the divergence is zero. If the mean error is below
        this threshold, the function returns True, indicating that the static divergence condition holds.
    **kwargs : dict, optional
        Additional keyword arguments to be passed to the respective geometry-based function
        (`check_static_div_equation_cylindrical` or `check_static_div_equation_cartesian`). This may include
        specific index positions (e.g., `ir`, `ix`, `iy`) or options to plot the difference between
        divergence components.

    Returns
    -------
    bool
        True if the mean relative error of the computed divergence is below the specified `rtol` threshold,
        indicating that the static divergence condition is satisfied; False otherwise.

    Raises
    ------
    AssertionError
        If `FM.is_static` is False, indicating that non-static fields were provided.
    ValueError
        If the `geometry` attribute of `FM` is not "cylindrical" or "rectangular".

    Notes
    -----
    This function serves as a wrapper that checks the geometry type of the `FieldMesh` (FM) and then
    calls the appropriate divergence check function based on that geometry.

    Examples
    --------
    >>> check_static_div_equation(FM, plot=True, rtol=1e-5, ix=10, iy=5)

    This call checks the static divergence condition in FM's specified geometry, with a stricter tolerance
    of 1e-5, and plots the divergence components and difference if available for that geometry.

    """

    assert FM.is_static, "Must provide a static FieldMesh"

    if FM.geometry == "cylindrical":
        return check_static_div_equation_cylindrical(FM, plot=plot, rtol=rtol, **kwargs)
    elif FM.geometry == "rectangular":
        return check_static_div_equation_cartesian(FM, plot=plot, rtol=rtol, **kwargs)

    else:
        raise ValueError("Unknown FieldMesh geometry")


def check_static_div_equation_cartesian(
    FM, ix=None, iy=None, plot=False, rtol=1e-4, plot_diff=True
):
    """
    Checks the static divergence equation in Cartesian coordinates for either electric or magnetic fields.

    This function verifies the static divergence condition, `∇·E = 0` for electric fields or `∇·B = 0`
    for magnetic fields, in a Cartesian coordinate system. It calculates the divergence components based
    on the spatial derivatives of the field components along each axis and evaluates them at specified
    x and y indices. Optionally, it plots the divergence components and the difference between them.

    Parameters
    ----------
    FM : FieldMesh
        Object containing the static electric or magnetic field data in Cartesian coordinates.
        Must have `is_static`, `geometry`, `is_pure_electric`, and `is_pure_magnetic` attributes
        to determine the field type and properties.
    ix : int, optional
        Index for the x-coordinate at which to evaluate the fields. If None, the function selects
        the index closest to the origin (x=0).
    iy : int, optional
        Index for the y-coordinate at which to evaluate the fields. If None, the function selects
        the index closest to the origin (y=0).
    plot : bool, default=False
        If True, plots the components of the divergence equation and the divergence error (if `plot_diff`
        is also True).
    rtol : float, default=1e-4
        The relative tolerance for verifying that the divergence is zero. If the mean error is below
        this threshold, the function returns True, indicating that the static divergence condition holds.
    plot_diff : bool, default=True
        If True and `plot` is also True, plots the difference between the x-y and z divergence
        components on a secondary y-axis.

    Returns
    -------
    bool
        True if the mean relative error of the computed divergence is below the specified `rtol` threshold,
        indicating that the static divergence condition is satisfied; False otherwise.

    Raises
    ------
    AssertionError
        If `FM.is_static` is False, indicating that non-static fields were provided.
        If `FM.geometry` is not "rectangular", indicating that a non-rectangular geometry was provided.
    ValueError
        If the field type in `FM` is mixed (contains both electric and magnetic components), which is
        invalid for this test.

    Notes
    -----
    The function computes divergence components as:
    - x-y component: `∂F_x / ∂x + ∂F_y / ∂y`
    - z component: `∂F_z / ∂z`
    where `F_x`, `F_y`, and `F_z` represent either the x, y, and z components of the electric or magnetic
    field, based on the type of field present in `FM`.

    Examples
    --------
    >>> check_static_div_equation_cartesian(FM, ix=10, iy=5, plot=True, rtol=1e-5)

    This call checks the static divergence condition at the specified x and y indices, with a stricter
    tolerance of 1e-5, and plots the divergence components and difference.

    """

    assert FM.is_static, "Must provide static FieldMesh"
    assert (
        FM.geometry == "rectangular"
    ), "Must provide FieldMesh with geometry = rectangular"

    if (
        (FM.axis_index("x") != 0)
        or (FM.axis_index("y") != 1)
        or (FM.axis_index("z") != 2)
    ):
        raise NotImplementedError(
            "Currently function assumes indexing of [x, y, z,]<->[0, 1, 2]."
        )

    dx = FM.dx
    dy = FM.dy
    dz = FM.dz

    if FM.is_pure_electric:
        Fx, Fy, Fz = FM["Ex"], FM["Ey"], FM["Ez"]
        units = r"($\text{V/m}^2)$"
        div_xy = r"$\frac{\partial E_x}{\partial x} + \frac{\partial E_y}{\partial y}$"
        div_z = r"$-\frac{\partial E_z}{\partial z}$"

    elif FM.is_pure_magnetic:
        Fx, Fy, Fz = FM["Bx"], FM["By"], FM["Bz"]
        units = r"($\text{T/m}$)"
        div_xy = r"$\frac{\partial B_x}{\partial x} + \frac{\partial B_y}{\partial y}$"
        div_z = r"$-\frac{\partial B_z}{\partial z}$"

    else:
        raise ValueError("Invalid field type: mixed for test")

    x, y, z = FM.coord_vec("x"), FM.coord_vec("y"), FM.coord_vec("z")

    # Get the point closest to the axis
    if ix is None:
        ix = np.argmin(np.abs(x))

    if iy is None:
        iy = np.argmin(np.abs(y))

    dFxdx = np.gradient(Fx, dx, axis=0, edge_order=2)
    dFydy = np.gradient(Fy, dy, axis=1, edge_order=2)
    dFzdz = np.gradient(Fz, dz, axis=2, edge_order=2)

    if plot:
        plt.plot(z, dFxdx[ix, iy, :] + dFydy[ix, iy, :], label=div_xy)
        plt.plot(z, -dFzdz[ix, iy, :], label=div_z)
        plt.xlabel("z (m)")
        plt.ylabel(div_xy + " " + units)
        plt.title(rf"Fields evaluated at $x$={x[ix]:0.6f}, $y$={y[iy]:0.6f} meters.")
        plt.legend()

        if plot_diff:
            ax = plt.gca()
            ax2 = ax.twinx()
            ax2.plot(
                z,
                +dFxdx[ix, iy, :] + dFydy[ix, iy, :] + dFzdz[ix, iy, :],
                color="black",
                alpha=0.15,
            )
            ax.set_zorder(ax2.get_zorder() + 1)  # Bring primary axis to the front
            ax.patch.set_visible(False)  # Hide the 'canvas' of the primary axis
            ax2.set_ylabel(r"$\Delta$ " + units)

    non_zero = np.abs(dFzdz) > 0.1 * np.max(np.abs(dFzdz[ix, iy, :]))

    err = (dFxdx + dFydy + dFzdz)[non_zero] / dFzdz[non_zero]

    return np.abs(np.mean(err)) < rtol


def check_static_div_equation_cylindrical(
    FM, ir=None, plot=False, rtol=1e-4, plot_diff=True, **kwargs
):
    """
    Checks the static divergence equation in cylindrical coordinates for either electric or magnetic fields.

    This function verifies the static divergence condition, `∇·E = 0` for electric fields or `∇·B = 0`
    for magnetic fields, in a cylindrical coordinate system. The divergence is calculated based on the
    spatial derivatives of the radial and axial field components, evaluated at a specific radial position.
    Optionally, the function plots the individual divergence components and the difference between them.

    Parameters
    ----------
    FM : FieldMesh
        Object containing the static electric or magnetic field data in cylindrical coordinates.
        Must have `is_static`, `geometry`, `is_pure_electric`, and `is_pure_magnetic` attributes to
        determine the field type and properties.
    ir : int, optional
        Index for the radial position at which to evaluate the fields. If None, the function selects
        the index closest to the origin (r=0).
    plot : bool, default=False
        If True, plots the components of the divergence equation and the divergence error (if `plot_diff`
        is also True).
    rtol : float, default=1e-4
        The relative tolerance for verifying that the divergence is zero. If the mean error is below
        this threshold, the function returns True, indicating that the static divergence condition holds.
    plot_diff : bool, default=True
        If True and `plot` is also True, plots the difference between the radial and axial divergence
        components on a secondary y-axis.

    Returns
    -------
    bool
        True if the mean relative error of the computed divergence is below the specified `rtol` threshold,
        indicating that the static divergence condition is satisfied; False otherwise.

    Raises
    ------
    AssertionError
        If `FM.is_static` is False, indicating that non-static fields were provided.
        If `FM.geometry` is not "cylindrical", indicating that a non-cylindrical geometry was provided.
    ValueError
        If the field type in `FM` is mixed (contains both electric and magnetic components), which is
        invalid for this test.

    Notes
    -----
    The function computes divergence components as:
    - Radial component: `(1/r) ∂(r * F_r) / ∂r`
    - Axial component: `∂F_z / ∂z`
    Where `F_r` and `F_z` represent either the radial and axial electric or magnetic field components
    based on the type of field present in `FM`.

    Examples
    --------
    >>> check_static_div_equation_cylindrical(FM, ir=5, plot=True, rtol=1e-5)

    This call checks the static divergence condition at the radial index 5, with a stricter tolerance
    of 1e-5, and plots the divergence components and difference.

    """

    assert FM.is_static, "Must provide static FieldMesh"
    assert FM.geometry == "cylindrical", "Must provide cylindrical FieldMesh"

    dr, dz = FM.dr, FM.dz
    r, z = FM.coord_vec("r"), FM.coord_vec("z")

    # Get the point closest to the axis
    if ir is None:
        ir = np.argmin(np.abs(r))

    if FM.is_pure_electric:
        Fr, Fz = np.squeeze(FM["Er"]), np.squeeze(FM["Ez"])
        units = r"($\text{V/m}^2)$"
        div_r = r"$\frac{1}{r}\frac{\partial}{\partial r}\left(rE_r\right)$"
        div_z = r"$-\frac{\partial E_z}{\partial z}$"
        title = (
            r"$\vec\nabla\cdot \vec{E}=0$, fields evaluated at $r$="
            + f"{r[ir]:0.6f} meters."
        )

    elif FM.is_pure_magnetic:
        Fr, Fz = np.squeeze(FM["Br"]), np.squeeze(FM["Bz"])
        units = r"($\text{T/m}$)"
        div_r = r"$\frac{1}{r}\frac{\partial}{\partial r}\left(rB_r\right)$"
        div_z = r"$-\frac{\partial B_z}{\partial z}$"
        title = (
            r"$\vec\nabla\cdot \vec{B}=0$, fields evaluated at $r$="
            + f"{r[ir]:0.6f} meters."
        )

    else:
        raise ValueError("Invalid field type: mixed for test")

    R, _ = np.meshgrid(r, z, indexing="ij")

    # Handle r = 0 part of cylindrical divergence
    drFrdr = np.gradient(R * Fr, dr, axis=0)
    drFrdr_r = np.zeros(drFrdr.shape)

    non_zero = R > 0
    drFrdr_r[non_zero] = drFrdr[non_zero] / R[non_zero]
    drFrdr_r[~non_zero] = 2 * np.gradient(Fr, dr, axis=0, edge_order=2)[~non_zero]

    dFzdz = np.gradient(Fz, dz, axis=1, edge_order=2)

    if plot:
        plt.plot(z, +drFrdr_r[ir, :], label=div_r)
        plt.plot(z, -dFzdz[ir, :], label=div_z)
        plt.xlabel("z (m)")
        plt.ylabel(div_r + " " + units)
        plt.title(title)
        plt.legend()

        if plot_diff:
            ax = plt.gca()
            ax2 = ax.twinx()
            ax2.plot(z, +drFrdr_r[ir, :] + dFzdz[ir, :], color="black", alpha=0.15)
            ax.set_zorder(ax2.get_zorder() + 1)  # Bring primary axis to the front
            ax.patch.set_visible(False)  # Hide the 'canvas' of the primary axis
            ax2.set_ylabel(r"$\Delta$ " + units)

    non_zero = np.abs(dFzdz) > 0.1 * np.max(np.abs(dFzdz))

    err = (drFrdr_r + dFzdz)[non_zero] / dFzdz[non_zero]

    return np.abs(np.mean(err)) < rtol


def plot_maxwell_curl_equations(FM, **kwargs):
    """
    Plots the curl equations for electric and magnetic fields based on the geometry of the field mesh.

    This function selects and calls the appropriate plotting function (`plot_curl_equations_cylindrical`
    or `plot_curl_equations_cartesian`) to visualize the curl of the electric (E) and magnetic (B) fields
    based on Maxwell's equations, depending on whether the `FieldMesh` geometry is cylindrical or rectangular.

    Parameters
    ----------
    FM : FieldMesh
        Object containing the electric and magnetic field data, spatial coordinates, and geometry information.
        Must specify a geometry via the `geometry` attribute, which can be either "cylindrical" or "rectangular".
        Must also contain a `is_static` attribute.
    **kwargs : dict, optional
        Additional keyword arguments to be passed to the respective plotting functions (`plot_curl_equations_cylindrical`
        or `plot_curl_equations_cartesian`). This may include parameters such as specific index positions or
        options to plot the difference between spatial and temporal curls.

    Raises
    ------
    AssertionError
        If `FM.is_static` is True, indicating the fields are static, which does not meet the requirement
        for oscillating fields.
    ValueError
        If the `geometry` attribute of `FM` is not "cylindrical" or "rectangular".

    Notes
    -----
    This function is a wrapper that checks the geometry type of the `FieldMesh` (FM) and then
    calls the appropriate curl plotting function based on that geometry.

    Examples
    --------
    >>> plot_curl_equations(FM, ir=5, plot_diff=True)

    This call will plot the curl equations for the fields in the `FieldMesh` (FM), assuming FM's geometry is
    either cylindrical or rectangular. The `ir=5` argument will be passed to the appropriate plotting function.

    """

    assert not FM.is_static, "Must provide a time varying FieldMesh"

    if FM.geometry == "cylindrical":
        return plot_maxwell_curl_equations_cylindrical(FM, **kwargs)
    elif FM.geometry == "rectangular":
        return plot_maxwell_curl_equations_cartesian(FM, **kwargs)

    else:
        raise ValueError("Unknown FieldMesh geometry")


def plot_maxwell_curl_equations_cylindrical(FM, ir=None, plot_diff=True):
    """
    Plots the cylindrical curl equations for electric and magnetic field components.

    This function computes and visualizes the curl of electric (E) and magnetic (B)
    fields according to Maxwell's equations in the cylindrical coordinate system.
    It evaluates the spatial derivatives of the field components and compares
    them to the corresponding time derivatives assuming harmonic time dependence of the fields.
    An optional plot of the difference between the spatial curl and the time derivative is shown
    on a secondary y-axis.

    Parameters
    ----------
    FM : FieldMesh
        Object containing the electric and magnetic field data as well as spatial
        coordinates and frequency. Must have attributes `Er`, `Ez`, and `Btheta`
        representing the field components, and `dr` and `dz` representing grid
        spacings in the r and z directions.
    ir : int, optional
        Index for the radial position to evaluate the fields at. If None, the function
        selects the index closest to the r-axis origin (0).
    plot_diff : bool, default=True
        If True, plots the difference between the computed spatial curl and
        the time derivative (based on Maxwell's equations) on a secondary y-axis.

    Raises
    ------
    AssertionError
        If `FM.is_static` is True, indicating the fields are static, which
        does not meet the requirement for oscillating fields.

    Notes
    -----
    This function requires `FM` to contain oscillating fields, as it computes
    temporal derivative terms (curl E and curl B) based on frequency. For a
    zero-frequency (static) field, this function is not valid.

    Examples
    --------
    >>> plot_curl_equations_cylindrical(FM, ir=5, plot_diff=True)

    This call will plot the cylindrical curl equations for the electric and
    magnetic fields, evaluated at the radial index 5, and include a difference
    plot on a secondary axis.

    """

    assert not FM.is_static, "Test requires oscillating fields"

    fig, axs = plt.subplots(3, 1, constrained_layout=True, figsize=(8, 6))

    dr = FM.dr
    dz = FM.dz

    r, z = FM.coord_vec("r"), FM.coord_vec("z")
    R, _ = np.meshgrid(r, z, indexing="ij")

    # Get the point closest to the axis
    if ir is None:
        ir = np.argmin(np.abs(r))

    w = FM.frequency * 2 * np.pi

    Er, Ez, Bth = np.squeeze(FM["Er"]), np.squeeze(FM["Ez"]), np.squeeze(FM["Btheta"])

    dErdz = np.gradient(Er, dz, axis=1, edge_order=2)
    dEzdr = np.gradient(Ez, dr, axis=0, edge_order=2)

    axs[0].plot(
        z,
        np.real(dErdz - dEzdr)[ir, :],
        label=r"$\Re\left[\frac{\partial E_r}{\partial z}-\frac{\partial E_z}{\partial r}\right]$",
    )
    axs[0].plot(z, np.real(1j * w * Bth)[ir, :], label=r"$\Re[i\omega B_{\theta}]$")
    axs[0].set_xlabel("z (m)")
    axs[0].set_ylabel(r"$(\vec\nabla\times\vec{E})_{\theta}$ ($\text{V/m}^2$)")
    axs[0].set_title(rf"Fields evaluated at $r=${r[ir]:0.6f} meters.")
    axs[0].legend()

    if plot_diff:
        ax02 = axs[0].twinx()
        ax02.plot(
            z,
            np.real(dErdz - dEzdr)[ir, :] - np.real(1j * w * Bth)[ir, :],
            color="black",
            alpha=0.15,
        )
        axs[0].set_zorder(ax02.get_zorder() + 1)  # Bring primary axis to the front
        axs[0].patch.set_visible(False)  # Hide the 'canvas' of the primary axis
        ax02.tick_params(axis="y", colors="black")  # Set tick color
        ax02.spines["right"].set_color("black")
        ax02.set_ylabel(r"$\Delta$ ($\text{V/m}^2$)")

    dBthdz = np.gradient(Bth, dz, axis=1, edge_order=2)

    axs[1].plot(
        z,
        -np.imag(dBthdz)[ir, :],
        label=r"$-\Im\left[\frac{\partial B_{\theta}}{\partial z}\right]$",
    )
    axs[1].plot(
        z, -np.imag(1j * w / c_light**2 * Er)[ir, :], label=r"$-\Im[i(\omega/c^2) E_r]$"
    )
    axs[1].set_xlabel("z (m)")
    axs[1].set_ylabel(r"$(\vec\nabla\times\vec{B})_{r}$ ($\text{V/m}^3$)")
    axs[1].legend()

    if plot_diff:
        ax12 = axs[1].twinx()
        ax12.plot(
            z,
            -np.imag(dBthdz)[ir, :] + np.imag(1j * w / c_light**2 * Er)[ir, :],
            color="black",
            alpha=0.15,
        )
        axs[1].set_zorder(ax12.get_zorder() + 1)  # Bring primary axis to the front
        axs[1].patch.set_visible(False)  # Hide the 'canvas' of the primary axis
        # ax12.tick_params(axis='y', colors='tab:red')  # Set tick color
        # ax12.spines['right'].set_color('tab:red')
        ax12.set_ylabel(r"$\Delta$ ($\text{V/m}^3$)")

    R, _ = np.meshgrid(r, z, indexing="ij")

    # Handle r = 0 part of cylindrical divergence
    drBthdr = np.gradient(R * np.imag(Bth), dr, axis=0)
    drBthdr_r = np.zeros(drBthdr.shape)

    non_zero = R > 0
    drBthdr_r[non_zero] = drBthdr[non_zero] / R[non_zero]
    drBthdr_r[~non_zero] = (
        2 * np.gradient(np.imag(Bth), dr, axis=0, edge_order=2)[~non_zero]
    )

    axs[2].plot(
        z,
        drBthdr_r[ir, :],
        label=r"$-\Im\left[\frac{1}{r}\frac{\partial (rB_{\theta})}{\partial r}\right]$",
        color="tab:blue",
    )
    axs[2].plot(
        z,
        -np.imag(1j * w / c_light**2 * Ez)[ir, :],
        label=r"$-\Im[i(\omega/c^2) E_z]$",
        color="tab:orange",
    )
    axs[2].set_xlabel("z (m)")
    axs[2].set_ylabel(r"$(\vec\nabla\times\vec{B})_{z}$ ($\text{V/m}^3$)")
    axs[2].legend()

    if plot_diff:
        ax22 = axs[2].twinx()
        ax22.plot(
            z,
            drBthdr_r[ir, :] + np.imag(1j * w / c_light**2 * Ez)[ir, :],
            color="black",
            alpha=0.15,
        )
        axs[2].set_zorder(ax22.get_zorder() + 1)  # Bring primary axis to the front
        axs[2].patch.set_visible(False)  # Hide the 'canvas' of the primary axis
        # ax22.tick_params(axis='y', colors='tab:red')  # Set tick color
        # ax22.spines['right'].set_color('tab:red')
        ax22.set_ylabel(r"$\Delta$ ($\text{V/m}^3$)")


def plot_maxwell_curl_equations_cartesian(FM, ix=None, iy=None, plot_diff=True):
    """
    Plots the Cartesian curl equations for electric and magnetic field components.

    This function plots the curl of electric (E) and magnetic (B) fields as per
    Maxwell's equations in the Cartesian coordinate system. It computes the
    spatial derivatives of the field components and visualizes the results
    alongside the corresponding time derivatives assuming exp(-iwt) time dependence.
    An optional plot of the difference between the two is shown on a secondary y-axis.

    Parameters
    ----------
    FM : FieldMesh
        Object containing the electric and magnetic field data as well as
        spatial coordinates and frequency. Must have `Ex`, `Ey`, `Ez`,
        `Bx`, `By`, `Bz` attributes representing the field components and
        `dx`, `dy`, `dz` representing grid spacings in the x, y, and z directions.
    ix : int, optional
        Index for the x-coordinate position to evaluate the fields at. If None,
        the function selects the index closest to the x-axis origin (0).
    iy : int, optional
        Index for the y-coordinate position to evaluate the fields at. If None,
        the function selects the index closest to the y-axis origin (0).
    plot_diff : bool, default=True
        If True, plots the difference between the computed spatial curl and
        the time derivative (based on Maxwell's equations) on a secondary y-axis.

    Raises
    ------
    AssertionError
        If `FM.is_static` is True, indicating the fields are static, which
        does not meet the requirement for oscillating fields.

    Notes
    -----
    This function requires `FM` to have oscillating fields, as it computes the
    temporal derivative terms (curl E and curl B) based on frequency. For a
    zero-frequency (static) field, this function is not valid.

    Examples
    --------
    >>> plot_curl_equations_cartesian(FM, ix=5, iy=10, plot_diff=True)

    This call will plot the curl equations for the electric and magnetic fields,
    evaluated at the x and y indices 5 and 10, respectively, and include a
    difference plot on a secondary axis.

    """

    assert not FM.is_static, "Test requires oscillating fields"
    if (
        (FM.axis_index("x") != 0)
        or (FM.axis_index("y") != 1)
        or (FM.axis_index("z") != 2)
    ):
        raise NotImplementedError(
            "Currently function assumes indexing of [x, y, z,]<->[0, 1, 2]."
        )

    fig, axs = plt.subplots(3, 2, figsize=(8, 8))

    dx = FM.dx
    dy = FM.dy
    dz = FM.dz

    w = FM.frequency * 2 * np.pi

    x, y, z = FM.coord_vec("x"), FM.coord_vec("y"), FM.coord_vec("z")

    # Get the point closest to the axis
    if ix is None:
        ix = np.argmin(np.abs(x))

    if iy is None:
        iy = np.argmin(np.abs(y))

    Ex, Ey, Ez = FM["Ex"], FM["Ey"], FM["Ez"]
    Bx, By, Bz = FM["Bx"], FM["By"], FM["Bz"]

    # Curl(Evec) = iw Bvec
    DyEz = np.gradient(Ez, dy, axis=1, edge_order=2)
    DzEy = np.gradient(Ey, dz, axis=2, edge_order=2)

    DzEx = np.gradient(Ex, dz, axis=2, edge_order=2)
    DxEz = np.gradient(Ez, dx, axis=0, edge_order=2)

    DxEy = np.gradient(Ey, dx, axis=0, edge_order=2)
    DyEx = np.gradient(Ex, dy, axis=1, edge_order=2)

    axs[0, 0].plot(
        z,
        +np.real(DyEz - DzEy)[ix, iy, :],
        label=r"$\Re\left[\frac{\partial E_z}{\partial y} - \frac{\partial E_y}{\partial z}\right]$",
    )
    axs[0, 0].plot(z, +np.real(1j * w * Bx[ix, iy, :]), label=r"$-\Re[i\omega B_x]$")
    axs[0, 0].set_xlabel("z (m)")
    axs[0, 0].set_ylabel(r"($\vec\nabla\times\vec{E})_x$ $(\text{V/m}^2)$")
    axs[0, 0].legend()
    axs[0, 0].set_title(
        r"$\vec\nabla\times\vec{E} = -\frac{\partial \vec{B}}{\partial t}$"
    )

    if plot_diff:
        ax002 = axs[0, 0].twinx()
        ax002.plot(
            z,
            np.real(DyEz - DzEy)[ix, iy, :] - np.real(1j * w * Bx[ix, iy, :]),
            color="black",
            alpha=0.15,
        )
        axs[0, 0].set_zorder(ax002.get_zorder() + 1)  # Bring primary axis to the front
        axs[0, 0].patch.set_visible(False)  # Hide the 'canvas' of the primary axis
        # ax12.tick_params(axis='y', colors='tab:red')  # Set tick color
        # ax12.spines['right'].set_color('tab:red')
        ax002.set_ylabel(r"$\Delta$ ($V/m^2$)")

    axs[1, 0].plot(
        z,
        +np.real(DzEx - DxEz)[ix, iy, :],
        label=r"$\Re\left[\frac{\partial E_x}{\partial z} - \frac{\partial E_z}{\partial x}\right]$",
    )
    axs[1, 0].plot(z, +np.real(1j * w * By[ix, iy, :]), label=r"$-\Re[i\omega B_y]$")
    axs[1, 0].set_xlabel("z (m)")
    axs[1, 0].set_ylabel(r"($\vec\nabla\times\vec{E})_y$ $(\text{V/m}^2)$")
    axs[1, 0].legend()

    if plot_diff:
        ax102 = axs[1, 0].twinx()
        ax102.plot(
            z,
            np.real(DzEx - DxEz)[ix, iy, :] - np.real(1j * w * By[ix, iy, :]),
            color="black",
            alpha=0.15,
        )
        axs[1, 0].set_zorder(ax102.get_zorder() + 1)  # Bring primary axis to the front
        axs[1, 0].patch.set_visible(False)  # Hide the 'canvas' of the primary axis
        ax102.set_ylabel(r"$\Delta$ ($\text{V/m}^2$)")

    axs[2, 0].plot(
        z,
        +np.real(DxEy - DyEx)[ix, iy, :],
        label=r"$\Re\left[\frac{\partial E_y}{\partial x} - \frac{\partial E_x}{\partial y}\right]$",
    )
    axs[2, 0].plot(z, +np.real(1j * w * Bz[ix, iy, :]), label=r"$-\Re[i\omega B_z]$")
    axs[2, 0].set_xlabel("z (m)")
    axs[2, 0].set_ylabel(r"($\vec\nabla\times\vec{E})_z$ $(\text{V/m}^2)$")
    axs[2, 0].legend()

    if plot_diff:
        ax202 = axs[2, 0].twinx()
        ax202.plot(
            z,
            np.real(DxEy - DyEx)[ix, iy, :] - np.real(1j * w * Bz[ix, iy, :]),
            color="black",
            alpha=0.15,
        )
        axs[2, 0].set_zorder(ax202.get_zorder() + 1)  # Bring primary axis to the front
        axs[2, 0].patch.set_visible(False)  # Hide the 'canvas' of the primary axis
        ax202.set_ylabel(r"$\Delta$ ($\text{V/m}^2$)")

    # Curl(Bvec) = iw/c2 Evec
    DyBz = np.gradient(Bz, dy, axis=1, edge_order=2)
    DzBy = np.gradient(By, dz, axis=2, edge_order=2)

    DzBx = np.gradient(Bx, dz, axis=2, edge_order=2)
    DxBz = np.gradient(Bz, dx, axis=0, edge_order=2)

    DxBy = np.gradient(By, dx, axis=0, edge_order=2)
    DyBx = np.gradient(Bx, dy, axis=1, edge_order=2)

    axs[0, 1].plot(
        z,
        -np.imag(DyBz - DzBy)[ix, iy, :],
        label=r"$-\Im\left[\frac{\partial B_z}{\partial y} - \frac{\partial B_y}{\partial z}\right]$",
    )
    axs[0, 1].plot(
        z,
        +np.imag(1j * w * Ex[ix, iy, :]) / c_light**2,
        label=r"$\Im[i(\omega/c^2) E_x]$",
    )
    axs[0, 1].set_xlabel("z (m)")
    axs[0, 1].set_ylabel(r"$(\vec\nabla\times\vec{B})_x$ $(V/m^3)$")
    axs[0, 1].set_title(
        r"$\vec\nabla\times\vec{B} = \frac{1}{c^2}\frac{\partial \vec{E}}{\partial t}$"
    )
    axs[0, 1].legend()

    if plot_diff:
        ax012 = axs[0, 1].twinx()
        ax012.plot(
            z,
            -np.imag(DyBz - DzBy)[ix, iy, :]
            - np.imag(1j * w * Ex[ix, iy, :]) / c_light**2,
            color="black",
            alpha=0.15,
        )
        axs[0, 1].set_zorder(ax012.get_zorder() + 1)  # Bring primary axis to the front
        axs[0, 1].patch.set_visible(False)  # Hide the 'canvas' of the primary axis
        ax012.set_ylabel(r"$\Delta$ ($\text{V/m}^3$)")

    axs[1, 1].plot(
        z,
        -np.imag(DzBx - DxBz)[ix, iy, :],
        label=r"$-\Im\left[\frac{\partial B_x}{\partial z} - \frac{\partial B_z}{\partial x}\right]$",
    )
    axs[1, 1].plot(
        z,
        +np.imag(1j * w * Ey[ix, iy, :]) / c_light**2,
        label=r"$\Im[i(\omega/c^2) E_y]$",
    )
    axs[1, 1].set_xlabel("z (m)")
    axs[1, 1].set_ylabel(r"$(\vec\nabla\times\vec{B})_y$ $(\text{V/m}^3)$")
    axs[1, 1].legend()

    if plot_diff:
        ax112 = axs[1, 1].twinx()
        ax112.plot(
            z,
            -np.imag(DzBx - DxBz)[ix, iy, :]
            - np.imag(1j * w * Ey[ix, iy, :]) / c_light**2,
            color="black",
            alpha=0.15,
        )
        axs[1, 1].set_zorder(ax112.get_zorder() + 1)  # Bring primary axis to the front
        axs[1, 1].patch.set_visible(False)  # Hide the 'canvas' of the primary axis
        ax112.set_ylabel(r"$\Delta$ ($\text{V/m}^3$)")

    axs[2, 1].plot(
        z,
        -np.imag(DxBy - DyBx)[ix, iy, :],
        label=r"$-\Im\left[\frac{\partial B_y}{\partial x} - \frac{\partial B_x}{\partial y}\right]$",
    )
    axs[2, 1].plot(
        z,
        +np.imag(1j * w * Ez[ix, iy, :]) / c_light**2,
        label=r"$\Im[i(\omega/c^2) E_z]$",
    )
    axs[2, 1].set_xlabel("z (m)")
    axs[2, 1].set_ylabel(r"$(\vec\nabla\times\vec{B})_z$ $(\text{V/m}^3)$")
    axs[2, 1].legend()

    if plot_diff:
        ax212 = axs[2, 1].twinx()
        ax212.plot(
            z,
            -np.imag(DxBy - DyBx)[ix, iy, :]
            - np.imag(1j * w * Ez[ix, iy, :]) / c_light**2,
            color="black",
            alpha=0.15,
        )
        axs[1, 1].set_zorder(ax202.get_zorder() + 1)  # Bring primary axis to the front
        axs[1, 1].patch.set_visible(False)  # Hide the 'canvas' of the primary axis
        ax212.set_ylabel(r"$\Delta$ ($\text{V/m}^3$)")

    fig.suptitle(rf"Fields evaluated at $x=${x[ix]:0.6f}, $y=${y[iy]:0.6f} meters.")
    plt.tight_layout()
