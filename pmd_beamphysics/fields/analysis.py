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
    c = 299792458
    omega = 2 * np.pi * frequency
    k = omega / c
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
    assert FM.is_static, "Must provide a static FieldMesh"

    if FM.geometry == "cylindrical":
        return check_static_div_equation_cylindrical(FM, plot=plot, rtol=rtol, *kwargs)
    elif FM.geometry == "rectangular":
        return check_static_div_equation_cartesian(FM, plot=plot, rtol=rtol, *kwargs)

    else:
        raise ValueError("Unknown FieldMesh geometry")


def check_static_div_equation_cartesian(
    FM, ix=None, iy=None, plot=False, rtol=1e-4, plot_diff=True
):
    assert FM.is_static, "Must provide static FieldMesh"
    assert (
        FM.geometry == "rectangular"
    ), "Must provide FieldMesh with geometry = rectangular"

    dx = FM.dx
    dy = FM.dy
    dz = FM.dz

    if FM.is_pure_electric:
        Fx, Fy, Fz = FM["Ex"], FM["Ey"], FM["Ez"]
        units = r"($V/m^2)$"
        div_xy = r"$\frac{\partial E_x}{\partial x} + \frac{\partial E_y}{\partial y}$"
        div_z = r"$-\frac{\partial E_z}{\partial z}$"

    elif FM.is_pure_magnetic:
        Fx, Fy, Fz = FM["Bx"], FM["By"], FM["Bz"]
        units = r"($T/m$)"
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
        plt.ylabel(units)
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
            ax2.set_ylabel("$\Delta$ " + units)

    non_zero = np.abs(dFzdz) > 0.1 * np.max(np.abs(dFzdz[ix, iy, :]))

    err = (dFxdx + dFydy + dFzdz)[non_zero] / dFzdz[non_zero]

    return np.abs(np.mean(err)) < rtol


def check_static_div_equation_cylindrical(
    FM, ir=None, plot=False, rtol=1e-4, plot_diff=True, **kwargs
):
    assert FM.is_static, "Must provide static FieldMesh"
    assert FM.geometry == "cylindrical", "Must provide cylindrical FieldMesh"

    dr, dz = FM.dr, FM.dz
    r, z = FM.coord_vec("r"), FM.coord_vec("z")

    # Get the point closest to the axis
    if ir is None:
        ir = np.argmin(np.abs(r))

    if FM.is_pure_electric:
        Fr, Fz = np.squeeze(FM["Er"]), np.squeeze(FM["Ez"])
        units = r"($V/m^2)$"
        div_r = r"$\frac{1}{r}\frac{\partial}{\partial r}\left(rE_r\right)$"
        div_z = r"$-\frac{\partial E_z}{\partial z}$"
        title = (
            r"$\vec\nabla\cdot \vec{E}=0$, fields evaluated at $r$="
            + f"{r[ir]:0.6f} meters."
        )

    elif FM.is_pure_magnetic:
        Fr, Fz = np.squeeze(FM["Br"]), np.squeeze(FM["Bz"])
        units = r"($T/m$)"
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
        plt.ylabel(units)
        plt.title(title)
        plt.legend()

        if plot_diff:
            ax = plt.gca()
            ax2 = ax.twinx()
            ax2.plot(z, +drFrdr_r[ir, :] + dFzdz[ir, :], color="black", alpha=0.15)
            ax.set_zorder(ax2.get_zorder() + 1)  # Bring primary axis to the front
            ax.patch.set_visible(False)  # Hide the 'canvas' of the primary axis
            ax2.set_ylabel("$\Delta$ " + units)

    non_zero = np.abs(dFzdz) > 0.1 * np.max(np.abs(dFzdz))

    err = (drFrdr_r + dFzdz)[non_zero] / dFzdz[non_zero]

    return np.abs(np.mean(err)) < rtol


def plot_curl_equations(FM, **kwargs):
    assert not FM.is_static, "Must provide a static FieldMesh"

    if FM.geometry == "cylindrical":
        return plot_curl_equations_cylindrical(FM, **kwargs)
    elif FM.geometry == "rectangular":
        return plot_curl_equations_cartesian(FM, **kwargs)

    else:
        raise ValueError("Unknown FieldMesh geometry")


def plot_curl_equations_cylindrical(FM, ir=None, plot_diff=True):
    c = 299792458

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
    axs[0].set_ylabel("(V/m^2)")
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
        ax02.set_ylabel("$\Delta$ ($V/m^2$)")

    dBthdz = np.gradient(Bth, dz, axis=1, edge_order=2)

    axs[1].plot(
        z,
        -np.imag(dBthdz)[ir, :],
        label=r"$-\Im\left[\frac{\partial B_{\theta}}{\partial z}\right]$",
    )
    axs[1].plot(
        z, -np.imag(1j * w / c**2 * Er)[ir, :], label=r"$-\Im[i(\omega/c^2) E_r]$"
    )
    axs[1].set_xlabel("z (m)")
    axs[1].set_ylabel("($V/m^3$)")
    axs[1].legend()

    if plot_diff:
        ax12 = axs[1].twinx()
        ax12.plot(
            z,
            -np.imag(dBthdz)[ir, :] + np.imag(1j * w / c**2 * Er)[ir, :],
            color="black",
            alpha=0.15,
        )
        axs[1].set_zorder(ax12.get_zorder() + 1)  # Bring primary axis to the front
        axs[1].patch.set_visible(False)  # Hide the 'canvas' of the primary axis
        # ax12.tick_params(axis='y', colors='tab:red')  # Set tick color
        # ax12.spines['right'].set_color('tab:red')
        ax12.set_ylabel("$\Delta$ ($V/m^3$)")

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
        -np.imag(1j * w / c**2 * Ez)[ir, :],
        label=r"$-\Im[i(\omega/c^2) E_z]$",
        color="tab:orange",
    )
    axs[2].set_xlabel("z (m)")
    axs[2].set_ylabel("($V/m^3$)")
    axs[2].legend()

    if plot_diff:
        ax22 = axs[2].twinx()
        ax22.plot(
            z,
            drBthdr_r[ir, :] + np.imag(1j * w / c**2 * Ez)[ir, :],
            color="black",
            alpha=0.15,
        )
        axs[2].set_zorder(ax22.get_zorder() + 1)  # Bring primary axis to the front
        axs[2].patch.set_visible(False)  # Hide the 'canvas' of the primary axis
        # ax22.tick_params(axis='y', colors='tab:red')  # Set tick color
        # ax22.spines['right'].set_color('tab:red')
        ax22.set_ylabel("$\Delta$ ($V/m^3$)")

    fig.suptitle("Fields evaluated at $x$=")  # or plt.suptitle('Main title')


def plot_curl_equations_cartesian(FM, ix=None, iy=None, plot_diff=True):
    c = 299792458

    assert not FM.is_static, "Test requires oscillating fields"

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
    axs[0, 0].set_ylabel("$(V/m^2)$")
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
        ax002.set_ylabel("$\Delta$ ($V/m^2$)")

    axs[1, 0].plot(
        z,
        +np.real(DzEx - DxEz)[ix, iy, :],
        label=r"$\Re\left[\frac{\partial E_x}{\partial z} - \frac{\partial E_z}{\partial x}\right]$",
    )
    axs[1, 0].plot(z, +np.real(1j * w * By[ix, iy, :]), label=r"$-\Re[i\omega B_y]$")
    axs[1, 0].set_xlabel("z (m)")
    axs[1, 0].set_ylabel("$(V/m^2)$")
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
        ax102.set_ylabel("$\Delta$ ($V/m^2$)")

    axs[2, 0].plot(
        z,
        +np.real(DxEy - DyEx)[ix, iy, :],
        label=r"$\Re\left[\frac{\partial E_y}{\partial x} - \frac{\partial E_x}{\partial y}\right]$",
    )
    axs[2, 0].plot(z, +np.real(1j * w * Bz[ix, iy, :]), label=r"$-\Re[i\omega B_z]$")
    axs[2, 0].set_xlabel("z (m)")
    axs[2, 0].set_ylabel("$(V/m^2)$")
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
        ax202.set_ylabel("$\Delta$ ($V/m^2$)")

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
        z, +np.imag(1j * w * Ex[ix, iy, :]) / c**2, label=r"$\Im[i(\omega/c^2) E_x]$"
    )
    axs[0, 1].set_xlabel("z (m)")
    axs[0, 1].set_ylabel("$(V/m^3)$")
    axs[0, 1].set_title(
        r"$\vec\nabla\times\vec{B} = \frac{1}{c^2}\frac{\partial \vec{E}}{\partial t}$"
    )
    axs[0, 1].legend()

    if plot_diff:
        ax012 = axs[0, 1].twinx()
        ax012.plot(
            z,
            -np.imag(DyBz - DzBy)[ix, iy, :] - np.imag(1j * w * Ex[ix, iy, :]) / c**2,
            color="black",
            alpha=0.15,
        )
        axs[0, 1].set_zorder(ax012.get_zorder() + 1)  # Bring primary axis to the front
        axs[0, 1].patch.set_visible(False)  # Hide the 'canvas' of the primary axis
        ax012.set_ylabel("$\Delta$ ($V/m^3$)")

    axs[1, 1].plot(
        z,
        -np.imag(DzBx - DxBz)[ix, iy, :],
        label=r"$-\Im\left[\frac{\partial B_x}{\partial z} - \frac{\partial B_z}{\partial x}\right]$",
    )
    axs[1, 1].plot(
        z, +np.imag(1j * w * Ey[ix, iy, :]) / c**2, label=r"$\Im[i(\omega/c^2) E_y]$"
    )
    axs[1, 1].set_xlabel("z (m)")
    axs[1, 1].set_ylabel("$(V/m^3)$")
    axs[1, 1].legend()

    if plot_diff:
        ax112 = axs[1, 1].twinx()
        ax112.plot(
            z,
            -np.imag(DzBx - DxBz)[ix, iy, :] - np.imag(1j * w * Ey[ix, iy, :]) / c**2,
            color="black",
            alpha=0.15,
        )
        axs[1, 1].set_zorder(ax112.get_zorder() + 1)  # Bring primary axis to the front
        axs[1, 1].patch.set_visible(False)  # Hide the 'canvas' of the primary axis
        ax112.set_ylabel("$\Delta$ ($V/m^3$)")

    axs[2, 1].plot(
        z,
        -np.imag(DxBy - DyBx)[ix, iy, :],
        label=r"$-\Im\left[\frac{\partial B_y}{\partial x} - \frac{\partial B_x}{\partial y}\right]$",
    )
    axs[2, 1].plot(
        z, +np.imag(1j * w * Ez[ix, iy, :]) / c**2, label=r"$\Im[i(\omega/c^2) E_z]$"
    )
    axs[2, 1].set_xlabel("z (m)")
    axs[2, 1].set_ylabel("$(V/m^3)$")
    axs[2, 1].legend()

    if plot_diff:
        ax212 = axs[2, 1].twinx()
        ax212.plot(
            z,
            -np.imag(DxBy - DyBx)[ix, iy, :] - np.imag(1j * w * Ez[ix, iy, :]) / c**2,
            color="black",
            alpha=0.15,
        )
        axs[1, 1].set_zorder(ax202.get_zorder() + 1)  # Bring primary axis to the front
        axs[1, 1].patch.set_visible(False)  # Hide the 'canvas' of the primary axis
        ax212.set_ylabel("$\Delta$ ($V/m^3$)")

    fig.suptitle(rf"Fields evaluated at $x=${x[ix]:0.6f}, $y=${y[iy]:0.6f} meters.")
    plt.tight_layout()
