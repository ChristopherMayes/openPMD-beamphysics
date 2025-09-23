import os
from pathlib import Path
from collections import namedtuple
from typing import Union, Optional

import numpy as np
from h5py import File, Group

from ..statistics import twiss_calc
from ..units import c_light, mec2, unit, write_unit_h5

# Genesis 1.3
# -------------
# Version 2 routines


def genesis2_beam_data1(pg):
    """

    Calculate statistics of a single particlegroup,
    for use in a Genesis 1.3 v2 beam file

    Returns a dict of:
        zpos or tpos      : z or t of the mean in m or s
        curpeak           : current in A
        gamma0            : average relativistic gamma (dimensionless)
        emitx, emity      : Normalized emittances in m*rad
        rxbeam, rybeam    : sigma_x, sigma_y in m
        xbeam, ybeam      : <x>, <y> in m
        pxbeam, pybeam    : beta_x*gamma, beta_y*gamma (dimensionless)
        alpha_x, alpha_y  : Twiss alpha  (dinensionless)
    """

    d = {}

    # Handle z or t
    if len(set(pg.z)) == 1:
        # should be different t
        d["tpos"] = pg["mean_t"]
    elif len(set(pg.t)) == 1:
        # should be different z
        d["zpos"] = pg["mean_z"]
    else:
        raise ValueError(f"{pg} has mixed t and z coordinates.")

    d["curpeak"] = pg["average_current"]
    d["gamma0"] = pg["mean_gamma"]
    d["delgam"] = pg["sigma_gamma"]
    d["emitx"] = pg["norm_emit_x"]
    d["emity"] = pg["norm_emit_y"]
    d["rxbeam"] = pg["sigma_x"]
    d["rybeam"] = pg["sigma_y"]
    d["xbeam"] = pg["mean_x"]
    d["ybeam"] = pg["mean_y"]

    d["pxbeam"] = pg["mean_px"] / pg.mass  # beta_x*gamma
    d["pybeam"] = pg["mean_py"] / pg.mass  # beta_y*gamma

    # Twiss, for alpha
    twiss_x = twiss_calc(pg.cov("x", "xp"))
    twiss_y = twiss_calc(pg.cov("y", "yp"))
    d["alphax"] = twiss_x["alpha"]
    d["alphay"] = twiss_y["alpha"]

    return d


def genesis2_beam_data(pg, n_slice=None):
    """

    Slices a particlegroup into n_slice and forms the beam columns.

    n_slice is the number of slices. If not given, the beam will be divided
    so there are 100 particles in each slice.

    Returns a dict of beam_columns, for use with write_genesis2_beam_file

    See: genesis2_beam_data1
    """

    # Handle z or t
    if len(set(pg.z)) == 1:
        # should be different t
        slice_key = "t"
    elif len(set(pg.t)) == 1:
        # should be different z
        slice_key = "z"
    else:
        raise ValueError(f"{pg} has mixed t and z coordinates.")

    # Automatic slicing
    if not n_slice:
        n_slice = len(pg) // 100

    # Slice (split)
    pglist = pg.split(n_slice, key=slice_key)

    d = {}
    # Loop over the slices
    for pg in pglist:
        data1 = genesis2_beam_data1(pg)
        for k in data1:
            if k not in d:
                d[k] = []
            d[k].append(data1[k])
    for k in d:
        d[k] = np.array(d[k])

    return d


def write_genesis2_beam_file(fname, beam_columns, verbose=False):
    """
    Writes a Genesis 1.3 v2 beam file, using a dict beam_columns

    The header will be written as:
    ? VERSION=1.0
    ? SIZE=<length of the columns>
    ? COLUMNS <list of columns
    <data>

    This is a copy of the lume-genesis routine:

    genesis.writers.write_beam_file

    """

    # Get size
    names = list(beam_columns)
    size = len(beam_columns[names[0]])
    header = f"""? VERSION=1.0
? SIZE={size}
? COLUMNS {" ".join([n.upper() for n in names])}"""

    dat = np.array([beam_columns[name] for name in names]).T

    np.savetxt(
        fname, dat, header=header, comments="", fmt="%1.8e"
    )  # Genesis can't read format %1.16e - lines are too long?

    if verbose:
        print("Beam written:", fname)

    return header


def genesis2_dpa_to_data(dpa, *, xlamds, current, zsep=1, species="electron"):
    """
    Converts Genesis 1.3 v2 dpa data to ParticleGroup data.

    The Genesis 1.3 v2 dpa phase space coordinates and units are:
        gamma [1]
        phase [rad]
        x [m]
        y [m]
        px/mc [1]
        py/mc [1]
    The definition of the phase is different between the .par and .dpa files.
        .par file: phase = psi  = kw*z + field_phase
        .dpa file: phase = kw*z

    Parameters
    ----------

    dpa: array
        Parsed .dpa file as an array with shape (n_slice, 6, n_particles_per_slice)

    xlamds: float
        wavelength (m)

    zsep: int
        slice separation interval

    current: array
        current array of length n_slice (A)


    species: str, required to be 'electron'

    Returns
    -------
    data: dict with keys: x, px, y, py, z, pz, weight, species, status
        in the units of the openPMD-beamphysics Python package:
        m, eV/c, m, eV/c, m, eV/c, C

        These are returned in z-coordinates, with z=0.

    """

    assert species == "electron"
    mc2 = mec2

    dz = xlamds * zsep

    nslice, dims, n1 = dpa.shape  # n1 particles in a single slice
    assert dims == 6
    n_particle = n1 * nslice

    gamma = dpa[:, 0, :].flatten()
    phase = dpa[:, 1, :].flatten()
    x = dpa[:, 2, :].flatten()
    y = dpa[:, 3, :].flatten()
    px = dpa[:, 4, :].flatten() * mc2
    py = dpa[:, 5, :].flatten() * mc2
    pz = np.sqrt((gamma**2 - 1) * mc2**2 - px**2 - py * 2)

    i0 = np.arange(nslice)
    i_slice = np.repeat(i0[:, np.newaxis], n1, axis=1).flatten()

    # Spread particles out over zsep interval
    z = dz * (
        i_slice + np.mod(phase / (2 * np.pi * zsep), zsep)
    )  # z = (zsep * xlamds) * (i_slice + mod(dpa_phase/2pi, 1))
    z = z.flatten()
    # t = np.full(n_particle, 0.0)

    # z-coordinates
    t = -z / c_light
    z = np.full(n_particle, 0.0)

    weight = np.repeat(current[:, np.newaxis], n1, axis=1).flatten() * dz / c_light / n1

    return {
        "t": t,
        "x": x,
        "px": px,
        "y": y,
        "py": py,
        "z": z,
        "pz": pz,
        "species": species,
        "weight": weight,
        "status": np.full(n_particle, 1),
    }


# -------------
# Version 4 routines


def genesis4_beam_data(pg, n_slice=None):
    """
    Slices a particlegroup into n_slice and forms the sliced beam data.

    n_slice is the number of slices. If not given, the beam will be divided
    so there are 100 particles in each slice.

    Returns a dict of beam_columns, for use with write_genesis2_beam_file

    This uses the same routines as genesis2, with some relabeling
    See: genesis2_beam_data1
    """

    # Re-use genesis2_beam_data
    g2data = genesis2_beam_data(pg, n_slice=n_slice)

    # Old, new, unit
    relabel = [
        ("tpos", "t", "s"),
        ("zpos", "s", "m"),
        ("curpeak", "current", "A"),
        ("gamma0", "gamma", "1"),
        ("delgam", "delgam", "1"),
        ("emitx", "ex", "m"),
        ("emity", "ey", "m"),
        ("rxbeam", "sigma_x", "m"),
        ("rybeam", "sigma_y", "m"),
        ("xbeam", "xcenter", "m"),
        ("ybeam", "ycenter", "m"),
        ("pxbeam", "pxcenter", "1"),
        ("pybeam", "pycenter", "1"),
        ("alphax", "alphax", "1"),
        ("alphay", "alphay", "1"),
    ]

    data = {}
    units = {}
    for g2key, g4key, u in relabel:
        if g2key not in g2data:
            continue
        data[g4key] = g2data[g2key]
        units[g4key] = unit(u)

    # Re-calculate these
    data["betax"] = data["gamma"] * data.pop("sigma_x") ** 2 / data["ex"]
    data["betay"] = data["gamma"] * data.pop("sigma_y") ** 2 / data["ey"]

    units["betax"] = unit("m")
    units["betay"] = unit("m")

    if "s" in data:
        data["s"] -= data["s"].min()

    return data, units


def write_genesis4_beam(
    particle_group, h5_fname, n_slice=None, verbose=False, return_input_str=False
):
    """
    Writes sliced beam data to an HDF5 file

    """
    beam_data, units = genesis4_beam_data(particle_group, n_slice=n_slice)

    with File(h5_fname, "w") as h5:
        for k in beam_data:
            h5[k] = beam_data[k]
            write_unit_h5(h5[k], units[k])

    if verbose:
        print("Genesis4 beam file written:", h5_fname)

    if return_input_str:
        data_keys = list(beam_data)
        lines = genesis4_profile_file_input_str(data_keys, h5_fname)
        lines += genesis4_beam_input_str(data_keys)
        return lines


def _profile_file_lines(
    label, h5filename, xdata_key, ydata_key, isTime=False, reverse=False
):
    lines = f"""&profile_file
  label = {label}
  xdata = {h5filename}/{xdata_key}
  ydata = {h5filename}/{ydata_key}"""
    if isTime:
        lines += "\n  isTime = T"
    if reverse:
        lines += "\n  reverse = T"
    lines += "\n&end\n"
    return lines


def genesis4_profile_file_input_str(data_keys, h5filename):
    """
    Returns an input str suitable for the main Genesis4 input file
    for profile data.
    """

    h5filename = os.path.split(h5filename)[1]  # Genesis4 does not understand paths

    if "s" in data_keys:
        xdata_key = "s"
        isTime = False
        reverse = False
    elif "t" in data_keys:
        xdata_key = "t"
        isTime = True
        reverse = True
    else:
        raise ValueError("no s or t found")

    lines = ""
    for ydata_key in data_keys:
        if ydata_key == xdata_key:
            continue
        lines += _profile_file_lines(
            ydata_key, h5filename, xdata_key, ydata_key, isTime, reverse
        )

    return lines


def genesis4_beam_input_str(data_keys):
    """
    Returns an input str suitable for the main Genesis4 input file
    for profile data.
    """
    lines = ["&beam"]
    for k in data_keys:
        if k in ("s", "t"):
            continue
        lines.append(f"  {k} = @{k}")
    lines.append("&end")
    return "\n".join(lines)


def write_genesis4_distribution(particle_group, h5file, verbose=False):
    """


    Cooresponds to the `import distribution` section in the Genesis4 manual.

    Writes datesets to an h5 file:

    h5file: str or open h5 handle

    Datasets
        x is the horizontal coordinate in meters
        y is the vertical coordinate in meters
        xp = px/pz is the dimensionless trace space horizontal momentum
        yp = py/pz is the dimensionless trace space vertical momentum
        t is the time in seconds
        p  = relativistic gamma*beta is the total momentum divided by mc


        These should be the same as in .interfaces.elegant.write_elegant


    If particles are at different z, they will be drifted to the same z,
    because the output should have different times.

    If any of the weights are different, the bunch will be resampled
    to have equal weights.
    Note that this can be very slow for a large number of particles.

    """

    if isinstance(h5file, str):
        h5 = File(h5file, "w")
    else:
        h5 = h5file

    if len(set(particle_group.z)) > 1:
        if verbose:
            print("Drifting particles to the same z")
        # Work on a copy, because we will drift
        P = particle_group.copy()
        # Drift to z.
        P.drift_to_z()
    else:
        P = particle_group

    if len(set(P.weight)) > 1:
        n = len(P)
        if verbose:
            print(f"Resampling {n} weighted particles")
        P = P.resample(n, equal_weights=True)

    for k in ["x", "xp", "y", "yp", "t"]:
        h5[k] = P[k]

    # p is really beta*gamma
    h5["p"] = P["p"] / P.mass

    if verbose:
        print(f"Datasets x, xp, y, yp, t, p written to: {h5file}")


PARFILE_SLICE_FIELDS = ("x", "px", "y", "py", "gamma", "theta", "current")
ParfileSliceData = namedtuple("RawSliceData", PARFILE_SLICE_FIELDS)


# Known scalars
_parfile_scalar_datasets = [
    "beamletsize",
    "one4one",
    "refposition",
    "slicecount",
    "slicelength",
    "slicespacing",
]

_parfile_skip_groups = [
    "Meta",
]


def genesis4_parfile_scalars(h5):
    """
    Extract useful scalars and slice names from a Genesis4 .par file
    """
    # Allow for opening a file
    if isinstance(h5, (str, Path)):
        assert os.path.exists(h5), f"File does not exist: {h5}"
        h5 = File(h5, "r")

    params = {}
    for k in _parfile_scalar_datasets:
        if k not in h5 or h5[k].shape != (1,):
            raise ValueError(
                f"Expected scalar dataset '{k}' with shape (1,), got shape {h5[k].shape}"
            )
        params[k] = h5[k][0]

    return params


def genesis4_parfile_slice_groups(h5):
    """
    Extract useful scalars and slice names from a Genesis4 .par file
    """
    # Allow for opening a file
    if isinstance(h5, (str, Path)):
        assert os.path.exists(h5), f"File does not exist: {h5}"
        h5 = File(h5, "r")

    return sorted(
        g for g in h5 if g not in _parfile_scalar_datasets + _parfile_skip_groups
    )


def load_parfile_slice_data(group):
    """
    Returns ParfileSliceData with Genesis4's named fields in a slice group.
    """
    values = [group[field][:] for field in PARFILE_SLICE_FIELDS[:-1]]
    current = group["current"][:]  # This is really a scalar
    assert len(current) == 1
    values.append(current[0])
    return ParfileSliceData(*values)


def genesis4_par_to_data(
    h5: Union[str, Path, File],
    species: str = "electron",
    smear: bool = False,
    wrap: bool = False,
    z0: float = 0,
    slices: Optional[list[int]] = None,
    equal_weights: bool = False,
    cutoff: float = 1.6e-19,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """
    Convert Genesis 4 `.par` slice data from an HDF5 file into a dictionary
    compatible with `openPMD-beamphysics` ParticleGroup input.

    Each slice in the Genesis 4 output contains sampled macroparticles with
    position, momentum, and phase data. This function reconstructs full 6D
    coordinates and weights, with options for smearing, resampling, and filtering.

    Parameters
    ----------
    h5 : str, Path, or h5py.File
        Path to a Genesis 4 `.par` HDF5 file, or an open h5py file handle.

    species : str, default="electron"
        Particle species. Currently, only "electron" is supported.

    smear : bool, default=False
        Genesis4 often samples the beam by skipping slices
        in a step called 'sample'.
        This will smear out the theta coordinate over these slices,
        preserving the modulus.

    slices : list of int, optional
        Specific slice indices to include. If None, all available slices are used.

    wrap: bool, default = False
        If true, the z position within a slice will be modulo the slice length.

    z0 : float, default=0.0
        Initial z-position for slice index 1, in meters.

    equal_weights : bool, default=False
        If True, particles are resampled within each slice so that all output
        particles have equal charge weights. Useful for simulations that
        require uniform macroparticles.

    cutoff : float, default=1.6e-19
        Minimum per-particle weight in Coulombs. Slices resulting in
        sub-electron macroparticles are skipped to avoid numerical issues.

    rng : numpy.random.Generator or seed, optional
        Random number generator or seed used for smearing and resampling.
        If None, a new default RNG is created.


    Returns
    -------
    data : dict
        Dictionary containing particle phase space and metadata in openPMD-compatible format:
        {
            "x", "y", "z"         : position [m],
            "px", "py", "pz"      : momentum [eV/c],
            "t"                   : time [s],
            "status"              : particle status flag (all 1),
            "species"             : particle species string,
            "weight"              : particle weight [Coulombs],
        }

    Notes
    -----
    - The function expects Genesis slice groups named "slice000001", etc., each containing:
        - 'x': x position in meters
        - 'px': gamma * beta_x
        - 'y': y position in meters
        - 'py': gamma * beta_y
        - 'theta': longitudinal phase angle in radians
        - 'gamma': relativistic gamma
        - 'current': scalar slice current in Amperes

    - Scalar metadata (e.g., `slicespacing`, `slicelength`) must be stored as 1-element datasets.

    - z positions are reconstructed from theta (ponderomotive phase), with optional smearing
      to account for undersampled slices, and offset by slice position and `z0`.

    - Transverse momenta px and py are scaled by mec² to convert to eV/c.

    """
    # Allow for opening a file
    if isinstance(h5, (str, Path)):
        assert os.path.exists(h5), f"File does not exist: {h5}"
        h5 = File(h5, "r")

    if species != "electron":
        raise ValueError("Only electrons supported for Genesis4")

    # Extract scalar parameters
    params = genesis4_parfile_scalars(h5)

    # Useful local variables
    ds_slice = params["slicelength"]  # single slice length
    s_spacing = params["slicespacing"]  # spacing between slices
    sample = round(s_spacing / ds_slice)  # This should be an integer

    #
    xs = []
    pxs = []
    ys = []
    pys = []
    zs = []
    gammas = []
    weights = []

    if slices is None:
        snames = genesis4_parfile_slice_groups(h5)

    else:
        snames = [f"slice{ix:06}" for ix in slices]

    rng = np.random.default_rng(rng)

    for sname in snames:
        ix = int(sname[5:])  # Extract slice index

        # Skip missing
        if sname not in h5:
            continue

        g = h5[sname]
        if not isinstance(g, Group) or "current" not in g:
            # Groups like 'Meta' do not contain slice data.
            continue

        pdata = load_parfile_slice_data(g)

        current = pdata.current  # I * s_spacing/c = Q
        n1 = len(pdata.x)

        # Convert current to weight (C)
        # I * s_spacing/c = Q
        # Single charge
        q1 = current * s_spacing / c_light / n1

        # Skip subphysical particles
        if q1 < cutoff:
            continue

        # Skip zero current slices. These usually have nans in the particle data.
        if current == 0:
            continue

        # Calculate z
        theta = pdata.theta

        # Smear theta over sample slices
        irel = theta / (2 * np.pi)
        if wrap:
            irel = irel % 1  # Relative bin position (0,1)

        # Random smear
        if smear:
            z1 = (irel + rng.integers(0, sample, size=n1)) * ds_slice
        else:
            z1 = (irel) * ds_slice

        z1 = z1 + (ix - 1) * s_spacing + z0  # set absolute z

        # Collect arrays
        xs.append(pdata.x)
        pxs.append(pdata.px * mec2)
        ys.append(pdata.y)
        pys.append(pdata.py * mec2)
        gammas.append(pdata.gamma)
        zs.append(z1)
        weights.append(np.full(n1, q1))

    if equal_weights:
        # resample each slice

        n_slices = len(weights)

        counts = list(set([len(w) for w in weights]))
        assert (
            len(counts) == 1
        )  # All slices are supposed to have the same number of particles
        n1 = counts[0]

        slice_charges = np.array([np.sum(w) for w in weights])
        max_slice_charge = np.max(
            slice_charges
        )  # use this as an upper limit for sampling
        n_samples = (n1 * slice_charges / max_slice_charge).astype(int)
        total_charge = np.sum(slice_charges)
        n_samples_total = np.sum(n_samples)
        weight1 = total_charge / n_samples_total  # new equal weight

        # Loop over populated slices (note that some are skipped above)
        for i in range(n_slices):
            n_sample = n_samples[i]
            samples = rng.choice(n1, n_sample, replace=False)
            xs[i] = xs[i][samples]
            pxs[i] = pxs[i][samples]
            ys[i] = ys[i][samples]
            pys[i] = pys[i][samples]
            zs[i] = zs[i][samples]
            gammas[i] = gammas[i][samples]
            weights[i] = np.full(n_sample, weight1)

    # Stack
    x = np.hstack(xs)
    px = np.hstack(pxs)
    y = np.hstack(ys)
    py = np.hstack(pys)
    gamma = np.hstack(gammas)
    z = np.hstack(zs)
    weight = np.hstack(weights)

    # Form final particlegroup data
    n = len(weight)
    p = np.sqrt(gamma**2 - 1) * mec2
    pz = np.sqrt(p**2 - px**2 - py**2)

    status = 1
    data = {
        "x": x,
        "y": y,
        "z": z,
        "px": px,
        "py": py,
        "pz": pz,
        "t": np.full(n, 0.0),
        "status": np.full(n, status),
        "species": species,
        "weight": weight,
    }

    return data
