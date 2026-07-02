import os
import re
from pathlib import Path
from collections import namedtuple
from typing import Optional

import numpy as np
from h5py import File, Group

from ..statistics import twiss_calc
from ..units import Z0, c_light, e_charge, mec2, pmd_unit, write_unit_h5

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

    if species != "electron":
        raise ValueError(f"Only electron is supported, got: {species}")
    mc2 = mec2

    dz = xlamds * zsep

    nslice, dims, n1 = dpa.shape  # n1 particles in a single slice
    if dims != 6:
        raise ValueError(f".dpa data must have 6 phase-space dimensions, found {dims}")
    n_particle = n1 * nslice

    gamma = dpa[:, 0, :].flatten()
    phase = dpa[:, 1, :].flatten()
    x = dpa[:, 2, :].flatten()
    y = dpa[:, 3, :].flatten()
    px = dpa[:, 4, :].flatten() * mc2
    py = dpa[:, 5, :].flatten() * mc2
    pz = np.sqrt((gamma**2 - 1) * mc2**2 - px**2 - py**2)

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
        units[g4key] = pmd_unit(u)

    # Re-calculate these
    data["betax"] = data["gamma"] * data.pop("sigma_x") ** 2 / data["ex"]
    data["betay"] = data["gamma"] * data.pop("sigma_y") ** 2 / data["ey"]

    units["betax"] = pmd_unit("m")
    units["betay"] = pmd_unit("m")

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
ParfileSliceData = namedtuple("ParfileSliceData", PARFILE_SLICE_FIELDS)


# Known scalars
_parfile_scalar_datasets = [
    "beamletsize",
    "one4one",
    "refposition",
    "slicecount",
    "slicelength",
    "slicespacing",
]

# Slice groups are named 'slice000001', 'slice000002', ...
_parfile_slice_name_re = re.compile(r"slice\d+")


def genesis4_parfile_scalars(h5):
    """
    Extract useful scalars and slice names from a Genesis4 .par file
    """
    # Allow for opening a file
    if isinstance(h5, (str, Path)):
        h5 = File(h5, "r")

    params = {}
    for k in _parfile_scalar_datasets:
        if k not in h5:
            raise ValueError(f"Missing expected scalar dataset '{k}'")
        if h5[k].shape != (1,):
            raise ValueError(
                f"Expected scalar dataset '{k}' with shape (1,), got shape {h5[k].shape}"
            )
        params[k] = h5[k][0]

    return params


def genesis4_parfile_slice_groups(h5):
    """
    Slice group names ('slice000001', ...) in a Genesis4 .par file,
    sorted by slice index.

    Groups are matched by name pattern and structure rather than by
    excluding known metadata entries, so new metadata added by future
    Genesis4 versions is ignored automatically.
    """
    # Allow for opening a file
    if isinstance(h5, (str, Path)):
        h5 = File(h5, "r")

    return sorted(
        (
            g
            for g in h5
            if _parfile_slice_name_re.fullmatch(g) and isinstance(h5[g], Group)
        ),
        key=lambda s: int(s.removeprefix("slice")),
    )


def genesis4_parfile_slice_counts(h5) -> dict:
    """
    Particle count in each slice of a Genesis4 .par file.

    Reads only the slice dataset shapes (HDF5 metadata, no bulk particle
    data), so it is cheap even for very large files.

    Parameters
    ----------
    h5 : str, Path, or h5py.File
        Path to a Genesis4 `.par` file, or an open h5py file handle.

    Returns
    -------
    dict
        Mapping of slice group name to particle count, ordered by slice index.
    """
    # Allow for opening a file
    if isinstance(h5, (str, Path)):
        h5 = File(h5, "r")

    return {
        g: h5[g]["x"].shape[0]
        for g in genesis4_parfile_slice_groups(h5)
        if "x" in h5[g]
    }


def genesis4_parfile_n_particle(h5) -> int:
    """
    Total number of particles in a Genesis4 .par file.

    Reads only the slice dataset shapes (HDF5 metadata, no bulk particle data),
    so it is cheap even for very large files. Useful for choosing an
    `n_particle` subsample size for `genesis4_par_to_data` /
    `ParticleGroup.from_genesis4`.

    Parameters
    ----------
    h5 : str, Path, or h5py.File
        Path to a Genesis4 `.par` file, or an open h5py file handle.

    Returns
    -------
    int
        Total particle count summed over all slices.
    """
    return sum(genesis4_parfile_slice_counts(h5).values())


def load_parfile_slice_data(group, sel=None):
    """
    Returns ParfileSliceData with Genesis4's named fields in a slice group.

    Parameters
    ----------
    group : h5py.Group
        Slice group to read.
    sel : array of int, optional
        Sorted (increasing) particle row indices to read. If None, all
        particles in the slice are read.
    """
    if sel is None:
        values = [group[field][:] for field in PARFILE_SLICE_FIELDS[:-1]]
    else:
        values = [group[field][sel] for field in PARFILE_SLICE_FIELDS[:-1]]
    current = group["current"][:]  # This is really a scalar
    if len(current) != 1:
        raise ValueError(
            f"Expected a single 'current' value in {group.name}, found {len(current)}"
        )
    values.append(current[0])
    return ParfileSliceData(*values)


def genesis4_par_to_data(
    h5: str | Path | File,
    species: str = "electron",
    smear: bool = False,
    wrap: bool = False,
    z0: float = 0,
    slices: Optional[list[int]] = None,
    equal_weights: bool = False,
    cutoff: float = 0.0,
    n_particle: Optional[int] = None,
    rng: Optional[int | np.random.Generator] = None,
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
        particles have equal charge weights. Only existing particles are used
        (the beam is never upsampled), so the output count is generally
        smaller. Useful for simulations that require uniform macroparticles.
        Mutually exclusive with `n_particle`.

    cutoff : float, default=0.0
        Minimum per-particle weight in Coulombs. Quiet-loaded slices whose
        reconstructed weight falls below this value are skipped. By default
        (0.0) nothing is dropped for being sub-physical, leaving the choice to
        the user; pass a positive value (e.g. the elementary charge ~1.6e-19)
        to discard slices whose macroparticles would carry less than a single
        electron's charge. Zero-current and non-finite (nan) slices are always
        removed regardless of this value. The cutoff is ignored for one4one
        beams, where each macroparticle is a single real electron with weight
        exactly equal to the elementary charge.

    n_particle : int, optional
        Subsample the beam to exactly this many particles as it is read,
        retaining the per-slice weights (rescaled once so the total charge is
        conserved exactly). Each slice is thinned in proportion to its
        particle count, preserving the longitudinal/charge profile, and only
        the selected particles are read from the file — huge (e.g. one4one)
        files are downsampled immediately, without ever loading the full
        beam. Values >= the number of particles in the file are a no-op (all
        particles are returned). Use `genesis4_parfile_n_particle` to find
        the file's total count. Mutually exclusive with `equal_weights`.

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

    - Both quiet-loading and one4one beams are supported. In one4one mode each
      macroparticle represents a single electron, so the number of particles per
      slice varies with the current and some slices may be empty; empty slices are
      skipped and `equal_weights` handles the unequal per-slice counts.

    """
    # Allow for opening a file
    if isinstance(h5, (str, Path)):
        h5 = File(h5, "r")

    if species != "electron":
        raise ValueError("Only electrons supported for Genesis4")

    # Scalar arrays.
    # TODO: use refposition?
    scalars = [
        "beamletsize",
        "one4one",
        "refposition",
        "slicecount",
        "slicelength",
        "slicespacing",
    ]

    params = {}
    for k in scalars:
        if len(h5[k]) != 1:
            raise ValueError(f"Expected a single value for {k}, found {len(h5[k])}")
        params[k] = h5[k][0]

    # Useful local variables
    ds_slice = params["slicelength"]  # single slice length
    s_spacing = params["slicespacing"]  # spacing between slices
    sample = round(s_spacing / ds_slice)  # This should be an integer
    one4one = bool(params["one4one"])

    # In one4one mode each macroparticle is a single real electron with weight
    # exactly qe (see the weight assignment below), so the qe-scale cutoff would
    # be meaningless and is disabled; the non-finite / non-positive guard still
    # applies.
    effective_cutoff = 0.0 if one4one else cutoff

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

    if n_particle is not None and n_particle <= 0:
        raise ValueError(f"`n_particle` must be positive, got {n_particle}.")

    if n_particle is not None and equal_weights:
        raise ValueError(
            "`n_particle` and `equal_weights` are mutually exclusive. Use "
            "`n_particle` to thin a large file while retaining the per-slice "
            "weights, or `equal_weights` to resample the existing particles "
            "to a single common weight."
        )

    # Metadata pre-pass: from the per-slice counts (dataset shapes) and the
    # 1-element 'current' datasets — no bulk particle data — find the slices
    # that survive filtering, with their index, particle count, and
    # per-particle weight.
    counts_all = genesis4_parfile_slice_counts(h5)

    slice_names = []
    slice_ix = []
    slice_n = []
    slice_q = []
    for sname in snames:
        n1 = int(counts_all.get(sname, 0))

        # Skip missing and empty slices. In one4one mode the number of
        # particles per slice varies with the current, and low-current slices
        # can have no particles.
        if n1 == 0:
            continue

        g = h5[sname]
        if "current" not in g:
            continue

        current = g["current"][:]  # This is really a scalar
        if len(current) != 1:
            raise ValueError(
                f"Expected a single 'current' value in {g.name}, found {len(current)}"
            )

        # Convert current to weight (C)
        if one4one:
            # Each macroparticle is a single real electron, so its weight is
            # exactly the elementary charge. Genesis4 stores the smooth
            # requested current per slice, but populates it with round(ne)
            # integer particles; reconstructing q1 = ne*qe/round(ne) would
            # scatter the weights around qe. Genesis4's own &sort namelist
            # resolves this by resetting each slice current to npart*qe, which
            # is equivalent to assigning qe to every macroparticle here.
            q1 = e_charge
        else:
            # I * s_spacing/c = Q distributed over the n1 macroparticles.
            q1 = current[0] * s_spacing / c_light / n1

        # Skip subphysical or non-physical slices. The cutoff removes
        # quiet-loaded slices whose reconstructed weight is below the given
        # charge; zero-current slices (usually nans) and any non-finite
        # weight are always dropped. For one4one beams effective_cutoff is 0
        # and q1 == qe, so no slices are dropped here (empty slices were
        # already skipped above).
        if not np.isfinite(q1) or q1 <= 0 or q1 < effective_cutoff:
            continue

        slice_names.append(sname)
        slice_ix.append(int(sname.removeprefix("slice")))
        slice_n.append(n1)
        slice_q.append(q1)

    if not slice_names:
        raise ValueError(
            "No slices remain after filtering. All slices were empty, "
            f"zero-current, or below the cutoff ({cutoff} C)."
        )

    slice_n = np.array(slice_n, dtype=np.int64)
    slice_q = np.array(slice_q)
    total_count = int(slice_n.sum())
    full_charge = float(np.sum(slice_n * slice_q))  # before any subsampling

    # Optionally subsample on read to `n_particle` particles, apportioned over
    # the surviving slices in proportion to their counts, so slices dropped by
    # the filters above cannot consume any of the requested count. Systematic
    # (cumulative-rounding) apportionment preserves the longitudinal profile
    # while hitting the requested total exactly. Only the selected rows are
    # read from the file, so the full beam is never read or materialized; the
    # kept weights are rescaled once at the end to conserve the full-beam
    # charge exactly (and keep one4one weights uniform).
    keep_counts = None
    if n_particle is not None and n_particle < total_count:
        # Round the running cumulative target; consecutive differences give
        # each slice's keep count and sum exactly to n_particle.
        edges = np.round(np.cumsum(slice_n) / total_count * n_particle).astype(np.int64)
        keep_counts = np.diff(edges, prepend=0)

    for i, sname in enumerate(slice_names):
        n1 = int(slice_n[i])
        q1 = slice_q[i]
        ix = slice_ix[i]

        # Subsample this slice on read using its pre-computed keep count: only
        # the selected rows are read from the file. Dropped charge (including
        # whole slices that keep zero) is restored by a single global rescale
        # after the loop, so the total is conserved exactly and one4one
        # weights stay uniform.
        if keep_counts is not None:
            n_keep = int(keep_counts[i])
            if n_keep == 0:
                continue
            # h5py fancy indexing requires increasing indices; shuffle=False
            # is faster, and the order within a slice does not matter.
            sel = np.sort(rng.choice(n1, n_keep, replace=False, shuffle=False))
            pdata = load_parfile_slice_data(h5[sname], sel=sel)
            n1 = n_keep
        else:
            pdata = load_parfile_slice_data(h5[sname])

        # Calculate z
        irel = pdata.theta / (2 * np.pi)
        if wrap:
            irel = irel % 1  # Relative bin position (0,1)

        # Random smear over the `sample` slices each written slice represents
        if smear:
            z1 = (irel + rng.integers(0, sample, size=n1)) * ds_slice
        else:
            z1 = irel * ds_slice

        z1 = z1 + (ix - 1) * s_spacing + z0  # set absolute z

        # Collect arrays
        xs.append(pdata.x)
        pxs.append(pdata.px * mec2)
        ys.append(pdata.y)
        pys.append(pdata.py * mec2)
        gammas.append(pdata.gamma)
        zs.append(z1)
        weights.append(np.full(n1, q1))

    if equal_weights and len({float(w[0]) for w in weights}) > 1:
        # Resample each slice so that every output particle carries the same
        # weight. Slices may have different particle counts (e.g. in one4one
        # mode), so the sampling is done per slice. If every slice already has
        # the same per-particle weight (e.g. an unsubsampled one4one beam), the
        # output is already equal-weight and this block is skipped.
        n_slices = len(weights)

        counts = np.array([len(w) for w in weights])
        slice_charges = np.array([np.sum(w) for w in weights])

        # Highest feasible sampling rate (particles per Coulomb) that does not
        # require more particles than any slice actually has.
        rate = np.min(counts / slice_charges)
        n_samples = np.floor(rate * slice_charges).astype(int)

        total_charge = np.sum(slice_charges)
        n_samples_total = np.sum(n_samples)
        weight1 = total_charge / n_samples_total  # new equal weight

        # Loop over populated slices (note that some are skipped above)
        for i in range(n_slices):
            n_sample = n_samples[i]
            samples = rng.choice(counts[i], n_sample, replace=False)
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

    # If the beam was subsampled, rescale the kept weights so the total charge
    # exactly matches the full beam (this also keeps one4one weights uniform).
    if keep_counts is not None:
        weight = weight * (full_charge / weight.sum())

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


def load_genesis4_fields(h5):
    """
    Copied from: https://github.com/slaclab/lume-genesis/blob/52dc2815bb0bb42c1e057d1393c6ada07b8580b3/genesis/version4/readers.py#L8
    TODO: Point LUME-Genesis' function to here instead.

    Loads the field data into memory from an open h5 handle.

    Example usage:

    import h5py
    with h5py.File('rad_field.fld.h5', 'r') as h5:
        dfl, param = load_genesis4_fields(h5)

    Returns tuple (dfl, param) where

        dfl is a 3d complex dfl grid with shape (nx, ny, nz)

        param is a dict with:
            gridpoints:    number of gridpoints in one transverse dimension, equal to nx and ny above
            gridsize:      gridpoint spacing (meter)
            refposition:   starting position (meter)
            wavelength:    radiation wavelength (meter)
            slicecount:    number of slices
            slicespacing   slice spacing (meter)

        These params correspond to v2 params:
            gridpoints:   ncar
            gridsize:     dgrid*2 / (ncar-1)
            wavelength:   xlamds
            slicespacing: xlamds * zsep


    """

    # Get params
    param = {
        key: h5[key][0]
        for key in [
            "gridpoints",
            "gridsize",
            "refposition",
            "wavelength",
            "slicecount",
            "slicespacing",
        ]
    }

    # transverse grid points in each dimension
    nx = param["gridpoints"]

    # slice list
    slist = sorted(
        [
            g
            for g in h5
            if g.startswith("slice") and g not in ["slicecount", "slicespacing"]
        ]
    )

    # Note from Sven:
    #   The order of the 1D array of the wavefront is with the x coordinates as the inner loop.
    #   So the order is (x1,y1),(x2,y1), ... (xn,y1),(x1,y2),(x2,y2),.....
    #   This is done int he routine getLLGridpoint in the field class.
    # Therefore the transpose is needed below

    dfl = np.stack(
        [
            h5[g]["field-real"][:].reshape(nx, nx).T
            + 1j * h5[g]["field-imag"][:].reshape(nx, nx).T
            for g in slist
        ],
        axis=-1,
    )

    return dfl, param


def wavefront_write_genesis4(
    w,
    h5: File,
    polarization: str = None,
    refposition: float = 0,
) -> None:
    """
    Write the wavefront field data to a Genesis4-style HDF5 file.

    This function stores the full field data as a 3D array of complex numbers (`DFL`) in units of `sqrt(W)`,
    following the Genesis 4 format. The relation between the stored `DFL` data and the electric field `E` in V/m is:

    .. math::

        E = DFL \\times \\frac{\\sqrt{2Z_0}}{\\Delta}

    where `Z0` is the characteristic impedance of free space:

    .. math::

        Z_0 = \\pi \\times 119.9169832 \\text{ V}^2/\\text{W}

    and `Δ` represents the grid spacing.

    Parameters
    ----------
    w : Wavefront
        The `Wavefront` instance containing the field data to be written.

    h5 : h5py.File
        The HDF5 file object where the data will be stored in Genesis4 format.

    polarization : str, optional
        The polarization component to write. Must be either `"x"` or `"y"`. If `None`, the function
        will attempt to infer the correct component:
        - If only `Ex` exists, it will be written.
        - If only `Ey` exists, it will be written.
        - If both components exist, a `ValueError` is raised.

    refposition : float, optional
        The reference position in meters, stored as metadata in the output file. Default is `0`.

    Raises
    ------
    ValueError
        - If both `Ex` and `Ey` exist but no polarization is explicitly specified.
        - If `nx != ny`, as Genesis4 requires a square grid.
        - If `dx != dy`, as Genesis4 requires equal grid spacing in both transverse directions.
        - If `polarization` is specified but not `"x"` or `"y"`.

    Notes
    -----
    - The function ensures that the grid size and spacing meet Genesis4's requirements.
    - The data is stored in slices, following the indexing convention of Genesis4:
      The x-coordinates are stored as the inner loop, requiring a transpose before flattening.

    """
    nx, ny, nz = w.shape
    dx, dy, dz = w.dx, w.dy, w.dz
    wavelength = w.wavelength

    # Auto-select
    if polarization is None:
        if w.Ey is None:
            E = w.Ex
        elif w.Ex is None:
            E = w.Ey
        else:
            raise ValueError("Can only write one component: 'x' or 'y'")
    else:
        assert polarization in ("x", "y")
        if polarization == "x":
            E = w.Ex
        else:
            E = w.Ey

    dfl = E * dx / np.sqrt(2 * Z0)

    if nx != ny:
        raise ValueError(f"Genesis4 requires nx = ny. This data has {nx=}, {ny=}")

    if dx != dy:
        raise ValueError(f"Genesis4 requires dx = dy. This data has {dx=}, {dy=}")

    h5["gridpoints"] = np.asarray([nx])
    h5["gridsize"] = np.asarray([dx])
    h5["refposition"] = np.asarray([refposition])
    h5["wavelength"] = np.asarray([wavelength])
    h5["slicecount"] = np.asarray([nz])
    h5["slicespacing"] = np.asarray([dz])

    # Note from Sven:
    #   The order of the 1D array of the wavefront is with the x
    #   coordinates as the inner loop.
    #   So the order is (x1,y1),(x2,y1), ... (xn,y1),(x1,y2),(x2,y2),.....
    #   This is done in the routine getLLGridpoint in the field class.
    # Therefore the transpose is needed below
    for z in range(nz):
        slice_index = z + 1
        slice_group = h5.create_group(f"slice{slice_index:06}")
        slice_group["field-real"] = dfl[:, :, z].real.T.flatten()
        slice_group["field-imag"] = dfl[:, :, z].imag.T.flatten()
