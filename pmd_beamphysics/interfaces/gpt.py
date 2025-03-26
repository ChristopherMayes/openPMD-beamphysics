import os
import subprocess

import numpy as np

from pmd_beamphysics.interfaces.superfish import fish_complex_to_real_fields
from pmd_beamphysics.units import c_light, e_charge


def write_gpt(particle_group, outfile, asci2gdf_bin=None, verbose=False):
    """
    GPT uses a custom binary format GDF for particle data. This can be created with the
    asci2gdf utility as:
    asci2gdf -o particles.gdf particles.txt

    This routine makes ASCII particles, with column labels:
        'x', 'y', 'z', 'GBx', 'GBy', 'GBz', 't', 'q', 'm', 'nmacro'
    in SI units.

    """

    assert np.all(particle_group.weight >= 0), "ParticleGroup.weight must be >= 0"

    q = particle_group.species_charge
    mc2 = (
        particle_group.mass
    )  # returns pmd_beamphysics.species.mass_of(particle_group.species) [eV]
    m = mc2 * (e_charge / c_light**2)
    n = particle_group.n_particle
    gamma = particle_group.gamma

    dat = {
        "x": particle_group.x,
        "y": particle_group.y,
        "z": particle_group.z,
        "GBx": gamma * particle_group.beta_x,
        "GBy": gamma * particle_group.beta_y,
        "GBz": gamma * particle_group.beta_z,
        "t": particle_group.t,
        "q": np.full(n, q),
        "m": np.full(n, m),
        "nmacro": np.abs(particle_group.weight / q),
    }

    if hasattr(particle_group, "id"):
        dat["ID"] = particle_group.id
    else:
        dat["ID"] = np.arange(1, particle_group["n_particle"] + 1)

    header = " ".join(list(dat))

    outdat = np.array([dat[k] for k in dat]).T

    if verbose:
        print(f"writing {n} particles to {outfile}")

    # Write ASCII
    np.savetxt(outfile, outdat, header=header, comments="", fmt="%20.12e")

    if asci2gdf_bin:
        run_asci2gdf(outfile, asci2gdf_bin)
    else:
        print(
            f"ASCII particles written. Convert to GDF using: asci2df -o particles.gdf {outfile}"
        )


def run_asci2gdf(outfile, asci2gdf_bin, verbose=False):
    """
    Helper function to convert ASCII to GDF using asci2gdf
    """
    tempfile = outfile + ".txt"
    os.rename(outfile, tempfile)

    asci2gdf_bin = os.path.expandvars(asci2gdf_bin)
    assert os.path.exists(asci2gdf_bin), f"{asci2gdf_bin} does not exist"
    cmd = [asci2gdf_bin, "-o", outfile, tempfile]
    if verbose:
        print(" ".join(cmd))
    subprocess.run(cmd)

    # Cleanup
    os.remove(tempfile)

    if verbose:
        print("Written GDF file:", outfile)

    return outfile


def write_gpt_fieldmap(fm, outfile, asci2gdf_bin=None, verbose=False):
    """
    Writes a GPT fieldmap file from a FieldMesh object.

    Requires cylindrical geometry for now.
    """

    if fm.geometry == "cylindrical" and fm.coord_vec("r")[0] == fm.coord_vec("r")[-1]:
        return write_gpt_1d_fieldmap(
            fm, outfile, asci2gdf_bin=asci2gdf_bin, verbose=verbose
        )

    elif fm.geometry == "cylindrical":
        return write_gpt_2d_fieldmap(
            fm, outfile, asci2gdf_bin=asci2gdf_bin, verbose=verbose
        )

    elif fm.geometry == "rectangular":
        return write_gpt_3d_fieldmap(
            fm, outfile, asci2gdf_bin=asci2gdf_bin, verbose=verbose
        )

    else:
        raise ValueError(f"Unknown geometry {fm.geometry}")


def write_gpt_1d_fieldmap(fm, outfile, asci2gdf_bin=None, verbose=False):
    """
    Writes a GPT fieldmap file from a FieldMesh object.

    Requires cylindrical geometry for now.
    """

    assert fm.geometry == "cylindrical", f"Geometry: {fm.geometry} not implemented"

    assert fm.shape[1] == 1, "Cylindrical symmetry required"

    assert fm.coord_vec("r")[0] == 0, "r[0] must equal 0"

    dat = {}
    dat["Z"] = fm.coord_vec("z")

    keys = ["Z"]
    if fm.is_static:
        if fm.is_pure_magnetic:
            keys = ["Z", "Bz"]
            dat["Bz"] = np.real(fm["Bz"][0, 0, :])
        elif fm.is_pure_electric:
            keys = ["Z", "Ez"]
            dat["Er"] = np.real(fm["Er"][0, 0, :])
            dat["Ez"] = np.real(fm["Ez"][0, 0, :])
        else:
            raise ValueError("Mixed static field TODO")

    else:
        # Use internal Superfish routine
        keys = ["Z", "Ez"]
        _, dat["Ez"], _, _ = fish_complex_to_real_fields(fm, verbose=verbose)
        dat["Ez"] = dat["Ez"][0, 0, :]

    # Flatten dat
    gptdata = np.array([dat[k].flatten() for k in keys]).T

    # Write file.
    # Hack to delete final newline
    # https://stackoverflow.com/questions/28492954/numpy-savetxt-stop-newline-on-final-line
    with open(outfile, "w") as fout:
        NEWLINE_SIZE_IN_BYTES = 1  # 2 on Windows?
        np.savetxt(fout, gptdata, header=" ".join(keys), comments="")
        fout.seek(0, os.SEEK_END)  # Go to the end of the file.
        # Go backwards one byte from the end of the file.
        fout.seek(fout.tell() - NEWLINE_SIZE_IN_BYTES, os.SEEK_SET)
        fout.truncate()  # Truncate the file to this point.

    if asci2gdf_bin:
        run_asci2gdf(outfile, asci2gdf_bin, verbose=verbose)
    elif verbose:
        print(
            f"ASCII field data written. Convert to GDF using: asci2df -o field.gdf {outfile}"
        )

    return outfile


def write_gpt_2d_fieldmap(fm, outfile, asci2gdf_bin=None, verbose=False):
    """
    Writes a GPT fieldmap file from a FieldMesh object.

    Requires cylindrical geometry for now.
    """

    assert fm.geometry == "cylindrical", f"Geometry: {fm.geometry} not implemented"

    assert fm.shape[1] == 1, "Cylindrical symmetry required"

    dat = {}
    dat["R"], dat["Z"] = np.meshgrid(
        fm.coord_vec("r"), fm.coord_vec("z"), indexing="ij"
    )

    keys = ["R", "Z"]
    if fm.is_static:
        if fm.is_pure_magnetic:
            keys = ["R", "Z", "Br", "Bz"]
            dat["Br"] = np.real(fm["Br"][:, 0, :])
            dat["Bz"] = np.real(fm["Bz"][:, 0, :])
        elif fm.is_pure_electric:
            keys = ["R", "Z", "Er", "Ez"]
            dat["Er"] = np.real(fm["Er"][:, 0, :])
            dat["Ez"] = np.real(fm["Ez"][:, 0, :])
        else:
            raise ValueError("Mixed static field TODO")

    else:
        # Use internal Superfish routine
        keys = ["R", "Z", "Er", "Ez", "Bphi"]
        dat["Er"], dat["Ez"], dat["Bphi"], _ = fish_complex_to_real_fields(
            fm, verbose=verbose
        )

    # Flatten dat
    gptdata = np.array([dat[k].flatten() for k in keys]).T

    # Write file.
    # Hack to delete final newline
    # https://stackoverflow.com/questions/28492954/numpy-savetxt-stop-newline-on-final-line
    with open(outfile, "w") as fout:
        NEWLINE_SIZE_IN_BYTES = 1  # 2 on Windows?
        np.savetxt(fout, gptdata, header=" ".join(keys), comments="")
        fout.seek(0, os.SEEK_END)  # Go to the end of the file.
        # Go backwards one byte from the end of the file.
        fout.seek(fout.tell() - NEWLINE_SIZE_IN_BYTES, os.SEEK_SET)
        fout.truncate()  # Truncate the file to this point.

    if asci2gdf_bin:
        run_asci2gdf(outfile, asci2gdf_bin, verbose=verbose)

    elif verbose:
        print(
            f"ASCII field data written. Convert to GDF using: asci2df -o field.gdf {outfile}"
        )

    return outfile


def write_gpt_3d_fieldmap(fm, outfile, asci2gdf_bin=None, verbose=False):
    """
    Writes a 3D GPT fieldmap file from a FieldMesh object.
    """

    assert fm.geometry == "rectangular", f"Geometry: {fm.geometry} not implemented"

    dat = {}
    dat["X"], dat["Y"], dat["Z"] = np.meshgrid(
        fm.coord_vec("x"), fm.coord_vec("y"), fm.coord_vec("z"), indexing="ij"
    )

    keys = ["X", "Y", "Z"]
    if fm.is_static:
        if fm.is_pure_magnetic:
            keys = keys + ["Bx", "By", "Bz"]
            dat["Bx"] = np.real(fm["Bx"])
            dat["By"] = np.real(fm["By"])
            dat["Bz"] = np.real(fm["Bz"])

        elif fm.is_pure_electric:
            keys = keys + ["Ex", "Ey", "Ez"]
            dat["Ex"] = np.real(fm["Ex"])
            dat["Ey"] = np.real(fm["Ey"])
            dat["Ez"] = np.real(fm["Ez"])
        else:
            raise ValueError("Mixed static field TODO")

    else:
        # Use internal Superfish routine
        raise ValueError("Complex 3D Fields not implement yet!")
        # keys = ['X', 'Y', 'Z', + ['Ex', 'Ey', 'Ez']'Bx', 'By', 'Bz']
        # dat['Er'], dat['Ez'], dat['Bphi'], _ = fish_complex_to_real_fields(fm, verbose=verbose)

    # Flatten dat
    gptdata = np.array([dat[k].flatten() for k in keys]).T

    # Write file.
    # Hack to delete final newline
    # https://stackoverflow.com/questions/28492954/numpy-savetxt-stop-newline-on-final-line
    with open(outfile, "w") as fout:
        NEWLINE_SIZE_IN_BYTES = 1  # 2 on Windows?
        np.savetxt(fout, gptdata, header=" ".join(keys), comments="")
        fout.seek(0, os.SEEK_END)  # Go to the end of the file.
        # Go backwards one byte from the end of the file.
        fout.seek(fout.tell() - NEWLINE_SIZE_IN_BYTES, os.SEEK_SET)
        fout.truncate()  # Truncate the file to this point.

    if asci2gdf_bin:
        run_asci2gdf(outfile, asci2gdf_bin, verbose=verbose)
    elif verbose:
        print(
            f"ASCII field data written. Convert to GDF using: asci2df -o field.gdf {outfile}"
        )

    return outfile
