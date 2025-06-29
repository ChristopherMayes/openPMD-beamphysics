from math import pi
from dataclasses import replace

import numpy as np

from pmd_beamphysics.wavefront.wavefront import Wavefront


def drift_wavefront(w: Wavefront, z, backend=np, device="cpu"):
    """
    Propagate a wavefront `w` by distance `z` in real space,
    modifying each slice along the z-dimension.

    Paraxial approximation

    In k-space:
    Ehat -> Ehat * exp( -i z (kx^2 + ky^2)/ (2 k0))

    Experimental: works with torch and device='mps'

    """

    fft2 = backend.fft.fft2
    ifft2 = backend.fft.ifft2

    # Handle torch's 'mps'
    def fftfreq(*args, **kwargs):
        if device == "cpu":
            x = backend.fft.fftfreq(*args, **kwargs)
        else:
            x = backend.fft.fftfreq(*args, **kwargs, device=device)
        return x

    if w.in_kspace:
        # The fields only need the phase factor here
        kx, ky, _ = backend.meshgrid(w.kxvec, w.kyvec, w.kzvec, indexing="ij")
        kernel = backend.exp(-1j * z * w.wavelength * (kx**2 + ky**2) / (4 * np.pi))
        new_fields = []
        for field in (w.Ex, w.Ey):
            if field is None:
                new_fields.append(None)
                continue
            new_fields.append(kernel * field)
        Ex_drifted, Ey_drifted = new_fields
        return replace(w, Ex=Ex_drifted, Ey=Ey_drifted)

    # r-space calculation

    # Make simple kx vecs here.
    # No need for fftshift, we will fft and ifft all within this function
    kx_vec = fftfreq(w.nx, d=w.dx) * 2 * pi
    ky_vec = fftfreq(w.ny, d=w.dy) * 2 * pi
    kx, ky = backend.meshgrid(kx_vec, ky_vec, indexing="ij")

    # Compute the phase kernel for propagation
    kernel = backend.exp(-1j * z * w.wavelength * (kx**2 + ky**2) / (4 * backend.pi))

    # Propagate Ex and Ey
    new_fields = []
    for field in (w.Ex, w.Ey):
        if field is None:
            new_fields.append(None)
            continue

        # Allocate an output array of the same shape/dtype
        # Handle dtype (TODO: more general)
        if device == "mps" or field.dtype == backend.complex64:
            dtype = backend.complex64
        elif field.dtype == backend.float32:
            dtype = backend.complex64
        elif field.dtype == backend.float64:
            dtype = backend.complex128
        else:
            dtype = backend.complex128

        field_out = backend.empty_like(field, dtype=dtype)

        # Apply to each slice
        for iz in range(field.shape[2]):
            kmesh2 = fft2(field[:, :, iz])
            field_out[:, :, iz] = ifft2(kernel * kmesh2)

        new_fields.append(field_out)

    Ex_drifted, Ey_drifted = new_fields

    return replace(w, Ex=Ex_drifted, Ey=Ey_drifted)


def drift_wavefront_advanced(w: Wavefront, z, Rcurv=2):
    """
    Work in progress
    """
    if z == 0:
        return w.copy()

    x_mesh, y_mesh, _ = np.meshgrid(w.xvec, w.yvec, w.zvec, indexing="ij")

    curv = np.exp(-1j * np.pi * (x_mesh**2 + y_mesh**2) / w.wavelength / Rcurv)

    w = replace(w)
    w.Ex = w.Ex * curv if w.Ex is not None else None
    w.Ey = w.Ey * curv if w.Ey is not None else None

    z_eff = Rcurv * z / (Rcurv + z)  # float
    M = z_eff / z  # float

    w = drift_wavefront(w, z_eff)

    print("effective propagation distance: ", z_eff, "scaling factor: ", M)
    Fr = np.exp(
        -1j
        * np.pi
        / z
        / w.wavelength
        * (1 - M)
        * ((x_mesh / M) ** 2 + (y_mesh / M) ** 2)
    )

    w.Ex = w.Ex / Fr * M if w.Ex is not None else None
    w.Ey = w.Ey / Fr * M if w.Ey is not None else None
    w.dx = w.dx / M
    w.dy = w.dy / M

    return w
