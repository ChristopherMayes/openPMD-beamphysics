import copy
import pathlib
import typing

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pmd_beamphysics import Wavefront
from pmd_beamphysics.wavefront import (
    Plane,
    PlotKey,
    fix_padding,
    get_padded_shape,
    get_range_for_grid_spacing,
    wavefront_ids_from_file,
)
from pmd_beamphysics.tools import get_num_fft_workers, set_num_fft_workers


@pytest.fixture(autouse=True)
def _set_fft_jobs() -> None:
    set_num_fft_workers(1)
    assert get_num_fft_workers() == 1


@pytest.fixture
def wavefront() -> Wavefront:
    return Wavefront.gaussian_pulse(
        dims=(21, 21, 11),
        wavelength=1.35e-8,
        grid_spacing=(2.9e-5, 2.9e-5, 4.54),
        pad=(100, 100, 40),
        nphotons=1e12,
        zR=2.0,
        sigma_z=2.29e-7,
    )


@pytest.fixture(
    params=["xy", "yz", "xz", "kxky", "kykz", "kxkz"],
)
def projection_plane(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(params=typing.get_args(PlotKey))
def plot_key(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.mark.parametrize(
    ("grid_spacing", "dim", "expected_low", "expected_high"),
    [
        pytest.param(
            range_[1] - range_[0],
            len(range_),
            range_[0],
            range_[-1],
            id=f"-{range_[0]}_to_{range_[-1]}_{len(range_)}_steps",
        )
        for range_ in [
            np.linspace(-50, 50, 101),
            np.linspace(-50, 49, 100),
        ]
    ],
)
def test_get_range_for_grid_spacing(
    grid_spacing: float,
    dim: int,
    expected_low: float,
    expected_high: float,
) -> None:
    low, high = get_range_for_grid_spacing(grid_spacing=grid_spacing, dim=dim)
    assert (low, high) == (expected_low, expected_high)


def test_smoke_drift_z(wavefront: Wavefront) -> None:
    new = wavefront.drift(distance=0.0)
    assert new is not wavefront


def test_smoke_focusing_element_in_place(wavefront: Wavefront) -> None:
    wavefront.focus(plane="xy", focus=(1.0, 1.0))


def test_smoke_focusing_element(wavefront: Wavefront) -> None:
    new = wavefront.focus(plane="xy", focus=(1.0, 1.0))
    assert new is not wavefront


def test_other_type_equality(wavefront: Wavefront) -> None:
    assert wavefront != 0


def test_smoke_repr(wavefront: Wavefront) -> None:
    print(repr(wavefront))
    print(str(wavefront))


@pytest.mark.parametrize(
    ("grid", "padding", "expected_padding"),
    [
        pytest.param(
            (10,),
            (10,),
            (11,),
            id="1d",
        ),
        pytest.param(
            (10, 10),
            (10, 10),
            (11, 11),
            id="2d",
        ),
    ],
)
def test_padding_fix(
    grid: tuple[int, ...], padding: tuple[int, ...], expected_padding: tuple[int, ...]
) -> None:
    fixed = fix_padding(grid, padding)
    assert fixed == expected_padding


def test_smoke_properties(wavefront: Wavefront) -> None:
    # assert len(wavefront.phasors) == 3
    assert wavefront.rmesh.shape == (21, 21, 11)

    padded_shape = get_padded_shape(wavefront.grid, padding=wavefront.pad)
    assert wavefront.kmesh.shape == padded_shape
    assert np.isclose(wavefront.wavelength, 1.35e-8)
    assert wavefront.grid == (21, 21, 11)
    assert wavefront.pad == (100, 100, 44)

    for dim, pad in zip(wavefront.grid, wavefront.pad):
        assert (dim + pad) % 2 == 1


def test_smoke_ifft(wavefront: Wavefront) -> None:
    assert wavefront.rmesh.shape == (21, 21, 11)
    wavefront.kmesh  # FFT to generate kmesh
    # invalidate the rmesh
    wavefront._rmesh = None
    wavefront.rmesh  # IFFT to generate rmesh


def test_copy(wavefront: Wavefront) -> None:
    wavefront.rmesh
    wavefront.kmesh
    copied = copy.copy(wavefront)
    assert copied == wavefront
    assert copied is not wavefront
    assert copied.rmesh is wavefront.rmesh
    assert copied.kmesh is wavefront.kmesh


def test_deepcopy(wavefront: Wavefront) -> None:
    wavefront.rmesh
    wavefront.kmesh
    copied = copy.deepcopy(wavefront)
    assert copied == wavefront
    assert copied is not wavefront
    assert copied.rmesh is not wavefront.rmesh
    assert copied.kmesh is not wavefront.kmesh


def test_plot_projection_rspace(
    wavefront: Wavefront, projection_plane: Plane, plot_key: PlotKey
) -> None:
    wavefront.plot(plot_key, projection=projection_plane)
    plt.suptitle(f"- {projection_plane}")


def test_write_and_validate(
    wavefront: Wavefront,
    tmp_path: pathlib.Path,
    request: pytest.FixtureRequest,
):
    from openpmd_validator.check_h5 import check_file as pmd_validator

    fn = tmp_path / f"{request.node.name}.h5"
    wavefront.write(fn)

    errors, warnings, *_ = pmd_validator(fn, verbose=True)
    assert errors == 0
    assert warnings == 0


def test_write_and_read_genesis4(
    wavefront: Wavefront,
    tmp_path: pathlib.Path,
    request: pytest.FixtureRequest,
):
    fn = tmp_path / f"{request.node.name}.h5"
    wavefront.metadata.mesh.grid_global_offset = (0.0, 0.0, 0.0)

    wavefront.write_genesis4(fn)
    loaded = wavefront.from_genesis4(fn, pad=wavefront.pad)

    # TODO date is not stored in Genesis4 file
    loaded.metadata.base.date = wavefront.metadata.base.date

    assert wavefront.grid == loaded.grid
    # assert np.all(wavefront._rmesh == loaded._rmesh)
    # assert np.all(wavefront._kmesh == loaded._kmesh)
    assert wavefront.wavelength == loaded.wavelength
    assert wavefront.pad == loaded.pad
    assert wavefront.metadata == loaded.metadata

    assert np.allclose(wavefront.rmesh.real, loaded.rmesh.real)
    assert np.allclose(wavefront.rmesh.imag, loaded.rmesh.imag)
    loaded._rmesh = wavefront.rmesh
    assert wavefront == loaded


def test_write_and_read_genesis4_legacy_openpmd(
    wavefront: Wavefront,
    tmp_path: pathlib.Path,
    request: pytest.FixtureRequest,
):
    fn = tmp_path / f"{request.node.name}.h5"
    field_file = wavefront.to_genesis4_fieldfile()

    field_file.write_openpmd_wavefront(dest=fn)

    wavefront.metadata.mesh.grid_global_offset = (0.0, 0.0, 0.0)

    loaded = wavefront.from_file(fn).with_padding(wavefront.pad)

    # check these individually before testing full equality, so we don't get just a final failure
    assert wavefront.grid == loaded.grid
    assert np.allclose(wavefront.rmesh.real, loaded.rmesh.real)
    assert np.allclose(wavefront.rmesh.imag, loaded.rmesh.imag)
    assert np.isclose(wavefront.wavelength, loaded.wavelength)
    assert wavefront.pad == loaded.pad

    # NOTE the date isn't retained in the old format
    loaded.metadata.base.date = wavefront.metadata.base.date
    assert wavefront.metadata.base == loaded.metadata.base
    assert wavefront.metadata.iteration == loaded.metadata.iteration
    assert wavefront.metadata.mesh == loaded.metadata.mesh
    assert wavefront.metadata == loaded.metadata

    # Overwrite the above stuff that's close but not equal:
    loaded._rmesh = wavefront._rmesh
    loaded._kmesh = wavefront._kmesh
    loaded.wavelength = wavefront.wavelength
    assert wavefront == loaded


def detailed_compare(W1: Wavefront, W2: Wavefront) -> None:
    assert W1.grid == W2.grid
    assert np.all(W1._rmesh == W2._rmesh)
    assert np.all(W1._kmesh == W2._kmesh)
    assert W1.wavelength == W2.wavelength
    # TODO  we don't store padding
    assert W1.pad == W2.pad

    # TODO: we don't store microseconds
    # NOTE: the date should be retained here as when it was originally stored,
    # minus microseconds as above
    W2.metadata.base.date = W2.metadata.base.date.replace(
        microsecond=W1.metadata.base.date.microsecond
    )
    assert W1.metadata.base == W2.metadata.base
    assert W1.metadata.iteration == W2.metadata.iteration
    assert W1.metadata.mesh == W2.metadata.mesh
    assert W1.metadata == W2.metadata
    assert W1 == W2


def test_write_and_read_openpmd(
    wavefront: Wavefront,
    tmp_path: pathlib.Path,
    request: pytest.FixtureRequest,
):
    fn = tmp_path / f"{request.node.name}.h5"
    wavefront.metadata.mesh.grid_global_offset = (0.0, 0.0, 0.0)

    wavefront.write(fn)
    # TODO  we don't store padding
    loaded = wavefront.from_file(fn).with_padding(wavefront.pad)

    # TODO: we don't store microseconds
    # NOTE: the date should be retained here as when it was originally stored,
    # minus microseconds as above
    loaded.metadata.base.date = loaded.metadata.base.date.replace(
        microsecond=wavefront.metadata.base.date.microsecond
    )
    detailed_compare(wavefront, loaded)


def test_legacy_wavefront_ids_from_file(
    wavefront: Wavefront,
    tmp_path: pathlib.Path,
    request: pytest.FixtureRequest,
):
    fn = tmp_path / f"{request.node.name}.h5"
    field_file = wavefront.to_genesis4_fieldfile()
    field_file.write_openpmd_wavefront(dest=fn)

    assert wavefront_ids_from_file(fn) == ["000000"]


def test_wavefront_ids_from_file(
    wavefront: Wavefront,
    tmp_path: pathlib.Path,
    request: pytest.FixtureRequest,
):
    for iteration in range(3):
        fn = tmp_path / f"{request.node.name}-{iteration}.h5"
        wavefront.metadata.iteration.iteration = iteration
        wavefront.write(fn)

        assert wavefront_ids_from_file(fn) == [str(iteration)]


def test_plot_1d_kmesh_angular_divergence(
    wavefront: Wavefront,
    tmp_path: pathlib.Path,
    request: pytest.FixtureRequest,
) -> None:
    wavefront.plot_1d_kmesh_angular_divergence(
        save=tmp_path / f"{request.node.name}.png"
    )


def test_plot_1d_kmesh_spectrum(
    wavefront: Wavefront,
    tmp_path: pathlib.Path,
    request: pytest.FixtureRequest,
) -> None:
    wavefront.plot_1d_kmesh_spectrum(save=tmp_path / f"{request.node.name}.png")


def test_plot_1d_far_field_spectral_density(
    wavefront: Wavefront,
    tmp_path: pathlib.Path,
    request: pytest.FixtureRequest,
) -> None:
    wavefront.plot_1d_far_field_spectral_density(
        save=tmp_path / f"{request.node.name}.png"
    )


def test_plot_reciprocal(
    wavefront: Wavefront,
    tmp_path: pathlib.Path,
    request: pytest.FixtureRequest,
) -> None:
    wavefront.plot_reciprocal(save=tmp_path / f"{request.node.name}.png")


def test_from_metadata_dict(wavefront: Wavefront) -> None:
    md = wavefront.metadata.to_dict()
    W = Wavefront(
        rmesh=wavefront.rmesh,
        wavelength=wavefront.wavelength,
        metadata=md,
        pad=wavefront.pad,
    )
    W.metadata.base.date = W.metadata.base.date.replace(
        microsecond=wavefront.metadata.base.date.microsecond
    )
    detailed_compare(wavefront, W)


@pytest.fixture
def wigner_wavefront() -> Wavefront:
    return Wavefront.gaussian_pulse(
        dims=(101, 101, 513),
        wavelength=1.35e-8,
        grid_spacing=(6e-6, 6e-6, 2.9333e-8),
        pad=(100, 100, 256),
        nphotons=1e12,
        zR=2.0,
        sigma_z=2.29e-7,
    )


def test_plot_wigner(wigner_wavefront: Wavefront) -> None:
    phi = 2.0 * np.pi * (0.5 * wigner_wavefront.rmesh) ** 3
    W = wigner_wavefront.with_rmesh(wigner_wavefront.rmesh * np.exp(1j * phi))
    W.plot_wigner_distribution()
