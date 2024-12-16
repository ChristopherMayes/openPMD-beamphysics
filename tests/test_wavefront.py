import copy
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pmd_beamphysics import Wavefront
from pmd_beamphysics.wavefront import (
    Plane,
    fix_padding,
    get_padded_shape,
    get_num_fft_workers,
    get_range_for_grid_spacing,
    set_num_fft_workers,
)


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
    params=["xy"],
)
def projection_plane(request: pytest.FixtureRequest) -> str:
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


def test_plot_projection_rspace(wavefront: Wavefront, projection_plane: Plane) -> None:
    wavefront.plot(projection_plane, rspace=True)
    plt.suptitle(f"rspace - {projection_plane}")


@pytest.mark.xfail(reason="vstack shape")
def test_plot_projection_kspace(wavefront: Wavefront, projection_plane: Plane) -> None:
    # TODO check/fix
    wavefront.plot(projection_plane, rspace=False)
    plt.suptitle(f"kspace - {projection_plane}")


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

    # Differences in sign make this not quite equal:
    # E             array([[[0.+0.j, 0.+0.j, 0.+0.j, ..., 0.+0.j, 0.+0.j, 0.+0.j],
    # E           -         [0.+0.j, 0.+0.j, 0.+0.j, ..., 0.+0.j, 0.+0.j, 0.+0.j],
    # E           ?            ^       ^       ^            ^       ^       ^
    # E           +         [0.-0.j, 0.-0.j, 0.-0.j, ..., 0.-0.j, 0.-0.j, 0.-0.j],
    # E           ?            ^       ^       ^            ^       ^       ^...
    # Should probably switch to `allclose` in `__eq__` but I think Chris has a
    # preference for true equality
    # assert np.allclose(wavefront.rmesh.real, loaded.rmesh.real)
    # assert np.allclose(wavefront.rmesh.imag, loaded.rmesh.imag)
    # loaded._rmesh = wavefront.rmesh
    # assert wavefront == loaded
