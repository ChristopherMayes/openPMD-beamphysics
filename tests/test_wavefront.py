import copy
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pmd_beamphysics import Wavefront
from pmd_beamphysics.wavefront import (
    Plane,
    WavefrontPadding,
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
        dims=(11, 21, 21),
        wavelength=1.35e-8,
        grid_spacing=(4.54, 2.9e-5, 2.9e-5),
        pad=(40, 100, 100),
        nphotons=1e12,
        zR=2.0,
        sigma_t=5.0,
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


def test_smoke_drift_z_in_place(wavefront: Wavefront) -> None:
    # Implicitly calculates the FFT:
    wavefront.drift(direction="z", distance=0.0, inplace=True)
    # Use the property to calculate the inverse fft:
    wavefront.field_rspace


def test_smoke_drift_z(wavefront: Wavefront) -> None:
    new = wavefront.drift(direction="z", distance=0.0, inplace=False)
    assert new is not wavefront


def test_smoke_focusing_element_in_place(wavefront: Wavefront) -> None:
    wavefront.focus(plane="xy", focus=(1.0, 1.0), inplace=True)


def test_smoke_focusing_element(wavefront: Wavefront) -> None:
    new = wavefront.focus(plane="xy", focus=(1.0, 1.0), inplace=False)
    assert new is not wavefront


@pytest.mark.parametrize(
    ("padding", "expected"),
    [
        pytest.param(
            WavefrontPadding(grid=(10,), pad=(10,)),
            WavefrontPadding(grid=(10,), pad=(11,)),
            id="1d",
        ),
        pytest.param(
            WavefrontPadding(grid=(10, 10), pad=(10, 10)),
            WavefrontPadding(grid=(10, 10), pad=(11, 11)),
            id="2d",
        ),
    ],
)
def test_padding_fix(padding: WavefrontPadding, expected: WavefrontPadding) -> None:
    assert padding.fix() == expected


def test_smoke_properties(wavefront: Wavefront) -> None:
    assert len(wavefront.phasors) == 3
    assert wavefront.field_rspace.shape == (11, 21, 21)
    assert wavefront.field_kspace.shape == wavefront.pad.get_padded_shape(
        wavefront.field_rspace
    )
    assert np.isclose(wavefront.wavelength, 1.35e-8)
    assert wavefront.pad.grid == (11, 21, 21)
    assert wavefront.pad.pad == (44, 100, 100)


def test_copy(wavefront: Wavefront) -> None:
    wavefront.field_rspace
    wavefront.field_kspace
    copied = copy.copy(wavefront)
    assert copied == wavefront
    assert copied is not wavefront
    assert copied.field_rspace is wavefront.field_rspace
    assert copied.field_kspace is wavefront.field_kspace


def test_deepcopy(wavefront: Wavefront) -> None:
    wavefront.field_rspace
    wavefront.field_kspace
    copied = copy.deepcopy(wavefront)
    assert copied == wavefront
    assert copied is not wavefront
    assert copied.field_rspace is not wavefront.field_rspace
    assert copied.field_kspace is not wavefront.field_kspace


def test_plot_projection(wavefront: Wavefront, projection_plane: Plane) -> None:
    wavefront.plot(projection_plane, rspace=True)
    plt.suptitle(f"rspace - {projection_plane}")
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
