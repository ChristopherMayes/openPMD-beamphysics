import copy

import numpy as np
import pytest

from pmd_beamphysics import Wavefront
from pmd_beamphysics.wavefront import (
    WavefrontPadding,
    get_num_fft_workers,
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
        ranges=((0.0, 50.0), (-3e-4, 3e-4), (-3e-4, 3e-4)),
        pad=(40, 100, 100),
        nphotons=1e12,
        zR=2.0,
        sigma_t=5.0,
    )


def test_smoke_propagate_z_in_place(wavefront: Wavefront) -> None:
    # Implicitly calculates the FFT:
    wavefront.propagate(direction="z", distance=0.0, inplace=True)
    # Use the property to calculate the inverse fft:
    wavefront.field_rspace


def test_smoke_propagate_z(wavefront: Wavefront) -> None:
    new = wavefront.propagate(direction="z", distance=0.0, inplace=False)
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
            WavefrontPadding(grid=(11,), pad=(11,)),
            id="1d",
        ),
        pytest.param(
            WavefrontPadding(grid=(10, 10), pad=(10, 10)),
            WavefrontPadding(grid=(11, 11), pad=(11, 11)),
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
    assert wavefront.pad.pad == (44, 102, 102)


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
