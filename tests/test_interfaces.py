from __future__ import annotations

import numpy as np
import pytest

from beamphysics import ParticleGroup


def make_particle_group(pz):
    n = len(pz)
    data = {
        "x": np.zeros(n),
        "px": np.zeros(n),
        "y": np.zeros(n),
        "py": np.zeros(n),
        "z": np.zeros(n),
        "pz": np.array(pz, dtype=float),
        "t": np.zeros(n),
        "status": np.ones(n, dtype=int),
        "weight": np.full(n, 1e-12),
        "species": "electron",
    }
    return ParticleGroup(data=data)


def test_write_impact_does_not_mutate_caller(tmp_path):
    """The cathode-start small-pz shift must act on a copy, not the
    caller's internal pz array."""
    from beamphysics.interfaces.impact import write_impact

    P = make_particle_group([5.0, 1e6, 1e6])
    pz_before = P.pz.copy()
    write_impact(P, tmp_path / "impact.in", cathode_kinetic_energy_ref=1.0)
    np.testing.assert_array_equal(P.pz, pz_before)


def test_write_lucretia_with_stop_ix(tmp_path):
    """A valid stop_ix array must produce a file; a wrong-length one must
    raise instead of silently writing nothing."""
    from beamphysics.interfaces.lucretia import write_lucretia

    P = make_particle_group([1e6, 2e6, 3e6])
    outfile = tmp_path / "beam.mat"
    write_lucretia(P, str(outfile), stop_ix=[0, 0, 0], verbose=False)
    assert outfile.exists()

    with pytest.raises(ValueError, match="stop_ix"):
        write_lucretia(P, str(tmp_path / "bad.mat"), stop_ix=[0, 0], verbose=False)


def test_cst_get_scale_rejects_unknown_units():
    from beamphysics.interfaces.cst import get_scale

    assert get_scale("[mm]") == 1e-3
    assert get_scale("[V/m]") == 1
    with pytest.raises(ValueError, match="Unknown unit"):
        get_scale("[cm]")
