#!/usr/bin/env python3
"""Regenerate the ImpactX BeamMonitor test data in this directory.

Run with an environment that has ImpactX installed, e.g.::

    conda install -c conda-forge impactx
    python generate.py

It writes ``monitor.h5`` (a two-iteration beam monitor) and
``particles_lost.h5`` (the lost-particle output, which carries the runtime
``s_lost`` component). Both are read by ``tests/test_impactx.py``.
"""

import os
import pathlib
import shutil
import tempfile

from impactx import ImpactX, distribution, elements

HERE = pathlib.Path(__file__).resolve().parent

# ImpactX writes its diags/ tree into the working directory, so run somewhere
# disposable rather than leaving it next to the data files.
_cwd = pathlib.Path.cwd()
_tmp = tempfile.TemporaryDirectory()
os.chdir(_tmp.name)

sim = ImpactX()
sim.space_charge = False
sim.slice_step_diagnostics = False
sim.particle_lost_diagnostics_backend = "h5"
sim.init_grids()

# 2 GeV electron beam, as in the ImpactX FODO example
sim.beam.ref.set_species("electron").set_kin_energy_MeV(2.0e3)
distr = distribution.Waterbag(
    lambdaX=3.9984884770e-5,
    lambdaY=3.9984884770e-5,
    lambdaT=1.0e-3,
    lambdaPx=2.6623538760e-5,
    lambdaPy=2.6623538760e-5,
    lambdaPt=2.0e-3,
    muxpx=-0.846574929020762,
    muypy=0.846574929020762,
)
sim.add_particles(1.0e-9, distr, 200)

monitor = elements.BeamMonitor("monitor", backend="h5")
sim.lattice.extend(
    [
        monitor,
        elements.Drift(name="d1", ds=0.25, nslice=1),
        # a collimator tight enough to scrape the tails, so that
        # particles_lost.h5 has a few particles in it
        elements.Aperture(
            name="ap",
            aperture_x=1.55e-4,
            aperture_y=1.55e-4,
            shape="rectangular",
            action="transmit",
        ),
        elements.Quad(name="q1", ds=1.0, k=1.0, nslice=1),
        monitor,
    ]
)

sim.track_particles()
sim.finalize()

for name in ("monitor.h5", "particles_lost.h5"):
    shutil.copy(pathlib.Path("diags") / "openPMD" / name, HERE / name)
    print(f"wrote {HERE / name}")

os.chdir(_cwd)
_tmp.cleanup()
