"""
Benchmark: beamphysics TaylorWakefield vs ocelot Wake (wake3D).

Same particle distribution through the same wake tables; compare the
per-particle kicks (Px, Py, Pz in eV).

Requires ocelot (https://github.com/ocelot-collab/ocelot) to be installed,
with its repository checked out for the unit-test wake table. Run:

    python scripts/benchmark_taylor_wakefield_vs_ocelot.py [path/to/ocelot/repo]

Expected agreement: machine precision (~1e-14) for file-based wake tables;
~1.5e-10 for the analytic table generators, which comes from ocelot
hardcoding a slightly different value of the free-space impedance than
scipy's CODATA value.

Ocelot conventions: rparticles[0]=x, [2]=y, [4]=tau (positive = tail),
kicks applied as P/(E*1e9) to x', y', p. Standalone use: set s_start=s_stop
so L=0 -> full single kick.

beamphysics conventions: z (head at larger z) => tau = -z.
"""

import sys
from pathlib import Path

import numpy as np

import ocelot.cpbd.wake3D as ow
from ocelot.cpbd.beam import ParticleArray

from beamphysics.wakefields import TaylorWakefield

OCELOT_REPO = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else Path(__file__).resolve().parents[2] / "ocelot"
)
OCELOT_WAKE_TABLE = str(OCELOT_REPO / "unit_tests/ebeam_test/wake/wake_table.dat")

RNG = np.random.default_rng(42)


def make_bunch(
    n=20000,
    sigma_tau=10e-6,
    sigma_x=20e-6,
    sigma_y=20e-6,
    offset_x=0.0,
    offset_y=0.0,
    charge=250e-12,
    energy_gev=14.0,
):
    x = RNG.normal(offset_x, sigma_x, n)
    y = RNG.normal(offset_y, sigma_y, n)
    tau = RNG.normal(0, sigma_tau, n)
    q = np.full(n, charge / n)
    return x, y, tau, q, energy_gev


def ocelot_kicks(wake_table, x, y, tau, q, energy_gev):
    """Return (Px, Py, Pz) in eV from ocelot's Wake process."""
    n = len(x)
    p_array = ParticleArray(n)
    p_array.rparticles[0] = x
    p_array.rparticles[2] = y
    p_array.rparticles[4] = tau
    p_array.q_array[:] = q
    p_array.E = energy_gev

    wake = ow.Wake()
    wake.wake_table = wake_table
    wake.prepare(None)
    wake.s_start = 0.0
    wake.s_stop = 0.0  # L=0 -> full kick in one application
    wake.apply(p_array, dz=1.0)

    scale = energy_gev * 1e9
    return (
        p_array.rparticles[1] * scale,
        p_array.rparticles[3] * scale,
        p_array.rparticles[5] * scale,
    )


def bp_kicks(wakefield, x, y, tau, q):
    """Return (Px, Py, Pz) in eV from beamphysics TaylorWakefield."""
    return wakefield.particle_kicks_3d(x, y, -tau, q)


def compare(name, ours, theirs):
    ours = np.asarray(ours)
    theirs = np.asarray(theirs)
    denom = np.max(np.abs(theirs))
    if denom == 0:
        agree = np.max(np.abs(ours)) == 0
        print(f"  {name:3s}: both identically zero: {agree}")
        return 0.0
    err = np.max(np.abs(ours - theirs)) / denom
    print(f"  {name:3s}: max |kick| = {denom:12.5g} eV   max rel err = {err:.3g}")
    return err


def run_case(
    title, ocelot_table, bp_table, offset_x=0.0, offset_y=0.0, sigma_tau=10e-6
):
    print(f"\n=== {title} ===")
    x, y, tau, q, E = make_bunch(
        offset_x=offset_x, offset_y=offset_y, sigma_tau=sigma_tau
    )
    Px_o, Py_o, Pz_o = ocelot_kicks(ocelot_table, x, y, tau, q, E)
    Px_b, Py_b, Pz_b = bp_kicks(bp_table, x, y, tau, q)
    errs = [
        compare("Pz", Pz_b, Pz_o),
        compare("Px", Px_b, Px_o),
        compare("Py", Py_b, Py_o),
    ]
    return max(errs)


def main():
    worst = 0.0

    # --- Case 1: file-based wake table (ocelot unit-test table) ---
    ot = ow.WakeTable(OCELOT_WAKE_TABLE)
    bt = TaylorWakefield.from_file(OCELOT_WAKE_TABLE)
    worst = max(
        worst,
        run_case("File table (h00,h13,h24), on-axis beam", ot, bt, sigma_tau=100e-6),
    )
    worst = max(
        worst,
        run_case(
            "File table, offset beam (x=+30um, y=-50um)",
            ot,
            bt,
            offset_x=30e-6,
            offset_y=-50e-6,
            sigma_tau=100e-6,
        ),
    )

    # --- Case 2: analytic parallel-plate (first order, off-center) ---
    for orient_o, orient_b in [("horz", "horizontal"), ("vert", "vertical")]:
        ot = ow.WakeTableParallelPlate(
            b=250e-6,
            a=500e-6,
            t=250e-6,
            p=500e-6,
            length=2.0,
            sigma=10e-6,
            orient=orient_o,
        )
        bt = TaylorWakefield.parallel_plate(
            plate_distance=250e-6,
            half_gap=500e-6,
            corrugation_gap=250e-6,
            corrugation_period=500e-6,
            length=2.0,
            sigma=10e-6,
            orientation=orient_b,
        )
        worst = max(
            worst,
            run_case(
                f"ParallelPlate ({orient_b}), offset beam",
                ot,
                bt,
                offset_x=10e-6,
                offset_y=20e-6,
            ),
        )

    # --- Case 3: parallel-plate, beam centered (Y=0 branch) ---
    ot = ow.WakeTableParallelPlate(
        b=500e-6, a=500e-6, t=250e-6, p=500e-6, length=1.0, sigma=10e-6, orient="horz"
    )
    bt = TaylorWakefield.parallel_plate(
        plate_distance=500e-6,
        half_gap=500e-6,
        corrugation_gap=250e-6,
        corrugation_period=500e-6,
        length=1.0,
        sigma=10e-6,
        orientation="horizontal",
    )
    worst = max(
        worst,
        run_case(
            "ParallelPlate centered (Y=0 branch)", ot, bt, offset_x=5e-6, offset_y=-5e-6
        ),
    )

    # --- Case 4: zeroth-order (no decay) variant ---
    ot = ow.WakeTableParallelPlate_origin(
        b=300e-6, a=500e-6, t=250e-6, p=500e-6, length=1.0, sigma=10e-6, orient="horz"
    )
    bt = TaylorWakefield.parallel_plate(
        plate_distance=300e-6,
        half_gap=500e-6,
        corrugation_gap=250e-6,
        corrugation_period=500e-6,
        length=1.0,
        sigma=10e-6,
        orientation="horizontal",
        decay=False,
    )
    worst = max(worst, run_case("ParallelPlate zeroth order (decay=False)", ot, bt))

    # --- Case 5: dechirper off-axis (mode sum) ---
    for orient_o, orient_b in [("horz", "horizontal"), ("vert", "vertical")]:
        ot = ow.WakeTableDechirperOffAxis(
            b=500e-6,
            a=0.01,
            width=0.02,
            t=250e-6,
            p=500e-6,
            length=1.0,
            sigma=10e-6,
            orient=orient_o,
        )
        bt = TaylorWakefield.dechirper_off_axis(
            plate_distance=500e-6,
            half_gap=0.01,
            width=0.02,
            corrugation_gap=250e-6,
            corrugation_period=500e-6,
            length=1.0,
            sigma=10e-6,
            orientation=orient_b,
        )
        worst = max(
            worst,
            run_case(
                f"DechirperOffAxis ({orient_b})", ot, bt, offset_x=10e-6, offset_y=20e-6
            ),
        )

    # --- Case 6: longitudinal wake potential vs get_long_wake ---
    print("\n=== Wake potential vs ocelot get_long_wake/get_dipole_wake ===")
    ot = ow.WakeTable(OCELOT_WAKE_TABLE)
    bt = TaylorWakefield.from_file(OCELOT_WAKE_TABLE)
    s = np.linspace(-300e-6, 300e-6, 1000)  # ocelot tau grid
    current = 100 * np.exp(-0.5 * (s / 50e-6) ** 2)
    profile_ocelot = np.column_stack([s, current])

    w = ow.Wake()
    w.wake_table = ot
    w.prepare(None)
    x_o, W_o = w.get_long_wake(profile_ocelot)

    # beamphysics: z = -tau, ascending
    profile_bp = np.column_stack([-s[::-1], current[::-1]])
    z_b, W_b = bt.wake_potential(profile_bp, key=(0, 0))
    err = np.max(np.abs(W_b[::-1] - W_o)) / np.max(np.abs(W_o))
    zerr = np.max(np.abs(-z_b[::-1] - x_o))
    print(f"  long wake: max rel err = {err:.3g}, grid err = {zerr:.3g}")
    worst = max(worst, err)

    # Dipole: needs a table with an (0,4) component; use parallel plate.
    # (Note: ocelot's get_dipole_wake on the file table above would silently
    # convolve the wrong component since H[0,4]=0 also means "missing".)
    ot = ow.WakeTableParallelPlate(
        b=250e-6, a=500e-6, t=250e-6, p=500e-6, length=1.0, sigma=50e-6, orient="horz"
    )
    bt = TaylorWakefield.parallel_plate(
        plate_distance=250e-6,
        half_gap=500e-6,
        corrugation_gap=250e-6,
        corrugation_period=500e-6,
        length=1.0,
        sigma=50e-6,
        orientation="horizontal",
    )
    w = ow.Wake()
    w.wake_table = ot
    w.prepare(None)
    x_o, Wd_o = w.get_dipole_wake(profile_ocelot)
    z_b, Wd_b = bt.wake_potential(profile_bp, key=(0, 4))
    err = np.max(np.abs(Wd_b[::-1] - Wd_o)) / np.max(np.abs(Wd_o))
    print(f"  dipole wake: max rel err = {err:.3g}")
    worst = max(worst, err)

    print(f"\nWorst relative error across all cases: {worst:.3g}")
    assert worst < 1e-8, "Benchmark FAILED"
    print("Benchmark PASSED (all cases agree with ocelot)")


if __name__ == "__main__":
    main()
