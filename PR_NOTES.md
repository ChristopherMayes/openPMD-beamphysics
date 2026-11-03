# Add a 3D Taylor-expanded wakefield model (`TaylorWakefield`), ported from ocelot

## Summary

This PR adds a full 3D wakefield kick model to `beamphysics.wakefields`, ported from ocelot's `Wake` physics process (`ocelot.cpbd.wake3D`). The model represents the longitudinal point-charge wake function through a second-order Taylor expansion in the transverse coordinates of the source and witness particles, following I. Zagorodnov, K. Bane, and G. Stupakov, Phys. Rev. ST Accel. Beams 18, 104401 (2015). Transverse wakes are obtained from the longitudinal expansion through the Panofsky-Wenzel theorem, so the model produces longitudinal and transverse (dipole and quadrupole) kicks. This is a genuinely new capability for the package: all existing wakefield classes are purely longitudinal.

The implementation was benchmarked head-to-head against ocelot using identical particle distributions and identical wake tables. Per-particle kicks agree to machine precision (about 1e-14 relative) for file-based wake tables, and to about 1.5e-10 relative for the analytic table generators, where the residual is entirely explained by ocelot hardcoding a slightly different value of the free-space impedance than the CODATA value provided by scipy.

## The model

The longitudinal wake for a source particle at transverse position (x_s, y_s) and a witness particle at (x_w, y_w), separated longitudinally by s >= 0, is expanded as

$$w(x_s, y_s, x_w, y_w, s) = \sum_{a \le b} h_{ab}(s)\, u_a u_b, \qquad u = (1,\, x_s,\, y_s,\, x_w,\, y_w)$$

so the full 3D wake is represented by a set of one-dimensional components $h_{ab}(s)$. The index meaning is 0: constant, 1: source x, 2: source y, 3: witness x, and 4: witness y. For example, (0, 0) is the monopole longitudinal wake, (0, 4) is the vertical dipole wake, and (3, 3) and (2, 4) are quadrupole-like terms. To compute kicks, the particle charges (and transverse-moment-weighted "generalized currents") are deposited onto a smoothed longitudinal grid, each component is convolved with the appropriate current, and the transverse kicks are accumulated through the Panofsky-Wenzel integral.

Unlike the 1D wakefields in this package, the wake amplitudes are in V/C for the whole structure, because the structure length is baked into the wake table. Kicks are therefore returned in eV/c rather than eV/m.

## What is added

- `beamphysics/wakefields/taylor.py` provides the new module.
  - `TaylorWakeComponent` is a dataclass holding one component $h_{ab}(s)$, consisting of a tabulated wake, an optional tabulated derivative-coupled (inductive-like) term, and optional lumped R, L, and 1/C circuit terms.
  - `TaylorWakefield` holds the component set and computes kicks via `particle_kicks_3d(x, y, z, weight, n_points=500, filter_order=20, factor=1.0)`, which returns `(dpx, dpy, dpz)` in eV/c using beamphysics conventions (the bunch head is at larger z). The `factor` argument scales all kicks and is equivalent to ocelot's `Wake.factor`, for example to represent several identical structures.
  - `TaylorWakefield.from_file` and `TaylorWakefield.to_file` read and write the ocelot/Zagorodnov numeric wake table format, so tables can be exchanged with ocelot and with the ECHO family of codes (for example, the European XFEL `*_WAKE_TAYLOR.dat` tables).
  - `TaylorWakefield.parallel_plate` is an analytic generator for corrugated parallel-plate (dechirper) structures with the beam possibly offset from the center, ported from ocelot's `WakeTableParallelPlate`. The `decay=False` option reproduces ocelot's zeroth-order `WakeTableParallelPlate_origin` variant. It is based on K. Bane, G. Stupakov, and I. Zagorodnov, Phys. Rev. Accel. Beams 19, 084401 (2016).
  - `TaylorWakefield.dechirper_off_axis` is a mode-sum generator for a beam near a single corrugated plate of finite width, ported from ocelot's `WakeTableDechirperOffAxis` and based on https://doi.org/10.1016/j.nima.2016.09.001.
  - `TaylorWakefield.wake_potential` convolves a single component with a current profile, applying the Panofsky-Wenzel integral for transverse witness components, and `TaylorWakefield.plot` displays the tabulated components.
  - The generator parameters use descriptive names (`half_gap`, `plate_distance`, `corrugation_gap`, `corrugation_period`, `length`, `sigma`, `orientation`) in place of ocelot's single-letter names (`a`, `b`, `t`, `p`), with the correspondence documented in the docstrings.
- `ParticleGroup.apply_wakefield` in `beamphysics/particles.py` was extended to support the new model. The `length` argument is now optional: it remains required for 1D longitudinal wakefields, and it must be omitted for `TaylorWakefield` objects because the structure length is part of the wake table. For 3D wakefields, `px`, `py`, and `pz` are all updated, and extra keyword arguments such as `n_points`, `filter_order`, and `factor` are forwarded to `particle_kicks_3d`. Usage is simply `P2 = P.apply_wakefield(wake)`. The argument handling is strict in both directions: unexpected keyword arguments for a 1D wakefield raise a `TypeError` instead of being silently ignored, and passing `include_self_kick=False` with a 3D wakefield raises a `ValueError` because the half self-term is always included in the Taylor convolution.
- `ParticleGroup.wakefield_plot` raises a clear `TypeError` when given a 3D `TaylorWakefield`, directing the user to apply the wake and plot the momentum changes directly, instead of failing with an obscure `AttributeError` inside the plotting internals.
- `beamphysics/wakefields/__init__.py` exports `TaylorWakefield` and `TaylorWakeComponent`.

## Benchmark against ocelot

The script `scripts/benchmark_taylor_wakefield_vs_ocelot.py` pushes identical 20,000-particle Gaussian bunches through ocelot's `Wake.apply` and through this implementation, and compares the per-particle kicks Px, Py, and Pz. The only requirement is the ocelot package from PyPI (`pip install ocelot-collab`); no ocelot source checkout is needed. The benchmark was run against ocelot-collab 26.6.1.

For the file-based cases, the script writes a synthetic wake table with `TaylorWakefield.to_file` and reads it back through ocelot's own `WakeTable` parser. The synthetic table contains monopole, dipole, and quadrupole-like components and deliberately exercises every term of the convolution: the tabulated wake W0, the derivative-coupled term W1, and the lumped R, L, and 1/C circuit terms. The benchmark covers nine cases and the results are as follows.

| Case | Max relative error |
| --- | --- |
| File-based wake table (W0, W1, R, L, 1/C terms), on-axis beam | 4.4e-14 |
| File-based wake table, offset beam (x = +30 µm, y = -50 µm) | 9.6e-13 |
| Longitudinal and dipole wake potentials versus `get_long_wake` and `get_dipole_wake` | exact / 2.6e-14 |
| Analytic parallel plate, horizontal orientation, offset beam | 1.5e-10 |
| Analytic parallel plate, vertical orientation, offset beam | 1.5e-10 |
| Analytic parallel plate, beam centered (Y = 0 branch) | 1.5e-10 |
| Analytic parallel plate, zeroth order (`decay=False`) | 1.5e-10 |
| Dechirper off-axis mode sum, horizontal orientation | 1.5e-10 |
| Dechirper off-axis mode sum, vertical orientation | 1.5e-10 |

The 1.5e-10 residual in the analytic-generator cases comes from a single constant: ocelot hardcodes the free-space impedance as 376.7303134695850 Ohm, while this implementation uses the package's own `beamphysics.units.Z0` (equal to mu_0 times c, matching the scipy CODATA value to 8e-14 relative). The file-based cases, which share no such constant, agree to machine precision.

## Intentional differences from ocelot

- Components are stored in a dictionary keyed by the index pair (a, b) rather than in ocelot's H index matrix. Ocelot tests for the presence of a component with `H[n, m] > 0`, which cannot distinguish a missing component from a component stored at index 0. As a consequence, ocelot's `get_dipole_wake` silently convolves the wrong component when a table has no (0, 4) term. This implementation raises a `KeyError` instead.
- The numba-accelerated charge deposition loop was replaced with a vectorized `np.bincount` implementation, so the package gains no new dependency and no optional-dependency code path. The summation-order difference contributes only at the 1e-14 level.
- Coordinate conventions follow beamphysics: the bunch head is at larger z, and internally the ocelot coordinate tau = -z is used so the algorithm is otherwise line-for-line identical. Kicks are applied directly to `px`, `py`, and `pz` in eV/c, whereas ocelot divides by the reference energy to update its dimensionless coordinates.
- Input validation is stricter than ocelot's. Constructing a `TaylorWakefield` with a component outside the 13 supported index pairs (for example (2, 2) or (4, 4), which ocelot would silently ignore) raises a `ValueError`. `particle_kicks_3d` validates `n_points` and `filter_order` and rejects empty or zero-length distributions with clear messages, and `wake_potential` verifies that the supplied current profile is on a strictly increasing, uniform z grid instead of silently returning wrong results.
- Each `TaylorWakeComponent` copies its input arrays, so components never share buffers. In ocelot's analytic generators (and in a direct port), the (0, 2) and (0, 4) components of the parallel-plate table alias the same array, and modifying one in place would silently corrupt the other.

## Tests

The new file `tests/test_wakefields_taylor.py` contains 29 tests covering component construction and validation, file round trips including the lumped R, L, and 1/C terms, physics checks (causality, net energy loss, dipole kick direction for an offset beam, quadrupole antisymmetry for a centered beam, exact horizontal/vertical orientation symmetry, and linear scaling with charge), a statistical regression against values generated after the ocelot benchmark was verified, and the `ParticleGroup.apply_wakefield` integration including its argument validation. The validation and safety behaviors added after code review (unsupported component keys, buffer independence, `n_points` and grid validation, the `factor` scaling, strict keyword handling, and the `wakefield_plot` guard) each have dedicated tests. The full test suite passes with 1,651 tests, and the changed files are clean under ruff check and ruff format.

## Documentation

- A new example notebook `docs/examples/wakefields/taylor_wakefield_3d.ipynb` builds a dechirper wake table, plots the components and the wake potentials for a Gaussian current profile, applies the wake to a `ParticleGroup` and shows the induced energy chirp and transverse kick, demonstrates the ocelot-compatible file round trip, and shows the single-plate mode-sum table. It is registered in the `mkdocs.yml` navigation and executes cleanly.
- The API page `docs/api/wakefields.md` gains entries for `TaylorWakefield` and `TaylorWakeComponent`.

## Scope and follow-ups

This PR covers ocelot's second-order `Wake` and `WakeTable`, which is the main 3D model, together with the parallel-plate and off-axis dechirper table generators. Ocelot's third-order `Wake3` and `WakeTable3` variant is a natural follow-up, and the component design extends directly to index triples (a, b, c).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
