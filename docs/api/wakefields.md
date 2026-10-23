# Wakefields

## Resistive Wall Wakefield Classes

::: beamphysics.wakefields.ResistiveWallWakefield

::: beamphysics.wakefields.ResistiveWallPseudomode

## Base Classes

::: beamphysics.wakefields.WakefieldBase

::: beamphysics.wakefields.PseudomodeWakefield

::: beamphysics.wakefields.ImpedanceWakefield

::: beamphysics.wakefields.TabularWakefield

::: beamphysics.wakefields.Pseudomode

## Tabulation and Interchange

A `TabularWakefield` is the common format for exporting a wake to tracking codes that read it as a uniform table. IMPACT-Z requires a table, and Bmad accepts one as an alternative to pseudomodes. Any model can be resampled onto a uniform table with `TabularWakefield.from_wakefield`, and the resistive wall classes provide `to_tabular` as a convenience.

The tabulation range is chosen automatically when `zmax` is omitted. Each model reports a `default_zmax`: ten envelope decay lengths for a pseudomode model, `100 * s0` for the impedance model, and the extent of the table for a `TabularWakefield`.

The number of rows is also chosen automatically when `n` is omitted. Each model reports a `min_wavelength`, the shortest length scale it resolves, and `default_n_samples` converts it into a row count at 128 samples per length scale. A consumer that interpolates the table linearly, as IMPACT-Z does from version 2.7.9, then reproduces the resistive wall wakes to within `1e-03` of `W0`.

### IMPACT-Z

IMPACT-Z applies a tabulated short-range wake through its zero-length `-41` element, which reads a uniform four-column table named `rfdata{file_id}.in`. The table can be built as an array and written separately, so that a driver such as lume-impact can carry it in memory and decide where the file belongs at run time. Both `parse_impact_z_wakefield` and `TabularWakefield.from_impact_z` accept either a path or such an array. The method `TabularWakefield.to_impact_z` builds the table from a `TabularWakefield` and can also write it.

::: beamphysics.interfaces.impact.create_impact_z_wakefield_rfdata

::: beamphysics.interfaces.impact.parse_impact_z_wakefield

::: beamphysics.interfaces.impact.write_impact_z_wakefield

### IMPACT-T

IMPACT-T applies a tabulated short-range wake through its `-6` element when `Bnseg` is positive and `V6` is not positive, and reads it from the file `fort.{Bmpstp}`. The file has no position column: its rows hold the longitudinal wake and the two dipole wakes, equally spaced from the source particle to the distance `V7`. The table is therefore returned together with that length, which must be entered as `V7`, and IMPACT-T reads at most 1000 rows. `TabularWakefield.from_impact_t` reads such a table back when given the same length, and `TabularWakefield.to_impact_t` builds and optionally writes it from a `TabularWakefield`.

::: beamphysics.interfaces.impact.create_impact_t_wakefield_data

::: beamphysics.interfaces.impact.parse_impact_t_wakefield

::: beamphysics.interfaces.impact.write_impact_t_wakefield

### Bmad

Bmad reads a tabulated longitudinal wake through the `z_long` substructure of `sr_wake`. `TabularWakefield.to_bmad` returns the `sr_wake` value and can also write it to a file for use as `sr_wake = call::<file>`. The tabulated range must exceed the full length of the bunch, because Bmad applies the table by a circular convolution over that range.

The `scale_with_length` setting determines how Bmad interprets the table. With the default of `True`, the table is a wake per unit length in V/C/m, as it is throughout this package, and Bmad multiplies it by the length of the element that carries it. The `sr_wake` should then be attached to an element whose length equals that of the wake-producing structure, such as a pipe. With `False`, Bmad applies the table once as a wake in V/C regardless of the element length.

## Low-level Functions

::: beamphysics.wakefields.longitudinal_impedance_round

::: beamphysics.wakefields.longitudinal_impedance_flat

::: beamphysics.wakefields.wakefield_from_impedance

::: beamphysics.wakefields.wakefield_from_impedance_fft

::: beamphysics.wakefields.ac_conductivity

::: beamphysics.wakefields.surface_impedance

::: beamphysics.wakefields.characteristic_length
