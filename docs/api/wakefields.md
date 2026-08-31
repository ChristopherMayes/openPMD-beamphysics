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

A `TabularWakefield` is the interchange format between the models above and external
codes. Any model can be resampled onto a uniform table, and the resistive wall classes
provide a convenience that picks a default range from the characteristic length `s0`.

The range need not be chosen by hand. Every model that knows its own decay reports a
`default_zmax`, which is used whenever `zmax` is omitted: ten envelope decay lengths for
a pseudomode model, `100 * s0` for the impedance model, and its own extent for a table.

Neither does the number of rows. Every model that knows its own structure reports a
`min_wavelength`, the shortest length scale it resolves, and `default_n_samples` turns
that into a row count at a fixed density of 128 samples per length scale. A consumer
that interpolates the table linearly, as IMPACT-Z does, then reproduces the resistive
wall wakes to better than `1e-03` of `W0`.

::: beamphysics.wakefields.PseudomodeWakefield.decay_length

::: beamphysics.wakefields.WakefieldBase.min_wavelength

::: beamphysics.wakefields.WakefieldBase.default_n_samples

::: beamphysics.wakefields.TabularWakefield.from_wakefield

::: beamphysics.wakefields.TabularWakefield.from_impact_z

::: beamphysics.wakefields.ResistiveWallWakefieldBase.to_tabular

### IMPACT-Z

IMPACT-Z applies a tabulated short-range wake through its zero-length `-41` element,
which reads a uniform four-column table named `rfdata{file_id}.in`. The table can be
built as an array and written separately, so that a driver such as lume-impact can
carry it in memory and decide where the file belongs at run time. The reader and
`TabularWakefield.from_impact_z` accept either a path or such an array.

::: beamphysics.interfaces.impact.create_impact_z_wakefield_rfdata

::: beamphysics.interfaces.impact.parse_impact_z_wakefield

::: beamphysics.interfaces.impact.write_impact_z_wakefield

## Low-level Functions

::: beamphysics.wakefields.longitudinal_impedance_round

::: beamphysics.wakefields.longitudinal_impedance_flat

::: beamphysics.wakefields.wakefield_from_impedance

::: beamphysics.wakefields.wakefield_from_impedance_fft

::: beamphysics.wakefields.ac_conductivity

::: beamphysics.wakefields.surface_impedance

::: beamphysics.wakefields.characteristic_length
