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

`TabularWakefield` is the common format for exchanging wakes with tracking codes. `TabularWakefield.from_wakefield`, or `to_tabular` on the resistive wall classes, resamples any model onto a uniform table. When `zmax` or `n` is omitted, each model supplies `default_zmax` and `default_n_samples`.

### IMPACT-Z

IMPACT-Z reads a four-column table, `rfdata{file_id}.in`, through its `-41` element. The builder returns an array, so that a driver such as lume-impact can decide where to write it, and the readers accept either a path or an array.

::: beamphysics.interfaces.impact.create_impact_z_wakefield_rfdata

::: beamphysics.interfaces.impact.parse_impact_z_wakefield

::: beamphysics.interfaces.impact.write_impact_z_wakefield

### IMPACT-T

IMPACT-T reads a three-column table, `fort.{Bmpstp}`, through its `-6` element. The file has no position column, so the builder also returns the length of the table, which is entered as `V7`.

::: beamphysics.interfaces.impact.create_impact_t_wakefield_data

::: beamphysics.interfaces.impact.parse_impact_t_wakefield

::: beamphysics.interfaces.impact.write_impact_t_wakefield

### Bmad

`TabularWakefield.to_bmad` formats a table as the value of a Bmad `sr_wake`, and `TabularWakefield.from_bmad` reads back the table that pytao returns for `tao.ele_wake(ele_id, who="sr_z_long_table")`. The table must be longer than the bunch, because Bmad applies it by a circular convolution.

## Low-level Functions

::: beamphysics.wakefields.longitudinal_impedance_round

::: beamphysics.wakefields.longitudinal_impedance_flat

::: beamphysics.wakefields.wakefield_from_impedance

::: beamphysics.wakefields.wakefield_from_impedance_fft

::: beamphysics.wakefields.ac_conductivity

::: beamphysics.wakefields.surface_impedance

::: beamphysics.wakefields.characteristic_length
