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

::: beamphysics.wakefields.TabularWakefield.from_wakefield

::: beamphysics.wakefields.TabularWakefield.from_impact_z

::: beamphysics.wakefields.ResistiveWallWakefieldBase.to_tabular

### IMPACT-Z

IMPACT-Z applies a tabulated short-range wake through its zero-length `-41` element,
which reads a uniform four-column table named `rfdata{file_id}.in`.

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
