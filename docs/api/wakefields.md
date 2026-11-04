# Wakefields

## Resistive Wall Wakefield Classes

::: beamphysics.wakefields.ResistiveWallWakefield

::: beamphysics.wakefields.ResistiveWallPseudomode

## Cavity Wakefield Classes

::: beamphysics.wakefields.ParametricWakefield

## Base Classes

::: beamphysics.wakefields.WakefieldBase

::: beamphysics.wakefields.PseudomodeWakefield

::: beamphysics.wakefields.ImpedanceWakefield

::: beamphysics.wakefields.TabularWakefield

::: beamphysics.wakefields.BinnedWakefield

::: beamphysics.wakefields.Pseudomode

## Tabulation and Interchange

`TabularWakefield` holds interpolated point samples of a smooth wake. `BinnedWakefield` holds fixed-width bin averages for convolution and tracking; it has no pointwise `wake` method. Both classes provide `from_wakefield` and the same Bmad and IMPACT readers and writers. When `zmax` or `n` is omitted, the source supplies `default_zmax` and `default_n_samples`.

### Singular wakes and bin averages

A wake that diverges at the source particle, such as the diffraction term of `ParametricWakefield`, cannot be sampled at points. `BinnedWakefield.from_wakefield` integrates it into bins using the model's `integrated_wake(z)`, the integral from z to zero (V/C), including the full delta contribution for z < 0. This is the integrated Green function of Qiang et al. (2006). Bmad, IMPACT-T, IMPACT-Z and `convolve_density` halve the first row for the self-kick, so the binned kernel stores twice the self-bin coefficient in its first row.

A binned wake's `dz` is fixed. Its convolution and particle tracking methods use that spacing and reject a mismatched explicit dz. `rebin(dz)` changes the spacing conservatively; interpolation of point values is not equivalent. Bmad bins at the table spacing, while IMPACT-T and IMPACT-Z bin at their own mesh width, so rebin to that width before exporting.

```python
from beamphysics.wakefields import BinnedWakefield, ParametricWakefield, ZAGORODNOV_2021_TABLE_IV

wake = ParametricWakefield(length=10.5, **ZAGORODNOV_2021_TABLE_IV["AH1"])
bins = BinnedWakefield.from_wakefield(wake, zmax=20e-3, n=2001)
impact_bins = bins.rebin(dz=50e-6)
rfdata = impact_bins.to_impact_z()
restored = BinnedWakefield.from_impact_z(rfdata)
```

The reader class determines the interpretation: use `BinnedWakefield.from_impact_z` for bin averages and `TabularWakefield.from_impact_z` for point samples. These file formats do not record that distinction.

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
