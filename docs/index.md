# openPMD-beamphysics

Tools for analyzing and viewing particle data in the openPMD standard, extension beamphysics defined in: [beamphysics extension](https://github.com/DavidSagan/openPMD-standard/blob/EXT_BeamPhysics/EXT_BeamPhysics.md)

## Python classes

This package provides two feature-rich classes for handling openPMD-beamphysics standard data:

* `ParticleGroup` for handling particle data

* `FieldMesh` - for handling external field mesh data

For usage see the examples.


## Installation

Installing `openpmd-beamphysics` from the `conda-forge` channel can be achieved by adding `conda-forge` to your channels with:

```
conda config --add channels conda-forge
```

Once the `conda-forge` channel has been enabled, `openpmd-beamphysics` can be installed with:

```
conda install openpmd-beamphysics
```

It is possible to list all of the versions of `openpmd-beamphysics` available on your platform with:

```
conda search openpmd-beamphysics --channel conda-forge
```
