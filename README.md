# openPMD-beamphysics

| **`Documentation`**                                                                                                                          |
| -------------------------------------------------------------------------------------------------------------------------------------------- |
| [![Documentation](https://img.shields.io/badge/beamphysics-documentation-blue.svg)](https://christophermayes.github.io/openPMD-beamphysics/) |

Tools for analyzing and viewing particle data in the openPMD standard, extension beamphysics.

<https://github.com/openPMD/openPMD-standard/blob/upcoming-2.0.0/EXT_BeamPhysics.md>

# Installing openpmd-beamphysics

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

## Development environment

A conda environment file is provided in this repository and may be used for a
development environment.

To create a new conda environment using this file, do the following:

```bash
git clone https://github.com/ChristopherMayes/openPMD-beamphysics
cd openPMD-beamphysics
conda env create -n pmd_beamphysics-dev -f environment.yml
conda activate beamphysics-dev
python -m pip install --no-deps -e .
```

Alternatively, with a virtualenv and pip:

```bash
git clone https://github.com/ChristopherMayes/openPMD-beamphysics
cd openPMD-beamphysics

python -m venv beamphysics-venv
source pmd_beamphysics-venv/bin/activate
python -m pip install -e .
```
