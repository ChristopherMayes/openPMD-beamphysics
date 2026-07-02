#!/bin/bash
# Generate the Genesis4 example data in docs/examples/data/genesis4.
#
# The generated files (end.par.h5, one4one.par.h5, and the .out.h5 outputs) are
# not stored in the repository; the documentation build and the test suite use
# this script to create them on the fly.
#
# Genesis4 is intentionally not a dependency of the beamphysics-dev
# environment. This script installs the conda-forge `genesis4` package into an
# isolated conda environment (created only if it does not already exist) and
# runs the data-generation script `run.sh` with it.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$REPO_ROOT/docs/examples/data/genesis4"
ENV_NAME="genesis4-example-data"

# Prefer mamba for speed; fall back to conda.
if command -v mamba &> /dev/null; then
    CONDA=mamba
elif command -v conda &> /dev/null; then
    CONDA=conda
else
    echo "Error: conda or mamba is required to create the isolated Genesis4 environment." >&2
    exit 1
fi

# Create the isolated environment only if it does not already exist.
if ! $CONDA env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Creating isolated conda environment '$ENV_NAME' with genesis4..."
    $CONDA create -y -n "$ENV_NAME" -c conda-forge genesis4
fi

cd "$DATA_DIR"
$CONDA run -n "$ENV_NAME" bash run.sh

echo "Genesis4 example data generated in $DATA_DIR:"
ls -l "$DATA_DIR"/*.h5
