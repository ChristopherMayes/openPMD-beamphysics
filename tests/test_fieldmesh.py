from beamphysics.fields.corrector_modeling import make_dipole_corrector_fieldmesh
from beamphysics import FieldMesh
import numpy as np
import tempfile

import pytest


@pytest.fixture(scope="module")
def fm_dipole_corrector():
    """Fixture to create and return the FM object for dipole corrector fieldmesh."""
    R = 0.02
    L = 0.1
    theta = np.pi / 2
    current = 1

    # Create the FM object (this will only be done once per test module)
    FM = make_dipole_corrector_fieldmesh(
        current=current,
        xmin=-R,
        xmax=R,
        nx=11,
        ymin=-R,
        ymax=R,
        ny=11,
        zmin=-5 * L / 2,
        zmax=5 * L / 2,
        nz=11,
        mode="saddle",
        R=R,
        L=L,
        theta=theta,
        npts=20,
    )
    return FM


# First test using the FM fixture
def test_dipole_corrector_read_impact_emfield_cartesian(fm_dipole_corrector):
    FM = fm_dipole_corrector

    # Test read/write impact_emfield_cartesian
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_filename = temp_file.name
        FM.write_impact_emfield_cartesian(temp_filename)
        FM2 = FieldMesh.from_impact_emfield_cartesian(
            temp_filename, eleAnchorPt=FM.attrs["eleAnchorPt"]
        )

    assert FM == FM2


# Another test that uses the same FM object
def test_fieldmesh_limits(fm_dipole_corrector):
    FM = fm_dipole_corrector

    # Example test to check attributes
    assert FM.xmax == 0.02
    assert FM.ymax == 0.02

    # Test setting zmax
    L1 = FM.zmax - FM.zmin
    FM.zmax = 100
    L2 = FM.zmax - FM.zmin
    assert L1 == L2
    FM.zmin = 100
    L3 = FM.zmax - FM.zmin
    assert L1 == L3


@pytest.mark.parametrize(
    "component_keys",
    [
        ("electricField/z", "Ez"),  # canonical first
        ("Ez", "electricField/z"),  # alias first
    ],
)
def test_duplicate_component_alias_rejected(component_keys):
    """A data dict containing both a canonical component name and its alias
    must raise, not silently overwrite one with the other — regardless of
    dict insertion order."""
    from beamphysics.fields.fieldmesh import load_field_data_dict

    arr = np.ones((2, 2, 2))
    attrs = {
        "eleAnchorPt": "beginning",
        "gridGeometry": "rectangular",
        "axisLabels": ("x", "y", "z"),
        "gridLowerBound": (0, 0, 0),
        "gridOriginOffset": (0.0, 0.0, 0.0),
        "gridSpacing": (0.01, 0.01, 0.01),
        "gridSize": (2, 2, 2),
        "harmonic": 0,
        "fundamentalFrequency": 0,
    }
    k1, k2 = component_keys
    data = {
        "attrs": attrs,
        "components": {k1: arr, k2: 2 * arr},
    }
    with pytest.raises(ValueError, match="Duplicate component"):
        load_field_data_dict(data, verbose=False)
