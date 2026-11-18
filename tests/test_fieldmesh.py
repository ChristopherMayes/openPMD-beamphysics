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


@pytest.fixture
def fm_onaxis():
    from beamphysics import FieldMesh

    FM = FieldMesh.from_onaxis(z=np.arange(3.0), Bz=np.array([-2.0, 0.0, 1.0]))
    FM.scale = 2.0
    return FM


def test_record_name_key_deprecated(fm_onaxis):
    FM = fm_onaxis
    with pytest.warns(DeprecationWarning, match=r"components\['magneticField/z'\]"):
        assert FM["magneticField/z"] is FM.components["magneticField/z"]


@pytest.mark.parametrize(
    ("key", "replacement"),
    [
        ("abs_magneticField/z", "abs_Bz"),
        ("re_magneticField/z", "re_Bz"),
        ("im_magneticField/z", "im_Bz"),
    ],
)
def test_prefixed_record_name_key_deprecated(fm_onaxis, key, replacement):
    FM = fm_onaxis
    with pytest.warns(DeprecationWarning, match=f"use '{replacement}' instead"):
        np.testing.assert_array_equal(FM[key], FM[replacement])
    with pytest.warns(DeprecationWarning, match=f"use '{replacement}' instead"):
        assert FM.units(key) == FM.units(replacement)


def test_alias_keys_not_deprecated(fm_dipole_corrector, recwarn):
    FM = fm_dipole_corrector
    FM["Bz"]
    FM.units("abs_Bz")
    assert FM.is_pure_magnetic
    assert not FM.is_pure_electric
    assert not [w for w in recwarn if w.category is DeprecationWarning]


def test_write_does_not_warn(fm_dipole_corrector, tmp_path, recwarn):
    fm_dipole_corrector.write(str(tmp_path / "fm.h5"))
    assert not [w for w in recwarn if w.category is DeprecationWarning]


@pytest.mark.parametrize(
    "key",
    ["magneticField/r", "electricField/z", "abs_electricField/z", "electricField"],
)
def test_absent_record_name_key_raises(fm_dipole_corrector, key):
    with pytest.raises(ValueError):
        fm_dipole_corrector[key]


def test_scaled_component_record_name_deprecated(fm_onaxis):
    FM = fm_onaxis
    with pytest.warns(DeprecationWarning, match="use 'Bz' instead") as record:
        np.testing.assert_array_equal(
            FM.scaled_component("magneticField/z"), FM.scaled_component("Bz")
        )
    assert record[0].filename == __file__
