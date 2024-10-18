from pmd_beamphysics.fields.corrector_modeling import make_dipole_corrector_fieldmesh
from pmd_beamphysics import FieldMesh
import numpy as np
import tempfile


def test_dipole_corrector():
    R = 0.02
    L = 0.1
    theta = np.pi / 2
    current = 1

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

    # Test read/write impact_emfield_cartesian
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_filename = temp_file.name
        FM.write_impact_emfield_cartesian(temp_filename)
        FM2 = FieldMesh.from_impact_emfield_cartesian(
            temp_filename, eleAnchorPt=FM.attrs["eleAnchorPt"]
        )

    assert FM == FM2
