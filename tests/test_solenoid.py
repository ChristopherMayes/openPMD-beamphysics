import numpy as np
import pytest

from pmd_beamphysics.fields.solenoid import C_full, cel, make_solenoid_fieldmesh


@pytest.mark.parametrize(
    ("kc", "p", "c", "s", "mathematica"),
    [
        pytest.param(0.5, 1.0, 1.0, 0.5, 1.5262092342121871, id="kc0.5"),
        pytest.param(0.8, 0.9, 1.2, -0.3, 0.7192092915373303, id="kc0.8"),
        pytest.param(0.3, 0.5, 2.0, 0.7, 4.371297871647941, id="kc0.3"),
        pytest.param(0.0, 1.0, 1.0, 0.0, 1.0, id="edge-case"),
    ],
)
def test_cel_vs_c_full(kc: float, p: float, c: float, s: float, mathematica: float):
    r"""
    Compare with Mathematica:

    CEL[kc_, p_, c_, s_] :=
     Integrate[(c  Cos[\[Phi]]^2 + s  Sin[\[Phi]]^2)/((Cos[\[Phi]]^2 + p  Sin[\[Phi]]^2)  Sqrt[Cos[\[Phi]]^2 + kc^2  Sin[\[Phi]]^2]), {\[Phi], 0, \[Pi]/2}, Assumptions -> {kc >= 0, p > 0}]

     CEL[0.5, 1.0, 1.0, 0.5] = 1.5262092342121871

     CEL[0.8, 0.9, 1.2, -0.3] = 0.7192092915373303

     CEL[0.3, 0.5, 2.0, 0.7] = 4.371297871647941

     CEL[0.0, 1.0, 1.0, 0.0] = 1.0

    """

    cel_result = cel(kc, p, c, s)
    c_full_result = C_full(kc, p, c, s)
    difference = abs(cel_result - c_full_result)

    print(
        f"Test case: kc={kc}, p={p}, c={c}, s={s}\n"
        f"  cel result: {cel_result}\n"
        f"  C_full result: {c_full_result}\n"
        f"  Mathematica: {mathematica}\n"
        f"  Difference: {difference}\n"
    )

    np.testing.assert_allclose(cel_result, c_full_result)
    np.testing.assert_allclose(mathematica, c_full_result)


@pytest.mark.parametrize(
    ("nI", "B0"),
    [
        pytest.param(None, 1.0, id="only_B0"),
        pytest.param(2000.0, None, id="only_nI"),
    ],
)
def test_make_solenoid_fieldmesh_valid_scenarios(nI, B0):
    """Test valid combinations of nI and B0 parameters."""

    params = {
        "rmin": 0,
        "rmax": 0.01,
        "zmin": -0.2,
        "zmax": 0.2,
        "nr": 10,
        "nz": 20,
        "radius": 0.1,
        "nI": nI,
        "B0": B0,
        "L": 0.4,
    }

    print("Calling make_solenoid_fieldmesh with:", params)
    field_mesh = make_solenoid_fieldmesh(**params)

    assert field_mesh is not None
    assert hasattr(field_mesh, "data")
    assert "components" in field_mesh.data
    assert "magneticField/r" in field_mesh.data["components"]
    assert "magneticField/z" in field_mesh.data["components"]

    Br = field_mesh.data["components"]["magneticField/r"]
    Bz = field_mesh.data["components"]["magneticField/z"]
    assert Br.shape == (10, 1, 20)  # (nr, 1, nz)
    assert Bz.shape == (10, 1, 20)  # (nr, 1, nz)


def test_make_solenoid_fieldmesh_error_both():
    with pytest.raises(ValueError, match="Must specify exactly one of.*"):
        make_solenoid_fieldmesh(L=0.4, nI=2000.0, B0=1.0)


def test_make_solenoid_fieldmesh_error_neither():
    with pytest.raises(ValueError, match="Must specify exactly one of.*"):
        make_solenoid_fieldmesh(L=0.4)
