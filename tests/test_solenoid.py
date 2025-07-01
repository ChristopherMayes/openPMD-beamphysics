import numpy as np
import pytest

from pmd_beamphysics.fields.solenoid import C_full, cel


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

    assert np.isclose(cel_result, c_full_result)
    assert np.isclose(mathematica, c_full_result)
