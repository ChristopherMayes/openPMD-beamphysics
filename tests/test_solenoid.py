from pmd_beamphysics.fields.solenoid import C_full, cel
import numpy as np


def test_cel_vs_c_full():
    r"""
    Compare with Mathematica:

    CEL[kc_, p_, c_, s_] :=
     Integrate[(c  Cos[\[Phi]]^2 + s  Sin[\[Phi]]^2)/((Cos[\[Phi]]^2 + p  Sin[\[Phi]]^2)  Sqrt[Cos[\[Phi]]^2 + kc^2  Sin[\[Phi]]^2]), {\[Phi], 0, \[Pi]/2}, Assumptions -> {kc >= 0, p > 0}]

     CEL[0.5, 1.0, 1.0, 0.5] = 1.5262092342121871

     CEL[0.8, 0.9, 1.2, -0.3] = 0.7192092915373303

     CEL[0.3, 0.5, 2.0, 0.7] = 4.371297871647941

     CEL[0.0, 1.0, 1.0, 0.0] = 1.0

    """

    test_cases = [
        {"kc": 0.5, "p": 1.0, "c": 1.0, "s": 0.5, "mma": 1.5262092342121871},
        {"kc": 0.8, "p": 0.9, "c": 1.2, "s": -0.3, "mma": 0.7192092915373303},
        {"kc": 0.3, "p": 0.5, "c": 2.0, "s": 0.7, "mma": 4.371297871647941},
        {"kc": 0.0, "p": 1.0, "c": 1.0, "s": 0.0, "mma": 1.0},  # Edge case
    ]

    print("Testing cel vs C_full:")
    for i, params in enumerate(test_cases):
        kc, p, c, s = params["kc"], params["p"], params["c"], params["s"]
        cel_result = cel(kc, p, c, s)
        c_full_result = C_full(kc, p, c, s)
        difference = abs(cel_result - c_full_result)

        mathematica = params["mma"]

        print(
            f"Test case {i + 1}: kc={kc}, p={p}, c={c}, s={s}\n"
            f"  cel result: {cel_result}\n"
            f"  C_full result: {c_full_result}\n"
            f"  Mathematica: {mathematica}\n"
            f"  Difference: {difference}\n"
        )

        assert np.isclose(cel_result, c_full_result)
        assert np.isclose(mathematica, c_full_result)
