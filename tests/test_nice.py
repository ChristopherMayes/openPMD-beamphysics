from pmd_beamphysics.units import nice_array
import numpy as np


def test_nice_array():
    nice_array(1)

    nice_array([1])

    nice_array(
        [
            1,
            2,
        ]
    )

    nice_array(np.array([1, 2]))
