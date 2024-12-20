import pathlib
import re

import matplotlib
import matplotlib.pyplot as plt
import pytest

matplotlib.use("Agg")

test_root = pathlib.Path(__file__).parent.resolve()
test_artifacts = test_root / "artifacts"


@pytest.fixture(autouse=True, scope="function")
def _plot_show_to_savefig(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
):
    index = 0

    def savefig():
        nonlocal index
        test_artifacts.mkdir(exist_ok=True)
        for fignum in plt.get_fignums():
            plt.figure(fignum)
            name = re.sub(r"[/\\]", "_", request.node.name)
            filename = test_artifacts / f"{name}_{index}.png"
            print(f"Saving figure (_plot_show_to_savefig fixture) to {filename}")
            plt.savefig(filename)
            index += 1
        plt.close("all")

    monkeypatch.setattr(plt, "show", savefig)
    yield
    plt.show()
