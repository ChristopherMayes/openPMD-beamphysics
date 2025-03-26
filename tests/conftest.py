import logging
import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest

mpl.use("Agg")

test_root = pathlib.Path(__file__).resolve().parent
repo_root = test_root.parent

test_artifacts = test_root / "artifacts"
test_artifacts.mkdir(exist_ok=True)

logging.getLogger("matplotlib").setLevel("INFO")


@pytest.fixture(autouse=True, scope="function")
def _plot_show_to_savefig(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = 0
    node_name = request.node.name.replace("/", "_")

    def savefig():
        nonlocal index
        filename = test_artifacts / f"{node_name}_{index}.png"
        print(f"Saving figure (_plot_show_to_savefig fixture) to {filename}")
        plt.savefig(filename)
        index += 1

    monkeypatch.setattr(plt, "show", savefig)
