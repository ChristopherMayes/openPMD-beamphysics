"""
Digitize the ECHO wake potentials of Fig. 7 (left) of Zagorodnov et al. (2021).

The figure, arXiv:2010.10204v2 Fig007.png, shows the longitudinal wake potentials
of Gaussian bunches in the AH1 module for nine bunch lengths, from the code ECHO
in solid blue with the analytical fit of Eq. 18 dashed in red. The curves are
traced column by column from a seed column where all nine are separated, the
axes are calibrated from the tick-label positions, and the result is written to
docs/examples/data/zagorodnov_2021_fig7_echo.csv with columns sigma (µm),
s/sigma and W (V/pC).

Usage:
    python scripts/digitize_zagorodnov_fig7.py [Fig007.png]
"""

import io
import pathlib
import sys
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter

URL = "https://arxiv.org/html/2010.10204v2/Fig007.png"
OUT = pathlib.Path(__file__).resolve().parents[1] / (
    "docs/examples/data/zagorodnov_2021_fig7_echo.csv"
)

SIGMAS_UM = [10, 15, 25, 50, 100, 250, 500, 1000, 2000]  # top to bottom at s = sigma

# Left panel frame in pixels, and the axis calibration from the tick labels:
# s/sigma = 0 at column 132 with 30.25 px per unit; W = 0 at row 232.5 with
# 32 px per 200 V/pC.
X0, X1, Y0, Y1 = 41, 284, 40, 232
COL0, PX_PER_UNIT = 132.0, (253.0 - 132.0) / 4
ROW0, PX_PER_V = 232.5, 32 / 200


def load(source: str | None) -> np.ndarray:
    if source is None:
        data = urllib.request.urlopen(URL).read()  # noqa: S310
        rgba = plt.imread(io.BytesIO(data), format="png")
    else:
        rgba = plt.imread(source)
    alpha = rgba[..., 3:4]
    return rgba[..., :3] * alpha + (1 - alpha)


def curve_runs(curve: np.ndarray, col: int) -> list[float]:
    """Row centres of the curve pixels in one column."""
    rows = np.where(curve[Y0 + 1 : Y1, col])[0] + Y0 + 1
    if len(rows) == 0:
        return []
    groups = np.split(rows, np.where(np.diff(rows) > 1)[0] + 1)
    return [float(g.mean()) for g in groups]


def track(runs: dict[int, list[float]], seed: int, start: float, order, tol=2.5):
    """Follow one curve from the seed, predicting each row from the local slope."""
    points = {seed: start}
    hist = [(seed, start)]
    for c in order:
        if len(hist) >= 2:
            (c1, r1), (c2, r2) = hist[-2], hist[-1]
            pred = r2 + (r2 - r1) / (c2 - c1) * (c - c2)
        else:
            pred = hist[-1][1]
        rs = runs[c]
        if rs:
            j = int(np.argmin([abs(r - pred) for r in rs]))
            if abs(rs[j] - pred) <= tol:
                points[c] = rs[j]
                hist.append((c, rs[j]))
        if abs(c - hist[-1][0]) > 12:
            break
    return points


def clean(cols: np.ndarray, rows: np.ndarray):
    """Drop points that leave a rolling median; the curves are smooth."""
    keep = np.ones(len(cols), dtype=bool)
    # A wide window first, to reject sustained excursions onto a neighbour.
    for size, tol in ((25, 3.0), (9, 2.0), (9, 2.0)):
        med = median_filter(rows[keep], size=size, mode="nearest")
        bad = np.abs(rows[keep] - med) > tol
        idx = np.where(keep)[0]
        keep[idx[bad]] = False
    cols, rows = cols[keep], rows[keep]
    # A fragment beyond a gap is where the tracker lost the curve and re-latched.
    segments = np.split(np.arange(len(cols)), np.where(np.diff(cols) > 8)[0] + 1)
    longest = max(segments, key=len)
    return cols[longest], rows[longest]


def enforce_order(tracks: list[dict[int, float]], tol=1.0) -> None:
    """
    Drop points where two tracks coincide or a shorter bunch lies below a longer one.

    The curves are ordered by bunch length, except that the three shortest
    bunches merge beyond s = 2.5 sigma, so either case means a track jumped onto
    a neighbour.
    """
    merge_col = int(COL0 + 2.5 * PX_PER_UNIT)
    for i, (upper, lower) in enumerate(zip(tracks[:-1], tracks[1:])):
        limit = merge_col if i < 2 else X1
        for c in [c for c in upper if c in lower and c < limit]:
            if lower[c] - upper[c] < tol:
                del upper[c]
                del lower[c]


def main(source: str | None) -> None:
    im = load(source)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    blue = (b > 0.55) & (r < 0.5) & (g < 0.5)
    red = (r > 0.55) & (g < 0.5) & (b < 0.5)
    curve = blue | red

    cols = np.arange(X0 + 1, X1)
    runs = {c: curve_runs(curve, c) for c in cols}
    # Seed near s = sigma, where the nine curves are ordered by bunch length.
    seed = min(
        (c for c in cols if len(runs[c]) == 9),
        key=lambda c: abs((c - COL0) / PX_PER_UNIT - 1.0),
    )

    tracks = []
    for start in runs[seed]:
        pts = track(runs, seed, start, range(seed + 1, X1))
        pts.update(track(runs, seed, start, range(seed - 1, X0, -1)))
        tracks.append(pts)
    enforce_order(tracks)

    rows_out = []
    for sigma, pts in zip(SIGMAS_UM, tracks):
        # Ahead of s = -1.5 sigma the curves pile up within a few pixels of zero.
        c = np.array([k for k in sorted(pts) if (k - COL0) / PX_PER_UNIT >= -1.5])
        c, rr = clean(c, np.array([pts[k] for k in c]))
        s_over_sigma = (c - COL0) / PX_PER_UNIT
        W = (ROW0 - rr) / PX_PER_V
        rows_out.append(np.column_stack([np.full(len(c), sigma), s_over_sigma, W]))

    table = np.vstack(rows_out)
    header = (
        "ECHO wake potentials digitized from Fig. 7 (left) of Zagorodnov, Tomin, "
        "Chen, Brinker, NIM A 995, 165111 (2021), arXiv:2010.10204v2, by "
        "scripts/digitize_zagorodnov_fig7.py. Pixel resolution is about 0.03 in "
        "s/sigma and 6 V/pC in W.\nsigma_um,s_over_sigma,W_V_per_pC"
    )
    np.savetxt(OUT, table, fmt=["%g", "%.3f", "%.1f"], delimiter=",", header=header)
    print(f"wrote {len(table)} points to {OUT}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
