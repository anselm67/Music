"""Compare PDMX vs KernSheet appearance, and calibrate ScanAugment against it.

Samples N random PNGs per source and accumulates two distributions:
  * grayscale pixel values (256-bin) — the photometric gap (paper tone / pedestal)
  * stroke width (2x distance-transform over ink pixels) — the "fatter" gap

With ``--augment P`` a third "PDMX+aug" source is added: each sampled PDMX page is
passed through ``ScanAugment(P)`` so its augmented pixel + stroke-width curves can be
overlaid on KernSheet's. The goal is to tune ScanAugment so "PDMX+aug" lands on
KernSheet rather than guess the parameters.

Usage:
    uv run python scripts/pixel_hist.py [-n 400] [--augment 1.0] [-o out.png]
"""

import argparse
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from sheetmusic import ScanAugment

DATASETS = {
    "PDMX": Path("/home/anselm/datasets/PDMX/build/png"),
    "KernSheet": Path("/home/anselm/datasets/KernSheet/build/png"),
}
# Max stroke half-width tracked (px); 2x => widths up to 2*MAX_DT.
MAX_DT = 20


def stroke_widths(arr: np.ndarray) -> np.ndarray:
    """Per-ink-pixel stroke width (px) via the distance transform.

    Binarise ink (dark), distance-transform the ink mask, and return 2x the
    distance at each ink pixel — a fatter stroke yields larger interior
    distances, so the distribution shifts right as ink spreads.
    """
    ink = (arr < 128).astype(np.uint8)
    if ink.sum() == 0:
        return np.empty(0, dtype=np.float32)
    dt = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    return 2.0 * dt[ink > 0]


def augment_arr(arr: np.ndarray, aug: ScanAugment) -> np.ndarray:
    """Apply ScanAugment to a uint8 grayscale array, return uint8."""
    x = torch.from_numpy(arr).float().unsqueeze(0) / 255.0
    y = aug(x).clamp(0.0, 1.0)
    return (y.squeeze(0).numpy() * 255.0).astype(np.uint8)


def source_stats(
    root: Path, n: int, seed: int, aug: ScanAugment | None
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (256-bin pixel counts, stroke-width histogram, n_images)."""
    files = list(root.rglob("*.png"))
    random.Random(seed).shuffle(files)
    files = files[:n]

    pix = np.zeros(256, dtype=np.int64)
    sw = np.zeros(2 * MAX_DT + 1, dtype=np.int64)
    sw_bins = np.arange(2 * MAX_DT + 2)
    for p in files:
        arr = np.asarray(Image.open(p).convert("L"), dtype=np.uint8)
        if aug is not None:
            arr = augment_arr(arr, aug)
        pix += np.bincount(arr.ravel(), minlength=256)
        w = np.clip(stroke_widths(arr), 0, 2 * MAX_DT)
        sw += np.histogram(w, bins=sw_bins)[0]
    return pix, sw, len(files)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", type=int, default=400, help="images sampled per source")
    ap.add_argument("-o", default="build/pixel_hist.png", help="output PNG")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--augment",
        type=float,
        default=0.0,
        help="If >0, add a PDMX+aug source via ScanAugment(this prob).",
    )
    args = ap.parse_args()

    sources = [(name, root, None) for name, root in DATASETS.items()]
    if args.augment > 0:
        sources.append(("PDMX+aug", DATASETS["PDMX"], ScanAugment(args.augment)))

    fig, (ax_lin, ax_log, ax_sw) = plt.subplots(1, 3, figsize=(20, 5))
    bins = np.arange(256)
    sw_centers = np.arange(2 * MAX_DT + 1)

    for name, root, aug in sources:
        pix, sw, k = source_stats(root, args.n, args.seed, aug)
        density = pix / pix.sum()
        sw_density = sw / sw.sum()
        label = f"{name} (n={k})"
        ax_lin.plot(bins, density, label=label, alpha=0.8)
        ax_log.semilogy(bins, density, label=label, alpha=0.8)
        ax_sw.plot(sw_centers, sw_density, label=label, alpha=0.8)
        mean = (bins * pix).sum() / pix.sum()
        white_frac = pix[250:].sum() / pix.sum()
        black_frac = pix[:6].sum() / pix.sum()
        sw_mean = (sw_centers * sw).sum() / sw.sum()
        sw_med = np.searchsorted(np.cumsum(sw), sw.sum() / 2)
        print(
            f"{name:10s}  imgs={k:4d}  mean={mean:6.1f}  "
            f"black(<6)={black_frac:6.3%}  white(>=250)={white_frac:6.3%}  "
            f"stroke_px mean={sw_mean:4.1f} median={sw_med:2d}"
        )

    for ax, title, xlabel in (
        (ax_lin, "pixels (linear)", "value (0=black, 255=white)"),
        (ax_log, "pixels (log-y)", "value (0=black, 255=white)"),
        (ax_sw, "stroke width", "width (px)"),
    ):
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("density")
        ax.legend()
        ax.grid(True, alpha=0.3)

    out = Path(args.o)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
