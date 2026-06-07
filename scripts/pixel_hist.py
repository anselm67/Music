"""Compare grayscale pixel-value distributions: PDMX vs KernSheet.

Samples N random PNGs from each dataset, accumulates a 256-bin histogram of
grayscale pixel values, normalises to a density, and overlays both on one plot.

Usage:
    uv run python scripts/pixel_hist.py [-n 400] [-o build/pixel_hist.png]
"""

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

DATASETS = {
    "PDMX": Path("/home/anselm/datasets/PDMX/build/png"),
    "KernSheet": Path("/home/anselm/datasets/KernSheet/build/png"),
}


def dataset_hist(root: Path, n: int, seed: int) -> tuple[np.ndarray, int]:
    """Return (256-bin counts over [0,255], n_images_sampled)."""
    files = list(root.rglob("*.png"))
    random.Random(seed).shuffle(files)
    files = files[:n]

    counts = np.zeros(256, dtype=np.int64)
    for p in files:
        arr = np.asarray(Image.open(p).convert("L"), dtype=np.uint8)
        counts += np.bincount(arr.ravel(), minlength=256)
    return counts, len(files)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", type=int, default=400, help="images sampled per dataset")
    ap.add_argument("-o", default="build/pixel_hist.png", help="output PNG")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(14, 5))
    bins = np.arange(256)

    for name, root in DATASETS.items():
        counts, k = dataset_hist(root, args.n, args.seed)
        density = counts / counts.sum()
        label = f"{name} (n={k})"
        ax_lin.plot(bins, density, label=label, alpha=0.8)
        ax_log.semilogy(bins, density, label=label, alpha=0.8)
        # Quick stats to stdout.
        total = counts.sum()
        mean = (bins * counts).sum() / total
        white_frac = counts[250:].sum() / total
        black_frac = counts[:6].sum() / total
        print(
            f"{name:10s}  imgs={k:4d}  mean={mean:6.1f}  "
            f"black(<6)={black_frac:6.3%}  white(>=250)={white_frac:6.3%}"
        )

    for ax, title in ((ax_lin, "linear"), (ax_log, "log-y")):
        ax.set_title(f"Grayscale pixel distribution ({title})")
        ax.set_xlabel("pixel value (0=black, 255=white)")
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
