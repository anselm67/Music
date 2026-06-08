#!/usr/bin/env python
"""Per-dimension box delta diagnostic for staffer predictions vs GT stave boxes.

Reports Δleft, Δright, Δtop, Δbottom (pred − GT, pixels) for matched stave
boxes.  Signed mean shows bias direction; absolute stats show error magnitude.

  uv run python scripts/staffer_box_deltas.py --staffer stave-primary-grid-full
  uv run python scripts/staffer_box_deltas.py \
      --staffer stave-primary-grid-full-kernsheet \
      --kern-home /home/anselm/datasets/KernSheet
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import fields
from pathlib import Path

import torch
from scipy.optimize import linear_sum_assignment
from torchvision.transforms import v2
from tqdm import tqdm

from kernsheet import KernSheet, KernSheetSource
from pdmx import PDMX, PdmxSource
from sheetmusic import PerImageNormalize, Source
from staffer import StafferConfig, StafferDataset, StafferModule


def _make_transform(cfg: StafferConfig) -> v2.Transform:
    # Mirror StafferDataset.transform: stretch resize + per-image normalisation.
    return v2.Compose(
        [
            v2.Grayscale(),
            v2.Resize(
                cfg.image_shape,
                interpolation=cfg.interpolation,
                antialias=cfg.antialias,
            ),
            v2.ToDtype(torch.float, scale=True),
            PerImageNormalize(),
        ]
    )


def _config_from_ckpt(path: Path) -> StafferConfig:
    hp = torch.load(path, weights_only=False)["hyper_parameters"]
    known = {f.name for f in fields(StafferConfig)}
    return StafferConfig(**{k: v for k, v in hp.items() if k in known})


@torch.no_grad()
def _predict_boxes(
    model: StafferModule, img: torch.Tensor
) -> list[tuple[float, float, float, float]]:
    """Normalised (left, top, right, bottom) for active staves, top→bottom."""
    stave_tb, stave_logits, boundary_logits, sys_lr, _ = (
        t.squeeze(0) for t in model.forward(img)
    )
    logit = stave_logits.squeeze(-1)
    active = (logit > 0.0).nonzero(as_tuple=True)[0]
    if active.numel() == 0:
        return []
    active = active[stave_tb[active, 0].argsort()]
    boundary = (boundary_logits.squeeze(-1)[active] > 0.0).long()
    group = (boundary.cumsum(0) - 1).clamp(0, sys_lr.shape[0] - 1)
    boxes = []
    for i, q in enumerate(active.tolist()):
        left, right = sys_lr[group[i]].tolist()
        top, bot = stave_tb[q].tolist()
        boxes.append((left, top, right, bot))
    return boxes


def _stats(xs: list[float]) -> str:
    if not xs:
        return "  n=0"
    s = sorted(xs)
    n = len(s)
    mean = sum(s) / n
    p50 = s[n // 2]
    p90 = s[int(0.90 * n)]
    p99 = s[int(0.99 * n)]
    mx = max(s)
    return (
        f"mean={mean:+6.2f}  p50={p50:+6.2f}  p90={p90:+6.2f}"
        f"  p99={p99:+6.2f}  max={mx:+7.2f}"
    )


def _abs_stats(xs: list[float]) -> str:
    return _stats([abs(x) for x in xs])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--staffer", default="stave-primary-grid-full")
    ap.add_argument("--home", type=Path, default=Path("/home/anselm/datasets/PDMX"))
    ap.add_argument("--kern-home", type=Path, default=None)
    ap.add_argument("--csv", default="System2.csv")
    ap.add_argument("--pages", type=int, default=300)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    )

    ckpt = Path("checkpoints") / "staffer" / args.staffer / "last.ckpt"
    cfg = _config_from_ckpt(ckpt)

    source: Source
    if args.kern_home is not None:
        source = KernSheetSource(KernSheet(args.kern_home))
    else:
        source = PdmxSource(PDMX(args.home, args.csv, -1, args.limit))

    dataset = StafferDataset(cfg, source)
    model = (
        StafferModule.load_from_checkpoint(
            ckpt, config=cfg, weights_only=False, map_location=device
        )
        .to(device)
        .eval()
    )
    transform = _make_transform(cfg)

    pages = list({(sid, pno) for sid, pno, _, _ in dataset.items})
    random.shuffle(pages)
    pages = pages[: args.pages]

    dleft: list[float] = []
    dright: list[float] = []
    dtop: list[float] = []
    dbot: list[float] = []
    n_gt = n_missed = n_extra = 0

    for score_id, page_number in tqdm(pages, desc="pages"):
        try:
            raw = source.image(score_id, page_number)
            img = transform(raw).unsqueeze(0).to(device)
        except Exception as e:
            print(f"skip {score_id} p{page_number}: {e}", file=sys.stderr)
            continue

        score = source.score(score_id)
        page = score.pages[page_number - 1]
        W, H = page.image_width, page.image_height
        # Un-normalise predicted boxes to original page pixels (where the GT boxes
        # live) by inverting StafferDataset's stretch normalisation: it scales
        # coords by sx, sy = 1/W, 1/H, so original = pred / s.
        sx, sy = 1.0 / W, 1.0 / H

        gt_staves = [stave for system in page.systems for stave in system.staves]
        if not gt_staves:
            continue
        n_gt += len(gt_staves)

        pred = _predict_boxes(model, img)  # normalised ltrb
        n_extra += max(0, len(pred) - len(gt_staves))

        if not pred:
            n_missed += len(gt_staves)
            continue

        gt_cy = [(s.box.top + s.box.bottom) / 2.0 for s in gt_staves]
        pred_cy = [((t + b) / 2.0) / sy for (_, t, _, b) in pred]

        cost = [[abs(g - p) for p in pred_cy] for g in gt_cy]
        rows, cols = linear_sum_assignment(cost)
        matched: dict[int, int] = {}
        for r, c in zip(rows, cols):
            thresh = 0.5 * gt_staves[r].box.height
            if cost[r][c] <= thresh:
                matched[r] = c

        n_missed += len(gt_staves) - len(matched)

        for gi, pi in matched.items():
            pl, pt, pr, pb = pred[pi]
            gt = gt_staves[gi].box
            dleft.append(pl / sx - gt.left)
            dright.append(pr / sx - gt.right)
            dtop.append(pt / sy - gt.top)
            dbot.append(pb / sy - gt.bottom)

    print(f"\nBox deltas (pred − GT, px)  staffer={args.staffer}")
    print(
        f"{len(pages)} pages · {n_gt} GT staves · "
        f"miss {n_missed} ({100 * n_missed / max(n_gt, 1):.1f}%) · extra {n_extra}\n"
    )
    print(f"{'':>7}  {'mean':>7}  {'p50':>7}  {'p90':>7}  {'p99':>7}  {'max':>8}")
    for label, xs in (
        ("Δleft", dleft),
        ("Δright", dright),
        ("Δtop", dtop),
        ("Δbot", dbot),
    ):
        if not xs:
            print(f"  {label:<6}  n=0")
            continue
        print(f"  {label:<6}  signed: {_stats(xs)}")
        print(f"  {'':6}  absol.: {_abs_stats(xs)}")


if __name__ == "__main__":
    main()
