#!/usr/bin/env python
"""One-off PRE-MERGE DIAGNOSTIC — not wired into any CLI, no src/ changes.

Measures how much noter's transcription degrades when fed staffer's PREDICTED
stave crops instead of ground-truth crops.

It answers the question behind the end-to-end merge: noter is trained on
geometrically perfect GT crops; at merge time it sees the detector's jittering
boxes. This quantifies (a) the token-similarity drop and (b) the matched-box
error magnitude (Δtop/Δbottom px) + detection miss/extra rate — i.e. the SPEC
for the noter box-jitter augmentation.

Both models run inference only. Everything is imported/reused from src/; the
single copied block is staffer predict's active-stave grouping (see below).

Run from /home/anselm/projects/Music (use --device cpu to avoid contending with
a running training on the GPU):

  uv run python scripts/eval_predicted_boxes.py \
      --noter enhanced3 --staffer stave-primary-grid \
      --csv System2.csv --pages 300 --device cuda
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from dataclasses import fields
from pathlib import Path

import torch
from scipy.optimize import linear_sum_assignment
from torchvision.io import decode_image
from torchvision.transforms import v2
from tqdm import tqdm

from noter import NoterConfig, NoterDataset, NoterModule, Vocab
from pdmx import PDMX
from sheetmusic import Box
from staffer import StafferConfig, StafferModule
from utils import sequence_edit_distance, strip_eos

# Mirror StafferDataset.transform (the normalize constants are dataset stats).
_NORM_MEAN, _NORM_STD = 0.9563435316085815, 0.16557540870879858


def make_staffer_transform(cfg: StafferConfig) -> v2.Transform:
    return v2.Compose(
        [
            v2.Grayscale(),
            v2.Resize(
                cfg.image_shape,
                interpolation=cfg.interpolation,
                antialias=cfg.antialias,
            ),
            v2.ToDtype(torch.float, scale=True),
            v2.Normalize(mean=[_NORM_MEAN], std=[_NORM_STD]),
        ]
    )


@torch.no_grad()
def staffer_active_boxes(
    model: StafferModule, img: torch.Tensor
) -> list[tuple[float, float, float, float]]:
    """Predicted stave boxes as normalised (left, top, right, bottom), top→bottom.

    Copied from `staffer predict` (cli/staffer.py): active queries are
    non-contiguous, so the boundary cumsum runs over active staves sorted
    top-to-bottom; each stave inherits (left, right) from its grouped system.
    """
    stave_tb, stave_logits, boundary_logits, sys_lr, _sys_logits = (
        t.squeeze(0) for t in model.forward(img)
    )
    stave_logit = stave_logits.squeeze(-1)
    active = (stave_logit > 0.0).nonzero(as_tuple=True)[0]
    if active.numel() == 0:
        return []
    active = active[stave_tb[active, 0].argsort()]  # top-to-bottom
    boundary = (boundary_logits.squeeze(-1)[active] > 0.0).long()
    group = (boundary.cumsum(0) - 1).clamp(0, sys_lr.shape[0] - 1)
    boxes = []
    for i, q in enumerate(active.tolist()):
        left, right = sys_lr[group[i]].tolist()
        top, bot = stave_tb[q].tolist()
        boxes.append((left, top, right, bot))
    return boxes


def similarity(
    gt_seq: torch.Tensor, pred_tokens: torch.Tensor, max_chords: int
) -> float:
    """1 - normalised edit distance, matching cli/noter.run_eval."""
    gt_content = strip_eos(gt_seq[1:], Vocab.EOS)
    pred_content = strip_eos(pred_tokens.cpu(), Vocab.EOS)
    edit = sequence_edit_distance(gt_content, pred_content, Vocab.PAD)
    max_cost = max(len(gt_content), len(pred_content)) * max_chords
    return 1.0 - edit / max_cost if max_cost > 0 else 1.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--noter", default="enhanced3")
    ap.add_argument("--staffer", default="stave-primary-grid")
    ap.add_argument("--home", type=Path, default=Path("/home/anselm/datasets/PDMX"))
    ap.add_argument("--csv", default="System2.csv")
    ap.add_argument("--pages", type=int, default=300, help="random pages to evaluate")
    ap.add_argument("--limit", type=int, default=-1, help="PDMX rows to load (-1=all)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    )

    pdmx = PDMX(args.home, args.csv, -1, args.limit)

    # noter
    n_ckpt = Path("checkpoints") / "noter" / args.noter / "last.ckpt"
    n_hp = torch.load(n_ckpt, weights_only=False)["hyper_parameters"]
    n_keep = {f.name for f in fields(NoterConfig)}
    n_cfg = NoterConfig(**{k: v for k, v in n_hp.items() if k in n_keep})
    dataset = NoterDataset(n_cfg, pdmx)
    noter = (
        NoterModule.load_from_checkpoint(
            n_ckpt, config=n_cfg, weights_only=False, map_location=device
        )
        .to(device)
        .eval()
    )

    # staffer
    s_ckpt = Path("checkpoints") / "staffer" / args.staffer / "last.ckpt"
    s_hp = torch.load(s_ckpt, weights_only=False)["hyper_parameters"]
    s_keep = {f.name for f in fields(StafferConfig)}
    s_cfg = StafferConfig(**{k: v for k, v in s_hp.items() if k in s_keep})
    staffer = (
        StafferModule.load_from_checkpoint(
            s_ckpt, config=s_cfg, weights_only=False, map_location=device
        )
        .to(device)
        .eval()
    )
    s_transform = make_staffer_transform(s_cfg)

    page_h, page_w = n_cfg.page_shape  # (966, 680)

    # Group GT staves (noter items) by page image.
    page_to_idx: dict[str, list[int]] = defaultdict(list)
    for idx, item in enumerate(dataset.items):
        page_to_idx[str(item[1])].append(idx)
    pages = list(page_to_idx)
    random.shuffle(pages)
    pages = pages[: args.pages]

    # Accumulators.
    base: list[float] = []  # GT-box similarity
    pred: list[float] = []  # predicted-box similarity
    dtop: list[float] = []  # |pred_top - gt_top| px
    dbot: list[float] = []  # |pred_bot - gt_bot| px
    n_gt = n_missed = n_extra = 0

    for png in tqdm(pages, desc="pages"):
        idxs = page_to_idx[png]
        # --- staffer inference on the page ---
        try:
            page_img = s_transform(decode_image(png)).unsqueeze(0).to(device)
        except Exception:
            continue
        pred_boxes = staffer_active_boxes(staffer, page_img)
        pred_cy = [((t + b) / 2.0) * page_h for (_l, t, _r, b) in pred_boxes]

        # GT staves on this page: (idx, center_y_px, height_px)
        gts = []
        for idx in idxs:
            box = dataset.items[idx][2]
            gts.append((idx, (box.top + box.bottom) / 2.0, box.height))
        n_gt += len(gts)

        # --- match GT <-> predicted by center-y (1-D optimal + threshold) ---
        match = {}  # gt position -> pred position
        if pred_boxes and gts:
            cost = torch.tensor(
                [[abs(cy - pcy) for pcy in pred_cy] for (_i, cy, _h) in gts]
            )
            rows, cols = linear_sum_assignment(cost.numpy())
            for r, c in zip(rows, cols):
                thresh = 0.5 * gts[r][2]  # half the GT staff height
                if cost[r, c].item() <= thresh:
                    match[r] = c
        # match values are distinct preds (1-1 assignment), so unmatched preds
        # = total preds - matched. Misses (unmatched GTs) are counted per-GT below.
        n_extra += len(pred_boxes) - len(match)

        # --- score each GT stave: GT-box baseline vs predicted-box ---
        for gpos, (idx, cy, _h) in enumerate(gts):
            mxl, png_path, gt_box, spine, fb, lb = dataset.items[idx]

            res = dataset._load_image(mxl, png_path, gt_box)
            seq = dataset._load_sequence(mxl, spine, fb, lb)
            if res is None or seq is None:
                continue
            img0, w0 = res
            p0 = noter.predict(
                img0.unsqueeze(0).to(device), torch.tensor([w0]).to(device)
            )
            base.append(similarity(seq, p0[0], n_cfg.max_chords))

            if gpos not in match:  # detection miss → whole staff lost
                pred.append(0.0)
                n_missed += 1
                continue
            pl, pt, pr, pb = pred_boxes[match[gpos]]
            pbox = Box(
                (int(pl * page_w), int(pt * page_h)),
                (int(pr * page_w), int(pb * page_h)),
            )
            dtop.append(abs(pt * page_h - gt_box.top))
            dbot.append(abs(pb * page_h - gt_box.bottom))
            res2 = dataset._load_image(mxl, png_path, pbox)
            if res2 is None:
                pred.append(0.0)
                continue
            img1, w1 = res2
            p1 = noter.predict(
                img1.unsqueeze(0).to(device), torch.tensor([w1]).to(device)
            )
            pred.append(similarity(seq, p1[0], n_cfg.max_chords))

    # --- report ---
    def avg(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else float("nan")

    def pct(xs: list[float], q: float) -> float:
        if not xs:
            return float("nan")
        s = sorted(xs)
        return s[min(len(s) - 1, int(q * len(s)))]

    print(f"\nPredicted-box tolerance — noter={args.noter} staffer={args.staffer}")
    print(f"{len(pages)} pages, {n_gt} GT staves, page_shape={n_cfg.page_shape}")
    miss_pct = 100 * n_missed / max(n_gt, 1)
    print(
        f"detection: matched {n_gt - n_missed}/{n_gt} "
        f"(miss {n_missed}, {miss_pct:.1f}%) · extra preds {n_extra}\n"
    )
    print(
        f"  similarity: base={avg(base):.4f} pred={avg(pred):.4f} "
        f"Δ={avg(base) - avg(pred):.4f}"
    )
    print("  matched-box error (jitter spec):")
    print(f"  {'':>6} {'mean':>6} {'p50':>6} {'p90':>6} {'p99':>6} {'max':>7}")
    for label, xs in (("Δtop", dtop), ("Δbot", dbot)):
        print(
            f"  {label:>6} {avg(xs):>5.2f}p {pct(xs, 0.50):>5.2f}p "
            f"{pct(xs, 0.90):>5.2f}p {pct(xs, 0.99):>5.2f}p {max(xs):>6.1f}p"
        )


if __name__ == "__main__":
    main()
