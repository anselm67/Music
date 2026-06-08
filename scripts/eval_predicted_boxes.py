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

  # KernSheet (real scans) — real-scan jitter spec for the noter retrain:
  uv run python scripts/eval_predicted_boxes.py \
      --kern-home /home/anselm/datasets/KernSheet \
      --noter enhanced3-kernsheet --staffer stave-primary-grid-full-kernsheet \
      --pages 300 --device cuda
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from dataclasses import fields
from pathlib import Path

import torch
from scipy.optimize import linear_sum_assignment
from torchvision.transforms import v2
from tqdm import tqdm

from kernsheet import KernSheet, KernSheetSource
from noter import NoterConfig, NoterDataset, NoterModule, Vocab
from pdmx import PDMX, PdmxSource
from sheetmusic import Box, LetterboxResize, PerImageNormalize, Source, letterbox_scale
from staffer import StafferConfig, StafferModule
from utils import sequence_edit_distance, strip_eos


def make_staffer_transform(cfg: StafferConfig) -> v2.Transform:
    # Mirror StafferDataset.transform: letterbox + per-image normalisation.
    return v2.Compose(
        [
            v2.Grayscale(),
            LetterboxResize(
                cfg.image_shape,
                interpolation=cfg.interpolation,
                antialias=cfg.antialias,
                fill=255,
            ),
            v2.ToDtype(torch.float, scale=True),
            PerImageNormalize(),
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
    ap.add_argument(
        "--home",
        type=Path,
        default=Path("/home/anselm/datasets/PDMX"),
        help="PDMX dataset root (used unless --kern-home is given).",
    )
    ap.add_argument(
        "--kern-home",
        type=Path,
        default=None,
        help="KernSheet dataset root; selects KernSheet instead of PDMX. "
        "Use the KernSheet-fine-tuned models, e.g. --noter enhanced3-kernsheet "
        "--staffer stave-primary-grid-full-kernsheet.",
    )
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

    if args.kern_home is not None:
        source: Source = KernSheetSource(KernSheet(args.kern_home))
        vocab_home: Path = args.kern_home
    else:
        source = PdmxSource(PDMX(args.home, args.csv, -1, args.limit))
        vocab_home = args.home
    vocab = Vocab.load(vocab_home / "build/vocab.json")

    # noter
    n_ckpt = Path("checkpoints") / "noter" / args.noter / "last.ckpt"
    n_hp = torch.load(n_ckpt, weights_only=False)["hyper_parameters"]
    n_keep = {f.name for f in fields(NoterConfig)}
    n_cfg = NoterConfig(**{k: v for k, v in n_hp.items() if k in n_keep})
    dataset = NoterDataset(n_cfg, source, vocab)
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

    page_h, page_w = n_cfg.page_shape  # (960, 688)

    # Group GT staves (noter items) by page image.
    page_to_idx: dict[tuple[str, int], list[int]] = defaultdict(list)
    for idx, item in enumerate(dataset.items):
        page_to_idx[(item[0], item[1])].append(idx)
    pages = list(page_to_idx)
    random.shuffle(pages)
    pages = pages[: args.pages]

    # Accumulators.
    base: list[float] = []  # GT-box similarity
    pred: list[float] = []  # predicted-box similarity
    dtop: list[float] = []  # |pred_top - gt_top| px
    dbot: list[float] = []  # |pred_bot - gt_bot| px
    dleft: list[float] = []  # |pred_left - gt_left| px
    dright: list[float] = []  # |pred_right - gt_right| px
    n_gt = n_missed = n_extra = 0
    # Per-miss diagnostics: (page, gt_cy_norm, n_preds, n_gts, nearest_pred_px).
    misses: list[tuple[str, float, int, int, float]] = []
    # Per-page: (raw_brightness_0to1, n_gt, n_missed_on_page).
    page_stats: list[tuple[float, int, int]] = []

    for page_key in tqdm(pages, desc="pages"):
        idxs = page_to_idx[page_key]
        score_id, page_number = page_key
        # --- staffer inference on the page ---
        try:
            raw = dataset.source.image(score_id, page_number)
            page_img = s_transform(raw).unsqueeze(0).to(device)
        except Exception:
            continue
        brightness = float(raw.float().mean()) / 255.0
        page_missed = 0
        # Un-normalise staffer-space predictions into the noter-resized page pixels
        # where the GT boxes live. The two branches letterbox into different
        # canvases, so go via original px: pred / (scale_s/target_s) → original,
        # then * scale_n → noter-resized (both StafferDataset & NoterDataset use the
        # single aspect-preserving letterbox_scale).
        h0, w0 = raw.shape[-2], raw.shape[-1]
        th_s, tw_s = s_cfg.image_shape
        scale_s = letterbox_scale(h0, w0, th_s, tw_s)
        scale_n = letterbox_scale(h0, w0, page_h, page_w)
        convx, convy = tw_s / scale_s * scale_n, th_s / scale_s * scale_n
        pred_boxes = staffer_active_boxes(staffer, page_img)
        pred_cy = [((t + b) / 2.0) * convy for (_l, t, _r, b) in pred_boxes]

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
            sid, pno, gt_box, spine, fb, lb = dataset.items[idx]

            res = dataset._load_image(sid, pno, gt_box)
            seq = dataset.load_sequence(sid, spine, fb, lb)
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
                page_missed += 1
                nearest = min((abs(cy - pcy) for pcy in pred_cy), default=float("inf"))
                misses.append(
                    (f"{sid} p{pno}", cy / page_h, len(pred_boxes), len(gts), nearest)
                )
                continue
            pl, pt, pr, pb = pred_boxes[match[gpos]]
            pbox = Box(
                (int(pl * convx), int(pt * convy)),
                (int(pr * convx), int(pb * convy)),
            )
            dtop.append(abs(pt * convy - gt_box.top))
            dbot.append(abs(pb * convy - gt_box.bottom))
            dleft.append(abs(pl * convx - gt_box.left))
            dright.append(abs(pr * convx - gt_box.right))
            res2 = dataset._load_image(sid, pno, pbox)
            if res2 is None:
                pred.append(0.0)
                continue
            img1, w1 = res2
            p1 = noter.predict(
                img1.unsqueeze(0).to(device), torch.tensor([w1]).to(device)
            )
            pred.append(similarity(seq, p1[0], n_cfg.max_chords))

        page_stats.append((brightness, len(gts), page_missed))

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
    for label, xs in (
        ("Δtop", dtop),
        ("Δbot", dbot),
        ("Δleft", dleft),
        ("Δright", dright),
    ):
        print(
            f"  {label:>6} {avg(xs):>5.2f}p {pct(xs, 0.50):>5.2f}p "
            f"{pct(xs, 0.90):>5.2f}p {pct(xs, 0.99):>5.2f}p {max(xs):>6.1f}p"
        )

    # --- miss breakdown: undercount (fewer preds than GT) vs threshold reject ---
    undercount = sum(1 for _p, _cy, npred, ngt, _d in misses if npred < ngt)
    near = sum(1 for *_x, d in misses if d != float("inf") and d < 30)
    print(
        f"\n  misses: {len(misses)} — undercount(pred<gt) {undercount}, "
        f"a pred within 30px {near} (matched-but-rejected/dense)"
    )
    print("  by vertical position (gt_cy / page_h):")
    ybins = [0] * 10
    for _p, cy, *_x in misses:
        ybins[min(9, int(cy * 10))] += 1
    print("   " + " ".join(f"{b:>3}" for b in ybins) + "  (deciles top→bottom)")
    print("  sample missed pages (page · cy · n_pred/n_gt · nearest_px):")
    for p, cy, npred, ngt, d in misses[:20]:
        ds = "inf" if d == float("inf") else f"{d:.0f}"
        print(f"    {p:<28} cy={cy:.2f} {npred}/{ngt} near={ds}")

    # --- darkness vs misses (capacity-controlled) ---
    # M=16 stave queries: pages with >16 GT staves miss by capacity regardless of
    # brightness, so analyse within-capacity (n_gt<=16) pages separately.
    M = 16
    within = [(b, ng, nm) for (b, ng, nm) in page_stats if ng <= M]
    over = [(b, ng, nm) for (b, ng, nm) in page_stats if ng > M]
    clean = [b for (b, _ng, nm) in within if nm == 0]
    dirty = [b for (b, _ng, nm) in within if nm > 0]
    print(
        f"\n  darkness vs misses — {len(page_stats)} pages "
        f"({len(within)} within capacity n_gt<=16, {len(over)} over):"
    )
    print(
        f"    within-cap pages WITH a miss:  n={len(dirty)} brightness "
        f"mean={avg(dirty):.3f} p50={pct(dirty, 0.5):.3f}"
    )
    print(
        f"    within-cap pages with NO miss: n={len(clean)} brightness "
        f"mean={avg(clean):.3f} p50={pct(clean, 0.5):.3f}"
    )
    if over:
        ob = [b for (b, _n, _m) in over]
        om = sum(nm for (_b, _n, nm) in over)
        print(
            f"    over-capacity pages (>16 staves): n={len(over)} "
            f"brightness mean={avg(ob):.3f}, total misses on them={om}"
        )
    # miss-rate by brightness quartile, within-capacity only
    print("    within-cap miss-rate by brightness quartile (dark→light):")
    sw = sorted(within, key=lambda t: t[0])
    if sw:
        q = max(1, len(sw) // 4)
        for k in range(4):
            chunk = sw[k * q : (k + 1) * q] if k < 3 else sw[3 * q :]
            if not chunk:
                continue
            tot_gt = sum(ng for (_b, ng, _m) in chunk)
            tot_miss = sum(nm for (_b, _n, nm) in chunk)
            br = avg([b for (b, _n, _m) in chunk])
            rate = 100 * tot_miss / max(tot_gt, 1)
            print(
                f"      Q{k + 1} brightness~{br:.3f}: "
                f"miss {tot_miss}/{tot_gt} ({rate:.1f}%)"
            )


if __name__ == "__main__":
    main()
