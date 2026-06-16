"""Decompose the KernSheet end-to-end similarity ceiling.

The scorer eval reports only min/avg/max matched-stave similarity. A ~91% MEAN can
be a uniform 9% error OR a fat tail of near-zero pages (matching failures, edition
mismatch, broken GT) dragging an otherwise-high median. This probe runs the same
eval loop but keeps EVERY per-stave similarity with its (score_id, page) identity,
then prints the distribution + the worst staves with decoded GT-vs-prediction tokens
so the failures can be classified by eye (model misread vs edition mismatch vs
alignment) against the actual scan.

    uv run python scripts/scorer_ceiling_probe.py mahler-ks --size 500 --worst 25

Read-only; no training. KernSheet only (the real-scan corpus whose ceiling we chase).
"""
# ruff: noqa: E501  — diagnostic script; long report f-strings/comments read better unwrapped

import argparse
import random
from pathlib import Path

import torch
from tqdm import tqdm

from cli.scorer import _load_for_inference, _match_staves, _similarity
from kernsheet import KernSheet, KernSheetSource
from noter.noter_vocab import Vocab
from scorer.scorer_dataset import ScorerDataset


def strip_dots(seq: torch.Tensor, dot_id: int, sil: int) -> torch.Tensor:
    """Drop pure-continuation (`.`) timesteps from a (T, max_chords) sequence so a
    cross-stave time-grid mismatch isn't scored as a musical error."""
    dot_only = (seq[:, 0] == dot_id) & (seq[:, 1:] == sil).all(dim=1)
    return seq[~dot_only]


def max_run(toks: list[str]) -> int:
    """Longest run of identical consecutive timesteps (a runaway-decode signature)."""
    best = run = 0
    prev = None
    for t in toks:
        run = run + 1 if t == prev else 1
        best = max(best, run)
        prev = t
    return best


def classify(gt_len: int, pred_len: int, pr_t: list[str]) -> str:
    if max_run(pr_t) >= 6 or pred_len > 1.8 * gt_len + 5:
        return "runaway-repeat"
    if pred_len < 0.6 * gt_len:
        return "early-EOS/truncated"
    return "note-divergence"


def decode_gt(seq: torch.Tensor, vocab: Vocab) -> list[str]:
    return vocab.i2tok(seq[1:])  # drop SOS, stop at EOS (mirrors _similarity)


def decode_pred(seq: torch.Tensor, vocab: Vocab) -> list[str]:
    return vocab.i2tok(seq)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--home", default="/home/anselm/datasets/KernSheet")
    ap.add_argument("--size", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--worst", type=int, default=25)
    args = ap.parse_args()

    home = Path(args.home)
    config, module = _load_for_inference(args.name)
    vocab = Vocab.load(home / "build/vocab.json")
    source = KernSheetSource(KernSheet(home))
    dataset = ScorerDataset(config, source, vocab)

    n = min(args.size, len(dataset))
    indices = random.Random(args.seed).sample(range(len(dataset)), n)
    mc = config.noter.max_chords
    dot_id, sil = vocab.encode("."), vocab.SIL

    records = []  # dict per matched stave
    for idx in tqdm(indices, desc="probe"):
        image, _gt_sys, gt_stave, gt_assign, stave_tokens = dataset[idx]
        score_id, page = dataset.items[idx]
        num_gt = int((gt_assign != -1).sum())
        boxes, tokens, _ = module.predict(
            image.unsqueeze(0).to(module.device), barline_ids=None, beam_width=args.beam
        )
        H = image.shape[-2]
        gt_cy = ((gt_stave[:num_gt, 1] + gt_stave[:num_gt, 3]) / 2).numpy()
        pred_cy = (((boxes[:, 2] + boxes[:, 4]) / 2) / H).cpu().numpy()
        for g, p in _match_staves(gt_cy, pred_cy):
            gt_seq, pred_seq = stave_tokens[g], tokens[p].cpu()
            sim = _similarity(gt_seq, pred_seq, mc)
            # same metric, continuation timesteps removed = the cross-stave-recoverable part
            dot_sim = _similarity(
                strip_dots(gt_seq, dot_id, sil), strip_dots(pred_seq, dot_id, sil), mc
            )
            gt_t, pr_t = decode_gt(gt_seq, vocab), decode_pred(pred_seq, vocab)
            records.append(
                {
                    "sim": sim,
                    "dot_sim": dot_sim,
                    "sid": score_id,
                    "page": page,
                    "gt_len": len(gt_t),
                    "pred_len": len(pr_t),
                    "gt_t": gt_t,
                    "pr_t": pr_t,
                }
            )

    N = len(records)
    sims = sorted(r["sim"] for r in records)
    mean = sum(sims) / N
    median = sims[N // 2]
    dot_mean = sum(r["dot_sim"] for r in records) / N
    print(f"\n=== {args.name} · KernSheet · {n} pages · {N} matched staves ===")
    print(f"raw          mean {mean:.1%}   median {median:.1%}   min {sims[0]:.1%}")
    print(
        f"`.`-stripped mean {dot_mean:.1%}  (+{dot_mean - mean:.1%} — the continuation-grid"
        f" slice a cross-stave/system decode could recover)"
    )

    print("\n--- distribution (raw per-stave similarity) ---")
    bins = [0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0001]
    labels = [
        "0-10",
        "10-30",
        "30-50",
        "50-70",
        "70-80",
        "80-90",
        "90-95",
        "95-99",
        "99-100",
    ]
    for lo, hi, lab in zip(bins, bins[1:], labels):
        c = sum(1 for s in sims if lo <= s < hi)
        print(f"  {lab:>7}%  {c:4d} ({100 * c / N:4.1f}%) {'#' * round(50 * c / N)}")
    for thr in (0.5, 0.8, 0.9):
        print(f"  staves < {thr:.0%}: {sum(s < thr for s in sims)}")

    print("\n--- mean similarity by GT length (is the tail the long/dense staves?) ---")
    for lo, hi in [(0, 30), (30, 60), (60, 100), (100, 1_000_000)]:
        grp = [r for r in records if lo <= r["gt_len"] < hi]
        if grp:
            m = sum(r["sim"] for r in grp) / len(grp)
            print(
                f"  gt {lo:>3}-{hi if hi < 1000 else '+':<4} tok: {len(grp):4d} staves  mean {m:.1%}"
            )

    tail = [r for r in records if r["sim"] < 0.8]
    print(f"\n--- failure-mode mix of the <80% tail ({len(tail)} staves) ---")
    from collections import Counter

    modes = Counter(classify(r["gt_len"], r["pred_len"], r["pr_t"]) for r in tail)
    for mode, c in modes.most_common():
        print(f"  {mode:<22} {c:4d} ({100 * c / max(len(tail), 1):4.1f}% of tail)")

    print(f"\n--- {args.worst} worst matched staves (GT vs prediction) ---")
    for r in sorted(records, key=lambda r: r["sim"])[: args.worst]:
        mode = classify(r["gt_len"], r["pred_len"], r["pr_t"])
        print(
            f"\n[{r['sim']:.0%}] {r['sid']} p{r['page']}  "
            f"(gt {r['gt_len']} / pred {r['pred_len']} tok · {mode})"
        )
        print(f"  GT  : {' '.join(r['gt_t'])[:220]}")
        print(f"  PRED: {' '.join(r['pr_t'])[:220]}")

    # The adjudication set: note-divergence staves in the mid band (right length, wrong
    # notes — NOT the catastrophic runaways). Open each scan and decide per stave:
    # model-misread vs edition-mismatch (scanned edition != .krn).
    band = [
        r
        for r in records
        if 0.55 <= r["sim"] < 0.80
        and classify(r["gt_len"], r["pred_len"], r["pr_t"]) == "note-divergence"
    ]
    band.sort(key=lambda r: r["sim"])
    print(
        f"\n=== ADJUDICATE: {args.worst} note-divergence staves in 55-80% (of {len(band)}) ==="
    )
    print("    (open the scan for each — model-misread vs edition-mismatch?)")
    for r in band[: args.worst]:
        print(
            f"\n[{r['sim']:.0%}] {r['sid']} p{r['page']}  (gt {r['gt_len']} / pred {r['pred_len']} tok)"
        )
        print(f"  GT  : {' '.join(r['gt_t'])[:260]}")
        print(f"  PRED: {' '.join(r['pr_t'])[:260]}")


if __name__ == "__main__":
    main()
