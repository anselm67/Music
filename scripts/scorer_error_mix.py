"""Diagnostic: is the scorer's error budget substitution- or structure-dominated?

Settles the "spellcheck layer" design question with a number. For each predicted
stave it aligns the record sequence to its GT stave (Levenshtein with backtrace)
and tallies:

  * substitutions  — wrong token at an aligned record  -> a soft, position-wise
                     refiner CAN fix these.
  * insertions/deletions — pred has an extra / missing record -> STRUCTURAL; a
                     fixed-length spellcheck layer CANNOT fix these (needs a
                     length-changing seq2seq pass).

It also measures the motivating symptom directly: within a multi-stave system, do
the spines agree on the number of barline (`=`) records? Pred vs GT.

Run:
    uv run python scripts/scorer_error_mix.py ravel -n 300
    uv run python scripts/scorer_error_mix.py merge1 -n 300 --pdmx-home <PDMX>
"""

import argparse
from pathlib import Path

from tqdm import tqdm

from kernsheet import KernSheet, KernSheetSource
from noter import Vocab
from pdmx import PDMX, PdmxSource
from scorer import ScorerDataset

from cli.scorer import _group_by_system, _load_for_inference


def align_ops(gt: list[str], pred: list[str]) -> tuple[int, int, int, int]:
    """Levenshtein backtrace -> (matches, substitutions, insertions, deletions).

    insertion = pred has a record GT lacks; deletion = GT record missing from pred.
    Both are structural; substitution is position-wise (spellcheck-fixable).
    """
    n, m = len(gt), len(pred)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if gt[i - 1] == pred[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j - 1] + cost, dp[i - 1][j] + 1, dp[i][j - 1] + 1)
    i, j = n, m
    match = sub = ins = dele = 0
    while i > 0 or j > 0:
        diag = i > 0 and j > 0
        if diag and gt[i - 1] == pred[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            match += 1
            i, j = i - 1, j - 1
        elif diag and dp[i][j] == dp[i - 1][j - 1] + 1:
            sub += 1
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            dele += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return match, sub, ins, dele


def barline_counts(seqs: list[list[str]]) -> list[int]:
    return [sum(1 for r in seq if r.startswith("=")) for seq in seqs]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("-n", type=int, default=200, help="Pages to evaluate.")
    ap.add_argument(
        "--kern-home", type=Path, default=Path("/home/anselm/datasets/KernSheet")
    )
    ap.add_argument("--pdmx-home", type=Path, default=None)
    ap.add_argument("--csv", default="System2.csv")
    args = ap.parse_args()

    config, module = _load_for_inference(args.model)
    if args.pdmx_home is not None:
        home = args.pdmx_home
        source = PdmxSource(PDMX(home, args.csv, -1, args.n))
    else:
        home = args.kern_home
        source = KernSheetSource(KernSheet(home))
    vocab = Vocab.load(home / "build/vocab.json")
    dataset = ScorerDataset(config, source, vocab)

    tot = {"match": 0, "sub": 0, "ins": 0, "dele": 0}
    paired_staves = 0
    stave_count_match = stave_count_total = 0
    bar_agree_pred = bar_agree_gt = bar_total = 0

    n = min(args.n, len(dataset))
    for idx in tqdm(range(n), desc="pages"):
        image, _gs, _gv, gt_assign, stave_tokens = dataset[idx]
        boxes, tokens, owners = module.predict(image.unsqueeze(0).to(module.device))
        num_gt = int((gt_assign != -1).sum())
        gt_by_sys = _group_by_system(gt_assign, num_gt)
        pred_by_sys = _group_by_system(owners, boxes.shape[0])

        for sys_id, gt_ks in gt_by_sys.items():
            pred_ks = pred_by_sys.get(sys_id, [])
            gt_seqs = [vocab.i2tok(stave_tokens[k][1:]) for k in gt_ks]
            pred_seqs = [vocab.i2tok(tokens[k].cpu()) for k in pred_ks]

            stave_count_total += 1
            stave_count_match += int(len(gt_seqs) == len(pred_seqs))

            # Substitution-vs-structure ratio over order-paired staves.
            for gt_seq, pred_seq in zip(gt_seqs, pred_seqs):
                mt, sb, ins, de = align_ops(gt_seq, pred_seq)
                tot["match"] += mt
                tot["sub"] += sb
                tot["ins"] += ins
                tot["dele"] += de
                paired_staves += 1

            # Barline-agreement symptom over GT multi-stave systems (one shared
            # population so the GT/pred fractions are comparable). Pred "agrees"
            # only if it detected >=2 spines AND their barline counts match.
            if len(gt_seqs) >= 2:
                bar_total += 1
                bar_agree_gt += int(len(set(barline_counts(gt_seqs))) == 1)
                bar_agree_pred += int(
                    len(pred_seqs) >= 2 and len(set(barline_counts(pred_seqs))) == 1
                )

    errors = tot["sub"] + tot["ins"] + tot["dele"]
    structural = tot["ins"] + tot["dele"]
    print(f"\n{n} pages | {paired_staves} order-paired staves")
    print(f"stave-count match (pred==GT): {stave_count_match}/{stave_count_total}")
    print("\n--- error mix (over paired staves) ---")
    print(f"  matches:       {tot['match']:>7}")
    print(f"  substitutions: {tot['sub']:>7}   (spellcheck-fixable)")
    print(f"  insertions:    {tot['ins']:>7}   (structural)")
    print(f"  deletions:     {tot['dele']:>7}   (structural)")
    if errors:
        print(f"\n  substitution share: {tot['sub'] / errors:6.1%} of errors")
        print(f"  structural share:   {structural / errors:6.1%} of errors")
    print("\n--- barline-count agreement across spines (multi-stave systems) ---")
    if bar_total:
        print(f"  GT   spines agree: {bar_agree_gt}/{bar_total}  (sanity, want ~100%)")
        print(f"  Pred spines agree: {bar_agree_pred}/{bar_total}")


if __name__ == "__main__":
    main()
