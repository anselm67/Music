#!/usr/bin/env python
"""Does the scorer's joint fine-tune (the differentiable crop bridge) actually
buy transcription accuracy over a plain two-stage staffer→crop→noter pipeline?

Runs the *identical* end-to-end eval on the same seeded pages for two models:

  A) the trained scorer checkpoint (joint fine-tuned), and
  B) a zero-shot MERGE of the same run's standalone staffer + noter branches
     (``ScorerModule.load_from_checkpoints``, no joint training). At inference the
     differentiable bridge crops the same pixels a plain crop would, so (B) *is*
     the plain-crop baseline.

The gap A−B is the joint fine-tune's contribution.

  uv run python scripts/bridge_benefit.py \
      --scorer sibelius --staffer sibelius-mixed --noter sibelius-mixed \
      --kern-home ~/datasets/KernSheet --size 500
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from tqdm import tqdm

from cli.scorer import _load_for_inference, _match_staves, _similarity
from kernsheet import KernSheet, KernSheetSource
from noter import Vocab
from pdmx import PDMX, PdmxSource
from scorer import ScorerConfig, ScorerDataset, ScorerModule
from sheetmusic import Source


def _merge(
    staffer: str, noter: str, vocab: Vocab, device: torch.device
) -> ScorerModule:
    """Zero-shot merge of the standalone branches — no joint training."""
    config = ScorerConfig()
    config.use_vocab(vocab)
    staffer_ckpt = Path("checkpoints") / "staffer" / staffer / "last.ckpt"
    noter_ckpt = Path("checkpoints") / "noter" / noter / "last.ckpt"
    module = ScorerModule.load_from_checkpoints(config, staffer_ckpt, noter_ckpt)
    return module.to(device).eval()


@torch.no_grad()
def evaluate(
    module: ScorerModule,
    dataset: ScorerDataset,
    indices: list[int],
    max_chords: int,
    device: torch.device,
) -> tuple[float, float, int, int, int]:
    """Mirror cli.scorer.run_eval: matched-stave similarity + detection counts."""
    sims: list[float] = []
    total_gt = total_pred = miscount = 0
    for idx in tqdm(indices, desc="eval", leave=False):
        image, _gt_sys, gt_stave, gt_assign, stave_tokens, _arts = dataset[idx]
        num_gt = int((gt_assign != -1).sum())
        boxes, tokens, _pred_arts, _owners = module.predict(
            image.unsqueeze(0).to(device)
        )
        num_pred = tokens.shape[0]
        total_gt += num_gt
        total_pred += num_pred
        miscount += int(num_pred != num_gt)
        H = image.shape[-2]
        gt_cy = ((gt_stave[:num_gt, 1] + gt_stave[:num_gt, 3]) / 2).numpy()
        pred_cy = (((boxes[:, 2] + boxes[:, 4]) / 2) / H).cpu().numpy()
        for g, p in _match_staves(gt_cy, pred_cy):
            sims.append(_similarity(stave_tokens[g], tokens[p].cpu(), max_chords))
    avg = sum(sims) / len(sims) if sims else 0.0
    mn = min(sims) if sims else 0.0
    return avg, mn, total_pred, total_gt, miscount


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scorer", default="sibelius")
    ap.add_argument("--staffer", default="sibelius-mixed")
    ap.add_argument("--noter", default="sibelius-mixed")
    ap.add_argument("--home", type=Path, default=Path("/home/anselm/datasets/PDMX"))
    ap.add_argument("--kern-home", type=Path, default=None)
    ap.add_argument("--csv", default="System2.csv")
    ap.add_argument("--size", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(
        args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    )

    if args.kern_home is not None:
        home = args.kern_home
        source: Source = KernSheetSource(KernSheet(home))
    else:
        home = args.home
        source = PdmxSource(PDMX(home, args.csv, -1, -1))
    vocab = Vocab.load(home / "build" / "vocab.json")

    # The trained scorer (A) defines the dataset/transform; the merge (B) shares
    # the same staffer image_shape, so one dataset + one index set feeds both.
    config, trained = _load_for_inference(args.scorer)
    trained = trained.to(device)
    dataset = ScorerDataset(config, source, vocab)
    n = min(args.size, len(dataset))
    indices = random.Random(args.seed).sample(range(len(dataset)), n)
    max_chords = config.noter.max_chords

    merged = _merge(args.staffer, args.noter, vocab, device)

    print(f"\n=== bridge-benefit eval · {n} pages · seed {args.seed} · {home} ===")
    rows = []
    for tag, model in (
        (f"MERGE (plain crop) {args.staffer}+{args.noter}", merged),
        (f"SCORER (joint FT)  {args.scorer}", trained),
    ):
        avg, mn, pred, gt, miss = evaluate(model, dataset, indices, max_chords, device)
        rows.append((tag, avg, mn, pred, gt, miss))
        print(
            f"\n{tag}\n"
            f"  matched-stave similarity  avg {avg:.1%}   min {mn:.1%}\n"
            f"  staves {pred} pred / {gt} GT · miscount {miss}/{n} pages"
        )

    d = rows[1][1] - rows[0][1]
    print(f"\nΔ(scorer − merge) avg similarity: {d:+.1%}")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
