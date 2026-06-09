#!/usr/bin/env python
"""Beam-recall diagnostic for the barline-count invariant (noter).

Question this answers: of the staves whose *best* (beam[0]) decode has the wrong
number of barlines, what fraction have a candidate with the right count somewhere
in the top-K beam?  That number is the ceiling for a rerank step that penalises
barline-count disagreement — if it's high, rerank alone is worth building; if
it's low, the right structure isn't being generated and a localized-repair path
is needed instead.

Within a system every spine shares the same barline skeleton, so the bar count is
a cross-staff invariant; GT gives the per-system target.  Barlines live in slot 0
(the primary/structural token beam search ranks over), so counting `=` tokens in
the slot-0 stream is all this needs — no durations, no meter, no Fraction.

  uv run python scripts/noter_bar_recall.py \
      --noter enhanced3-center+jitter-kernsheet \
      --kern-home /home/anselm/datasets/KernSheet --limit 400 --beam 8
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

import torch
from torch import Tensor
from tqdm import tqdm

from kernsheet import KernSheet, KernSheetSource
from noter import NoterConfig, NoterDataset, NoterModule, Vocab


def slot0_stream(seq: Tensor, vocab: Vocab) -> list[str]:
    """Decode the slot-0 (primary) token per timestep, stopping at EOS."""
    toks: list[str] = []
    for t in range(seq.shape[0]):
        i0 = int(seq[t, 0].item())
        if i0 == Vocab.EOS:
            break
        if i0 in (Vocab.PAD, Vocab.SOS, Vocab.SIL):
            continue
        toks.append(vocab.decode(i0))
    return toks


def bar_count(slot0: list[str]) -> int:
    """Number of barlines (`=` / `==`) in a slot-0 token stream."""
    return sum(1 for tok in slot0 if tok.startswith("="))


@torch.no_grad()
def beam_all(module: NoterModule, image: Tensor, width: Tensor, beam: int) -> Tensor:
    """Beam search returning ALL `beam` candidates (best-first), (beam, T, chords).

    Mirrors ScorerModule._beam_single: beams over slot 0, slots 1+ greedy from the
    chosen beam's logits.  Returns every beam (SOS stripped) rather than just [0].
    """
    c = module.config
    device = module.device
    model = module.model

    memory, src_pad = model.encode(image, width)
    memory = memory.repeat_interleave(beam, dim=0)
    src_pad = src_pad.repeat_interleave(beam, dim=0)

    generated = torch.full(
        (beam, 1, c.max_chords), Vocab.SOS, dtype=torch.long, device=device
    )
    scores = torch.full((beam,), float("-inf"), device=device)
    scores[0] = 0.0
    done = torch.zeros(beam, dtype=torch.bool, device=device)

    for _ in range(c.max_seqlen - 1):
        tgt_pad = (generated == Vocab.SIL).all(dim=-1)
        logits = model.decode(
            generated, memory, module._causal_mask(generated.shape[1]), tgt_pad, src_pad
        )
        step_logits = logits[:, -1, :, :]  # (beam, max_chords, V)
        V = step_logits.shape[-1]

        slot0_lp = step_logits[:, 0, :].log_softmax(-1)
        slot0_lp[done] = float("-inf")
        slot0_lp[done, Vocab.EOS] = 0.0

        top_scores, top_idx = (
            (scores.unsqueeze(-1) + slot0_lp).view(beam * V).topk(beam)
        )
        beam_from = top_idx // V
        token0 = top_idx % V

        scores = top_scores
        done = done[beam_from]
        generated = generated[beam_from]

        next_tokens = step_logits[beam_from].argmax(-1)
        next_tokens[:, 0] = token0
        next_tokens[done] = Vocab.EOS
        done = done | (next_tokens[:, 0] == Vocab.EOS)

        generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)
        if bool(done.all()):
            break

    return generated[:, 1:]  # (beam, T, max_chords), best-first; SOS stripped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--noter", default="enhanced3-center+jitter-kernsheet")
    ap.add_argument(
        "--kern-home", type=Path, default=Path("/home/anselm/datasets/KernSheet")
    )
    ap.add_argument("--limit", type=int, default=400, help="val staves to evaluate")
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    )

    from cli.noter import config_from_checkpoint

    ckpt = Path("checkpoints") / "noter" / args.noter / "last.ckpt"
    config: NoterConfig = config_from_checkpoint(ckpt)
    vocab = Vocab.load(args.kern_home / "build" / "vocab.json")
    source = KernSheetSource(KernSheet(args.kern_home))

    # Reproduce the seed-42 train/val split the model was trained against, then
    # evaluate on the held-out val indices only.
    full = NoterDataset(
        config, source, vocab, count=config.train_len + config.valid_len
    )
    _, val_ds = torch.utils.data.random_split(
        full,
        [config.train_len, config.valid_len],
        generator=torch.Generator().manual_seed(42),
    )
    val_indices = list(val_ds.indices)
    random.shuffle(val_indices)

    module = NoterModule.load_from_checkpoint(
        ckpt, config=config, weights_only=False, map_location=device
    )
    module.to(device).eval()

    n_eval = 0
    argmax_clean = 0
    argmax_violating = 0
    recoverable = 0
    delta_histo: Counter[int] = Counter()  # (argmax bar count − GT), signed

    progress = tqdm(val_indices, desc="staves")
    for idx in progress:
        try:
            image, width, gt_seq = full[idx]
        except Exception:
            continue
        gt_bars = bar_count(slot0_stream(gt_seq[1:], vocab))  # strip SOS
        if gt_bars < 1:
            continue  # nothing to match against

        beams = beam_all(
            module,
            image.unsqueeze(0).to(device),
            width.unsqueeze(0).to(device),
            args.beam,
        )
        cand_bars = [
            bar_count(slot0_stream(beams[k], vocab)) for k in range(beams.shape[0])
        ]

        n_eval += 1
        delta_histo[cand_bars[0] - gt_bars] += 1
        if cand_bars[0] == gt_bars:
            argmax_clean += 1
        else:
            argmax_violating += 1
            if any(cb == gt_bars for cb in cand_bars):
                recoverable += 1

        if argmax_violating:
            progress.set_postfix(
                viol=argmax_violating, recov=recoverable, clean=argmax_clean
            )
        if n_eval >= args.limit:
            break

    print(
        f"\nBeam-recall (barline-count invariant)  noter={args.noter}  beam={args.beam}"
    )
    print(f"  evaluated {n_eval} val staves\n")
    print(
        f"  argmax (beam[0]) correct count: {argmax_clean:4d} "
        f"({100 * argmax_clean / max(n_eval, 1):.1f}%)"
    )
    print(
        f"  argmax wrong count:             {argmax_violating:4d} "
        f"({100 * argmax_violating / max(n_eval, 1):.1f}%)"
    )
    if argmax_violating:
        print(
            f"  └─ recoverable in top-{args.beam}:  {recoverable:4d} "
            f"({100 * recoverable / argmax_violating:.1f}% of violating)  "
            f"← rerank ceiling"
        )
    print("\n  argmax bar-count delta (pred − GT):")
    for d in sorted(delta_histo):
        print(f"    {d:+d}: {delta_histo[d]}")


if __name__ == "__main__":
    main()
