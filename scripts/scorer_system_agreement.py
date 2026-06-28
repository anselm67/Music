#!/usr/bin/env python
"""Per-system cross-staff barline-agreement diagnostic on the end-to-end SCORER.

The scorer's reranker would favour a system's joint beam combination whose staves
AGREE on barline positions.  This measures, on the REAL predicted-box pipeline
(detect -> crop -> transcribe, the debussy scorer), how much room that idea has.

Why the scorer, not a standalone-noter harness: the scorer IS the predicted-box
pipeline, and `predict()` returns `owners` — the detected system index per stave —
so the system grouping is detection-derived, not a GT bar-range key (which predicted
boxes would break).  Two earlier noter-level probes that wired GT crops + GT grouping
were dropped for exactly that reason.

Signature, not count: within a system every spine shares the same record rows, so a
barline falls on the SAME timestep index in each stave; the cross-staff invariant is
the set of `=` positions (a shared count with a shifted bar still disagrees).

Per predicted system (>=2 staves), over the staves' beams it reports:
  1. disagreement rate     — argmax signatures differ (what the rerank can act on).
  2. consistency ceiling   — does a mutually-consistent combo exist in the beams
                             (a signature present in EVERY stave's candidate set)?
  3. correctness + gap      — on systems whose staves cleanly match GT staves
                             (center-y Hungarian, no miss/extra): is GT reachable,
                             and is the logprob-best consistent combo actually GT?
                             A low gap ==> staves agree on a SHARED error.

`ScorerModule._beam_single` returns only beam[0], so this carries a `beam_all` that
returns ALL candidates + their cumulative slot-0 logprobs — the same surface the
reranker itself will need.

  uv run python scripts/scorer_system_agreement.py \
      --scorer debussy --kern-home /home/anselm/datasets/KernSheet \
      --pages 300 --beam 16 --device cuda
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch import Tensor
from tqdm import tqdm

from cli.scorer import _group_by_system, _match_staves, config_from_checkpoint
from kernsheet import KernSheet, KernSheetSource
from noter import Vocab
from scorer import ScorerConfig, ScorerDataset, ScorerModule, build_stave_boxes
from scorer.scorer_module import active_grouping


def bar_positions(seq: Tensor, vocab: Vocab) -> tuple[int, ...]:
    """Timestep indices (record rows) carrying a barline in slot 0, up to EOS.

    Raw timestep `t` is the shared cross-staff coordinate: spines of one system have
    the same record rows, so equal position tuples == agreeing barlines.
    """
    positions: list[int] = []
    for t in range(seq.shape[0]):
        i0 = int(seq[t, 0].item())
        if i0 == Vocab.EOS:
            break
        if i0 in (Vocab.PAD, Vocab.SOS, Vocab.SIL):
            continue
        if vocab.decode(i0).startswith("="):
            positions.append(t)
    return tuple(positions)


@torch.no_grad()
def beam_all(
    module: ScorerModule, crop: Tensor, width: Tensor, beam: int
) -> tuple[Tensor, Tensor]:
    """Beam search for ONE stave crop returning ALL candidates + their scores.

    Mirrors `ScorerModule._beam_single` (beams over slot 0, slots 1+ greedy from the
    chosen beam) but keeps every beam and its cumulative slot-0 logprob instead of
    returning only beam[0].  Returns (generated (beam, T, chords) SOS-stripped,
    scores (beam,)), best-first.
    """
    c = module.config.noter
    device = crop.device
    noter = module.model.noter

    memory, src_pad = noter.encode(crop, width)
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
        logits = noter.decode(
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

    return generated[:, 1:], scores  # best-first; SOS stripped


@torch.no_grad()
def detect_and_decode(
    module: ScorerModule, image: Tensor, beam: int, vocab: Vocab
) -> tuple[Tensor, Tensor, list[dict[tuple[int, ...], float]], list[tuple[int, ...]]]:
    """Run detect -> crop -> beam_all per stave.

    Returns (boxes (K, 5) px, owners (K,), per-stave sig->best-logprob dicts,
    per-stave argmax signature).  Mirrors `ScorerModule.predict` but keeps the full
    beam.  Empty tensors / lists when nothing fires.
    """
    stave_tb, stave_logits, boundary_logits, sys_lr, _ = module.model.detect(image)
    sel, owners = active_grouping(
        stave_tb[0], stave_logits[0], boundary_logits[0], sys_lr.shape[1]
    )
    hw = (int(image.shape[-2]), int(image.shape[-1]))
    boxes = build_stave_boxes(stave_tb, sys_lr, [sel], [owners], hw)
    if boxes.shape[0] == 0:
        return boxes, owners, [], []

    crops, widths = module.model.crop(image, boxes)
    sig_scores: list[dict[tuple[int, ...], float]] = []
    argmax_sigs: list[tuple[int, ...]] = []
    for k in range(crops.shape[0]):
        cands, scores = beam_all(module, crops[k : k + 1], widths[k : k + 1], beam)
        sigs = [bar_positions(cands[b], vocab) for b in range(cands.shape[0])]
        best: dict[tuple[int, ...], float] = {}
        for sig, sc in zip(sigs, scores.tolist()):
            if sig not in best or sc > best[sig]:
                best[sig] = sc
        sig_scores.append(best)
        argmax_sigs.append(sigs[0])
    return boxes, owners, sig_scores, argmax_sigs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scorer", default="debussy")
    ap.add_argument(
        "--kern-home", type=Path, default=Path("/home/anselm/datasets/KernSheet")
    )
    ap.add_argument("--pages", type=int, default=300, help="random pages to evaluate")
    ap.add_argument("--beam", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    )

    ckpt = Path("checkpoints") / "scorer" / args.scorer / "last.ckpt"
    config: ScorerConfig = config_from_checkpoint(ckpt)
    vocab = Vocab.load(args.kern_home / "build" / "vocab.json")
    source = KernSheetSource(KernSheet(args.kern_home))
    dataset = ScorerDataset(config, source, vocab)

    module = ScorerModule.load_from_checkpoint(
        ckpt, config=config, weights_only=False, map_location=device
    )
    module.to(device).eval()

    n_pages = min(args.pages, len(dataset))
    indices = random.sample(range(len(dataset)), n_pages)

    total_gt = total_pred = miscount_pages = 0
    n_sys = 0  # predicted multi-stave systems
    agree = disagree = consist_reachable = 0  # among multi-stave systems
    clean = 0  # systems cleanly aligned to a GT system (no miss/extra)
    # among clean systems: correct (both==GT), agree-but-wrong (shared error)
    argmax_correct = agree_but_wrong = correct_reachable = rerank_correct = 0

    for idx in tqdm(indices, desc="pages"):
        image, _gt_sys, gt_stave, gt_assign, stave_tokens, _arts = dataset[idx]
        image = image.unsqueeze(0).to(device)
        num_gt = int((gt_assign != -1).sum())
        boxes, owners, sig_scores, argmax_sigs = detect_and_decode(
            module, image, args.beam, vocab
        )
        num_pred = boxes.shape[0]
        total_gt += num_gt
        total_pred += num_pred
        miscount_pages += int(num_pred != num_gt)
        if num_pred == 0:
            continue

        # Center-y match (normalised): GT boxes already normalised ltrb; predicted
        # boxes are px, so divide by page height.  Threshold by half the GT staff
        # height so a miss/extra doesn't forge a cross-staff pair.
        H = image.shape[-2]
        gt_cy = ((gt_stave[:num_gt, 1] + gt_stave[:num_gt, 3]) / 2).numpy()
        gt_h = (gt_stave[:num_gt, 3] - gt_stave[:num_gt, 1]).numpy()
        pred_cy = (((boxes[:, 2] + boxes[:, 4]) / 2) / H).cpu().numpy()
        pred2gt: dict[int, int] = {}
        for g, p in _match_staves(gt_cy, pred_cy):
            if abs(gt_cy[g] - pred_cy[p]) < 0.5 * gt_h[g]:
                pred2gt[p] = g
        gt_sig = {g: bar_positions(stave_tokens[g][1:], vocab) for g in range(num_gt)}
        gt_by_sys = _group_by_system(gt_assign, num_gt)

        for ks in _group_by_system(owners, num_pred).values():
            if len(ks) < 2:
                continue  # no cross-staff agreement to measure
            n_sys += 1
            sigs0 = [argmax_sigs[k] for k in ks]
            agrees = all(s == sigs0[0] for s in sigs0)
            if agrees:
                agree += 1
            else:
                disagree += 1
            # A signature present in EVERY stave's candidate set is a consistent combo.
            common = set.intersection(*(set(sig_scores[k]) for k in ks))
            if common:
                consist_reachable += 1

            # Correctness needs a clean GT alignment: every pred stave matched a GT
            # stave, all in ONE gt system, that gt system fully covered (no miss).
            if not all(k in pred2gt for k in ks):
                continue
            matched = [pred2gt[k] for k in ks]
            sys_of = {int(gt_assign[g].item()) for g in matched}
            if len(sys_of) != 1:
                continue
            gsys = sys_of.pop()
            if len(gt_by_sys.get(gsys, [])) != len(ks):
                continue
            targets = {gt_sig[g] for g in matched}  # GT spines share positions
            if len(targets) != 1:
                continue
            target = targets.pop()

            clean += 1
            if all(s == target for s in sigs0):
                argmax_correct += 1  # correct ==> staves also agree
            elif agrees:
                agree_but_wrong += 1  # staves agree on a SHARED wrong timeline
            if target in common:  # each stave has a candidate equal to GT
                correct_reachable += 1
            if common:  # the reranker's pick: logprob-best consistent signature
                pick = max(common, key=lambda s: sum(sig_scores[k][s] for k in ks))
                if pick == target:
                    rerank_correct += 1

    def pct(x: int, d: int) -> str:
        return f"{100 * x / max(d, 1):.1f}%"

    print(f"\nScorer per-system agreement  scorer={args.scorer}  beam={args.beam}")
    print(f"  evaluated {n_pages} pages")
    print(f"  staves: {total_pred} predicted / {total_gt} GT")
    print(f"  pages with miscount: {miscount_pages} / {n_pages}\n")
    print(f"  predicted multi-stave systems: {n_sys}  (agree + disagree)")
    print(f"  ├─ argmax AGREE:               {agree:4d} ({pct(agree, n_sys)})")
    print(
        f"  └─ argmax DISAGREE (target):   {disagree:4d} ({pct(disagree, n_sys)})  "
        f"<- what the rerank can act on"
    )
    print(
        f"     consistent combo in beams:  {consist_reachable:4d} "
        f"({pct(consist_reachable, n_sys)})  <- agreement ceiling"
    )
    print(f"\n  cleanly GT-aligned systems: {clean} (of {n_sys})")
    if clean:
        disagree_clean = clean - argmax_correct - agree_but_wrong
        print(
            f"  ├─ argmax jointly correct:     {argmax_correct:4d} "
            f"({pct(argmax_correct, clean)})  (agree AND == GT)"
        )
        print(
            f"  ├─ agree but WRONG:            {agree_but_wrong:4d} "
            f"({pct(agree_but_wrong, clean)})  <- staves share an error"
        )
        print(
            f"  └─ disagree:                  {disagree_clean:4d} "
            f"({pct(disagree_clean, clean)})"
        )
        print(
            f"     GT reachable in beams:     {correct_reachable:4d} "
            f"({pct(correct_reachable, clean)})  <- correctness ceiling"
        )
        print(
            f"     rerank pick CORRECT:       {rerank_correct:4d} "
            f"({pct(rerank_correct, clean)})  <- vs ceiling = the agreement gap"
        )


if __name__ == "__main__":
    main()
