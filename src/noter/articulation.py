"""Shared articulation-head helpers: training loss + free-running eval metrics.

Used by both the standalone noter (``NoterModule`` / ``noter`` CLI) and the
end-to-end scorer (``ScorerModule`` / ``scorer`` CLI) so the two never drift on
how the per-note multi-hot (tie / staccato / fermata / accent) is scored.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from kern import ARTICULATIONS
from utils import align_sequences

from .noter_vocab import Vocab

# Display names aligned to the ``kern.ARTICULATIONS`` flag order.
ARTICULATION_NAMES = ("tie-to-next", "tie-from-prev", "staccato", "fermata", "accent")


def articulation_loss(
    art_logits: Tensor,  # (..., NUM_ARTICULATIONS)
    art_labels: Tensor,  # (..., NUM_ARTICULATIONS)
    slot_mask: Tensor,  # (...) — True on real (non-PAD) note slots
) -> tuple[Tensor, Tensor, Tensor]:
    """Masked multi-label BCE over the articulation head → (loss, acc, recall).

    Scored only on the note slots ``slot_mask`` marks (the same non-PAD slots the
    token loss covers). ``recall`` (positive recall) catches an all-negatives
    collapse, since most notes carry no articulation. The caller applies its own
    ``articulation_weight`` and logging.
    """
    if not slot_mask.any():
        z = art_logits.sum() * 0.0
        return z, z, z
    logits = art_logits[slot_mask]  # (N, A)
    labels = art_labels[slot_mask]  # (N, A)
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    with torch.no_grad():
        preds = logits > 0  # logit > 0 <=> prob > 0.5
        pos = labels > 0.5
        acc = (preds == pos).float().mean()
        recall = (preds & pos).sum() / pos.sum().clamp(min=1)
    return loss, acc, recall


def tally_articulations(
    gt_tokens: Tensor,  # (Lg, max_chords)
    gt_arts: Tensor,  # (Lg, max_chords, A)
    pred_tokens: Tensor,  # (Lp, max_chords)
    pred_arts: Tensor,  # (Lp, max_chords, A)
    tp: np.ndarray,  # type: ignore[type-arg]
    fp: np.ndarray,  # type: ignore[type-arg]
    fn: np.ndarray,  # type: ignore[type-arg]
) -> None:
    """Accumulate per-flag TP/FP/FN of free-running articulation bits.

    Rows are paired by the token edit-distance alignment; bits are scored only on
    real note slots (token != PAD). An unmatched GT row contributes its set bits
    as false negatives, an unmatched predicted row as false positives.

    Within a paired chord, GT and predicted bits are compared positionally
    (slot k vs slot k), relying on the canonical low->high pitch sort the
    tokenizer bakes into both. Row pairing itself is order-blind (chord_distance
    sorts), so a chord predicted with a swapped/extra slot can misattribute that
    one row's bits — rare given canonical targets, and it only perturbs that row.
    """
    mc, num = gt_arts.shape[1], gt_arts.shape[2]
    zeros = torch.zeros(mc, num, dtype=torch.bool)
    for i, j in align_sequences(gt_tokens, pred_tokens, Vocab.PAD):
        if i is not None:
            g = (gt_arts[i] > 0.5) & (gt_tokens[i] != Vocab.PAD).unsqueeze(-1)
        else:
            g = zeros
        if j is not None:
            p = (pred_arts[j] > 0.5) & (pred_tokens[j] != Vocab.PAD).unsqueeze(-1)
        else:
            p = zeros
        tp += (g & p).sum(0).numpy()
        fp += (p & ~g).sum(0).numpy()
        fn += (g & ~p).sum(0).numpy()


def report_articulations(
    tp: np.ndarray,  # type: ignore[type-arg]
    fp: np.ndarray,  # type: ignore[type-arg]
    fn: np.ndarray,  # type: ignore[type-arg]
) -> None:
    """Print the per-flag + micro-average precision / recall / F1 table."""
    print("\nArticulations (free-running, per flag):")
    print(f"  {'flag':<16}{'prec':>7}{'rec':>7}{'F1':>7}{'support':>9}")
    for k, code in enumerate(ARTICULATIONS):
        support = int(tp[k] + fn[k])
        prec = tp[k] / (tp[k] + fp[k]) if tp[k] + fp[k] else 0.0
        rec = tp[k] / support if support else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        label = f"{code} {ARTICULATION_NAMES[k]}"
        print(f"  {label:<16}{prec:>7.1%}{rec:>7.1%}{f1:>7.1%}{support:>9}")
    t, f_, n_ = int(tp.sum()), int(fp.sum()), int(fn.sum())
    prec = t / (t + f_) if t + f_ else 0.0
    rec = t / (t + n_) if t + n_ else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    print(f"  {'(micro avg)':<16}{prec:>7.1%}{rec:>7.1%}{f1:>7.1%}{t + n_:>9}")
