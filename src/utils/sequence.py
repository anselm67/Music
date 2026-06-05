import logging
from itertools import zip_longest
from typing import Protocol

import numpy as np
import torch
from torch import Tensor

_RED = "\033[31m"
_RESET = "\033[0m"


class _Source(Protocol):
    def records(self, id: str, first_bar: int, last_bar: int) -> list[str] | None: ...


class _Vocab(Protocol):
    PAD: int
    SOS: int
    EOS: int

    def tok2i(self, tokens: list[str], max_chords: int) -> Tensor: ...


def load_sequence(
    source: _Source,
    vocab: _Vocab,
    score_id: str,
    spine_number: int,
    first_bar: int,
    last_bar: int,
    max_seqlen: int,
    max_chords: int,
) -> Tensor | None:
    """Load one stave's token sequence from source, shape (max_seqlen, max_chords).

    Returns SOS-prefixed, EOS-terminated tensor, or None if the bars are missing,
    the sequence is too long, or any record can't be decoded.
    """
    try:
        records = source.records(score_id, first_bar, last_bar)
    except Exception as e:
        logging.error(f"{score_id}: {e}")
        return None
    if records is None:
        logging.error(f"{score_id}: bars {first_bar}:{last_bar} not found.")
        return None
    if len(records) + 2 > max_seqlen:
        logging.error(
            f"{score_id}: bars {first_bar}:{last_bar}, "
            f"sequence too long {len(records)} (max {max_seqlen - 2})"
        )
        return None
    s_sos = torch.full((1, max_chords), vocab.SOS)
    s_eos = torch.full((1, max_chords), vocab.EOS)
    body = torch.full((max_seqlen - 1, max_chords), vocab.PAD)
    for idx, text in enumerate(records):
        try:
            # Real KernSheet records occasionally have fewer spines than the
            # system's staff count (malformed/misaligned bar range); skip the
            # sample rather than letting the IndexError crash the worker.
            str_tok = text.split("\t")[spine_number]
            body[idx, :] = vocab.tok2i(str_tok.strip().split(), max_chords=max_chords)
        except Exception as e:
            logging.error(f"{score_id}: {e}")
            return None
    body[len(records), :] = s_eos
    return torch.cat([s_sos, body])


def format_sequence_columns(
    left: list[str],
    right: list[str],
    left_header: str = "GT",
    right_header: str = "Pred",
    highlight_mismatches: bool = True,
) -> str:
    """Return a two-column string comparing left and right token sequences.

    Mismatching tokens in the right column are highlighted red when
    highlight_mismatches is True (default). Fill slots (when one sequence is
    shorter) are never highlighted.
    """
    left_width = max(max((len(s) for s in left), default=0), len(left_header)) + 2
    right_width = max(max((len(s) for s in right), default=0), len(right_header))

    rows: list[str] = []
    rows.append(f"{left_header:<{left_width}}| {right_header}")
    rows.append("-" * left_width + "+" + "-" * (right_width + 2))

    for l_tok, r_tok in zip_longest(left, right, fillvalue=""):
        is_fill = l_tok == "" or r_tok == ""
        mismatch = highlight_mismatches and not is_fill and l_tok != r_tok
        r_display = f"{_RED}{r_tok}{_RESET}" if mismatch else r_tok
        rows.append(f"{l_tok:<{left_width}}| {r_display}")

    return "\n".join(rows)


def chord_distance(chord1: Tensor, chord2: Tensor, pad_id: int) -> int:
    """Edit distance between the actual notes in two PAD-padded chord tensors."""
    c1 = sorted(t for t in chord1.tolist() if t != pad_id)
    c2 = sorted(t for t in chord2.tolist() if t != pad_id)
    m, n = len(c1), len(c2)
    dp = np.zeros((m + 1, n + 1), dtype=int)
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if c1[i - 1] == c2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return int(dp[m][n])


def sequence_edit_distance(seq1: Tensor, seq2: Tensor, pad_id: int) -> int:
    """Sequence-level edit distance where each element is a chord.

    Insertion/deletion of a whole chord costs max_chords so it stays on the
    same scale as substitution (which is at most max_chords via chord_distance).
    """
    len1, len2 = seq1.size(0), seq2.size(0)
    max_chords = seq1.size(1)
    dp = np.zeros((len1 + 1, len2 + 1), dtype=int)
    for i in range(len1 + 1):
        dp[i][0] = i * max_chords
    for j in range(len2 + 1):
        dp[0][j] = j * max_chords
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = chord_distance(seq1[i - 1], seq2[j - 1], pad_id)
            dp[i][j] = min(
                dp[i - 1][j] + max_chords,
                dp[i][j - 1] + max_chords,
                dp[i - 1][j - 1] + cost,
            )
    return int(dp[len1][len2])


def strip_eos(seq: Tensor, eos_id: int) -> Tensor:
    """Return seq sliced up to (not including) the first EOS timestep."""
    eos_pos = (seq[:, 0] == eos_id).nonzero(as_tuple=False)
    return seq[: int(eos_pos[0].item())] if eos_pos.numel() > 0 else seq
