"""Codec between a simplified token's `pitch/duration` text and a separate
duration scalar.

The noter/scorer decoder predicts duration as a classification head over an
analytic duration table (``NUM_DUR_BINS`` bins, derived from
``log2(note length)``) rather than baking it into the flat token vocab (so the
vocab carries pitch only). This module is the codec for that split:

* :func:`split_duration` pulls the numeric duration off a note/rest token,
  returning ``(base_token, log2_length)``.
* :func:`length_to_bin` / :func:`bin_to_suffix` map a ``log2_length`` to its
  table bin index and back to a duration suffix; :func:`bin_boundaries` /
  :func:`bin_log2_lengths` expose the table for the tensor-side loss/decode.
* :func:`snap_suffix` / :func:`join_duration` quantise a predicted
  ``log2_length`` back onto the table and re-attach it.

The table is enumerated from ``(denominator, dots)`` combinations, NOT built
from training counts, so a duration that is rare or unseen in training still
resolves to the nearest musically representable value.
"""

import bisect
import math
import re
from fractions import Fraction

# Tokens that carry a *numeric* duration: notes (`C#/4`, `cc:x/8`) and rests
# (`rest/4`), optionally dotted (`/4:1`). Gracenotes (`C/q`), meters (`4/4`),
# clefs, keys, bars and specials carry no numeric duration and are left intact.
_DUR_TOKEN_RE = re.compile(r"^(rest|[A-Ga-g]+(?::x)?(?:#+|-+)?)/(\d+(?::\d+)?)$")

# A *base* token (duration already stripped) that a numeric duration attaches
# to — i.e. one the duration head must predict a length for. Same left side as
# _DUR_TOKEN_RE but with no `/duration`. Used at inference to tell which
# predicted tokens carry a duration to feed back / emit.
_BASE_TOKEN_RE = re.compile(r"^(rest|[A-Ga-g]+(?::x)?(?:#+|-+)?)$")

# Every (denominator, dots) duration we are willing to emit. Analytic, not
# data-derived, so the snap target covers tuplets (/12, /24, ...) and rare
# durations regardless of what training happened to contain.
_DENOMINATORS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128)
# Dots 0-2 covers the overwhelming majority; rarer 3-/4-dot durations exist in
# the corpus and `split_duration` encodes their exact length, but the snap will
# round them to the nearest 0-2-dot value. Intentional ceiling for the pilot.
_DOTS = (0, 1, 2)


def _suffix_length(suffix: str) -> Fraction:
    """Exact length (fraction of a whole note) of a duration suffix like ``4:1``."""
    if ":" in suffix:
        denom_s, dots_s = suffix.split(":")
        denom, dots = int(denom_s), int(dots_s)
    else:
        denom, dots = int(suffix), 0
    # A dot adds half the previous value: numerator 1, 3, 7 = 2^(dots+1) - 1.
    return Fraction(1, denom) * Fraction(2 ** (dots + 1) - 1, 2**dots)


def _build_table() -> tuple[list[float], list[str]]:
    # A dotted tuplet collides exactly with a plain note one level up (`6:1` ==
    # `4`); keep the fewest-dots spelling so each length maps to one token.
    best: dict[Fraction, tuple[int, str]] = {}
    for denom in _DENOMINATORS:
        for dots in _DOTS:
            length = _suffix_length(str(denom) if dots == 0 else f"{denom}:{dots}")
            suffix = str(denom) if dots == 0 else f"{denom}:{dots}"
            if length not in best or dots < best[length][0]:
                best[length] = (dots, suffix)
    table = sorted((math.log2(float(length)), s) for length, (_, s) in best.items())
    return [x for x, _ in table], [s for _, s in table]


_TABLE_LOG2, _TABLE_SUFFIX = _build_table()

# The duration head is a classifier over these bins (one per table entry).
NUM_DUR_BINS = len(_TABLE_SUFFIX)

# Midpoints between adjacent table entries (in log2 units): the decision
# boundaries for snapping a length to its nearest bin. `length_to_bin` and the
# loss-side `torch.bucketize` both round against these, so the two never drift.
_BOUNDARIES = [0.5 * (lo + hi) for lo, hi in zip(_TABLE_LOG2, _TABLE_LOG2[1:])]


def split_duration(token: str) -> tuple[str, float | None]:
    """Split a token into ``(base_token, log2_length)``.

    ``log2_length`` is ``None`` for tokens that carry no numeric duration
    (gracenotes, meters, clefs, keys, bars, specials), which are returned
    unchanged as the base token.
    """
    if m := _DUR_TOKEN_RE.match(token):
        suffix = m.group(2)
        # recip `0` is a breve/longa (a degenerate non-1/n duration, and the most
        # common is `rest/0`); it has no place on the 1/denominator grid, so leave
        # such tokens intact as flat vocab entries — and avoid a 1/0 division.
        if int(suffix.split(":")[0]) != 0:
            return m.group(1), math.log2(float(_suffix_length(suffix)))
    return token, None


def is_duration_bearing(base_token: str) -> bool:
    """Whether a (duration-stripped) vocab token is a note/rest that carries a
    numeric duration — i.e. the duration head predicts a length for it."""
    return _BASE_TOKEN_RE.match(base_token) is not None


def length_to_bin(log2_length: float) -> int:
    """Quantise a ``log2(length)`` to its nearest bin index (``0..NUM_DUR_BINS-1``)."""
    return bisect.bisect_left(_BOUNDARIES, log2_length)


def bin_to_suffix(bin_index: int) -> str:
    """The duration suffix for a bin index."""
    return _TABLE_SUFFIX[bin_index]


def bin_boundaries() -> list[float]:
    """The ``NUM_DUR_BINS-1`` log2 midpoints, for a tensor-side ``torch.bucketize``."""
    return _BOUNDARIES


def bin_log2_lengths() -> list[float]:
    """Each bin's canonical ``log2(length)``, for the decode/feedback lookup."""
    return _TABLE_LOG2


def snap_suffix(log2_length: float) -> str:
    """Quantise a predicted ``log2(length)`` to the nearest table duration suffix."""
    return _TABLE_SUFFIX[length_to_bin(log2_length)]


def join_duration(base_token: str, log2_length: float) -> str:
    """Inverse of :func:`split_duration`: re-attach a snapped duration suffix."""
    return f"{base_token}/{snap_suffix(log2_length)}"
