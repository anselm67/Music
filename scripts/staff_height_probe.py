#!/usr/bin/env python3
"""Probe (dry-run) for the legacy thirds-split staff-height artifact.

The legacy->native layout migration (``scripts/migrate_kernsheet_layout.py``,
``split_envelope``) only had the grand-staff *envelope* (``rh_top``/``lh_bot``) and
split it into exact thirds: treble = top third, bass = bottom third, gap = middle
third. A real 5-line staff is shorter than a third of the envelope, so the *inner*
edges overshoot into the inter-staff gap by ~(gap - staff_height)/3 (~3-4px). The
outer edges (treble top, bass bottom) are the true envelope and are correct.

This script re-measures the actual 5 staff lines inside each staff box from the
cached page image and reports, per edge, how far the box edge sits *outside* the
outermost measured staff line. It writes NOTHING — it only quantifies the overshoot
so the corpus-wide fixer can be sanity-checked before it mutates GT.

Usage:
    uv run python scripts/staff_height_probe.py --home ~/datasets/kern-sheet-edit
    uv run python scripts/staff_height_probe.py --home ~/datasets/kern-sheet-edit \
        --prefix bach --verbose
"""

import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import find_peaks

from kernsheet import KernSheet
from sheetmusic import Box

# A staff line spans most of the box width; rows below this dark-fraction can't be one.
LINE_DARK_FRACTION = 0.4
# Allow the true line to sit a little outside a (mispositioned) box when we search.
SEARCH_PAD_PX = 6
LINES_PER_STAFF = 5


def measure_staff_edges(gray: np.ndarray, box: Box) -> tuple[int, int] | None:
    """The outermost two of the 5 staff-line rows inside ``box`` (y px in image
    frame), or None if 5 clean full-width lines can't be found."""
    top = max(0, box.top - SEARCH_PAD_PX)
    bottom = min(gray.shape[0], box.bottom + SEARCH_PAD_PX)
    left = max(0, box.left)
    right = min(gray.shape[1], box.right)
    if bottom - top < LINES_PER_STAFF or right - left < 10:
        return None
    band = gray[top:bottom, left:right]
    # Per-row count of dark (ink) pixels: a staff line is a near-full-width dark row.
    dark = (band < 128).sum(axis=1).astype(np.float64)
    width = right - left
    min_height = LINE_DARK_FRACTION * width
    # Lines are ~ box.height/4 apart; require peaks no closer than a fraction of that.
    distance = max(2, box.height // 12)
    peaks, props = find_peaks(dark, height=min_height, distance=distance)
    if len(peaks) < LINES_PER_STAFF:
        return None
    # Keep the five strongest (note beams / lyrics can add weaker full-rows). This is
    # by peak height, not geometric spacing, so a strong spurious row could in theory
    # displace a true line — the fixer's MAX_MOVE_PX guard, not this pick, is what makes
    # a mis-measure safe (it rejects the resulting out-of-range snap).
    order = np.argsort(props["peak_heights"])[::-1][:LINES_PER_STAFF]
    rows = sorted(int(peaks[i]) for i in order)
    return top + rows[0], top + rows[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--home", type=Path, required=True)
    ap.add_argument("--prefix", default="", help="restrict to keys starting with this")
    ap.add_argument(
        "--verbose", action="store_true", help="print every staff's overshoot"
    )
    args = ap.parse_args()

    ks = KernSheet(args.home)

    top_over: Counter = Counter()  # measured_top - box.top  (box extends ABOVE line)
    bot_over: Counter = Counter()  # box.bottom - measured_bottom (extends BELOW line)
    scores_touched: set[str] = set()
    n_scores = n_staves = n_measured = n_unmeasured = 0

    for key, kern_score in ks.items(args.prefix, valid=True):
        try:
            score = ks.load_score(kern_score.id)
        except Exception as e:
            print(f"  load {kern_score.id}: {e}")
            continue
        n_scores += 1
        worst = 0
        for page in score.pages:
            png = ks.png_path(kern_score.id, page.page_number)
            if not png.is_file():
                continue
            gray = cv2.imread(png.as_posix(), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            for system in page.systems:
                for staff in system.staves:
                    n_staves += 1
                    edges = measure_staff_edges(gray, staff.box)
                    if edges is None:
                        n_unmeasured += 1
                        continue
                    n_measured += 1
                    m_top, m_bot = edges
                    dt = m_top - staff.box.top
                    db = staff.box.bottom - m_bot
                    top_over[dt] += 1
                    bot_over[db] += 1
                    worst = max(worst, dt, db)
                    if args.verbose and (dt > 2 or db > 2):
                        print(
                            f"  {kern_score.id} p{page.page_number}: "
                            f"box {staff.box.top}-{staff.box.bottom} "
                            f"line {m_top}-{m_bot}  top+{dt} bot+{db}"
                        )
        if worst > 2:
            scores_touched.add(kern_score.id)

    print(
        f"\n{n_scores} scores, {n_staves} staves "
        f"({n_measured} measured, {n_unmeasured} unmeasured)"
    )
    print(f"{len(scores_touched)} score(s) with an edge overshoot > 2px\n")

    def summarise(name: str, counter: Counter) -> None:
        if not counter:
            return
        total = sum(counter.values())
        vals = sorted(counter)
        # Reconstruct a flat list only for the percentiles (cheap at this size).
        flat = [v for v in vals for _ in range(counter[v])]
        p50 = flat[len(flat) // 2]
        p95 = flat[int(len(flat) * 0.95)]
        over2 = sum(c for v, c in counter.items() if v > 2)
        print(
            f"{name}: median {p50}px  p95 {p95}px  "
            f"min {vals[0]} max {vals[-1]}  ({over2}/{total} > 2px)"
        )

    summarise("top overshoot (above first line)", top_over)
    summarise("bottom overshoot (below last line)", bot_over)


if __name__ == "__main__":
    main()
