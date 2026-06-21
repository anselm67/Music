#!/usr/bin/env python3
"""Fix the legacy thirds-split staff-height artifact across the corpus.

Background and the measurement are in ``scripts/staff_height_probe.py``: the
legacy->native migration split each grand-staff envelope into exact thirds, so the
*inner* (gap-facing) edge of every staff box overshoots the real outermost staff
line by ~(gap - staff_height)/3. The outer edge is the true envelope and is correct.

This re-measures the 5 staff lines inside each box and snaps only the **inner** edge
in to the measured outermost line. It is conservative by construction:
  * the trusted outer edge is never touched;
  * the inner edge only ever moves *inward* (the box only shrinks);
  * a staff whose lines can't be measured cleanly is left untouched;
  * a move beyond MAX_MOVE_PX is treated as a mis-measure and skipped.

Dry-run by default; pass --write to save layout JSON (in the given --home only, so
git there is your rollback). --worklist writes the changed score ids, one per line,
for ``kernsheet edit --worklist <file>`` to review.

Usage:
    uv run python scripts/staff_height_fix.py --home ~/datasets/kern-sheet-edit
    uv run python scripts/staff_height_fix.py --home ~/datasets/kern-sheet-edit \
        --write --worklist /tmp/staff_height_fixed.txt
"""

import argparse
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from kernsheet import KernSheet
from sheetmusic import Box, Staff
from staff_height_probe import measure_staff_edges

# Only correct an overshoot of at least this many px (below it is annotation noise).
MIN_OVERSHOOT_PX = 3
# Reject a snap that would move an edge further than this — a likely mis-measure.
MAX_MOVE_PX = 20


def fix_staff(
    gray: np.ndarray, staff: Staff, fix_top: bool, fix_bottom: bool
) -> Staff | None:
    """A staff with its inner edge(s) snapped to the measured staff lines, or None
    if nothing was changed (unmeasurable, within noise, or beyond MAX_MOVE)."""
    edges = measure_staff_edges(gray, staff.box)
    if edges is None:
        return None
    m_top, m_bot = edges
    box = staff.box
    new_top, new_bottom = box.top, box.bottom
    # Inner top edge bleeds *up* into the gap: pull it down to the first line.
    if fix_top:
        over = m_top - box.top  # box extends above the measured first line
        if MIN_OVERSHOOT_PX <= over <= MAX_MOVE_PX:
            new_top = m_top
    # Inner bottom edge bleeds *down* into the gap: pull it up to the last line.
    if fix_bottom:
        over = box.bottom - m_bot  # box extends below the measured last line
        if MIN_OVERSHOOT_PX <= over <= MAX_MOVE_PX:
            new_bottom = m_bot
    if new_top == box.top and new_bottom == box.bottom:
        return None
    return replace(staff, box=Box(box.left, new_top, box.right, new_bottom))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--home", type=Path, required=True)
    ap.add_argument("--prefix", default="", help="restrict to keys starting with this")
    ap.add_argument("--write", action="store_true", help="save (default: dry-run)")
    ap.add_argument("--worklist", type=Path, help="write changed score ids here")
    args = ap.parse_args()

    ks = KernSheet(args.home)
    changed_ids: list[str] = []
    n_scores = n_changed_scores = n_staves_fixed = 0

    for _, kern_score in ks.items(args.prefix, valid=True):
        try:
            score = ks.load_score(kern_score.id)
        except Exception as e:
            print(f"  load {kern_score.id}: {e}")
            continue
        n_scores += 1
        score_changed = False
        for page in score.pages:
            png = ks.png_path(kern_score.id, page.page_number)
            if not png.is_file():
                continue
            gray = cv2.imread(png.as_posix(), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            for system in page.systems:
                last = len(system.staves) - 1
                for i, staff in enumerate(system.staves):
                    # An edge is "inner" (gap-facing, possibly inflated) when another
                    # staff sits on that side within the system.
                    fixed = fix_staff(gray, staff, fix_top=i > 0, fix_bottom=i < last)
                    if fixed is not None:
                        system.staves[i] = fixed
                        n_staves_fixed += 1
                        score_changed = True
        if score_changed:
            n_changed_scores += 1
            changed_ids.append(kern_score.id)
            if args.write:
                ks.save_score(kern_score.id, score)

    verb = "fixed" if args.write else "would fix (dry-run; pass --write)"
    print(
        f"\n{n_scores} scores scanned; {verb}: "
        f"{n_staves_fixed} staves across {n_changed_scores} score(s)."
    )
    if args.worklist and changed_ids:
        args.worklist.write_text("\n".join(changed_ids) + "\n")
        print(f"wrote {len(changed_ids)} score id(s) to {args.worklist}")


if __name__ == "__main__":
    main()
