"""One-shot migration: drop the redundant per-staff x from on-disk layouts.

A staff used to store a full ltrb box whose ``left``/``right`` duplicated the
system's ``bars`` (``bars[0]``/``bars[-1]``). Staff x is now derived from the
system's bars on load, so the staff owns only its vertical band.

Before: ``"staves": [{"box": {"left": l, "top": t, "right": r, "bottom": b}}]``
After:  ``"staves": [{"top": t, "bottom": b}]``

The ~2% of KernSheet staves whose stored x had drifted from ``bars`` are
corrected by this drop (bars is the source of truth); PDMX is a no-op field-drop
(its staff x already equalled bars exactly). Idempotent: already-flat staves are
left untouched. Mirrors the prior scripts/migrate_box_ltrb.py pattern.
"""

import argparse
import json
from pathlib import Path

DEFAULT_ROOTS = [
    Path.home() / "datasets/PDMX/build/layout",
    Path.home() / "datasets/KernSheet/layout",
]


def migrate_staff(staff: dict) -> bool:
    """Flatten one staff dict in place; return True if it was changed."""
    box = staff.get("box")
    if box is None or "top" not in box or "bottom" not in box:
        return False  # already flat, or unrecognised
    staff.clear()
    staff["top"] = box["top"]
    staff["bottom"] = box["bottom"]
    return True


def migrate_score(score: dict) -> int:
    """Migrate every staff in a score dict in place; return staves converted."""
    converted = 0
    for page in score.get("pages", []):
        for system in page.get("systems", []):
            for staff in system.get("staves", []):
                converted += migrate_staff(staff)
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="*", type=Path, default=DEFAULT_ROOTS)
    parser.add_argument(
        "--dry-run", action="store_true", help="report counts without writing"
    )
    args = parser.parse_args()

    files = changed = staves = skipped = 0
    for root in args.roots:
        if not root.exists():
            print(f"skip (missing): {root}")
            continue
        for path in root.rglob("*.json"):
            files += 1
            try:
                score = json.loads(path.read_text())
            except (json.JSONDecodeError, UnicodeDecodeError):
                skipped += 1  # empty / failed build, leave untouched
                continue
            converted = migrate_score(score)
            if converted:
                changed += 1
                staves += converted
                if not args.dry_run:
                    path.write_text(json.dumps(score, indent=2))
            if files % 20000 == 0:
                print(f"  ...{files} files scanned, {changed} changed")

    verb = "would change" if args.dry_run else "changed"
    print(
        f"scanned {files} files; {verb} {changed} files, {staves} staves; "
        f"skipped {skipped} unparseable"
    )


if __name__ == "__main__":
    main()
