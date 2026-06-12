"""One-shot migration: rewrite on-disk layout boxes to flat ltrb fields.

Before: each staff box is stored as ``{"top_left": [l, t], "bot_right": [r, b]}``.
After:  ``{"left": l, "top": t, "right": r, "bottom": b}``.

This matches the in-memory ``Box`` after the ltrb refactor (commit 53ea6e2) and
lets the ``_box_to_disk`` / ``_box_from_disk`` shim in ``sheetmusic.layout`` be
deleted. Operates directly on the JSON (independent of that shim), is idempotent
(boxes already in flat form are left untouched), and re-dumps with ``indent=2`` so
only the box keys change.

Usage::

    uv run python scripts/migrate_box_ltrb.py [--dry-run] [ROOT ...]

ROOT defaults to the PDMX and KernSheet layout directories.
"""

import argparse
import json
from pathlib import Path

DEFAULT_ROOTS = [
    Path.home() / "datasets/PDMX/build/layout",
    Path.home() / "datasets/KernSheet/layout",
]


def migrate_box(box: dict) -> dict | None:
    """Return a flat-ltrb box dict, or None if already migrated / unrecognised."""
    if "top_left" not in box or "bot_right" not in box:
        return None
    left, top = box["top_left"]
    right, bottom = box["bot_right"]
    return {"left": left, "top": top, "right": right, "bottom": bottom}


def migrate_score(score: dict) -> int:
    """Migrate every staff box in a score dict in place; return boxes converted."""
    converted = 0
    for page in score.get("pages", []):
        for system in page.get("systems", []):
            for staff in system.get("staves", []):
                new_box = migrate_box(staff.get("box", {}))
                if new_box is not None:
                    staff["box"] = new_box
                    converted += 1
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="*", type=Path, default=DEFAULT_ROOTS)
    parser.add_argument(
        "--dry-run", action="store_true", help="report counts without writing"
    )
    args = parser.parse_args()

    files = 0
    changed = 0
    boxes = 0
    skipped = 0
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
                boxes += converted
                if not args.dry_run:
                    path.write_text(json.dumps(score, indent=2))
            if files % 20000 == 0:
                print(f"  ...{files} files scanned, {changed} changed")

    verb = "would change" if args.dry_run else "changed"
    print(
        f"scanned {files} files; {verb} {changed} files, {boxes} boxes; "
        f"skipped {skipped} unparseable"
    )


if __name__ == "__main__":
    main()
