"""One-shot migration: rescale KernSheet validated layouts to the 2× canvas.

The staffer/noter canvas doubled (patch_size 16→32 / crop 64→128 at held patch
grid), so ``build/png`` is re-rendered at higher resolution: PDMX via
``svg_to_png_command --width 2048`` (no layout change — its boxes are SVG-native
2100-space, aspect-invariant), and KernSheet via ``RENDER_DPI`` 200→300 with the
page baked at ``image_width`` (``rebuild_images`` → ``_transform``). KS boxes live
in ``image_width``-pixel space, so they must be rescaled alongside the render.

This normalises every KS page to ``TARGET_WIDTH`` px (aspect-preserved height)
via the authoritative ``Page.resize`` (which scales bars + staff top/bottom). It
is **idempotent**: a page already at ``TARGET_WIDTH`` resizes by factor 1.0 (a
no-op), so re-running is safe. Lossless (integer ×2 from the uniform 1200) and
git-backed (KernSheet is a git repo). PDMX needs no migration.

Mirrors scripts/migrate_staff_box_x.py. Run --dry-run first to see the counts.
"""

import argparse
import json
from pathlib import Path

from sheetmusic import Score

# The 1200-wide KS render doubles to fill the 2× staffer canvas (1376px wide,
# ~2400 gives the same headroom the 1200 render had over the old 688 canvas).
TARGET_WIDTH = 2400

DEFAULT_ROOT = Path.home() / "datasets/KernSheet/layout"


def migrate_score(score: Score) -> Score:
    """Return a copy with every page normalised to ``TARGET_WIDTH`` (aspect kept)."""
    pages = []
    for page in score.pages:
        new_height = round(page.image_height * TARGET_WIDTH / page.image_width)
        pages.append(page.resize(TARGET_WIDTH, new_height))
    return Score(score.id, pages)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--dry-run", action="store_true", help="report counts without writing"
    )
    args = parser.parse_args()

    if not args.root.exists():
        print(f"missing: {args.root}")
        return

    files = changed = skipped = 0
    for path in sorted(args.root.rglob("*.json")):
        files += 1
        try:
            score = Score.from_json(json.loads(path.read_text()))
        except Exception:
            skipped += 1  # empty / failed build / malformed, leave untouched
            continue
        if all(p.image_width == TARGET_WIDTH for p in score.pages):
            continue  # already migrated
        changed += 1
        if not args.dry_run:
            path.write_text(json.dumps(migrate_score(score).asdict(), indent=2))

    verb = "would rescale" if args.dry_run else "rescaled"
    print(
        f"scanned {files} layouts; {verb} {changed} to {TARGET_WIDTH}px wide; "
        f"skipped {skipped} unparseable"
    )


if __name__ == "__main__":
    main()
