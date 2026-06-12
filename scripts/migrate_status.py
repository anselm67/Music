#!/usr/bin/env python3
"""One-shot: migrate on-disk layout JSON from ``validated: bool`` to the richer
``status`` enum + ``reviewed`` list (see sheetmusic.Status / kernsheet.reviews).

Per page:  ``"validated": true``  -> ``"status": "validated"``
           ``"validated": false`` -> ``"status": "pending"``
and adds ``"reviewed": []``. Idempotent (pages already carrying ``status`` are left
alone), so it is safe to re-run. Walks both the PDMX build/layout tree and the
KernSheet layout tree; pass --dry-run to count without writing.

    uv run python scripts/migrate_status.py --dry-run
    uv run python scripts/migrate_status.py
"""

import argparse
import json
from pathlib import Path

PDMX_LAYOUT = Path("/home/anselm/datasets/PDMX/build/layout")
KS_LAYOUT = Path("/home/anselm/datasets/KernSheet/layout")


def migrate_page(page: dict) -> bool:
    """Rewrite one page dict in place; return True if it changed."""
    if "status" in page:
        return False
    validated = page.pop("validated")
    page["status"] = "validated" if validated else "pending"
    page.setdefault("reviewed", [])
    return True


def migrate_file(path: Path, write: bool) -> bool | None:
    """True if changed, False if already current, None if unparseable (skipped)."""
    text = path.read_text()
    if not text.strip():
        return None  # pre-existing 0-byte artifact (failed render)
    obj = json.loads(text)
    # NB: not any(...) — that short-circuits and leaves later pages un-migrated.
    changed = False
    for page in obj["pages"]:
        if migrate_page(page):
            changed = True
    if changed and write:
        path.write_text(json.dumps(obj, indent=2))
    return changed


def migrate_tree(root: Path, write: bool) -> None:
    if not root.is_dir():
        print(f"  (skip, no such dir: {root})")
        return
    total = changed = skipped = 0
    for path in root.rglob("*.json"):
        total += 1
        result = migrate_file(path, write)
        if result is None:
            skipped += 1
        elif result:
            changed += 1
        if total % 20000 == 0:
            print(f"  ...{total:,} scanned, {changed:,} migrated")
    verb = "migrated" if write else "would migrate (dry-run)"
    print(f"  {root}: {changed:,}/{total:,} {verb}, {skipped} unparseable skipped")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdmx", type=Path, default=PDMX_LAYOUT)
    ap.add_argument("--ks", type=Path, default=KS_LAYOUT)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    write = not args.dry_run
    for root in (args.ks, args.pdmx):
        migrate_tree(root, write)


if __name__ == "__main__":
    main()
