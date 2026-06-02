#!/usr/bin/env python3
"""Rewrite the ``id`` field of every build/layout/*.json to the home-relative mxl path.

Historically ``pdmx make`` wrote ``Score.id`` as the *absolute* mxl path (machine-
specific, and unused for resolution). The maker now emits the home-relative key
(``mxl/<...>.mxl``); this script migrates the already-built layout files in place so
they match, without an expensive full rebuild.

Safe to run while training reads the dataset: only the ``id`` field changes (which no
consumer resolves through), and writes are atomic (temp file + os.replace).

Usage:
    uv run python scripts/migrate_layout_id.py [--home DIR] [--dry-run] [--limit N]
"""

import argparse
import json
import os
from pathlib import Path


def relative_key(layout_root: Path, json_file: Path) -> str:
    rel = json_file.relative_to(layout_root).with_suffix(".mxl")
    return str(Path("mxl") / rel)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", type=Path, default=Path.home() / "datasets/PDMX")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=-1, help="Process at most N files.")
    args = ap.parse_args()

    layout_root = args.home / "build" / "layout"
    changed = already = errors = seen = 0
    for json_file in layout_root.rglob("*.json"):
        if 0 <= args.limit <= seen:
            break
        seen += 1
        key = relative_key(layout_root, json_file)
        try:
            obj = json.loads(json_file.read_text())
        except Exception as e:
            print(f"ERR  {json_file}: {e}")
            errors += 1
            continue
        if obj.get("id") == key:
            already += 1
            continue
        if args.dry_run:
            print(f"{obj.get('id')!r}  ->  {key!r}")
            changed += 1
            continue
        obj["id"] = key
        tmp = json_file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(obj, indent=2))
        os.replace(tmp, json_file)
        changed += 1

    verb = "would change" if args.dry_run else "changed"
    print(
        f"\nseen {seen:,}: {verb} {changed:,}, "
        f"already-ok {already:,}, errors {errors:,}"
    )


if __name__ == "__main__":
    main()
