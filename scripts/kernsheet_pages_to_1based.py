"""One-off: convert KernSheet layouts from 0-based to 1-based page_number.

KernSheet was migrated with 0-based page_number, but PDMX and the shared
staffer/scorer datasets are 1-based (they read pages[page_number - 1]). This bumps
every page_number by +1 in each layout/*.json, preserving all manual review, and
drops the stale 0-based PNG cache so it re-renders under the new naming.

Idempotent: a score is only bumped if it still has a page_number == 0.

Run: uv run python scripts/kernsheet_pages_to_1based.py
"""

import json
from pathlib import Path

HOME = Path("/home/anselm/datasets/KernSheet")


def main() -> None:
    layouts = sorted((HOME / "layout").rglob("*.json"))
    bumped = skipped = 0
    for path in layouts:
        data = json.loads(path.read_text())
        pages = data.get("pages", [])
        if not any(p["page_number"] == 0 for p in pages):
            skipped += 1  # already 1-based
            continue
        for p in pages:
            p["page_number"] += 1
        path.write_text(json.dumps(data, indent=2))
        bumped += 1
    print(f"layouts: {bumped} bumped to 1-based, {skipped} already 1-based")

    png_dir = HOME / "build" / "png"
    stale = list(png_dir.rglob("*.png")) if png_dir.exists() else []
    for png in stale:
        png.unlink()
    print(f"dropped {len(stale)} stale PNG(s) (will re-render 1-based on demand)")


if __name__ == "__main__":
    main()
