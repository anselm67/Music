#!/usr/bin/env python3
"""Thin shim around :func:`kernsheet.migrate.migrate` (now also `kernsheet migrate`).

Usage:
    uv run python scripts/migrate_kernsheet_layout.py [--home DIR] [--limit N] [--write]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kernsheet.migrate import migrate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", type=Path, default=Path("/home/anselm/datasets/KernSheet"))
    ap.add_argument("--limit", type=int, default=0, help="process at most N scores")
    ap.add_argument("--write", action="store_true", help="write layout/ files (else dry-run)")
    args = ap.parse_args()

    ok, skips = migrate(args.home, write=args.write, limit=args.limit)
    print(f"\nmigrated OK: {ok}   skipped: {len(skips)}")
    by_reason: dict[str, int] = {}
    for s in skips:
        kind = s.reason.split("(")[0].split("bar ")[0].strip()
        by_reason[kind] = by_reason.get(kind, 0) + 1
    for reason, count in sorted(by_reason.items(), key=lambda x: -x[1]):
        print(f"  {count:4d}  {reason}")
    if skips:
        print("\nexamples:")
        for s in skips[:12]:
            print(f"  {s.id}: {s.reason}")


if __name__ == "__main__":
    main()
