"""Migrate layout JSON: move bars[] from Staff up to System.

Before this migration each staff in a system carried its own ``bars`` list of
bar-line x-coordinates.  Since all staves in a system share the same bars, the
list is now stored once on the System instead.

Old format (per system):
    { "bar_numbers": [...], "staves": [{"box": ..., "bars": [x, ...]}, ...] }

New format:
    { "bar_numbers": [...], "bars": [x, ...], "staves": [{"box": ...}, ...] }

Usage:
    uv run python scripts/migrate_bars_to_system.py          # dry-run
    uv run python scripts/migrate_bars_to_system.py --write  # write in-place

Datasets migrated:
    /home/anselm/datasets/PDMX/build/layout   (~253 k files)
    /home/anselm/datasets/KernSheet/layout    (~581 files)
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def _migrate_system(system: dict) -> tuple[dict, bool]:
    """Return (migrated_system, changed).

    Already-migrated systems (top-level 'bars' key present, no 'bars' in
    staves) are returned unchanged with changed=False.
    """
    staves = system.get("staves", [])

    # Detect already-migrated: system has 'bars' and no staff has 'bars'.
    if "bars" in system and not any("bars" in s for s in staves):
        return system, False

    # Collect per-staff bar lists.
    staff_bars: list[list[int]] = [s.get("bars", []) for s in staves]

    # Warn on inconsistency (take first staff's bars as canonical).
    if staff_bars and not all(b == staff_bars[0] for b in staff_bars[1:]):
        print(
            f"  WARNING: staves have inconsistent bars; using staves[0]",
            file=sys.stderr,
        )

    system_bars: list[int] = staff_bars[0] if staff_bars else []

    new_system = {
        "bar_numbers": system.get("bar_numbers", []),
        "bars": system_bars,
        "staves": [
            {k: v for k, v in s.items() if k != "bars"} for s in staves
        ],
    }
    # Preserve any extra top-level keys (e.g. svg_bar_numbers).
    for k, v in system.items():
        if k not in ("bar_numbers", "bars", "staves"):
            new_system[k] = v

    return new_system, True


def migrate_file(path: Path) -> tuple[Path, int]:
    """Migrate one JSON file in-place.  Returns (path, systems_changed)."""
    raw = path.read_text()
    obj = json.loads(raw)

    changed_total = 0
    for page in obj.get("pages", []):
        new_systems = []
        for system in page.get("systems", []):
            new_system, changed = _migrate_system(system)
            new_systems.append(new_system)
            if changed:
                changed_total += 1
        page["systems"] = new_systems

    if changed_total:
        path.write_text(json.dumps(obj))

    return path, changed_total


def _migrate_file_worker(path_str: str) -> tuple[str, int]:
    path = Path(path_str)
    try:
        _, n = migrate_file(path)
        return str(path), n
    except Exception as e:
        return str(path), -1  # sentinel for errors


def run(roots: list[Path], write: bool) -> None:
    files: list[Path] = []
    for root in roots:
        files.extend(root.rglob("*.json"))

    print(f"Found {len(files):,} JSON files across {len(roots)} root(s).")

    if not write:
        # Dry-run: just show a sample.
        sample = files[:3]
        for p in sample:
            raw = json.loads(p.read_text())
            page = raw["pages"][0] if raw.get("pages") else {}
            system = page["systems"][0] if page.get("systems") else {}
            staves = system.get("staves", [])
            already = "bars" in system and not any("bars" in s for s in staves)
            migrated, _ = _migrate_system(system)
            print(f"\n--- {p} ({'already migrated' if already else 'needs migration'})")
            print("  Before:", json.dumps(system)[:200])
            print("  After: ", json.dumps(migrated)[:200])
        print(f"\nDry-run complete.  Pass --write to apply to all {len(files):,} files.")
        return

    errors = 0
    changed_files = 0
    changed_systems = 0

    with ProcessPoolExecutor() as executor:
        futs = {
            executor.submit(_migrate_file_worker, str(p)): p for p in files
        }
        done = 0
        for fut in as_completed(futs):
            _, n = fut.result()
            done += 1
            if n < 0:
                errors += 1
            elif n > 0:
                changed_files += 1
                changed_systems += n
            if done % 10_000 == 0:
                print(f"  {done:,}/{len(files):,} processed …")

    print(
        f"\nDone.  {changed_files:,} files updated "
        f"({changed_systems:,} systems migrated), {errors} errors."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write changes in-place (default: dry-run).",
    )
    parser.add_argument(
        "--pdmx",
        default="/home/anselm/datasets/PDMX/build/layout",
        metavar="DIR",
        help="PDMX layout root.",
    )
    parser.add_argument(
        "--kernsheet",
        default="/home/anselm/datasets/KernSheet/layout",
        metavar="DIR",
        help="KernSheet layout root.",
    )
    args = parser.parse_args()

    roots = [Path(args.pdmx), Path(args.kernsheet)]
    for r in roots:
        if not r.exists():
            print(f"WARNING: {r} does not exist, skipping.", file=sys.stderr)
    roots = [r for r in roots if r.exists()]

    if not roots:
        print("No dataset roots found, nothing to do.", file=sys.stderr)
        sys.exit(1)

    run(roots, write=args.write)


if __name__ == "__main__":
    main()
