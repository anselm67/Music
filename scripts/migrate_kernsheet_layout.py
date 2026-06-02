#!/usr/bin/env python3
"""One-off migration: KernSheet's legacy envelope layout -> native ``Score`` JSON.

Reads each catalog entry's legacy layout (``pages[].staves[]`` where every "staff"
is a grand-staff *system* storing only ``rh_top``/``lh_bot`` + barline x-positions),
and rewrites it as a ``sheetmusic.Score`` under ``<home>/layout/<key>.json``.

Per system we:
  * split the ``rh_top``/``lh_bot`` envelope into two ``Staff`` boxes by thirds
    (RH treble = top third, LH bass = bottom third, middle third = inter-staff gap);
  * assign absolute ``bar_numbers`` by walking systems in reading order, matching
    OMR's ``staff_editor.get_bar_offset`` (start at 0 if the kern has a pickup else 1,
    offset by ``first_bar - 1``, each system consumes ``len(bars) - 1`` measures).

Tokens are (re)generated via Music's ``kern.tokenize`` into ``<home>/build/tokens``;
``KernReader`` over those tokens supplies ``first_bar``/``has_bar_zero`` and validates
that every system's first bar is a real ``=N`` marker. A count mismatch flags & skips
the score (never a silent misalign).

Usage:
    uv run python scripts/migrate_kernsheet_layout.py [--home DIR] [--limit N] [--write]
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kern import KernReader, tokenize  # noqa: E402
from sheetmusic import Box, Page, Score, Staff, System  # noqa: E402


@dataclass
class Skip:
    key: str
    reason: str


class Skipped(Exception):
    """Raised to skip a score during migration, carrying a human-readable reason."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def split_envelope(rh_top: int, lh_bot: int, bars: list[int]) -> list[Staff]:
    """Thirds split of a grand-staff envelope into [treble, bass] Staff boxes."""
    delta = (lh_bot - rh_top) // 3
    left, right = bars[0], bars[-1]
    treble = Staff(box=Box((left, rh_top), (right, rh_top + delta)), bars=bars)
    bass = Staff(box=Box((left, lh_bot - delta), (right, lh_bot)), bars=bars)
    return [treble, bass]


def build_score(key: str, legacy: dict, kr: KernReader) -> Score:
    """Build a native Score; raises ValueError on a bar-count/alignment mismatch."""
    cursor = (0 if kr.has_bar_zero() else 1) + (kr.first_bar - 1)
    pages: list[Page] = []
    for p in legacy["pages"]:
        systems: list[System] = []
        for staff in p["staves"]:
            bars = staff["bars"]
            n = len(bars) - 1
            if n < 1:
                raise ValueError(f"system with <1 measure (bars={bars})")
            if cursor not in kr.bars:
                raise ValueError(f"bar {cursor} absent from kern =N markers")
            bar_numbers = list(range(cursor, cursor + n))
            systems.append(
                System(
                    bar_numbers=bar_numbers,
                    staves=split_envelope(staff["rh_top"], staff["lh_bot"], bars),
                )
            )
            cursor += n
        pages.append(
            Page(
                page_number=p["page_number"],
                image_width=p["image_width"],
                image_height=p["image_height"],
                systems=systems,
                validated=p["validated"],
                image_rotation=p.get("image_rotation", 0.0),
            )
        )
    # Count-match guard: the walk must land on the kern's last barline. The final
    # barline may be numbered (cursor == last marker) or bare `==` (cursor - 1).
    last_marker = max(kr.bars)
    if last_marker not in (cursor - 1, cursor):
        raise ValueError(
            f"annotated measures end at bar {cursor - 1}, kern last =N is {last_marker}"
        )
    return Score(id=key, pages=pages)


def migrate_one(key: str, score: dict, home: Path, tokens_dir: Path) -> Score:
    """Migrate one catalog score to a native Score, or raise Skipped(reason)."""
    jp = score.get("json_path", "")
    if not jp and score.get("pdf_path"):
        jp = str(Path(score["pdf_path"]).with_suffix(".json"))
    legacy_path = home / jp if jp else None
    if legacy_path is None or not legacy_path.exists():
        raise Skipped("no layout json")
    legacy = json.loads(legacy_path.read_text())
    if not all(p.get("validated") for p in legacy["pages"]):
        raise Skipped("has non-validated page")

    tokens_path = tokens_dir / f"{key}.tokens"
    if not tokens_path.exists():
        tokens_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            tokenize((home / key).with_suffix(".krn"), tokens_path)
        except (SyntaxError, ValueError, AssertionError) as e:
            tokens_path.unlink(missing_ok=True)
            raise Skipped(f"tokenize failed: {type(e).__name__}")
    try:
        kr = KernReader(tokens_path)
    except AssertionError:
        raise Skipped("tokens spine-count mismatch")
    if kr.first_bar < 0:
        raise Skipped("no numbered bars in kern")

    try:
        return build_score(key, legacy, kr)
    except ValueError as e:
        raise Skipped(str(e))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", type=Path, default=Path("/home/anselm/datasets/KernSheet"))
    ap.add_argument("--limit", type=int, default=0, help="process at most N scores")
    ap.add_argument("--write", action="store_true", help="write layout/ files (else dry-run)")
    args = ap.parse_args()

    catalog = json.loads((args.home / "catalog.json").read_text())["entries"]
    tokens_dir = args.home / "build" / "tokens"

    ok = 0
    skips: list[Skip] = []
    for key, entry in catalog.items():
        if args.limit and ok + len(skips) >= args.limit:
            break
        for score in entry["scores"]:
            try:
                out = migrate_one(key, score, args.home, tokens_dir)
            except Skipped as s:
                skips.append(Skip(key, s.reason))
                continue

            if args.write:
                dst = args.home / "layout" / f"{key}.json"
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(json.dumps(out.asdict(), indent=2))
            ok += 1

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
            print(f"  {s.key}: {s.reason}")


if __name__ == "__main__":
    main()
