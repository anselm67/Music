"""One-off migration: KernSheet's legacy envelope layout -> native ``Score`` JSON.

Reads each catalog entry's legacy layout (``pages[].staves[]`` where every "staff"
is a grand-staff *system* storing only ``rh_top``/``lh_bot`` + barline x-positions),
and rewrites it as a :class:`sheetmusic.Score` under ``<home>/layout/<id>.json``.

Per system we:
  * split the ``rh_top``/``lh_bot`` envelope into two ``Staff`` boxes by thirds
    (RH treble = top third, LH bass = bottom third, middle third = inter-staff gap);
  * assign absolute ``bar_numbers`` by walking systems in reading order, matching
    OMR's ``staff_editor.get_bar_offset`` (start at 0 if the kern has a pickup else 1,
    offset by ``first_bar - 1``, each system consumes ``len(bars) - 1`` measures).

Tokens are (re)generated via :func:`kern.tokenize` into ``<home>/build/tokens``;
``KernReader`` over those tokens supplies ``first_bar``/``has_bar_zero`` and validates
that every system's first bar is a real ``=N`` marker. A count mismatch flags & skips
the score (never a silent misalign).
"""

import json
from dataclasses import dataclass
from pathlib import Path

from kern import KernReader, tokenize
from sheetmusic import Box, Page, Score, Staff, Status, System


@dataclass
class Skip:
    id: str
    reason: str


class Skipped(Exception):
    """Raised to skip a score during migration, carrying a human-readable reason."""

    def __init__(self, reason: str, id: str = "") -> None:
        super().__init__(reason)
        self.reason = reason
        self.id = id


def split_envelope(rh_top: int, lh_bot: int, bars: list[int]) -> list[Staff]:
    """Thirds split of a grand-staff envelope into [treble, bass] Staff boxes."""
    delta = (lh_bot - rh_top) // 3
    left, right = bars[0], bars[-1]
    treble = Staff(box=Box(left, rh_top, right, rh_top + delta))
    bass = Staff(box=Box(left, lh_bot - delta, right, lh_bot))
    return [treble, bass]


def build_score(score_id: str, legacy: dict, kr: KernReader) -> Score:
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
                    bars=bars,
                    staves=split_envelope(staff["rh_top"], staff["lh_bot"], bars),
                )
            )
            cursor += n
        pages.append(
            Page(
                # Legacy page index is 0-based; KernSheet layouts are 1-based to
                # match PDMX and the staffer/scorer datasets (pages[page_number-1]).
                page_number=p["page_number"] + 1,
                image_width=p["image_width"],
                image_height=p["image_height"],
                systems=systems,
                status=Status.VALIDATED if p["validated"] else Status.PENDING,
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
    return Score(id=score_id, pages=pages)


def migrate_one(key: str, score: dict, home: Path, tokens_dir: Path) -> Score:
    """Migrate one catalog score to a native Score, or raise Skipped(reason)."""
    jp = score.get("json_path", "")
    if not jp and score.get("pdf_path"):
        jp = str(Path(score["pdf_path"]).with_suffix(".json"))
    # Per-score id = the layout-json stem: globally unique (authored per work+edition
    # in its work dir), so neither shared "-all" PDFs nor `-0`/`-1` editions collide.
    # The krn/tokens stay keyed by the entry `key` (shared across a work's editions).
    score_id = str(Path(jp).with_suffix("")) if jp else key
    try:
        return _build_one(score_id, key, jp, home, tokens_dir)
    except Skipped as s:
        s.id = score_id  # tag the skip with the per-edition id, not the entry key
        raise


def _build_one(score_id: str, key: str, jp: str, home: Path, tokens_dir: Path) -> Score:
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
        return build_score(score_id, legacy, kr)
    except ValueError as e:
        raise Skipped(str(e))


def migrate(home: Path, write: bool = False, limit: int = 0) -> tuple[int, list[Skip]]:
    """Migrate every catalog score; write ``layout/<id>.json`` when ``write``.

    Returns ``(ok_count, skips)``.
    """
    catalog = json.loads((home / "catalog.json").read_text())["entries"]
    tokens_dir = home / "build" / "tokens"

    ok = 0
    skips: list[Skip] = []
    for key, entry in catalog.items():
        if limit and ok + len(skips) >= limit:
            break
        for score in entry["scores"]:
            try:
                out = migrate_one(key, score, home, tokens_dir)
            except Skipped as s:
                skips.append(Skip(s.id or key, s.reason))
                continue
            if write:
                dst = home / "layout" / f"{out.id}.json"
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(json.dumps(out.asdict(), indent=2))
            ok += 1
    return ok, skips
