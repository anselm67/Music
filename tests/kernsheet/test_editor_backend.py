"""Unit tests for the editor backend's native Score <-> envelope conversion."""

import json
from pathlib import Path

from kernsheet.editor_backend import (
    EditorBackend,
    EditorPage,
    EditorStaff,
    _split_envelope,
    _to_editor_page,
)
from sheetmusic import Box, Page, Staff, System


def _native_page() -> Page:
    treble = Staff(box=Box((100, 10), (900, 40)), bars=[100, 500, 900])
    bass = Staff(box=Box((100, 60), (900, 90)), bars=[100, 500, 900])
    return Page(
        page_number=0,
        image_width=1000,
        image_height=200,
        systems=[System(bar_numbers=[1, 2], staves=[treble, bass])],
        validated=True,
    )


def test_to_editor_page_flattens_system_to_envelope() -> None:
    ep = _to_editor_page(_native_page())
    assert len(ep.staves) == 1
    es = ep.staves[0]
    assert (es.rh_top, es.lh_bot, es.bars) == (10, 90, [100, 500, 900])
    assert ep.validated and ep.image_width == 1000


def test_split_envelope_thirds_and_empty_bars() -> None:
    treble, bass = _split_envelope(EditorStaff(0, 90, [100, 900]), image_width=1000)
    assert treble.box.top == 0 and treble.box.bottom == 30  # delta = 90 // 3
    assert bass.box.top == 60 and bass.box.bottom == 90
    assert treble.box.left == 100 and treble.box.right == 900

    # No bars yet -> span the full page width.
    t, _ = _split_envelope(EditorStaff(0, 90, []), image_width=1000)
    assert t.box.left == 0 and t.box.right == 1000


def _make_home(tmp_path: Path) -> tuple[Path, str]:
    (tmp_path / "catalog.json").write_text(
        json.dumps(
            {
                "entries": {
                    "a/b/work": {
                        "scores": [
                            {"pdf_path": "a/b/work.pdf", "json_path": "a/b/work.json"}
                        ]
                    }
                }
            }
        )
    )
    tokens = tmp_path / "build" / "tokens" / "a/b/work.tokens"
    tokens.parent.mkdir(parents=True, exist_ok=True)
    tokens.write_text(
        "clef-G\tclef-F\n=1\t=1\nC/4\tC/4\n=2\t=2\nC/4\tC/4\n"
        "=3\t=3\nC/4\tC/4\n=4\t=4\nC/4\tC/4\n=5\t=5\n"
    )
    return tmp_path, "a/b/work"


def test_to_score_recomputes_bar_numbers_by_walk(tmp_path: Path) -> None:
    home, id = _make_home(tmp_path)
    backend = EditorBackend(home, id)
    # Two systems of 2 bars each (3 barline x-positions -> 2 segments).
    pages = (
        EditorPage(
            page_number=0,
            image_width=1000,
            image_height=200,
            staves=[
                EditorStaff(10, 90, [100, 500, 900]),
                EditorStaff(110, 190, [100, 500, 900]),
            ],
            validated=True,
        ),
    )
    score = backend._to_score(pages)
    systems = score.pages[0].systems
    assert [s.bar_numbers for s in systems] == [[1, 2], [3, 4]]
    # Each system carries the thirds-split grand staff.
    assert all(s.staff_count == 2 for s in systems)


def test_to_score_empty_system_contributes_no_bars(tmp_path: Path) -> None:
    home, id = _make_home(tmp_path)
    backend = EditorBackend(home, id)
    pages = (
        EditorPage(
            page_number=0,
            image_width=1000,
            image_height=200,
            staves=[
                EditorStaff(10, 90, [100, 500, 900]),  # 2 bars -> [1, 2]
                EditorStaff(110, 190, []),  # mid-edit, no bars -> [] (no undercount)
                EditorStaff(210, 290, [100, 900]),  # 1 bar -> [3]
            ],
            validated=True,
        ),
    )
    score = backend._to_score(pages)
    assert [s.bar_numbers for s in score.pages[0].systems] == [[1, 2], [], [3]]
