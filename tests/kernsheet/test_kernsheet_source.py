"""Unit tests for KernSheetSource — the KernSheet-backed Source implementation."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from kernsheet import KernSheet, KernSheetSource
from sheetmusic import Box, Page, Score, Source, Staff, System


def _score(json_path: str, pdf_path: str, id: str | None = None) -> dict[str, str]:
    score = {"pdf_path": pdf_path, "pdf_url": "", "json_path": json_path}
    if id is not None:
        score["id"] = id
    return score


def _entry(*scores: dict[str, str]) -> dict[str, object]:
    return {"source": "", "imslp_query": "", "imslp_url": "", "scores": list(scores)}


def _catalog(tmp_path: Path, entries: dict) -> None:
    (tmp_path / "catalog.json").write_text(json.dumps({"entries": entries}))


def _page(page_number: int, width: int = 40, validated: bool = True) -> Page:
    return Page(
        page_number=page_number,
        image_width=width,
        image_height=width,
        systems=[
            System(
                bar_numbers=[1],
                bars=[0, 10],
                staves=[Staff(box=Box((0, 0), (10, 10)))],
            )
        ],
        validated=validated,
    )


def _write_layout(tmp_path: Path, id: str, pages: list[Page]) -> None:
    dst = tmp_path / "layout" / f"{id}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(Score(id=id, pages=pages).asdict()))


def test_kernsheet_source_satisfies_protocol(tmp_path: Path) -> None:
    _catalog(tmp_path, {})
    src: Source = KernSheetSource(KernSheet(tmp_path))
    assert src is not None  # structural typing checked by mypy


def test_id_is_json_stem_and_scores_filtered_by_layout(tmp_path: Path) -> None:
    # Two editions of one work -> two json-stem ids sharing the single entry key.
    # The catalog's own `id` field (a stale `#N` encoding) is ignored in favour of
    # the json stem, so every consumer keys the same way Score.id does.
    _catalog(
        tmp_path,
        {
            "a/b/work": _entry(
                _score("a/b/work-0.json", "a/b/work-0.pdf", id="a/b/work#1"),
                _score("a/b/work-1.json", "a/b/work-1.pdf", id="a/b/work#2"),
            )
        },
    )
    _write_layout(tmp_path, "a/b/work-0", [_page(1)])  # only edition 0 migrated
    ks = KernSheet(tmp_path)
    src = KernSheetSource(ks)

    assert set(ks.id2key) == {"a/b/work-0", "a/b/work-1"}
    assert set(ks.id2key.values()) == {"a/b/work"}  # both editions -> entry key
    # scores() yields only editions whose layout exists on disk.
    assert [s.id for s in src.scores()] == ["a/b/work-0"]
    assert src.score("a/b/work-0").id == "a/b/work-0"


def test_scores_skips_unmigrated_empty_json_path(tmp_path: Path) -> None:
    # An unmigrated score carries an empty json_path; its layout_path resolves to
    # the layout/ directory itself, which exists() — scores() must still skip it
    # (it guards on is_file, else load_score would read_text a directory).
    _catalog(
        tmp_path,
        {
            "a/b/work": _entry(_score("a/b/work.json", "a/b/work.pdf")),
            "c/d/none": _entry(_score("", "c/d/none.pdf")),
        },
    )
    _write_layout(tmp_path, "a/b/work", [_page(1)])  # also creates the layout/ dir
    src = KernSheetSource(KernSheet(tmp_path))

    assert [s.id for s in src.scores()] == ["a/b/work"]


def test_pages_filters_unvalidated(tmp_path: Path) -> None:
    # scores() yields every migrated score; pages() is the training/eval feed and
    # drops un-validated pages (kernsheet detect writes them validated=False) while
    # keeping the valid pages of a partly-reviewed score.
    _catalog(
        tmp_path,
        {
            "a/done": _entry(_score("a/done.json", "a/done.pdf")),
            "a/fresh": _entry(_score("a/fresh.json", "a/fresh.pdf")),
            "a/mixed": _entry(_score("a/mixed.json", "a/mixed.pdf")),
        },
    )
    _write_layout(tmp_path, "a/done", [_page(1), _page(2)])  # all validated
    _write_layout(tmp_path, "a/fresh", [_page(1, validated=False)])  # detect output
    _write_layout(tmp_path, "a/mixed", [_page(1), _page(2, validated=False)])
    src = KernSheetSource(KernSheet(tmp_path))

    assert {s.id for s in src.scores()} == {"a/done", "a/fresh", "a/mixed"}
    assert [p.page_number for p in src.pages("a/done")] == [1, 2]
    assert src.pages("a/fresh") == []
    assert [p.page_number for p in src.pages("a/mixed")] == [1]


def test_records_keyed_by_entry_not_edition(tmp_path: Path) -> None:
    # Editions share one .krn/tokens file, keyed by the entry key (not the score id).
    _catalog(
        tmp_path,
        {"a/b/work": _entry(_score("a/b/work-1.json", "a/b/work-1.pdf"))},
    )
    src = KernSheetSource(KernSheet(tmp_path))

    with patch("kernsheet.kernsheet.KernReader") as reader_cls:
        reader_cls.return_value = MagicMock(
            get_text=MagicMock(return_value=["=1", "C/4\tC/4"])
        )
        out = src.records("a/b/work-1", 1, 2)

    tokens = tmp_path / "build" / "tokens" / "a/b/work.tokens"
    reader_cls.assert_called_once_with(tokens)
    assert out == ["=1", "C/4\tC/4"]


def test_make_renders_pages_into_annotation_space(tmp_path: Path) -> None:
    _catalog(
        tmp_path,
        {"a/b/work": _entry(_score("a/b/work.json", "a/b/work.pdf"))},
    )
    _write_layout(tmp_path, "a/b/work", [_page(1, width=40)])
    # A pre-existing tokens file makes make() skip tokenization and only render pngs.
    tokens = tmp_path / "build" / "tokens" / "a/b/work.tokens"
    tokens.parent.mkdir(parents=True, exist_ok=True)
    tokens.write_text("")
    ks = KernSheet(tmp_path)

    raw = Image.fromarray(
        np.zeros((50, 60, 3), dtype=np.uint8)
    )  # raw render (h=50, w=60)
    with patch("kernsheet.kernsheet.convert_from_path", return_value=[raw]) as conv:
        ks.make()

    conv.assert_called_once_with(tmp_path / "a/b/work.pdf")
    assert (tmp_path / "build" / "png" / "a/b/work-001.png").exists()

    # image() just decodes the cached png; width normalised to image_width=40,
    # height scaled by 40/60.
    img = KernSheetSource(ks).image("a/b/work", 1)
    assert img.shape == (3, int(50 * 40 / 60), 40)
