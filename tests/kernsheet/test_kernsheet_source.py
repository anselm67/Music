"""Unit tests for KernSheetSource — the KernSheet-backed Source implementation."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from kernsheet import KernSheetSource
from sheetmusic import Box, Page, Score, Source, Staff, System


def _catalog(tmp_path: Path, entries: dict) -> None:
    (tmp_path / "catalog.json").write_text(json.dumps({"entries": entries}))


def _page(page_number: int, width: int = 40) -> Page:
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
        validated=True,
    )


def _write_layout(tmp_path: Path, id: str, pages: list[Page]) -> None:
    dst = tmp_path / "layout" / f"{id}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(Score(id=id, pages=pages).asdict()))


def test_kernsheet_source_satisfies_protocol(tmp_path: Path) -> None:
    _catalog(tmp_path, {})
    src: Source = KernSheetSource(tmp_path)
    assert src is not None  # structural typing checked by mypy


def test_id_is_json_stem_and_scores_filtered_by_layout(tmp_path: Path) -> None:
    # One work, two editions -> two distinct ids; only the migrated one is yielded.
    _catalog(
        tmp_path,
        {
            "a/b/work": {
                "scores": [
                    {"pdf_path": "a/b/work-0.pdf", "json_path": "a/b/work-0.json"},
                    {"pdf_path": "a/b/work-1.pdf", "json_path": "a/b/work-1.json"},
                ]
            }
        },
    )
    _write_layout(tmp_path, "a/b/work-0", [_page(0)])  # only edition 0 migrated
    src = KernSheetSource(tmp_path)

    assert set(src._key) == {"a/b/work-0", "a/b/work-1"}
    assert [s.id for s in src.scores()] == ["a/b/work-0"]
    assert src.score("a/b/work-0").id == "a/b/work-0"


def test_records_keyed_by_entry_not_edition(tmp_path: Path) -> None:
    # Editions share one .krn/tokens file, keyed by the entry key (not the score id).
    _catalog(
        tmp_path,
        {
            "a/b/work": {
                "scores": [
                    {"pdf_path": "a/b/work-1.pdf", "json_path": "a/b/work-1.json"},
                ]
            }
        },
    )
    src = KernSheetSource(tmp_path)

    with patch("kernsheet.kernsheet_source.KernReader") as reader_cls:
        reader_cls.return_value = MagicMock(
            get_text=MagicMock(return_value=["=1", "C/4\tC/4"])
        )
        out = src.records("a/b/work-1", 1, 2)

    tokens = tmp_path / "build" / "tokens" / "a/b/work.tokens"
    reader_cls.assert_called_once_with(tokens)
    assert out == ["=1", "C/4\tC/4"]


def test_image_renders_into_annotation_space_and_caches(tmp_path: Path) -> None:
    _catalog(
        tmp_path,
        {
            "a/b/work": {
                "scores": [{"pdf_path": "a/b/work.pdf", "json_path": "a/b/work.json"}]
            }
        },
    )
    _write_layout(tmp_path, "a/b/work", [_page(0, width=40)])
    src = KernSheetSource(tmp_path)

    raw = Image.fromarray(
        np.zeros((50, 60, 3), dtype=np.uint8)
    )  # raw render (h=50, w=60)
    with patch("pdf2image.convert_from_path", return_value=[raw]) as conv:
        img = src.image("a/b/work", 0)

    conv.assert_called_once_with(tmp_path / "a/b/work.pdf")
    # Width normalised to image_width=40, height scaled by 40/60.
    assert img.shape == (3, int(50 * 40 / 60), 40)
    assert (tmp_path / "build" / "png" / "a/b/work-000.png").exists()

    # Second call hits the cache (no re-render).
    with patch("pdf2image.convert_from_path") as conv2:
        src.image("a/b/work", 0)
        conv2.assert_not_called()
