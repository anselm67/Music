"""Unit tests for PdmxSource — the PDMX-backed Source implementation."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from pdmx import PDMX, PdmxSource
from sheetmusic import Box, Page, Score, Source, Staff, Status, System


def _make_pdmx(tmp_path: Path, mxl_rel: str = "mxl/a/b/x.mxl") -> PDMX:
    (tmp_path / "PDMX.csv").write_text(f"mxl\n{mxl_rel}\n")
    return PDMX(tmp_path)


def _page(page_number: int) -> Page:
    return Page(
        page_number=page_number,
        image_width=100,
        image_height=100,
        systems=[
            System(
                bar_numbers=[1],
                bars=[0, 10],
                svg_bar_numbers=[1],
                staves=[Staff(box=Box(0, 0, 10, 10))],
            )
        ],
        status=Status.VALIDATED,
    )


def _write_layout(pdmx: PDMX, mxl_rel: str, pages: list[Page]) -> Score:
    score = Score(id=mxl_rel, pages=pages)
    layout = pdmx.get_path(pdmx.home / mxl_rel, "layout", mkdirs=True)
    layout.write_text(json.dumps(score.asdict()))
    return score


def test_pdmx_source_satisfies_protocol(tmp_path: Path) -> None:
    src: Source = PdmxSource(_make_pdmx(tmp_path))
    assert src is not None  # structural typing checked by mypy


def test_score_and_scores_load_relative_id(tmp_path: Path) -> None:
    pdmx = _make_pdmx(tmp_path)
    _write_layout(pdmx, "mxl/a/b/x.mxl", [_page(1)])
    src = PdmxSource(pdmx)

    score = src.score("mxl/a/b/x.mxl")
    assert score.id == "mxl/a/b/x.mxl"
    assert [s.id for s in src.scores()] == ["mxl/a/b/x.mxl"]


def test_image_uses_flat_png_when_present(tmp_path: Path) -> None:
    pdmx = _make_pdmx(tmp_path)
    src = PdmxSource(pdmx)
    flat = pdmx.get_path(pdmx.home / "mxl/a/b/x.mxl", "png", mkdirs=True)
    flat.touch()  # single-page: flat png exists

    with patch("pdmx.pdmx_source.decode_image") as dec:
        src.image("mxl/a/b/x.mxl", 1)
        assert dec.call_args[0][0] == flat.as_posix()


def test_image_uses_paged_png_when_flat_absent(tmp_path: Path) -> None:
    pdmx = _make_pdmx(tmp_path)
    src = PdmxSource(pdmx)
    # No flat png on disk -> multi-page `_NNN` path.
    paged = pdmx.get_page_path(pdmx.home / "mxl/a/b/x.mxl", "png", 2)

    with patch("pdmx.pdmx_source.decode_image") as dec:
        src.image("mxl/a/b/x.mxl", 2)
        assert dec.call_args[0][0] == paged.as_posix()


def test_records_reads_tokens(tmp_path: Path) -> None:
    pdmx = _make_pdmx(tmp_path)
    src = PdmxSource(pdmx)

    with patch("pdmx.pdmx_source.KernReader") as reader_cls:
        reader_cls.return_value = MagicMock(
            get_text=MagicMock(return_value=["=1", "C/4\tC/4"])
        )
        out = src.records("mxl/a/b/x.mxl", 1, 2)

    tokens = pdmx.get_path(pdmx.home / "mxl/a/b/x.mxl", "tokens")
    reader_cls.assert_called_once_with(tokens)
    reader_cls.return_value.get_text.assert_called_once_with(1, 2)
    assert out == ["=1", "C/4\tC/4"]


def test_spine_count_reads_tokens(tmp_path: Path) -> None:
    pdmx = _make_pdmx(tmp_path)
    src = PdmxSource(pdmx)

    with patch("pdmx.pdmx_source.KernReader") as reader_cls:
        reader_cls.return_value = MagicMock(spine_count=3)
        out = src.spine_count("mxl/a/b/x.mxl")

    tokens = pdmx.get_path(pdmx.home / "mxl/a/b/x.mxl", "tokens")
    reader_cls.assert_called_once_with(tokens)
    assert out == 3
