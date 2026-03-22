"""Tests for LayoutExtractor (verovio scraper)."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from dataset import Box, Page, Staff, System
from verovio.scraper import LayoutExtractor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SVG_NS = 'xmlns="http://www.w3.org/2000/svg"'


def write_svg(tmp_path: Path, content: str, name: str = "test.svg") -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(content))
    return p


def make_svg(body: str, width: int = 2100, height: int = 2970,
             margin: tuple[int, int] | None = (0, 0)) -> str:
    margin_open = ""
    margin_close = ""
    if margin is not None:
        margin_open = f'<g class="page-margin" transform="translate({margin[0]}, {margin[1]})">'
        margin_close = "</g>"
    return f"""\
<svg width="{width}px" height="{height}px" version="1.1" {SVG_NS}>
  {margin_open}
  {body}
  {margin_close}
</svg>"""


def staff_paths(top_y: int, bot_y: int, left_x: int = 100, right_x: int = 5000) -> str:
    """Two horizontal paths representing top and bottom staff lines."""
    return f"""\
<path d="M{left_x} {top_y} L{right_x} {top_y}"/>
<path d="M{left_x} {bot_y} L{right_x} {bot_y}"/>"""


def one_staff_system(top_y: int = 200, bot_y: int = 600) -> str:
    return f"""\
<g class="system">
  <g class="staff">
    {staff_paths(top_y, bot_y)}
  </g>
</g>"""


def two_staff_system(top1: int = 200, bot1: int = 600,
                     top2: int = 1200, bot2: int = 1600) -> str:
    return f"""\
<g class="system">
  <g class="staff">
    {staff_paths(top1, bot1)}
  </g>
  <g class="staff">
    {staff_paths(top2, bot2)}
  </g>
</g>"""


# ---------------------------------------------------------------------------
# Translation / margin
# ---------------------------------------------------------------------------

class TestTranslation:
    def test_zero_margin(self, tmp_path):
        svg = write_svg(tmp_path, make_svg(one_staff_system()))
        ex = LayoutExtractor(svg)
        assert ex.translation == (0, 0)

    def test_nonzero_margin_parsed(self, tmp_path):
        svg = write_svg(tmp_path, make_svg(
            one_staff_system(), margin=(500, 300)))
        ex = LayoutExtractor(svg)
        assert ex.translation == (500, 300)

    def test_no_margin_element(self, tmp_path):
        body = one_staff_system()
        content = f'<svg width="2100px" height="2970px" version="1.1" {SVG_NS}>{body}</svg>'
        svg = write_svg(tmp_path, content)
        ex = LayoutExtractor(svg)
        assert ex.translation == (0, 0)

    def test_translate_applies_offset(self, tmp_path):
        svg = write_svg(tmp_path, make_svg(
            one_staff_system(), margin=(500, 300)))
        ex = LayoutExtractor(svg)
        # (100 + 500) // 10 = 60, (200 + 300) // 10 = 50
        assert ex.translate((100, 200)) == (60, 50)

    def test_translate_zero_offset(self, tmp_path):
        svg = write_svg(tmp_path, make_svg(one_staff_system()))
        ex = LayoutExtractor(svg)
        assert ex.translate((100, 200)) == (10, 20)


# ---------------------------------------------------------------------------
# Page dimensions
# ---------------------------------------------------------------------------

class TestPageDimensions:
    def test_image_size(self, tmp_path):
        svg = write_svg(tmp_path, make_svg(one_staff_system()))
        page = LayoutExtractor(svg).parse()
        assert page.image_width == 2100
        assert page.image_height == 2970

    def test_custom_dimensions(self, tmp_path):
        svg = write_svg(tmp_path, make_svg(
            one_staff_system(), width=1200, height=800))
        page = LayoutExtractor(svg).parse()
        assert page.image_width == 1200
        assert page.image_height == 800

    def test_missing_width_raises(self, tmp_path):
        content = f'<svg height="2970px" version="1.1" {SVG_NS}/>'
        svg = write_svg(tmp_path, content)
        with pytest.raises(ValueError, match="width"):
            LayoutExtractor(svg).parse()

    def test_missing_height_raises(self, tmp_path):
        content = f'<svg width="2100px" version="1.1" {SVG_NS}/>'
        svg = write_svg(tmp_path, content)
        with pytest.raises(ValueError, match="height"):
            LayoutExtractor(svg).parse()

    def test_page_number_passthrough(self, tmp_path):
        svg = write_svg(tmp_path, make_svg(one_staff_system()))
        page = LayoutExtractor(svg).parse(page_number=5)
        assert page.page_number == 5

    def test_validated_true(self, tmp_path):
        svg = write_svg(tmp_path, make_svg(one_staff_system()))
        assert LayoutExtractor(svg).parse().validated is True


# ---------------------------------------------------------------------------
# System parsing
# ---------------------------------------------------------------------------

class TestSystemParsing:
    def test_single_system(self, tmp_path):
        svg = write_svg(tmp_path, make_svg(one_staff_system()))
        page = LayoutExtractor(svg).parse()
        assert len(page.systems) == 1

    def test_two_systems(self, tmp_path):
        body = one_staff_system(200, 600) + one_staff_system(1200, 1600)
        svg = write_svg(tmp_path, make_svg(body))
        page = LayoutExtractor(svg).parse()
        assert len(page.systems) == 2

    def test_empty_svg_no_systems(self, tmp_path):
        content = f'<svg width="2100px" height="2970px" version="1.1" {SVG_NS}/>'
        svg = write_svg(tmp_path, content)
        page = LayoutExtractor(svg).parse()
        assert page.systems == []

    def test_system_bar_number_starts_at_1(self, tmp_path):
        svg = write_svg(tmp_path, make_svg(one_staff_system()))
        page = LayoutExtractor(svg).parse()
        assert page.systems[0].bar_number == 1

    def test_system_bar_number_custom_start(self, tmp_path):
        svg = write_svg(tmp_path, make_svg(one_staff_system()))
        page = LayoutExtractor(svg).parse(bar_number=5)
        assert page.systems[0].bar_number == 5

    def test_two_systems_bar_numbers_increment(self, tmp_path):
        body = f"""\
<g class="system">
  <g class="staff">
    <path d="M100 200 L1000 200"/>
    <path d="M100 600 L1000 600"/>
    <path d="M2000 200 L3000 200"/>
    <path d="M2000 600 L3000 600"/>
  </g>
</g>
<g class="system">
  <g class="staff">
    {staff_paths(1200, 1600)}
  </g>
</g>"""
        svg = write_svg(tmp_path, make_svg(body))
        page = LayoutExtractor(svg).parse()
        assert page.systems[0].bar_number == 1
        assert page.systems[1].bar_number == 1 + page.systems[0].bar_count


# ---------------------------------------------------------------------------
# Staff parsing
# ---------------------------------------------------------------------------

class TestStaffParsing:
    def test_single_staff_top_bottom(self, tmp_path):
        # M100 200 → translate (10, 20), M100 600 → (10, 60)
        svg = write_svg(tmp_path, make_svg(one_staff_system(200, 600)))
        staff = LayoutExtractor(svg).parse().systems[0].staves[0]
        assert staff.top == 20
        assert staff.bottom == 60

    def test_two_staves_ordered_top_to_bottom(self, tmp_path):
        svg = write_svg(tmp_path, make_svg(two_staff_system()))
        staves = LayoutExtractor(svg).parse().systems[0].staves
        assert len(staves) == 2
        assert staves[0].top < staves[1].top

    def test_non_matching_paths_ignored(self, tmp_path):
        body = f"""\
<g class="system">
  <g class="staff">
    <path d="M100 200 L5000 200"/>
    <path d="C100 200 300 400 500 600"/>
    <path d="M100 600 L5000 600"/>
  </g>
</g>"""
        svg = write_svg(tmp_path, make_svg(body))
        page = LayoutExtractor(svg).parse()
        assert len(page.systems[0].staves) == 1
        staff = page.systems[0].staves[0]
        assert staff.top == 20
        assert staff.bottom == 60

    def test_staff_left_right_bars(self, tmp_path):
        # left_x=100 → 10, right_x=5000 → 500
        svg = write_svg(tmp_path, make_svg(one_staff_system()))
        staff = LayoutExtractor(svg).parse().systems[0].staves[0]
        assert staff.bars[0] == 10
        assert staff.bars[-1] == 500

    def test_staff_with_margin_offset(self, tmp_path):
        svg = write_svg(tmp_path, make_svg(
            one_staff_system(200, 600), margin=(500, 300)))
        staff = LayoutExtractor(svg).parse().systems[0].staves[0]
        # top: (200 + 300) // 10 = 50, bottom: (600 + 300) // 10 = 90
        assert staff.top == 50
        assert staff.bottom == 90


# ---------------------------------------------------------------------------
# System box
# ---------------------------------------------------------------------------

class TestSystemBox:
    def test_single_staff_system_box_matches_staff(self, tmp_path):
        svg = write_svg(tmp_path, make_svg(one_staff_system(200, 600)))
        system = LayoutExtractor(svg).parse().systems[0]
        assert system.top == system.staves[0].top
        assert system.bottom == system.staves[0].bottom

    def test_two_staff_system_box_spans_both(self, tmp_path):
        svg = write_svg(tmp_path, make_svg(
            two_staff_system(200, 600, 1200, 1600)))
        system = LayoutExtractor(svg).parse().systems[0]
        assert system.top == system.staves[0].top
        assert system.bottom == system.staves[-1].bottom

    def test_system_left_right_from_first_staff(self, tmp_path):
        svg = write_svg(tmp_path, make_svg(one_staff_system()))
        system = LayoutExtractor(svg).parse().systems[0]
        assert system.left == system.staves[0].left
        assert system.right == system.staves[0].right


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------

class TestReturnTypes:
    def test_returns_page(self, tmp_path):
        svg = write_svg(tmp_path, make_svg(one_staff_system()))
        assert isinstance(LayoutExtractor(svg).parse(), Page)

    def test_systems_are_system_instances(self, tmp_path):
        svg = write_svg(tmp_path, make_svg(one_staff_system()))
        page = LayoutExtractor(svg).parse()
        assert all(isinstance(s, System) for s in page.systems)

    def test_staves_are_staff_instances(self, tmp_path):
        svg = write_svg(tmp_path, make_svg(two_staff_system()))
        page = LayoutExtractor(svg).parse()
        for system in page.systems:
            assert all(isinstance(s, Staff) for s in system.staves)
