"""Tests for verovio scrapper: extract_layout and parse_staff_group."""

import textwrap
from pathlib import Path

import pytest

from dataset import Page, Staff
from verovio.scraper import extract_layout


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_svg(tmp_path: Path, content: str, name: str = "test.svg") -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(content))
    return p


SVG_NS = 'xmlns="http://www.w3.org/2000/svg"'

# A minimal valid SVG with one staff group and two horizontal paths
# representing top and bottom lines of a staff.
# Coordinates are in verovio's internal units (tenths of a pixel).
# One staff, one bar: M100 200 L5000 200 (top line), M100 600 L5000 600 (bot)
SIMPLE_SVG = f"""\
    <svg width="2100px" height="2970px" version="1.1" {SVG_NS}>
      <g class="page-margin" transform="translate(0, 0)">
        <g class="staff">
          <path d="M100 200 L5000 200"/>
          <path d="M100 600 L5000 600"/>
        </g>
      </g>
    </svg>
"""

# Same but with a non-zero page margin offset.
MARGIN_SVG = f"""\
    <svg width="2100px" height="2970px" version="1.1" {SVG_NS}>
      <g class="page-margin" transform="translate(500, 300)">
        <g class="staff">
          <path d="M100 200 L5000 200"/>
          <path d="M100 600 L5000 600"/>
        </g>
      </g>
    </svg>
"""

# Two staves on the same page.
TWO_STAVES_SVG = f"""\
    <svg width="2100px" height="2970px" version="1.1" {SVG_NS}>
      <g class="page-margin" transform="translate(0, 0)">
        <g class="staff">
          <path d="M100 200 L5000 200"/>
          <path d="M100 600 L5000 600"/>
        </g>
        <g class="staff">
          <path d="M100 1200 L5000 1200"/>
          <path d="M100 1600 L5000 1600"/>
        </g>
      </g>
    </svg>
"""

# SVG missing width/height attributes — should raise ValueError.
NO_DIMS_SVG = f"""\
    <svg version="1.1" {SVG_NS}>
      <g class="staff">
        <path d="M100 200 L5000 200"/>
      </g>
    </svg>
"""

# SVG with no page-margin element — margin should default to (0, 0).
NO_MARGIN_SVG = f"""\
    <svg width="2100px" height="2970px" version="1.1" {SVG_NS}>
      <g class="staff">
        <path d="M100 200 L5000 200"/>
        <path d="M100 600 L5000 600"/>
      </g>
    </svg>
"""

# Staff group containing a non-matching path that should be ignored.
MIXED_PATHS_SVG = f"""\
    <svg width="2100px" height="2970px" version="1.1" {SVG_NS}>
      <g class="page-margin" transform="translate(0, 0)">
        <g class="staff">
          <path d="M100 200 L5000 200"/>
          <path d="C100 200 300 400 500 600"/>
          <path d="M100 600 L5000 600"/>
        </g>
      </g>
    </svg>
"""


# ---------------------------------------------------------------------------
# Page dimensions
# ---------------------------------------------------------------------------

class TestPageDimensions:
    def test_image_size(self, tmp_path):
        svg = write_svg(tmp_path, SIMPLE_SVG)
        page = extract_layout(svg)
        assert page.image_width == 2100
        assert page.image_height == 2970

    def test_missing_dimensions_raises(self, tmp_path):
        svg = write_svg(tmp_path, NO_DIMS_SVG)
        with pytest.raises(ValueError, match="width"):
            extract_layout(svg)

    def test_page_number_passthrough(self, tmp_path):
        svg = write_svg(tmp_path, SIMPLE_SVG)
        page = extract_layout(svg, page_number=3)
        assert page.page_number == 3

    def test_validated_is_true(self, tmp_path):
        svg = write_svg(tmp_path, SIMPLE_SVG)
        page = extract_layout(svg)
        assert page.validated is True


# ---------------------------------------------------------------------------
# Margin handling
# ---------------------------------------------------------------------------

class TestMargin:
    def test_zero_margin(self, tmp_path):
        svg = write_svg(tmp_path, SIMPLE_SVG)
        page = extract_layout(svg)
        # With zero margin: coords / 10
        # top line y=200 → 20, bottom line y=600 → 60
        assert page.staves[0].rh_top == 20
        assert page.staves[0].lh_bot == 60

    def test_nonzero_margin_offset(self, tmp_path):
        svg = write_svg(tmp_path, MARGIN_SVG)
        page = extract_layout(svg)
        # margin (500, 300): top = (200 + 300) // 10 = 50
        assert page.staves[0].rh_top == 50
        assert page.staves[0].lh_bot == 90  # (600 + 300) // 10

    def test_no_margin_element_defaults_to_zero(self, tmp_path):
        svg = write_svg(tmp_path, NO_MARGIN_SVG)
        page = extract_layout(svg)
        assert page.staves[0].rh_top == 20


# ---------------------------------------------------------------------------
# Staff parsing
# ---------------------------------------------------------------------------

class TestStaffParsing:
    def test_single_staff_bar_count(self, tmp_path):
        svg = write_svg(tmp_path, SIMPLE_SVG)
        page = extract_layout(svg)
        assert len(page.staves) == 1
        # bars: [left, right] = [10, 500]
        assert page.staves[0].bars == [10, 500]

    def test_two_staves(self, tmp_path):
        svg = write_svg(tmp_path, TWO_STAVES_SVG)
        page = extract_layout(svg)
        assert len(page.staves) == 2

    def test_two_staves_order(self, tmp_path):
        svg = write_svg(tmp_path, TWO_STAVES_SVG)
        page = extract_layout(svg)
        # First staff should have smaller rh_top
        assert page.staves[0].rh_top < page.staves[1].rh_top

    def test_non_matching_paths_ignored(self, tmp_path):
        svg = write_svg(tmp_path, MIXED_PATHS_SVG)
        page = extract_layout(svg)
        # Cubic bezier path should be skipped, staff still parsed correctly
        assert len(page.staves) == 1
        assert page.staves[0].rh_top == 20
        assert page.staves[0].lh_bot == 60

    def test_empty_svg_has_no_staves(self, tmp_path):
        empty = f'<svg width="2100px" height="2970px" version="1.1" {SVG_NS}/>'
        svg = write_svg(tmp_path, empty)
        page = extract_layout(svg)
        assert page.staves == []


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

class TestReturnType:
    def test_returns_page(self, tmp_path):
        svg = write_svg(tmp_path, SIMPLE_SVG)
        assert isinstance(extract_layout(svg), Page)

    def test_staves_are_staff_instances(self, tmp_path):
        svg = write_svg(tmp_path, TWO_STAVES_SVG)
        page = extract_layout(svg)
        assert all(isinstance(s, Staff) for s in page.staves)