"""Unit tests for ClassicalStaffer's barline detector (the editor-facing path).

Full ``detect`` needs a PDF; ``detect_bars`` is the deterministic, image-only piece
the ``c`` recompute-bars key relies on, so it is exercised here on a synthetic page.
"""

import numpy as np

from kernsheet import ClassicalStaffer


def _page_with_bars(
    bar_xs: list[int], top: int, bottom: int, width: int = 600, height: int = 400
) -> np.ndarray:
    """A white page with 1px black vertical lines at ``bar_xs``, spanning rows
    ``[top, bottom)`` only."""
    img = np.full((height, width, 3), 255, np.uint8)
    for x in bar_xs:
        img[top:bottom, x : x + 1] = 0
    return img


def test_detect_bars_finds_vertical_lines() -> None:
    img = _page_with_bars([50, 300, 550], top=100, bottom=300)
    assert ClassicalStaffer().detect_bars(img, 100, 300) == [50, 300, 550]


def test_detect_bars_only_within_band() -> None:
    # A line that stops above the queried band must not be detected.
    img = _page_with_bars([200], top=0, bottom=80)
    assert ClassicalStaffer().detect_bars(img, 100, 300) == []


def test_detect_bars_accepts_grayscale() -> None:
    img = _page_with_bars([100, 400], top=50, bottom=250)
    gray = img[:, :, 0]
    assert ClassicalStaffer().detect_bars(gray, 50, 250) == [100, 400]
