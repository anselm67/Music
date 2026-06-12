"""Classical (cv2) staff detector, ported from the OMR project's ``Staffer``.

A projection-profile pipeline — grayscale, Otsu, deskew, then horizontal/vertical
projections with scipy peak-finding — that turns a PDF into a native :class:`Score`.
Unlike the trained ``staffer`` model it also recovers barline x-positions, and it
needs no training, so it doubles as a baseline to A/B the model against.

Assumptions (inherited from the OMR pipeline): every system is a piano grand staff
of two 5-line staves, and the page is rendered at ``width`` px (the peak-height
thresholds below are tuned for 1200px).
"""

import logging
from pathlib import Path

import cv2
import numpy as np
from cv2.typing import MatLike
from numpy.typing import NDArray
from pdf2image import convert_from_path
from scipy.signal import find_peaks

from sheetmusic import Box, Page, Score, Staff, Status, System

# Render/processing width; the peak-height thresholds in _decode_page are tuned to it.
WIDTH = 1200

# A grand staff is two 5-line staves; LINES_PER_SYSTEM staff lines bound one System.
LINES_PER_STAFF = 5
LINES_PER_SYSTEM = 2 * LINES_PER_STAFF


class ClassicalStaffer:
    """Detect systems/staves/barlines from a PDF using projection profiles."""

    def __init__(self, width: int = WIDTH) -> None:
        self.width = width

    def detect(self, pdf_path: Path, id: str) -> Score:
        """Render every page of ``pdf_path`` and decode it into a :class:`Score`."""
        images = convert_from_path(pdf_path)
        next_bar = 0
        pages = []
        for pageno, image in enumerate(images, start=1):
            page, next_bar = self._decode_page(np.array(image), pageno, next_bar)
            pages.append(page)
        return Score(id=id, pages=pages)

    # --- image preprocessing -------------------------------------------------

    def _denoise(self, image: MatLike) -> MatLike:
        _, image = cv2.threshold(image, 127, 255, cv2.THRESH_OTSU)
        return image

    def _average_angle(self, image: MatLike, hough_threshold: int = 500) -> float:
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        hough_lines = cv2.HoughLines(
            edges,
            1,
            np.pi / 14400,
            hough_threshold,
            min_theta=np.radians(80),
            max_theta=np.radians(100),
        )
        if hough_lines is None:
            return 0.0
        median_angle = float(np.median(hough_lines.squeeze(1)[:, 1]))
        return np.degrees(median_angle) - 90.0

    def _deskew(
        self, image: MatLike, min_rotation_angle_degrees: float = 0.05
    ) -> tuple[float, MatLike]:
        angle = self._average_angle(image)
        if abs(angle) < min_rotation_angle_degrees:
            return 0.0, image
        height, width = image.shape
        matrix = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1)
        return angle, cv2.warpAffine(image, matrix, (width, height))

    def _transform(self, orig_image: MatLike) -> tuple[float, MatLike]:
        image = cv2.cvtColor(np.array(orig_image), cv2.COLOR_RGB2GRAY)
        h, w = image.shape
        scale = self.width / w
        image = cv2.resize(
            image, (self.width, int(h * scale)), interpolation=cv2.INTER_AREA
        )
        image = self._denoise(image)
        rotation_angle, image = self._deskew(image)
        # deskew's warpAffine de-binarises the image; re-threshold.
        image = self._denoise(image)
        return float(rotation_angle), image

    # --- peak finding --------------------------------------------------------

    def _find_bars(self, image: MatLike, fill_rate: float = 0.8) -> list[int]:
        x_proj = np.sum(image, 0)
        # A barline spans the staff top-to-bottom; require a near-full vertical line.
        peak_height = int(image.shape[0] * fill_rate * 255)
        bar_peaks, _ = find_peaks(x_proj, distance=50, height=peak_height)
        return [int(p) for p in bar_peaks]

    def _close_vertical(self, ink: MatLike) -> MatLike:
        """Bridge vertical gaps so barlines read as solid columns for find_bars."""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        return cv2.morphologyEx(ink, cv2.MORPH_CLOSE, kernel)

    def detect_bars(self, page_image: MatLike, top: int, bottom: int) -> list[int]:
        """Barline x-positions within the page's horizontal band ``[top:bottom]``.

        Exposed for the editor's recompute-bars: it runs the same vertical-projection
        bar finder used during full detection, on a band the caller already has staff
        boxes for — no staff detection repeated. ``page_image`` is taken as-is (no
        resize/deskew), so ``top``/``bottom`` are pixel rows in its own frame.
        """
        gray = page_image
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        ink = cv2.bitwise_not(self._denoise(gray))
        return self._find_bars(self._close_vertical(ink)[top:bottom, :])

    def _filter_staff_peaks(
        self, staff_peaks: NDArray[np.int_], peak_heights: list[float]
    ) -> list[int]:
        # An odd staff count means a spurious peak: a title band above the first
        # staff or a legend below the last (an outsized gap), else the weakest peak.
        gaps = staff_peaks[1:] - staff_peaks[:-1]
        if len(gaps) > 0:
            average_gap = sum(gaps) / len(gaps)
            if gaps[0] > 2 * average_gap:
                return [int(p) for p in staff_peaks[1:]]
            if gaps[-1] > 2 * average_gap:
                return [int(p) for p in staff_peaks[:-1]]
        sorted_peaks = sorted(
            zip(staff_peaks, peak_heights), key=lambda x: x[1], reverse=True
        )
        return [int(peak) for peak, _ in sorted_peaks[:-1]]

    def _staff_lines(self, y_lines: NDArray[np.float64], peak: int) -> list[int]:
        """The five staff-line y-positions around a staff peak."""
        peak_height = 75000
        start = max(0, peak - 40)
        staff_range = y_lines[start : peak + 40]
        line_peaks, properties = find_peaks(staff_range, height=peak_height, distance=7)
        if len(line_peaks) > LINES_PER_STAFF:
            # Too lax: keep the five strongest peaks.
            sorted_peaks = sorted(
                zip(line_peaks, properties["peak_heights"].tolist()),
                key=lambda x: x[1],
                reverse=True,
            )
            line_peaks = np.array([lp for lp, _ in sorted_peaks[:LINES_PER_STAFF]])
        elif len(line_peaks) < LINES_PER_STAFF:
            # Lower the threshold until five lines emerge.
            while peak_height > 0 and len(line_peaks) < LINES_PER_STAFF:
                peak_height -= 2500
                line_peaks, _ = find_peaks(staff_range, height=peak_height, distance=7)
        return [int(p + start) for p in line_peaks]

    # --- page decode ---------------------------------------------------------

    def _decode_page(
        self, orig_image: MatLike, pageno: int, next_bar: int
    ) -> tuple[Page, int]:
        image_rotation, image = self._transform(orig_image)
        image_height, image_width = image.shape

        # Horizontal close + row projection → rough staff positions.
        ink = cv2.bitwise_not(image)
        h_ink = cv2.morphologyEx(
            ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        )
        y_lines = np.sum(h_ink, 1).astype(np.float64)

        wsize = 50
        offset = (wsize - 1) // 2
        y_avg = np.convolve(y_lines, np.ones(wsize) / wsize, mode="valid")
        y_proj = np.zeros(y_lines.shape)
        y_proj[offset : offset + len(y_avg)] = y_avg
        staff_peaks, properties = find_peaks(
            y_proj, width=20, distance=50, height=40_000
        )

        peaks: list[int] = [int(p) for p in staff_peaks]
        if len(peaks) % 2 != 0:
            peaks = self._filter_staff_peaks(
                staff_peaks, properties["peak_heights"].tolist()
            )

        staff_lines: list[int] = []
        for peak in peaks:
            staff_lines.extend(self._staff_lines(y_lines, peak))
        staff_lines.sort()

        # Keep whole grand staves only.
        usable = LINES_PER_SYSTEM * (len(staff_lines) // LINES_PER_SYSTEM)
        if usable != len(staff_lines):
            logging.warning(
                "page %d: trimming %d stray staff line(s)",
                pageno,
                len(staff_lines) - usable,
            )
            staff_lines = staff_lines[:usable]

        # Vertical close → image used to locate barlines within each band.
        v_ink = self._close_vertical(ink)

        systems = []
        for base in range(0, len(staff_lines), LINES_PER_SYSTEM):
            lines = staff_lines[base : base + LINES_PER_SYSTEM]
            bars = self._find_bars(v_ink[lines[0] : lines[-1], :])
            left = bars[0] if bars else 0
            right = bars[-1] if bars else image_width
            treble = Staff(Box(left, lines[0], right, lines[4]))
            bass = Staff(Box(left, lines[5], right, lines[9]))
            bar_count = max(len(bars) - 1, 0)
            bar_numbers = list(range(next_bar, next_bar + bar_count)) or [next_bar]
            next_bar += bar_count
            systems.append(
                System(bar_numbers=bar_numbers, bars=bars, staves=[treble, bass])
            )

        page = Page(
            page_number=pageno,
            image_width=image_width,
            image_height=image_height,
            systems=systems,
            status=Status.PENDING,
            image_rotation=image_rotation,
        )
        return page, next_bar
