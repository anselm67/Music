"""Verivio scrapper: scrapes page layout from verovio svg output"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from xml.etree.ElementTree import Element, ElementTree, parse

from sheetmusic import Box, Page, Staff, System


class LayoutExtractor:
    svg_file: Path
    tree: ElementTree[Element[str]]  # type: ignore[type-arg]
    ns: dict[str, str]
    translation: tuple[int, int]

    def __init__(self, svg_file: Path) -> None:
        self.svg_file = svg_file
        self.tree = parse(svg_file)
        self.ns = {
            "svg": "http://www.w3.org/2000/svg",
            "xlink": "http://www.w3.org/1999/xlink",
        }
        self.parse_translation()

    def parse_translation(self) -> None:
        self.translation = (0, 0)
        page_margin = self.tree.getroot().find(
            ".//svg:g[@class='page-margin']", self.ns
        )
        if page_margin is not None:
            attr = page_margin.attrib.get("transform")
            if attr is not None and (
                match := re.search(r"translate\((\d+),\s*(\d+)\)", attr)
            ):
                self.translation = int(match.group(1)), int(match.group(2))

    def translate(self, point: tuple[int, int]) -> tuple[int, int]:
        return (point[0] + self.translation[0]) // 10, (
            point[1] + self.translation[1]
        ) // 10

    MULTIREST_RE = re.compile(r"^#E08(?P<digit>\d)-.*$")

    def check_multirest(self, staff_group: Element) -> int | None:
        multirest = staff_group.find(".//svg:g[@class='multiRest']", self.ns)
        if multirest is None:
            return None
        digits: list[str] = list()
        for use in multirest.findall("svg:use", self.ns):
            href = use.get(f"{{{self.ns['xlink']}}}href")
            if href and (m := self.MULTIREST_RE.match(href)) is not None:
                digits.append(m.group("digit"))
        if digits:
            return int("".join(digits))
        else:
            return None

    def parse_staff_group(self, staff_group: Element) -> tuple[int, Box]:
        """Returns the bar count (multirest) and bounding box for the staff."""
        (top, bottom, left, right) = (-1, -1, -1, -1)
        for path in staff_group.findall("svg:path", self.ns):
            d_attr = path.get("d", "")
            match = re.match(r"^M(\d+)\s+(\d+)\s+L(\d+)\s+(\d+)$", d_attr)
            if match is None:
                continue
            coords = tuple(map(int, match.groups()))
            (x, y) = self.translate((coords[0], coords[1]))
            (to_x, to_y) = self.translate((coords[2], coords[3]))
            top = y if top < 0 else min(top, y)
            bottom = to_y if bottom < 0 else max(bottom, to_y)
            left = x if left < 0 else min(left, x)
            right = to_x if right < 0 else max(right, to_x)

        # Checks for a multirest measure.
        bar_count = self.check_multirest(staff_group)
        return (
            1 if bar_count is None else bar_count,
            Box((left, top), (right, bottom)),
        )

    def check_bar_number(self, measure_group: Element) -> int | None:
        for g in measure_group.findall(".//svg:g", self.ns):
            g_class = g.get("class") or ""
            if ("mNum" in g_class) or g_class == "reh":
                text = g.find("svg:text", self.ns)
                if text is None:
                    return None
                # Verovio sometimes wraps in tspan
                tspans = list(text.iterfind(".//svg:tspan", self.ns))
                if not tspans:
                    return None
                raw = tspans[-1].text or ""
                return int(raw) if raw.strip().isdigit() else None
        return None

    def parse_system(
        self, system_group: Element, bar_number: int = 1
    ) -> tuple[int, System]:
        # Collects the bounding box for all bars within that system.
        boxes: dict[int, list[Box]] = dict()
        bar_count = 0
        bar_numbers: list[int] = list()
        svg_bar_numbers: list[int | None] = list()
        for measure in system_group.findall(".//svg:g[@class='measure']", self.ns):
            svg_bar_numbers.append(self.check_bar_number(measure))
            measure_bar_count: int = -1
            for group in measure.findall(".//svg:g[@class='staff']", self.ns):
                count, box = self.parse_staff_group(group)
                if measure_bar_count < 0:
                    measure_bar_count = count
                elif measure_bar_count != count:
                    logging.warning(
                        f"{self.svg_file}: "
                        f"measure {measure.get('id')} bar count mismatch."
                    )
                boxes.setdefault(box.top, list()).append(box)
            bar_numbers.append(bar_number + bar_count)
            bar_count += measure_bar_count

        # Transform bars bounding boxes into a list of staves.
        staves: list[Staff] = list()
        for _, bars in sorted(boxes.items()):
            staves.append(
                Staff(
                    box=Box(bars[0].top_left, bars[-1].bot_right),
                    bars=[bars[0].left, bars[0].right, *[x.right for x in bars[1:]]],
                )
            )
        # TODO Move bars[] for the system into the System and out of the Staff.
        if not staves:
            raise ValueError(f"{self.svg_file} has a system with no staff.")
        first_bars = staves[0].bars
        if not all(staff.bars == first_bars for staff in staves):
            raise ValueError(f"{self.svg_file}: system has incompatible bar values.")
        if len(svg_bar_numbers) != len(first_bars) - 1:
            raise ValueError(
                f"{self.svg_file}: svg bar numbers disagrees with stave bars."
            )
        return bar_number + bar_count, System(bar_numbers, staves, svg_bar_numbers)

    def parse(self, page_number: int = 1, bar_number: int = 1) -> Page:
        root = self.tree.getroot()

        # Collects the dimensions of the page.
        try:
            image_width = int(root.attrib["width"].removesuffix("px"))
            image_height = int(root.attrib["height"].removesuffix("px"))
        except KeyError as e:
            raise ValueError(
                f"{self.svg_file}: missing {e} attribute on root element."
            ) from e

        # Collects the bounding box for all bars on that page.
        systems: list[System] = list()
        for group in root.findall(".//svg:g[@class='system']", self.ns):
            bar_number, system = self.parse_system(group, bar_number)
            systems.append(system)

        return Page(
            page_number=page_number,
            image_width=image_width,
            image_height=image_height,
            systems=systems,
            validated=True,
        )


# vscode - End of File
