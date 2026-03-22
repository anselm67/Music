"""Verivio scrapper: scrapes page layout from verovio svg output
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

from dataset import Box, Page, Staff, System


class LayoutExtractor:
    svg_file: Path
    tree: ET.ElementTree[ET.Element]
    namespaces: dict[str, str]
    translation: tuple[int, int]

    def __init__(self, svg_file: Path):
        self.svg_file = svg_file
        self.tree = ET.parse(svg_file)
        self.namespaces = {'svg': 'http://www.w3.org/2000/svg'}
        self.parse_translation()

    def parse_translation(self):
        self.translation = (0, 0)
        page_margin = self.tree.getroot().find(
            ".//{http://www.w3.org/2000/svg}g[@class='page-margin']")
        if page_margin is not None:
            attr = page_margin.attrib.get('transform')
            if attr is not None and (match := re.search(r"translate\((\d+),\s*(\d+)\)", attr)):
                self.translation = int(match.group(1)), int(match.group(2))

    def translate(self, point: tuple[int, int]) -> tuple[int, int]:
        return (point[0] + self.translation[0]) // 10, (point[1] + self.translation[1]) // 10

    def parse_staff_group(self, staff_group) -> Box:
        (top, bottom, left, right) = (-1, -1, -1, -1)
        for path in staff_group.findall("svg:path", self.namespaces):
            d_attr = path.get('d', '')
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
        return Box((left, top), (right, bottom))

    def parse_system(self, system_group, bar_number: int = 1) -> tuple[int, System]:
        # Collects the bounding box for all bars within that system.
        boxes: dict[int, list[Box]] = dict()
        for group in system_group.findall(".//svg:g[@class='staff']", self.namespaces):
            box = self.parse_staff_group(group)
            boxes.setdefault(box.top, list()).append(box)

        # Transform bars bounding boxes into a list of staves.
        staves: list[Staff] = list()
        bar_count = -1
        for _, bars in sorted(boxes.items()):
            if bar_count < 0:
                bar_count = len(bars)
            elif bar_count != len(bars):
                raise ValueError("Bar count mismatch.")
            staves.append(Staff(
                box=Box(bars[0].top_left, bars[-1].bot_right),
                bars=[bars[0].left, bars[0].right,
                      *[x.right for x in bars[1:]]]
            ))

        if not staves:
            raise ValueError(f"{self.svg_file} has a system with no staff.")
        return bar_count, System(staves=staves, bar_number=bar_number)

    def parse(self, page_number: int = 1, bar_number: int = 1) -> Page:
        root = self.tree.getroot()

        # Collects the dimensions of the page.
        try:
            image_width = int(root.attrib['width'].removesuffix('px'))
            image_height = int(root.attrib['height'].removesuffix('px'))
        except KeyError as e:
            raise ValueError(
                f"{self.svg_file}: missing {e} attribute on root element.") from e

        # Collects the bounding box for all bars on that page.
        systems: list[System] = list()
        for group in root.findall(".//svg:g[@class='system']", self.namespaces):
            bar_count, system = self.parse_system(group, bar_number)
            systems.append(system)
            bar_number += bar_count

        return Page(
            page_number=page_number,
            image_width=image_width,
            image_height=image_height,
            systems=systems,
            validated=True
        )
