"""Verivio scrapper: scrapes page layout from verovio svg output
"""

import re
import xml.etree.ElementTree as ET
from pathlib import Path

from dataset import Page, Staff


def parse_staff_group(namespaces: dict[str, str], group, trans: tuple[int, int]) -> tuple[int, int, int, int]:
    def translate_to_pixels(point: tuple[int, int]) -> tuple[int, int]:
        return (point[0] + trans[0]) // 10, (point[1] + trans[1]) // 10

    (top, bottom, left, right) = (-1, -1, -1, -1)
    for path in group.findall("svg:path", namespaces):
        d_attr = path.get('d', '')
        match = re.match(r"^M(\d+)\s+(\d+)\s+L(\d+)\s+(\d+)$", d_attr)
        if match is None:
            continue
        coords = tuple(map(int, match.groups()))
        (x, y) = translate_to_pixels((coords[0], coords[1]))
        (to_x, to_y) = translate_to_pixels((coords[2], coords[3]))
        top = y if top < 0 else min(top, y)
        bottom = to_y if bottom < 0 else max(bottom, to_y)
        left = x if left < 0 else min(left, x)
        right = to_x if right < 0 else max(right, to_x)
    return (top, bottom, left, right)


def extract_layout(svg_file: Path, page_number: int = 1) -> Page:
    namespaces = {'svg': 'http://www.w3.org/2000/svg'}
    tree = ET.parse(svg_file)
    root = tree.getroot()

    boxes: dict[int, list[tuple[int, int, int, int]]] = dict()

    # Collects the dimensions of the page.
    try:
        image_width = int(root.attrib['width'].removesuffix('px'))
        image_height = int(root.attrib['height'].removesuffix('px'))
    except KeyError as e:
        raise ValueError(
            f"{svg_file}: missing {e} attribute on root element.") from e

    # Looks up for optional margins.
    trans = (0, 0)
    page_margin = root.find(
        ".//{http://www.w3.org/2000/svg}g[@class='page-margin']")
    if page_margin is not None:
        attr = page_margin.attrib.get('transform')
        if attr is not None and (match := re.search(r"translate\((\d+),\s*(\d+)\)", attr)):
            trans = int(match.group(1)), int(match.group(2))

    # Collects the bounding box for all bars on that page.
    for group in root.findall(".//svg:g[@class]", namespaces):
        if 'staff' != group.get('class', ''):
            continue
        # We're on a staff group.
        (top, bottom, left, right) = parse_staff_group(
            namespaces, group, trans)
        boxes.setdefault(top, list()).append((top, bottom, left, right))

    # Transform bounding boxes into a Page layout.
    staves = list()
    for top, (bar, *bars) in boxes.items():
        staves.append(Staff(
            rh_top=bar[0],
            lh_bot=bar[1],
            bars=[bar[2], bar[3], *[x[3] for x in bars]]
        ))
    return Page(
        page_number=page_number,
        image_width=image_width,
        image_height=image_height,
        staves=staves,
        validated=True
    )
