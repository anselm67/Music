"""Verivio scrapper: scrapes page layout from verovio svg output
"""

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path


def translate_to_pixels(measure: int) -> int:
    return (500 + measure) // 10


def parse_staff_group(namespaces: dict[str, str], group) -> tuple[int, int, int, int]:
    (top, bottom, left, right) = (-1, -1, -1, -1)
    for path in group.findall("svg:path", namespaces):
        d_attr = path.get('d', '')
        match = re.match(r"^M(\d+)\s+(\d+)\s+L(\d+)\s+(\d+)$", d_attr)
        if match is None:
            continue
        (x, y, to_x, to_y) = tuple(translate_to_pixels(int(n))
                                   for n in match.groups())
        top = y if top < 0 else min(top, y)
        bottom = to_y if bottom < 0 else max(bottom, to_y)
        left = x if left < 0 else min(left, x)
        right = to_x if right < 0 else max(right, to_x)
    return (top, bottom, left, right)


def extract_layout(svg_file: Path, json_file: Path):
    namespaces = {'svg': 'http://www.w3.org/2000/svg'}
    tree = ET.parse(svg_file)
    root = tree.getroot()

    staves = []

    for group in root.findall(".//svg:g[@class]", namespaces):
        if 'staff' != group.get('class', ''):
            continue
        # We're on a staff group.
        (top, bottom, left, right) = parse_staff_group(namespaces, group)
        staves.append({
            'top': top,
            'bottom': bottom,
            'left': left,
            'right': right,
        })
    with open(json_file, 'w') as f:
        json.dump(staves, f, indent=2)
    return json_file
