#!/usr/bin/env python3
import json
import logging
import tempfile
from pathlib import Path

import click
import cv2

from verovio import extract_layout
from verovio import render as verovio_render
from verovio import svg_to_png


@click.group()
def cli():
    pass


@click.command()
@click.argument("svg_file",
                type=click.Path(dir_okay=False, file_okay=True,
                                exists=True, readable=True, path_type=Path),
                required=True)
@click.option("--output", "-o",
              type=click.Path(dir_okay=False, file_okay=True, path_type=Path))
def scrape(svg_file: Path, output: Path):
    """Scrapes an verovio generated .svg file for page layout info.
    """
    json_file = output or svg_file.with_suffix(".json")
    page = extract_layout(svg_file)
    with open(json_file, "w") as f:
        json.dump(page.asdict(), f, indent=2)


@click.command()
@click.argument("mxl_file",
                type=click.Path(dir_okay=False, file_okay=True,
                                exists=True, readable=True, path_type=Path),
                required=True)
@click.option("--output", "-o",
              type=click.Path(dir_okay=False, file_okay=True, path_type=Path))
def render(mxl_file: Path, output: Path):
    """Renders the mxl file into .svg files, one per page."""
    svg_file = output or mxl_file.with_suffix(".svg")
    verovio_render(mxl_file, svg_file)


def mouse_positon_handler(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"({x}, {y})")


@click.command()
@click.argument("svg_file",
                type=click.Path(dir_okay=False, file_okay=True,
                                exists=True, readable=True, path_type=Path),
                required=True)
def show(svg_file):
    """Displays the provided image and layout info when available."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
        png_file = Path(tmp.name)
        svg_to_png(svg_file, png_file)

        img = cv2.imread(png_file)
        assert img is not None, f"Failed to read {png_file}"

        page = extract_layout(svg_file)
        print(json.dumps(page.asdict(), indent=2))
        # Renders the page layout on top of the image.
        box_color = (155, 0, 0)
        bar_color = (0, 255, 0)
        for staff in page.staves:
            (top_left, bot_right) = staff.box()
            cv2.rectangle(img, top_left, bot_right, box_color, 2)
            for bar in staff.bars:
                cv2.line(img, (bar, staff.rh_top),
                         (bar, staff.lh_bot), bar_color, 2)

        cv2.imshow("layout", img)
        cv2.setMouseCallback("layout", mouse_positon_handler)
        while True:
            if cv2.waitKey() == ord('q'):
                break

        cv2.destroyAllWindows()


cli.add_command(render)
cli.add_command(scrape)
cli.add_command(show)


def main():
    logging.basicConfig(level=logging.INFO)
    cli()


if __name__ == "__main__":
    main()
