#!/usr/bin/env python3
import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import click
import cv2

from dataset import PDMX
from verovio import extract_layout, mxl_to_kern
from verovio import render as verovio_render
from verovio import svg_to_png

HOME = Path("/home/anselm/datasets/PDMX")


@dataclass
class ClickContext:
    home: Path
    pdmx: PDMX


@click.group()
@click.option("--home", "-h", type=click.Path(dir_okay=True, file_okay=False,
                                              exists=True, readable=True,
                                              path_type=Path),
              default=HOME, show_default=True)
@click.pass_context
def cli(ctx, home: Path):
    pdmx = PDMX(home)
    ctx.obj = ClickContext(home=home, pdmx=pdmx)


@click.command()
@click.argument("query_string")
@click.option("--columns", "-c", multiple=True, help="Columns to display")
@click.option("--limit", "-n", default=20, show_default=True)
@click.pass_obj
def query(ctx: ClickContext, query_string: str, columns: tuple[str], limit: int):
    """Query the underlying PDMX.csv database as a DataFrame."""
    result = ctx.pdmx.query(query_string)
    if columns:
        result = result[list(columns)]
    print(result.head(limit).to_string())


@click.command()
@click.option("--force", "-f", default=False, is_flag=True, show_default=True)
@click.option("--dry-run", "-n", default=False, is_flag=True, show_default=True)
@click.pass_obj
def to_svg(ctx: ClickContext, force: bool, dry_run: bool):
    """Converts all .mxl files into .svg files.

    --force will enforce the conversion even if the .svg file exists and is more recent than its source.
    --dry-run tells what's to be done without doing it.
    """
    total, failed = ctx.pdmx.to_svg(force, dry_run)
    print(f"{total} mxl files processed, {failed} failed.")


@click.command()
@click.option("--force", "-f", default=False, is_flag=True, show_default=True)
@click.option("--dry-run", "-n", default=False, is_flag=True, show_default=True)
@click.pass_obj
def to_kern(ctx: ClickContext, force: bool, dry_run: bool):
    """Converts all .mxl files into .krn files.

    --force will enforce the conversion even if the .svg file exists and is more recent than its source.
    --dry-run tells what's to be done without doing it.
    """
    total, failed = ctx.pdmx.to_kern(force, dry_run)
    print(f"{total} mxl files processed, {failed} failed.")


@click.command()
@click.argument("svg_file",
                type=click.Path(dir_okay=False, file_okay=True,
                                exists=True, readable=True, path_type=Path),
                required=True)
@click.option("--output", "-o",
              type=click.Path(dir_okay=False, file_okay=True, path_type=Path))
def scrape(svg_file: Path, output: Path):
    """Scrapes a verovio generated .svg file for page layout info.
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
    print(f"Output written to {svg_file}.")


@click.command()
@click.argument("mxl_file",
                type=click.Path(dir_okay=False, file_okay=True,
                                exists=True, readable=True, path_type=Path),
                required=True)
@click.option("--output", "-o",
              type=click.Path(dir_okay=False, file_okay=True, path_type=Path))
def from_mxl(mxl_file: Path, output: Path):
    """Converts ab mxl file into a kern file."""
    output = output or mxl_file.with_suffix(".krn")
    mxl_to_kern(mxl_file, output)
    print(f"Output written to {output}.")


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


cli.add_command(query)
cli.add_command(to_svg)
cli.add_command(to_kern)
cli.add_command(scrape)
cli.add_command(render)
cli.add_command(from_mxl)
cli.add_command(show)


def main():
    logging.basicConfig(level=logging.INFO)
    cli()


if __name__ == "__main__":
    main()
