#!/usr/bin/env python3
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import click
import cv2

from dataset import PDMX, Prepare, Score
from utils import from_json
from verovio import LayoutExtractor, mxl_to_kern
from verovio import render as verovio_render

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
@click.option("--parts", "-p", type=int, help="With this number of parts.", default=-1)
@click.option("--columns", "-c", multiple=True, help="Columns to display")
@click.option("--limit", "-n", default=20, show_default=True)
@click.pass_obj
def query(ctx: ClickContext, query_string: str, parts: int, columns: tuple[str], limit: int):
    """Query the underlying PDMX.csv database as a DataFrame."""
    result = ctx.pdmx.query(query_string, parts_count=parts)
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
@click.option("--force", "-f", default=False, is_flag=True, show_default=True)
@click.pass_obj
def to_layout(ctx: ClickContext, force: bool):
    """Extracts layout structure from .svg files into json.

    --force will enforce the conversion even if the .svg file exists and is more recent than its source.
    """
    total, failed = ctx.pdmx.to_layout(force)
    print(f"{total} svg files processed, {failed} failed.")


@click.command()
@click.option("--force", "-f", default=False, is_flag=True, show_default=True)
@click.option("--dry-run", "-n", default=False, is_flag=True, show_default=True)
@click.pass_obj
def to_png(ctx: ClickContext, force: bool, dry_run: bool):
    """Renders all .svg files into .png files.

    --force will enforce the conversion even if the .svg file exists and is more recent than its source.
    --dry-run tells what's to be done without doing it.
    """
    total, failed = ctx.pdmx.to_png(force, dry_run)
    print(f"{total} svg files processed, {failed} failed.")


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
@click.argument("any_file",
                type=click.Path(dir_okay=False, file_okay=True,
                                exists=True, readable=True, path_type=Path),
                required=True)
@click.pass_obj
def show(ctx: ClickContext, any_file: Path):
    """Displays the provided image and layout info when available."""
    pdmx = ctx.pdmx
    with open(pdmx.get_path(any_file, 'layout'), 'r') as f:
        obj = json.load(f)
    score = cast(Score, from_json(Score, obj))

    page_index = 0
    while True:
        page = score.pages[page_index]
        if len(score.pages) > 1:
            img_path = pdmx.get_page_path(any_file, 'png', page.page_number)
        else:
            img_path = pdmx.get_path(any_file, 'png')
        img = cv2.imread(img_path)
        assert img is not None, f"Can't read image {img_path}"
        print(
            f"{len(page.systems)} systems, first system {len(page.systems[0].staves)} staves.")
        # Renders the page layout on top of the image.
        system_color = (255, 0, 0)
        staff_color = (0, 255, 0)
        bar_color = (0, 0, 255)
        for system in page.systems:
            cv2.rectangle(img, system.box.top_left,
                          system.box.bot_right, system_color, 8)
            for staff in system.staves:
                cv2.rectangle(img, staff.box.top_left,
                              staff.box.bot_right, staff_color, 4)
                for bar in staff.bars:
                    cv2.line(img, (bar, staff.box.top),
                             (bar, staff.box.bottom), bar_color, 2)
        scale = 0.4
        (width, height) = tuple(map(lambda x: int(x * scale), img.shape[:2]))
        cv2.imshow("layout", cv2.resize(img, (height, width)))
        cv2.setMouseCallback("layout", mouse_positon_handler)

        if (key := cv2.waitKey()) == ord('q'):
            break
        elif key == ord('p'):
            page_index = max(0, page_index - 1)
        elif key == ord('n'):
            page_index = min(len(score.pages), page_index + 1)

    cv2.destroyAllWindows()


@click.command()
@click.option("--force", "-f", default=False, is_flag=True, show_default=True)
@click.option("--dry-run", "-n", default=False, is_flag=True, show_default=True)
@click.argument("mxl_file",
                type=click.Path(dir_okay=False, file_okay=True,
                                exists=True, readable=True, path_type=Path),
                required=False, default=None)
@click.pass_obj
def prepare(ctx: ClickContext, mxl_file: Path | None, force: bool, dry_run: bool):
    p = Prepare(ctx.pdmx, force, dry_run)
    p.prepare(mxl_file, num_worker=1)


cli.add_command(query)
cli.add_command(to_svg)
cli.add_command(to_kern)
cli.add_command(to_layout)
cli.add_command(to_png)
cli.add_command(render)
cli.add_command(from_mxl)
cli.add_command(show)
cli.add_command(prepare)


def main():
    logging.basicConfig(level=logging.INFO)
    cli()


if __name__ == "__main__":
    main()
