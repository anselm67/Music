#!/usr/bin/env python3
"""Tool to make and manage the music dataset from PDMX.

PDMX Main repo is https://zenodo.org/records/14648209

"""
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import click
import cv2

from dataset import PDMX, Score
from dataset.layout import Page
from kern import KernReader
from utils import print_histogram
from verovio import mxl_to_kern
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
@click.option("--log-file", type=click.Path(file_okay=True, writable=True, path_type=Path),
              help="Name of pdmx's log file.")
@click.option("--csv", default="PDMX.csv", show_default=True,
              help="Name of the .csv master file.")
@click.option("--log-level", default="INFO",
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False))
@click.option("--count", "-n", type=int, default=-1, show_default="all",
              help="How many rows of the dataset should we consider.")
@click.option("--offset", "-o", type=int, default=-1, show_default="start",
              help="Offset at which to start picking rows from the dataset.")
@click.pass_context
def cli(ctx, home: Path, csv: str, log_file: None | Path, log_level: str, offset: int, count: int):
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        filename=log_file,
        format="%(asctime)s | %(levelname)s | %(module)s.%(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    pdmx = PDMX(home, csv, offset, count)
    ctx.obj = ClickContext(home=home, pdmx=pdmx)


@click.command()
@click.argument("query_string")
@click.option("--metadata", "-m", type=str, default=None,
              help="Metadata query as a json filter.")
@click.option("--score", "-s", type=str, default=None,
              help="Score query as a json filter.")
@click.option("--columns", "-c", multiple=True,
              help="Names of columns to display")
@click.option("--output", "-o",
              type=click.Path(dir_okay=False, file_okay=True, path_type=Path),
              default=None,
              help="Save the filtered set to the given output file.")
@click.pass_obj
def query(ctx: ClickContext, query_string: str, metadata: str | None, score: str | None, columns: tuple[str], output: Path | None):
    """Query the underlying PDMX.csv database as a DataFrame.

    QUERY_STRING: The base panda query to run.
    """
    result = ctx.pdmx.query(query_string, metadata=metadata, score=score)
    if columns:
        result = result[list(columns)]
    if output is not None:
        result.to_csv(output, index=False)
    else:
        print(result.to_string())


@click.command()
@click.argument("mxl_file",
                type=click.Path(dir_okay=False, file_okay=True,
                                exists=True, readable=True, path_type=Path),
                required=True)
@click.option("--output", "-o",
              type=click.Path(dir_okay=False, file_okay=True, path_type=Path))
def render(mxl_file: Path, output: Path):
    """Renders MXL_FILE into .svg files, one per page."""
    svg_file = output or mxl_file.with_suffix(".svg")
    verovio_render(mxl_file, svg_file)
    print(f"Output written to {svg_file}.")


@click.command()
@click.argument("mxl_file",
                type=click.Path(dir_okay=False, file_okay=True,
                                exists=True, readable=True, path_type=Path),
                required=True)
@click.option("--output", "-o",
              type=click.Path(dir_okay=False, file_okay=True, path_type=Path),
              help="Output file, defaults to the mxl file with .krn extension.")
def from_mxl(mxl_file: Path, output: Path):
    """Converts MXL_FILE into a kern file.

        MXL_FILE: The mxl file to convert to kern.
    """
    output = output or mxl_file.with_suffix(".krn")
    mxl_to_kern(mxl_file, output)
    print(f"Output written to {output}.")


@dataclass
class ClickParams:
    page: Page
    kern_reader: KernReader


def mouse_positon_handler(event, x, y, _, click_params):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    print(f"x: {x}, y: {y}")
    # Converts the (x, y) into a system, staff and bar to highlight.
    params = cast(ClickParams, click_params)
    for system_index, system in enumerate(params.page.systems):
        if system.box.contains((x, y)):
            for staff_index, staff in enumerate(system.staves):
                if staff.box.contains((x, y)):
                    bar_number = system.bar_number
                    bar_count = 0
                    while bar_count+1 < len(staff.bars) and staff.bars[bar_count+1] < x:
                        bar_count += 1
                    print(
                        f"page number: {params.page.page_number}\n"
                        f"     system: {system_index}\n"
                        f"      staff: {staff_index}\n"
                        f" bar number: {bar_number + bar_count}"
                    )
                    tokens = params.kern_reader.get_text(
                        bar_number + bar_count)
                    if tokens:
                        for line in tokens:
                            print(line)
                    else:
                        print("*** No matching tokens.")
                    return
    print("Click on a bar to get its coordinates.")


@click.command()
@click.argument("any_path",
                type=click.Path(dir_okay=False, file_okay=True,
                                readable=True, path_type=Path),
                required=True)
@click.option("--scale", "-s", default=0.8, show_default=True,
              help="Resize scale of image and structure for display.")
@click.pass_obj
def show(ctx: ClickContext, any_path: Path, scale: float):
    """Displays the provided image and layout info when available.


        ANY_PATH: Any file that refers to a PDMX item, e.g. its mxl or svg path.
    """
    pdmx = ctx.pdmx
    with open(pdmx.get_path(any_path, 'layout'), 'r') as f:
        obj = json.load(f)
    score = Score.from_json(obj)
    kern_reader = KernReader(pdmx.get_path(any_path, 'tokens'))
    page_index = 0
    hide_truth: bool = False
    while True:
        page = score.pages[page_index]
        # Loads the page image.
        if score.page_count > 1:
            img_path = pdmx.get_page_path(any_path, 'png', page.page_number)
        else:
            img_path = pdmx.get_path(any_path, 'png')
        img = cv2.imread(img_path)
        assert img is not None, f"Can't load image {img_path}"

        # Resizes it according to provided scale.
        # We could drawinto the un-resized image and then resize, but resizing
        # them separately allows to check that Score.resize() works fine.
        (height, width) = tuple(map(lambda x: int(x * scale), img.shape[:2]))
        img = cv2.resize(img, (width, height))
        page = page.resize(width, height)

        # Renders the page layout on top of the image.
        if not hide_truth:
            print(
                f"Page {page_index+1}: {page.image_width} x {page.image_height} px {len(page.systems)} systems...")
            for index, system in enumerate(page.systems):
                print(
                    f"\tsystem {index+1}: {len(page.systems[0].staves)} staves.")
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
        cv2.imshow("layout", img)
        cv2.setMouseCallback(
            "layout",
            mouse_positon_handler,
            param=ClickParams(page, kern_reader)
        )

        if (key := cv2.waitKey()) == ord('q'):
            break
        elif key == ord('p'):
            if (page_index := page_index - 1) < 0:
                page_index = len(score.pages) - 1
        elif key == ord('n'):
            if (page_index := page_index + 1) >= score.page_count:
                page_index = 0
        elif key == ord('i'):
            mxl_file = pdmx.get_path(any_path, 'mxl')
            if (infos := pdmx.info(mxl_file)) is None:
                print(f"{mxl_file}: not found.")
            else:
                for title, value in infos:
                    print(f"{title}\n\t\033[1;31m{value}\033[0m")
        elif key == ord('h'):
            hide_truth = not hide_truth
        else:
            print(
                "(p)revious page,\n"
                "(n)ext page,\n"
                "(i)nfos about tghe score,\n"
                "(h)ide/show ground truth boxes."
            )

    cv2.destroyAllWindows()


@click.command()
@click.option("--force", "-f", default=False, is_flag=True, show_default=True,
              help="Recomputes all dependent files even if they're newer than their mxl source.")
@click.option("--dry-run", "-n", default=False, is_flag=True, show_default=True,
              help="Say what you'd do but don't do it.")
@click.option("--num-workers", type=int, default=None,
              help="Number of workers for the dataset loader.")
@click.argument("mxl_file",
                type=click.Path(dir_okay=False, file_okay=True,
                                path_type=Path),
                required=False, default=None)
@click.pass_obj
def make(ctx: ClickContext, mxl_file: Path | None, force: bool, dry_run: bool, num_workers: int | None):
    """Computes all dependent files from PDMX mxl files.

    \b
    For a given mxl file, this includes:
    - The corresponding humdrum kern file,
    - Verovio rendering of the mxl file as a set of svg files - one per page.
    - For each svg file, the corresponding png file.
    - For each mxl file, it's layout infos as page, system, staves and bars.


    MXL_FILE: Optional; When provided only this item is checked and rebuild.
    """
    pdmx = ctx.pdmx
    # Resolves relative path if needed.
    if mxl_file is not None:
        mxl_file = pdmx.get_path(mxl_file, 'mxl')
    pdmx.make(mxl_file, force=force, dry_run=dry_run, num_workers=num_workers)


@click.command()
@click.option("--num-workers", type=int, default=None,
              help="Number of workers for the dataset loader.")
@click.pass_obj
def stats(ctx: ClickContext, num_workers: int | None):
    """Computes layout statistics for the PDMX dataset.
    """
    stats = ctx.pdmx.stats(num_worker=num_workers)
    print(f"{stats.layout_count:,} layout files, {stats.score_count:,} scores:")
    print(f"  Page count: {stats.page_count:,}")
    print(f"System count: {stats.system_count:,}")
    print(f" Staff count: {stats.staff_count:,}")
    print(f"   Bar count: {stats.bar_count}:,")

    print_histogram(stats.part_histo, title="Parts per score:")

    print_histogram(stats.system_histo, title="Systems per page:")
    print_histogram(stats.staff_histo, title="Staves per page:")
    print_histogram(stats.width100_histo, title="Page widths:")
    print_histogram(stats.height100_histo, title="Page heights:")

    print_histogram(stats.bar_histo, title="Bars per system:")


@click.command()
@click.argument("mxl_file",
                type=click.Path(dir_okay=False, file_okay=True,
                                path_type=Path),
                required=False, default=None)
@click.pass_obj
def info(ctx: ClickContext, mxl_file: Path):
    """Displays all infos about a given PDMX score.

    MXL_FILE: Optional; When provided only this item is checked and rebuild.
    """
    pdmx = ctx.pdmx
    # Resolves relative path if needed.
    if mxl_file is not None:
        mxl_file = pdmx.get_path(mxl_file, 'mxl')
    if (infos := pdmx.info(mxl_file)) is None:
        print(f"{mxl_file}: not found.")
    else:
        for title, value in infos:
            print(f"{title}\n\t\033[1;31m{value}\033[0m")


cli.add_command(query)
cli.add_command(render)
cli.add_command(from_mxl)
cli.add_command(show)
cli.add_command(make)
cli.add_command(stats)
cli.add_command(info)
# TODO Add a validate command to check that all bars within a system are the same.


def main():
    cli()


if __name__ == "__main__":
    main()
