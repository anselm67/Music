#!/usr/bin/env python3
"""Tool to make and manage the music dataset from PDMX.

PDMX Main repo is https://zenodo.org/records/14648209

"""
import logging
from dataclasses import dataclass
from pathlib import Path

import click

from dataset import PDMX, MxlEditor
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


@click.command()
@click.argument("any_path",
                type=click.Path(dir_okay=False, file_okay=True,
                                readable=True, path_type=Path),
                default=None)
@click.option("--scale", "-s", default=0.8, show_default=True,
              help="Resize scale of image and structure for display.")
@click.pass_obj
def show(ctx: ClickContext, any_path: Path | None, scale: float):
    """Displays the provided image and layout info when available.


        ANY_PATH: Any file that refers to a PDMX item, e.g. its mxl or svg path.
    """
    pdmx = ctx.pdmx
    mxl_path = None if any_path is None else pdmx.get_path(any_path, 'mxl')
    editor = MxlEditor(pdmx, mxl_path)
    editor.run()
    editor.close()


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
