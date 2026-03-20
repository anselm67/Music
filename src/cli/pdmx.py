#!/usr/bin/env python3
import logging
from dataclasses import dataclass
from pathlib import Path

import click

from dataset import PDMX

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


cli.add_command(query)


def main():
    logging.basicConfig(level=logging.INFO)
    cli()


if __name__ == "__main__":
    main()
