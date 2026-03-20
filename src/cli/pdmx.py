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
@click.pass_obj
def filter(ctx: ClickContext):
    ctx.pdmx.filter()


cli.add_command(filter)


def main():
    logging.basicConfig(level=logging.INFO)
    cli()


if __name__ == "__main__":
    main()
