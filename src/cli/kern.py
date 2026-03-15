#!/usr/bin/env python3
import logging
from dataclasses import dataclass
from pathlib import Path

import click

from kern.empty import EmptyHandler
from kern.parser import Parser


@dataclass
class ClickContext:
    silent: bool = False


@click.group()
@click.option("--silent", "-s", is_flag=True, default=False)
@click.pass_context
def cli(ctx, silent: bool):
    ctx.ensure_object(ClickContext)
    ctx.obj = ClickContext(silent=silent)


@click.command()
@click.argument("files", nargs=-1,
                type=click.Path(dir_okay=False, file_okay=True,
                                exists=True, readable=True),
                required=True)
@click.pass_obj
def validate(ctx: ClickContext, files: list[Path]):
    """Parse and validate one or more kern files, reporting any errors found."""
    failed_count = 0
    for file in files:
        if not ctx.silent:
            print(f"{file}...", end="", flush=True)
        try:
            parser = Parser.from_file(file, EmptyHandler())
            parser.parse()
            if not ctx.silent:
                print(f"done")
        except Exception as e:
            failed_count += 1
            if not ctx.silent:
                print(f"failed: {e}")
    print(f"{len(files)} parsed, {failed_count} failed.")


cli.add_command(validate)


def main():
    logging.basicConfig(level=logging.INFO)
    cli()


if __name__ == "__main__":
    cli()
if __name__ == "__main__":
    cli()
