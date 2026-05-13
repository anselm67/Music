#!/usr/bin/env python3
"""CLI to manipulate kern files.
"""
import logging
from dataclasses import dataclass
from pathlib import Path

import click

from kern import EmptyHandler, Parser, to_midi
from kern import tokenize as kern_tokenize


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
                print("done")
        except Exception as e:
            failed_count += 1
            if not ctx.silent:
                print(f"failed: {e}")
    print(f"{len(files)} parsed, {failed_count} failed.")


@click.command()
@click.argument("kern_file",
                type=click.Path(dir_okay=False, file_okay=True,
                                exists=True, readable=True, path_type=Path),
                required=True)
@click.option("--output", "-o",
              type=click.Path(dir_okay=False, file_okay=True, path_type=Path))
@click.option("--tempo", "-t", type=click.IntRange(1, 279), default=60)
@click.pass_obj
def midi(ctx: ClickContext, kern_file: Path, output: Path, tempo: int):
    """Converts a kern file to midi."""
    output = output or kern_file.with_suffix(".mid")
    to_midi(kern_file, output, tempo=tempo)


@click.command()
@click.argument("kern_file",
                type=click.Path(dir_okay=False, file_okay=True,
                                exists=True, readable=True, path_type=Path),
                required=True)
@click.option("--output", "-o",
              type=click.Path(dir_okay=False, file_okay=True, path_type=Path),
              default=None)
@click.pass_obj
def tokenize(ctx: ClickContext, kern_file: Path, output: Path | None):
    """Tokenize a kern file into a unique normal form."""
    kern_tokenize(kern_file, output, enable_warnings=not ctx.silent)


cli.add_command(validate)
cli.add_command(midi)
cli.add_command(tokenize)


def main():
    logging.basicConfig(level=logging.INFO)
    cli()


if __name__ == "__main__":
    main()

# vscode - End of File
