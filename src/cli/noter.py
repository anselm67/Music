#!/usr/bin/env python3
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import click
import cv2
import torch

from dataset import PDMX, NoterDataset, Vocab
from models import Config

HOME = Path("/home/anselm/datasets/PDMX")


@dataclass
class ClickContext:
    home: Path
    pdmx: PDMX
    config: Config


@click.group()
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    help="Select a logging level.",
)
@click.option(
    "--log-file",
    type=click.Path(file_okay=True, writable=True, path_type=Path),
    help="Name of staffer's log file.",
)
@click.option(
    "--home",
    "-h",
    type=click.Path(
        dir_okay=True, file_okay=False, exists=True, readable=True, path_type=Path
    ),
    default=HOME,
    show_default=True,
    help="Root directory of the PDMX dataset.",
)
@click.option(
    "--csv",
    default="Staff16.csv",
    show_default=True,
    help="Name of the .csv master file.",
)
@click.option(
    "--count",
    "-n",
    type=int,
    default=-1,
    show_default="all",
    help="How many rows of the dataset should we consider.",
)
@click.option(
    "--offset",
    "-o",
    type=int,
    default=-1,
    show_default="start",
    help="Offset at which to start picking rows from the dataset.",
)
@click.pass_context
def cli(
    ctx,
    log_level: str,
    log_file: None | Path,
    home: Path,
    csv: str,
    offset: int,
    count: int,
):
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        filename=log_file,
        format="%(asctime)s | %(levelname)s | %(module)s.%(funcName)s:%(lineno)d | %(message)s",  # noqa: E501
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    pdmx = PDMX(home, csv, offset, count)
    ctx.obj = ClickContext(home, pdmx, Config())


@click.command()
@click.pass_obj
def vocab(ctx: ClickContext):
    """Generates the vocab pickle file from PDMX token files."""
    vocab = Vocab.from_files(ctx.home / "build" / "tokens")
    vocab.save(ctx.home / "build" / "vocab.json")


@click.command()
@click.pass_obj
def show(ctx: ClickContext):
    """Displays random samples from the dataset."""
    dataset = NoterDataset(ctx.config, ctx.pdmx)
    vocab = Vocab.load(ctx.home / "build/vocab.json")
    while True:
        index = random.randint(0, len(dataset) - 1)
        img_tensor, seq_tensor = dataset[index]
        img = img_tensor.squeeze(0).cpu().numpy()
        print(f"     Image size: {img.shape}")
        print(f"Sequence length: {seq_tensor.shape}")
        tokens = vocab.i2tok(seq_tensor)
        print(tokens)
        cv2.imshow("Staff", img)
        if cv2.waitKey(0) == ord("q"):
            break


cli.add_command(vocab)
cli.add_command(show)


def main():
    torch.set_float32_matmul_precision("high")
    cli()


if __name__ == "__main__":
    main()
