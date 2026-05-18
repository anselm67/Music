#!/usr/bin/env python3
import logging
import math
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import click
import cv2
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PDMX
from noter import NoterConfig, NoterDataset, NoterModule, Vocab
from utils import print_histogram

HOME = Path("/home/anselm/datasets/PDMX")


@dataclass
class ClickContext:
    home: Path
    pdmx: PDMX
    config: NoterConfig


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
    default="System2.csv",
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
    ctx: click.Context,
    log_level: str,
    log_file: None | Path,
    home: Path,
    csv: str,
    offset: int,
    count: int,
) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        filename=log_file,
        format="%(asctime)s | %(levelname)s | %(module)s.%(funcName)s:%(lineno)d | %(message)s",  # noqa: E501
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.info("Running: %s", " ".join(sys.argv))
    pdmx = PDMX(home, csv, offset, count)
    ctx.obj = ClickContext(home, pdmx, NoterConfig())


@click.command()
@click.pass_obj
def vocab(ctx: ClickContext) -> None:
    """Generates the vocab pickle file from PDMX token files."""
    vocab = Vocab.from_files(ctx.home / "build" / "tokens")
    vocab.save(ctx.home / "build" / "vocab.json")


@click.command()
@click.pass_obj
def show(ctx: ClickContext) -> None:
    """Displays random samples from the dataset."""
    dataset = NoterDataset(ctx.config, ctx.pdmx)
    while True:
        index = random.randint(0, len(dataset) - 1)
        img_tensor, seq_tensor = dataset[index]
        img = img_tensor.squeeze(0).cpu().numpy()
        tokens = dataset.vocab.i2tok(seq_tensor)
        print(tokens)
        cv2.imshow("Staff", img)
        if cv2.waitKey(0) == ord("q"):
            break


@click.command()
@click.option(
    "--num-workers",
    type=int,
    default=8,
    help="Number of workers for the dataset loader.",
)
@click.pass_obj
def stats(ctx: ClickContext, num_workers: int) -> None:
    """Computes stats for images from the dataset."""
    # We can't use DataLoader just yet, because the image size isn't padded.
    dataset = NoterDataset(ctx.config, ctx.pdmx)
    max_height, max_width, max_seqlen = 0, 0, 0
    seqlen_histo: Counter[int] = Counter()
    for idx in tqdm(range(0, len(dataset)), desc="Computing stats"):
        try:
            (height, width), seqlen = dataset.get_item_stats(idx)
        except Exception as e:
            logging.error(f"get_item_stats({idx}): {e}")
            continue
        max_height = max(max_height, height)
        max_width = max(max_width, width)
        max_seqlen = max(max_seqlen, seqlen)
        seqlen_histo[seqlen] += 1
    print(f"Scanned {len(dataset)} items.")
    print(f"Image max size (w x h): {max_width} x {max_height}")
    print(f"      Sequence max len: {max_seqlen}")
    print_histogram(seqlen_histo, title="Sequence lengths:")


@click.command()
@click.option(
    "--num-workers",
    type=int,
    default=8,
    help="Number of workers for the dataset loader.",
)
@click.pass_obj
def image_stats(ctx: ClickContext, num_workers: int) -> None:
    """Computes the mean and std of a subset of images from the dataset.."""
    ds = NoterDataset(ctx.config, ctx.pdmx)
    loader = DataLoader[tuple[Tensor, Tensor]](
        ds, num_workers=num_workers, batch_size=ctx.config.batch_size
    )
    pix_sum = 0
    pix_sum2 = 0
    pix_count = 0
    for images, _ in loader:
        for batch_index in range(len(images)):
            img = images[batch_index].squeeze(0).cpu().numpy()
            pix_sum += img.sum()
            pix_sum2 += (img**2).sum()
            pix_count += img.shape[0] * img.shape[1]
    mean = pix_sum / pix_count
    std = math.sqrt(pix_sum2 / pix_count - mean**2)
    print(f"Scanned {len(ds)} images.")
    print(f"mean: {mean}")
    print(f" std: {std}")


@click.command()
@click.pass_obj
def summary(ctx: ClickContext) -> None:
    """Displays a nice summary of the underlying NoterModel model."""
    config = NoterConfig()
    config.use_vocab(Vocab.load(ctx.pdmx.home / "build/vocab.json"))
    B = config.batch_size
    T = config.max_seqlen
    H = config.max_chords

    # We can't use torchinfo.summary() because it fails on nested tensors used
    # by the TransformerDecoder.
    model = NoterModule(config)
    model.forward(
        torch.zeros(B, config.in_channels, *config.input_shape),  # source
        torch.full((B,), config.input_shape[1]),  # source_widths
        torch.zeros(B, T, H, dtype=torch.long),  # target
    )
    print(model)


cli.add_command(vocab)
cli.add_command(show)
cli.add_command(stats)
cli.add_command(image_stats)
cli.add_command(summary)


def main() -> None:
    torch.set_float32_matmul_precision("high")
    cli()


if __name__ == "__main__":
    main()
