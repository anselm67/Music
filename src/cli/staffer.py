#!/usr/bin/env python3
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import click
import cv2
from torch import Tensor
from torch.utils.data import DataLoader
from torchinfo import summary as model_summary

from dataset import PDMX, Box, StafferDataset
from models import Config, HierarchicalDETR

HOME = Path("/home/anselm/datasets/PDMX")


@dataclass
class ClickContext:
    config: Config
    home: Path
    pdmx: PDMX


@click.group()
@click.option("--log-level", default="INFO",
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False))
@click.option("--home", "-h", type=click.Path(dir_okay=True, file_okay=False,
                                              exists=True, readable=True,
                                              path_type=Path),
              default=HOME, show_default=True)
@click.option("--csv", default="Staff16.csv", show_default=True,
              help="Name of the .csv master file.")
@click.pass_context
def cli(ctx, log_level: str, home: Path, csv: str):
    logging.basicConfig(level=getattr(logging, log_level.upper()))
    pdmx = PDMX(home, csv)
    ctx.obj = ClickContext(Config(), home, pdmx)


@click.command()
def summary():
    config = Config()
    model = HierarchicalDETR(config)
    model_summary(model, input_size=(config.batch_size,
                  config.in_channels, *config.image_shape))


@click.command()
@click.pass_obj
def show(ctx: ClickContext):
    """Displays random samples from the dataset."""
    ds = StafferDataset(ctx.config, ctx.pdmx, count=10)
    loader = DataLoader[tuple[Tensor, Tensor, Tensor, Tensor]](
        ds, num_workers=1, batch_size=ctx.config.batch_size
    )
    for images, syses, staves, assigns in loader:
        for batch_index in range(len(images)):
            img, sys, staff, assign = images[batch_index], syses[batch_index], staves[batch_index], assigns[batch_index]
            img = img.squeeze(0).cpu().numpy()
            (height, width) = img.shape
            # Try to make sense of the ground truth data.
            for sys_index in range(sys.shape[0]):
                cx, cy, w, h = sys[sys_index]
                box = Box.from_cxcywh((width, height), cx, cy, w, h)
                cv2.rectangle(img, box.top_left, box.bot_right, 0, 2)
            for staff_index in range(assign.shape[0]):
                if assign[staff_index] < 0:
                    break
                cx, cy, w, h = staff[staff_index]
                box = Box.from_cxcywh((width, height), cx, cy, w, h)
                cv2.rectangle(img, box.top_left, box.bot_right, 0, 2)
            print(f"Image size: {img.shape}")
            print(f"    Assign: {assign}")
            cv2.imshow("Page", img)

            if cv2.waitKey(0) == ord('q'):
                return


@click.command()
@click.option("--count", "-c", type=int, default=50000,
              help="Number of samples to pick to compute the stats.")
@click.option("--num-workers", type=int, default=8,
              help="Number of workers for the dataset loader.")
@click.pass_obj
def stats(ctx: ClickContext, count: int, num_workers: int):
    """Computes the mean and std of a subset of images from the dataset.."""
    ds = StafferDataset(ctx.config, ctx.pdmx)
    loader = DataLoader[tuple[Tensor, Tensor, Tensor, Tensor]](
        ds, num_workers=num_workers, batch_size=ctx.config.batch_size
    )
    scanned = count
    pix_sum = 0
    pix_sum2 = 0
    pix_count = 0
    for images, _, _, _ in loader:
        if count <= 0:
            break
        for batch_index in range(len(images)):
            img = images[batch_index].squeeze(0).cpu().numpy()
            pix_sum += img.sum()
            pix_sum2 += (img**2).sum()
            pix_count += img.shape[0] * img.shape[1]
            count -= 1
            if count <= 0:
                break
        if count % 100 == 0:
            logging.info(f"{count} left.")
    mean = pix_sum / pix_count
    std = math.sqrt(pix_sum2 / pix_count - mean ** 2)
    print(f"Scanned {scanned} images.")
    print(f"mean: {mean}")
    print(f" std: {std}")


cli.add_command(summary)
cli.add_command(show)
cli.add_command(stats)


def main():
    cli()


if __name__ == '__main__':
    main()
