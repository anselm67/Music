#!/usr/bin/env python3
import logging
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
        for batch_index in range(ctx.config.batch_size):
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


cli.add_command(summary)
cli.add_command(show)


def main():
    cli()


if __name__ == '__main__':
    main()
    main()
