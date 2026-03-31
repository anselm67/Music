#!/usr/bin/env python3
import logging
import math
from dataclasses import dataclass, replace
from pathlib import Path

import click
import cv2
import lightning as L
import matplotlib.pyplot as plt
import pandas as pd
import torch
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch import Tensor
from torch.utils.data import DataLoader
from torchinfo import summary as model_summary

from dataset import PDMX, Box, StafferDataModule, StafferDataset
from models import Config, HierarchicalDETR, StafferModule

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
def check():
    config = Config()
    model = HierarchicalDETR(config)

    model.eval()
    x = torch.randn(config.batch_size, config.in_channels, *config.image_shape)
    print(f"input:          {x.shape}")

    with torch.no_grad():
        sys_boxes, sys_logits, stave_boxes, stave_logits, assign_logits = model(
            x)

    print(f"sys_boxes:      {sys_boxes.shape}")       # (B, 16, 4)
    print(f"sys_logits:     {sys_logits.shape}")      # (B, 16, 1)
    print(f"stave_boxes:    {stave_boxes.shape}")     # (B, 16, 4)
    print(f"stave_logits:   {stave_logits.shape}")    # (B, 16, 1)
    print(f"assign_logits:  {assign_logits.shape}")   # (B, 16, 16)


@click.command()
@click.pass_obj
def show(ctx: ClickContext):
    """Displays random samples from the dataset."""
    ds = StafferDataset(ctx.config, ctx.pdmx, sample_count=10)
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


@click.command()
@click.argument("name", type=str)
@click.option("--max-steps", "-s", type=int, default=25_000,    # B=8, N=50,000 -> 4 epochs
              help="Number of training steps to run.")
@click.option("--sample-count", "-c", type=int, default=50_000,
              help="Number of sample to use from the dataset.")
@click.option("--hide-progress", "-h", type=bool, is_flag=True, default=False,
              help="Hide progress report, e.g. to see the logging info.")
@click.pass_obj
def train(ctx: ClickContext, name: str, max_steps: int, sample_count: int, hide_progress: bool):
    """Trains the Staffer model.

    NAME: sets id/name of the model being trained.
    """
    config = replace(
        ctx.config,
        id_name=name,
        sample_count=sample_count,
        max_steps=max_steps
    )

    data_module = StafferDataModule(config, ctx.pdmx)
    model = StafferModule(config)

    logger = CSVLogger(save_dir="logs", name="staffer", version=config.id_name)

    callbacks: list[Callback] = [
        ModelCheckpoint(
            dirpath=f"checkpoints/{config.id_name}",
            filename="{epoch}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True
        ),
        EarlyStopping(
            monitor="val/loss",
            patience=5,
            mode="min",
        ),
    ]

    # Resume training if we have an existing checkpoint.
    last_checkpoint: Path | None = Path(
        f"checkpoints/{config.id_name}/last.ckpt")
    if last_checkpoint.exists():
        logging.info(f"Resuming training from {last_checkpoint}")
    else:
        last_checkpoint = None

    trainer = L.Trainer(
        max_steps=max_steps,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=50,
        val_check_interval=250,
        precision="16-mixed",
        enable_progress_bar=not hide_progress
    )

    # Following incantation required to load checkpoints.
    from torchvision.transforms.functional import InterpolationMode
    torch.serialization.add_safe_globals([InterpolationMode])

    trainer.fit(
        model,
        data_module,
        ckpt_path=last_checkpoint
    )


@click.argument("name", type=str)
@click.command()
def logs(name: str):
    csv_path = Path(f"logs/staffer/{name}/metrics.csv")
    if not csv_path.exists():
        logging.error(f"No csv log {str(csv_path)}, bye !")
    plt.ion()
    fig, (ax_loss, ax_metrics) = plt.subplots(1, 2, figsize=(16, 6))

    def on_key(event):
        if event.key == 'q':
            plt.close("all")

    fig.canvas.mpl_connect('key_press_event', on_key)
    last_mod = 0

    while plt.get_fignums():
        mtime = csv_path.stat().st_mtime
        if mtime != last_mod:
            last_mod = mtime
            df = pd.read_csv(csv_path)

            # Train and validation losses.
            ax_loss.cla()
            for col, label in [("train/loss", "train"), ("val/loss", "val")]:
                if col in df.columns:
                    d = df[["step", col]].dropna()
                    ax_loss.plot(d["step"], d[col], label=label)
            ax_loss.set_title("loss")
            ax_loss.set_xlabel("step")
            ax_loss.legend()

            # Validation metrics: containment, stave_iou and sys_iou.
            ax_metrics.cla()
            for col, label in [
                ("val/containment", "containment"),
                ("val/stave_iou", "stave iou"),
                ("val/sys_iou", "sys iou"),
            ]:
                if col in df.columns:
                    d = df[["step", col]].dropna()
                    ax_metrics.plot(d["step"], d[col], label=label)
            ax_metrics.set_title("val metrics")
            ax_metrics.set_xlabel("step")
            ax_metrics.legend()

            fig.tight_layout()
            fig.canvas.draw()

        plt.pause(5)
    print("Bye!")


cli.add_command(summary)
cli.add_command(check)
cli.add_command(show)
cli.add_command(stats)
cli.add_command(train)
cli.add_command(logs)


def main():
    torch.set_float32_matmul_precision("high")
    cli()


if __name__ == '__main__':
    main()
    main()
    main()
