#!/usr/bin/env python3
import logging
import math
import random
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import click
import cv2
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStoppingReason
from lightning.pytorch.loggers import CSVLogger
from torch import Tensor
from torch.utils.data import DataLoader
from torchinfo import summary as model_summary
from torchvision.io import decode_image

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
              type=click.Choice(["DEBUG", "INFO", "WARNING",
                                "ERROR"], case_sensitive=False),
              help="Select a logging level.")
@click.option("--log-file", type=click.Path(file_okay=True, writable=True, path_type=Path),
              help="Name of staffer's log file.")
@click.option("--home", "-h", type=click.Path(dir_okay=True, file_okay=False,
                                              exists=True, readable=True,
                                              path_type=Path),
              default=HOME, show_default=True,
              help="Root directory of the PDMX dataset.")
@click.option("--csv", default="Staff16.csv", show_default=True,
              help="Name of the .csv master file.")
@click.option("--count", "-n", type=int, default=-1, show_default="all",
              help="How many rows of the dataset should we consider.")
@click.option("--offset", "-o", type=int, default=-1, show_default="start",
              help="Offset at which to start picking rows from the dataset.")
@click.pass_context
def cli(ctx, log_level: str, log_file: None | Path, home: Path, csv: str, offset: int, count: int):
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        filename=log_file,
        format="%(asctime)s | %(levelname)s | %(module)s.%(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    pdmx = PDMX(home, csv, offset, count)
    ctx.obj = ClickContext(Config(), home, pdmx)


@click.command()
def summary():
    """Displays a nice summary of the underlying HierarchicalDETR model."""
    config = Config()
    model = HierarchicalDETR(config)
    model_summary(model, input_size=(config.batch_size,
                  config.in_channels, *config.image_shape))


@click.command()
def check():
    """Checks the model input / output dimensions."""
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
    dataset = StafferDataset(ctx.config, ctx.pdmx)
    while True:
        index = random.randint(0, len(dataset) - 1)
        img, sys, staff, assign = dataset[index]
        img = img.squeeze(0).cpu().numpy()
        img = np.stack([img] * 3, axis=-1) * 255
        width_height = img.shape[1], img.shape[0]
        # Try to make sense of the ground truth data.
        for sys_index in range(sys.shape[0]):
            cx, cy, w, h = tuple(map(lambda x: x.item(), sys[sys_index]))
            box = Box.from_cxcywh(width_height, cx, cy, w, h)
            cv2.rectangle(img, box.top_left, box.bot_right, (255, 0, 0), 2)
        for staff_index in range(assign.shape[0]):
            if assign[staff_index] < 0:
                break
            cx, cy, w, h = tuple(map(lambda x: x.item(), staff[staff_index]))
            box = Box.from_cxcywh(width_height, cx, cy, w, h)
            cv2.rectangle(img, box.top_left, box.bot_right, (0, 0, 255), 1)
        print(f"Image size: {img.shape}")
        print(f"    Assign: {assign}")
        cv2.imshow("Page", img)

        if cv2.waitKey(0) == ord('q'):
            return


@click.command()
@click.option("--num-workers", type=int, default=8,
              help="Number of workers for the dataset loader.")
@click.pass_obj
def stats(ctx: ClickContext, num_workers: int):
    """Computes the mean and std of a subset of images from the dataset.."""
    ds = StafferDataset(ctx.config, ctx.pdmx)
    loader = DataLoader[tuple[Tensor, Tensor, Tensor, Tensor]](
        ds, num_workers=num_workers, batch_size=ctx.config.batch_size
    )
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
    print(f"Scanned {len(ds)} images.")
    print(f"mean: {mean}")
    print(f" std: {std}")


@click.command()
@click.argument("name", type=str)
@click.option("--hide-progress", "-h", type=bool, is_flag=True, default=False,
              help="Hide progress report, e.g. to see the logging info.")
@click.option("--early-stopping", "-s", type=click.FloatRange(min=0), default=0.0,
              help="Enable early stopping with a patience of this amount of an epoch.")
@click.option("--epochs", "-e", type=int, default=4,
              help="Numberof epochs to train for.")
@click.pass_obj
def train(ctx: ClickContext,
          name: str,
          hide_progress: bool,
          early_stopping: float,
          epochs: int):
    """Trains and/or resume training of a Staffer model instance.

    NAME: sets id/name of the model being trained.
    """
    VAL_CHECK_INTERVAL = 250

    # Resume training if we have an existing checkpoint.
    ckpt_path = Path("checkpoints") / "staffer" / name / "last.ckpt"
    if ckpt_path.exists():
        logging.info(f"Resuming training from {ckpt_path}")
        config = config_from_checkpoint(ckpt_path)
    else:
        ckpt_path = None
        config = replace(
            ctx.config,
            id_name=name,
        )
    config.max_steps = epochs * (config.train_len // config.batch_size)
    logging.info(f"Training for {epochs} epochs, "
                 f"or {config.max_steps} steps of {config.batch_size}.")
    early_stopping_callback = None
    if early_stopping > 0:
        steps = int(early_stopping * (config.train_len // config.batch_size))
        steps = steps // VAL_CHECK_INTERVAL
        logging.info(
            f"EarlyStopping: patience is {steps} validaton steps.")
        early_stopping_callback = EarlyStopping(
            monitor="val/loss",
            patience=steps,
            mode="min",
            min_delta=1e-4,  # Ignore "noise"
        )

    callbacks: list[Callback] = [callback for callback in [
        ModelCheckpoint(
            dirpath=f"checkpoints/staffer/{config.id_name}",
            filename="{epoch}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            save_on_train_epoch_end=True,
            save_on_exception=True
        ),
        early_stopping_callback
    ] if callback is not None]

    # Sets up the logger.
    logger_path = Path("logs/staffer") / config.id_name / "metrics.csv"
    if logger_path.exists():
        all_path = logger_path.with_stem("cumulated_metrics")
        if all_path.exists():
            all = pd.read_csv(all_path)
            new = pd.read_csv(logger_path)
            pd.concat([all, new], ignore_index=True).to_csv(
                all_path, index=False)
        else:
            shutil.copy(logger_path, all_path)
        logger_path.unlink()

    logger = CSVLogger(save_dir="logs", name="staffer", version=config.id_name)

    trainer = L.Trainer(
        max_steps=config.max_steps,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=100,
        val_check_interval=VAL_CHECK_INTERVAL,
        precision="16-mixed",
        enable_model_summary=False,
        enable_progress_bar=not hide_progress
    )

    # Following incantation required to load checkpoints.
    from torchvision.transforms.functional import InterpolationMode
    torch.serialization.add_safe_globals([InterpolationMode])

    trainer.fit(
        StafferModule(config),
        StafferDataModule(config, ctx.pdmx),
        ckpt_path=ckpt_path
    )

    if early_stopping_callback is not None and early_stopping_callback.stopping_reason != EarlyStoppingReason.NOT_STOPPED:
        logging.info(
            f"Early stopping: {early_stopping_callback.stopping_reason}")
        logging.info(
            f"       message: {early_stopping_callback.stopping_reason_message}")
        logging.info(
            f"         epoch: {early_stopping_callback.stopped_epoch}")


LOG_VARIABLES = [
    # From StafferModule._step():
    "loss",
    "lr",           # Training only.
    "stave_iou",
    "sys_iou",

    # From LossDict:
    "sys_box",
    "sys_giou",
    "sys_obj",
    "stave_box",
    "stave_giou",
    "stave_obj",
    "assign",
    "containment",
    "alignment"
]


def plot_one(ax_metrics: Any, name: str,
             columns: tuple[str, ...],
             ls='solid') -> None:
    csv_path = Path(f"logs/staffer/{name}/metrics.csv")
    all_path = csv_path.with_stem("cumulated_metrics")
    all_df = None
    if all_path.exists():
        all_df = pd.read_csv(all_path)

    if csv_path.exists():
        df = pd.read_csv(csv_path) if all_df is None else pd.concat(
            [all_df, pd.read_csv(csv_path)])
    elif all_df is not None:
        df = all_df
    else:
        raise click.UsageError(f"No metrics file found for {name}.")

    # Train and validation losses.
    labels = tuple(f"{name}:{col}" for col in columns)
    for col, label in zip(columns, labels):
        if col in df.columns:
            d = df[["step", col]].dropna()
            ax_metrics.plot(d["step"], d[col], label=label, ls=ls)


@click.command()
@click.argument("names", type=str, nargs=-1)
@click.option("--train-columns", "-t",
              type=click.Choice(LOG_VARIABLES, case_sensitive=False),
              multiple=True,
              help="Select one or more training metrics to plot.")
@click.option("--valid-columns", "-v",
              type=click.Choice(LOG_VARIABLES, case_sensitive=False),
              multiple=True,
              help="Select one or more validation metrics to plot.")
def logs(names: tuple[str], train_columns: tuple[str, ...], valid_columns: tuple[str, ...]):
    """Displays training logs from multiple experiments in a single graph.

    NAMES: List of the names of the model experiments you want graphed."""
    columns = (tuple(f"train/{s}" for s in train_columns)
               + tuple(f"val/{s}" for s in valid_columns))
    if len(columns) == 0:
        raise click.UsageError("Select at least one metric to plot.")
    (name, *others) = names
    csv_path = Path(f"logs/staffer/{name}/metrics.csv")

    plt.ion()
    fig, ax_metrics = plt.subplots(1, 1)

    def on_key(event):
        if event.key == 'q':
            plt.close("all")

    fig.canvas.mpl_connect('key_press_event', on_key)
    last_mod = 0

    while plt.get_fignums():
        mtime = csv_path.stat().st_mtime
        if mtime != last_mod:
            last_mod = mtime

            ax_metrics.cla()
            plot_one(ax_metrics, name, columns)
            for other in others:
                plot_one(ax_metrics, other, columns, ls='dashed')

            ax_metrics.set_title("Training metrics")
            ax_metrics.set_xlabel("step")
            ax_metrics.legend()

            fig.canvas.draw()

        plt.pause(5000)
    print("Bye!")


def config_from_checkpoint(checkpoint_path: Path) -> Config:
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    return Config(**checkpoint["hyper_parameters"])


def unbox(size: tuple[int, int], t: Tensor) -> Box:
    value = tuple(map(lambda x: x.item(), t))
    assert len(value) == 4, f"Box tensor expect shape [4], but got {t.shape}"
    cx, cy, w, h = value
    return Box.from_cxcywh(size, cx, cy, w, h)


@click.command()
@click.argument("name", type=str)
@click.argument("img_paths", nargs=-1,
                type=click.Path(file_okay=True,
                                exists=True, readable=True,
                                path_type=Path))
@click.pass_obj
def predict(ctx: ClickContext, name: str, img_paths: tuple[Path]) -> None:
    """Predicts bounding boxes for system and staves for a list of images.

    NAME: The model version to use to make the predictions.
    IMG_PATHS: The list of images to process and visualize.
    """
    ckpt_path = Path("checkpoints") / "staffer" / name / "last.ckpt"
    config = config_from_checkpoint(ckpt_path)
    dataset = StafferDataset(config, ctx.pdmx)
    model = StafferModule.load_from_checkpoint(
        ckpt_path, config=config, weights_only=False)
    model.eval()
    for img_path in img_paths:
        print(f"Path: {img_path.as_posix()}")
        img = decode_image(img_path.as_posix())
        img = dataset.transform(img).cuda()
        with torch.no_grad():
            (
                pred_sys_boxes, pred_sys_logits,
                pred_stave_boxes, pred_stave_logits,
                # (stave_query, system_query)
                pred_assign
            ) = tuple(map(lambda t: t.squeeze(0), model.forward(img.unsqueeze(0))))

        img = img.squeeze(0).cpu().numpy()
        img = np.stack([img] * 3, axis=-1) * 255
        width_height = img.shape[1], img.shape[0]
        # Try to make sense of the ground truth data.
        for sys_index in range(pred_sys_boxes.shape[0]):
            print(
                f"\tsystem[{sys_index}]: {pred_sys_logits[sys_index].item():.2f}")
            if pred_sys_logits[sys_index].item() > 0.0:
                box = unbox(width_height, pred_sys_boxes[sys_index])
                cv2.rectangle(img, box.top_left, box.bot_right, (0, 0, 255), 2)
        stave_assignment = torch.argmax(pred_assign, dim=1)
        for staff_index in range(pred_assign.shape[0]):
            print(
                f"\tstaff[{staff_index}]: {pred_stave_logits[staff_index].item():.2f}, system: {stave_assignment[staff_index].item()}")
            if pred_stave_logits[staff_index].item() > 0.0:
                box = unbox(width_height, pred_stave_boxes[staff_index])
                cv2.rectangle(img, box.top_left, box.bot_right, (0, 255, 0), 1)
        cv2.imshow("Page", img)

        if cv2.waitKey(0) == ord('q'):
            return
        cv2.destroyAllWindows()


cli.add_command(summary)
cli.add_command(check)
cli.add_command(show)
cli.add_command(stats)
cli.add_command(train)
cli.add_command(logs)
cli.add_command(predict)


def main():
    torch.set_float32_matmul_precision("high")
    cli()


if __name__ == '__main__':
    main()

# vscode - End of file
