#!/usr/bin/env python3
import logging
import math
import random
import shutil
import sys
from collections.abc import Iterable
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, cast

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
from matplotlib.backend_bases import Event, KeyEvent
from torch import Tensor
from torch.utils.data import DataLoader
from torchinfo import summary as model_summary
from torchvision.io import decode_image
from tqdm import tqdm

from pdmx import PDMX, Box
from staffer import (
    StafferConfig,
    StafferModel,
    StafferDataModule,
    StafferDataset,
    StafferModule,
    assign_staves,
)

HOME = Path("/home/anselm/datasets/PDMX")


@dataclass
class ClickContext:
    config: StafferConfig
    home: Path
    pdmx: PDMX


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
    logging.getLogger("lightning.pytorch").handlers = []
    if log_file is not None:
        logging.getLogger("lightning.pytorch").addHandler(
            logging.FileHandler(log_file.as_posix())
        )
    logging.info("Running: %s", " ".join(sys.argv))
    pdmx = PDMX(home, csv, offset, count)
    ctx.obj = ClickContext(StafferConfig(), home, pdmx)


@click.command()
def summary() -> None:
    """Displays a nice summary of the StafferModel architecture."""
    config = StafferConfig()
    model = StafferModel(config)
    model_summary(
        model, input_size=(config.batch_size, config.in_channels, *config.image_shape)
    )


@click.command()
def check() -> None:
    """Checks the model input / output dimensions."""
    config = StafferConfig()
    model = StafferModel(config)

    model.eval()
    x = torch.randn(config.batch_size, config.in_channels, *config.image_shape)
    print(f"input:          {x.shape}")

    with torch.no_grad():
        stave_tb, stave_logits, boundary_logits, sys_lr, sys_logits = model(x)

    print(f"stave_tb:       {stave_tb.shape}")  # (B, 16, 2) — top, bottom
    print(f"stave_logits:   {stave_logits.shape}")  # (B, 16, 1)
    print(f"boundary_logits:{boundary_logits.shape}")  # (B, 16, 1)
    print(f"sys_lr:         {sys_lr.shape}")  # (B, 16, 2) — left, right
    print(f"sys_logits:     {sys_logits.shape}")  # (B, 16, 1)


@click.command()
@click.pass_obj
def show(ctx: ClickContext) -> None:
    """Displays random samples from the dataset."""
    dataset = StafferDataset(ctx.config, ctx.pdmx)
    cv2.namedWindow("Page")
    while True:
        index = random.randint(0, len(dataset) - 1)
        img_tensor, sys, staff, assign = dataset[index]
        img = img_tensor.squeeze(0).cpu().numpy()
        img = np.stack([img] * 3, axis=-1) * 255
        width_height = img.shape[1], img.shape[0]
        # Try to make sense of the ground truth data.
        for sys_index in range(sys.shape[0]):
            box = unbox(width_height, sys[sys_index])
            cv2.rectangle(img, box.top_left, box.bot_right, (255, 0, 0), 2)
        for staff_index in range(assign.shape[0]):
            if assign[staff_index] < 0:
                break
            box = unbox(width_height, staff[staff_index])
            cv2.rectangle(img, box.top_left, box.bot_right, (0, 0, 255), 1)
        print(f"Image size: {img.shape}")
        print(f"    Assign: {assign}")
        cv2.imshow("Page", img)

        if cv2.waitKey(0) == ord("q"):
            break
    cv2.destroyAllWindows()


@click.command()
@click.option(
    "--num-workers",
    type=int,
    default=8,
    help="Number of workers for the dataset loader.",
)
@click.pass_obj
def stats(ctx: ClickContext, num_workers: int) -> None:
    """Computes the mean and std of a subset of images from the dataset.."""
    ds = StafferDataset(ctx.config, ctx.pdmx)
    loader = DataLoader[tuple[Tensor, Tensor, Tensor, Tensor]](
        ds, num_workers=num_workers, batch_size=ctx.config.batch_size
    )
    pix_sum = 0
    pix_sum2 = 0
    pix_count = 0
    for images, _, _, _ in loader:
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
@click.argument("name", type=str)
@click.option(
    "--hide-progress",
    "-h",
    type=bool,
    is_flag=True,
    default=False,
    help="Hide progress report, e.g. to see the logging info.",
)
@click.option(
    "--early-stopping",
    "-s",
    type=click.FloatRange(min=0),
    default=0.0,
    help="Enable early stopping with a patience of this amount of an epoch.",
)
@click.option(
    "--epochs", "-e", type=int, default=4, help="Numberof epochs to train for."
)
@click.option(
    "--use-sampler",
    type=bool,
    is_flag=True,
    default=False,
    help="Use a weighted sampler loader to equalize the distribution.",
)
@click.option(
    "--num-workers",
    type=int,
    default=8,
    help="Number of workers for the dataset loader.",
)
@click.pass_obj
def train(
    ctx: ClickContext,
    name: str,
    hide_progress: bool,
    early_stopping: float,
    epochs: int,
    use_sampler: bool,
    num_workers: int,
) -> None:
    """Trains and/or resume training of a Staffer model instance.

    NAME: sets id/name of the model being trained.
    """
    VAL_CHECK_INTERVAL = 250

    # Resume training if we have an existing checkpoint.
    ckpt_path: Path | None = None
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
    logging.info(
        f"Training for {epochs} epochs, "
        f"or {config.max_steps} steps of {config.batch_size}."
    )
    early_stopping_callback = None
    if early_stopping > 0:
        steps = int(early_stopping * (config.train_len // config.batch_size))
        steps = steps // VAL_CHECK_INTERVAL
        logging.info(f"EarlyStopping: patience is {steps} validaton steps.")
        early_stopping_callback = EarlyStopping(
            monitor="val/loss",
            patience=steps,
            mode="min",
            min_delta=1e-4,  # Ignore "noise"
        )

    callbacks: list[Callback] = [
        callback
        for callback in [
            ModelCheckpoint(
                dirpath=f"checkpoints/staffer/{config.id_name}",
                filename="{epoch}-{val/loss:.4f}",
                monitor="val/loss",
                mode="min",
                save_top_k=3,
                save_last=True,
                save_on_train_epoch_end=True,
                save_on_exception=True,
            ),
            early_stopping_callback,
        ]
        if callback is not None
    ]

    # Sets up the logger.
    logger_path = Path("logs/staffer") / config.id_name / "metrics.csv"
    if logger_path.exists():
        all_path = logger_path.with_stem("cumulated_metrics")
        if all_path.exists():
            prev_df = pd.read_csv(all_path)
            new_df = pd.read_csv(logger_path)
            pd.concat([prev_df, new_df], ignore_index=True).to_csv(
                all_path, index=False
            )
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
        precision="bf16-mixed",
        enable_model_summary=False,
        enable_progress_bar=not hide_progress,
    )

    # Following incantation required to load checkpoints.
    from torchvision.transforms.functional import InterpolationMode

    torch.serialization.add_safe_globals([InterpolationMode])

    trainer.fit(
        StafferModule(config),
        StafferDataModule(config, ctx.pdmx, use_sampler, num_workers=num_workers),
        ckpt_path=ckpt_path,
    )

    if (
        early_stopping_callback is not None
        and early_stopping_callback.stopping_reason != EarlyStoppingReason.NOT_STOPPED
    ):
        logging.info(f"Early stopping: {early_stopping_callback.stopping_reason}")
        logging.info(
            f"       message: {early_stopping_callback.stopping_reason_message}"
        )
        logging.info(f"         epoch: {early_stopping_callback.stopped_epoch}")


LOG_VARIABLES = [
    # From StafferModule._step():
    "loss",
    "lr",  # Training only.
    "sys_iou",
    # From LossDict:
    "stave_l1",
    "stave_obj",
    "boundary",
    "sys_lr",
    "sys_obj",
    "sys_giou",
]


def plot_one(
    ax_metrics: Any, name: str, columns: tuple[str, ...], ls: str = "solid"
) -> None:
    csv_path = Path(f"logs/staffer/{name}/metrics.csv")
    all_path = csv_path.with_stem("cumulated_metrics")
    all_df = None
    if all_path.exists():
        all_df = pd.read_csv(all_path)

    if csv_path.exists():
        df = (
            pd.read_csv(csv_path)
            if all_df is None
            else pd.concat([all_df, pd.read_csv(csv_path)])
        )
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
@click.option(
    "--train-columns",
    "-t",
    type=str,
    multiple=True,
    metavar="METRIC,METRIC,...",
    help="Select one or more training metrics to plot.",
)
@click.option(
    "--valid-columns",
    "-v",
    type=str,
    multiple=True,
    metavar="METRIC,METRIC,...",
    help="Select one or more validation metrics to plot.",
)
@click.option(
    "--both-columns",
    "-a",
    type=str,
    multiple=True,
    metavar="METRIC,METRIC,...",
    help="Selects one or more train and validation metrics to plot.",
)
def logs(
    names: tuple[str, ...],
    train_columns: tuple[str, ...],
    valid_columns: tuple[str, ...],
    both_columns: tuple[str, ...],
) -> None:
    """Displays training logs from multiple experiments in a single graph.

    NAMES: List of the names of the model experiments you want graphed.

    \b
    The following METRIC are available:
    - loss, lr (training only), sys_iou,
    - stave_l1, stave_obj, boundary, sys_lr, sys_obj, sys_giou,
    """
    # Parses the metric names and select train/valid variant.
    train_columns = tuple(i for c in train_columns for i in c.split(","))
    valid_columns = tuple(i for c in valid_columns for i in c.split(","))
    both_columns = tuple(i for c in both_columns for i in c.split(","))
    for c in train_columns + valid_columns + both_columns:
        if c not in LOG_VARIABLES:
            raise click.UsageError(f"metric {c} doesn't exist.")
    columns = tuple(f"train/{s}" for s in train_columns) + tuple(
        f"val/{s}" for s in valid_columns
    )
    if len(both_columns) > 0:
        columns += tuple(f"train/{s}" for s in both_columns)
        columns += tuple(f"val/{s}" for s in both_columns)
    if len(columns) == 0:
        raise click.UsageError("Select at least one metric to plot.")

    if not names:
        raise click.UsageError("Provide at least one NAME to plot.")

    # Plots the metrics.
    (name, *others) = names
    csv_path = Path(f"logs/staffer/{name}/metrics.csv")

    plt.ion()
    fig, ax_metrics = plt.subplots(1, 1)

    def on_key(event: Event) -> None:
        key_event = cast(KeyEvent, event)
        if key_event.key == "q":
            plt.close("all")

    fig.canvas.mpl_connect("key_press_event", on_key)
    last_mod = 0.0

    while plt.get_fignums():
        try:
            mtime = csv_path.stat().st_mtime
        except FileNotFoundError:
            mtime = 0.0
        if mtime != last_mod:
            last_mod = mtime

            ax_metrics.cla()
            plot_one(ax_metrics, name, columns)
            for other in others:
                plot_one(ax_metrics, other, columns, ls="dashed")

            ax_metrics.set_title("Training metrics")
            ax_metrics.set_xlabel("step")
            ax_metrics.legend()

            fig.canvas.draw()

        plt.pause(5.0)
    print("Bye!")


def config_from_checkpoint(checkpoint_path: Path) -> StafferConfig:
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    known = {f.name for f in fields(StafferConfig)}
    hp = checkpoint["hyper_parameters"]
    unknown = set(hp) - known
    if unknown:
        logging.warning("config_from_checkpoint: ignoring unknown fields %s", unknown)
    return StafferConfig(**{k: v for k, v in hp.items() if k in known})


def unbox(size: tuple[int, int], t: Tensor) -> Box:
    left, top, right, bot = t.tolist()
    W, H = size
    return Box((int(left * W), int(top * H)), (int(right * W), int(bot * H)))


@click.command()
@click.argument("name", type=str)
@click.argument(
    "img_paths",
    nargs=-1,
    type=click.Path(file_okay=True, exists=True, readable=True, path_type=Path),
)
@click.pass_obj
def predict(ctx: ClickContext, name: str, img_paths: tuple[Path, ...]) -> None:
    """Predicts bounding boxes for system and staves for a list of images.

    NAME: The model version to use to make the predictions.
    IMG_PATHS: The list of images to process and visualize. When omitted,
    picks random images from the PDMX dataset (respecting --offset / --count).
    """
    ckpt_path = Path("checkpoints") / "staffer" / name / "last.ckpt"
    config = config_from_checkpoint(ckpt_path)
    dataset = StafferDataset(config, ctx.pdmx)
    model = StafferModule.load_from_checkpoint(
        ckpt_path, config=config, weights_only=False
    )
    model.eval()
    if img_paths:
        source: Iterable[Path] = img_paths
    else:
        shuffled = list(dataset.items)
        random.shuffle(shuffled)
        source = (item[1] for item in shuffled)
    cv2.namedWindow("Page")
    for img_path in source:
        print(f"Path: {img_path.as_posix()}")
        img = decode_image(img_path.as_posix())
        img = dataset.transform(img).cuda()
        with torch.no_grad():
            (
                pred_stave_tb,
                pred_stave_logits,
                pred_boundary_logits,
                pred_sys_lr,
                pred_sys_logits,
            ) = tuple(map(lambda t: t.squeeze(0), model.forward(img.unsqueeze(0))))

        img = img.squeeze(0).cpu().numpy()
        img = np.stack([img] * 3, axis=-1) * 255
        width_height = img.shape[1], img.shape[0]
        # Stave -> system grouping. Active queries are non-contiguous, so the
        # boundary cumsum must run over only the active staves, ordered top-to-
        # bottom, then scatter the resulting group id back onto their slots.
        stave_logit = pred_stave_logits.squeeze(-1)
        active = (stave_logit > 0.0).nonzero(as_tuple=True)[0]
        active = active[pred_stave_tb[active, 0].argsort()]  # top-to-bottom
        active_boundary = (pred_boundary_logits.squeeze(-1)[active] > 0.0).long()
        active_group = (active_boundary.cumsum(0) - 1).clamp(
            0, pred_sys_lr.shape[0] - 1
        )
        group = torch.full(
            (pred_stave_tb.shape[0],), -1, dtype=torch.long, device=pred_stave_tb.device
        )
        group[active] = active_group
        # System boxes are derived as the hull of each system's active staves.
        for sys_index in range(pred_sys_logits.shape[0]):
            print(f"\tsystem[{sys_index}]: {pred_sys_logits[sys_index].item():.2f}")
            mask = (group == sys_index) & (stave_logit > 0.0)
            if pred_sys_logits[sys_index].item() > 0.0 and bool(mask.any()):
                top = pred_stave_tb[mask][:, 0].min()
                bot = pred_stave_tb[mask][:, 1].max()
                left, right = pred_sys_lr[sys_index]
                box = unbox(width_height, torch.stack([left, top, right, bot]))
                cv2.rectangle(img, box.top_left, box.bot_right, (0, 0, 255), 2)
        for staff_index in range(pred_stave_tb.shape[0]):
            sys_idx = int(group[staff_index].item())
            print(
                f"\tstaff[{staff_index}]: {stave_logit[staff_index].item():.2f}, "
                f"system: {sys_idx}"
            )
            if stave_logit[staff_index].item() > 0.0:
                top, bot = pred_stave_tb[staff_index]
                left, right = pred_sys_lr[sys_idx]
                box = unbox(width_height, torch.stack([left, top, right, bot]))
                cv2.rectangle(img, box.top_left, box.bot_right, (0, 255, 0), 1)
        cv2.imshow("Page", img)

        if cv2.waitKey(0) == ord("q"):
            break
    cv2.destroyAllWindows()


@click.command()
@click.argument("name", type=str)
@click.option(
    "--size",
    type=int,
    default=500,
    show_default=True,
    help="Number of random samples to evaluate.",
)
@click.pass_obj
def run_eval(ctx: ClickContext, name: str, size: int) -> None:
    """Evaluates stave top/bottom accuracy bucketed by vertical position.

    NAME: The model version to evaluate.
    """
    NUM_BINS = 16
    ckpt_path = Path("checkpoints") / "staffer" / name / "last.ckpt"
    config = config_from_checkpoint(ckpt_path)
    H = config.image_shape[0]
    bin_size = H / NUM_BINS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = StafferDataset(config, ctx.pdmx)
    model = StafferModule.load_from_checkpoint(
        ckpt_path, config=config, weights_only=False, map_location=device
    )
    model.eval()
    anchor_c = model.model.heads.anchor_centers().cpu()

    # bins[b] = list of (top_err_px, bot_err_px)
    bins: list[list[tuple[float, float]]] = [[] for _ in range(NUM_BINS)]

    n = min(size, len(dataset))
    indices = random.sample(range(len(dataset)), n)

    with torch.no_grad():
        for idx in tqdm(indices, desc="Evaluating"):
            img_tensor, _gt_sys, gt_staff, gt_assign = dataset[idx]
            pred_stave_tb, _, _, _, _ = model.forward(
                img_tensor.unsqueeze(0).to(device)
            )
            pred_stave_tb = pred_stave_tb.squeeze(0)  # (num_stave_queries, 2)

            # GT stave i is owned by query q[i], not slot i (free assignment).
            num_gt = int((gt_assign != -1).sum())
            q = assign_staves(anchor_c, gt_staff, num_gt)
            for i in range(num_gt):
                qi = int(q[i])
                gt_top = gt_staff[i, 1].item() * H
                gt_bot = gt_staff[i, 3].item() * H
                pred_top = pred_stave_tb[qi, 0].item() * H
                pred_bot = pred_stave_tb[qi, 1].item() * H
                top_err = abs(pred_top - gt_top)
                bot_err = abs(pred_bot - gt_bot)
                bin_idx = min(int(gt_top / bin_size), NUM_BINS - 1)
                bins[bin_idx].append((top_err, bot_err))

    print(f"\nStave coordinate error by vertical position ({n} samples, H={H}px):")
    print(f"  {'bin':>3}  {'y range':>11}  {'n':>5}  {'top_err':>8}  {'bot_err':>8}")
    for b, errors in enumerate(bins):
        y_lo = int(b * bin_size)
        y_hi = int((b + 1) * bin_size)
        if errors:
            avg_top = sum(e[0] for e in errors) / len(errors)
            avg_bot = sum(e[1] for e in errors) / len(errors)
            print(
                f"  {b:>3}  {y_lo:>5}-{y_hi:<5}  {len(errors):>5}"
                f"  {avg_top:>7.1f}px  {avg_bot:>7.1f}px"
            )
        else:
            print(f"  {b:>3}  {y_lo:>5}-{y_hi:<5}  {0:>5}  {'—':>8}  {'—':>8}")


cli.add_command(summary)
cli.add_command(check)
cli.add_command(show)
cli.add_command(stats)
cli.add_command(train)
cli.add_command(logs)
cli.add_command(predict)
cli.add_command(run_eval, name="eval")


def main() -> None:
    torch.set_float32_matmul_precision("high")
    cli()


if __name__ == "__main__":
    main()

# vscode - End of file
