#!/usr/bin/env python3
import logging
import math
import random
import shutil
import sys
from collections import Counter
from dataclasses import dataclass, replace
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
from tqdm import tqdm

from noter import NoterConfig, NoterDataModule, NoterDataset, NoterModule, Vocab
from pdmx import PDMX, PdmxSource
from utils import (
    format_sequence_columns,
    print_histogram,
    sequence_edit_distance,
    strip_eos,
)

HOME = Path("/home/anselm/datasets/PDMX")


@dataclass
class ClickContext:
    home: Path
    source: PdmxSource
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
    ctx.obj = ClickContext(home, PdmxSource(pdmx), NoterConfig())


@click.command()
@click.pass_obj
def vocab(ctx: ClickContext) -> None:
    """Generates the vocab pickle file from PDMX token files."""
    vocab = Vocab.from_pdmx(ctx.source.pdmx)
    vocab.save(ctx.home / "build" / "vocab.json")


@click.command()
@click.argument(
    "kernsheet_home",
    type=click.Path(
        dir_okay=True, file_okay=False, exists=True, readable=True, path_type=Path
    ),
)
@click.pass_obj
def extend_vocab(ctx: ClickContext, kernsheet_home: Path) -> None:
    """Extends the PDMX vocab with KernSheet tokens for fine-tuning.

    Loads the frozen PDMX vocab (which the noter checkpoint was trained on),
    appends KernSheet tokens seen at least twice at the tail (preserving every
    PDMX id so the checkpoint stays loadable), and writes the combined vocab to
    KERNSHEET_HOME/build/vocab.json.
    """
    base = Vocab.load(ctx.home / "build" / "vocab.json")
    tokens_dir = kernsheet_home / "build" / "tokens"
    files = sorted(tokens_dir.rglob("*.tokens"))
    if not files:
        raise click.ClickException(f"No .tokens files found under {tokens_dir}")
    combined = base.extend_from_files(files)
    out = kernsheet_home / "build" / "vocab.json"
    combined.save(out)
    logging.info(f"Wrote combined vocab ({len(combined):,} tokens) -> {out}")


@click.command()
@click.pass_obj
def show(ctx: ClickContext) -> None:
    """Displays random samples from the dataset."""
    vocab = Vocab.load(ctx.home / "build/vocab.json")
    dataset = NoterDataset(ctx.config, ctx.source, vocab)
    cv2.namedWindow("Staff")
    while True:
        index = random.randint(0, len(dataset) - 1)
        img_tensor, _, seq_tensor = dataset[index]
        img = img_tensor.squeeze(0).cpu().numpy()
        tokens = dataset.vocab.i2tok(seq_tensor)
        print(tokens)
        cv2.imshow("Staff", img)
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
    """Computes stats for images from the dataset."""
    # We can't use DataLoader just yet, because the image size isn't padded.
    vocab = Vocab.load(ctx.home / "build/vocab.json")
    dataset = NoterDataset(ctx.config, ctx.source, vocab)
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
    vocab = Vocab.load(ctx.home / "build/vocab.json")
    ds = NoterDataset(ctx.config, ctx.source, vocab)
    loader = DataLoader[tuple[Tensor, Tensor]](
        ds, num_workers=num_workers, batch_size=ctx.config.batch_size
    )
    pix_sum = 0
    pix_sum2 = 0
    pix_count = 0
    for images, _, _ in loader:
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
    config.use_vocab(Vocab.load(ctx.home / "build/vocab.json"))
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


def config_from_checkpoint(checkpoint_path: Path) -> NoterConfig:
    from torchvision.transforms.functional import InterpolationMode

    torch.serialization.add_safe_globals([InterpolationMode])
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    hyper_params = checkpoint["hyper_parameters"]
    hyper_params.pop("max_steps", None)
    return NoterConfig(**hyper_params)


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
    "--epochs", "-e", type=int, default=4, help="Number of epochs to train for."
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
    num_workers: int,
) -> None:
    """Trains and/or resumes training of a Noter model instance.

    NAME: sets id/name of the model being trained.
    """
    VAL_CHECK_INTERVAL = 250

    vocab = Vocab.load(ctx.home / "build" / "vocab.json")
    ckpt_path: Path | None = None
    ckpt_path = Path("checkpoints") / "noter" / name / "last.ckpt"
    if ckpt_path.exists():
        logging.info(f"Resuming training from {ckpt_path}")
        config = config_from_checkpoint(ckpt_path)
    else:
        ckpt_path = None
        config = replace(ctx.config, id_name=name)
        config.use_vocab(vocab)

    config.max_steps = epochs * (config.train_len // config.batch_size)
    logging.info(
        f"Training for {epochs} epochs, "
        f"or {config.max_steps} steps of {config.batch_size}."
    )

    early_stopping_callback = None
    if early_stopping > 0:
        steps = int(early_stopping * (config.train_len // config.batch_size))
        steps = steps // VAL_CHECK_INTERVAL
        logging.info(f"EarlyStopping: patience is {steps} validation steps.")
        early_stopping_callback = EarlyStopping(
            monitor="val/loss",
            patience=steps,
            mode="min",
            min_delta=1e-4,
        )

    callbacks: list[Callback] = [
        callback
        for callback in [
            ModelCheckpoint(
                dirpath=f"checkpoints/noter/{config.id_name}",
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

    logger_path = Path("logs/noter") / config.id_name / "metrics.csv"
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

    logger = CSVLogger(save_dir="logs", name="noter", version=config.id_name)

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

    trainer.fit(
        NoterModule(config),
        NoterDataModule(config, ctx.source, vocab, num_workers=num_workers),
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
    "loss",
    "lr",
    "accuracy",
]


def plot_one(
    ax_metrics: Any, name: str, columns: tuple[str, ...], ls: str = "solid"
) -> None:
    csv_path = Path(f"logs/noter/{name}/metrics.csv")
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
    - loss, lr (training only), accuracy
    """
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

    (name, *others) = names
    csv_path = Path(f"logs/noter/{name}/metrics.csv")

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


@click.command()
@click.argument("name", type=str)
@click.option(
    "--size",
    type=int,
    default=1024,
    show_default=True,
    help="Number of random samples to evaluate.",
)
@click.pass_obj
def run_eval(ctx: ClickContext, name: str, size: int) -> None:
    """Evaluates prediction accuracy on N random samples from the dataset.

    NAME: The model version to use to make the predictions.
    """
    ckpt_path = Path("checkpoints") / "noter" / name / "last.ckpt"
    config = config_from_checkpoint(ckpt_path)
    vocab = Vocab.load(ctx.home / "build/vocab.json")
    dataset = NoterDataset(config, ctx.source, vocab)
    module = NoterModule.load_from_checkpoint(
        ckpt_path, config=config, weights_only=False
    )
    module.eval()

    n = min(size, len(dataset))
    indices = random.sample(range(len(dataset)), n)

    similarities: list[float] = []
    for idx in tqdm(indices, desc="Evaluating"):
        image, source_width, gt_sequence = dataset[idx]

        device = module.device
        predicted = module.predict(
            image.unsqueeze(0).to(device), source_width.unsqueeze(0).to(device)
        )

        gt_content = strip_eos(gt_sequence[1:], Vocab.EOS)
        pred_content = strip_eos(predicted[0].cpu(), Vocab.EOS)
        edit_dist = sequence_edit_distance(gt_content, pred_content, Vocab.PAD)
        max_cost = max(len(gt_content), len(pred_content)) * config.max_chords
        similarity = 1.0 - edit_dist / max_cost if max_cost > 0 else 1.0
        similarities.append(similarity)

    if not similarities:
        print("No samples to evaluate.")
        return
    print(f"\nEvaluated {n} samples from '{name}':")
    print(f"  min: {min(similarities):.1%}")
    print(f"  avg: {sum(similarities) / len(similarities):.1%}")
    print(f"  max: {max(similarities):.1%}")


@click.command()
@click.argument("name", type=str)
@click.pass_obj
def predict(ctx: ClickContext, name: str) -> None:
    """Predicts token sequences for random samples from the dataset.

    NAME: The model version to use to make the predictions.
    """
    ckpt_path = Path("checkpoints") / "noter" / name / "last.ckpt"
    config = config_from_checkpoint(ckpt_path)
    vocab = Vocab.load(ctx.home / "build/vocab.json")
    dataset = NoterDataset(config, ctx.source, vocab)
    module = NoterModule.load_from_checkpoint(
        ckpt_path, config=config, weights_only=False
    )
    module.eval()

    shuffled_indices = list(range(len(dataset)))
    random.shuffle(shuffled_indices)

    total_similarity = 0.0
    n_samples = 0

    cv2.namedWindow("Staff")
    for idx in shuffled_indices:
        image, source_width, gt_sequence = dataset[idx]

        device = module.device
        predicted = module.predict(
            image.unsqueeze(0).to(device), source_width.unsqueeze(0).to(device)
        )  # (1, T, max_chords)

        gt_tokens = dataset.vocab.i2tok(gt_sequence[1:])  # skip SOS
        pred_tokens = dataset.vocab.i2tok(predicted[0].cpu())

        gt_content = strip_eos(gt_sequence[1:], Vocab.EOS)
        pred_content = strip_eos(predicted[0].cpu(), Vocab.EOS)
        edit_dist = sequence_edit_distance(gt_content, pred_content, Vocab.PAD)
        max_cost = max(len(gt_content), len(pred_content)) * config.max_chords
        similarity = 1.0 - edit_dist / max_cost if max_cost > 0 else 1.0

        total_similarity += similarity
        n_samples += 1
        avg_similarity = total_similarity / n_samples

        click.clear()
        print(f"Item {idx}")
        print(format_sequence_columns(gt_tokens, pred_tokens))
        print(f"\nSimilarity: {similarity:.1%}  (edit {edit_dist} / max {max_cost})")
        print(f"   Avg sim: {avg_similarity:.1%}  ({n_samples} samples)")

        # Denormalize and display.
        img = image.squeeze(0).cpu().numpy()
        img = img * 0.17525607175008864 + 0.9482423663139343
        img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
        cv2.imshow("Staff", np.stack([img] * 3, axis=-1))

        if cv2.waitKey(0) == ord("q"):
            break
    cv2.destroyAllWindows()


cli.add_command(vocab)
cli.add_command(extend_vocab, name="extend-vocab")
cli.add_command(show)
cli.add_command(stats)
cli.add_command(image_stats)
cli.add_command(summary)
cli.add_command(train)
cli.add_command(logs)
cli.add_command(run_eval, name="eval")
cli.add_command(predict)


def main() -> None:
    torch.set_float32_matmul_precision("high")
    cli()


if __name__ == "__main__":
    main()
