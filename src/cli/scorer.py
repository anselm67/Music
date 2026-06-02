#!/usr/bin/env python3
import logging
import random
import shutil
import sys
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
from tqdm import tqdm

from noter import NoterConfig, Vocab
from pdmx import PDMX, PdmxSource
from scorer import (
    STAFFER_NORM,
    ScorerConfig,
    ScorerDataModule,
    ScorerDataset,
    ScorerModule,
    build_stave_boxes,
)
from staffer import StafferConfig
from utils import format_sequence_columns, sequence_edit_distance, strip_eos

# STAFFER_NORM = (mean, std) page normalisation; unpacked to denormalise for display.
PAGE_MEAN, PAGE_STD = STAFFER_NORM

HOME = Path("/home/anselm/datasets/PDMX")


@dataclass
class ClickContext:
    home: Path
    source: PdmxSource
    config: ScorerConfig


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
    help="Name of scorer's log file.",
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
    logging.getLogger("lightning.pytorch").handlers = []
    if log_file is not None:
        logging.getLogger("lightning.pytorch").addHandler(
            logging.FileHandler(log_file.as_posix())
        )
    logging.info("Running: %s", " ".join(sys.argv))
    pdmx = PDMX(home, csv, offset, count)
    ctx.obj = ClickContext(home, PdmxSource(pdmx), ScorerConfig())


def config_from_checkpoint(checkpoint_path: Path) -> ScorerConfig:
    """Rebuild a ScorerConfig from a saved checkpoint's nested hyper-parameters."""
    from torchvision.transforms.functional import InterpolationMode

    torch.serialization.add_safe_globals([InterpolationMode])
    hp = torch.load(checkpoint_path, weights_only=False)["hyper_parameters"]
    staffer = StafferConfig(**hp["staffer"])
    noter = NoterConfig(**hp["noter"])
    rest = {k: v for k, v in hp.items() if k not in ("staffer", "noter", "max_steps")}
    return ScorerConfig(staffer=staffer, noter=noter, **rest)


@click.command()
@click.pass_obj
def check(ctx: ClickContext) -> None:
    """Wires the model on synthetic data and prints the bridge shapes."""
    config = ScorerConfig()
    config.use_vocab(Vocab.load(ctx.home / "build/vocab.json"))
    module = ScorerModule(config)
    model = module.model
    staffer_params = sum(p.numel() for p in model.staffer.parameters())
    noter_params = sum(p.numel() for p in model.noter.parameters())
    print(f"staffer params: {staffer_params:,}")
    print(f"noter   params: {noter_params:,}")

    B = 2
    image = torch.randn(B, config.staffer.in_channels, *config.staffer.image_shape)
    stave_tb, _, _, sys_lr, _ = model.detect(image)
    print(f"detect: stave_tb {tuple(stave_tb.shape)}, sys_lr {tuple(sys_lr.shape)}")

    # One active stave per page, both owned by system 0.
    sel = [torch.tensor([0]), torch.tensor([0])]
    sys_ids = [torch.tensor([0]), torch.tensor([0])]
    boxes = build_stave_boxes(
        stave_tb, sys_lr, sel, sys_ids, config.staffer.image_shape
    )
    crops, widths = model.crop(image, boxes)
    print(f"bridge: boxes {tuple(boxes.shape)}, crops {tuple(crops.shape)}")

    memory, src_pad = model.noter.encode(crops, widths)
    T = config.noter.max_seqlen - 1
    target = torch.full((boxes.shape[0], T, config.noter.max_chords), Vocab.SOS)
    tgt_pad = (target == Vocab.PAD).all(dim=-1)
    logits = model.noter.decode(
        target, memory, module._causal_mask(T), tgt_pad, src_pad
    )
    print(f"noter:  memory {tuple(memory.shape)}, logits {tuple(logits.shape)}")
    print("ok — detect → crop → encode → decode wired.")


@click.command()
@click.argument("name", type=str)
@click.option(
    "--staffer",
    default="stave-primary-grid-full",
    show_default=True,
    help="Staffer checkpoint to initialise the detector branch from (fresh runs).",
)
@click.option(
    "--noter",
    default="enhanced3",
    show_default=True,
    help="Noter checkpoint to initialise the transcriber branch from (fresh runs).",
)
@click.option(
    "--hide-progress",
    "-p",
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
    staffer: str,
    noter: str,
    hide_progress: bool,
    early_stopping: float,
    epochs: int,
    num_workers: int,
) -> None:
    """Trains and/or resumes training of a Scorer model instance.

    On a fresh run the two branches are initialised from the standalone STAFFER and
    NOTER checkpoints; resuming restores the Scorer's own training state.

    NAME: sets id/name of the model being trained.
    """
    VAL_CHECK_INTERVAL = 250

    vocab = Vocab.load(ctx.home / "build" / "vocab.json")
    ckpt_path: Path | None = None
    candidate = Path("checkpoints") / "scorer" / name / "last.ckpt"
    if candidate.exists():
        logging.info(f"Resuming training from {candidate}")
        ckpt_path = candidate
        config = config_from_checkpoint(candidate)
        module = ScorerModule(config)
    else:
        config = replace(ctx.config, id_name=name)
        config.use_vocab(vocab)
        staffer_ckpt = Path("checkpoints") / "staffer" / staffer / "last.ckpt"
        noter_ckpt = Path("checkpoints") / "noter" / noter / "last.ckpt"
        for label, path in [("staffer", staffer_ckpt), ("noter", noter_ckpt)]:
            if not path.exists():
                raise click.UsageError(f"{label} checkpoint not found: {path}")
        logging.info(f"Initialising branches from {staffer_ckpt} and {noter_ckpt}")
        module = ScorerModule.load_from_checkpoints(config, staffer_ckpt, noter_ckpt)

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
                dirpath=f"checkpoints/scorer/{config.id_name}",
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

    logger_path = Path("logs/scorer") / config.id_name / "metrics.csv"
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

    logger = CSVLogger(save_dir="logs", name="scorer", version=config.id_name)

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
        module,
        ScorerDataModule(config, ctx.source, vocab, num_workers=num_workers),
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
    # From ScorerModule._step():
    "loss",  # joint = λ_det·det + λ_tr·tr
    "det_loss",
    "tr_loss",
    "accuracy",  # transcription token accuracy
    "sys_iou",  # detection quality (1 − IoU; lower is better)
    "stave_err_px",  # clean mean stave-edge error (px)
    "lr",  # Training only.
    # From the detection LossDict:
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
    csv_path = Path(f"logs/scorer/{name}/metrics.csv")
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
    - loss, det_loss, tr_loss, accuracy, sys_iou, stave_err_px, lr (training only),
    - stave_l1, stave_obj, boundary, sys_lr, sys_obj, sys_giou,
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
    csv_path = Path(f"logs/scorer/{name}/metrics.csv")

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


def _similarity(gt_seq: torch.Tensor, pred_seq: torch.Tensor, max_chords: int) -> float:
    """Edit-distance similarity between a GT (SOS-led) and predicted token sequence."""
    gt_content = strip_eos(gt_seq[1:], Vocab.EOS)  # drop SOS, cut at EOS
    pred_content = strip_eos(pred_seq, Vocab.EOS)
    edit = sequence_edit_distance(gt_content, pred_content, Vocab.PAD)
    max_cost = max(len(gt_content), len(pred_content)) * max_chords
    return 1.0 - edit / max_cost if max_cost > 0 else 1.0


def _load_for_inference(name: str) -> tuple[ScorerConfig, ScorerModule]:
    """Load a trained Scorer checkpoint for evaluation, on the best available device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path("checkpoints") / "scorer" / name / "last.ckpt"
    config = config_from_checkpoint(ckpt_path)
    module = ScorerModule.load_from_checkpoint(
        ckpt_path, config=config, weights_only=False, map_location=device
    )
    module.eval()
    return config, module


@click.command()
@click.argument("name", type=str)
@click.pass_obj
def predict(ctx: ClickContext, name: str) -> None:
    """Detects, crops, and transcribes every stave on random pages.

    NAME: The model version to use to make the predictions.
    """
    config, module = _load_for_inference(name)
    vocab = Vocab.load(ctx.home / "build/vocab.json")
    dataset = ScorerDataset(config, ctx.source, vocab)
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    cv2.namedWindow("Page")
    for idx in indices:
        image, _gt_sys, _gt_stave, gt_assign, stave_tokens = dataset[idx]
        boxes, tokens = module.predict(image.unsqueeze(0).to(module.device))
        num_gt = int((gt_assign != -1).sum())

        img = image.squeeze(0).cpu().numpy() * PAGE_STD + PAGE_MEAN
        img = np.stack([(np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)] * 3, axis=-1)

        click.clear()
        print(f"Item {idx}: detected {boxes.shape[0]} staves (GT {num_gt})")
        for k in range(boxes.shape[0]):
            _, left, top, right, bot = boxes[k].tolist()
            cv2.rectangle(
                img, (int(left), int(top)), (int(right), int(bot)), (0, 255, 0), 1
            )
            pred_tokens = dataset.vocab.i2tok(tokens[k].cpu())
            print(f"\n── stave[{k}] ──")
            if k < num_gt:
                gt_tokens = dataset.vocab.i2tok(stave_tokens[k][1:])  # skip SOS
                print(format_sequence_columns(gt_tokens, pred_tokens))
            else:
                print(" ".join(pred_tokens))  # spurious stave — no GT to compare
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
    help="Number of random pages to evaluate.",
)
@click.pass_obj
def run_eval(ctx: ClickContext, name: str, size: int) -> None:
    """Evaluates end-to-end transcription on N random pages.

    Reports token similarity on order-matched (top-to-bottom) predicted vs GT staves,
    plus detection counts so geometric slop and miss/extra are visible separately.

    NAME: The model version to evaluate.
    """
    config, module = _load_for_inference(name)
    vocab = Vocab.load(ctx.home / "build/vocab.json")
    dataset = ScorerDataset(config, ctx.source, vocab)
    n = min(size, len(dataset))
    indices = random.sample(range(len(dataset)), n)

    similarities: list[float] = []
    total_gt, total_pred, pages_miscount = 0, 0, 0
    for idx in tqdm(indices, desc="Evaluating"):
        image, _gt_sys, _gt_stave, gt_assign, stave_tokens = dataset[idx]
        num_gt = int((gt_assign != -1).sum())
        _boxes, tokens = module.predict(image.unsqueeze(0).to(module.device))
        num_pred = tokens.shape[0]
        total_gt += num_gt
        total_pred += num_pred
        pages_miscount += int(num_pred != num_gt)
        for k in range(min(num_gt, num_pred)):
            similarities.append(
                _similarity(stave_tokens[k], tokens[k].cpu(), config.noter.max_chords)
            )

    if not similarities:
        print("No matched staves to evaluate.")
        return
    print(f"\nEvaluated {n} pages from '{name}':")
    print(f"  staves: {total_pred} predicted / {total_gt} GT")
    print(f"  pages with miscount: {pages_miscount} / {n}")
    print(f"  matched-stave similarity  min {min(similarities):.1%}")
    print(
        f"                            avg {sum(similarities) / len(similarities):.1%}"
    )
    print(f"                            max {max(similarities):.1%}")


cli.add_command(check)
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
