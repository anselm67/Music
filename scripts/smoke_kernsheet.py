"""Throwaway smoke test: train a few steps of staffer / noter / scorer on
KernSheetSource to confirm the revamped KernSheet plumbing feeds all three models.

Not a real training run — tiny lengths, 2 optimiser steps each, garbage weights.
The goal is only to exercise: KernSheetSource.scores/score/image/records ->
dataset -> datamodule -> model forward+backward, end to end.

Run: uv run python scripts/smoke_kernsheet.py
"""

import logging
from pathlib import Path

import lightning as L

from kernsheet import KernSheet, KernSheetSource
from noter import NoterConfig, NoterDataModule, NoterModule, Vocab
from scorer import ScorerConfig, ScorerDataModule, ScorerModule
from staffer import StafferConfig, StafferDataModule, StafferModule

HOME = Path("/home/anselm/datasets/KernSheet")
CKPT = Path("checkpoints")
# Tiny run: total dataset = TRAIN + VALID items, BATCH-sized, 2 steps.
BATCH, TRAIN, VALID = 4, 8, 4
RENDER_SCORES = 24  # enough leading scores to cover the first TRAIN+VALID items


def make_trainer() -> L.Trainer:
    return L.Trainer(
        max_steps=2,
        limit_train_batches=2,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="auto",
        devices=1,
        precision="32",
    )


def render_subset(ks: KernSheet, src: KernSheetSource) -> int:
    """Render PNGs for the first RENDER_SCORES renderable scores (catalog order =
    the order the datasets consume), so the leading dataset items have images."""
    rendered = 0
    for score in src.scores():
        try:
            ks._rebuild_images(ks.id2score[score.id], score)
            rendered += 1
        except Exception as e:
            logging.warning(f"render skip {score.id}: {e}")
        if rendered >= RENDER_SCORES:
            break
    logging.info(f"rendered {rendered} scores")
    return rendered


def load_vocab() -> Vocab:
    path = HOME / "build" / "vocab.json"
    if path.exists():
        return Vocab.load(path)
    files = sorted((HOME / "build" / "tokens").rglob("*.tokens"))
    vocab = Vocab.from_files(files)
    vocab.save(path)
    logging.info(f"built vocab ({len(vocab)} tokens) from {len(files)} files")
    return vocab


def smoke_staffer(src: KernSheetSource) -> None:
    cfg = StafferConfig(
        id_name="smoke", batch_size=BATCH, train_len=TRAIN, valid_len=VALID
    )
    module = StafferModule(cfg)
    dm = StafferDataModule(cfg, src, use_sampler=False, num_workers=0)
    trainer = make_trainer()
    trainer.fit(module, dm)
    ckpt = CKPT / "staffer" / "smoke" / "last.ckpt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(ckpt)
    print(f"  staffer: OK ({trainer.global_step} steps) -> {ckpt}")


def smoke_noter(src: KernSheetSource, vocab: Vocab) -> None:
    cfg = NoterConfig(
        id_name="smoke", batch_size=BATCH, train_len=TRAIN, valid_len=VALID
    )
    cfg.use_vocab(vocab)
    module = NoterModule(cfg)
    dm = NoterDataModule(cfg, src, vocab, num_workers=0)
    trainer = make_trainer()
    trainer.fit(module, dm)
    ckpt = CKPT / "noter" / "smoke" / "last.ckpt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(ckpt)
    print(f"  noter: OK ({trainer.global_step} steps) -> {ckpt}")


def smoke_scorer(src: KernSheetSource, vocab: Vocab) -> None:
    cfg = ScorerConfig(
        id_name="smoke", batch_size=BATCH, train_len=TRAIN, valid_len=VALID
    )
    cfg.use_vocab(vocab)
    module = ScorerModule.load_from_checkpoints(
        cfg,
        CKPT / "staffer" / "smoke" / "last.ckpt",
        CKPT / "noter" / "smoke" / "last.ckpt",
    )
    dm = ScorerDataModule(cfg, src, vocab, num_workers=0)
    trainer = make_trainer()
    trainer.fit(module, dm)
    print(f"  scorer: OK ({trainer.global_step} steps)")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ks = KernSheet(HOME)
    src = KernSheetSource(ks)

    print("== rendering a subset of pages ==")
    render_subset(ks, src)
    vocab = load_vocab()

    results: dict[str, str] = {}
    print("== staffer ==")
    try:
        smoke_staffer(src)
        results["staffer"] = "OK"
    except Exception as e:
        logging.exception("staffer failed")
        results["staffer"] = f"FAIL: {e}"

    print("== noter ==")
    try:
        smoke_noter(src, vocab)
        results["noter"] = "OK"
    except Exception as e:
        logging.exception("noter failed")
        results["noter"] = f"FAIL: {e}"

    print("== scorer ==")
    try:
        smoke_scorer(src, vocab)
        results["scorer"] = "OK"
    except Exception as e:
        logging.exception("scorer failed")
        results["scorer"] = f"FAIL: {e}"

    print("\n== summary ==")
    for name, status in results.items():
        print(f"  {name:8s} {status}")


if __name__ == "__main__":
    main()
