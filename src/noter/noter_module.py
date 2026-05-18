import math

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from .noter_model import NoterConfig, NoterModel
from .noter_vocab import Vocab


class NoterModule(L.LightningModule):
    def __init__(self, config: NoterConfig) -> None:
        super().__init__()
        self.config = config
        self.model = NoterModel(config)
        self.save_hyperparameters(config.asdict())

    def _causal_mask(self, size: int) -> Tensor:
        return torch.ones(size, size, device=self.device, dtype=torch.bool).triu(
            diagonal=1
        )

    def forward(self, source: Tensor, source_widths: Tensor, target: Tensor) -> Tensor:
        attention_mask = self._causal_mask(target.shape[1])
        tgt_pad_mask = (target == Vocab.PAD).all(
            dim=-1
        )  # (B, T) — True if all chord slots are PAD
        return self.model(source, source_widths, target, attention_mask, tgt_pad_mask)

    def _step(self, batch: tuple, stage: str) -> Tensor:
        source, source_widths, target = batch
        # target: (B, T, max_chords) — teacher-forced input is target[:, :-1]
        # labels: target[:, 1:] shifted by one
        logits = self.forward(
            source, source_widths, target[:, :-1]
        )  # (B, T-1, max_chords, vocab_size)
        labels = target[:, 1:]  # (B, T-1, max_chords)

        B, T, H, V = logits.shape
        loss = F.cross_entropy(
            logits.reshape(B * T * H, V),
            labels.reshape(B * T * H),
            ignore_index=Vocab.PAD,
        )

        # Token accuracy (ignoring PAD)
        with torch.no_grad():
            mask = labels != Vocab.PAD
            correct = (logits.argmax(-1) == labels) & mask
            accuracy = correct.sum() / mask.sum().clamp(min=1)

        self.log(f"{stage}/loss", loss, prog_bar=True)
        self.log(f"{stage}/accuracy", accuracy, prog_bar=True)
        if stage == "train":
            self.log("train/lr", self.trainer.optimizers[0].param_groups[0]["lr"])

        return loss

    def training_step(self, batch: tuple, batch_idx: int) -> Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        self._step(batch, "val")

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        optimizer = AdamW(
            self.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            eps=1e-4,
        )

        def lr_lambda(step: int) -> float:
            if step < self.config.warmup_steps:
                return step / max(1, self.config.warmup_steps)
            progress = (step - self.config.warmup_steps) / max(
                1, self.config.max_steps - self.config.warmup_steps
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
