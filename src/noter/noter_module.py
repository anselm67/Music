import math

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from kern import NUM_ARTICULATIONS

from .articulation import articulation_loss
from .noter_model import NoterConfig, NoterModel
from .noter_vocab import Vocab


class NoterModule(L.LightningModule):
    _causal_mask_buf: Tensor

    def __init__(self, config: NoterConfig) -> None:
        super().__init__()
        self.config = config
        self.model = NoterModel(config)
        self.save_hyperparameters(config.asdict())
        self.register_buffer(
            "_causal_mask_buf",
            torch.ones(config.max_seqlen, config.max_seqlen, dtype=torch.bool).triu(
                diagonal=1
            ),
        )

    def _causal_mask(self, size: int) -> Tensor:
        return self._causal_mask_buf[:size, :size]

    def forward(
        self, source: Tensor, source_widths: Tensor, target: Tensor, stave_mask: Tensor
    ) -> Tensor:
        # target: (B, S, T, max_chords); the model builds the per-staff and
        # cross-stave masks internally from stave_mask.
        causal = self._causal_mask(target.shape[2])
        return self.model(source, source_widths, target, stave_mask, causal)

    def _step(self, batch: tuple, stage: str) -> Tensor:
        source, source_widths, target, articulations, stave_mask = batch
        # target: (B, S, T, max_chords) — teacher-forced input is target[:, :, :-1],
        # labels shifted by one. Padded staves are excluded from the loss.
        tgt_in = target[:, :, :-1]
        causal = self._causal_mask(tgt_in.shape[2])
        labels = target[:, :, 1:].clone()  # (B, S, T-1, max_chords)
        labels[~stave_mask] = Vocab.PAD

        # The input note's articulation is fed back alongside its token; the
        # head predicts the *next* note's articulation (shifted, like tokens).
        logits, art_logits = self.model.forward_both(
            source,
            source_widths,
            tgt_in,
            stave_mask,
            causal,
            articulations[:, :, :-1],
        )

        V = logits.shape[-1]
        loss = F.cross_entropy(
            logits.reshape(-1, V),
            labels.reshape(-1),
            ignore_index=Vocab.PAD,
        )

        # Token accuracy (ignoring PAD)
        slot_mask = labels != Vocab.PAD
        with torch.no_grad():
            correct = (logits.argmax(-1) == labels) & slot_mask
            accuracy = correct.sum() / slot_mask.sum().clamp(min=1)

        self.log(f"{stage}/loss", loss, prog_bar=True)
        self.log(f"{stage}/accuracy", accuracy, prog_bar=True)

        loss = loss + self._articulation_loss(
            art_logits, articulations[:, :, 1:], slot_mask, stage
        )

        if stage == "train":
            self.log("train/lr", self.trainer.optimizers[0].param_groups[0]["lr"])

        return loss

    def _articulation_loss(
        self, art_logits: Tensor, art_labels: Tensor, slot_mask: Tensor, stage: str
    ) -> Tensor:
        """Weighted, logged articulation loss over the head (see ``articulation_loss``).

        ``art_logits``/``art_labels``: (B, S, T-1, max_chords, NUM_ARTICULATIONS);
        ``slot_mask``: (B, S, T-1, max_chords) — the same non-PAD note slots the
        token loss covers (padded staves already folded in)."""
        loss, acc, recall = articulation_loss(art_logits, art_labels, slot_mask)
        self.log(f"{stage}/art_loss", loss, prog_bar=True)
        self.log(f"{stage}/art_acc", acc)
        self.log(f"{stage}/art_recall", recall, prog_bar=True)
        return self.config.articulation_weight * loss

    @torch.no_grad()
    def predict(
        self,
        source: Tensor,
        source_widths: Tensor,
        stave_mask: Tensor,
        return_articulations: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Lockstep decode of a batch of systems → ``(B, S, T-1, max_chords)``.

        Each row ``t`` is generated for all ``S`` staves at once, so a staff sees its
        siblings' rows ``< t`` through the cross-stave attention. A staff finishes at
        its own EOS; padded staves start finished.

        With ``return_articulations``, also returns the row-aligned generated
        articulation multi-hot ``(B, S, T-1, max_chords, NUM_ARTICULATIONS)``.
        """
        B, S, c = source.shape[0], source.shape[1], self.config
        flat_src = source.reshape(B * S, *source.shape[2:])
        memory, mem_pad = self.model.encode(flat_src, source_widths.reshape(B * S))
        generated = torch.full(
            (B, S, 1, c.max_chords), Vocab.SOS, device=self.device, dtype=torch.long
        )
        # Carry a parallel buffer of generated articulations (SOS row = zeros) and
        # feed it back as input, mirroring the teacher-forced training path. The
        # buffer is row-aligned with `generated`.
        art = torch.zeros(B, S, 1, c.max_chords, NUM_ARTICULATIONS, device=self.device)
        done = ~stave_mask.clone()  # padded staves emit EOS immediately
        for _ in range(c.max_seqlen - 1):
            T = generated.shape[2]
            causal = self._causal_mask(T)
            logits, art_logits = self.model.decode_both(
                generated, memory, mem_pad, stave_mask, causal, art
            )
            next_tokens = logits[:, :, -1, :, :].argmax(dim=-1)  # (B, S, max_chords)
            next_tokens[done] = Vocab.EOS  # keep finished staves at EOS
            done = done | (next_tokens[..., 0] == Vocab.EOS)
            generated = torch.cat([generated, next_tokens.unsqueeze(2)], dim=2)
            next_art = (art_logits[:, :, -1] > 0).float()  # (B, S, mc, A)
            next_art[done] = 0.0  # finished staves carry no articulation
            art = torch.cat([art, next_art.unsqueeze(2)], dim=2)
            if done.all():
                break
        if return_articulations:
            return generated[:, :, 1:], art[:, :, 1:]  # strip SOS
        return generated[:, :, 1:]  # strip SOS

    def training_step(self, batch: tuple, batch_idx: int) -> Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        self._step(batch, "val")

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        optimizer = AdamW(
            self.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
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
