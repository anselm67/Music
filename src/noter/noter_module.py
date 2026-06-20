import math

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from .duration import bin_boundaries, bin_log2_lengths
from .noter_model import NoterConfig, NoterModel
from .noter_vocab import Vocab


class NoterModule(L.LightningModule):
    _causal_mask_buf: Tensor
    _duration_boundaries: Tensor
    _dur_bin_log2: Tensor

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
        # Duration-table bin boundaries (for bucketizing GT log2 lengths to bin
        # labels) and each bin's canonical log2 length (for decoding an argmax
        # bin back to a length to feed forward).
        self.register_buffer("_duration_boundaries", torch.tensor(bin_boundaries()))
        self.register_buffer("_dur_bin_log2", torch.tensor(bin_log2_lengths()))

    def _causal_mask(self, size: int) -> Tensor:
        return self._causal_mask_buf[:size, :size]

    def forward(
        self,
        source: Tensor,
        source_widths: Tensor,
        target: Tensor,
        stave_mask: Tensor,
        target_dur: Tensor | None = None,
        target_dur_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        # target: (B, S, T, max_chords); the model builds the per-staff and
        # cross-stave masks internally from stave_mask.
        causal = self._causal_mask(target.shape[2])
        return self.model(
            source,
            source_widths,
            target,
            stave_mask,
            causal,
            target_dur,
            target_dur_mask,
        )

    def _step(self, batch: tuple, stage: str) -> Tensor:
        source, source_widths, target, target_dur, target_dur_mask, stave_mask = batch
        # target: (B, S, T, max_chords) — teacher-forced input is target[:, :, :-1],
        # labels shifted by one. Padded staves are excluded from the loss. The
        # duration channels are teacher-forced and shifted in lockstep.
        logits, dur_logits = self.forward(
            source,
            source_widths,
            target[:, :, :-1],
            stave_mask,
            target_dur[:, :, :-1],
            target_dur_mask[:, :, :-1],
        )  # (B, S, T-1, max_chords, vocab_size), (..., NUM_DUR_BINS)
        labels = target[:, :, 1:].clone()  # (B, S, T-1, max_chords)
        labels[~stave_mask] = Vocab.PAD

        V = logits.shape[-1]
        token_loss = F.cross_entropy(
            logits.reshape(-1, V),
            labels.reshape(-1),
            ignore_index=Vocab.PAD,
        )

        # Duration classification (CE over the analytic table's bins), only on
        # slots that carry a duration and belong to a real staff. GT log2 lengths
        # are bucketized to bin labels against the table midpoints.
        dur_bins = torch.bucketize(
            target_dur[:, :, 1:].contiguous(), self._duration_boundaries
        )
        dur_loss_mask = target_dur_mask[:, :, 1:].clone()
        dur_loss_mask[~stave_mask] = False
        if dur_loss_mask.any():
            dur_loss = F.cross_entropy(
                dur_logits[dur_loss_mask], dur_bins[dur_loss_mask]
            )
        else:
            # No duration-bearing slot this batch (e.g. all-pad staves): a
            # constant with no autograd edge, so the head simply isn't updated.
            dur_loss = dur_logits.new_zeros(())
        loss = token_loss + self.config.dur_loss_weight * dur_loss

        with torch.no_grad():
            mask = labels != Vocab.PAD
            accuracy = ((logits.argmax(-1) == labels) & mask).sum() / mask.sum().clamp(
                min=1
            )
            # Exact snap-match: the predicted bin equals the GT bin.
            if dur_loss_mask.any():
                pred_bins = dur_logits.argmax(-1)
                dur_acc = (
                    (pred_bins[dur_loss_mask] == dur_bins[dur_loss_mask]).float().mean()
                )
            else:
                dur_acc = torch.ones((), device=loss.device)

        self.log(f"{stage}/loss", loss, prog_bar=True)
        self.log(f"{stage}/accuracy", accuracy, prog_bar=True)
        self.log(f"{stage}/dur_loss", dur_loss, prog_bar=True)
        self.log(f"{stage}/dur_acc", dur_acc, prog_bar=True)
        if stage == "train":
            self.log("train/lr", self.trainer.optimizers[0].param_groups[0]["lr"])

        return loss

    @torch.no_grad()
    def predict(
        self,
        source: Tensor,
        source_widths: Tensor,
        stave_mask: Tensor,
        duration_bearing: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Lockstep decode of a batch of systems → ``(tokens, durations)``, each
        ``(B, S, T-1, max_chords)``.

        Each row ``t`` is generated for all ``S`` staves at once, so a staff sees its
        siblings' rows ``< t`` through the cross-stave attention. A staff finishes at
        its own EOS; padded staves start finished. ``duration_bearing`` is a
        ``(vocab_size,)`` bool flagging which token ids carry a duration; the
        predicted bin's log2 length for those slots is fed back as the next step's
        input, and ``durations`` is returned as log2 lengths (snap-canonical).
        """
        B, S, c = source.shape[0], source.shape[1], self.config
        flat_src = source.reshape(B * S, *source.shape[2:])
        memory, mem_pad = self.model.encode(flat_src, source_widths.reshape(B * S))
        generated = torch.full(
            (B, S, 1, c.max_chords), Vocab.SOS, device=self.device, dtype=torch.long
        )
        gen_dur = torch.zeros((B, S, 1, c.max_chords), device=self.device)
        gen_dmask = torch.zeros(
            (B, S, 1, c.max_chords), device=self.device, dtype=torch.bool
        )
        done = ~stave_mask.clone()  # padded staves emit EOS immediately
        for _ in range(c.max_seqlen - 1):
            T = generated.shape[2]
            logits, dur_logits = self.model.decode(
                generated,
                memory,
                mem_pad,
                stave_mask,
                self._causal_mask(T),
                gen_dur,
                gen_dmask,
            )
            next_tokens = logits[:, :, -1, :, :].argmax(dim=-1)  # (B, S, max_chords)
            next_bin = dur_logits[:, :, -1, :, :].argmax(dim=-1)  # (B, S, max_chords)
            next_dur = self._dur_bin_log2[next_bin]  # bin -> log2 length
            next_tokens[done] = Vocab.EOS  # keep finished staves at EOS
            next_dmask = duration_bearing[next_tokens]  # (B, S, max_chords)
            next_dur = torch.where(next_dmask, next_dur, torch.zeros_like(next_dur))
            done = done | (next_tokens[..., 0] == Vocab.EOS)
            generated = torch.cat([generated, next_tokens.unsqueeze(2)], dim=2)
            gen_dur = torch.cat([gen_dur, next_dur.unsqueeze(2)], dim=2)
            gen_dmask = torch.cat([gen_dmask, next_dmask.unsqueeze(2)], dim=2)
            if done.all():
                break
        return generated[:, :, 1:], gen_dur[:, :, 1:]  # strip SOS

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
