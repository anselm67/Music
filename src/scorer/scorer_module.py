"""Lightning module for the Scorer end-to-end model: joint detection + transcription."""

import math
from dataclasses import fields
from pathlib import Path

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torchvision.transforms.functional import InterpolationMode

from noter import Vocab
from staffer import StafferLoss, assign_staves
from staffer.staffer_loss import generalized_iou

from .scorer_model import ScorerConfig, ScorerModel, build_stave_boxes


def mean_sys_iou(
    pred_stave_tb: Tensor,  # (B, M, 2) — top, bottom
    pred_sys_lr: Tensor,  # (B, N, 2) — left, right
    gt_sys_boxes: Tensor,  # (B, N, 4) ltrb padded
    gt_assign: Tensor,  # (B, M) padded with -1
    assign_q: list[Tensor],  # per page: (G,) query slot owning each GT stave
) -> Tensor:
    """1 − IoU of the derived system boxes (hull of each system's staves).

    Same convention as ``StafferModule._mean_sys_iou`` — lower is better.
    """
    ious = []
    for i in range(pred_stave_tb.shape[0]):
        assign, q = gt_assign[i], assign_q[i]
        num_gt_sys = int(assign[assign != -1].max().item()) + 1
        a = assign[: q.shape[0]]
        tb = pred_stave_tb[i][q]
        derived = []
        for j in range(num_gt_sys):
            mask = a == j
            top = tb[mask][:, 0].min()
            bot = tb[mask][:, 1].max()
            derived.append(
                torch.stack([pred_sys_lr[i][j, 0], top, pred_sys_lr[i][j, 1], bot])
            )
        iou = generalized_iou(torch.stack(derived), gt_sys_boxes[i][:num_gt_sys])
        ious.append(1.0 - iou.clamp(min=0).mean())
    return torch.stack(ious).mean()


def mean_stave_l1(
    pred_tb: Tensor,  # (B, M, 2) — top, bottom
    gt_stave_boxes: Tensor,  # (B, M, 4) ltrb padded
    assign_q: list[Tensor],  # per page: (G,) query slot owning each GT stave
) -> Tensor:
    """Mean L1 error on (top, bottom) across GT staves (normalised units)."""
    errors = []
    for i in range(pred_tb.shape[0]):
        q = assign_q[i]
        matched = pred_tb[i][q]
        gt = gt_stave_boxes[i][: q.shape[0], [1, 3]]
        errors.append(torch.abs(matched - gt).mean())
    return torch.stack(errors).mean()


def active_grouping(
    stave_tb: Tensor,  # (M, 2) — top, bottom
    stave_logits: Tensor,  # (M, 1) — stave objectness
    boundary_logits: Tensor,  # (M, 1) — >0 ⇒ starts a new system
    num_systems: int,
) -> tuple[Tensor, Tensor]:
    """Inference-time stave→system grouping for one page (no ground truth).

    Mirrors ``staffer predict``: the active queries (objectness > 0), sorted
    top-to-bottom, *are* the detected staves; the boundary cumsum over them recovers
    each one's system. Returns (sel, owners) — the active query indices and the system
    index each inherits (left, right) from — ready for ``build_stave_boxes``.
    """
    logit = stave_logits.squeeze(-1)
    active = (logit > 0).nonzero(as_tuple=True)[0]
    if active.numel() == 0:
        return active, active
    active = active[stave_tb[active, 0].argsort()]  # top-to-bottom
    boundary = (boundary_logits.squeeze(-1)[active] > 0).long()
    boundary[0] = 1  # the first detected stave always opens system 0
    owners = (boundary.cumsum(0) - 1).clamp(0, num_systems - 1)
    return active, owners


class ScorerModule(L.LightningModule):
    _causal_mask_buf: Tensor

    def __init__(self, config: ScorerConfig) -> None:
        super().__init__()
        self.config = config
        self.model = ScorerModel(config)
        self.loss_fn = StafferLoss(config.staffer)
        self.save_hyperparameters(config.asdict())
        self.register_buffer(
            "_causal_mask_buf",
            torch.ones(
                config.noter.max_seqlen, config.noter.max_seqlen, dtype=torch.bool
            ).triu(diagonal=1),
        )
        # Staffer starts frozen so the noter adapts to predicted crops first;
        # unfrozen at freeze_staffer_steps (see on_train_batch_start).
        self._staffer_frozen = config.freeze_staffer_steps > 0
        if self._staffer_frozen:
            self.model.staffer.requires_grad_(False)

    def _causal_mask(self, size: int) -> Tensor:
        return self._causal_mask_buf[:size, :size]

    def _step(self, batch: tuple, stage: str) -> Tensor:
        image, gt_sys, gt_stave, gt_assign, stave_tokens = batch
        B = image.shape[0]

        # --- Detection (staffer) ---
        stave_tb, stave_logits, boundary_logits, sys_lr, sys_logits = self.model.detect(
            image
        )

        # Route each GT stave to its nearest-anchor query — drives both the
        # detection loss and which predicted boxes feed the noter.
        anchor_c = self.model.staffer.heads.anchor_centers()
        assign_q = [
            assign_staves(anchor_c, gt_stave[i], int((gt_assign[i] != -1).sum()))
            for i in range(B)
        ]
        det = self.loss_fn.forward(
            stave_tb,
            stave_logits,
            boundary_logits,
            sys_lr,
            sys_logits,
            gt_sys,
            gt_stave,
            gt_assign,
            assign_q,
        )

        # --- Transcription (noter) over the GT-routed predicted crops ---
        sys_ids = [gt_assign[i][: assign_q[i].shape[0]] for i in range(B)]
        hw = (int(image.shape[-2]), int(image.shape[-1]))
        boxes = build_stave_boxes(stave_tb, sys_lr, assign_q, sys_ids, hw)
        # Targets, flattened in the same (page, GT-stave) order as the boxes.
        targets = torch.cat(
            [stave_tokens[i, : assign_q[i].shape[0]] for i in range(B)], dim=0
        )  # (K, T, max_chords)

        tr = torch.zeros((), device=image.device)
        accuracy = torch.zeros((), device=image.device)
        if targets.shape[0] > 0:
            crops, widths = self.model.crop(image, boxes)
            memory, src_pad = self.model.noter.encode(crops, widths)
            tgt_in, labels = targets[:, :-1], targets[:, 1:]
            tgt_pad = (tgt_in == Vocab.PAD).all(dim=-1)
            logits = self.model.noter.decode(
                tgt_in, memory, self._causal_mask(tgt_in.shape[1]), tgt_pad, src_pad
            )
            K, T, H, V = logits.shape
            tr = F.cross_entropy(
                logits.reshape(K * T * H, V),
                labels.reshape(K * T * H),
                ignore_index=Vocab.PAD,
            )
            with torch.no_grad():
                mask = labels != Vocab.PAD
                accuracy = (
                    (logits.argmax(-1) == labels) & mask
                ).sum() / mask.sum().clamp(min=1)

        det_loss = det.total()
        total = self.config.lambda_det * det_loss + self.config.lambda_tr * tr

        # Detection-quality metrics — watch these for drift while the
        # transcription loss tugs on the boxes during joint fine-tuning.
        with torch.no_grad():
            sys_iou = mean_sys_iou(stave_tb, sys_lr, gt_sys, gt_assign, assign_q)
            stave_err_px = (
                mean_stave_l1(stave_tb, gt_stave, assign_q)
                * self.config.staffer.image_shape[0]
            )

        self.log(f"{stage}/loss", total, prog_bar=True)
        self.log(f"{stage}/det_loss", det_loss, prog_bar=True)
        self.log(f"{stage}/tr_loss", tr, prog_bar=True)
        self.log(f"{stage}/accuracy", accuracy, prog_bar=True)
        self.log(f"{stage}/sys_iou", sys_iou)
        self.log(f"{stage}/stave_err_px", stave_err_px)
        for f in fields(det):
            self.log(f"{stage}/{f.name}", getattr(det, f.name))
        if stage == "train":
            self.log("train/lr", self.trainer.optimizers[0].param_groups[0]["lr"])
        return total

    def on_train_batch_start(self, batch: tuple, batch_idx: int) -> None:
        # Unfreeze once past the threshold. On resume, global_step is restored, so a
        # run resumed beyond freeze_staffer_steps unfreezes on its first batch.
        if (
            self._staffer_frozen
            and self.global_step >= self.config.freeze_staffer_steps
        ):
            self.model.staffer.requires_grad_(True)
            self._staffer_frozen = False

    def training_step(self, batch: tuple, batch_idx: int) -> Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        self._step(batch, "val")

    @torch.no_grad()
    def predict(self, image: Tensor, use_beam: bool = True) -> tuple[Tensor, Tensor, Tensor]:
        """End-to-end inference for a single page: detect → crop → transcribe.

        ``image``: ``(1, 1, H, W)``. Returns ``(boxes, tokens, owners)``:
          ``boxes`` ``(K, 5)`` — ``[batch_idx, left, top, right, bot]`` px, active
          staves top-to-bottom; ``tokens`` ``(K, T, max_chords)`` — generated ids,
          SOS stripped; ``owners`` ``(K,)`` — system index per stave.
          ``K`` is the detected stave count (0 if none fired).
        """
        stave_tb, stave_logits, boundary_logits, sys_lr, _ = self.model.detect(image)
        sel, owners = active_grouping(
            stave_tb[0], stave_logits[0], boundary_logits[0], sys_lr.shape[1]
        )
        hw = (int(image.shape[-2]), int(image.shape[-1]))
        boxes = build_stave_boxes(stave_tb, sys_lr, [sel], [owners], hw)
        if boxes.shape[0] == 0:
            tokens = image.new_zeros(
                (0, self.config.noter.max_seqlen - 1, self.config.noter.max_chords),
                dtype=torch.long,
            )
            return boxes, tokens, torch.empty(0, dtype=torch.long, device=image.device)
        crops, widths = self.model.crop(image, boxes)
        generate = self._generate_beam if use_beam else self._generate_greedy
        return boxes, generate(crops, widths), owners

    @torch.no_grad()
    def _generate_greedy(self, crops: Tensor, widths: Tensor) -> Tensor:
        """Autoregressively decode token sequences for K staff crops (greedy)."""
        c = self.config.noter
        K = crops.shape[0]
        generated = torch.full(
            (K, 1, c.max_chords), Vocab.SOS, dtype=torch.long, device=crops.device
        )
        memory, src_pad = self.model.noter.encode(crops, widths)
        done = torch.zeros(K, dtype=torch.bool, device=crops.device)
        for _ in range(c.max_seqlen - 1):
            tgt_pad = (generated == Vocab.SIL).all(dim=-1)
            logits = self.model.noter.decode(
                generated,
                memory,
                self._causal_mask(generated.shape[1]),
                tgt_pad,
                src_pad,
            )
            next_tokens = logits[:, -1, :, :].argmax(dim=-1)  # (K, max_chords)
            next_tokens[done] = Vocab.EOS
            done = done | (next_tokens[:, 0] == Vocab.EOS)
            generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)
            if bool(done.all()):
                break
        return generated[:, 1:]  # strip SOS

    @torch.no_grad()
    def _generate_beam(self, crops: Tensor, widths: Tensor, beam_width: int = 4) -> Tensor:
        """Autoregressively decode token sequences for K staff crops (beam search).

        Processes each stave independently so peak memory scales with beam_width,
        not K*beam_width.  Beams over slot 0 (the primary/structural token that
        governs null records and sequence length); slots 1+ are decoded greedily
        from the selected beam's logits.
        """
        return torch.cat(
            [
                self._beam_single(crops[k : k + 1], widths[k : k + 1], beam_width)
                for k in range(crops.shape[0])
            ],
            dim=0,
        )

    @torch.no_grad()
    def _beam_single(self, crop: Tensor, width: Tensor, B: int) -> Tensor:
        """Beam search for one stave crop → (1, T, max_chords)."""
        c = self.config.noter
        device = crop.device

        memory, src_pad = self.model.noter.encode(crop, width)
        memory = memory.repeat_interleave(B, dim=0)   # (B, S, D)
        src_pad = src_pad.repeat_interleave(B, dim=0)  # (B, S)

        generated = torch.full((B, 1, c.max_chords), Vocab.SOS, dtype=torch.long, device=device)
        scores = torch.full((B,), float("-inf"), device=device)
        scores[0] = 0.0
        done = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(c.max_seqlen - 1):
            tgt_pad = (generated == Vocab.SIL).all(dim=-1)
            logits = self.model.noter.decode(
                generated,
                memory,
                self._causal_mask(generated.shape[1]),
                tgt_pad,
                src_pad,
            )
            step_logits = logits[:, -1, :, :]  # (B, max_chords, V)
            V = step_logits.shape[-1]

            slot0_lp = step_logits[:, 0, :].log_softmax(-1)  # (B, V)
            slot0_lp[done] = float("-inf")
            slot0_lp[done, Vocab.EOS] = 0.0

            # (B, V) candidates → keep top B
            top_scores, top_idx = (scores.unsqueeze(-1) + slot0_lp).view(B * V).topk(B)
            beam_from = top_idx // V
            token0 = top_idx % V

            scores = top_scores
            done = done[beam_from]
            generated = generated[beam_from]

            next_tokens = step_logits[beam_from].argmax(-1)  # (B, max_chords)
            next_tokens[:, 0] = token0
            next_tokens[done] = Vocab.EOS
            done = done | (next_tokens[:, 0] == Vocab.EOS)

            generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)
            if bool(done.all()):
                break

        best = generated[0:1, 1:]  # (1, T, max_chords); strip SOS
        # Pad to max_seqlen - 1 so staves can be concatenated across independent runs.
        pad_len = c.max_seqlen - 1 - best.shape[1]
        if pad_len > 0:
            pad = torch.full((1, pad_len, c.max_chords), Vocab.PAD, dtype=torch.long, device=device)
            best = torch.cat([best, pad], dim=1)
        return best

    @classmethod
    def load_from_checkpoints(
        cls, config: ScorerConfig, staffer_ckpt: Path, noter_ckpt: Path
    ) -> "ScorerModule":
        """Build a Scorer and load both standalone checkpoints whole.

        Each Lightning checkpoint stores its model under a ``model.`` prefix
        (``StafferModule.model`` / ``NoterModule.model``); we strip it and load the
        state dict straight into the matching Scorer sub-module — every key transfers.
        """
        torch.serialization.add_safe_globals([InterpolationMode])
        module = cls(config)

        def submodel_state(ckpt: Path) -> dict[str, Tensor]:
            sd = torch.load(ckpt, weights_only=False, map_location="cpu")["state_dict"]
            prefix = "model."
            return {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}

        module.model.staffer.load_state_dict(submodel_state(staffer_ckpt))
        module.model.noter.load_state_dict(submodel_state(noter_ckpt))
        return module

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
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
