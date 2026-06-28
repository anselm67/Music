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

from noter import NoterModel, Vocab
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


def group_systems(
    assign_q: list[Tensor],
    sys_ids: list[Tensor],
    max_staves: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Group the flat per-stave batch into systems for the cross-stave decode.

    The flat batch is ordered page-by-page then by each page's GT-stave order — the
    same order ``build_stave_boxes`` and the target ``cat`` use. Staves sharing a page
    and a system index form one system, capped and padded to ``max_staves``. Returns
    ``(grouped_idx, stave_mask)`` both ``(num_systems, max_staves)``: ``grouped_idx``
    indexes the flat batch (0 on pad slots — the caller masks them via ``stave_mask``,
    which is True only on real staves).
    """
    groups: list[list[int]] = []
    k = 0
    for q, sid in zip(assign_q, sys_ids):
        by_sys: dict[int, list[int]] = {}
        for j in range(q.shape[0]):
            by_sys.setdefault(int(sid[j].item()), []).append(k)
            k += 1
        groups.extend(by_sys[s][:max_staves] for s in sorted(by_sys))
    ng = len(groups)
    grouped_idx = torch.zeros((ng, max_staves), dtype=torch.long, device=device)
    stave_mask = torch.zeros((ng, max_staves), dtype=torch.bool, device=device)
    for g, members in enumerate(groups):
        for slot, kk in enumerate(members):
            grouped_idx[g, slot] = kk
            stave_mask[g, slot] = True
    return grouped_idx, stave_mask


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
            # Cut the transcription gradient to the box coords so it can't shrink
            # the detector's boxes; det loss alone trains geometry.
            crops, widths = self.model.crop(image, boxes.detach())
            memory, mem_pad = self.model.noter.encode(crops, widths)  # (K,P,D),(K,P)
            # Group the flat staves into systems so the noter couples each system's
            # staves in the cross-stave decode (shared barline grid).
            max_staves = self.config.noter.max_staves
            grouped_idx, stave_mask = group_systems(
                assign_q, sys_ids, max_staves, image.device
            )
            ng = grouped_idx.shape[0]
            mem_g = memory[grouped_idx].reshape(ng * max_staves, *memory.shape[1:])
            pad_g = mem_pad[grouped_idx].reshape(ng * max_staves, -1)
            tgt = targets[grouped_idx]  # (ng, max_staves, T, max_chords)
            tgt_in = tgt[:, :, :-1]
            labels = tgt[:, :, 1:].clone()
            labels[~stave_mask] = Vocab.PAD  # exclude padded staves from the loss
            logits = self.model.noter.decode(
                tgt_in, mem_g, pad_g, stave_mask, self._causal_mask(tgt_in.shape[2])
            )  # (ng, max_staves, T-1, max_chords, V)
            V = logits.shape[-1]
            tr = F.cross_entropy(
                logits.reshape(-1, V), labels.reshape(-1), ignore_index=Vocab.PAD
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
    def predict(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """End-to-end inference for a single page: detect → crop → transcribe.

        ``image``: ``(1, 1, H, W)``. Returns ``(boxes, tokens, owners)``:
          ``boxes`` ``(K, 5)`` — ``[batch_idx, left, top, right, bot]`` px, active
          staves top-to-bottom; ``tokens`` ``(K, T, max_chords)`` — generated ids,
          SOS stripped; ``owners`` ``(K,)`` — system index per stave.
          ``K`` is the detected stave count (0 if none fired).

        Each detected system's staves are decoded together in lockstep (the cross-stave
        noter), so cross-staff barline agreement is structural — superseding the former
        per-stave beam + agreement reranker.
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
        tokens = self._generate_grouped(crops, widths, owners)
        return boxes, tokens, owners

    @torch.no_grad()
    def _generate_grouped(
        self, crops: Tensor, widths: Tensor, owners: Tensor
    ) -> Tensor:
        """Lockstep-decode each system's staves together → ``(K, T-1, max_chords)``.

        Staves sharing an ``owners`` index are decoded as one system: row ``t`` is
        generated for all its staves at once, each seeing its siblings' rows ``< t``
        through the cross-stave attention. Returns the tokens in the original (K) stave
        order, SOS stripped.
        """
        c = self.config.noter
        K, device = crops.shape[0], crops.device
        smax = c.max_staves
        # Group flat staves by owner into (Ng, smax) slots (pad slots index 0, masked).
        by_sys: dict[int, list[int]] = {}
        for k in range(K):
            by_sys.setdefault(int(owners[k].item()), []).append(k)
        groups = [by_sys[s][:smax] for s in sorted(by_sys)]
        ng = len(groups)
        grouped_idx = torch.zeros((ng, smax), dtype=torch.long, device=device)
        stave_mask = torch.zeros((ng, smax), dtype=torch.bool, device=device)
        for g, members in enumerate(groups):
            for slot, k in enumerate(members):
                grouped_idx[g, slot] = k
                stave_mask[g, slot] = True

        memory, mem_pad = self.model.noter.encode(crops, widths)  # (K,P,D),(K,P)
        mem_g = memory[grouped_idx].reshape(ng * smax, *memory.shape[1:])
        pad_g = mem_pad[grouped_idx].reshape(ng * smax, -1)

        generated = torch.full(
            (ng, smax, 1, c.max_chords), Vocab.SOS, dtype=torch.long, device=device
        )
        done = ~stave_mask.clone()  # padded staves emit EOS immediately
        for _ in range(c.max_seqlen - 1):
            T = generated.shape[2]
            logits = self.model.noter.decode(
                generated, mem_g, pad_g, stave_mask, self._causal_mask(T)
            )
            next_tokens = logits[:, :, -1, :, :].argmax(dim=-1)  # (ng, smax, mc)
            next_tokens[done] = Vocab.EOS
            done = done | (next_tokens[..., 0] == Vocab.EOS)
            generated = torch.cat([generated, next_tokens.unsqueeze(2)], dim=2)
            if bool(done.all()):
                break

        gen = generated[:, :, 1:]  # (ng, smax, T-1, mc); strip SOS
        out = gen.new_full((K, gen.shape[2], c.max_chords), Vocab.PAD)
        for g, members in enumerate(groups):
            for slot, k in enumerate(members):
                out[k] = gen[g, slot]
        return out

    @classmethod
    def load_from_checkpoints(
        cls, config: ScorerConfig, staffer_ckpt: Path, noter_ckpt: Path
    ) -> "ScorerModule":
        """Build a Scorer and load both standalone checkpoints.

        Each Lightning checkpoint stores its model under a ``model.`` prefix
        (``StafferModule.model`` / ``NoterModule.model``); we strip it and load the
        state dict into the matching Scorer sub-module. The staffer transfers whole. A
        single-stave noter checkpoint is remapped onto the cross-stave encoder/decoder
        (``NoterModel.remap_legacy_state_dict``) and loaded non-strict — the cross-stave
        params have no legacy counterpart and stay at init (zero gate ⇒ identical to the
        base noter); anything missing beyond those is an error.
        """
        torch.serialization.add_safe_globals([InterpolationMode])
        module = cls(config)

        def submodel_state(ckpt: Path) -> dict[str, Tensor]:
            sd = torch.load(ckpt, weights_only=False, map_location="cpu")["state_dict"]
            prefix = "model."
            return {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}

        module.model.staffer.load_state_dict(submodel_state(staffer_ckpt))
        noter_state = NoterModel.remap_legacy_state_dict(submodel_state(noter_ckpt))
        result = module.model.noter.load_state_dict(noter_state, strict=False)
        stray = [
            k
            for k in result.missing_keys
            if not any(s in k for s in ("cross_stave", "norm_xs", "xs_gate"))
        ]
        if stray or result.unexpected_keys:
            raise RuntimeError(
                f"noter checkpoint mismatch: missing {stray}, "
                f"unexpected {result.unexpected_keys}"
            )
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
