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

from kern import NUM_ARTICULATIONS
from noter import NoterModel, Vocab, articulation_loss
from staffer import StafferLoss, assign_staves
from staffer.staffer_loss import generalized_iou

from .scorer_model import ScorerConfig, ScorerModel, build_stave_boxes

# Max staves encoded together during inference. The noter encoder's O(P²) attention
# over the 3072-patch crops (patch_height=4) OOMs a 16GB GPU on pages with many
# staves; chunking the per-crop-independent encode bounds peak memory.
_ENCODE_CHUNK = 4


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
    and a system index form one system, capped at ``max_staves`` and padded to the
    batch-max staff count ``smax`` (≤ ``max_staves``) — padding to the batch-max, not
    the global ceiling, keeps a batch of ≤2-staff systems at ``smax=2`` so bumping
    ``max_staves`` to 4 costs nothing on the common case. Returns ``(grouped_idx,
    stave_mask)`` both ``(num_systems, smax)``: ``grouped_idx`` indexes the flat batch
    (0 on pad slots — the caller masks them via ``stave_mask``, True only on real
    staves).
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
    smax = max((len(m) for m in groups), default=1)
    grouped_idx = torch.zeros((ng, smax), dtype=torch.long, device=device)
    stave_mask = torch.zeros((ng, smax), dtype=torch.bool, device=device)
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
        image, gt_sys, gt_stave, gt_assign, stave_tokens, stave_arts = batch
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
        arts = torch.cat(
            [stave_arts[i, : assign_q[i].shape[0]] for i in range(B)], dim=0
        )  # (K, T, max_chords, A)

        tr = torch.zeros((), device=image.device)
        accuracy = torch.zeros((), device=image.device)
        art_loss = torch.zeros((), device=image.device)
        art_acc = torch.zeros((), device=image.device)
        art_recall = torch.zeros((), device=image.device)
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
            ng, smax = grouped_idx.shape
            mem_g = memory[grouped_idx].reshape(ng * smax, *memory.shape[1:])
            pad_g = mem_pad[grouped_idx].reshape(ng * smax, -1)
            tgt = targets[grouped_idx]  # (ng, smax, T, max_chords)
            art = arts[grouped_idx]  # (ng, smax, T, max_chords, A)
            tgt_in = tgt[:, :, :-1]
            labels = tgt[:, :, 1:].clone()
            labels[~stave_mask] = Vocab.PAD  # exclude padded staves from the loss
            # Decode tokens + articulations together; the input note's articulation
            # is fed back alongside its token (shifted, like the teacher-forced tokens).
            logits, art_logits = self.model.noter.decode_both(
                tgt_in,
                mem_g,
                pad_g,
                stave_mask,
                self._causal_mask(tgt_in.shape[2]),
                art[:, :, :-1],
            )  # (ng, smax, T-1, max_chords, V/A)
            V = logits.shape[-1]
            tr = F.cross_entropy(
                logits.reshape(-1, V), labels.reshape(-1), ignore_index=Vocab.PAD
            )
            slot_mask = labels != Vocab.PAD
            with torch.no_grad():
                accuracy = (
                    (logits.argmax(-1) == labels) & slot_mask
                ).sum() / slot_mask.sum().clamp(min=1)
            art_loss, art_acc, art_recall = articulation_loss(
                art_logits, art[:, :, 1:], slot_mask
            )

        det_loss = det.total()
        # tr is the pure token CE (logged as tr_loss); the articulation loss is folded
        # into the transcription term for the total, weighted but logged separately.
        tr_total = tr + self.config.noter.articulation_weight * art_loss
        total = self.config.lambda_det * det_loss + self.config.lambda_tr * tr_total

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
        self.log(f"{stage}/art_loss", art_loss)
        self.log(f"{stage}/art_acc", art_acc)
        self.log(f"{stage}/art_recall", art_recall, prog_bar=True)
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
    def predict(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """End-to-end inference for a single page: detect → crop → transcribe.

        ``image``: ``(1, 1, H, W)``. Returns ``(boxes, tokens, articulations, owners)``:
          ``boxes`` ``(K, 5)`` — ``[batch_idx, left, top, right, bot]`` px, active
          staves top-to-bottom; ``tokens`` ``(K, T, max_chords)`` — generated ids,
          SOS stripped; ``articulations`` ``(K, T, max_chords, A)`` — the row-aligned
          per-note multi-hot; ``owners`` ``(K,)`` — system index per stave.
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
            t, mc = self.config.noter.max_seqlen - 1, self.config.noter.max_chords
            tokens = image.new_zeros((0, t, mc), dtype=torch.long)
            arts = image.new_zeros((0, t, mc, NUM_ARTICULATIONS))
            owners = torch.empty(0, dtype=torch.long, device=image.device)
            return boxes, tokens, arts, owners
        crops, widths = self.model.crop(image, boxes)
        tokens, arts = self._generate_grouped(crops, widths, owners)
        return boxes, tokens, arts, owners

    @torch.no_grad()
    def _generate_grouped(
        self, crops: Tensor, widths: Tensor, owners: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Lockstep-decode each system's staves together → ``(tokens, articulations)``,
        shapes ``(K, T-1, max_chords)`` and ``(K, T-1, max_chords, A)``.

        Staves sharing an ``owners`` index are decoded as one system: row ``t`` is
        generated for all its staves at once, each seeing its siblings' rows ``< t``
        through the cross-stave attention. Returns them in the original (K) stave
        order, SOS stripped.
        """
        c = self.config.noter
        K, device = crops.shape[0], crops.device
        # Group flat staves by owner; cap each system at max_staves and pad to the
        # page's batch-max staff count (a page of ≤2-staff systems stays smax=2).
        by_sys: dict[int, list[int]] = {}
        for k in range(K):
            by_sys.setdefault(int(owners[k].item()), []).append(k)
        groups = [by_sys[s][: c.max_staves] for s in sorted(by_sys)]
        ng = len(groups)
        smax = max((len(m) for m in groups), default=1)
        grouped_idx = torch.zeros((ng, smax), dtype=torch.long, device=device)
        stave_mask = torch.zeros((ng, smax), dtype=torch.bool, device=device)
        for g, members in enumerate(groups):
            for slot, k in enumerate(members):
                grouped_idx[g, slot] = k
                stave_mask[g, slot] = True

        # Encode staves in sub-batches. The encoder's O(P²) self-attention over the
        # 3072-patch crops (patch_height=4) materialises K×heads×3072² scores at once,
        # which OOMs a page with many staves; chunking bounds peak memory. Encode is
        # per-crop independent and P is fixed at 3072, so the chunks concatenate.
        if K <= _ENCODE_CHUNK:
            memory, mem_pad = self.model.noter.encode(crops, widths)  # (K,P,D),(K,P)
        else:
            parts = [
                self.model.noter.encode(
                    crops[i : i + _ENCODE_CHUNK], widths[i : i + _ENCODE_CHUNK]
                )
                for i in range(0, K, _ENCODE_CHUNK)
            ]
            memory = torch.cat([m for m, _ in parts], dim=0)  # (K,P,D)
            mem_pad = torch.cat([p for _, p in parts], dim=0)  # (K,P)
        mem_g = memory[grouped_idx].reshape(ng * smax, *memory.shape[1:])
        pad_g = mem_pad[grouped_idx].reshape(ng * smax, -1)

        generated = torch.full(
            (ng, smax, 1, c.max_chords), Vocab.SOS, dtype=torch.long, device=device
        )
        # Parallel articulation buffer (SOS row = zeros), fed back like the tokens.
        art = torch.zeros(ng, smax, 1, c.max_chords, NUM_ARTICULATIONS, device=device)
        done = ~stave_mask.clone()  # padded staves emit EOS immediately
        for _ in range(c.max_seqlen - 1):
            T = generated.shape[2]
            logits, art_logits = self.model.noter.decode_both(
                generated, mem_g, pad_g, stave_mask, self._causal_mask(T), art
            )
            next_tokens = logits[:, :, -1, :, :].argmax(dim=-1)  # (ng, smax, mc)
            next_tokens[done] = Vocab.EOS
            done = done | (next_tokens[..., 0] == Vocab.EOS)
            generated = torch.cat([generated, next_tokens.unsqueeze(2)], dim=2)
            next_art = (art_logits[:, :, -1] > 0).float()  # (ng, smax, mc, A)
            next_art[done] = 0.0  # finished staves carry no articulation
            art = torch.cat([art, next_art.unsqueeze(2)], dim=2)
            if bool(done.all()):
                break

        gen = generated[:, :, 1:]  # (ng, smax, T-1, mc); strip SOS
        gen_art = art[:, :, 1:]  # (ng, smax, T-1, mc, A); strip SOS
        out = gen.new_full((K, gen.shape[2], c.max_chords), Vocab.PAD)
        out_art = gen_art.new_zeros((K, gen.shape[2], c.max_chords, NUM_ARTICULATIONS))
        for g, members in enumerate(groups):
            for slot, k in enumerate(members):
                out[k] = gen[g, slot]
                out_art[k] = gen_art[g, slot]
        return out, out_art

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
            if not any(
                s in k
                for s in ("cross_stave", "norm_xs", "xs_gate", "art_proj", "art_head")
            )
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
