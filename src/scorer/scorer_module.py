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
    def predict(
        self,
        image: Tensor,
        use_beam: bool = True,
        barline_ids: set[int] | None = None,
        beam_width: int = 4,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """End-to-end inference for a single page: detect → crop → transcribe.

        ``image``: ``(1, 1, H, W)``. Returns ``(boxes, tokens, owners)``:
          ``boxes`` ``(K, 5)`` — ``[batch_idx, left, top, right, bot]`` px, active
          staves top-to-bottom; ``tokens`` ``(K, T, max_chords)`` — generated ids,
          SOS stripped; ``owners`` ``(K,)`` — system index per stave.
          ``K`` is the detected stave count (0 if none fired).

        When ``barline_ids`` is given, decode with the per-system agreement reranker
        (``_generate_rerank``, which implies beam search); otherwise beam/greedy.
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
        if barline_ids is not None:
            tokens = self._generate_rerank(
                crops, widths, owners, barline_ids, beam_width
            )
        elif use_beam:
            tokens = self._generate_beam(crops, widths, beam_width)
        else:
            tokens = self._generate_greedy(crops, widths)
        return boxes, tokens, owners

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
    def _generate_beam(
        self, crops: Tensor, widths: Tensor, beam_width: int = 4
    ) -> Tensor:
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
    def _beam_some(
        self, crop: Tensor, width: Tensor, beam_width: int, keep: int
    ) -> tuple[Tensor, Tensor]:
        """Beam search for one stave crop → the top-``keep`` candidates + scores.

        Beams over slot 0 (the primary/structural token governing null records and
        sequence length); slots 1+ greedy from the selected beam. Returns
        ``(cands, scores)``, best-first, with ``keep <= beam_width``: ``cands``
        ``(keep, max_seqlen-1, max_chords)`` is SOS-stripped and PAD-padded so
        independent staves concatenate; ``scores`` ``(keep,)`` is the slot-0 logprob.
        """
        assert keep <= beam_width
        c = self.config.noter
        device = crop.device
        B = beam_width

        memory, src_pad = self.model.noter.encode(crop, width)
        memory = memory.repeat_interleave(B, dim=0)  # (B, S, D)
        src_pad = src_pad.repeat_interleave(B, dim=0)  # (B, S)

        generated = torch.full(
            (B, 1, c.max_chords), Vocab.SOS, dtype=torch.long, device=device
        )
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

            # (B, V) candidates → keep top B, best-first (the rerank relies on
            # the returned candidates being sorted by score).
            top_scores, top_idx = (
                (scores.unsqueeze(-1) + slot0_lp).view(B * V).topk(B, sorted=True)
            )
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

        cands = generated[:keep, 1:]  # (keep, T, max_chords); strip SOS
        # Pad to max_seqlen - 1 so staves can be concatenated across independent runs.
        pad_len = c.max_seqlen - 1 - cands.shape[1]
        if pad_len > 0:
            pad = torch.full(
                (keep, pad_len, c.max_chords),
                Vocab.PAD,
                dtype=torch.long,
                device=device,
            )
            cands = torch.cat([cands, pad], dim=1)
        return cands, scores[:keep]

    @torch.no_grad()
    def _beam_single(self, crop: Tensor, width: Tensor, B: int) -> Tensor:
        """Beam search for one stave crop → best candidate ``(1, max_seqlen-1, mc)``."""
        cands, _ = self._beam_some(crop, width, B, 1)
        return cands

    @torch.no_grad()
    def _generate_rerank(
        self,
        crops: Tensor,
        widths: Tensor,
        owners: Tensor,
        barline_ids: set[int],
        beam_width: int = 4,
    ) -> Tensor:
        """Per-system agreement rerank over slot-0 beam candidates.

        Each stave keeps its top-``keep`` beam candidates + scores. The staves of a
        system (sharing an ``owners`` index) must share one barline skeleton, so pick
        the barline signature present in *every* stave's candidates that maximises the
        summed logprob, and emit each stave's best candidate for it. Falls back to each
        stave's argmax (beam 0) when the staves share no signature. Returns
        ``(K, max_seqlen-1, max_chords)`` in stave order, like ``_generate_beam``.
        """
        cands: list[Tensor] = []
        scores: list[Tensor] = []
        for k in range(crops.shape[0]):
            ck, sk = self._beam_some(
                crops[k : k + 1], widths[k : k + 1], beam_width, beam_width
            )
            cands.append(ck)
            scores.append(sk)

        def signature(seq: Tensor) -> tuple[int, ...]:
            """Timestep indices carrying a barline in slot 0, up to EOS."""
            positions: list[int] = []
            for t in range(seq.shape[0]):
                tok = int(seq[t, 0].item())
                if tok == Vocab.EOS:
                    break
                if tok in barline_ids:
                    positions.append(t)
            return tuple(positions)

        chosen = [0] * crops.shape[0]  # candidate index per stave; default = argmax
        by_sys: dict[int, list[int]] = {}
        for k in range(crops.shape[0]):
            by_sys.setdefault(int(owners[k].item()), []).append(k)

        for ks in by_sys.values():
            if len(ks) < 2:
                continue  # no cross-staff agreement to enforce
            # per stave: barline signature -> (best logprob, candidate index)
            best: list[dict[tuple[int, ...], tuple[float, int]]] = []
            for k in ks:
                d: dict[tuple[int, ...], tuple[float, int]] = {}
                for b in range(cands[k].shape[0]):
                    sig = signature(cands[k][b])
                    sc = float(scores[k][b].item())
                    if sig not in d or sc > d[sig][0]:
                        d[sig] = (sc, b)
                best.append(d)
            sets = [set(d) for d in best]
            common = sets[0].intersection(*sets[1:])
            if not common:
                continue  # no shared skeleton → keep each stave's argmax
            target = max(common, key=lambda s: sum((d[s][0] for d in best), 0.0))
            for k, d in zip(ks, best):
                chosen[k] = d[target][1]

        return torch.cat(
            [cands[k][chosen[k] : chosen[k] + 1] for k in range(crops.shape[0])], dim=0
        )

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
