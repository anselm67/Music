"""Lignhtning module for the Staffer model."""

import math
from dataclasses import fields

import lightning as L
import torch
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from .staffer_loss import StafferLoss, assign_staves, generalized_iou
from .staffer_model import StafferConfig, StafferModel


class StafferModule(L.LightningModule):
    def __init__(self, config: StafferConfig) -> None:
        super().__init__()
        self.config = config
        self.model = StafferModel(config)
        self.loss_fn = StafferLoss(config)
        self.save_hyperparameters(config.asdict())

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        return self.model(x)

    def _step(self, batch: tuple, stage: str) -> Tensor:
        images, gt_sys_boxes, gt_stave_boxes, gt_assign = batch
        (
            pred_stave_tb,
            pred_stave_logits,
            pred_boundary_logits,
            pred_sys_lr,
            pred_sys_logits,
        ) = self.model(images)

        # Route each GT stave to the query whose fixed anchor sits nearest it. The
        # same assignment drives both the loss and the IoU/L1 metrics.
        anchor_c = self.model.heads.anchor_centers()
        assign_q = [
            assign_staves(anchor_c, gt_stave_boxes[i], int((gt_assign[i] != -1).sum()))
            for i in range(len(gt_assign))
        ]

        loss = self.loss_fn.forward(
            pred_stave_tb,
            pred_stave_logits,
            pred_boundary_logits,
            pred_sys_lr,
            pred_sys_logits,
            gt_sys_boxes,
            gt_stave_boxes,
            gt_assign,
            assign_q,
        )

        # IoU metrics — the system box is derived from its staves' hull.
        sys_iou = self._mean_sys_iou(
            pred_stave_tb, pred_sys_lr, gt_sys_boxes, gt_assign, assign_q
        )
        stave_err_px = (
            self._mean_stave_l1(pred_stave_tb, gt_stave_boxes, assign_q)
            * self.config.image_shape[0]
        )

        self.log(f"{stage}/loss", loss.total(), prog_bar=True)
        self.log(f"{stage}/sys_iou", sys_iou)
        # The clean mean stave-edge error in px. Logged under a distinct key from
        # the `stave_l1` loss term below: LossDict has a `stave_l1` field, so the
        # fields(loss) loop also emits `{stage}/stave_l1`. They previously shared
        # the key — Lightning silently kept the (multiplied) loss term and the
        # clean metric was lost. See docs/staffer-training.html.
        self.log(f"{stage}/stave_err_px", stave_err_px)
        for f in fields(loss):
            self.log(f"{stage}/{f.name}", getattr(loss, f.name))

        if stage == "train":
            self.log("train/lr", self.trainer.optimizers[0].param_groups[0]["lr"])

        return loss.total()

    def _mean_sys_iou(
        self,
        pred_stave_tb: Tensor,  # (B, M, 2) — top, bottom
        pred_sys_lr: Tensor,  # (B, N, 2) — left, right
        gt_boxes: list[Tensor],  # list of (N, 4) padded
        gt_assign: list[Tensor],
        assign_q: list[Tensor],  # list of (G,) query slot owning each GT stave
    ) -> Tensor:
        """IoU of the derived system boxes (hull of each system's staves)."""
        ious = []
        for i in range(pred_stave_tb.shape[0]):
            assign = gt_assign[i]
            q = assign_q[i]
            num_gt_staves = q.shape[0]
            num_gt_sys = int(assign[assign != -1].max().item()) + 1
            a = assign[:num_gt_staves]
            tb = pred_stave_tb[i][q]
            derived = []
            for j in range(num_gt_sys):
                mask = a == j
                top = tb[mask][:, 0].min()
                bot = tb[mask][:, 1].max()
                derived.append(
                    torch.stack([pred_sys_lr[i][j, 0], top, pred_sys_lr[i][j, 1], bot])
                )
            matched = torch.stack(derived)
            gt = gt_boxes[i][:num_gt_sys]
            iou = generalized_iou(matched, gt).clamp(min=0).mean()
            ious.append(1.0 - iou)
        return torch.stack(ious).mean()

    def _mean_stave_l1(
        self,
        pred_tb: Tensor,  # (B, M, 2) — top, bottom
        gt_boxes: list[Tensor],  # list of (M, 4) padded
        assign_q: list[Tensor],  # list of (G,) query slot owning each GT stave
    ) -> Tensor:
        """Mean L1 error on (top, bottom) across GT staves."""
        errors = []
        for i in range(pred_tb.shape[0]):
            q = assign_q[i]
            matched = pred_tb[i][q]
            gt = gt_boxes[i][: q.shape[0], [1, 3]]
            errors.append(torch.abs(matched - gt).mean())
        return torch.stack(errors).mean()

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
