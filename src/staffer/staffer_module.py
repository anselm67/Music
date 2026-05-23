"""Lignhtning module for the Staffer model."""

import math
from dataclasses import fields

import lightning as L
import torch
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from .staffer_loss import StafferLoss, generalized_iou, relative_stave_yh
from .staffer_model import StafferConfig, StafferModel


class StafferModule(L.LightningModule):
    def __init__(self, config: StafferConfig) -> None:
        super().__init__()
        self.config = config
        self.model = StafferModel(config)
        self.loss_fn = StafferLoss(config)
        self.save_hyperparameters(config.asdict())

    def forward(
        self, x: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        return self.model(x)

    def _step(self, batch: tuple, stage: str) -> Tensor:
        images, gt_sys_boxes, gt_stave_boxes, gt_assign = batch
        (
            pred_sys_boxes,
            pred_sys_logits,
            pred_stave_yh,
            pred_stave_logits,
            pred_assign,
        ) = self.model(images)

        loss = self.loss_fn.forward(
            pred_sys_boxes,
            pred_sys_logits,
            pred_stave_yh,
            pred_stave_logits,
            pred_assign,
            gt_sys_boxes,
            gt_stave_boxes,
            gt_assign,
        )

        # IoU metrics
        sys_iou = self._mean_sys_iou(pred_sys_boxes, gt_sys_boxes, gt_assign)
        stave_l1 = self._mean_stave_l1(pred_stave_yh, gt_stave_boxes, gt_assign, gt_sys_boxes)

        self.log(f"{stage}/loss", loss.total(), prog_bar=True)
        self.log(f"{stage}/sys_iou", sys_iou)
        self.log(f"{stage}/stave_l1", stave_l1)
        for f in fields(loss):
            self.log(f"{stage}/{f.name}", getattr(loss, f.name))

        if stage == "train":
            self.log("train/lr", self.trainer.optimizers[0].param_groups[0]["lr"])

        return loss.total()

    def _mean_sys_iou(
        self,
        pred_boxes: Tensor,      # (B, N, 4)
        gt_boxes: list[Tensor],  # list of (N, 4) padded
        gt_assign: list[Tensor],
    ) -> Tensor:
        ious = []
        for i in range(pred_boxes.shape[0]):
            num_gt = int(gt_assign[i][gt_assign[i] != -1].max().item()) + 1
            matched = pred_boxes[i][:num_gt]
            gt = gt_boxes[i][:num_gt]
            iou = generalized_iou(matched, gt).clamp(min=0).mean()
            ious.append(1.0 - iou)
        return torch.stack(ious).mean()

    def _mean_stave_l1(
        self,
        pred_yh: Tensor,             # (B, M, 2) — cy_delta, h
        gt_boxes: list[Tensor],      # list of (M, 4) padded cxcywh
        gt_assign: list[Tensor],
        gt_sys_boxes: list[Tensor],  # list of (N, 4) padded cxcywh
    ) -> Tensor:
        """Mean L1 error on (cy_delta, h) across GT staves."""
        errors = []
        for i in range(pred_yh.shape[0]):
            num_gt = int((gt_assign[i] != -1).sum().item())
            matched = pred_yh[i][:num_gt]
            gt = relative_stave_yh(gt_boxes[i], gt_assign[i], gt_sys_boxes[i], num_gt)
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
