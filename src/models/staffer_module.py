import math

import lightning as L
import torch
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR




from .staffer_loss import HierarchicalLoss, generalized_iou
from .staffer_model import Config, HierarchicalDETR


class StafferModule(L.LightningModule):

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.model = HierarchicalDETR(config)
        self.loss_fn = HierarchicalLoss(config)
        self.save_hyperparameters(config.asdict())

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        return self.model(x)

    def _step(self, batch: tuple, stage: str) -> Tensor:
        images, gt_sys_boxes, gt_stave_boxes, gt_assign = batch
        pred_sys_boxes, pred_sys_logits, pred_stave_boxes, pred_stave_logits, pred_assign = self.model(
            images)

        loss = self.loss_fn(
            pred_sys_boxes, pred_sys_logits,
            pred_stave_boxes, pred_stave_logits,
            pred_assign,
            gt_sys_boxes, gt_stave_boxes, gt_assign,
        )

        # IoU metrics
        sys_iou = self._mean_iou(
            pred_sys_boxes, gt_sys_boxes, gt_assign, is_sys=True)
        stave_iou = self._mean_iou(
            pred_stave_boxes, gt_stave_boxes, gt_assign, is_sys=False)

        # Containment metric
        containment = self.loss_fn._containment_loss(
            pred_sys_boxes[0], pred_stave_boxes[0],
            gt_assign[0],
            int((gt_assign[0] != -1).sum().item()),
            int(gt_assign[0][gt_assign[0] != -1].max().item()) + 1,
        )

        self.log(f"{stage}/loss", loss, prog_bar=True)
        self.log(f"{stage}/sys_iou", sys_iou, prog_bar=True)
        self.log(f"{stage}/stave_iou", stave_iou, prog_bar=True)
        self.log(f"{stage}/containment", containment, prog_bar=True)

        return loss

    def _mean_iou(
        self,
        pred_boxes: Tensor,        # (B, N, 4)
        gt_boxes: list[Tensor],    # list of (N, 4) padded
        gt_assign: list[Tensor],   # list of (M,) padded with -1
        is_sys: bool,
    ) -> Tensor:
        ious = []
        for i in range(pred_boxes.shape[0]):
            if is_sys:
                num_gt = int(gt_assign[i][gt_assign[i] != -1].max().item()) + 1
            else:
                num_gt = int((gt_assign[i] != -1).sum().item())
            matched = pred_boxes[i][:num_gt]
            gt = gt_boxes[i][:num_gt]
            iou = generalized_iou(matched, gt).clamp(min=0).mean()
            ious.append(iou)
        return torch.stack(ious).mean()

    def training_step(self, batch: tuple, batch_idx: int) -> Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        self._step(batch, "val")

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.config.lr,
                          weight_decay=self.config.weight_decay)

        def lr_lambda(step: int) -> float:
            if step < self.config.warmup_steps:
                return step / max(1, self.config.warmup_steps)
            progress = (step - self.config.warmup_steps) / \
                max(1, self.config.max_steps - self.config.warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            }
        }


