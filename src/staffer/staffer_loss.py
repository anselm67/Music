"""Loss module for the Staffer model."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .staffer_model import StafferConfig


def relative_stave_yh(
    gt_stave_boxes: Tensor,  # (M, 4) padded cxcywh
    gt_assign: Tensor,       # (M,) padded with -1
    gt_sys_boxes: Tensor,    # (N, 4) padded cxcywh
    num_gt: int,
) -> Tensor:                 # (num_gt, 2) — cy_delta, h
    """GT target for stave prediction: cy relative to parent system, plus h."""
    parent_cy = gt_sys_boxes[gt_assign[:num_gt].clamp(min=0), 1]
    gt_cy_delta = gt_stave_boxes[:num_gt, 1] - parent_cy
    return torch.stack([gt_cy_delta, gt_stave_boxes[:num_gt, 3]], dim=1)


@dataclass
class LossDict:
    sys_box: Tensor
    sys_giou: Tensor
    sys_obj: Tensor
    stave_l1: Tensor
    stave_obj: Tensor
    assign: Tensor

    def total(self) -> Tensor:
        return (
            self.sys_box
            + self.sys_giou
            + self.sys_obj
            + self.stave_l1
            + self.stave_obj
            + self.assign
        )


def box_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def generalized_iou(pred: Tensor, target: Tensor) -> Tensor:
    """Generalized Inter over Union distance between two sets of boxes.

    Args:
        pred (Tensor): Predicted boxes (P, 4)
        target (Tensor): Ground truth (G, 4)

    Returns:
        Tensor: Loss.
    """
    pred = box_cxcywh_to_xyxy(pred)
    target = box_cxcywh_to_xyxy(target)
    inter_x1 = torch.max(pred[:, 0], target[:, 0])
    inter_y1 = torch.max(pred[:, 1], target[:, 1])
    inter_x2 = torch.min(pred[:, 2], target[:, 2])
    inter_y2 = torch.min(pred[:, 3], target[:, 3])
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    area_pred = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    area_tgt = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = area_pred + area_tgt - inter
    iou = inter / union.clamp(min=1e-6)
    enclosing_x1 = torch.min(pred[:, 0], target[:, 0])
    enclosing_y1 = torch.min(pred[:, 1], target[:, 1])
    enclosing_x2 = torch.max(pred[:, 2], target[:, 2])
    enclosing_y2 = torch.max(pred[:, 3], target[:, 3])
    enclosing = (enclosing_x2 - enclosing_x1) * (enclosing_y2 - enclosing_y1)
    return iou - (enclosing - union) / enclosing.clamp(min=1e-6)


class StafferLoss(nn.Module):
    def __init__(self, config: StafferConfig):
        super().__init__()
        self.config = config

    def _box_loss(
        self,
        pred_boxes: Tensor,  # (N, 4)
        pred_logits: Tensor,  # (N, 1)
        gt_boxes: Tensor,  # (N, 4) padded
        num_gt: int,
        num_queries: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        matched = pred_boxes[:num_gt]
        gt = gt_boxes[:num_gt]

        box_loss = F.l1_loss(matched, gt)
        giou_loss = (1 - generalized_iou(matched, gt)).mean()

        obj_target = torch.zeros(num_queries, device=pred_boxes.device)
        obj_target[:num_gt] = 1.0
        obj_loss = F.binary_cross_entropy_with_logits(
            pred_logits.squeeze(-1), obj_target
        )

        return box_loss, giou_loss, obj_loss

    def _stave_yh_loss(
        self,
        pred_yh: Tensor,  # (M, 2) — cy_delta, h predictions
        pred_logits: Tensor,  # (M, 1)
        gt_boxes: Tensor,  # (M, 4) padded cxcywh
        gt_assign: Tensor,  # (M,) padded with -1
        gt_sys_boxes: Tensor,  # (N, 4) padded cxcywh
        num_gt_staves: int,
        num_queries: int,
    ) -> tuple[Tensor, Tensor]:
        gt_yh = relative_stave_yh(gt_boxes, gt_assign, gt_sys_boxes, num_gt_staves)
        yh_loss = F.l1_loss(pred_yh[:num_gt_staves], gt_yh)

        obj_target = torch.zeros(num_queries, device=pred_yh.device)
        obj_target[:num_gt_staves] = 1.0
        obj_loss = F.binary_cross_entropy_with_logits(
            pred_logits.squeeze(-1), obj_target
        )

        return yh_loss, obj_loss

    def _assignment_loss(
        self,
        pred_assign: Tensor,  # (M, N)
        gt_assign: Tensor,  # (M,) padded with -1
        num_gt_staves: int,
    ) -> Tensor:
        return F.cross_entropy(
            pred_assign[:num_gt_staves],
            gt_assign[:num_gt_staves],
        )

    def forward(
        self,
        pred_sys_boxes: Tensor,
        pred_sys_logits: Tensor,
        pred_stave_yh: Tensor,  # (B, M, 2) — cy_delta, h
        pred_stave_logits: Tensor,
        pred_assign: Tensor,
        gt_sys_boxes: Tensor,
        gt_stave_boxes: Tensor,
        gt_assign: Tensor,
    ) -> LossDict:
        B = pred_sys_boxes.shape[0]

        sys_box = torch.tensor(0.0, device=pred_sys_boxes.device)
        sys_giou = torch.tensor(0.0, device=pred_sys_boxes.device)
        sys_obj = torch.tensor(0.0, device=pred_sys_boxes.device)
        stave_l1 = torch.tensor(0.0, device=pred_sys_boxes.device)
        stave_obj = torch.tensor(0.0, device=pred_sys_boxes.device)
        assign = torch.tensor(0.0, device=pred_sys_boxes.device)

        for i in range(B):
            num_gt_staves = int((gt_assign[i] != -1).sum().item())
            num_gt_sys = int(gt_assign[i][gt_assign[i] != -1].max().item()) + 1

            b, g, o = self._box_loss(
                pred_sys_boxes[i],
                pred_sys_logits[i],
                gt_sys_boxes[i],
                num_gt_sys,
                self.config.num_system_queries,
            )
            sys_box = sys_box + b
            sys_giou = sys_giou + g
            sys_obj = sys_obj + o

            b, o = self._stave_yh_loss(
                pred_stave_yh[i],
                pred_stave_logits[i],
                gt_stave_boxes[i],
                gt_assign[i],
                gt_sys_boxes[i],
                num_gt_staves,
                self.config.num_stave_queries,
            )
            stave_l1 = stave_l1 + b
            stave_obj = stave_obj + o

            assign = assign + self._assignment_loss(
                pred_assign[i],
                gt_assign[i],
                num_gt_staves,
            )

        return LossDict(
            sys_box=self.config.box_loss_multiplier * (sys_box / B),
            sys_giou=self.config.box_loss_multiplier * (sys_giou / B),
            sys_obj=sys_obj / B,
            stave_l1=self.config.box_loss_multiplier * (stave_l1 / B),
            stave_obj=stave_obj / B,
            assign=assign / B,
        )
