"""Loss module for the Staffer model."""
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .staffer_model import Config


@dataclass
class LossDict:
    sys_box: Tensor
    sys_giou: Tensor
    sys_obj: Tensor
    stave_box: Tensor
    stave_giou: Tensor
    stave_obj: Tensor
    assign: Tensor
    containment: Tensor
    alignment: Tensor

    def total(self) -> Tensor:
        return (
            self.sys_box + self.sys_giou + self.sys_obj +
            self.stave_box + self.stave_giou + self.stave_obj +
            self.assign + self.containment + self.alignment
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


class HierarchicalLoss(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

    def _box_loss(
        self,
        pred_boxes: Tensor,    # (N, 4)
        pred_logits: Tensor,   # (N, 1)
        gt_boxes: Tensor,      # (N, 4) padded
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
            pred_logits.squeeze(-1), obj_target)

        return box_loss, giou_loss, obj_loss

    def _assignment_loss(
        self,
        pred_assign: Tensor,   # (M, N)
        gt_assign: Tensor,     # (M,) padded with -1
        num_gt_staves: int,
    ) -> Tensor:
        return F.cross_entropy(
            pred_assign[:num_gt_staves],
            gt_assign[:num_gt_staves],
        )

    def _containment_loss(
        self,
        pred_sys_boxes: Tensor,    # (N, 4)
        pred_stave_boxes: Tensor,  # (M, 4)
        gt_assign: Tensor,         # (M,) padded with -1
        num_gt_staves: int,
        num_gt_sys: int,
    ) -> Tensor:
        loss = torch.tensor(0.0, device=pred_sys_boxes.device)

        for sys_idx in range(num_gt_sys):
            sys_box = pred_sys_boxes[sys_idx]
            stave_mask = gt_assign[:num_gt_staves] == sys_idx
            if not stave_mask.any():
                continue

            stave_boxes = pred_stave_boxes[:num_gt_staves][stave_mask]
            sys_xyxy = box_cxcywh_to_xyxy(sys_box.unsqueeze(0)).squeeze(0)
            staves_xyxy = box_cxcywh_to_xyxy(stave_boxes)

            stave_areas = (
                (staves_xyxy[:, 2] - staves_xyxy[:, 0]) *
                (staves_xyxy[:, 3] - staves_xyxy[:, 1])
            ).clamp(min=1e-6)

            inter_x1 = torch.max(
                staves_xyxy[:, 0], sys_xyxy[0].expand_as(staves_xyxy[:, 0]))
            inter_y1 = torch.max(
                staves_xyxy[:, 1], sys_xyxy[1].expand_as(staves_xyxy[:, 1]))
            inter_x2 = torch.min(
                staves_xyxy[:, 2], sys_xyxy[2].expand_as(staves_xyxy[:, 2]))
            inter_y2 = torch.min(
                staves_xyxy[:, 3], sys_xyxy[3].expand_as(staves_xyxy[:, 3]))
            inter_areas = (
                (inter_x2 - inter_x1).clamp(0) *
                (inter_y2 - inter_y1).clamp(0)
            )

            outside_fraction = (stave_areas - inter_areas) / stave_areas
            loss = loss + outside_fraction.mean()

        return loss / num_gt_sys

    def _alignment_loss(
        self,
        pred_sys_boxes: Tensor,
        pred_stave_boxes: Tensor,
        gt_assign: Tensor,
        num_gt_staves: int,
        num_gt_sys: int,
    ) -> Tensor:
        loss = torch.tensor(0.0, device=pred_sys_boxes.device)

        for sys_idx in range(num_gt_sys):
            sys_xyxy = box_cxcywh_to_xyxy(
                pred_sys_boxes[sys_idx].unsqueeze(0)).squeeze(0)
            stave_mask = gt_assign[:num_gt_staves] == sys_idx
            if not stave_mask.any():
                continue

            staves_xyxy = box_cxcywh_to_xyxy(
                pred_stave_boxes[:num_gt_staves][stave_mask])

            # Top of first stave == top of system
            top_loss = (staves_xyxy[0, 1] - sys_xyxy[1]).abs()

            # Bottom of last stave == bottom of system
            bottom_loss = (staves_xyxy[-1, 3] - sys_xyxy[3]).abs()

            # Left and right of all staves == left and right of system
            left_loss = (staves_xyxy[:, 0] - sys_xyxy[0]).abs().mean()
            right_loss = (staves_xyxy[:, 2] - sys_xyxy[2]).abs().mean()

            loss = loss + top_loss + bottom_loss + left_loss + right_loss

        return loss / num_gt_sys

    def forward(
        self,
        pred_sys_boxes: Tensor,
        pred_sys_logits: Tensor,
        pred_stave_boxes: Tensor,
        pred_stave_logits: Tensor,
        pred_assign: Tensor,
        gt_sys_boxes: list[Tensor],
        gt_stave_boxes: list[Tensor],
        gt_assign: list[Tensor],
    ) -> LossDict:
        B = pred_sys_boxes.shape[0]

        sys_box = torch.tensor(0.0, device=pred_sys_boxes.device)
        sys_giou = torch.tensor(0.0, device=pred_sys_boxes.device)
        sys_obj = torch.tensor(0.0, device=pred_sys_boxes.device)
        stave_box = torch.tensor(0.0, device=pred_sys_boxes.device)
        stave_giou = torch.tensor(0.0, device=pred_sys_boxes.device)
        stave_obj = torch.tensor(0.0, device=pred_sys_boxes.device)
        assign = torch.tensor(0.0, device=pred_sys_boxes.device)
        containment = torch.tensor(0.0, device=pred_sys_boxes.device)
        alignment = torch.tensor(0.0, device=pred_sys_boxes.device)

        for i in range(B):
            num_gt_staves = int((gt_assign[i] != -1).sum().item())
            num_gt_sys = int(gt_assign[i][gt_assign[i] != -1].max().item()) + 1

            b, g, o = self._box_loss(
                pred_sys_boxes[i], pred_sys_logits[i],
                gt_sys_boxes[i], num_gt_sys, self.config.num_system_queries,
            )
            sys_box = sys_box + b
            sys_giou = sys_giou + g
            sys_obj = sys_obj + o

            b, g, o = self._box_loss(
                pred_stave_boxes[i], pred_stave_logits[i],
                gt_stave_boxes[i], num_gt_staves, self.config.num_stave_queries,
            )
            stave_box = stave_box + b
            stave_giou = stave_giou + g
            stave_obj = stave_obj + o

            assign = assign + self._assignment_loss(
                pred_assign[i], gt_assign[i], num_gt_staves,
            )
            containment = containment + self._containment_loss(
                pred_sys_boxes[i], pred_stave_boxes[i],
                gt_assign[i], num_gt_staves, num_gt_sys,
            )
            alignment = alignment + self._alignment_loss(
                pred_sys_boxes[i], pred_stave_boxes[i],
                gt_assign[i], num_gt_staves, num_gt_sys,
            )

        return LossDict(
            sys_box=self.config.box_loss_multiplier * (sys_box / B),
            sys_giou=self.config.box_loss_multiplier * (sys_giou / B),
            sys_obj=sys_obj / B,
            stave_box=self.config.box_loss_multiplier * (stave_box / B),
            stave_giou=self.config.box_loss_multiplier * (stave_giou / B),
            stave_obj=stave_obj / B,
            assign=assign / B,
            containment=containment / B,
            alignment=alignment / B,
        )
