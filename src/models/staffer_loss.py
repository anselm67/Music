import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .staffer_model import Config


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
        pred_boxes: Tensor,      # (N, 4)
        pred_logits: Tensor,     # (N, 1)
        gt_boxes: Tensor,        # (N, 4) padded
        num_gt: int,
        num_queries: int,
    ) -> tuple[Tensor, Tensor, Tensor, list[int]]:
        """Returns box_loss, giou_loss, obj_loss, pred_idx (sorted by cy)."""
        # Aligns the predictions to the ground truth by sorting them by their y's
        pred_idx = pred_boxes[:, 1].argsort()[:num_gt].tolist()
        matched = pred_boxes[pred_idx]
        gt = gt_boxes[:num_gt]

        box_loss = F.l1_loss(matched, gt)
        giou_loss = (1 - generalized_iou(matched, gt)).mean()

        obj_target = torch.zeros(num_queries, device=pred_boxes.device)
        obj_target[pred_idx] = 1.0
        obj_loss = F.binary_cross_entropy_with_logits(
            pred_logits.squeeze(-1), obj_target)

        return box_loss, giou_loss, obj_loss, pred_idx

    def _containment_loss(
        self,
        pred_sys_boxes: Tensor,    # (N, 4) sorted
        pred_stave_boxes: Tensor,  # (M, 4) sorted
        gt_assign: Tensor,         # (M,) padded with -1
        num_gt_staves: int,
        num_gt_sys: int,
    ) -> Tensor:
        # pred_sys_boxes and pred_stave_boxes are top-y sorted
        loss = torch.tensor(0.0, device=pred_sys_boxes.device)

        for sys_idx in range(num_gt_sys):
            sys_box = pred_sys_boxes[sys_idx]
            stave_mask = gt_assign[:num_gt_staves] == sys_idx
            if not stave_mask.any():
                continue

            stave_boxes = pred_stave_boxes[:num_gt_staves][stave_mask]

            sys_xyxy = box_cxcywh_to_xyxy(sys_box.unsqueeze(0)).squeeze(0)
            staves_xyxy = box_cxcywh_to_xyxy(stave_boxes)

            # Area of each stave
            stave_areas = (
                (staves_xyxy[:, 2] - staves_xyxy[:, 0]) *
                (staves_xyxy[:, 3] - staves_xyxy[:, 1])
            ).clamp(min=1e-6)

            # Intersection of each stave with its system
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

            # Fraction of each stave outside its system
            outside_fraction = (stave_areas - inter_areas) / stave_areas
            loss = loss + outside_fraction.mean()

        return loss / num_gt_sys  # normalise by number of systems

    def _assignment_loss(
        self,
        pred_assign: Tensor,     # (M, N)
        stave_pred_idx: list[int],
        sys_pred_idx: list[int],
        gt_assign: Tensor,       # (M,) padded with -1
        num_gt_staves: int,
    ) -> Tensor:
        remapped = torch.tensor(
            [sys_pred_idx[int(gt_assign[j].item())]
             for j in range(num_gt_staves)],
            dtype=torch.long,
            device=pred_assign.device,
        )
        return F.cross_entropy(pred_assign[stave_pred_idx], remapped)

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
    ) -> Tensor:
        B = pred_sys_boxes.shape[0]
        total_loss = torch.tensor(0.0, device=pred_sys_boxes.device)

        for i in range(B):
            num_gt_staves = int((gt_assign[i] != -1).sum().item())
            num_gt_sys = int(gt_assign[i][gt_assign[i] != -1].max().item()) + 1

            sys_box_loss, sys_giou_loss, sys_obj_loss, sys_pred_idx = self._box_loss(
                pred_sys_boxes[i], pred_sys_logits[i],
                gt_sys_boxes[i], num_gt_sys, self.config.num_system_queries,
            )
            stave_box_loss, stave_giou_loss, stave_obj_loss, stave_pred_idx = self._box_loss(
                pred_stave_boxes[i], pred_stave_logits[i],
                gt_stave_boxes[i], num_gt_staves, self.config.num_stave_queries,
            )
            sorted_sys_boxes = pred_sys_boxes[i][sys_pred_idx]
            sorted_stave_boxes = pred_stave_boxes[i][stave_pred_idx]

            assign_loss = self._assignment_loss(
                pred_assign[i], stave_pred_idx, sys_pred_idx, gt_assign[i], num_gt_staves)

            containment_loss = self._containment_loss(
                sorted_sys_boxes, sorted_stave_boxes,
                gt_assign[i], num_gt_staves, num_gt_sys,
            )

            # Claude says: += on tensors that require grad can cause issues with autograd.
            total_loss = total_loss + sys_box_loss + sys_giou_loss + sys_obj_loss
            total_loss = total_loss + stave_box_loss + stave_giou_loss + stave_obj_loss
            total_loss = total_loss + assign_loss + containment_loss

        return total_loss / B
