import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn

from .staffer_model import Config


def box_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def generalized_iou(pred: Tensor, target: Tensor) -> Tensor:
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


def hungarian_match(pred_boxes: Tensor, gt_boxes: Tensor) -> tuple[list[int], list[int]]:
    """Match predicted boxes to GT boxes for a single image."""
    N = pred_boxes.shape[0]
    M = gt_boxes.shape[0]
    # Cost matrix: L1 + GIoU
    cost_l1 = torch.cdist(pred_boxes, gt_boxes, p=1)  # (N, M)
    giou = generalized_iou(
        pred_boxes.unsqueeze(1).expand(-1, M, -1).reshape(-1, 4),
        gt_boxes.unsqueeze(0).expand(N, -1, -1).reshape(-1, 4),
    ).reshape(N, M)
    cost = cost_l1 - giou  # lower is better
    pred_idx, gt_idx = linear_sum_assignment(cost.detach().cpu().numpy())
    return pred_idx.tolist(), gt_idx.tolist()


class HierarchicalLoss(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

    def forward(
        self,
        pred_sys_boxes: Tensor,    # (B, N, 4)
        pred_sys_logits: Tensor,   # (B, N, 1)
        pred_stave_boxes: Tensor,  # (B, M, 4)
        pred_stave_logits: Tensor,  # (B, M, 1)
        pred_assign: Tensor,       # (B, M, N)
        gt_sys_boxes: list[Tensor],    # list of (num_sys, 4) per image
        gt_stave_boxes: list[Tensor],  # list of (num_staves, 4) per image
        # list of (num_staves,) stave→system index per image
        gt_assign: list[Tensor],
    ) -> Tensor:
        B = pred_sys_boxes.shape[0]
        total_loss = torch.tensor(0.0, device=pred_sys_boxes.device)

        for i in range(B):
            # TODO Check this out, added to original code.
            num_gt_staves = (gt_assign[i] != -1).sum().item()
            num_gt_sys = gt_assign[i][gt_assign[i] != -1].max().item() + 1

            sys_pred_idx, sys_gt_idx = hungarian_match(
                pred_sys_boxes[i], gt_sys_boxes[i][:num_gt_sys])

            matched_sys_boxes = pred_sys_boxes[i][sys_pred_idx]
            sys_box_loss = F.l1_loss(
                matched_sys_boxes, gt_sys_boxes[i][sys_gt_idx])
            sys_giou_loss = (
                1 - generalized_iou(matched_sys_boxes, gt_sys_boxes[i][sys_gt_idx])).mean()

            sys_obj_target = torch.zeros(
                self.config.num_system_queries, device=pred_sys_boxes.device)
            sys_obj_target[sys_pred_idx] = 1.0
            sys_obj_loss = F.binary_cross_entropy_with_logits(
                pred_sys_logits[i].squeeze(-1), sys_obj_target)

            # --- Stave loss ---

            stave_pred_idx, stave_gt_idx = hungarian_match(
                pred_stave_boxes[i], gt_stave_boxes[i][:num_gt_staves])

            matched_stave_boxes = pred_stave_boxes[i][stave_pred_idx]
            stave_box_loss = F.l1_loss(
                matched_stave_boxes, gt_stave_boxes[i][stave_gt_idx])
            stave_giou_loss = (
                1 - generalized_iou(matched_stave_boxes, gt_stave_boxes[i][stave_gt_idx])).mean()

            stave_obj_target = torch.zeros(
                self.config.num_stave_queries, device=pred_stave_boxes.device)
            stave_obj_target[stave_pred_idx] = 1.0
            stave_obj_loss = F.binary_cross_entropy_with_logits(
                pred_stave_logits[i].squeeze(-1), stave_obj_target)

            # --- Assignment loss ---
            # remap gt assign indices through the system matching
            sys_gt_to_pred = {gt: pred for pred,
                              gt in zip(sys_pred_idx, sys_gt_idx)}
            remapped_assign = torch.tensor(
                [sys_gt_to_pred[int(gt_assign[i][j].item())]
                 for j in stave_gt_idx],
                dtype=torch.long,
                device=pred_assign.device,
            )
            assign_loss = F.cross_entropy(
                pred_assign[i][stave_pred_idx], remapped_assign)

            total_loss += sys_box_loss + sys_giou_loss + sys_obj_loss
            total_loss += stave_box_loss + stave_giou_loss + stave_obj_loss
            total_loss += assign_loss

        return total_loss / B
