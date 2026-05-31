"""Loss module for the Staffer model."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn

from .staffer_model import StafferConfig


def assign_staves(
    anchor_centers: Tensor,  # (M,) fixed per-slot vertical centers
    gt_stave_boxes: Tensor,  # (M, 4) ltrb padded
    num_gt: int,
) -> Tensor:
    """Route each GT stave to a query by nearest fixed anchor.

    Optimal 1-D bipartite matching on |anchor_center - gt_center|. Returns a
    (num_gt,) long tensor giving, for GT stave k (top-to-bottom), the query slot
    responsible for it. The mapping is free/non-contiguous: GT stave 4 may be
    served by query 6. In 1-D the optimal matching never crosses, so the returned
    indices are monotonic and stay consistent with the stave->system grouping.
    """
    device = anchor_centers.device
    if num_gt == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    gt_c = (gt_stave_boxes[:num_gt, 1] + gt_stave_boxes[:num_gt, 3]) / 2  # (G,)
    cost = (anchor_centers[:, None] - gt_c[None, :]).abs()  # (M, G)
    q_idx, g_idx = linear_sum_assignment(cost.detach().cpu().numpy())
    q_for_gt = q_idx[g_idx.argsort()]  # reorder to GT (top-to-bottom) order
    return torch.as_tensor(q_for_gt, dtype=torch.long, device=device)


@dataclass
class LossDict:
    stave_l1: Tensor
    stave_obj: Tensor
    boundary: Tensor
    sys_lr: Tensor
    sys_obj: Tensor
    sys_giou: Tensor

    def total(self) -> Tensor:
        return (
            self.stave_l1
            + self.stave_obj
            + self.boundary
            + self.sys_lr
            + self.sys_obj
            + self.sys_giou
        )


def generalized_iou(pred: Tensor, target: Tensor) -> Tensor:
    """Generalized Inter over Union distance between two sets of boxes.

    Args:
        pred (Tensor): Predicted boxes (P, 4) in (left, top, right, bottom) format
        target (Tensor): Ground truth (G, 4) in (left, top, right, bottom) format

    Returns:
        Tensor: Loss.
    """
    inter_x1 = torch.max(pred[:, 0], target[:, 0])
    inter_y1 = torch.max(pred[:, 1], target[:, 1])
    inter_x2 = torch.min(pred[:, 2], target[:, 2])
    inter_y2 = torch.min(pred[:, 3], target[:, 3])
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    area_pred = (pred[:, 2] - pred[:, 0]).clamp(min=0) * (
        pred[:, 3] - pred[:, 1]
    ).clamp(min=0)
    area_tgt = (target[:, 2] - target[:, 0]).clamp(min=0) * (
        target[:, 3] - target[:, 1]
    ).clamp(min=0)
    union = area_pred + area_tgt - inter
    iou = inter / union.clamp(min=1e-6)
    enclosing_x1 = torch.min(pred[:, 0], target[:, 0])
    enclosing_y1 = torch.min(pred[:, 1], target[:, 1])
    enclosing_x2 = torch.max(pred[:, 2], target[:, 2])
    enclosing_y2 = torch.max(pred[:, 3], target[:, 3])
    enclosing = (enclosing_x2 - enclosing_x1).clamp(min=0) * (
        enclosing_y2 - enclosing_y1
    ).clamp(min=0)
    return iou - (enclosing - union) / enclosing.clamp(min=1e-6)


class StafferLoss(nn.Module):
    def __init__(self, config: StafferConfig):
        super().__init__()
        self.config = config

    def _obj_loss(
        self,
        pred_logits: Tensor,  # (Q, 1)
        matched_idx: Tensor,  # (G,) indices of the queries that own a GT object
        num_queries: int,
    ) -> Tensor:
        obj_target = torch.zeros(num_queries, device=pred_logits.device)
        obj_target[matched_idx] = 1.0
        return F.binary_cross_entropy_with_logits(pred_logits.squeeze(-1), obj_target)

    def _tb_l1(
        self,
        pred_tb: Tensor,  # (M, 2) — top, bottom predictions
        gt_boxes: Tensor,  # (M, 4) ltrb padded
        q: Tensor,  # (G,) query slot owning GT stave k
    ) -> Tensor:
        return F.l1_loss(pred_tb[q], gt_boxes[: q.shape[0], [1, 3]])

    def _boundary_loss(
        self,
        pred_boundary: Tensor,  # (M, 1)
        gt_assign: Tensor,  # (M,) padded with -1
        q: Tensor,  # (G,) query slot owning GT stave k
    ) -> Tensor:
        # Target: 1 where a stave starts a new system (the first stave always
        # does); cumsum of this flag recovers the stave->system grouping. The
        # boundary prediction is read from each GT stave's owning query.
        num_gt_staves = q.shape[0]
        target = torch.zeros(num_gt_staves, device=pred_boundary.device)
        target[0] = 1.0
        if num_gt_staves > 1:
            a = gt_assign[:num_gt_staves]
            target[1:] = (a[1:] != a[:-1]).float()
        return F.binary_cross_entropy_with_logits(pred_boundary[q].squeeze(-1), target)

    def _sys_lr_l1(
        self,
        pred_sys_lr: Tensor,  # (N, 2) — left, right
        gt_sys_boxes: Tensor,  # (N, 4) ltrb padded
        num_gt_sys: int,
    ) -> Tensor:
        return F.l1_loss(pred_sys_lr[:num_gt_sys], gt_sys_boxes[:num_gt_sys][:, [0, 2]])

    def _derived_sys_giou(
        self,
        pred_stave_tb: Tensor,  # (M, 2) — top, bottom
        pred_sys_lr: Tensor,  # (N, 2) — left, right
        gt_assign: Tensor,  # (M,) padded with -1
        gt_sys_boxes: Tensor,  # (N, 4) ltrb padded
        num_gt_sys: int,
        q: Tensor,  # (G,) query slot owning GT stave k
    ) -> Tensor:
        # Derive each system box as the hull of its assigned staves:
        # (sys.left, min stave tops, sys.right, max stave bottoms), then GIoU vs GT.
        # min/max route gradient into the extreme (first/last) staves of a system.
        assign = gt_assign[: q.shape[0]]
        tb = pred_stave_tb[q]  # (S, 2) — predicted edges of each GT stave's query
        losses = []
        for j in range(num_gt_sys):
            mask = assign == j
            if not mask.any():
                continue
            top = tb[mask][:, 0].min()
            bot = tb[mask][:, 1].max()
            derived = torch.stack([pred_sys_lr[j, 0], top, pred_sys_lr[j, 1], bot])
            giou = generalized_iou(derived.unsqueeze(0), gt_sys_boxes[j].unsqueeze(0))
            losses.append(1.0 - giou.squeeze(0))
        if not losses:
            return torch.tensor(0.0, device=pred_stave_tb.device)
        return torch.stack(losses).mean()

    def forward(
        self,
        pred_stave_tb: Tensor,  # (B, M, 2) — top, bottom
        pred_stave_logits: Tensor,  # (B, M, 1)
        pred_boundary_logits: Tensor,  # (B, M, 1)
        pred_sys_lr: Tensor,  # (B, N, 2) — left, right
        pred_sys_logits: Tensor,  # (B, N, 1)
        gt_sys_boxes: Tensor,  # list of (N, 4) ltrb padded
        gt_stave_boxes: Tensor,  # list of (M, 4) ltrb padded
        gt_assign: Tensor,  # list of (M,) padded with -1
        assign_q: list[Tensor],  # list of (G,) query slot owning each GT stave
    ) -> LossDict:
        B = pred_stave_tb.shape[0]

        device = pred_stave_tb.device
        stave_l1 = torch.tensor(0.0, device=device)
        stave_obj = torch.tensor(0.0, device=device)
        boundary = torch.tensor(0.0, device=device)
        sys_lr = torch.tensor(0.0, device=device)
        sys_obj = torch.tensor(0.0, device=device)
        sys_giou = torch.tensor(0.0, device=device)

        for i in range(B):
            q = assign_q[i]
            num_gt_sys = int(gt_assign[i][gt_assign[i] != -1].max().item()) + 1
            sys_idx = torch.arange(num_gt_sys, device=device)

            stave_l1 = stave_l1 + self._tb_l1(pred_stave_tb[i], gt_stave_boxes[i], q)

            stave_obj = stave_obj + self._obj_loss(
                pred_stave_logits[i], q, self.config.num_stave_queries
            )
            boundary = boundary + self._boundary_loss(
                pred_boundary_logits[i], gt_assign[i], q
            )
            sys_lr = sys_lr + self._sys_lr_l1(
                pred_sys_lr[i], gt_sys_boxes[i], num_gt_sys
            )
            sys_obj = sys_obj + self._obj_loss(
                pred_sys_logits[i], sys_idx, self.config.num_system_queries
            )
            sys_giou = sys_giou + self._derived_sys_giou(
                pred_stave_tb[i],
                pred_sys_lr[i],
                gt_assign[i],
                gt_sys_boxes[i],
                num_gt_sys,
                q,
            )

        mult = self.config.box_loss_multiplier
        return LossDict(
            stave_l1=mult * (stave_l1 / B),
            stave_obj=stave_obj / B,
            boundary=boundary / B,
            sys_lr=mult * (sys_lr / B),
            sys_obj=sys_obj / B,
            sys_giou=mult * (sys_giou / B),
        )
