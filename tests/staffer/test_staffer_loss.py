import pytest
import torch
from torch import Tensor

from staffer import StafferConfig, StafferLoss, LossDict


def make_boxes(n: int, padded: int) -> Tensor:
    """Make n random valid normalised ltrb boxes sorted by top, padded."""
    boxes = torch.zeros(padded, 4)
    if n > 0:
        coords = torch.rand(n, 4).clamp(0.01, 0.99)
        x = coords[:, [0, 2]].sort(dim=1).values
        y = coords[:, [1, 3]].sort(dim=1).values
        raw = torch.stack([x[:, 0], y[:, 0], x[:, 1], y[:, 1]], dim=1)
        raw = raw[raw[:, 1].argsort()]  # sort by top
        boxes[:n] = raw
    return boxes


def make_assign(num_staves: int, num_sys: int, padded: int) -> Tensor:
    """Make stave->system assignments sorted consistently, padded with -1."""
    assigns = torch.full((padded,), -1, dtype=torch.long)
    if num_staves > 0 and num_sys > 0:
        assigns[:num_staves] = torch.arange(num_staves) % num_sys
        assigns[:num_staves] = assigns[:num_staves].sort().values
    return assigns


class TestStafferLoss:
    @pytest.fixture
    def config(self) -> StafferConfig:
        return StafferConfig()

    @pytest.fixture
    def loss(self, config: StafferConfig) -> StafferLoss:
        return StafferLoss(config)

    def _make_inputs(
        self, config: StafferConfig, num_sys: int, num_staves: int, B: int = 2
    ) -> tuple[
        Tensor, Tensor, Tensor, Tensor, Tensor, list[Tensor], list[Tensor], list[Tensor]
    ]:
        N, M = config.num_system_queries, config.num_stave_queries
        pred_stave_tb = torch.rand(B, M, 2)
        pred_stave_logits = torch.randn(B, M, 1)
        pred_boundary_logits = torch.randn(B, M, 1)
        pred_sys_lr = torch.rand(B, N, 2)
        pred_sys_logits = torch.randn(B, N, 1)
        gt_sys_boxes = [make_boxes(num_sys, N) for _ in range(B)]
        gt_stave_boxes = [make_boxes(num_staves, M) for _ in range(B)]
        gt_assign = [make_assign(num_staves, num_sys, M) for _ in range(B)]
        return (
            pred_stave_tb,
            pred_stave_logits,
            pred_boundary_logits,
            pred_sys_lr,
            pred_sys_logits,
            gt_sys_boxes,
            gt_stave_boxes,
            gt_assign,
        )

    def test_loss_is_scalar(self, loss: StafferLoss, config: StafferConfig) -> None:
        inputs = self._make_inputs(config, num_sys=3, num_staves=5)
        result = loss(*inputs)
        assert isinstance(result, LossDict)
        assert result.total() > 0

    def test_loss_single_system_single_stave(
        self, loss: StafferLoss, config: StafferConfig
    ) -> None:
        inputs = self._make_inputs(config, num_sys=1, num_staves=1)
        result = loss(*inputs)
        assert result.total() > 0

    def test_loss_max_queries(self, loss: StafferLoss, config: StafferConfig) -> None:
        N, M = config.num_system_queries, config.num_stave_queries
        inputs = self._make_inputs(config, num_sys=N, num_staves=M)
        result = loss(*inputs)
        assert result.total() > 0

    def test_loss_decreases_with_better_predictions(
        self, loss: StafferLoss, config: StafferConfig
    ) -> None:
        """Loss should be lower when predictions match GT than when far away."""
        B = 1
        N, M = config.num_system_queries, config.num_stave_queries

        gt_sys = make_boxes(3, N)
        gt_stave = make_boxes(5, M)
        gt_assign = make_assign(5, 3, M)

        stave_logits = torch.zeros(B, M, 1)
        boundary_logits = torch.zeros(B, M, 1)
        sys_logits = torch.zeros(B, N, 1)

        good_stave_tb = torch.ones(B, M, 2)
        good_stave_tb[0, :5] = gt_stave[:5, [1, 3]]
        good_sys_lr = torch.ones(B, N, 2)
        good_sys_lr[0, :3] = gt_sys[:3][:, [0, 2]]

        bad_stave_tb = torch.ones(B, M, 2) * 0.9
        bad_sys_lr = torch.ones(B, N, 2) * 0.9

        good_loss = loss(
            good_stave_tb,
            stave_logits,
            boundary_logits,
            good_sys_lr,
            sys_logits,
            [gt_sys],
            [gt_stave],
            [gt_assign],
        )
        bad_loss = loss(
            bad_stave_tb,
            stave_logits,
            boundary_logits,
            bad_sys_lr,
            sys_logits,
            [gt_sys],
            [gt_stave],
            [gt_assign],
        )

        assert good_loss.total() < bad_loss.total()

    def test_loss_backward(self, loss: StafferLoss, config: StafferConfig) -> None:
        """Loss should be differentiable into stave, boundary and system heads."""
        B = 2
        N, M = config.num_system_queries, config.num_stave_queries

        pred_stave_tb = torch.rand(B, M, 2, requires_grad=True)
        pred_stave_logits = torch.randn(B, M, 1, requires_grad=True)
        pred_boundary_logits = torch.randn(B, M, 1, requires_grad=True)
        pred_sys_lr = torch.rand(B, N, 2, requires_grad=True)
        pred_sys_logits = torch.randn(B, N, 1, requires_grad=True)

        gt_sys_boxes = [make_boxes(3, N) for _ in range(B)]
        gt_stave_boxes = [make_boxes(5, M) for _ in range(B)]
        gt_assign = [make_assign(5, 3, M) for _ in range(B)]

        result = loss(
            pred_stave_tb,
            pred_stave_logits,
            pred_boundary_logits,
            pred_sys_lr,
            pred_sys_logits,
            gt_sys_boxes,
            gt_stave_boxes,
            gt_assign,
        )
        result.total().backward()

        assert pred_stave_tb.grad is not None
        assert pred_boundary_logits.grad is not None
        assert pred_sys_lr.grad is not None
        assert pred_sys_logits.grad is not None

    def test_boundary_cumsum_recovers_assign(self) -> None:
        """cumsum of the boundary flag (minus 1) reproduces the GT grouping."""
        assign = torch.tensor([0, 0, 1, 1, 1, 2])
        num_gt = assign.numel()
        boundary = torch.zeros(num_gt, dtype=torch.long)
        boundary[0] = 1
        boundary[1:] = (assign[1:] != assign[:-1]).long()
        assert torch.equal(boundary, torch.tensor([1, 0, 1, 0, 0, 1]))
        assert torch.equal(boundary.cumsum(0) - 1, assign)

    def test_derived_sys_giou_zero_when_consistent(
        self, loss: StafferLoss, config: StafferConfig
    ) -> None:
        """When staves + left/right match GT, the derived system box == GT box."""
        N, M = config.num_system_queries, config.num_stave_queries
        # 3 staves in one system, shared left/right; system box is their hull.
        staves = torch.tensor(
            [
                [0.1, 0.10, 0.9, 0.20],
                [0.1, 0.25, 0.9, 0.35],
                [0.1, 0.40, 0.9, 0.50],
            ]
        )
        sys_box = torch.tensor([0.1, 0.10, 0.9, 0.50])  # hull of the staves
        gt_stave = torch.zeros(M, 4)
        gt_stave[:3] = staves
        gt_sys = torch.zeros(N, 4)
        gt_sys[0] = sys_box
        gt_assign = torch.full((M,), -1, dtype=torch.long)
        gt_assign[:3] = 0

        pred_stave_tb = torch.zeros(1, M, 2)
        pred_stave_tb[0, :3] = staves[:, [1, 3]]
        pred_sys_lr = torch.zeros(1, N, 2)
        pred_sys_lr[0, 0] = torch.tensor([0.1, 0.9])

        result = loss(
            pred_stave_tb,
            torch.zeros(1, M, 1),
            torch.zeros(1, M, 1),
            pred_sys_lr,
            torch.zeros(1, N, 1),
            [gt_sys],
            [gt_stave],
            [gt_assign],
        )
        assert result.sys_giou.abs() < 1e-4
