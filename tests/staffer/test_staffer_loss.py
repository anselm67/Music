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
        pred_sys_boxes = torch.stack([make_boxes(N, N) for _ in range(B)])
        pred_sys_logits = torch.randn(B, N, 1)
        pred_stave_tb = torch.rand(B, M, 2)
        pred_stave_logits = torch.randn(B, M, 1)
        pred_assign = torch.randn(B, M, N)
        gt_sys_boxes = [make_boxes(num_sys, N) for _ in range(B)]
        gt_stave_boxes = [make_boxes(num_staves, M) for _ in range(B)]
        gt_assign = [make_assign(num_staves, num_sys, M) for _ in range(B)]
        return (
            pred_sys_boxes,
            pred_sys_logits,
            pred_stave_tb,
            pred_stave_logits,
            pred_assign,
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

        logits = torch.zeros(B, N, 1)
        stave_logits = torch.zeros(B, M, 1)
        pred_assign = torch.zeros(B, M, N)

        good_sys = torch.ones(B, N, 4)
        good_sys[0, :3] = gt_sys[:3]
        good_stave_tb = torch.ones(B, M, 2)
        good_stave_tb[0, :5] = gt_stave[:5, [1, 3]]

        bad_sys = torch.ones(B, N, 4) * 0.9
        bad_stave_tb = torch.ones(B, M, 2) * 0.9

        good_loss = loss(
            good_sys,
            logits,
            good_stave_tb,
            stave_logits,
            pred_assign,
            [gt_sys],
            [gt_stave],
            [gt_assign],
        )
        bad_loss = loss(
            bad_sys,
            logits,
            bad_stave_tb,
            stave_logits,
            pred_assign,
            [gt_sys],
            [gt_stave],
            [gt_assign],
        )

        assert good_loss.total() < bad_loss.total()

    def test_loss_backward(self, loss: StafferLoss, config: StafferConfig) -> None:
        """Loss should be differentiable."""
        B = 2
        N, M = config.num_system_queries, config.num_stave_queries

        pred_sys_boxes = torch.stack(
            [make_boxes(3, N) for _ in range(B)]
        ).requires_grad_(True)
        pred_sys_logits = torch.randn(B, N, 1, requires_grad=True)
        pred_stave_tb = torch.rand(B, M, 2, requires_grad=True)
        pred_stave_logits = torch.randn(B, M, 1, requires_grad=True)
        pred_assign = torch.randn(B, M, N, requires_grad=True)

        gt_sys_boxes = [make_boxes(3, N) for _ in range(B)]
        gt_stave_boxes = [make_boxes(5, M) for _ in range(B)]
        gt_assign = [make_assign(5, 3, M) for _ in range(B)]

        result = loss(
            pred_sys_boxes,
            pred_sys_logits,
            pred_stave_tb,
            pred_stave_logits,
            pred_assign,
            gt_sys_boxes,
            gt_stave_boxes,
            gt_assign,
        )
        result.total().backward()

        assert pred_sys_boxes.grad is not None
        assert pred_stave_tb.grad is not None
        assert pred_assign.grad is not None
