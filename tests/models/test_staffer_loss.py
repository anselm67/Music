import pytest
import torch
from torch import Tensor

from models import Config, HierarchicalLoss


def make_boxes(n: int, padded: int) -> Tensor:
    """Make n random normalised boxes sorted by cy, padded to padded size."""
    boxes = torch.zeros(padded, 4)
    if n > 0:
        raw = torch.rand(n, 4).clamp(0.01, 0.99)
        raw = raw[raw[:, 1].argsort()]  # sort by cy
        boxes[:n] = raw
    return boxes


def make_assign(num_staves: int, num_sys: int, padded: int) -> Tensor:
    """Make stave->system assignments sorted consistently, padded with -1."""
    assigns = torch.full((padded,), -1, dtype=torch.long)
    if num_staves > 0 and num_sys > 0:
        # distribute staves evenly across systems, in order
        assigns[:num_staves] = torch.arange(num_staves) % num_sys
        assigns[:num_staves] = assigns[:num_staves].sort().values
    return assigns


class TestHierarchicalLoss:

    @pytest.fixture
    def config(self) -> Config:
        return Config()

    @pytest.fixture
    def loss(self, config: Config) -> HierarchicalLoss:
        return HierarchicalLoss(config)

    def _make_inputs(self, config: Config, num_sys: int, num_staves: int, B: int = 2):
        N, M = config.num_system_queries, config.num_stave_queries
        pred_sys_boxes = torch.rand(B, N, 4)
        pred_sys_logits = torch.randn(B, N, 1)
        pred_stave_boxes = torch.rand(B, M, 4)
        pred_stave_logits = torch.randn(B, M, 1)
        pred_assign = torch.randn(B, M, N)
        gt_sys_boxes = [make_boxes(num_sys, N) for _ in range(B)]
        gt_stave_boxes = [make_boxes(num_staves, M) for _ in range(B)]
        gt_assign = [make_assign(num_staves, num_sys, M) for _ in range(B)]
        return (pred_sys_boxes, pred_sys_logits, pred_stave_boxes,
                pred_stave_logits, pred_assign, gt_sys_boxes, gt_stave_boxes, gt_assign)

    def test_loss_is_scalar(self, loss: HierarchicalLoss, config: Config):
        """Loss should return a scalar tensor."""
        inputs = self._make_inputs(config, num_sys=3, num_staves=5)
        result = loss(*inputs)
        assert result.shape == torch.Size([])
        assert result.item() > 0

    def test_loss_single_system_single_stave(self, loss: HierarchicalLoss, config: Config):
        """Minimal case — one system, one stave."""
        inputs = self._make_inputs(config, num_sys=1, num_staves=1)
        result = loss(*inputs)
        assert result.item() > 0

    def test_loss_max_queries(self, loss: HierarchicalLoss, config: Config):
        """Loss should handle num_gt == num_queries without crashing."""
        N, M = config.num_system_queries, config.num_stave_queries
        inputs = self._make_inputs(config, num_sys=N, num_staves=M)
        result = loss(*inputs)
        assert result.item() > 0

    def test_loss_decreases_with_better_boxes(self, loss: HierarchicalLoss, config: Config):
        """Loss should be lower when predicted boxes are sorted to match GT."""
        B = 1
        N, M = config.num_system_queries, config.num_stave_queries

        gt_sys = make_boxes(3, N)
        gt_stave = make_boxes(5, M)
        gt_assign = make_assign(5, 3, M)

        # Good prediction: top N predictions already sorted by cy matching GT
        good_sys_boxes = torch.rand(B, N, 4)
        good_sys_boxes[0, :3] = gt_sys[:3]  # first 3 match GT exactly
        good_sys_boxes[0] = good_sys_boxes[0,
                                           good_sys_boxes[0, :, 1].argsort()]

        good_stave_boxes = torch.rand(B, M, 4)
        good_stave_boxes[0, :5] = gt_stave[:5]
        good_stave_boxes[0] = good_stave_boxes[0,
                                               good_stave_boxes[0, :, 1].argsort()]

        # Random prediction
        rand_sys_boxes = torch.rand(B, N, 4)
        rand_stave_boxes = torch.rand(B, M, 4)

        logits = torch.zeros(B, N, 1)
        stave_logits = torch.zeros(B, M, 1)
        pred_assign = torch.randn(B, M, N)

        good_loss = loss(good_sys_boxes, logits, good_stave_boxes, stave_logits,
                         pred_assign, [gt_sys], [gt_stave], [gt_assign])
        rand_loss = loss(rand_sys_boxes, logits, rand_stave_boxes, stave_logits,
                         pred_assign, [gt_sys], [gt_stave], [gt_assign])

        assert good_loss.item() < rand_loss.item()

    def test_loss_backward(self, loss: HierarchicalLoss, config: Config):
        """Loss should be differentiable."""
        B = 2
        N, M = config.num_system_queries, config.num_stave_queries

        pred_sys_boxes = torch.rand(B, N, 4, requires_grad=True)
        pred_sys_logits = torch.randn(B, N, 1, requires_grad=True)
        pred_stave_boxes = torch.rand(B, M, 4, requires_grad=True)
        pred_stave_logits = torch.randn(B, M, 1, requires_grad=True)
        pred_assign = torch.randn(B, M, N, requires_grad=True)

        gt_sys_boxes = [make_boxes(3, N) for _ in range(B)]
        gt_stave_boxes = [make_boxes(5, M) for _ in range(B)]
        gt_assign = [make_assign(5, 3, M) for _ in range(B)]

        result = loss(pred_sys_boxes, pred_sys_logits, pred_stave_boxes,
                      pred_stave_logits, pred_assign, gt_sys_boxes, gt_stave_boxes, gt_assign)
        result.backward()

        assert pred_sys_boxes.grad is not None
        assert pred_stave_boxes.grad is not None
        assert pred_assign.grad is not None

    def test_containment_loss_isolated(self, loss: HierarchicalLoss, config: Config):
        """Directly test _containment_loss — stave outside system yields higher loss."""
        N, M = config.num_system_queries, config.num_stave_queries

        gt_assign = make_assign(2, 1, M)  # 2 staves, 1 system

        # Good: sorted top-y stave boxes inside system box
        good_sys = torch.zeros(N, 4)
        good_sys[0] = torch.tensor([0.5, 0.5, 0.8, 0.6])   # large system
        good_staves = torch.zeros(M, 4)
        good_staves[0] = torch.tensor([0.5, 0.4, 0.4, 0.1])  # stave inside
        good_staves[1] = torch.tensor([0.5, 0.6, 0.4, 0.1])  # stave inside

        # Bad: top-y stave boxes outside system box
        bad_sys = torch.zeros(N, 4)
        bad_sys[0] = torch.tensor([0.5, 0.5, 0.1, 0.05])   # tiny system
        bad_staves = torch.zeros(M, 4)
        bad_staves[0] = torch.tensor([0.5, 0.4, 0.8, 0.3])  # stave outside
        bad_staves[1] = torch.tensor([0.5, 0.6, 0.8, 0.3])  # stave outside

        good = loss._containment_loss(
            good_sys, good_staves, gt_assign, 2, 1)
        bad = loss._containment_loss(
            bad_sys, bad_staves, gt_assign, 2, 1)

        assert good.item() == pytest.approx(0.0, abs=1e-5)
        assert bad.item() > 0.0

    def test_containment_loss_isolated2(self, loss: HierarchicalLoss, config: Config):
        """Directly test _containment_loss — stave outside system yields higher loss."""
        N, M = config.num_system_queries, config.num_stave_queries

        gt_assign = make_assign(1, 1, M)  # 1 stave, 1 system

        # Good: stave is inside system
        good_sys = torch.zeros(N, 4)
        good_sys[0] = torch.tensor([0.5, 0.3, 0.8, 0.2])
        good_stave = torch.zeros(M, 4)
        good_stave[0] = torch.tensor([0.5, 0.3, 0.4, 0.05])

        good = loss._containment_loss(
            good_sys, good_stave, gt_assign, 1, 1)

        assert good.item() == pytest.approx(0.0, abs=1e-5)

    def test_containment_loss_activates(self, loss: HierarchicalLoss, config: Config):
        """Containment loss should be higher when staves stick out of their system."""
        B = 1
        N, M = config.num_system_queries, config.num_stave_queries

        sys_logits = torch.zeros(B, N, 1)
        stave_logits = torch.zeros(B, M, 1)
        pred_assign = torch.randn(B, M, N)

        # One system containing one stave — GT matches predictions exactly
        # so box/giou/obj losses are identical between good and bad cases
        gt_assign = make_assign(1, 1, M)

        # Good: stave is inside system
        good_sys = torch.ones(B, N, 4)
        good_sys[0, 0] = torch.tensor([0.5, 0.3, 0.8, 0.2])
        good_stave = torch.ones(B, M, 4)
        good_stave[0, 0] = torch.tensor([0.5, 0.3, 0.4, 0.05])
        gt_sys_good = good_sys[0].clone()
        gt_stave_good = good_stave[0].clone()

        # Bad: stave is outside system (same cy so sorting is stable)
        bad_sys = torch.ones(B, N, 4)
        bad_sys[0, 0] = torch.tensor([0.5, 0.3, 0.4, 0.05])
        bad_stave = torch.ones(B, M, 4)
        bad_stave[0, 0] = torch.tensor([0.5, 0.3, 0.8, 0.2])
        gt_sys_bad = bad_sys[0].clone()
        gt_stave_bad = bad_stave[0].clone()

        good_loss = loss(good_sys, sys_logits, good_stave, stave_logits,
                         pred_assign, [gt_sys_good], [gt_stave_good], [gt_assign])
        bad_loss = loss(bad_sys, sys_logits, bad_stave, stave_logits,
                        pred_assign, [gt_sys_bad], [gt_stave_bad], [gt_assign])

        assert bad_loss.item() > good_loss.item()
