import pytest
import torch

from models import Config, HierarchicalLoss


def make_boxes(n: int, padded: int) -> torch.Tensor:
    """Make n random normalised boxes, padded to padded size."""
    boxes = torch.zeros(padded, 4)
    boxes[:n] = torch.rand(n, 4).clamp(0.01, 0.99)
    return boxes


def make_assign(num_staves: int, num_sys: int, padded: int) -> torch.Tensor:
    """Make stave->system assignments, padded with -1."""
    assigns = torch.full((padded,), -1, dtype=torch.long)
    assigns[:num_staves] = torch.randint(0, num_sys, (num_staves,))
    return assigns


class TestHierarchicalLoss:
    @pytest.fixture
    def config(self) -> Config:
        return Config()

    @pytest.fixture
    def loss(self, config: Config) -> HierarchicalLoss:
        return HierarchicalLoss(config)

    def test_loss_is_scalar(self, loss: HierarchicalLoss, config: Config):
        """Loss should return a scalar tensor."""
        B = 2
        N, M = config.num_system_queries, config.num_stave_queries
        D = config.embed_dim

        pred_sys_boxes = torch.rand(B, N, 4)
        pred_sys_logits = torch.randn(B, N, 1)
        pred_stave_boxes = torch.rand(B, M, 4)
        pred_stave_logits = torch.randn(B, M, 1)
        pred_assign = torch.randn(B, M, N)

        gt_sys_boxes = [make_boxes(3, N) for _ in range(B)]
        gt_stave_boxes = [make_boxes(5, M) for _ in range(B)]
        gt_assign = [make_assign(5, 3, M) for _ in range(B)]

        l = loss(pred_sys_boxes, pred_sys_logits, pred_stave_boxes,
                 pred_stave_logits, pred_assign, gt_sys_boxes, gt_stave_boxes, gt_assign)

        assert l.shape == torch.Size([])
        assert l.item() > 0

    def test_loss_decreases_with_perfect_boxes(self, loss: HierarchicalLoss, config: Config):
        """Loss should be lower when predicted boxes match GT exactly."""
        B = 1
        N, M = config.num_system_queries, config.num_stave_queries

        gt_sys = make_boxes(3, N)
        gt_stave = make_boxes(5, M)
        gt_assign = make_assign(5, 3, M)

        # Perfect prediction — boxes match GT exactly
        pred_sys_boxes = gt_sys.unsqueeze(0)
        pred_stave_boxes = gt_stave.unsqueeze(0)
        pred_sys_logits = torch.zeros(B, N, 1)
        pred_stave_logits = torch.zeros(B, M, 1)
        pred_assign = torch.randn(B, M, N)

        perfect_loss = loss(pred_sys_boxes, pred_sys_logits, pred_stave_boxes,
                            pred_stave_logits, pred_assign, [gt_sys], [gt_stave], [gt_assign])

        # Random prediction
        rand_sys_boxes = torch.rand(B, N, 4)
        rand_stave_boxes = torch.rand(B, M, 4)

        random_loss = loss(rand_sys_boxes, pred_sys_logits, rand_stave_boxes,
                           pred_stave_logits, pred_assign, [gt_sys], [gt_stave], [gt_assign])

        assert perfect_loss.item() < random_loss.item()

    def test_loss_single_system_single_stave(self, loss: HierarchicalLoss, config: Config):
        """Minimal case — one system, one stave."""
        B = 1
        N, M = config.num_system_queries, config.num_stave_queries

        gt_sys = make_boxes(1, N)
        gt_stave = make_boxes(1, M)
        gt_assign = make_assign(1, 1, M)

        pred_sys_boxes = torch.rand(B, N, 4)
        pred_sys_logits = torch.randn(B, N, 1)
        pred_stave_boxes = torch.rand(B, M, 4)
        pred_stave_logits = torch.randn(B, M, 1)
        pred_assign = torch.randn(B, M, N)

        l = loss(pred_sys_boxes, pred_sys_logits, pred_stave_boxes,
                 pred_stave_logits, pred_assign, [gt_sys], [gt_stave], [gt_assign])

        assert l.item() > 0

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

        l = loss(pred_sys_boxes, pred_sys_logits, pred_stave_boxes,
                 pred_stave_logits, pred_assign, gt_sys_boxes, gt_stave_boxes, gt_assign)
        l.backward()

        assert pred_sys_boxes.grad is not None
        assert pred_stave_boxes.grad is not None
        assert pred_sys_boxes.grad is not None
        assert pred_stave_boxes.grad is not None
