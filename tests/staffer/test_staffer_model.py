import torch

from staffer import StafferConfig, StafferModel


def test_forward_shapes_and_backward() -> None:
    # Small image so the ViT backbone is cheap on CPU.
    config = StafferConfig(max_height=128, max_width=128)
    model = StafferModel(config)
    N, M = config.num_system_queries, config.num_stave_queries
    B = 2
    H, W = config.image_shape

    x = torch.randn(B, config.in_channels, H, W)
    stave_tb, stave_logits, boundary_logits, sys_lr, sys_logits = model(x)

    assert stave_tb.shape == (B, M, 2)
    assert stave_logits.shape == (B, M, 1)
    assert boundary_logits.shape == (B, M, 1)
    assert sys_lr.shape == (B, N, 2)
    assert sys_logits.shape == (B, N, 1)

    # Boxes are sigmoid-bounded to [0, 1].
    assert stave_tb.min() >= 0.0 and stave_tb.max() <= 1.0
    assert sys_lr.min() >= 0.0 and sys_lr.max() <= 1.0

    stave_tb.sum().backward()


def test_stave_anchors_have_positive_initial_height() -> None:
    # Bottom anchor must sit below the top anchor at init for every slot, so the
    # derived system hull is non-degenerate before any training.
    config = StafferConfig()
    model = StafferModel(config)
    top = model.heads.stave_top_ref.sigmoid()
    bot = model.heads.stave_bottom_ref.sigmoid()
    assert (bot > top).all()
