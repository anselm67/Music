import torch

from noter import NoterConfig, Vocab
from scorer import ScorerConfig, ScorerModule
from staffer import StafferConfig


def _tiny_module() -> tuple[ScorerModule, ScorerConfig, Vocab]:
    vocab = Vocab(
        {"PAD": 0, "UNK": 1, "SOS": 2, "EOS": 3, "SIL": 4, "C": 5, "rest": 6, "=": 7}
    )
    config = ScorerConfig(
        # Small image so the ViT backbone is cheap on CPU.
        staffer=StafferConfig(
            max_height=128,
            max_width=128,
            embed_dim=16,
            num_heads=2,
            mlp_dim=32,
            num_encoder_layers=1,
            num_decoder_layers=1,
        ),
        noter=NoterConfig(
            embed_dim=16,
            num_head=2,
            mlp_dim=32,
            num_encoder_layers=1,
            num_decoder_layers=1,
            max_seqlen=8,
            max_chords=2,
            max_staves=2,
        ),
        freeze_staffer_steps=0,
    )
    config.use_vocab(vocab)
    return ScorerModule(config), config, vocab


def _batch(config: ScorerConfig, b: int = 2) -> tuple[torch.Tensor, ...]:
    sc, nc = config.staffer, config.noter
    M, N = sc.num_stave_queries, sc.num_system_queries
    H, W = sc.image_shape
    t, mc = nc.max_seqlen, nc.max_chords
    image = torch.randn(b, sc.in_channels, H, W)
    gt_sys = torch.zeros(b, N, 4)
    gt_stave = torch.zeros(b, M, 4)
    gt_assign = torch.full((b, M), -1, dtype=torch.long)
    # Two staves per page, both in system 0 (normalised ltrb).
    for i in range(b):
        gt_stave[i, 0] = torch.tensor([0.1, 0.1, 0.9, 0.3])
        gt_stave[i, 1] = torch.tensor([0.1, 0.5, 0.9, 0.7])
        gt_assign[i, 0] = 0
        gt_assign[i, 1] = 0
        gt_sys[i, 0] = torch.tensor([0.1, 0.1, 0.9, 0.7])
    stave_tokens = torch.randint(0, config.noter.vocab_size, (b, M, t, mc))
    stave_durs = torch.randn(b, M, t, mc)
    stave_dmask = torch.rand(b, M, t, mc) > 0.5
    return image, gt_sys, gt_stave, gt_assign, stave_tokens, stave_durs, stave_dmask


def test_step_runs_and_duration_head_backprops() -> None:
    module, config, _ = _tiny_module()
    # _step (not training_step) so the train/lr log doesn't need a Trainer.
    loss = module._step(_batch(config), "val")
    assert loss.requires_grad and torch.isfinite(loss)
    loss.backward()
    # The transcription loss reached the noter's duration head.
    grad = module.model.noter.dur_head.weight.grad
    assert grad is not None and grad.abs().sum() > 0


def test_generate_grouped_returns_durations_aligned_with_tokens() -> None:
    # Drive the cross-stave generation directly with controlled full-width crops
    # (the detector's random geometry would otherwise flake the encoder).
    module, config, vocab = _tiny_module()
    module.eval()
    h, w = config.noter.input_shape
    K = 2
    crops = torch.randn(K, 1, h, w)
    widths = torch.full((K,), w)
    owners = torch.zeros(K, dtype=torch.long)  # both staves in system 0
    bearing = torch.tensor(
        [vocab.decode(i) in {"C", "rest"} for i in range(len(vocab))]
    )
    tokens, durations = module._generate_grouped(crops, widths, owners, bearing)
    assert tokens.shape == durations.shape
    assert tokens.shape[0] == K
    assert tokens.shape[2] == config.noter.max_chords
    # Non-duration-bearing tokens carry a zero length; the rest are finite.
    assert torch.isfinite(durations).all()
