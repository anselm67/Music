import torch

from kern import NUM_ARTICULATIONS
from scorer import ScorerConfig, ScorerModule


def _tiny_config() -> ScorerConfig:
    cfg = ScorerConfig()
    cfg.noter.embed_dim = 32
    cfg.noter.num_head = 2
    cfg.noter.num_encoder_layers = 1
    cfg.noter.num_decoder_layers = 1
    cfg.noter.mlp_dim = 64
    cfg.noter.max_seqlen = 8
    cfg.noter.max_chords = 2
    cfg.noter.vocab_size = 12
    cfg.noter.pad_idx = 0
    cfg.freeze_staffer_steps = 0
    return cfg


def test_generate_grouped_returns_tokens_and_articulations() -> None:
    cfg = _tiny_config()
    module = ScorerModule(cfg).eval()
    # 3 staves: system 0 has 2, system 1 has 1.
    K = 3
    crops = torch.rand(K, 1, *cfg.noter.input_shape)
    widths = torch.full((K,), cfg.noter.input_shape[1])
    owners = torch.tensor([0, 0, 1])
    tokens, arts = module._generate_grouped(crops, widths, owners)
    assert tokens.shape[0] == K
    assert arts.shape == (*tokens.shape, NUM_ARTICULATIONS)


def test_step_runs_with_articulation_loss() -> None:
    cfg = _tiny_config()
    module = ScorerModule(cfg)
    s = cfg.staffer
    B, H, W = 1, *s.image_shape
    T, mc = cfg.noter.max_seqlen, cfg.noter.max_chords

    image = torch.rand(B, 1, H, W)
    gt_sys = torch.zeros(B, s.num_system_queries, 4)
    gt_sys[0, 0] = torch.tensor([0.1, 0.1, 0.9, 0.4])
    gt_stave = torch.zeros(B, s.num_stave_queries, 4)
    gt_stave[0, 0] = torch.tensor([0.1, 0.12, 0.9, 0.22])
    gt_stave[0, 1] = torch.tensor([0.1, 0.28, 0.9, 0.38])
    gt_assign = torch.full((B, s.num_stave_queries), -1, dtype=torch.long)
    gt_assign[0, 0] = 0
    gt_assign[0, 1] = 0
    tokens = torch.zeros(B, s.num_stave_queries, T, mc, dtype=torch.long)
    for k in (0, 1):
        tokens[0, k, 0] = 2  # SOS
        tokens[0, k, 1] = 5 + k  # a note
        tokens[0, k, 2] = 3  # EOS
    arts = (torch.rand(B, s.num_stave_queries, T, mc, NUM_ARTICULATIONS) > 0.7).float()

    loss = module._step((image, gt_sys, gt_stave, gt_assign, tokens, arts), "val")
    assert torch.isfinite(loss)
