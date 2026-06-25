import torch
from torch import Tensor

from kern import NUM_ARTICULATIONS
from noter import NoterConfig, NoterModule
from noter.noter_model import NoterModel


def _tiny_config(articulations: bool) -> NoterConfig:
    cfg = NoterConfig(
        articulations=articulations,
        embed_dim=32,
        num_head=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        mlp_dim=64,
        max_seqlen=8,
        max_chords=2,
        max_staves=2,
        input_shape=[64, 64],
    )
    cfg.vocab_size = 12
    cfg.pad_idx = 0
    return cfg


def _batch(
    cfg: NoterConfig, B: int = 2
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    S, T, C = cfg.max_staves, cfg.max_seqlen, cfg.max_chords
    source = torch.rand(B, S, 1, *cfg.input_shape)
    widths = torch.full((B, S), cfg.input_shape[1])
    # Real tokens in [5, vocab); SOS at row 0.
    target = torch.randint(5, cfg.vocab_size, (B, S, T, C))
    target[:, :, 0, :] = 2  # SOS
    arts = (torch.rand(B, S, T, C, NUM_ARTICULATIONS) > 0.7).float()
    stave_mask = torch.ones(B, S, dtype=torch.bool)
    return source, widths, target, arts, stave_mask


def test_heads_only_built_when_enabled() -> None:
    off = NoterModel(_tiny_config(False))
    assert off.art_head is None and off.target_embedder.art_proj is None
    on = NoterModel(_tiny_config(True))
    assert on.art_head is not None and on.target_embedder.art_proj is not None


def test_forward_both_shapes_and_backward() -> None:
    cfg = _tiny_config(True)
    model = NoterModel(cfg)
    source, widths, target, arts, stave_mask = _batch(cfg)
    causal = torch.ones(target.shape[2], target.shape[2], dtype=torch.bool).triu(1)
    tok_logits, art_logits = model.forward_both(
        source, widths, target, stave_mask, causal, arts
    )
    B, S, T, C = target.shape
    assert tok_logits.shape == (B, S, T, C, cfg.vocab_size)
    assert art_logits.shape == (B, S, T, C, NUM_ARTICULATIONS)
    (tok_logits.sum() + art_logits.sum()).backward()
    assert model.art_head is not None
    assert model.art_head.weight.grad is not None


def test_zero_init_feedback_is_identity_for_tokens() -> None:
    """At init the articulation feedback is zero-init, so the token logits are
    identical whether or not articulations are fed — the token path is unperturbed."""
    cfg = _tiny_config(True)
    model = NoterModel(cfg).eval()
    source, widths, target, arts, stave_mask = _batch(cfg)
    causal = torch.ones(target.shape[2], target.shape[2], dtype=torch.bool).triu(1)
    flat = source.reshape(-1, *source.shape[2:])
    memory, mem_pad = model.encode(flat, widths.reshape(-1))
    tok_no_art = model.decode(target, memory, mem_pad, stave_mask, causal)
    tok_with_art, _ = model.decode_both(
        target, memory, mem_pad, stave_mask, causal, arts
    )
    assert torch.allclose(tok_no_art, tok_with_art, atol=1e-6)


def test_module_step_and_predict() -> None:
    cfg = _tiny_config(True)
    module = NoterModule(cfg)
    batch = _batch(cfg)
    loss = module._step(batch, "val")
    assert torch.isfinite(loss)
    source, widths, _, _, stave_mask = batch
    preds = module.predict(source, widths, stave_mask)
    assert isinstance(preds, torch.Tensor)
    assert preds.shape[:2] == (source.shape[0], cfg.max_staves)

    tokens, arts = module.predict(source, widths, stave_mask, return_articulations=True)
    assert tokens.shape[:2] == (source.shape[0], cfg.max_staves)
    assert arts.shape == (*tokens.shape, NUM_ARTICULATIONS)
