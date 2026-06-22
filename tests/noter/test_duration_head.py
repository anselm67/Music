import torch

from noter import NoterConfig, NoterModule, Vocab
from noter.duration import NUM_DUR_BINS


def _tiny_module() -> tuple[NoterModule, NoterConfig, Vocab]:
    vocab = Vocab(
        {"PAD": 0, "UNK": 1, "SOS": 2, "EOS": 3, "SIL": 4, "C": 5, "rest": 6, "=": 7}
    )
    config = NoterConfig(
        embed_dim=16,
        num_head=2,
        mlp_dim=32,
        num_encoder_layers=1,
        num_decoder_layers=1,
        max_seqlen=8,
        max_chords=2,
        max_staves=2,
    )
    config.use_vocab(vocab)
    return NoterModule(config), config, vocab


def _batch(config: NoterConfig, b: int = 2) -> tuple[torch.Tensor, ...]:
    s, t, mc = config.max_staves, config.max_seqlen, config.max_chords
    h, w = config.input_shape
    source = torch.randn(b, s, 1, h, w)
    widths = torch.full((b, s), w)
    target = torch.randint(0, config.vocab_size, (b, s, t, mc))
    target_dur = torch.randn(b, s, t, mc)
    target_dur_mask = torch.rand(b, s, t, mc) > 0.5
    stave_mask = torch.ones(b, s, dtype=torch.bool)
    return source, widths, target, target_dur, target_dur_mask, stave_mask


def test_training_step_runs_and_backprops() -> None:
    module, config, _ = _tiny_module()
    # _step (not training_step) so the train/lr log doesn't need a Trainer.
    loss = module._step(_batch(config), "val")
    assert loss.requires_grad and torch.isfinite(loss)
    loss.backward()
    # The duration head received gradient.
    assert module.model.dur_head.weight.grad is not None
    assert module.model.dur_head.weight.grad.abs().sum() > 0


def test_forward_returns_token_and_duration_heads() -> None:
    module, config, _ = _tiny_module()
    source, widths, target, dur, dmask, smask = _batch(config)
    logits, dur_logits = module.forward(source, widths, target, smask, dur, dmask)
    b, s, t, mc = target.shape
    assert logits.shape == (b, s, t, mc, config.vocab_size)
    assert dur_logits.shape == (b, s, t, mc, NUM_DUR_BINS)


def test_detach_duration_severs_decoder_gradient() -> None:
    # The scorer sets detach_duration so the duration loss cannot flow back through
    # the decoder (and thus the crop bridge) into the detector's boxes.
    module, config, _ = _tiny_module()
    model = module.model
    source, widths, target, dur, dmask, smask = _batch(config, b=1)
    b, s = source.shape[:2]
    memory, mem_pad = model.encode(
        source.reshape(b * s, *source.shape[2:]), widths.reshape(b * s)
    )
    causal = module._causal_mask(target.shape[2])

    # Detached: the duration head gets gradient, the decoder does not.
    model.zero_grad()
    _, dur_logits = model.decode(
        target, memory, mem_pad, smask, causal, dur, dmask, detach_duration=True
    )
    dur_logits.sum().backward()
    head_grad = model.dur_head.weight.grad
    assert head_grad is not None and head_grad.abs().sum() > 0
    assert all(
        p.grad is None or p.grad.abs().sum() == 0 for p in model.decoder.parameters()
    )

    # Not detached (the standalone-noter default): the decoder co-trains.
    model.zero_grad()
    _, dur_logits = model.decode(
        target, memory, mem_pad, smask, causal, dur, dmask, detach_duration=False
    )
    dur_logits.sum().backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.decoder.parameters()
    )


def test_predict_returns_tokens_and_durations() -> None:
    module, config, vocab = _tiny_module()
    module.eval()
    source, widths, *_, smask = _batch(config, b=1)
    bearing = torch.tensor(
        [vocab.decode(i) in {"C", "rest"} for i in range(len(vocab))]
    )
    tokens, durs = module.predict(source, widths, smask, bearing)
    assert tokens.shape[:2] == (1, config.max_staves)
    assert durs.shape == tokens.shape
