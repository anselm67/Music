import pytest
import torch

from noter.grow_checkpoint import (
    EMBED_KEY,
    HEAD_BIAS_KEY,
    HEAD_WEIGHT_KEY,
    grow_state_dict,
)


def _make_sd(vocab: int, dim: int, max_chords: int) -> dict[str, torch.Tensor]:
    # Distinct, recognisable values so splicing can be verified row-by-row.
    return {
        "_causal_mask_buf": torch.ones(3, 3),
        "model.transformer.weight": torch.arange(dim * dim, dtype=torch.float).reshape(
            dim, dim
        ),
        EMBED_KEY: torch.arange(vocab * dim, dtype=torch.float).reshape(vocab, dim),
        HEAD_WEIGHT_KEY: torch.arange(
            max_chords * vocab * dim, dtype=torch.float
        ).reshape(max_chords * vocab, dim),
        HEAD_BIAS_KEY: torch.arange(max_chords * vocab, dtype=torch.float),
    }


class TestGrowStateDict:
    def test_preserves_shared_and_grows_vocab(self) -> None:
        V, VP, D, H = 4, 7, 3, 2
        old = _make_sd(V, D, H)
        new = _make_sd(VP, D, H)
        # make the fresh rows obviously different so we can tell them apart
        for k in (EMBED_KEY, HEAD_WEIGHT_KEY, HEAD_BIAS_KEY):
            new[k] = new[k] + 1000.0

        out = grow_state_dict(old, new, max_chords=H)

        # vocab-independent tensors copied verbatim from the checkpoint
        assert torch.equal(
            out["model.transformer.weight"], old["model.transformer.weight"]
        )
        assert torch.equal(out["_causal_mask_buf"], old["_causal_mask_buf"])

        # embedding: old rows kept, shape grown
        assert out[EMBED_KEY].shape == (VP, D)
        assert torch.equal(out[EMBED_KEY][:V], old[EMBED_KEY])

        # head: per chord-slot splice (slot-major / vocab-minor layout)
        assert out[HEAD_WEIGHT_KEY].shape == (H * VP, D)
        ow = old[HEAD_WEIGHT_KEY].view(H, V, D)
        nw = out[HEAD_WEIGHT_KEY].view(H, VP, D)
        assert torch.equal(nw[:, :V, :], ow)

        assert out[HEAD_BIAS_KEY].shape == (H * VP,)
        ob = old[HEAD_BIAS_KEY].view(H, V)
        nb = out[HEAD_BIAS_KEY].view(H, VP)
        assert torch.equal(nb[:, :V], ob)

    def test_appended_rows_keep_fresh_init(self) -> None:
        V, VP, D, H = 4, 7, 3, 2
        old = _make_sd(V, D, H)
        new = _make_sd(VP, D, H)
        for k in (EMBED_KEY, HEAD_WEIGHT_KEY, HEAD_BIAS_KEY):
            new[k] = new[k] + 1000.0

        out = grow_state_dict(old, new, max_chords=H)

        # the new tail rows come from new_sd, not old
        assert torch.equal(out[EMBED_KEY][V:], new[EMBED_KEY][V:])
        nw = out[HEAD_WEIGHT_KEY].view(H, VP, D)
        assert torch.equal(nw[:, V:, :], new[HEAD_WEIGHT_KEY].view(H, VP, D)[:, V:, :])

    def test_equal_vocab_is_verbatim(self) -> None:
        old = _make_sd(5, 3, 2)
        new = _make_sd(5, 3, 2)
        for k in (EMBED_KEY, HEAD_WEIGHT_KEY, HEAD_BIAS_KEY):
            new[k] = new[k] + 1000.0
        out = grow_state_dict(old, new, max_chords=2)
        for k in old:
            assert torch.equal(out[k], old[k])

    def test_mismatched_keys_raise(self) -> None:
        old = _make_sd(4, 3, 2)
        new = _make_sd(7, 3, 2)
        del new["_causal_mask_buf"]
        with pytest.raises(ValueError, match="keys differ"):
            grow_state_dict(old, new, max_chords=2)
