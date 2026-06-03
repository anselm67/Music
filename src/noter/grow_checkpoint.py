"""Grow a trained NoterModule state dict to a larger vocabulary.

Fine-tuning a pretrained checkpoint on an extended vocab needs the two
vocab-sized tensors resized while keeping every learned row in place:

- ``model.target_embedder.embedding.weight`` ``(V, D)`` — the input token
  embedding; a plain row append.
- ``model.mlp.weight`` ``(max_chords * V, D)`` / ``model.mlp.bias``
  ``(max_chords * V,)`` — the output head. ``NoterModel.forward`` reshapes its
  output ``view(B, T, max_chords, -1)``, so the head rows are laid out
  slot-major / vocab-minor (per-slot stride ``V``). The old rows must therefore
  be spliced per chord slot into the new ``V'`` stride, NOT copied as a flat
  prefix.

Every other tensor (transformer, source embedder, positional embeds, the causal
mask buffer) is vocab-independent and copied verbatim.
"""

from torch import Tensor

EMBED_KEY = "model.target_embedder.embedding.weight"
HEAD_WEIGHT_KEY = "model.mlp.weight"
HEAD_BIAS_KEY = "model.mlp.bias"


def grow_state_dict(
    old_sd: dict[str, Tensor],
    new_sd: dict[str, Tensor],
    *,
    max_chords: int,
) -> dict[str, Tensor]:
    """Splice a smaller-vocab state dict into a fresh larger-vocab one.

    ``new_sd`` is the state dict of a freshly-constructed module at the target
    vocab size (its new token rows are already correctly initialised). Returns a
    new state dict where every shape-matching tensor is taken verbatim from
    ``old_sd`` and the three vocab-sized tensors keep their old rows, with the
    appended rows left at ``new_sd``'s fresh initialisation.
    """
    if set(old_sd) != set(new_sd):
        raise ValueError(
            "state dict keys differ between checkpoint and fresh model: "
            f"missing={set(old_sd) - set(new_sd)}, extra={set(new_sd) - set(old_sd)}"
        )

    out = {k: v.clone() for k, v in new_sd.items()}
    for key, old in old_sd.items():
        new = new_sd[key]
        if old.shape == new.shape:
            out[key] = old.clone()
        elif key == EMBED_KEY:
            v = old.shape[0]
            out[key][:v] = old
        elif key == HEAD_WEIGHT_KEY:
            v, d = old.shape[0] // max_chords, old.shape[1]
            vp = new.shape[0] // max_chords
            out[key].view(max_chords, vp, d)[:, :v, :] = old.view(max_chords, v, d)
        elif key == HEAD_BIAS_KEY:
            v = old.shape[0] // max_chords
            vp = new.shape[0] // max_chords
            out[key].view(max_chords, vp)[:, :v] = old.view(max_chords, v)
        else:
            raise ValueError(
                f"unexpected shape change for {key}: {tuple(old.shape)} -> "
                f"{tuple(new.shape)}"
            )
    return out
