"""A ViT model from converting staves to kern tokens."""

from dataclasses import asdict, dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.transforms import InterpolationMode

from kern import NUM_ARTICULATIONS
from staffer import StafferConfig
from utils import current_commit

from .noter_vocab import Vocab


@dataclass
class NoterConfig:
    id_name: str = "default"
    git_hash: str = current_commit()

    # Derived from the staffer canvas (StafferConfig.image_shape, itself
    # computed by scale_to_patch): the scorer crops the noter's input from a
    # page letterboxed to the staffer shape, so the standalone noter must train
    # on the same page geometry. Sourced here rather than copy-pasting the
    # computed [960, 688] so the two never drift apart again. NB: when used as a
    # ScorerConfig sub-config this is force-set to the scorer's staffer canvas
    # (ScorerConfig.__post_init__), so an explicit value passed there is ignored.
    page_shape: list[int] = field(default_factory=lambda: StafferConfig().image_shape)

    input_shape: list[int] = field(default_factory=lambda: [64, 6 * 128])
    max_chords: int = 8
    max_seqlen: int = 128  # Also known as T
    # Staves per system — the cross-stave decode unit. A system's staves are
    # row-aligned (one shared barline grid), batched together and padded to this
    # width (a stave_mask marks the real ones). 2 covers the System2 corpus.
    max_staves: int = 2
    vocab_size: int = -1
    pad_idx: int = -1

    # Separate articulation flow: a per-note multi-hot (tie, staccato, fermata,
    # accent) predicted by its own head and fed back on the input side, alongside
    # the pitch x duration token (which is unchanged). Off by default so existing
    # checkpoints and the scorer are unaffected.
    articulations: bool = False
    # BCE weight for the articulation head. The duration-head pilot showed a high
    # auxiliary weight taxes the primary (pitch) task; keep this at parity (1.0).
    articulation_weight: float = 1.0

    interpolation: InterpolationMode = InterpolationMode.BILINEAR
    antialias: bool = False

    # Model parameters.
    in_channels: int = 1
    embed_dim: int = 256  # Also known as D
    patch_width: int = 4
    patch_height: int = -1
    num_head: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    mlp_dim: int = 1024
    dropout: float = 0.1

    # Training config.
    batch_size: int = 16
    train_len: int = -1
    valid_len: int = -1
    lr: float = 3e-4
    weight_decay: float = 1e-2
    warmup_steps: int = 500
    max_steps: int = field(init=False)

    def __post_init__(self) -> None:
        assert self.embed_dim % self.num_head == 0, (
            f"embed_dim={self.embed_dim} not divisible by num_head={self.num_head}"
        )
        if self.patch_height == -1:
            self.patch_height = self.input_shape[0]
        if self.train_len == -1:
            self.train_len = 12500 * self.batch_size
        if self.valid_len == -1:
            self.valid_len = 100 * self.batch_size
        self.max_steps = 4 * (self.train_len // self.batch_size)

    def asdict(self) -> dict[str, object]:
        obj = asdict(self)
        obj.pop("max_steps")
        return obj

    def use_vocab(self, vocab: Vocab) -> None:
        self.vocab_size = len(vocab)
        self.pad_idx = vocab.PAD


class SourceEmbedding(nn.Module):
    def __init__(self, config: NoterConfig) -> None:
        super().__init__()
        self.config = config
        self.proj = nn.Conv2d(
            config.in_channels,
            config.embed_dim,
            kernel_size=(config.patch_height, config.patch_width),
            stride=(config.patch_height, config.patch_width),
        )
        num_patches = (config.input_shape[0] // config.patch_height) * (
            config.input_shape[1] // config.patch_width
        )
        self.pos_embed = nn.Parameter(0.02 * torch.randn(num_patches, config.embed_dim))
        self.norm = nn.LayerNorm(config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)  # (B, D, H, W)
        x = x.flatten(2).transpose(1, 2)  # B, num_patches, D)
        x += self.pos_embed
        return self.dropout(self.norm(x))


class TargetEmbedder(nn.Module):
    def __init__(self, config: NoterConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embed_dim,
            padding_idx=config.pad_idx,
        )
        self.chord_proj = nn.Linear(
            config.max_chords * config.embed_dim, config.embed_dim
        )
        self.chord_pos_embed = nn.Parameter(
            0.02 * torch.randn(config.max_chords, config.embed_dim)
        )
        self.pos_embed = nn.Parameter(
            0.02 * torch.randn(config.max_seqlen, config.embed_dim)
        )
        # Per-slot articulation injected on the input side as a learned projection
        # of the multi-hot, mirroring the (separate) articulation head. Zero-init
        # so the token path is unperturbed at the start of training (identity), and
        # so a checkpoint trained without articulations loads unchanged.
        self.art_proj: nn.Linear | None = None
        if config.articulations:
            self.art_proj = nn.Linear(NUM_ARTICULATIONS, config.embed_dim)
            nn.init.zeros_(self.art_proj.weight)
            nn.init.zeros_(self.art_proj.bias)
        self.norm = nn.LayerNorm(config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, target: Tensor, articulations: Tensor | None = None) -> Tensor:
        embeds = self.embedding(target)  # (B, T, H, D)
        embeds = embeds + self.chord_pos_embed  # per-slot position signal
        if articulations is not None and self.art_proj is not None:
            embeds = embeds + self.art_proj(articulations)  # (B, T, H, D)
        B, T, H, D = embeds.shape
        assert T <= self.pos_embed.shape[0], (
            f"T={T} exceeds max_seqlen={self.pos_embed.shape[0]}"
        )
        x = self.chord_proj(embeds.view(B, T, H * D))
        return self.dropout(self.norm(x + self.pos_embed[:T]))


class CrossStaveDecoderLayer(nn.Module):
    """A post-norm Transformer decoder layer + a gated cross-stave attention branch.

    The first three sublayers (causal self-attention, image cross-attention, FFN)
    and their submodule names (``self_attn`` / ``multihead_attn`` / ``linear1`` /
    ``linear2`` / ``norm1-3``) match ``nn.TransformerDecoderLayer`` exactly, so a
    single-stave noter checkpoint loads straight in (see ``remap_legacy_state_dict``).
    The 4th sublayer lets a staff attend to the **other staves of its system** along
    the row axis (kern is a row-aligned spine grid). Its contribution is scaled by a
    **zero-initialised gate**, so at init the layer is identical to the pretrained one
    and a single-staff system — whose siblings are all masked — stays identical.
    """

    def __init__(self, config: NoterConfig) -> None:
        super().__init__()
        d, h, p = config.embed_dim, config.num_head, config.dropout
        # --- standard decoder sublayers (names match nn.TransformerDecoderLayer) ---
        self.self_attn = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)
        self.linear1 = nn.Linear(d, config.mlp_dim)
        self.linear2 = nn.Linear(config.mlp_dim, d)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.norm3 = nn.LayerNorm(d)
        self.dropout = nn.Dropout(p)
        self.dropout1 = nn.Dropout(p)
        self.dropout2 = nn.Dropout(p)
        self.dropout3 = nn.Dropout(p)
        # --- new: gated cross-stave attention branch (identity at init) ---
        self.cross_stave_attn = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)
        self.norm_xs = nn.LayerNorm(d)
        self.dropout_xs = nn.Dropout(p)
        self.xs_gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: Tensor,  # (B*S, T, D)
        memory: Tensor,  # (B*S, P, D)
        tgt_mask: Tensor,  # (T, T) causal
        tgt_kpm: Tensor,  # (B*S, T)
        mem_kpm: Tensor,  # (B*S, P)
        xs_mask: Tensor,  # (S*T, S*T) cross-stave time-causal
        xs_kpm: Tensor,  # (B, S*T) — True on padded staves
        bsz: int,
        staves: int,
    ) -> Tensor:
        a = self.self_attn(
            x, x, x, attn_mask=tgt_mask, key_padding_mask=tgt_kpm, need_weights=False
        )[0]
        x = self.norm1(x + self.dropout1(a))
        a = self.multihead_attn(
            x, memory, memory, key_padding_mask=mem_kpm, need_weights=False
        )[0]
        x = self.norm2(x + self.dropout2(a))
        # Cross-stave: couple a system's staves along the (shared) row axis.
        t, d = x.shape[1], x.shape[2]
        xq = x.view(bsz, staves, t, d).reshape(bsz, staves * t, d)
        h = self.norm_xs(xq)
        a = self.cross_stave_attn(
            h, h, h, attn_mask=xs_mask, key_padding_mask=xs_kpm, need_weights=False
        )[0]
        xq = xq + self.xs_gate * self.dropout_xs(a)
        x = xq.view(bsz, staves, t, d).reshape(bsz * staves, t, d)
        a = self.linear2(self.dropout(F.relu(self.linear1(x))))
        return self.norm3(x + self.dropout3(a))


class CrossStaveDecoder(nn.Module):
    """Stack of ``CrossStaveDecoderLayer`` + a final norm (matches legacy decoder)."""

    def __init__(self, config: NoterConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            CrossStaveDecoderLayer(config) for _ in range(config.num_decoder_layers)
        )
        self.norm = nn.LayerNorm(config.embed_dim)

    def forward(self, x: Tensor, *args: object) -> Tensor:
        for layer in self.layers:
            x = layer(x, *args)  # type: ignore[arg-type]
        return self.norm(x)


class NoterModel(nn.Module):
    def __init__(self, config: NoterConfig) -> None:
        super().__init__()
        self.config = config
        self.source_embedder = SourceEmbedding(config)
        self.target_embedder = TargetEmbedder(config)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_head,
            dim_feedforward=config.mlp_dim,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            config.num_encoder_layers,
            norm=nn.LayerNorm(config.embed_dim),
        )
        self.decoder = CrossStaveDecoder(config)
        self.mlp = nn.Linear(config.embed_dim, config.max_chords * config.vocab_size)
        # Separate per-slot articulation head (multi-label logits), only when enabled.
        self.art_head: nn.Linear | None = None
        if config.articulations:
            self.art_head = nn.Linear(
                config.embed_dim, config.max_chords * NUM_ARTICULATIONS
            )

    def make_src_padding_mask(self, widths: Tensor) -> Tensor:
        """
        widths: (N,) actual image widths in pixels
        returns: (N, num_patches) True where patch is padding
        """
        c = self.config
        num_patches_h = c.input_shape[0] // c.patch_height
        num_patches_w = c.input_shape[1] // c.patch_width
        valid_patches_w = widths // c.patch_width  # (N,)

        # patch index in the flattened sequence: row * num_patches_w + col
        # a patch is padding if its column >= valid_patches_w
        col_indices = torch.arange(num_patches_w, device=widths.device)  # (W,)
        col_mask = col_indices.unsqueeze(0) >= valid_patches_w.unsqueeze(
            1
        )  # (N, num_patches_w)

        # expand to all rows: (N, num_patches_h * num_patches_w)
        return (
            col_mask.unsqueeze(1).expand(-1, num_patches_h, -1).reshape(len(widths), -1)
        )

    def _cross_stave_mask(
        self, staves: int, seqlen: int, device: torch.device
    ) -> Tensor:
        """``(S*T, S*T)`` bool, True where disallowed: query ``(i, t)`` may attend
        key ``(j, s)`` for any staff pair iff ``s <= t`` — causal in the shared row
        clock, free across staves."""
        times = torch.arange(seqlen, device=device).repeat(staves)  # (S*T,)
        return times.unsqueeze(0) > times.unsqueeze(1)  # [query, key]: key_t > query_t

    def encode(self, source: Tensor, source_widths: Tensor) -> tuple[Tensor, Tensor]:
        """Per-staff encode of flat crops ``(N, 1, H, W)`` → memory ``(N, P, D)``."""
        src_pad_mask = self.make_src_padding_mask(source_widths)
        memory = self.encoder(
            self.source_embedder(source), src_key_padding_mask=src_pad_mask
        )
        return memory, src_pad_mask

    def _decode_hidden(
        self,
        target: Tensor,  # (B, S, T, max_chords)
        memory: Tensor,  # (B*S, P, D)
        memory_pad_mask: Tensor,  # (B*S, P)
        stave_mask: Tensor,  # (B, S) — True on real staves
        target_mask: Tensor,  # (T, T) causal
        articulations: Tensor | None,  # (B, S, T, max_chords, NUM_ARTICULATIONS)
    ) -> tuple[Tensor, tuple[int, int, int, int]]:
        """Run the decoder → hidden states ``(B*S, T, D)`` + the ``(B,S,T,mc)`` shape,
        shared by the token and articulation heads (one decoder pass)."""
        bsz, staves, t, mc = target.shape
        tgt_flat = target.reshape(bsz * staves, t, mc)
        art_flat = (
            articulations.reshape(bsz * staves, t, mc, NUM_ARTICULATIONS)
            if articulations is not None
            else None
        )
        tgt_embeds = self.target_embedder(tgt_flat, art_flat)  # (B*S, T, D)
        tgt_kpm = (tgt_flat == self.config.pad_idx).all(dim=-1)  # (B*S, T)
        xs_mask = self._cross_stave_mask(staves, t, target.device)
        xs_kpm = (~stave_mask).unsqueeze(-1).expand(bsz, staves, t).reshape(bsz, -1)
        outs = self.decoder(
            tgt_embeds,
            memory,
            target_mask,
            tgt_kpm,
            memory_pad_mask,
            xs_mask,
            xs_kpm,
            bsz,
            staves,
        )
        return outs, (bsz, staves, t, mc)

    def decode(
        self,
        target: Tensor,  # (B, S, T, max_chords)
        memory: Tensor,  # (B*S, P, D)
        memory_pad_mask: Tensor,  # (B*S, P)
        stave_mask: Tensor,  # (B, S) — True on real staves
        target_mask: Tensor,  # (T, T) causal
    ) -> Tensor:
        """Decode a batch of systems → logits ``(B, S, T, max_chords, vocab_size)``."""
        outs, (bsz, staves, t, mc) = self._decode_hidden(
            target, memory, memory_pad_mask, stave_mask, target_mask, None
        )
        return self.mlp(outs).view(bsz, staves, t, mc, -1)

    def decode_both(
        self,
        target: Tensor,  # (B, S, T, max_chords)
        memory: Tensor,  # (B*S, P, D)
        memory_pad_mask: Tensor,  # (B*S, P)
        stave_mask: Tensor,  # (B, S) — True on real staves
        target_mask: Tensor,  # (T, T) causal
        articulations: Tensor | None,  # (B, S, T, max_chords, NUM_ARTICULATIONS)
    ) -> tuple[Tensor, Tensor]:
        """Decode → (token logits ``(B,S,T,mc,vocab)``, articulation logits
        ``(B,S,T,mc,NUM_ARTICULATIONS)``). Requires ``config.articulations``."""
        assert self.art_head is not None, "decode_both needs config.articulations"
        outs, (bsz, staves, t, mc) = self._decode_hidden(
            target, memory, memory_pad_mask, stave_mask, target_mask, articulations
        )
        tok_logits = self.mlp(outs).view(bsz, staves, t, mc, -1)
        art_logits = self.art_head(outs).view(bsz, staves, t, mc, NUM_ARTICULATIONS)
        return tok_logits, art_logits

    def forward(
        self,
        source: Tensor,  # (B, S, 1, H, W)
        source_widths: Tensor,  # (B, S)
        target: Tensor,  # (B, S, T, max_chords)
        stave_mask: Tensor,  # (B, S)
        target_mask: Tensor,  # (T, T) causal
    ) -> Tensor:
        bsz, staves = source.shape[:2]
        flat_src = source.reshape(bsz * staves, *source.shape[2:])
        memory, mem_pad = self.encode(flat_src, source_widths.reshape(bsz * staves))
        return self.decode(target, memory, mem_pad, stave_mask, target_mask)

    def forward_both(
        self,
        source: Tensor,  # (B, S, 1, H, W)
        source_widths: Tensor,  # (B, S)
        target: Tensor,  # (B, S, T, max_chords)
        stave_mask: Tensor,  # (B, S)
        target_mask: Tensor,  # (T, T) causal
        articulations: Tensor,  # (B, S, T, max_chords, NUM_ARTICULATIONS)
    ) -> tuple[Tensor, Tensor]:
        """``forward`` with the articulation head: (token logits, articulation
        logits). Requires ``config.articulations``."""
        bsz, staves = source.shape[:2]
        flat_src = source.reshape(bsz * staves, *source.shape[2:])
        memory, mem_pad = self.encode(flat_src, source_widths.reshape(bsz * staves))
        return self.decode_both(
            target, memory, mem_pad, stave_mask, target_mask, articulations
        )

    @staticmethod
    def remap_legacy_state_dict(state: dict[str, Tensor]) -> dict[str, Tensor]:
        """Map a single-stave (``nn.Transformer``) noter state dict onto this split
        encoder/decoder. Keys are prefix-renamed; the new cross-stave params have no
        legacy counterpart and load fresh (zero gate). Works on bare or ``model.``-
        prefixed dicts."""
        return {
            k.replace("transformer.encoder", "encoder").replace(
                "transformer.decoder", "decoder"
            ): v
            for k, v in state.items()
        }
