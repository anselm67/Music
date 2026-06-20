"""A ViT model from converting staves to kern tokens."""

from dataclasses import asdict, dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.transforms import InterpolationMode

from staffer import StafferConfig
from utils import current_commit

from .duration import NUM_DUR_BINS
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
    # Weight on the duration-head classification loss, added to the token CE.
    dur_loss_weight: float = 2.0
    warmup_steps: int = 500
    # Train-only box-jitter augmentation: probability a train sample's box is
    # jittered (0 = disabled). Applied to the train split only.
    jitter: float = 0.0
    # Train-only scan-augmentation: probability a train page is passed through
    # ScanAugment (0 = disabled). Applied to the train split only.
    augment: float = 0.0
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
        # Per-slot duration injected on the input side: [log2(length)·has, has].
        # The duration left the token (pitch-only vocab), so this is how a slot's
        # note value reaches the decoder. Zero-init so it contributes nothing at
        # the start of training and the token path is unperturbed.
        self.dur_proj = nn.Linear(2, config.embed_dim)
        nn.init.zeros_(self.dur_proj.weight)
        nn.init.zeros_(self.dur_proj.bias)
        self.norm = nn.LayerNorm(config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        target: Tensor,  # (B, T, H) token ids
        dur: Tensor | None = None,  # (B, T, H) log2 length
        dur_mask: Tensor | None = None,  # (B, T, H) True where dur is present
    ) -> Tensor:
        embeds = self.embedding(target)  # (B, T, H, D)
        embeds = embeds + self.chord_pos_embed  # per-slot position signal
        if dur is not None:
            assert dur_mask is not None
            has = dur_mask.to(embeds.dtype)
            dur_in = torch.stack([dur * has, has], dim=-1)  # (B, T, H, 2)
            embeds = embeds + self.dur_proj(dur_in)
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
        # Duration classification head: a softmax over the analytic duration
        # table's NUM_DUR_BINS bins per chord slot, parallel to the token head.
        # Decoded by argmax → bin suffix (noter.duration).
        self.dur_head = nn.Linear(config.embed_dim, config.max_chords * NUM_DUR_BINS)

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

    def decode(
        self,
        target: Tensor,  # (B, S, T, max_chords)
        memory: Tensor,  # (B*S, P, D)
        memory_pad_mask: Tensor,  # (B*S, P)
        stave_mask: Tensor,  # (B, S) — True on real staves
        target_mask: Tensor,  # (T, T) causal
        target_dur: Tensor | None = None,  # (B, S, T, max_chords) log2 length
        target_dur_mask: Tensor | None = None,  # (B, S, T, max_chords) dur present
    ) -> tuple[Tensor, Tensor]:
        """Decode a batch of systems → ``(logits, dur_logits)`` of shapes
        ``(B, S, T, max_chords, vocab_size)`` and
        ``(B, S, T, max_chords, NUM_DUR_BINS)``."""
        bsz, staves, t, mc = target.shape
        tgt_flat = target.reshape(bsz * staves, t, mc)
        dur_flat = (
            target_dur.reshape(bsz * staves, t, mc) if target_dur is not None else None
        )
        dmask_flat = (
            target_dur_mask.reshape(bsz * staves, t, mc)
            if target_dur_mask is not None
            else None
        )
        tgt_embeds = self.target_embedder(tgt_flat, dur_flat, dmask_flat)  # (B*S, T, D)
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
        logits = self.mlp(outs).view(bsz, staves, t, mc, -1)
        dur_logits = self.dur_head(outs).view(bsz, staves, t, mc, NUM_DUR_BINS)
        return logits, dur_logits

    def forward(
        self,
        source: Tensor,  # (B, S, 1, H, W)
        source_widths: Tensor,  # (B, S)
        target: Tensor,  # (B, S, T, max_chords)
        stave_mask: Tensor,  # (B, S)
        target_mask: Tensor,  # (T, T) causal
        target_dur: Tensor | None = None,  # (B, S, T, max_chords)
        target_dur_mask: Tensor | None = None,  # (B, S, T, max_chords)
    ) -> tuple[Tensor, Tensor]:
        bsz, staves = source.shape[:2]
        flat_src = source.reshape(bsz * staves, *source.shape[2:])
        memory, mem_pad = self.encode(flat_src, source_widths.reshape(bsz * staves))
        return self.decode(
            target,
            memory,
            mem_pad,
            stave_mask,
            target_mask,
            target_dur,
            target_dur_mask,
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
