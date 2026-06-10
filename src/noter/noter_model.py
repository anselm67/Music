"""A ViT model from converting staves to kern tokens."""

from dataclasses import asdict, dataclass, field

import torch
from torch import Tensor, nn
from torchvision.transforms import InterpolationMode

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
    warmup_steps: int = 500
    # Train-only box-jitter augmentation: probability a train sample's box is
    # jittered (0 = disabled). Applied to the train split only.
    jitter: float = 0.0
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
        self.norm = nn.LayerNorm(config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, target: Tensor) -> Tensor:
        embeds = self.embedding(target)  # (B, T, H, D)
        embeds = embeds + self.chord_pos_embed  # per-slot position signal
        B, T, H, D = embeds.shape
        assert T <= self.pos_embed.shape[0], (
            f"T={T} exceeds max_seqlen={self.pos_embed.shape[0]}"
        )
        x = self.chord_proj(embeds.view(B, T, H * D))
        return self.dropout(self.norm(x + self.pos_embed[:T]))


class NoterModel(nn.Module):
    def __init__(self, config: NoterConfig) -> None:
        super().__init__()
        self.config = config
        self.source_embedder = SourceEmbedding(config)
        self.target_embedder = TargetEmbedder(config)
        self.transformer = nn.Transformer(
            d_model=config.embed_dim,
            nhead=config.num_head,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dim_feedforward=config.mlp_dim,
            dropout=config.dropout,
            batch_first=True,
        )
        self.mlp = nn.Linear(config.embed_dim, config.max_chords * config.vocab_size)

    def make_src_padding_mask(self, widths: Tensor) -> Tensor:
        """
        widths: (B,) actual image widths in pixels
        returns: (B, num_patches) True where patch is padding
        """
        c = self.config
        num_patches_h = c.input_shape[0] // c.patch_height
        num_patches_w = c.input_shape[1] // c.patch_width
        valid_patches_w = widths // c.patch_width  # (B,)

        # patch index in the flattened sequence: row * num_patches_w + col
        # a patch is padding if its column >= valid_patches_w
        col_indices = torch.arange(num_patches_w, device=widths.device)  # (W,)
        col_mask = col_indices.unsqueeze(0) >= valid_patches_w.unsqueeze(
            1
        )  # (B, num_patches_w)

        # expand to all rows: (B, num_patches_h * num_patches_w)
        return (
            col_mask.unsqueeze(1).expand(-1, num_patches_h, -1).reshape(len(widths), -1)
        )

    def forward(
        self,
        source: Tensor,
        source_widths: Tensor,
        target: Tensor,
        attention_mask: Tensor,
        tgt_pad_mask: Tensor,
    ) -> Tensor:
        source_embeds = self.source_embedder(source)
        target_embeds = self.target_embedder(target)
        src_padding_mask = self.make_src_padding_mask(source_widths)
        outs = self.transformer(
            src=source_embeds,
            tgt=target_embeds,
            tgt_mask=attention_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_padding_mask,
        )
        B, T, _ = target.shape
        return self.mlp(outs).view(B, T, self.config.max_chords, -1)

    def encode(self, source: Tensor, source_widths: Tensor) -> tuple[Tensor, Tensor]:
        src_pad_mask = self.make_src_padding_mask(source_widths)
        memory = self.transformer.encoder(
            self.source_embedder(source), src_key_padding_mask=src_pad_mask
        )
        return memory, src_pad_mask

    def decode(
        self,
        target: Tensor,
        memory: Tensor,
        target_mask: Tensor,
        tgt_pad_mask: Tensor,
        memory_pad_mask: Tensor,
    ) -> Tensor:
        target_embeds = self.target_embedder(target)
        outs = self.transformer.decoder(
            target_embeds,
            memory,
            tgt_mask=target_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=memory_pad_mask,
        )
        B, T, _ = outs.shape
        return self.mlp(outs).view(B, T, self.config.max_chords, -1)
