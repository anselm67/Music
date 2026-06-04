"""StafferModel: hierarchical layout detector for pages of sheet music."""

from dataclasses import asdict, dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn, randn
from torchvision.transforms import InterpolationMode

from utils import current_commit


@dataclass
class StafferConfig:
    id_name: str = "default"
    git_hash: str = current_commit()
    image_shape: list[int] = field(init=False)

    # Maximums as obtained with the "stats" command.
    max_width: int = 1024
    max_height: int = 1449

    in_channels: int = 1
    divider: float = 1.5
    embed_dim: int = 256  # Also known as D
    mlp_dim: int = 1024

    num_heads: int = 8  # Also known as H
    patch_size: int = 16
    dropout: float = 0.1
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4

    # Decoder config.
    # These numbers are from running the following commands:
    # pdmx query -o Stafff16.csv 'index==index' --score 'pages.*.staff_count < 16'
    # pdmx --csv Staff16.csv stats

    num_system_queries: int = 16  # Also known as N
    num_stave_queries: int = 16  # Also known as M

    interpolation: InterpolationMode = InterpolationMode.BILINEAR
    antialias: bool = False

    # Training config.
    batch_size: int = 16
    train_len: int = -1
    valid_len: int = -1
    max_steps: int = field(init=False)
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 4000
    box_loss_multiplier: int = 2
    bottom_bias: float = 3.0  # sampler weight multiplier for bottom-of-page systems

    def scale_to_patch(self, value: int) -> int:
        ret = value // self.divider
        return int(round(ret / self.patch_size) * self.patch_size)

    def __post_init__(self) -> None:
        self.image_shape = [
            self.scale_to_patch(self.max_height),
            self.scale_to_patch(self.max_width),
        ]
        if self.train_len == -1:
            self.train_len = 21875 * self.batch_size
        if self.valid_len == -1:
            self.valid_len = 100 * self.batch_size
        # Trains for 6 epochs by default.
        self.max_steps = 6 * (self.train_len // self.batch_size)

    def asdict(self) -> dict[str, object]:
        obj = asdict(self)
        obj.pop("image_shape")
        obj.pop("max_steps")
        return obj


class PatchEmbedding(nn.Module):
    config: StafferConfig

    def __init__(self, config: StafferConfig):
        super().__init__()
        self.config = config
        num_patch = (
            config.image_shape[0] // config.patch_size,
            config.image_shape[1] // config.patch_size,
        )
        self.proj = nn.Conv2d(
            config.in_channels,
            config.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.pos_embed = nn.Parameter(
            0.02 * randn(num_patch[0] * num_patch[1], config.embed_dim)
        )
        self.norm = nn.LayerNorm(config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x += self.pos_embed
        return self.dropout(self.norm(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: StafferConfig):
        super().__init__()
        self.config = config
        self.qkv = nn.Linear(config.embed_dim, 3 * config.embed_dim)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, config.mlp_dim),
            nn.GELU(),
            nn.Linear(config.mlp_dim, config.embed_dim),
            nn.Dropout(config.dropout),
        )
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.norm2 = nn.LayerNorm(config.embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        B, N, D = x.shape
        x_norm = self.norm1(x)
        qkv = self.qkv(x_norm).reshape(
            B, N, 3, self.config.num_heads, D // self.config.num_heads
        )
        q, k, v = qkv.unbind(2)  # each (B, N, num_heads, head_dim)
        q, k, v = (
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )  # (B, num_heads, N, head_dim)
        x_attn = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.config.dropout if self.training else 0.0
        )
        x_attn = x_attn.transpose(1, 2).reshape(B, N, D)
        x = x + self.proj(x_attn)
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    config: StafferConfig

    def __init__(self, config: StafferConfig):
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbedding(config)
        self.blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.num_encoder_layers)]
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x)
        x = self.blocks(x)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, config: StafferConfig):
        super().__init__()
        D = config.embed_dim
        H = config.num_heads

        # System stream
        self.sys_self_attn_norm = nn.LayerNorm(D)
        self.sys_self_attn = nn.MultiheadAttention(
            D, H, dropout=config.dropout, batch_first=True
        )

        self.sys_cross_attn_norm = nn.LayerNorm(D)
        self.sys_cross_attn = nn.MultiheadAttention(
            D, H, dropout=config.dropout, batch_first=True
        )

        self.sys_ffn_norm = nn.LayerNorm(D)
        self.sys_ffn = nn.Sequential(
            nn.Linear(D, config.mlp_dim),
            nn.GELU(),
            nn.Linear(config.mlp_dim, D),
            nn.Dropout(config.dropout),
        )

        # Stave stream
        self.stave_self_attn_norm = nn.LayerNorm(D)
        self.stave_self_attn = nn.MultiheadAttention(
            D, H, dropout=config.dropout, batch_first=True
        )

        self.stave_cross_attn_norm = nn.LayerNorm(D)
        self.stave_cross_attn = nn.MultiheadAttention(
            D, H, dropout=config.dropout, batch_first=True
        )

        self.stave_ffn_norm = nn.LayerNorm(D)
        self.stave_ffn = nn.Sequential(
            nn.Linear(D, config.mlp_dim),
            nn.GELU(),
            nn.Linear(config.mlp_dim, D),
            nn.Dropout(config.dropout),
        )

    def forward(
        self, sys_q: Tensor, stave_q: Tensor, memory: Tensor
    ) -> tuple[Tensor, Tensor]:
        # System stream
        normed = self.sys_self_attn_norm(sys_q)
        sys_q = sys_q + self.sys_self_attn(normed, normed, normed)[0]

        query = self.sys_cross_attn_norm(sys_q)
        sys_q = sys_q + self.sys_cross_attn(query, memory, memory)[0]
        sys_q = sys_q + self.sys_ffn(self.sys_ffn_norm(sys_q))

        # Stave stream — primary and independent of systems (no group attention);
        # staves never inherit from the system queries.
        normed = self.stave_self_attn_norm(stave_q)
        stave_q = stave_q + self.stave_self_attn(normed, normed, normed)[0]

        query = self.stave_cross_attn_norm(stave_q)
        stave_q = stave_q + self.stave_cross_attn(query, memory, memory)[0]
        stave_q = stave_q + self.stave_ffn(self.stave_ffn_norm(stave_q))

        return sys_q, stave_q


class StafferDecoder(nn.Module):
    def __init__(self, config: StafferConfig):
        super().__init__()
        self.sys_queries = nn.Embedding(config.num_system_queries, config.embed_dim)
        self.stave_queries = nn.Embedding(config.num_stave_queries, config.embed_dim)
        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.num_decoder_layers)]
        )

    def forward(self, memory: Tensor) -> tuple[Tensor, Tensor]:
        B = memory.shape[0]
        sys_q = self.sys_queries.weight.unsqueeze(0).expand(B, -1, -1)
        stave_q = self.stave_queries.weight.unsqueeze(0).expand(B, -1, -1)
        for layer in self.layers:
            sys_q, stave_q = layer(sys_q, stave_q, memory)
        # (B, N, D), (B, M, D) — only the final layer feeds the heads.
        return sys_q, stave_q


class PredictionHeads(nn.Module):
    def __init__(self, config: StafferConfig):
        super().__init__()
        D = config.embed_dim

        # System stream predicts the horizontal extent (left, right) only — the one
        # genuinely system-level quantity. Vertical extent is derived from the
        # system's staves downstream, so there is no system vertical head/anchor.
        self.sys_lr_head = nn.Sequential(
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, 2),  # left, right
        )
        self.sys_obj_head = nn.Linear(D, 1)

        # Stave stream is primary: predict (top, bottom) as a residual to a FIXED
        # per-slot vertical anchor grid. The anchors are frozen (requires_grad=False)
        # so each slot's y position is enforced; GT staves are routed to the
        # nearest-anchor query at train time (see assign_staves), making the
        # slot->stave mapping free and non-contiguous rather than positional.
        self.stave_box_head = nn.Sequential(
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, 2),  # predict top, bottom — x inherited from parent system
        )
        self.stave_obj_head = nn.Linear(D, 1)
        # Slot i is centred at (i + 0.5) / M with a small initial height, so a query
        # starts as a thin box at its band rather than a zero-height line (top ==
        # bottom would make the derived system hull degenerate at init). The box head
        # learns a logit-space residual from these frozen anchors.
        M = config.num_stave_queries
        centers = (torch.arange(M) + 0.5) / M
        half_height = 0.25 / M  # a quarter of the per-slot pitch
        self.stave_top_ref = nn.Parameter(
            torch.logit((centers - half_height).clamp(1e-4, 1 - 1e-4)),
            requires_grad=False,
        )
        self.stave_bottom_ref = nn.Parameter(
            torch.logit((centers + half_height).clamp(1e-4, 1 - 1e-4)),
            requires_grad=False,
        )

        # Boundary flag: 1 when a stave starts a new system. cumsum recovers the
        # stave->system grouping (contiguous by construction).
        self.boundary_head = nn.Linear(D, 1)

    def anchor_centers(self) -> Tensor:
        """Fixed per-slot vertical centers (M,) used to route GT staves to queries."""
        return (self.stave_top_ref.sigmoid() + self.stave_bottom_ref.sigmoid()) / 2

    def forward(
        self,
        sys_feats: Tensor,  # (B, N, D)
        stave_feats: Tensor,  # (B, M, D)
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        # Stave vertical edges: predict a residual to the frozen per-slot anchor
        # grid. The anchor fixes the slot's y band; the residual places the box.
        residual = self.stave_box_head(stave_feats)  # (B, M, 2)
        top = (residual[..., 0] + self.stave_top_ref).sigmoid()
        bot = (residual[..., 1] + self.stave_bottom_ref).sigmoid()
        stave_tb = torch.stack([top, bot], dim=-1)  # (B, M, 2)

        # System horizontal extent, unanchored.
        sys_lr = self.sys_lr_head(sys_feats).sigmoid()  # (B, N, 2)

        sys_logits = self.sys_obj_head(sys_feats)  # (B, N, 1)
        stave_logits = self.stave_obj_head(stave_feats)  # (B, M, 1)
        boundary_logits = self.boundary_head(stave_feats)  # (B, M, 1)
        return (
            stave_tb,
            stave_logits,
            boundary_logits,
            sys_lr,
            sys_logits,
        )


class StafferModel(nn.Module):
    def __init__(self, config: StafferConfig):
        super().__init__()
        self.config = config
        self.backbone = ViT(config)
        self.decoder = StafferDecoder(config)
        self.heads = PredictionHeads(config)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        memory = self.backbone(x)  # (B, P, D)
        sys_feats, stave_feats = self.decoder(memory)  # (B, N, D), (B, M, D)
        return self.heads(sys_feats, stave_feats)


# vscode - End of file.
