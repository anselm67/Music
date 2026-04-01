"""A model to computes the layout of a page of sheet music.
"""
from dataclasses import asdict, dataclass, field

import torch.nn.functional as F
from torch import Tensor, nn, randn
from torchvision.transforms import InterpolationMode


@dataclass
class Config:
    id_name: str = "default"
    image_shape: tuple[int, int] = field(init=False)

    # Maximums as obtained with the "stats" command.
    max_width: int = 1024
    max_height: int = 1449

    in_channels: int = 1
    divider: float = 1.5
    embed_dim: int = 256                # Also known as D
    mlp_dim: int = 1024

    num_heads: int = 8                  # Also known as H
    patch_size: int = 16
    dropout: float = 0.1
    num_layers = 4

    # Decoder config.
    # These numbers are from running the following commands:
    # pdmx query -o Stafff16.csv 'index==index' --score 'pages.*.staff_count < 16'
    # pdmx --csv Staff16.csv stats

    num_system_queries: int = 16        # Also known as N
    num_stave_queries: int = 16         # Also knwon as M
    num_decoder_layers: int = 4

    interpolation: InterpolationMode = InterpolationMode.NEAREST
    antialias: bool = False

    # Training config.
    batch_size: int = 16
    train_len: int = 6250 * batch_size
    valid_len: int = 100 * batch_size
    max_steps: int = 4 * (train_len // batch_size)
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 1000

    def scale_to_patch(self, value: int) -> int:
        ret = value // self.divider
        return int(round(ret / self.patch_size) * self.patch_size)

    def __post_init__(self):
        self.image_shape = (
            self.scale_to_patch(self.max_height),
            self.scale_to_patch(self.max_width),
        )
        assert self.patch_size ** 2 == self.embed_dim

    def asdict(self):
        obj = asdict(self)
        obj.pop("image_shape")
        return obj


class PatchEmbedding(nn.Module):

    config: Config

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.num_patch = (
            config.image_shape[0] // config.patch_size,
            config.image_shape[1] // config.patch_size)
        self.proj = nn.Conv2d(config.in_channels, config.embed_dim,
                              kernel_size=config.patch_size, stride=config.patch_size)
        self.pos_embed = nn.Parameter(0.02 * randn(
            self.num_patch[0] * self.num_patch[1], config.embed_dim))

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x += self.pos_embed
        return self.dropout(x)


class TransformerBlock(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.qkv = nn.Linear(config.embed_dim, 3 * config.embed_dim)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, config.mlp_dim),
            nn.GELU(),
            nn.Linear(config.mlp_dim, config.embed_dim),
            nn.Dropout(config.dropout)
        )
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.norm2 = nn.LayerNorm(config.embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        B, N, D = x.shape
        x_norm = self.norm1(x)
        qkv = self.qkv(x_norm).reshape(
            B, N, 3, self.config.num_heads, D // self.config.num_heads)
        q, k, v = qkv.unbind(2)  # each (B, N, num_heads, head_dim)
        q, k, v = q.transpose(1, 2), k.transpose(
            1, 2), v.transpose(1, 2)  # (B, num_heads, N, head_dim)
        x_attn = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.config.dropout if self.training else 0.0)
        x_attn = x_attn.transpose(1, 2).reshape(B, N, D)
        x = x + self.proj(x_attn)
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):

    config: Config

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbedding(config)
        self.blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.num_layers)]
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x)
        x = self.blocks(x)

        return x


class DecoderLayer(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        D = config.embed_dim
        H = config.num_heads

        # System stream
        self.sys_self_attn_norm = nn.LayerNorm(D)
        self.sys_self_attn = nn.MultiheadAttention(
            D, H, dropout=config.dropout, batch_first=True)

        self.sys_cross_attn_norm = nn.LayerNorm(D)
        self.sys_cross_attn = nn.MultiheadAttention(
            D, H, dropout=config.dropout, batch_first=True)

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
            D, H, dropout=config.dropout, batch_first=True)

        self.stave_cross_attn_norm = nn.LayerNorm(D)
        self.stave_cross_attn = nn.MultiheadAttention(
            D, H, dropout=config.dropout, batch_first=True)

        self.stave_group_norm = nn.LayerNorm(D)
        self.stave_group_attn = nn.MultiheadAttention(
            D, H, dropout=config.dropout, batch_first=True)

        self.stave_ffn_norm = nn.LayerNorm(D)
        self.stave_ffn = nn.Sequential(
            nn.Linear(D, config.mlp_dim),
            nn.GELU(),
            nn.Linear(config.mlp_dim, D),
            nn.Dropout(config.dropout),
        )

    def forward(self, sys_q: Tensor, stave_q: Tensor, memory: Tensor) -> tuple[Tensor, Tensor]:
        # System stream
        normed = self.sys_self_attn_norm(sys_q)
        sys_q = sys_q + self.sys_self_attn(normed, normed, normed)[0]

        query = self.sys_cross_attn_norm(sys_q)
        sys_q = sys_q + self.sys_cross_attn(query, memory, memory)[0]
        sys_q = sys_q + self.sys_ffn(self.sys_ffn_norm(sys_q))

        # Stave stream
        normed = self.stave_self_attn_norm(stave_q)
        stave_q = stave_q + self.stave_self_attn(normed, normed, normed)[0]

        query = self.stave_cross_attn_norm(stave_q)
        stave_q = stave_q + self.stave_cross_attn(query, memory, memory)[0]

        normed = self.stave_group_norm(stave_q)
        stave_q = stave_q + self.stave_group_attn(normed, sys_q, sys_q)[0]
        stave_q = stave_q + self.stave_ffn(self.stave_ffn_norm(stave_q))

        return sys_q, stave_q


class HierarchicalDecoder(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        self.sys_queries = nn.Embedding(
            config.num_system_queries, config.embed_dim)
        self.stave_queries = nn.Embedding(
            config.num_stave_queries, config.embed_dim)
        self.layers = nn.ModuleList([DecoderLayer(config)
                                    for _ in range(config.num_decoder_layers)])

    def forward(self, memory: Tensor) -> tuple[Tensor, Tensor]:
        B = memory.shape[0]
        sys_q = self.sys_queries.weight.unsqueeze(0).expand(B, -1, -1)
        stave_q = self.stave_queries.weight.unsqueeze(0).expand(B, -1, -1)
        for layer in self.layers:
            sys_q, stave_q = layer(sys_q, stave_q, memory)
        return sys_q, stave_q  # (B, N, D), (B, M, D)


class PredictionHeads(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        D = config.embed_dim

        self.sys_box_head = nn.Sequential(
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, 4),
        )
        self.sys_obj_head = nn.Linear(D, 1)

        self.stave_box_head = nn.Sequential(
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, 4),
        )
        self.stave_obj_head = nn.Linear(D, 1)
        self.assign_head = nn.Linear(D, config.num_system_queries)

    def forward(self, sys_feats: Tensor, stave_feats: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        # (B, N, 4) normalised 0-1
        sys_boxes = self.sys_box_head(sys_feats).sigmoid()
        sys_logits = self.sys_obj_head(sys_feats)               # (B, N, 1)
        stave_boxes = self.stave_box_head(stave_feats).sigmoid()  # (B, M, 4)
        stave_logits = self.stave_obj_head(stave_feats)           # (B, M, 1)
        assign_logits = self.assign_head(stave_feats)             # (B, M, N)
        return sys_boxes, sys_logits, stave_boxes, stave_logits, assign_logits


class HierarchicalDETR(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        self.backbone = ViT(config)
        self.decoder = HierarchicalDecoder(config)
        self.heads = PredictionHeads(config)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        memory = self.backbone(x)           # (B, num_patches, D)
        sys_feats, stave_feats = self.decoder(memory)  # (B, N, D), (B, M, D)
        return self.heads(sys_feats, stave_feats)
        # returns: sys_boxes, sys_logits, stave_boxes, stave_logits, assign_logits
