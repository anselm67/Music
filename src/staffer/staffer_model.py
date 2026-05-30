"""StafferModel: hierarchical layout detector for pages of sheet music."""

from dataclasses import asdict, dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, arange, log, nn, randn
from torchvision.transforms import InterpolationMode

from utils import current_commit


@dataclass
class StafferConfig:
    id_name: str = "default"
    git_hash: str = current_commit()
    image_shape: tuple[int, int] = field(init=False)

    # Maximums as obtained with the "stats" command.
    max_width: int = 1024
    max_height: int = 1449

    in_channels: int = 1
    divider: float = 1.5
    embed_dim: int = 128  # Also known as D
    mlp_dim: int = 512

    num_heads: int = 4  # Also known as H
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
        self.image_shape = (
            self.scale_to_patch(self.max_height),
            self.scale_to_patch(self.max_width),
        )
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

        self.stave_group_norm = nn.LayerNorm(D)
        self.stave_group_attn = nn.MultiheadAttention(
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

        # Stave stream
        normed = self.stave_self_attn_norm(stave_q)
        stave_q = stave_q + self.stave_self_attn(normed, normed, normed)[0]

        query = self.stave_cross_attn_norm(stave_q)
        stave_q = stave_q + self.stave_cross_attn(query, memory, memory)[0]

        normed = self.stave_group_norm(stave_q)
        stave_q = stave_q + self.stave_group_attn(normed, sys_q, sys_q)[0]
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
        sys_layers, stave_layers = [], []
        for layer in self.layers:
            sys_q, stave_q = layer(sys_q, stave_q, memory)
            sys_layers.append(sys_q)
            stave_layers.append(stave_q)
        # (L, B, N, D), (L, B, M, D)
        return torch.stack(sys_layers), torch.stack(stave_layers)


def _even_anchor_logits(num_queries: int) -> Tensor:
    """Logit-space references for evenly-spaced vertical anchors.

    Query i anchors at (i + 0.5) / num_queries in [0, 1], returned as the
    inverse-sigmoid (logit) so it can be added to a head delta before sigmoid.
    """
    anchors = (arange(num_queries) + 0.5) / num_queries
    return log(anchors) - log(1.0 - anchors)


class PredictionHeads(nn.Module):
    def __init__(self, config: StafferConfig):
        super().__init__()
        D = config.embed_dim

        self.sys_box_head = nn.Sequential(
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, 4),
        )
        self.sys_obj_head = nn.Linear(D, 1)
        # Learnable logit-space anchors for the vertical edges (top, bottom),
        # initialised evenly spaced down the page. left/right are unanchored.
        sys_anchor = _even_anchor_logits(config.num_system_queries)
        self.sys_top_ref = nn.Parameter(sys_anchor.clone())
        self.sys_bottom_ref = nn.Parameter(sys_anchor.clone())

        self.stave_box_head = nn.Sequential(
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, 2),  # predict top, bottom — x inherited from parent system
        )
        self.stave_obj_head = nn.Linear(D, 1)
        stave_anchor = _even_anchor_logits(config.num_stave_queries)
        self.stave_top_ref = nn.Parameter(stave_anchor.clone())
        self.stave_bottom_ref = nn.Parameter(stave_anchor.clone())

        self.assign_head = nn.Linear(D, config.num_system_queries)

    def forward(
        self,
        sys_layers: Tensor,  # (L, B, N, D)
        stave_layers: Tensor,  # (L, B, M, D)
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        # Iterative box refinement: each decoder layer predicts a logit-space
        # residual to the running vertical reference; the detached sigmoid output
        # (re-linearised via logit, clamped) becomes the next layer's reference.
        # Detaching means every layer must emit a valid box (trained via deep
        # supervision), making the anchor content/count-conditional rather than a
        # static per-slot constant. left/right stay unanchored, predicted per layer.
        sys_ref_top: Tensor = self.sys_top_ref
        sys_ref_bot: Tensor = self.sys_bottom_ref
        sys_boxes_per_layer = []
        for sys_feats in sys_layers:
            d = self.sys_box_head(sys_feats)  # (B, N, 4) ltrb deltas
            top = (d[..., 1] + sys_ref_top).sigmoid()
            bot = (d[..., 3] + sys_ref_bot).sigmoid()
            sys_boxes_per_layer.append(
                torch.stack(
                    [d[..., 0].sigmoid(), top, d[..., 2].sigmoid(), bot], dim=-1
                )
            )
            sys_ref_top = torch.logit(top.detach(), eps=1e-6)
            sys_ref_bot = torch.logit(bot.detach(), eps=1e-6)
        sys_boxes_all = torch.stack(sys_boxes_per_layer)  # (L, B, N, 4)

        stave_ref_top: Tensor = self.stave_top_ref
        stave_ref_bot: Tensor = self.stave_bottom_ref
        stave_tb_per_layer = []
        for stave_feats in stave_layers:
            d = self.stave_box_head(stave_feats)  # (B, M, 2)
            top = (d[..., 0] + stave_ref_top).sigmoid()
            bot = (d[..., 1] + stave_ref_bot).sigmoid()
            stave_tb_per_layer.append(torch.stack([top, bot], dim=-1))
            stave_ref_top = torch.logit(top.detach(), eps=1e-6)
            stave_ref_bot = torch.logit(bot.detach(), eps=1e-6)
        stave_tb_all = torch.stack(stave_tb_per_layer)  # (L, B, M, 2)

        sys_logits = self.sys_obj_head(sys_layers[-1])  # (B, N, 1)
        stave_logits = self.stave_obj_head(stave_layers[-1])  # (B, M, 1)
        assign_logits = self.assign_head(stave_layers[-1])  # (B, M, N)
        return (
            sys_boxes_all[-1],
            sys_logits,
            stave_tb_all[-1],
            stave_logits,
            assign_logits,
            sys_boxes_all,
            stave_tb_all,
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
        sys_layers, stave_layers = self.decoder(memory)  # (L, B, N, D), (L, B, M, D)
        (
            sys_boxes,
            sys_logits,
            stave_tb,
            stave_logits,
            assign_logits,
            aux_sys_boxes,
            aux_stave_tb,
        ) = self.heads(sys_layers, stave_layers)
        # Per-layer boxes for deep supervision; read by the module during training.
        self.aux_sys_boxes = aux_sys_boxes
        self.aux_stave_tb = aux_stave_tb
        return sys_boxes, sys_logits, stave_tb, stave_logits, assign_logits


# vscode - End of file.
