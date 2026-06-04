"""ScorerModel: end-to-end OMR — Staffer + Noter joined by a differentiable crop.

The two standalone models are kept intact (see docs/architecture.html). Staffer
detects stave boxes on the full page; a differentiable ``grid_sample`` crop samples
each detected stave straight from the page image into the noter's input shape; the
noter transcribes each crop. Because the crop is differentiable in the box coordinates,
the transcription loss flows back into the detector — that gradient path is the only
thing the merge adds over running the two models back-to-back. (``roi_align`` is NOT
used: torchvision's implementation does not backprop to the box coordinates.)
"""

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from utils import current_commit

from noter import NoterConfig, NoterModel, Vocab
from staffer import StafferConfig, StafferModel

# Per-model grayscale normalisation (mean, std) — hardcoded in each dataset's
# v2.Normalize. Staffer normalises the page; the noter was trained on crops
# normalised with its own stats, so crops are recoloured staffer→noter space
# before entering the noter branch (a single affine of the same [0,1] image).
STAFFER_NORM = (0.9563435316085815, 0.16557540870879858)
NOTER_NORM = (0.9482423663139343, 0.17525607175008864)


@dataclass
class ScorerConfig:
    id_name: str = "default"
    git_hash: str = current_commit()

    # The two branches keep their own configs — d_model etc. are independent
    # (the bridge passes an image crop, not features).
    staffer: StafferConfig = field(default_factory=StafferConfig)
    noter: NoterConfig = field(default_factory=NoterConfig)

    # Joint loss weights.
    lambda_det: float = 1.0
    lambda_tr: float = 1.0

    # Joint training.
    batch_size: int = 8
    train_len: int = -1
    valid_len: int = -1
    max_steps: int = field(init=False)
    lr: float = 1e-4
    weight_decay: float = 1e-2
    warmup_steps: int = 500
    # Freeze the staffer (ViT + decoder + heads) for the first N steps so the
    # noter branch adapts to predicted crops before the detector starts moving.
    freeze_staffer_steps: int = 500

    def __post_init__(self) -> None:
        if self.train_len == -1:
            self.train_len = 12500 * self.batch_size
        if self.valid_len == -1:
            self.valid_len = 100 * self.batch_size
        self.max_steps = 4 * (self.train_len // self.batch_size)

    def use_vocab(self, vocab: Vocab) -> None:
        """Wire the noter vocab — required before building the model."""
        self.noter.use_vocab(vocab)

    def asdict(self) -> dict[str, object]:
        obj = asdict(self)
        obj.pop("max_steps")
        # Sub-config dicts drop their own derived fields the same way.
        obj["staffer"] = self.staffer.asdict()
        obj["noter"] = self.noter.asdict()
        return obj


def build_stave_boxes(
    stave_tb: Tensor,  # (B, M, 2) — normalised [top, bottom]
    sys_lr: Tensor,  # (B, N, 2) — normalised [left, right]
    sel_queries: list[Tensor],  # per page: (G,) query slots to crop, in target order
    sys_ids: list[Tensor],  # per page: (G,) system index owning each selected stave
    image_hw: Sequence[int],  # (H, W) of the page in pixels
) -> Tensor:
    """Build the ``(K, 5)`` raw stave-box tensor for the selected staves.

    Each row is ``[batch_idx, left, top, right, bot]`` in image-pixel coords — the
    stave's own (top, bottom) and the owning system's (left, right). The noter crop
    *window* (context, padding, masking) is derived from these in ``ScorerModel.crop``;
    this function only places the boxes. ``K = sum(G_i)``, ordered page-by-page then in
    each page's target order — so row ``k`` lines up with target row ``k``.
    Differentiable in the box edges.
    """
    H, W = image_hw
    boxes: list[Tensor] = []
    for b, (sel, owners) in enumerate(zip(sel_queries, sys_ids)):
        if sel.numel() == 0:
            continue
        top = stave_tb[b, sel, 0] * H
        bot = stave_tb[b, sel, 1] * H
        lr = sys_lr[b, owners]  # (G, 2)
        left = lr[:, 0] * W
        right = lr[:, 1] * W
        # batch_idx rides in the float box tensor (coords must stay float for
        # grad); crop() casts column 0 back to long before indexing.
        batch_idx = torch.full_like(left, float(b))
        boxes.append(torch.stack([batch_idx, left, top, right, bot], dim=-1))
    if not boxes:
        return stave_tb.new_zeros((0, 5))
    return torch.cat(boxes, dim=0)


class ScorerModel(nn.Module):
    """Intact ``StafferModel`` + ``NoterModel`` joined by a differentiable crop."""

    def __init__(self, config: ScorerConfig) -> None:
        super().__init__()
        self.config = config
        self.staffer = StafferModel(config.staffer)
        self.noter = NoterModel(config.noter)
        # crop → noter input shape, recoloured into noter normalisation space.
        std_s, std_n = STAFFER_NORM[1], NOTER_NORM[1]
        mean_s, mean_n = STAFFER_NORM[0], NOTER_NORM[0]
        self._renorm_scale = std_s / std_n
        self._renorm_shift = (mean_s - mean_n) / std_n

    def detect(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Run the staffer branch: page image → box predictions."""
        return self.staffer(image)

    def crop(self, image: Tensor, boxes: Tensor) -> tuple[Tensor, Tensor]:
        """Crop the selected staves from the page → ``(K, 1, 64, 768)`` crops.

        Reproduces the standalone noter crop window, differentiably. The noter takes a
        **1:1 (un-stretched)** window: a 64px-tall slice starting one stave-height above
        the stave top, left-aligned at the system's left edge, with the real stave width
        reported as ``source_widths`` so the right of the canvas is padded and masked
        (``make_src_padding_mask``). This matters because the last system of a score
        usually ends mid-page — stretching it to fill 768px would distort note spacing
        (timing). So the window size is fixed at ``input_shape`` *pixels* (1:1) and only
        its position depends on the box → gradient flows to the stave top/bottom and the
        system left (position), not to a spurious scale.

        Built with ``affine_grid`` + ``grid_sample`` (not ``torchvision.ops.roi_align``,
        which does not backprop to box coordinates). Crops are recoloured into the
        noter's normalisation space. NB: exact for staves where ``3·h ≥ 64px`` (the
        System2 norm); very short staves get page context instead of white vertical pad
        — a minor deviation to revisit if it bites.
        """
        out_h, out_w = self.config.noter.input_shape
        _, C, H, W = image.shape
        idx = boxes[:, 0].long()
        src = image[idx]  # (K, C, H, W) — gather each box's source page
        left, top, right, bot = boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
        h = bot - top
        # 1:1 window of (out_h, out_w) source pixels, anchored top-left at
        # (top - h, left) — the noter's (box.top - box.height, box.left) origin.
        cy = (top - h) + out_h / 2  # window centre (px)
        cx = left + out_w / 2
        theta = boxes.new_zeros((boxes.shape[0], 2, 3))
        theta[:, 0, 0] = out_w / W  # 1:1 horizontal (no stretch)
        theta[:, 0, 2] = cx / W * 2 - 1
        theta[:, 1, 1] = out_h / H  # 1:1 vertical
        theta[:, 1, 2] = cy / H * 2 - 1
        grid = F.affine_grid(
            theta, [boxes.shape[0], C, out_h, out_w], align_corners=False
        )
        crops = F.grid_sample(
            src, grid, align_corners=False, padding_mode="border"
        )  # (K, C, out_h, out_w)
        crops = crops * self._renorm_scale + self._renorm_shift
        # Real stave width in px → masks the right padding, exactly like the noter.
        widths = (right - left).clamp(min=1, max=out_w).long()
        return crops, widths

    def forward(
        self,
        image: Tensor,  # (B, 1, H, W) normalised page (staffer space)
        sel_queries: list[Tensor],  # per page: query slots to transcribe (target order)
        sys_ids: list[Tensor],  # per page: owning system index per selected stave
    ) -> tuple[tuple[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor, Tensor]:
        """Detect, crop, and encode the selected staves.

        Returns ``(staffer_outputs, memory, src_pad_mask)``; the caller runs the noter
        decoder (it owns the causal mask and the teacher-forced / autoregressive split).
        """
        staffer_out = self.detect(image)
        stave_tb, _stave_logits, _boundary, sys_lr, _sys_logits = staffer_out
        hw = (int(image.shape[-2]), int(image.shape[-1]))
        boxes = build_stave_boxes(stave_tb, sys_lr, sel_queries, sys_ids, hw)
        crops, widths = self.crop(image, boxes)
        memory, src_pad_mask = self.noter.encode(crops, widths)
        return staffer_out, memory, src_pad_mask
