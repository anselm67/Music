"""Torch Dataset for training models against a Source (PDMX or KernSheet)."""

import logging
import math
from collections import Counter
from typing import cast

import torch
from torch import Tensor
from torch.utils.data import Dataset, Subset, WeightedRandomSampler
from torchvision.transforms import v2
from tqdm import tqdm

from sheetmusic import (
    LetterboxResize,
    PerImageNormalize,
    ScanAugment,
    Source,
    letterbox_scale,
)

from .staffer_model import StafferConfig


class StafferDataset(Dataset[tuple[Tensor, Tensor, Tensor, Tensor]]):
    source: Source
    # score id, page number, part_count, max_sys_bottom
    # The last two items are used when use_sampler is enabled.
    items: list[tuple[str, int, int, float]]

    transform: v2.Transform

    def __init__(self, config: StafferConfig, source: Source, count: int = -1):
        self.config = config
        self.source = source
        # Clean transform (no augment); the datamodule calls enable_augment on the
        # train view only, leaving validation on un-augmented pages.
        self.transform = self._build_transform(0.0)
        # Build flat list of (score id, page_number) pairs
        logging.info("Initializing StafferDataset...")
        self.items = []
        for score in tqdm(source.scores(), desc="Loading dataset"):
            part_count = max(score.staff_count, 1) // max(score.system_count, 1)
            for page in source.pages(score.id):
                if not page.systems:  # skip blank/cover pages (no GT to supervise)
                    continue
                raw = max(
                    (s.box.bottom / page.image_height for s in page.systems),
                    default=0.0,
                )
                if raw > 1.0:
                    logging.warning(
                        "%s page %d: max_sys_bottom=%.3f > 1.0, clamping",
                        score.id,
                        page.page_number,
                        raw,
                    )
                max_sys_bottom = min(raw, 1.0)
                self.items.append(
                    (
                        score.id,
                        page.page_number,
                        part_count,
                        max_sys_bottom,
                    )
                )
            if count >= 0 and len(self.items) >= count:
                self.items = self.items[:count]
                break
        logging.info(f"\tStafferDataset: {len(self.items)} samples.")

    def _build_transform(self, augment: float) -> v2.Transform:
        """Image transform; ScanAugment (prob ``augment``, 0 = off) runs on the
        float [0,1] page before per-image normalisation."""
        return v2.Compose(
            [
                v2.Grayscale(),
                LetterboxResize(
                    self.config.image_shape,
                    interpolation=self.config.interpolation,
                    antialias=self.config.antialias,
                    fill=255,
                ),
                v2.ToDtype(torch.float, scale=True),
                ScanAugment(augment),
                PerImageNormalize(),
            ]
        )

    def enable_augment(self, prob: float) -> None:
        """Rebuild the transform with scan augmentation at probability ``prob``.

        Called by the datamodule on the (shallow-copied) train view only, so
        validation keeps the clean transform built in ``__init__``.
        """
        self.transform = self._build_transform(prob)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        while True:
            score_id, page_number, _, _ = self.items[idx]
            # Attempts to decode this image, or next one when that fails.
            try:
                image = self.transform(self.source.image(score_id, page_number))
            except Exception as e:
                logging.error(f"{score_id}: {e}")
                idx += 1
                continue

            # Converts the Score to expected ground truth tensors.
            is_ok = True
            score = self.source.score(score_id)
            # items() enrols only source.pages() (validated, for KernSheet) by their
            # page_number; the lookup below relies on pages being dense 1-based.
            page = score.pages[page_number - 1]
            assert page.page_number == page_number

            sys_boxes = torch.zeros(self.config.num_system_queries, 4)
            staff_boxes = torch.zeros(self.config.num_stave_queries, 4)
            assigns = torch.full((self.config.num_stave_queries,), -1, dtype=torch.long)
            # Normalise boxes into the letterboxed canvas: scale the original-pixel
            # coords by the aspect-preserving factor, then by the canvas dims (content
            # is anchored at the top-left, matching LetterboxResize's bottom/right pad).
            W, H = page.image_width, page.image_height
            target_h, target_w = self.config.image_shape
            scale = letterbox_scale(H, W, target_h, target_w)
            sx, sy = scale / target_w, scale / target_h
            staff_idx = 0
            for sys_idx, system in enumerate(page.systems):
                if sys_idx >= self.config.num_system_queries:
                    is_ok = False
                    break
                sys_boxes[sys_idx] = torch.tensor(
                    [
                        system.box.left * sx,
                        system.box.top * sy,
                        system.box.right * sx,
                        system.box.bottom * sy,
                    ]
                )
                for staff in system.staves:
                    # ltrb; only cols [1,3] (top, bottom) are used in the loss
                    staff_boxes[staff_idx] = torch.tensor(
                        [
                            staff.box.left * sx,
                            staff.box.top * sy,
                            staff.box.right * sx,
                            staff.box.bottom * sy,
                        ]
                    )
                    assigns[staff_idx] = sys_idx
                    staff_idx += 1

            if is_ok:
                return image, sys_boxes, staff_boxes, assigns
            idx += 1


def build_sampler(
    ds: Subset[tuple[Tensor, Tensor, Tensor, Tensor]],
    bottom_bias: float = 3.0,
) -> WeightedRandomSampler:
    logging.info("Computing sample weights...")
    part_counts: list[int] = []
    max_sys_bottoms: list[float] = []
    part_histo: Counter[int] = Counter()
    dataset = cast(StafferDataset, ds.dataset)
    for i in ds.indices:
        _, _, part_count, max_sys_bottom = dataset.items[i]
        part_counts.append(part_count)
        max_sys_bottoms.append(max_sys_bottom)
        part_histo[part_count] += 1

    sqrt_inv: dict[int, float] = {n: 1.0 / math.sqrt(c) for n, c in part_histo.items()}

    weights = [
        sqrt_inv[count] * (1.0 + bottom_bias * cy)
        for count, cy in zip(part_counts, max_sys_bottoms)
    ]

    return WeightedRandomSampler(
        weights=weights, num_samples=len(ds.indices), replacement=True
    )
