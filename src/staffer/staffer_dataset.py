"""Torch Dataset for training models against the PDMX dataset."""

import json
import logging
import math
from collections import Counter
from pathlib import Path
from typing import cast

import torch
from torch import Tensor
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision.io import decode_image
from torchvision.transforms import v2
from tqdm import tqdm

from pdmx import PDMX, Score
from .staffer_model import StafferConfig


class StafferDataset(Dataset):
    pdmx: PDMX
    # layout path, png path, page number, part_count, is_last_page
    # The last two items are used when use_sampler is enabled.
    items: list[tuple[Path, Path, int, int, bool]]

    transform: v2.Transform

    def __init__(self, config: StafferConfig, pdmx: PDMX, count: int = -1):
        self.config = config
        self.pdmx = pdmx
        self.transform = v2.Compose(
            [
                v2.Grayscale(),
                v2.Resize(
                    config.image_shape,
                    interpolation=config.interpolation,
                    antialias=config.antialias,
                ),
                v2.ToDtype(torch.float, scale=True),
                # Values from running: staffer stats
                v2.Normalize(mean=[0.9563435316085815], std=[0.16557540870879858]),
            ]
        )
        # Build flat list of (mxl_path, page_number) pairs
        logging.info("Initializing StafferDataset...")
        self.items = []
        for _, row in tqdm(
            pdmx.df.iterrows(), total=len(pdmx.df), desc="Loading dataset"
        ):
            mxl_file = pdmx.home / row["mxl"]
            layout_file = pdmx.get_path(mxl_file, "layout")
            score = Score.from_json(json.loads(layout_file.read_text()))
            part_count = score.staff_count // score.system_count
            for page in score.pages:
                if score.page_count > 1:
                    png_file = pdmx.get_page_path(mxl_file, "png", page.page_number)
                else:
                    png_file = pdmx.get_path(mxl_file, "png")
                self.items.append(
                    (
                        layout_file,
                        png_file,
                        page.page_number,
                        part_count,
                        (page.page_number == score.page_count - 1),
                    )
                )
            if count >= 0 and len(self.items) >= count:
                self.items = self.items[:count]
                break
        logging.info(f"\tStafferDataset: {len(self.items)} samples.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        while True:
            layout_path, png_path, page_number, _, _ = self.items[idx]
            # Attempts to decode this image, or next one when that fails.
            try:
                image = decode_image(png_path.as_posix())
                image = self.transform(image)
            except Exception as e:
                mxl_path = self.pdmx.get_path(layout_path, "mxl")
                logging.error(f"{mxl_path}: {e}")
                idx += 1
                continue

            # Converts the Score to expected ground truth tensors.
            is_ok = True
            score = Score.from_json(json.loads(layout_path.read_text()))
            page = score.pages[page_number - 1]

            sys_boxes = torch.zeros(self.config.num_system_queries, 4)
            staff_boxes = torch.zeros(self.config.num_stave_queries, 4)
            assigns = torch.full((self.config.num_stave_queries,), -1, dtype=torch.long)
            staff_idx = 0
            for sys_idx, system in enumerate(page.systems):
                sys_boxes[sys_idx] = torch.tensor(
                    system.box.to_cxcywh(page.image_width, page.image_height)
                )
                for staff in system.staves:
                    staff_boxes[staff_idx] = torch.tensor(
                        staff.box.to_cxcywh(page.image_width, page.image_height)
                    )
                    assigns[staff_idx] = sys_idx
                    staff_idx += 1

            if is_ok:
                if self.config.vflip > 0 and torch.rand(1).item() < self.config.vflip:
                    image, sys_boxes, staff_boxes, assigns = _vflip_gt(
                        image, sys_boxes, staff_boxes, assigns
                    )
                return image, sys_boxes, staff_boxes, assigns


def _vflip_gt(
    image: Tensor,
    sys_boxes: Tensor,
    staff_boxes: Tensor,
    assigns: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Vertical (top↔bottom) flip of image and GT tensors.

    After flipping the image, the system that was at the bottom is now at the
    top, so system order is reversed and stave assignments are remapped
    accordingly.  All boxes remain in cxcywh format; only cy changes.
    """
    image = torch.flip(image, dims=[-2])

    num_staves = int((assigns != -1).sum().item())
    if num_staves == 0:
        return image, sys_boxes, staff_boxes, assigns

    num_sys = int(assigns[:num_staves].max().item()) + 1

    sys_boxes[:num_sys, 1] = 1.0 - sys_boxes[:num_sys, 1]  # flip cy
    sys_boxes[:num_sys] = sys_boxes[:num_sys].flip(0)        # reverse order

    staff_boxes[:num_staves, 1] = 1.0 - staff_boxes[:num_staves, 1]  # flip cy

    assigns[:num_staves] = (num_sys - 1) - assigns[:num_staves]  # remap sys indices

    # Re-sort staves top-to-bottom by their new cy
    sort_idx = staff_boxes[:num_staves, 1].argsort()
    staff_boxes[:num_staves] = staff_boxes[:num_staves][sort_idx]
    assigns[:num_staves] = assigns[:num_staves][sort_idx]

    return image, sys_boxes, staff_boxes, assigns


def build_sampler(ds: Dataset, last_page_weight: float = 1.5) -> WeightedRandomSampler:
    logging.info("Computing sample weights...")
    part_counts: list[int] = []
    is_last_pages: list[bool] = []
    part_histo: Counter[int] = Counter()
    dataset = cast(StafferDataset, ds.dataset)  # type: ignore
    for i in ds.indices:  # type: ignore
        _, _, _, part_count, is_last_page = dataset.items[i]
        part_counts.append(part_count)
        is_last_pages.append(is_last_page)
        part_histo[part_count] += 1

    sqrt_inv: dict[int, float] = {n: 1.0 / math.sqrt(c) for n, c in part_histo.items()}

    weights = [
        sqrt_inv[count] * (last_page_weight if last_page else 1.0)
        for count, last_page in zip(part_counts, is_last_pages)
    ]

    return WeightedRandomSampler(
        weights=weights, num_samples=dataset.config.train_len, replacement=True
    )
