"""Torch Dataset for training models against the PDMX dataset."""

import json
import logging
import math
from collections import Counter
from pathlib import Path
from typing import cast

import torch
from torch import Tensor
from torch.utils.data import Dataset, Subset, WeightedRandomSampler
from torchvision.io import decode_image
from torchvision.transforms import v2
from tqdm import tqdm

from pdmx import PDMX
from sheetmusic import Score
from .staffer_model import StafferConfig


class StafferDataset(Dataset[tuple[Tensor, Tensor, Tensor, Tensor]]):
    pdmx: PDMX
    # layout path, png path, page number, part_count, max_sys_bottom
    # The last two items are used when use_sampler is enabled.
    items: list[tuple[Path, Path, int, int, float]]

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
            part_count = max(score.staff_count, 1) // max(score.system_count, 1)
            for page in score.pages:
                if score.page_count > 1:
                    png_file = pdmx.get_page_path(mxl_file, "png", page.page_number)
                else:
                    png_file = pdmx.get_path(mxl_file, "png")
                raw = max(
                    (s.box.bottom / page.image_height for s in page.systems),
                    default=0.0,
                )
                if raw > 1.0:
                    logging.warning(
                        "%s page %d: max_sys_bottom=%.3f > 1.0, clamping",
                        layout_file,
                        page.page_number,
                        raw,
                    )
                max_sys_bottom = min(raw, 1.0)
                self.items.append(
                    (
                        layout_file,
                        png_file,
                        page.page_number,
                        part_count,
                        max_sys_bottom,
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
                if sys_idx >= self.config.num_system_queries:
                    is_ok = False
                    break
                W, H = page.image_width, page.image_height
                sys_boxes[sys_idx] = torch.tensor(
                    [
                        system.box.left / W,
                        system.box.top / H,
                        system.box.right / W,
                        system.box.bottom / H,
                    ]
                )
                for staff in system.staves:
                    # ltrb; only cols [1,3] (top, bottom) are used in the loss
                    staff_boxes[staff_idx] = torch.tensor(
                        [
                            staff.box.left / W,
                            staff.box.top / H,
                            staff.box.right / W,
                            staff.box.bottom / H,
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
        _, _, _, part_count, max_sys_bottom = dataset.items[i]
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
