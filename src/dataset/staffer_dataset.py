"""Torch Dataset for training models against the PDMX dataset.
"""
import json
import logging
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import decode_image
from torchvision.transforms import v2

from dataset import PDMX, Score
from models import Config


class StafferDataset(Dataset):

    pdmx: PDMX
    # layout path, png path, page number
    items: list[tuple[Path, Path, int]]

    transform: v2.Transform

    def __init__(self, config: Config, pdmx: PDMX):
        self.config = config
        self.pdmx = pdmx
        self.transform = v2.Compose([
            v2.Grayscale(),
            v2.Resize(
                config.image_shape,
                interpolation=config.interpolation,
                antialias=config.antialias),
            v2.ToDtype(torch.float, scale=True),
            # Values from running: staffer stats
            v2.Normalize(mean=[0.9563435316085815], std=[0.16557540870879858]),
        ])
        # Build flat list of (mxl_path, page_number) pairs
        logging.info("Initializing StafferDataset...")
        self.items = []
        for _, row in pdmx.df.iterrows():
            mxl_file = pdmx.home / row['mxl']
            layout_file = pdmx.get_path(mxl_file, 'layout')
            score = Score.from_json(json.loads(layout_file.read_text()))
            for page in score.pages:
                if score.page_count > 1:
                    png_file = pdmx.get_page_path(
                        mxl_file, 'png', page.page_number)
                else:
                    png_file = pdmx.get_path(mxl_file, 'png')
                self.items.append((layout_file, png_file, page.page_number))
        logging.info(f"\tStafferDataset: {len(self.items)} samples.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        while True:
            layout_path, png_path, page_number = self.items[idx]
            # Attempts to decode this image, or next one when that fails.
            try:
                image = decode_image(png_path.as_posix())
                image = self.transform(image)
            except Exception as e:
                mxl_path = self.pdmx.get_path(layout_path, 'mxl')
                logging.error(f"{mxl_path}: {e}")
                idx += 1
                continue

            # Converts the Score to expected ground truth tensors.
            score = Score.from_json(json.loads(layout_path.read_text()))
            page = score.pages[page_number - 1]

            sys_boxes = torch.zeros(self.config.num_system_queries, 4)
            staff_boxes = torch.zeros(self.config.num_stave_queries, 4)
            assigns = torch.full(
                (self.config.num_stave_queries,), -1, dtype=torch.long)

            staff_idx = 0
            for sys_idx, system in enumerate(page.systems):
                sys_boxes[sys_idx] = torch.tensor(system.box.to_cxcywh(
                    page.image_width, page.image_height))
                for staff in system.staves:
                    staff_boxes[staff_idx] = torch.tensor(staff.box.to_cxcywh(
                        page.image_width, page.image_height))
                    assigns[staff_idx] = sys_idx
                    staff_idx += 1

            return image, sys_boxes, staff_boxes, assigns
