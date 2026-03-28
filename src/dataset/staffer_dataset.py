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
    config: Config
    pdmx: PDMX
    # layout path, png path, page number
    items: list[tuple[Path, Path, int]]

    transform: v2.Transform

    def __init__(self, config: Config, pdmx: PDMX, count: int = -1):
        self.config = config
        self.pdmx = pdmx
        self.transform = v2.Compose([
            v2.Grayscale(),
            v2.Resize(
                config.image_shape,
                interpolation=config.interpolation,
                antialias=config.antialias),
            v2.ToDtype(torch.float, scale=True),
            # v2.Normalize(mean=[], std=[]),
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
            if count > 0:
                count -= 1
                if count <= 0:
                    break
        logging.info(f"\tStafferDataset: {len(self.items)} samples.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        layout_path, png_path, page_number = self.items[idx]
        score = Score.from_json(json.loads(layout_path.read_text()))
        page = score.pages[page_number - 1]

        image = decode_image(png_path.as_posix())
        image = self.transform(image)

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
