import json
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
    items: list[tuple[Path, int]]

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
            v2.Normalize(mean=[], std=[]),
        ])
        # Build flat list of (mxl_path, page_number) pairs
        self.items = []
        for _, row in pdmx.df.iterrows():
            mxl_file = pdmx.home / row['mxl']
            layout_path = pdmx.get_path(mxl_file, 'layout')
            score = Score.from_json(json.loads(layout_path.read_text()))
            for page in score.pages:
                png_file = pdmx.get_page_path(
                    mxl_file, 'png', page.page_number)
                self.items.append((png_file, page.page_number))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        mxl_path, page_number = self.items[idx]
        layout_path = self.pdmx.get_path(mxl_path, 'layout')
        score = Score.from_json(json.loads(layout_path.read_text()))
        page = score.pages[page_number]

        png_path = self.pdmx.get_page_path(mxl_path, 'png', page_number)
        image = decode_image(png_path.as_posix())
        image = self.transform(image)

        gt_sys_boxes = []
        gt_stave_boxes = []
        gt_assign = []

        for sys_idx, system in enumerate(page.systems):
            gt_sys_boxes.append(system.box.to_cxcywh(
                page.image_width, page.image_height))
            for staff in system.staves:
                gt_stave_boxes.append(staff.box.to_cxcywh(
                    page.image_width, page.image_height))
                gt_assign.append(sys_idx)

        return (
            image,
            torch.tensor(gt_sys_boxes, dtype=torch.float32),
            torch.tensor(gt_stave_boxes, dtype=torch.float32),
            torch.tensor(gt_assign, dtype=torch.long),
        )
