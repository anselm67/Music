import json
import logging
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from dataset import PDMX, Score


class NoterDataset(Dataset):
    pdmx: PDMX
    items: list[tuple[Path, Path, int]]

    def __init__(self, pdmx: PDMX, count=-1):
        self.pdmx = pdmx
        logging.info("Initializing NoterDataset...")
        self.items = []
        for _, row in tqdm(
            pdmx.df.iterrows(), total=len(pdmx.df), desc="Loading dataset"
        ):
            mxl_file = pdmx.home / row["mxl"]
            layout_file = pdmx.get_path(mxl_file, "layout")
            score = Score.from_json(json.loads(layout_file.read_text()))
            for page in score.pages:
                if score.page_count > 1:
                    png_file = pdmx.get_page_path(mxl_file, "png", page.page_number)
                else:
                    png_file = pdmx.get_path(mxl_file, "png")
                self.items.append((layout_file, png_file, page.page_number))
                if count >= 0 and len(self.items) >= count:
                    self.items = self.items[:count]
                    break
        logging.info(f"\tNoterDataset: {len(self.items)} samples.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return (torch.tensor([]), torch.tensor([]))
