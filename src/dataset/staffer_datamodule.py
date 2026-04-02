"""Lightning dataset module for PDMX.
"""
import lightning as L
from torch.utils.data import DataLoader, random_split

from models import Config

from .pdmx import PDMX
from .staffer_dataset import StafferDataset


class StafferDataModule(L.LightningDataModule):
    config: Config
    pdmx: PDMX

    def __init__(self, config: Config, pdmx: PDMX):
        super().__init__()
        self.config = config
        self.pdmx = pdmx

    def setup(self, stage: str | None = None):
        full = StafferDataset(self.config, self.pdmx,
                              self.config.train_len + self.config.valid_len)
        self.train_ds, self.val_ds = random_split(
            full, [self.config.train_len, self.config.valid_len])

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
