"""Lightning dataset module wrapping StafferDataset over a Source."""

import logging

import lightning as L
from torch.utils.data import DataLoader, random_split

from sheetmusic import Source

from .staffer_dataset import StafferDataset, build_sampler
from .staffer_model import StafferConfig


class StafferDataModule(L.LightningDataModule):
    config: StafferConfig
    source: Source
    use_sampler: bool
    num_workers: int

    def __init__(
        self,
        config: StafferConfig,
        source: Source,
        use_sampler: bool,
        num_workers: int = 8,
    ):
        super().__init__()
        self.config = config
        self.source = source
        self.use_sampler = use_sampler
        self.num_workers = num_workers

    def setup(self, stage: str | None = None) -> None:
        full = StafferDataset(
            self.config,
            self.source,
            count=self.config.train_len + self.config.valid_len,
        )
        self.train_ds, self.val_ds = random_split(
            full, [self.config.train_len, self.config.valid_len]
        )

    def train_dataloader(self) -> DataLoader:
        if self.use_sampler:
            logging.info(f"train_dataloader: {self.num_workers} workers with sampler.")
            return DataLoader(
                self.train_ds,
                batch_size=self.config.batch_size,
                sampler=build_sampler(self.train_ds, self.config.bottom_bias),
                num_workers=self.num_workers,
                pin_memory=True,
            )
        else:
            logging.info(f"train_dataloader: {self.num_workers} workers no sampler.")
            return DataLoader(
                self.train_ds,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
            )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )


# vscode - End of File
