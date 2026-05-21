import lightning as L
from torch.utils.data import DataLoader, Dataset, random_split

from pdmx import PDMX

from .noter_dataset import NoterDataset
from .noter_model import NoterConfig


class NoterDataModule(L.LightningDataModule):
    config: NoterConfig
    pdmx: PDMX
    num_workers: int
    train_ds: Dataset
    val_ds: Dataset

    def __init__(self, config: NoterConfig, pdmx: PDMX, num_workers: int = 8):
        super().__init__()
        self.config = config
        self.pdmx = pdmx
        self.num_workers = num_workers

    def setup(self, stage: str | None = None) -> None:
        full = NoterDataset(
            self.config, self.pdmx, count=self.config.train_len + self.config.valid_len
        )
        self.train_ds, self.val_ds = random_split(
            full, [self.config.train_len, self.config.valid_len]
        )

    def train_dataloader(self) -> DataLoader:
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
            num_workers=self.num_workers,
            pin_memory=True,
        )
