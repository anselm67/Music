import lightning as L
from torch.utils.data import DataLoader, random_split

from models import Config

from .pdmx import PDMX
from .staffer_dataset import StafferDataset


class StafferDataModule(L.LightningDataModule):
    config: Config
    pdmx: PDMX
    sample_count: int

    def __init__(self, config: Config, pdmx: PDMX, sample_count: int = 100_000):
        super().__init__()
        self.config = config
        self.pdmx = pdmx
        self.sample_count = sample_count

    def setup(self, stage: str | None = None):
        full = StafferDataset(self.config, self.pdmx, self.sample_count)
        n_val = int(len(full) * (1 - self.config.valid_split))
        n_train = len(full) - n_val
        self.train_ds, self.val_ds = random_split(full, [n_train, n_val])

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
