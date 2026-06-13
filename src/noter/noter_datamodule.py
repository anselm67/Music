import copy

import torch
import lightning as L
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from sheetmusic import Source

from .noter_dataset import NoterDataset
from .noter_model import NoterConfig
from .noter_vocab import Vocab


class NoterDataModule(L.LightningDataModule):
    config: NoterConfig
    source: Source
    vocab: Vocab
    num_workers: int
    train_ds: Dataset
    val_ds: Dataset

    def __init__(
        self, config: NoterConfig, source: Source, vocab: Vocab, num_workers: int = 8
    ):
        super().__init__()
        self.config = config
        self.source = source
        self.vocab = vocab
        self.num_workers = num_workers

    def setup(self, stage: str | None = None) -> None:
        full = NoterDataset(
            self.config,
            self.source,
            self.vocab,
            count=self.config.train_len + self.config.valid_len,
        )
        self.train_ds, self.val_ds = random_split(
            full,
            [self.config.train_len, self.config.valid_len],
            generator=torch.Generator().manual_seed(42),
        )
        if self.config.jitter > 0 or self.config.augment > 0:
            # Shallow copy shares the (read-only) items but lets the train view
            # jitter boxes / scan-augment pages while validation (self.val_ds, on
            # `full`) stays clean. enable_augment rebuilds the train view's own
            # transform, so the shared clean transform on `full` is untouched.
            train_full = copy.copy(full)
            train_full.jitter = self.config.jitter
            if self.config.augment > 0:
                train_full.enable_augment(self.config.augment)
            self.train_ds = Subset(train_full, self.train_ds.indices)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
