from collections import defaultdict
from collections.abc import Iterator
from functools import partial

import torch
import lightning as L
from torch.utils.data import DataLoader, Sampler

from sheetmusic import Source

from .noter_dataset import NoterDataset, collate_systems
from .noter_model import NoterConfig
from .noter_vocab import Vocab


class BucketBatchSampler(Sampler[list[int]]):
    """Batches dataset indices so every batch holds systems of ONE staff count.

    A bucket of ``G``-staff systems is chunked into batches of ``crop_budget // G``
    systems, so the encoded crop count per batch (``G`` × systems) stays ~constant —
    peak memory is flat whatever the staves per system — and, because a batch is
    staff-count-homogeneous, ``collate_systems`` pads nothing. When ``shuffle`` the
    within-bucket order and the batch order are reseeded each epoch.
    """

    def __init__(
        self,
        indices: list[int],
        staff_counts: list[int],
        crop_budget: int,
        shuffle: bool,
        seed: int = 0,
    ) -> None:
        self.buckets: dict[int, list[int]] = defaultdict(list)
        for i in indices:
            self.buckets[staff_counts[i]].append(i)
        self.crop_budget = crop_budget
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def _batch_size(self, staves: int) -> int:
        return max(1, self.crop_budget // staves)

    def __iter__(self) -> Iterator[list[int]]:
        gen = torch.Generator()
        gen.manual_seed(self.seed + self.epoch)
        self.epoch += 1
        batches: list[list[int]] = []
        for staves, idxs in self.buckets.items():
            order = (
                torch.randperm(len(idxs), generator=gen).tolist()
                if self.shuffle
                else range(len(idxs))
            )
            ordered = [idxs[i] for i in order]
            bs = self._batch_size(staves)
            batches += [ordered[i : i + bs] for i in range(0, len(ordered), bs)]
        if self.shuffle:
            batches = [batches[i] for i in torch.randperm(len(batches), generator=gen)]
        yield from batches

    def __len__(self) -> int:
        return sum(
            -(-len(idxs) // self._batch_size(staves))
            for staves, idxs in self.buckets.items()
        )


class NoterDataModule(L.LightningDataModule):
    config: NoterConfig
    source: Source
    vocab: Vocab
    num_workers: int
    full: NoterDataset
    train_idx: list[int]
    val_idx: list[int]

    def __init__(
        self, config: NoterConfig, source: Source, vocab: Vocab, num_workers: int = 8
    ):
        super().__init__()
        self.config = config
        self.source = source
        self.vocab = vocab
        self.num_workers = num_workers

    def setup(self, stage: str | None = None) -> None:
        want = self.config.train_len + self.config.valid_len
        self.full = NoterDataset(self.config, self.source, self.vocab, count=want)
        # The dataset caps at `count`, so a short corpus under-fills it. The old
        # random_split raised on a length mismatch; keep that loud failure rather than
        # silently truncating train_idx and handing val an empty split (which would
        # disable validation/checkpointing without a word).
        assert len(self.full) == want, (
            f"dataset yielded {len(self.full)} systems, need train_len+valid_len="
            f"{want}; enlarge the subset (--csv / -n) or lower --train-len/--valid-len"
        )
        # Per-item staff count from the stored boxes (index 2), so the sampler can
        # bucket without loading images.
        self.staff_counts = [len(item[2]) for item in self.full.items]
        # Index-level split (not random_split, whose Subset hides the index->staff-
        # count map the bucket sampler needs).
        perm = torch.randperm(
            len(self.full), generator=torch.Generator().manual_seed(42)
        ).tolist()
        self.train_idx = perm[: self.config.train_len]
        self.val_idx = perm[
            self.config.train_len : self.config.train_len + self.config.valid_len
        ]

    def _loader(self, indices: list[int], shuffle: bool) -> DataLoader:
        sampler = BucketBatchSampler(
            indices, self.staff_counts, self.config.crop_budget, shuffle=shuffle
        )
        return DataLoader(
            self.full,
            batch_sampler=sampler,
            collate_fn=partial(collate_systems, config=self.config, vocab=self.vocab),
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_idx, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_idx, shuffle=False)
