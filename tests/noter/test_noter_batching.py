import torch

from noter import BucketBatchSampler, NoterConfig, NoterDataset, Vocab, collate_systems
from sheetmusic import Box


def _item(g: int) -> tuple[str, int, list[Box], list[int], int, int]:
    # items[i][2] is the staff-box list; only its length (the staff count) matters.
    return ("s", 0, [Box(0, 0, 1, 1)] * g, list(range(g)), 1, 2)


def test_next_same_count_keeps_staff_count_on_fallback() -> None:
    ds = NoterDataset.__new__(NoterDataset)
    ds.items = [_item(g) for g in [2, 1, 4, 2, 4]]
    # From a 2-staff system (idx 0) the next 2-staff system is idx 3, not idx 1 (1).
    assert ds._next_same_count(0) == 3
    # Wraps: from the last 4-staff (idx 4) back to the first 4-staff (idx 2).
    assert ds._next_same_count(4) == 2
    # No other system shares the count -> neighbour fallback.
    ds.items = [_item(2), _item(1)]
    assert ds._next_same_count(0) == 1


def _vocab() -> Vocab:
    return Vocab({"PAD": 0, "UNK": 1, "SOS": 2, "EOS": 3, "SIL": 4, "C/4": 5})


def _system(
    g: int, config: NoterConfig
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """A fake unpadded item as ``__getitem__`` returns it: G real staves, all-True."""
    h, w = config.input_shape
    images = torch.ones(g, 1, h, w)
    widths = torch.full((g,), w)
    sequences = torch.full((g, config.max_seqlen, config.max_chords), Vocab.SOS)
    articulations = torch.zeros(g, config.max_seqlen, config.max_chords, 5)
    mask = torch.ones(g, dtype=torch.bool)
    return images, widths, sequences, articulations, mask


def test_collate_pads_to_batch_max_staves() -> None:
    config, vocab = NoterConfig(), _vocab()
    batch = [_system(1, config), _system(3, config), _system(2, config)]
    images, widths, sequences, articulations, mask = collate_systems(
        batch, config, vocab
    )
    # Padded to the batch max (3), NOT the global max_staves (4).
    assert images.shape[:2] == (3, 3)
    assert mask.tolist() == [
        [True, False, False],
        [True, True, True],
        [True, True, False],
    ]
    # A padding slot is a full-width zero image (no all-masked encoder patch) with
    # an SOS-only sequence.
    assert images[0, 1].abs().sum().item() == 0.0
    assert widths[0, 1].item() == config.input_shape[1]
    assert sequences[0, 1, 0, 0].item() == Vocab.SOS
    assert sequences[0, 1, 1, 0].item() == Vocab.PAD


def test_collate_homogeneous_batch_pads_nothing() -> None:
    config, vocab = NoterConfig(), _vocab()
    batch = [_system(2, config), _system(2, config)]
    _, _, _, _, mask = collate_systems(batch, config, vocab)
    assert mask.all()


def test_bucket_sampler_batches_are_staff_homogeneous() -> None:
    # counts: five 2-staff, three 1-staff, two 4-staff systems.
    staff_counts = [2, 2, 1, 4, 2, 1, 2, 4, 2, 1]
    indices = list(range(len(staff_counts)))
    sampler = BucketBatchSampler(indices, staff_counts, crop_budget=4, shuffle=True)
    seen: list[int] = []
    for batch in sampler:
        counts = {staff_counts[i] for i in batch}
        assert len(counts) == 1, "a batch mixes staff counts"
        k = counts.pop()
        assert len(batch) <= max(1, 4 // k)  # crop_budget // staves
        seen += batch
    assert sorted(seen) == indices  # every index emitted exactly once
    assert len(sampler) == 3 + 1 + 2  # 2-staff:⌈5/2⌉ 1-staff:⌈3/4⌉ 4-staff:⌈2/1⌉


def test_bucket_sampler_reshuffles_each_epoch() -> None:
    staff_counts = [2] * 8
    sampler = BucketBatchSampler(
        list(range(8)), staff_counts, crop_budget=4, shuffle=True
    )
    first = [b for b in sampler]
    second = [b for b in sampler]
    assert first != second  # different order across epochs


def test_bucket_sampler_no_shuffle_is_stable_and_ordered() -> None:
    staff_counts = [2] * 6
    sampler = BucketBatchSampler(
        list(range(6)), staff_counts, crop_budget=4, shuffle=False
    )
    assert [b for b in sampler] == [[0, 1], [2, 3], [4, 5]]
    assert [b for b in sampler] == [[0, 1], [2, 3], [4, 5]]  # stable
