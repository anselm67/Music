from unittest.mock import MagicMock

from noter import NoterConfig, NoterDataset, Vocab


def _dataset() -> tuple[NoterDataset, MagicMock]:
    vocab = Vocab(
        {"PAD": 0, "UNK": 1, "SOS": 2, "EOS": 3, "SIL": 4, "C/4": 5, "D/4": 6}
    )
    source = MagicMock()
    source.scores.return_value = []  # empty: __init__ builds no items
    return NoterDataset(NoterConfig(), source, vocab), source


def test_load_sequence_skips_record_with_too_few_spines() -> None:
    # A 2-staff system wants spine 1, but the bar range has a single-spine line.
    # Must skip the sample (return None), not raise IndexError into the worker.
    ds, source = _dataset()
    source.records.return_value = ["C/4"]  # one column, no tab -> no spine 1
    assert (
        ds._load_sequence("x", spine_number=1, first_bar_number=1, last_bar_number=2)
        is None
    )


def test_load_sequence_reads_requested_spine() -> None:
    ds, source = _dataset()
    source.records.return_value = ["C/4\tD/4"]
    seq = ds._load_sequence("x", spine_number=1, first_bar_number=1, last_bar_number=2)
    assert seq is not None
    # spine 1 is "D/4" -> the first non-SOS row's first chord slot decodes to D/4
    assert ds.vocab.decode(int(seq[1, 0].item())) == "D/4"
