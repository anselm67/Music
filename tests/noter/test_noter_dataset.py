import torch
from unittest.mock import MagicMock

from noter import NoterConfig, Vocab, load_sequence


def _fixture() -> tuple[Vocab, MagicMock]:
    vocab = Vocab(
        {"PAD": 0, "UNK": 1, "SOS": 2, "EOS": 3, "SIL": 4, "C/4": 5, "D/4": 6}
    )
    source = MagicMock()
    return vocab, source


def test_load_sequence_skips_record_with_too_few_spines() -> None:
    # A 2-staff system wants spine 1, but the bar range has a single-spine line.
    # Must skip the sample (return None), not raise IndexError into the worker.
    vocab, source = _fixture()
    source.records.return_value = ["C/4"]  # one column, no tab -> no spine 1
    cfg = NoterConfig()
    s_sos = torch.full((1, cfg.max_chords), vocab.SOS)
    result = load_sequence(
        source, vocab, "x", 1, 1, 2, cfg.max_seqlen, cfg.max_chords, s_sos
    )
    assert result is None


def test_load_sequence_reads_requested_spine() -> None:
    vocab, source = _fixture()
    source.records.return_value = ["C/4\tD/4"]
    cfg = NoterConfig()
    s_sos = torch.full((1, cfg.max_chords), vocab.SOS)
    seq = load_sequence(
        source, vocab, "x", 1, 1, 2, cfg.max_seqlen, cfg.max_chords, s_sos
    )
    assert seq is not None
    # spine 1 is "D/4" -> the first non-SOS row's first chord slot decodes to D/4
    assert vocab.decode(int(seq[1, 0].item())) == "D/4"
