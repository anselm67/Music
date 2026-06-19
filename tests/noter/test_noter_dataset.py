from unittest.mock import MagicMock

from noter import NoterConfig, SequenceLoader, Vocab


def _loader() -> tuple[SequenceLoader, Vocab, MagicMock]:
    # Vocab is pitch-only; the duration head predicts `/4` separately.
    vocab = Vocab({"PAD": 0, "UNK": 1, "SOS": 2, "EOS": 3, "SIL": 4, "C": 5, "D": 6})
    source = MagicMock()
    cfg = NoterConfig()
    return SequenceLoader(source, vocab, cfg.max_seqlen, cfg.max_chords), vocab, source


def test_load_sequence_skips_record_with_too_few_spines() -> None:
    # A 2-staff system wants spine 1, but the bar range has a single-spine line.
    # Must skip the sample (return None), not raise IndexError into the worker.
    load_sequence, _, source = _loader()
    source.records.return_value = ["C/4"]  # one column, no tab -> no spine 1
    assert load_sequence("x", spine_number=1, first_bar=1, last_bar=2) is None


def test_load_sequence_reads_requested_spine() -> None:
    load_sequence, vocab, source = _loader()
    source.records.return_value = ["C/4\tD/4"]
    seq = load_sequence("x", spine_number=1, first_bar=1, last_bar=2)
    assert seq is not None
    # spine 1 is "D/4" -> duration stripped, first non-SOS chord slot decodes to D
    assert vocab.decode(int(seq[1, 0].item())) == "D"
