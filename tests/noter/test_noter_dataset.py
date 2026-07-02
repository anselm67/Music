from unittest.mock import MagicMock

from noter import NoterConfig, SequenceLoader, Vocab


def _loader() -> tuple[SequenceLoader, Vocab, MagicMock]:
    vocab = Vocab(
        {"PAD": 0, "UNK": 1, "SOS": 2, "EOS": 3, "SIL": 4, "C/4": 5, "D/4": 6}
    )
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
    # spine 1 is "D/4" -> the first non-SOS row's first chord slot decodes to D/4
    assert vocab.decode(int(seq[1, 0].item())) == "D/4"


def test_load_splits_articulation_multihot() -> None:
    load_sequence, vocab, source = _loader()
    # spine 1 = arc-start + fermata D. The suffix is stripped for the token id
    # and surfaces in the parallel articulation tensor instead.
    source.records.return_value = ["C/4@s\tD/4@<f"]
    result = load_sequence.load("x", spine_number=1, first_bar=1, last_bar=2)
    assert result is not None
    seq, arts = result
    # token id unchanged by the suffix (still D/4, no new vocab entry)
    assert vocab.decode(int(seq[1, 0].item())) == "D/4"
    # row 1 (first note after SOS), slot 0: [arc-start, arc-end, s, f, a]
    assert arts[1, 0].tolist() == [1.0, 0.0, 0.0, 1.0, 0.0]
    # SOS row carries no articulations
    assert arts[0].sum().item() == 0.0
