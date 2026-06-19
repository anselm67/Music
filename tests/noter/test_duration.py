import math

import pytest

from noter import Vocab
from noter.duration import join_duration, snap_suffix, split_duration


class TestSplitDuration:
    @pytest.mark.parametrize(
        "token, base, length",
        [
            ("C/4", "C", 1 / 4),
            ("C#/4:1", "C#", 3 / 8),  # dotted quarter
            ("C-/2:2", "C-", 7 / 8),  # double-dotted half
            ("rest/8", "rest", 1 / 8),
            ("C:x/4", "C:x", 1 / 4),  # drum note
            ("cc/8", "cc", 1 / 8),  # multi-letter (octave) pitch
            ("E/12", "E", 1 / 12),  # triplet eighth
        ],
    )
    def test_splits_numeric_duration(
        self, token: str, base: str, length: float
    ) -> None:
        got_base, got_log2 = split_duration(token)
        assert got_base == base
        assert got_log2 == pytest.approx(math.log2(length))

    @pytest.mark.parametrize(
        "token",
        ["C/q", "4/4", "12/8", "clef-G", "key-", "key##", "=", "==", "=||", ".", "SOS"],
    )
    def test_leaves_non_durations_intact(self, token: str) -> None:
        assert split_duration(token) == (token, None)

    @pytest.mark.parametrize("token", ["rest/0", "D/0", "GG/0", "A-/0", "C:x/0"])
    def test_breve_zero_duration_stays_intact(self, token: str) -> None:
        # recip `0` (breve/longa) has no 1/n length: leave it whole, don't divide.
        assert split_duration(token) == (token, None)


class TestSnap:
    @pytest.mark.parametrize("suffix", ["1", "2", "4", "8", "16", "4:1", "2:2", "12"])
    def test_round_trips_canonical_durations(self, suffix: str) -> None:
        base, log2_len = split_duration(f"C/{suffix}")
        assert log2_len is not None
        assert join_duration(base, log2_len) == f"C/{suffix}"

    def test_dotted_tuplet_snaps_to_plain_collision(self) -> None:
        # 6:1 (dotted triplet-eighth) has the exact length of a plain quarter.
        _, log2_len = split_duration("C/6:1")
        assert log2_len is not None
        assert log2_len == pytest.approx(math.log2(1 / 4))
        assert snap_suffix(log2_len) == "4"

    def test_clamps_beyond_table_ends(self) -> None:
        assert snap_suffix(-99.0) == "128"  # shortest in table
        assert snap_suffix(99.0) == "1:2"  # longest in table


class TestVocabPitchOnly:
    def test_from_files_drops_duration(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        # Each token appears >1 time (from_files drops singletons).
        f = tmp_path / "a.tokens"
        f.write_text(
            "clef-G\t clef-F\n"
            "clef-G\t clef-F\n"
            "C/4 E/4\t rest/4\n"
            "C/8 E/8\t rest/8\n"
            "C:x/4\t C:x/8\n"
            "4/4\t 4/4\n"
            "=\t =\n"
        )
        vocab = Vocab.from_files([f])
        keys = set(vocab._tok2i)
        # pitch/meter/clef/bar survive; the duration suffix is gone
        assert {"C", "E", "rest", "C:x", "clef-G", "clef-F", "4/4", "="} <= keys
        assert not any("/" in k and k not in {"4/4", "clef-F", "clef-G"} for k in keys)
        assert "C/4" not in keys and "C/8" not in keys

    def test_encode_ignores_duration(self) -> None:
        vocab = Vocab({"PAD": 0, "UNK": 1, "SOS": 2, "EOS": 3, "SIL": 4, "C": 5})
        # different durations of the same pitch encode to the same id
        assert vocab.encode("C/4") == 5
        assert vocab.encode("C/8") == 5
        assert vocab.encode("C/4:1") == 5
