import tempfile
from pathlib import Path

import pytest
import torch

from noter import Vocab


class TestVocab:
    @pytest.fixture
    def sample_tok2i(self) -> dict[str, int]:
        return {"token1": 5, "token2": 6, "token3": 7, "=": 8, "==": 9}

    @pytest.fixture
    def vocab(self, sample_tok2i: dict[str, int]) -> Vocab:
        return Vocab(sample_tok2i)

    def test_init(self, sample_tok2i: dict[str, int]) -> None:
        vocab = Vocab(sample_tok2i)
        assert vocab._tok2i == sample_tok2i
        assert vocab._i2tok == {v: k for k, v in sample_tok2i.items()}

    def test_len(self, vocab: Vocab, sample_tok2i: dict[str, int]) -> None:
        assert len(vocab) == len(sample_tok2i)

    def test_encode_plain_token(self, vocab: Vocab) -> None:
        assert vocab.encode("token1") == 5

    def test_encode_unknown_token(self, vocab: Vocab) -> None:
        assert vocab.encode("unknown") == Vocab.UNK

    def test_encode_flat_token_not_in_vocab(self, vocab: Vocab) -> None:
        # "token1/8" is an opaque string; UNK if not in vocab
        assert vocab.encode("token1/8") == Vocab.UNK

    def test_encode_bar(self, vocab: Vocab) -> None:
        encoded = vocab.encode("==12")
        assert encoded == 9

    def test_tok2i_normal(self, vocab: Vocab) -> None:
        tokens = ["token1", "token2"]
        max_chords = 4
        result = vocab.tok2i(tokens, max_chords)
        expected = torch.tensor([5, 6, 4, 4])  # 4 is SIL
        assert torch.equal(result, expected)

    def test_tok2i_with_unknown(self, vocab: Vocab) -> None:
        tokens = ["token1", "unknown", "token2"]
        max_chords = 3
        result = vocab.tok2i(tokens, max_chords)
        expected = torch.tensor([5, 1, 6])  # 1 is UNK
        assert torch.equal(result, expected)

    def test_tok2i_empty_tokens(self, vocab: Vocab) -> None:
        tokens: list[str] = []
        max_chords = 2
        result = vocab.tok2i(tokens, max_chords)
        expected = torch.tensor([4, 4])  # 4 is SIL
        assert torch.equal(result, expected)

    def test_tok2i_max_chords_zero(self, vocab: Vocab) -> None:
        tokens = ["token1"]
        max_chords = 0
        with pytest.raises(ValueError, match="Number of tokens .* exceeds max_chords"):
            vocab.tok2i(tokens, max_chords)

    def test_tok2i_more_tokens_than_max(self, vocab: Vocab) -> None:
        tokens = ["token1", "token2", "token3", "extra"]
        max_chords = 2
        with pytest.raises(ValueError, match="Number of tokens .* exceeds max_chords"):
            vocab.tok2i(tokens, max_chords)

    def test_i2tok_tensor(self, vocab: Vocab) -> None:
        ids = torch.tensor([[5], [1], [6]])
        result = vocab.i2tok(ids)
        expected = ["token1", "UNK", "token2"]
        assert result == expected

    def test_decode_bar_token(self, vocab: Vocab) -> None:
        # IDs 8 and 9 decode to the base bar forms, not to "=12"
        assert vocab.decode(8) == "="
        assert vocab.decode(9) == "=="

    def test_i2tok_unknown_id(self, vocab: Vocab) -> None:
        ids = torch.tensor([[5], [999], [6]])
        result = vocab.i2tok(ids)
        expected = ["token1", "UNK", "token2"]
        assert result == expected

    def test_i2tok_empty(self, vocab: Vocab) -> None:
        ids = torch.empty(0)
        result = vocab.i2tok(ids)
        assert result == []

    def test_save_and_load(self, vocab: Vocab, sample_tok2i: dict[str, int]) -> None:
        with tempfile.NamedTemporaryFile(delete=False) as f:
            path = Path(f.name)
        try:
            vocab.save(path)
            loaded_vocab = Vocab.load(path)
            assert loaded_vocab._tok2i == sample_tok2i
            assert loaded_vocab._i2tok == vocab._i2tok
        finally:
            path.unlink()

    def test_from_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dir_path = Path(temp_dir)
            # Create a .tokens file
            tokens_file = dir_path / "test.tokens"
            tokens_file.write_text(
                "token1 token2\ntoken3 token1\ntoken1 token2\ntoken3 token1\n"
            )
            # Create another file
            sub_dir = dir_path / "sub"
            sub_dir.mkdir()
            tokens_file2 = sub_dir / "another.tokens"
            tokens_file2.write_text("token4\ntoken4\n")

            vocab = Vocab.from_files([tokens_file, tokens_file2])
            expected_tok2i = {
                "EOS": 3,
                "PAD": 0,
                "SIL": 4,
                "SOS": 2,
                "UNK": 1,
                "token1": 5,
                "token2": 6,
                "token3": 7,
                "token4": 8,
            }
            assert vocab._tok2i == expected_tok2i

    def test_extend_from_files_preserves_ids_and_appends(self) -> None:
        base = Vocab({"PAD": 0, "UNK": 1, "SOS": 2, "EOS": 3, "SIL": 4, "old": 5})
        with tempfile.TemporaryDirectory() as temp_dir:
            tokens_file = Path(temp_dir) / "ks.tokens"
            # "new" twice (kept), "rare" once (dropped to UNK), "old" already known
            tokens_file.write_text("old new rare\nnew old\n")
            extended = base.extend_from_files([tokens_file])
        # every base id is preserved verbatim
        for tok, idx in base._tok2i.items():
            assert extended.encode(tok) == idx
        # kept token appended at the tail; rare singleton stays OOV -> UNK
        assert extended.encode("new") == 6
        assert extended.encode("rare") == Vocab.UNK
        assert len(extended) == len(base) + 1

    def test_extend_from_files_min_count(self) -> None:
        base = Vocab({"PAD": 0, "UNK": 1, "SOS": 2, "EOS": 3, "SIL": 4})
        with tempfile.TemporaryDirectory() as temp_dir:
            tokens_file = Path(temp_dir) / "ks.tokens"
            tokens_file.write_text("a a a b b c\n")
            extended = base.extend_from_files([tokens_file], min_count=3)
        assert extended.encode("a") == 5  # count 3 >= 3
        assert extended.encode("b") == Vocab.UNK  # count 2 < 3
        assert extended.encode("c") == Vocab.UNK  # count 1 < 3
        assert len(extended) == len(base) + 1

    def test_extend_from_files_no_files(self) -> None:
        base = Vocab({"PAD": 0, "UNK": 1, "SOS": 2, "EOS": 3, "SIL": 4})
        extended = base.extend_from_files([])  # must not ZeroDivisionError
        assert extended._tok2i == base._tok2i

    def test_from_files_no_files(self) -> None:
        vocab = Vocab.from_files([])
        assert vocab._tok2i == {"EOS": 3, "PAD": 0, "SIL": 4, "SOS": 2, "UNK": 1}

    def test_constants(self) -> None:
        assert Vocab.PAD_T == (0, "PAD")
        assert Vocab.UNK_T == (1, "UNK")
        assert Vocab.SOS_T == (2, "SOS")
        assert Vocab.EOS_T == (3, "EOS")
        assert Vocab.SIL_T == (4, "SIL")
        assert Vocab.RESERVED_TOKENS == [
            Vocab.PAD_T,
            Vocab.UNK_T,
            Vocab.SOS_T,
            Vocab.EOS_T,
            Vocab.SIL_T,
        ]
        assert Vocab.PAD == 0
        assert Vocab.UNK == 1
        assert Vocab.SOS == 2
        assert Vocab.EOS == 3
        assert Vocab.SIL == 4
