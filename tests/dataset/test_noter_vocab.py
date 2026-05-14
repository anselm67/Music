import tempfile
from pathlib import Path

import pytest
import torch

from dataset import Vocab


class TestVocab:
    @pytest.fixture
    def sample_tok2i(self):
        return {"token1": 5, "token2": 6, "token3": 7, "=": 8, "==": 9}

    @pytest.fixture
    def vocab(self, sample_tok2i):
        return Vocab(sample_tok2i)

    def test_init(self, sample_tok2i):
        vocab = Vocab(sample_tok2i)
        assert vocab._tok2i == sample_tok2i
        assert vocab._i2tok == {v: k for k, v in sample_tok2i.items()}

    def test_len(self, vocab, sample_tok2i):
        assert len(vocab) == len(sample_tok2i)

    def test_encode_plain_token(self, vocab):
        assert vocab.encode("token1") == 5

    def test_encode_unknown_token(self, vocab):
        assert vocab.encode("unknown") == Vocab.UNK

    def test_encode_duration(self, vocab):
        encoded = vocab.encode("token1/8")
        assert encoded == 5 | (8 << 16)

    def test_encode_one_dot_duration(self, vocab):
        encoded = vocab.encode("token1/8:1")
        assert encoded == 5 | (12 << 16)

    def test_encode_two_dot_duration(self, vocab):
        encoded = vocab.encode("token1/8:2")
        assert encoded == 5 | (18 << 16)

    def test_encode_bar(self, vocab):
        encoded = vocab.encode("==12")
        assert encoded == 9 | (12 << 16)

    def test_tok2i_normal(self, vocab):
        tokens = ["token1", "token2"]
        max_chords = 4
        result = vocab.tok2i(tokens, max_chords)
        expected = torch.tensor([5, 6, 4, 4])  # 4 is SIL
        assert torch.equal(result, expected)

    def test_tok2i_with_unknown(self, vocab):
        tokens = ["token1", "unknown", "token2"]
        max_chords = 3
        result = vocab.tok2i(tokens, max_chords)
        expected = torch.tensor([5, 1, 6])  # 1 is UNK
        assert torch.equal(result, expected)

    def test_tok2i_empty_tokens(self, vocab):
        tokens = []
        max_chords = 2
        result = vocab.tok2i(tokens, max_chords)
        expected = torch.tensor([4, 4])
        assert torch.equal(result, expected)

    def test_tok2i_max_chords_zero(self, vocab):
        tokens = ["token1"]
        max_chords = 0
        with pytest.raises(ValueError, match="Number of tokens .* exceeds max_chords"):
            vocab.tok2i(tokens, max_chords)

    def test_tok2i_more_tokens_than_max(self, vocab):
        tokens = ["token1", "token2", "token3", "extra"]
        max_chords = 2
        with pytest.raises(ValueError, match="Number of tokens .* exceeds max_chords"):
            vocab.tok2i(tokens, max_chords)

    def test_i2tok_tensor(self, vocab):
        ids = torch.tensor([[5], [1], [6]])
        result = vocab.i2tok(ids)
        expected = ["token1", "UNK", "token2"]
        assert result == expected

    def test_decode_duration_token(self, vocab):
        duration_id = 5 | (8 << 16)
        result = vocab.i2tok(torch.tensor([[duration_id]]))
        assert result == ["token1/8"]

    def test_decode_bar_token(self, vocab):
        bar_id = 8 | (12 << 16)
        result = vocab.i2tok(torch.tensor([[bar_id]]))
        assert result == ["=12"]

    def test_i2tok_unknown_id(self, vocab):
        ids = torch.tensor([[5], [999], [6]])
        result = vocab.i2tok(ids)
        expected = ["token1", "UNK", "token2"]
        assert result == expected

    def test_i2tok_empty(self, vocab):
        ids = []
        result = vocab.i2tok(ids)
        expected = []
        assert result == expected

    def test_save_and_load(self, vocab, sample_tok2i):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            path = Path(f.name)
        try:
            vocab.save(path)
            loaded_vocab = Vocab.load(path)
            assert loaded_vocab._tok2i == sample_tok2i
            assert loaded_vocab._i2tok == vocab._i2tok
        finally:
            path.unlink()

    def test_from_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dir_path = Path(temp_dir)
            # Create a .tokens file
            tokens_file = dir_path / "test.tokens"
            tokens_file.write_text("token1 token2\ntoken3 token1\n")
            # Create another file
            sub_dir = dir_path / "sub"
            sub_dir.mkdir()
            tokens_file2 = sub_dir / "another.tokens"
            tokens_file2.write_text("token4\n")

            vocab = Vocab.from_files(dir_path)
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

    def test_from_files_no_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dir_path = Path(temp_dir)
            vocab = Vocab.from_files(dir_path)
            assert vocab._tok2i == {"EOS": 3, "PAD": 0, "SIL": 4, "SOS": 2, "UNK": 1}

    def test_constants(self):
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
