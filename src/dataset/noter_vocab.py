import pickle
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor


class Vocab:
    PAD_T = (0, "PAD")  # Padding value for images and sequences.
    UNK_T = (1, "UNK")  # Unknown sequence token.
    SOS_T = (2, "SOS")  # Start of sequence.
    EOS_T = (3, "EOS")  # End of sequence.
    SIL_T = (4, "SIL")  # Padding value for chords.

    RESERVED_TOKENS = [PAD_T, UNK_T, SOS_T, EOS_T, SIL_T]

    PAD, UNK, SOS, EOS, SIL = map(lambda x: x[0], RESERVED_TOKENS)

    _tok2i: dict[str, int]
    _i2tok: dict[int, str]

    def __init__(self, tok2i: dict[str, int]):
        self._tok2i = tok2i
        self._i2tok = {i: s for s, i in tok2i.items()}

    def __len__(self):
        return len(self._tok2i)

    def tok2i(self, tokens: list[str], max_chords: int) -> Tensor:
        if len(tokens) > max_chords:
            raise ValueError(
                f"Number of tokens ({len(tokens)}) exceeds max_chords ({max_chords})"
            )
        tensor = torch.full((max_chords,), self.SIL)
        for idx, tok in enumerate(tokens):
            tensor[idx] = self._tok2i.get(tok, self.UNK)
        return tensor

    def i2tok(self, ids: Tensor | Iterable[int]) -> list[str]:
        if isinstance(ids, Tensor):
            ids = ids.tolist()
        return [self._i2tok.get(id, self.UNK_T[1]) for id in ids]

    def save(self, path: Path) -> None:
        with open(path, "wb+") as f:
            pickle.dump(self._tok2i, f)

    @staticmethod
    def from_files(dir: Path) -> "Vocab":
        """Generates the vocabulary from all .tokens files in DIR

        returns: A Vocab instance.
        """
        tok2i: dict[str, int] = dict()
        for tokens_file in dir.rglob("*.tokens"):
            with open(tokens_file, "r") as f:
                for record in f:
                    for token in record.strip().split():
                        if token not in tok2i:
                            tok2i[token] = len(tok2i)
        return Vocab(tok2i)

    @staticmethod
    def load(path: Path) -> "Vocab":
        with open(path, "rb") as f:
            tok2i = pickle.load(f)
        return Vocab(tok2i)
