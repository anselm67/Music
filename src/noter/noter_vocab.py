import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import torch
from torch import Tensor

from pdmx import PDMX


class Vocab:
    PAD_T = (0, "PAD")  # Padding value for images, sequences, and chord slots.
    UNK_T = (1, "UNK")  # Unknown sequence token.
    SOS_T = (2, "SOS")  # Start of sequence.
    EOS_T = (3, "EOS")  # End of sequence.

    RESERVED_TOKENS = [PAD_T, UNK_T, SOS_T, EOS_T]

    PAD, UNK, SOS, EOS = map(lambda x: x[0], RESERVED_TOKENS)

    _tok2i: dict[str, int]
    _i2tok: dict[int, str]

    # Bar tokens carry a bar number that we strip — the model only needs to know
    # whether it's a single or double barline, not which bar number it is.
    BAR_RE = re.compile(r"^(?P<base>==?)(?P<barno>\d+)$")

    def __init__(self, tok2i: dict[str, int]):
        self._tok2i = tok2i
        self._i2tok = {i: s for s, i in tok2i.items()}

    def __len__(self) -> int:
        return len(self._tok2i)

    @staticmethod
    def _strip_bar_number(str_tok: str) -> str:
        if m := Vocab.BAR_RE.match(str_tok):
            return m.group("base")
        return str_tok

    def encode(self, str_tok: str) -> int:
        return self._tok2i.get(Vocab._strip_bar_number(str_tok), self.UNK)

    def decode(self, int_tok: int) -> str:
        return self._i2tok.get(int_tok, self.UNK_T[1])

    def tok2i(self, tokens: list[str], max_chords: int) -> Tensor:
        if len(tokens) > max_chords:
            raise ValueError(
                f"Number of tokens ({len(tokens)}) exceeds max_chords ({max_chords})"
            )
        tensor = torch.full((max_chords,), self.PAD)
        for idx, tok in enumerate(tokens):
            tensor[idx] = self.encode(tok)
        return tensor

    def i2tok(self, ids: Tensor) -> list[str]:
        tokens: list[str] = []
        for i in range(0, len(ids)):
            tokens.append(
                " ".join(
                    self.decode(int(id.item())) for id in ids[i, :] if id != self.PAD
                )
            )
            if ids[i, 0] == self.EOS:
                break
        return tokens

    def save(self, path: Path) -> None:
        with open(path, "w+") as f:
            json.dump(self._tok2i, f, indent=2)

    @staticmethod
    def from_files(files: list[Path]) -> "Vocab":
        counts: dict[str, int] = defaultdict(int)
        logging.info(f"Parsing {len(files):,} .tokens files...")
        for tokens_file in files:
            with open(tokens_file, "r") as f:
                for record in f:
                    for token in record.strip().split():
                        counts[Vocab._strip_bar_number(token)] += 1

        tok2i: dict[str, int] = {s: i for i, s in Vocab.RESERVED_TOKENS}
        for key, value in counts.items():
            if value > 1:
                tok2i[key] = len(tok2i)

        vocab = Vocab(tok2i)
        logging.info(f"\t{len(vocab):,} tokens created.")
        return vocab

    @staticmethod
    def from_pdmx(pdmx: PDMX) -> "Vocab":
        files: list[Path] = []
        for _, row in pdmx.df.iterrows():
            mxl_str = row["mxl"]
            if not isinstance(mxl_str, str):
                continue
            files.append(pdmx.get_path(Path(mxl_str), "tokens"))
        return Vocab.from_files(files)

    @staticmethod
    def load(path: Path) -> "Vocab":
        with open(path, "r") as f:
            tok2i = json.load(f)
        return Vocab(tok2i)
