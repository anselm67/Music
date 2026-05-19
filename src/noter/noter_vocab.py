import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import torch
from torch import Tensor

from pdmx import PDMX


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

    def __len__(self) -> int:
        return len(self._tok2i)

    DURATION_RE = re.compile(
        r"^(?P<base>[^/]+)(?:/(?P<duration>\d+)(?::(?P<dots>\d+))?)$"
    )
    BAR_RE = re.compile(r"^(?P<base>==?)(?P<barno>\d+)$")

    @staticmethod
    def _get_base(str_tok: str) -> str:
        if m := Vocab.DURATION_RE.match(str_tok):
            return m.group("base")
        elif m := Vocab.BAR_RE.match(str_tok):
            return m.group("base")
        return str_tok

    def encode(self, str_tok: str) -> int:
        if m := self.DURATION_RE.match(str_tok):
            base = self._tok2i.get(m.group("base"), self.UNK)
            duration = int(m.group("duration"))
            dots = int(m.group("dots") or 0)
            while dots > 0:
                duration += duration // 2
                dots -= 1
            return base | (duration << 16)
        elif m := self.BAR_RE.match(str_tok):
            # We don't encode the bar number around.
            return self._tok2i.get(m.group("base"), self.UNK)
        return self._tok2i.get(str_tok, self.UNK)

    def decode(self, int_tok: int) -> str:
        base = self._i2tok.get(int_tok & 0xFFFF, self.UNK_T[1])
        arg = (int_tok & 0xFFFF0000) >> 16
        if arg == 0:
            return base
        if base in ("=", "=="):
            return base + str(arg)
        return f"{base}/{arg}"

    def tok2i(self, tokens: list[str], max_chords: int) -> Tensor:
        if len(tokens) > max_chords:
            raise ValueError(
                f"Number of tokens ({len(tokens)}) exceeds max_chords ({max_chords})"
            )
        tensor = torch.full((max_chords,), self.SIL)
        for idx, tok in enumerate(tokens):
            tensor[idx] = self.encode(tok)
        return tensor

    def i2tok(self, ids: Tensor) -> list[str]:
        tokens: list[str] = []
        for i in range(0, len(ids)):
            tokens.append(
                " ".join(
                    self.decode(int(id.item())) for id in ids[i, :] if id != self.SIL
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
        """Generates the vocabulary from all .tokens files in DIR

        returns: A Vocab instance.
        """
        counts: dict[str, int] = defaultdict(int)
        logging.info(f"Parsing all .tokens files in {dir}")
        for tokens_file in files:
            with open(tokens_file, "r") as f:
                for record in f:
                    for token in record.strip().split():
                        base = Vocab._get_base(token)
                        counts[base] += 1

        tok2i: dict[str, int] = {s: i for i, s in Vocab.RESERVED_TOKENS}
        for key, value in counts.items():
            if value > 1:
                tok2i[key] = len(tok2i)

        vocab = Vocab(tok2i)
        logging.info(f"\t{len(vocab):,} tokens created from {len(files):,} files.")
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
