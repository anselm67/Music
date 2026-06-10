import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import torch
from torch import Tensor

from pdmx import PDMX


class Vocab:
    PAD_T = (0, "PAD")  # Padding value for images and time-axis sequence positions.
    UNK_T = (1, "UNK")  # Unknown sequence token.
    SOS_T = (2, "SOS")  # Start of sequence.
    EOS_T = (3, "EOS")  # End of sequence.
    SIL_T = (4, "SIL")  # Padding value for empty chord slots.

    RESERVED_TOKENS = [PAD_T, UNK_T, SOS_T, EOS_T, SIL_T]

    PAD, UNK, SOS, EOS, SIL = map(lambda x: x[0], RESERVED_TOKENS)

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

    def barline_ids(self) -> set[int]:
        """Ids of barline tokens (the bar-number-stripped form starts with '=')."""
        return {i for s, i in self._tok2i.items() if s.startswith("=")}

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
            if ids[i, 0] == self.EOS:
                break
            tokens.append(
                " ".join(
                    self.decode(int(id.item())) for id in ids[i, :] if id != self.SIL
                )
            )
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

    def extend_from_files(self, files: list[Path], min_count: int = 2) -> "Vocab":
        """Return a copy of this vocab with new tokens from ``files`` appended.

        Existing token->id mappings are preserved verbatim; new tokens (seen at
        least ``min_count`` times and not already known) are assigned fresh ids at
        the tail. This keeps a pretrained checkpoint's embedding/output rows valid
        so it can be fine-tuned on a new corpus after growing those two tensors.
        """
        counts: dict[str, int] = defaultdict(int)
        logging.info(f"Parsing {len(files):,} .tokens files to extend vocab...")
        for tokens_file in files:
            with open(tokens_file, "r") as f:
                for record in f:
                    for token in record.strip().split():
                        counts[Vocab._strip_bar_number(token)] += 1

        oov = {t: c for t, c in counts.items() if t not in self._tok2i}
        tok2i = dict(self._tok2i)
        # Sort so appended ids are content-determined, not file-scan order: a
        # changed file set must not shift the ids of tokens a checkpoint already
        # learned during a prior extend.
        for key in sorted(oov):
            if oov[key] >= min_count:
                tok2i[key] = len(tok2i)

        added = len(tok2i) - len(self._tok2i)
        dropped = len(oov) - added
        total = sum(counts.values())
        oov_occ = sum(oov.values())
        pct = (100 * oov_occ / total) if total else 0.0
        logging.info(
            f"\tOOV {len(oov):,} unique / {oov_occ:,} occ "
            f"({pct:.4f}% weighted); "
            f"{added:,} added (count>={min_count}), "
            f"{dropped:,} rare dropped to UNK; "
            f"vocab {len(self._tok2i):,} -> {len(tok2i):,}."
        )
        return Vocab(tok2i)

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
