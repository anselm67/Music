"""Parses a Kern _token_ file to align it with bar numbers to a pdf/png score."""

import re
from pathlib import Path
from typing import Optional


class KernReader:
    """Parses a kern token file for bars, and create a bar number to record index."""

    lines: list[str]
    preambles: dict[int, list[str]]
    bars: dict[int, int]
    first_bar: int

    @property
    def bar_count(self) -> int:
        return len(self.bars)

    @property
    def spine_count(self) -> int:
        """Number of token columns (the tokenizer drops **dynam etc., so this is the
        count of musical spines = staves, unless a staff is voiced into >1 spine)."""
        return self._get_spline_count()

    def __init__(self, path: Path):
        super().__init__()
        self.path = path
        self.bars = dict()
        self.preambles = dict()
        self.first_bar = -1
        self.load_tokens()

    BAR_RE = re.compile(r"^=+\s*(\d+)?.*$")
    SIG_RE = re.compile(r"^\d+/\d+$")
    CLEF_RE = re.compile(r"^clef-.*$")
    KEYS_RE = re.compile(r"^key.*$")

    def _get_spline_count(self) -> int:
        if len(self.lines) == 0:
            return 0
        else:
            return len(self.lines[0].split("\t"))

    def _make_preamble(
        self, barno: int, clef: list[str], keys: list[str], sig: list[str]
    ) -> list[str]:
        preamble = []
        if all(s for s in clef):
            preamble.append("\t".join(clef))
        if all(s for s in keys):
            preamble.append("\t".join(keys))
        if barno == 1 and all(s for s in sig):
            preamble.append("\t".join(sig))
        return preamble

    def load_tokens(self) -> None:
        with open(self.path, "r") as fp:
            self.lines = [line.strip() for line in fp.readlines()]
        if (spine_count := self._get_spline_count()) == 0:
            return
        sig: list[str] = [""] * spine_count
        clef: list[str] = [""] * spine_count
        keys: list[str] = [""] * spine_count
        for lineno, line in enumerate(self.lines):
            for spine, tok in enumerate(line.split("\t")):
                assert spine < spine_count, "Invalid .tokens: spine count mismatch."
                if self.SIG_RE.match(tok):
                    sig[spine] = tok
                elif self.CLEF_RE.match(tok):
                    clef[spine] = tok
                elif self.KEYS_RE.match(tok):
                    keys[spine] = tok

            if m := self.BAR_RE.match(line):
                if m.group(1) is not None:
                    bar_number = int(m.group(1))
                    if bar_number > 0 and self.first_bar < 0:
                        self.first_bar = bar_number
                    self.bars[bar_number] = lineno
                    self.preambles[bar_number] = self._make_preamble(
                        bar_number, clef, keys, sig
                    )

        # The closing barline of a piece is not a measure: a final barline written
        # with a number at the very end — `==150`, or a bare trailing `=150` — opens a
        # bar that no music fills. It is the file's last line (an unnumbered `==`/`=||`
        # was never counted; content-ending scores have no trailing barline at all), so
        # drop the bar marker sitting on that last line. The count is then the real
        # measure count, matching the layout geometry, whatever the final glyph.
        last_line = next(
            (i for i in reversed(range(len(self.lines))) if self.lines[i]), -1
        )
        for bar_number in [b for b, ln in self.bars.items() if ln == last_line]:
            del self.bars[bar_number]
            self.preambles.pop(bar_number, None)

    def has_bar_zero(self) -> bool:
        return 0 in self.bars

    def get_text(
        self, start_barno: int, end_barno: int | None = None
    ) -> Optional[list[str]]:
        if start_barno < self.first_bar:
            start_barno = 0
        if end_barno is None:
            end_barno = start_barno + 1
        bos = self.bars.get(start_barno, -1)
        if bos < 0:
            return None
        preamble = self.preambles.get(start_barno, [])
        # End at the first bar marker at or after end_barno in FILE order (and
        # include that marker line — it reads more comfortably). Looking up by
        # ``end_barno`` directly breaks on non-consecutive numbering — a pickup
        # ``=0`` then ``=2`` (no bar 1) — where ``end_barno`` may not be a real bar;
        # the old code then fell through to the end of the piece. NB ``self.bars``
        # keeps only the LAST line for a repeated bar number, so first-ending voltas
        # (a bar number appearing twice) still slice imperfectly — a pre-existing
        # limitation that needs a list-per-bar index to fix.
        later = [ln for bn, ln in self.bars.items() if bn >= end_barno]
        eos = min(later) + 1 if later else len(self.lines)
        return preamble + self.lines[bos:eos]

    def header(self) -> list[str]:
        return self.lines[:10]
