import re
from pathlib import Path
from typing import Optional


class KernReader:
    """Parses a kern token file for bars, and create a bar number to record index.
    """
    lines: list[str]
    bars: dict[int, int]
    first_bar: int

    @property
    def bar_count(self) -> int:
        return len(self.bars)

    def __init__(self, path: Path):
        super().__init__()
        self.path = path
        self.bars = dict()
        self.first_bar = -1
        self.load_tokens()

    BAR_RE = re.compile(r'^=+\s*(\d+)?.*$')

    def load_tokens(self):
        with open(self.path.with_suffix(".tokens"), "r") as fp:
            self.lines = [line.strip() for line in fp.readlines()]
        # Constructs the bars index.
        for lineno in range(0, len(self.lines)):
            line = self.lines[lineno]
            if (m := self.BAR_RE.match(line)):
                if m.group(1) is not None:
                    bar_number = int(m.group(1))
                    if bar_number > 0 and self.first_bar < 0:
                        self.first_bar = bar_number
                    self.bars[bar_number] = lineno

    def has_bar_zero(self):
        return 0 in self.bars

    def get_text(self, barno: int) -> Optional[list[str]]:
        if barno < self.first_bar:
            barno = 0
        bos = self.bars.get(barno, -1)
        if bos >= 0:
            # Includes the marker for the next bar, feels more comfortable.
            eos = self.bars.get(barno + 1, -1) + 1
            return self.lines[bos:eos] if eos > 0 else self.lines[bos:]
        else:
            return None

    def header(self):
        return self.lines[:10]
