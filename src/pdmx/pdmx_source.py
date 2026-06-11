"""``Source`` implementation backed by a :class:`PDMX` dataset.

A thin adapter: it keeps ``PDMX``'s builder / query role separate from the read-only
data contract the datasets consume. The score key (``Score.id``) is the home-relative
mxl path (e.g. ``mxl/10/10/Qm….mxl``), so everything resolves via ``home / id`` and the
existing ``get_path`` rules.
"""

import json
from collections.abc import Iterator
from pathlib import Path

from torch import Tensor
from torchvision.io import decode_image

from kern import KernReader
from sheetmusic import MixedSource, Score, Source

from .pdmx import PDMX


class PdmxSource:
    def __init__(self, pdmx: PDMX) -> None:
        self.pdmx = pdmx

    def mix(self, secondary: Source, fraction: float) -> MixedSource:
        """Wrap this PDMX source as the primary of a rehearsal :class:`MixedSource`.

        ``fraction`` is ``secondary``'s share of each epoch (0.5 -> half
        ``secondary``, half PDMX); see :class:`MixedSource`.
        """
        return MixedSource(self, secondary, fraction)

    def scores(self) -> Iterator[Score]:
        for _, row in self.pdmx.df.iterrows():
            mxl = row["mxl"]
            if isinstance(mxl, str):
                yield self.score(mxl)

    def score(self, id: str) -> Score:
        mxl_file = self.pdmx.home / id
        layout_file = self.pdmx.get_path(mxl_file, "layout")
        return Score.from_json(json.loads(layout_file.read_text()))

    def image_path(self, id: str, page_number: int) -> Path:
        mxl_file = self.pdmx.home / id
        flat = self.pdmx.get_path(mxl_file, "png")
        return (
            flat
            if flat.exists()
            else self.pdmx.get_page_path(mxl_file, "png", page_number)
        )

    def image(self, id: str, page_number: int) -> Tensor:
        return decode_image(self.image_path(id, page_number).as_posix())

    def records(self, id: str, first_bar: int, last_bar: int) -> list[str] | None:
        tokens_file = self.pdmx.get_path(self.pdmx.home / id, "tokens")
        return KernReader(tokens_file).get_text(first_bar, last_bar)
