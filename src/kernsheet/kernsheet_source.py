"""``Source`` implementation backed by a :class:`KernSheet` dataset.

A thin adapter: keeps ``KernSheet``'s catalog/management role separate from the
read-only data contract the datasets consume.  The score key (``Score.id``) is the
layout-json stem (e.g. ``bach/fugue/bwv_856/bwv_856-0``): globally unique, so
neither shared "-all" PDFs nor ``-0``/``-1`` editions collide.  The ``.krn`` / tokens
live under the entry *key* (shared across a work's editions).
"""

from collections.abc import Iterator
from pathlib import Path

from torch import Tensor
from torchvision.io import decode_image

from sheetmusic import Score

from .kernsheet import KernSheet


class KernSheetSource:
    def __init__(self, kern_sheet: KernSheet) -> None:
        self.kern_sheet = kern_sheet

    def scores(self) -> Iterator[Score]:
        for _, score in self.kern_sheet.items():
            # is_file (not exists): an unmigrated score has an empty json_path,
            # whose layout_path resolves to the layout/ directory — which exists.
            if self.kern_sheet.layout_path(score).is_file():
                yield self.kern_sheet.load_score(score.id)

    def score(self, id: str) -> Score:
        return self.kern_sheet.load_score(id)

    def image_path(self, id: str, page_number: int) -> Path:
        return self.kern_sheet.png_path(id, page_number)

    def image(self, id: str, page_number: int) -> Tensor:
        return decode_image(self.image_path(id, page_number).as_posix())

    def records(self, id: str, first_bar: int, last_bar: int) -> list[str] | None:
        return self.kern_sheet.load_tokens(id).get_text(first_bar, last_bar)
