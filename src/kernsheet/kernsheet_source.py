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

from sheetmusic import Page, Score

from .kernsheet import KernSheet


class KernSheetSource:
    def __init__(self, kern_sheet: KernSheet) -> None:
        self.kern_sheet = kern_sheet

    def scores(self) -> Iterator[Score]:
        for _, score in self.kern_sheet.items():
            yield self.kern_sheet.load_score(score.id)

    def pages(self, id: str) -> list[Page]:
        # Training/eval feed: keep only VALIDATED pages (the layouts `kernsheet
        # detect` generates start status=PENDING) so un-reviewed or REJECTED pages
        # don't leak in until a human has approved them in the editor. Validated
        # pages of a partly-reviewed score are still kept.
        return [page for page in self.score(id).pages if page.validated]

    def score(self, id: str) -> Score:
        return self.kern_sheet.load_score(id)

    def image_path(self, id: str, page_number: int) -> Path:
        return self.kern_sheet.png_path(id, page_number)

    def image(self, id: str, page_number: int) -> Tensor:
        return decode_image(self.image_path(id, page_number).as_posix())

    def records(self, id: str, first_bar: int, last_bar: int) -> list[str] | None:
        return self.kern_sheet.load_tokens(id).get_text(first_bar, last_bar)

    def spine_count(self, id: str) -> int:
        return self.kern_sheet.load_tokens(id).spine_count
