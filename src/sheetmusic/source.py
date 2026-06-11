"""The ``Source`` protocol: the data contract every dataset backend implements.

A ``Source`` yields exactly what the ``staffer`` / ``noter`` / ``scorer`` datasets need,
hiding each dataset's on-disk layout behind a uniform interface. PDMX and KernSheet each
provide an implementation (``PdmxSource`` / ``KernSheetSource``).

The vocabulary is intentionally *not* part of this protocol: a single shared vocab is
built once and passed to the datasets alongside the source.
"""

from pathlib import Path
from typing import Iterable, Protocol

from torch import Tensor

from .layout import Page, Score


class Source(Protocol):
    def scores(self) -> Iterable[Score]:
        """Enumerate the dataset's scores (layout included). Frugal: yield lazily."""
        ...

    def pages(self, id: str) -> list[Page]:
        """The score's pages eligible for training/eval.

        Lets a source withhold pages the datasets must not learn from: KernSheet drops
        pages still awaiting human validation (e.g. ``kernsheet detect`` output), PDMX
        returns every page. Datasets enumerate pages through this, not ``Score.pages``.
        """
        ...

    def score(self, id: str) -> Score:
        """(Re)load one score's layout by its ``Score.id`` key.

        Used per ``__getitem__`` so datasets can keep only lightweight keys in memory
        rather than holding every parsed ``Score``.
        """
        ...

    def image(self, id: str, page_number: int) -> Tensor:
        """Raw page image ``(C, H, W)`` in original pixels; the dataset transforms."""
        ...

    def image_path(self, id: str, page_number: int) -> Path:
        """Path to the on-disk PNG for this page."""
        ...

    def records(self, id: str, first_bar: int, last_bar: int) -> list[str] | None:
        """Kern rows for the bar range, spines tab-separated; None if unavailable."""
        ...
