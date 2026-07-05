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

    def spine_count(self, id: str) -> int:
        """Number of tab-separated token spines (columns) in the score's token file.

        Well-formed data has one spine per staff; a score whose count exceeds a
        system's staff count has a voiced staff (or more staves than the scan shows),
        and the noter dataset would silently drop the extra spine — see the
        ``spine_count`` guard in ``NoterDataset``.
        """
        ...

    def staff_map(self, id: str) -> list[int]:
        """Token column index for each staff of the score, top staff first.

        From the Humdrum ``*staffN`` row when present, else the positional bass-first
        fallback (``reversed(range(spine_count))``). The noter zips this against a
        system's ``staff_boxes`` (also top-to-bottom) to route each staff to its
        token column. See :meth:`KernReader.staff_map`.
        """
        ...
