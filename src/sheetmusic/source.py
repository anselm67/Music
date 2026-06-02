"""The ``Source`` protocol: the data contract every dataset backend implements.

A ``Source`` yields exactly what the ``staffer`` / ``noter`` / ``scorer`` datasets need,
hiding each dataset's on-disk layout behind a uniform interface. PDMX and KernSheet each
provide an implementation (``PdmxSource`` / ``KernSheetSource``).

The vocabulary is intentionally *not* part of this protocol: a single shared vocab is
built once and passed to the datasets alongside the source.
"""

from typing import Iterable, Protocol

from torch import Tensor

from .layout import Score


class Source(Protocol):
    def scores(self) -> Iterable[Score]:
        """Enumerate the dataset's scores (layout included)."""
        ...

    def image(self, score: Score, page_number: int) -> Tensor:
        """Raw page image ``(C, H, W)`` in original pixels; the dataset transforms."""
        ...

    def records(self, score: Score, first_bar: int, last_bar: int) -> list[str] | None:
        """Kern rows for the bar range, spines tab-separated; None if unavailable."""
        ...
