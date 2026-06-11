"""A ``Source`` that interleaves two child Sources at a target fraction.

Rehearsal fine-tuning: train toward a small target corpus (KernSheet) without
catastrophically forgetting the large base corpus (PDMX) by mixing a controlled
fraction of base scores back into the stream. The mix happens here, at the
Source layer, so the datasets/datamodules (split, jitter, bottom-bias sampler)
are reused unchanged and the validation split inherits the same mix.
"""

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

from torch import Tensor

from .layout import Score
from .source import Source


class MixedSource:
    """Interleaves ``primary`` and ``secondary`` Sources at fraction ``mix``.

    ``scores()`` walks ``primary`` once and injects ``secondary`` scores so that
    a ``mix`` fraction of the emitted stream comes from ``secondary`` (cycling
    it, since it is typically far smaller). Score ids are namespaced ``"0::id"``
    / ``"1::id"`` so the id-keyed methods route back to the owning child. The
    ratio is exact at score granularity; per-item (staff) ratios approximate it.
    """

    SEP = "::"

    def __init__(self, primary: Source, secondary: Source, mix: float) -> None:
        assert 0.0 < mix < 1.0, f"mix must be in (0, 1), got {mix}"
        self.children = (primary, secondary)
        self.mix = mix

    def _tag(self, child: int, score: Score) -> Score:
        return replace(score, id=f"{child}{self.SEP}{score.id}")

    def _route(self, id: str) -> tuple[Source, str]:
        head, _, rest = id.partition(self.SEP)
        return self.children[int(head)], rest

    def _cycle(self, source: Source) -> Iterator[Score]:
        """Yield ``source``'s scores forever, restarting when exhausted."""
        while True:
            empty = True
            for score in source.scores():
                empty = False
                yield score
            if empty:
                raise ValueError("MixedSource secondary source yielded no scores")

    def scores(self) -> Iterator[Score]:
        primary, secondary = self.children
        secondary_scores = self._cycle(secondary)
        emitted_secondary = 0
        seen_primary = 0
        ratio = self.mix / (1.0 - self.mix)  # secondary : primary
        for score in primary.scores():
            yield self._tag(0, score)
            seen_primary += 1
            # Catch the secondary stream up to the running target so the emitted
            # fraction tracks `mix` regardless of where the consumer stops.
            target = round(seen_primary * ratio)
            while emitted_secondary < target:
                yield self._tag(1, next(secondary_scores))
                emitted_secondary += 1

    def score(self, id: str) -> Score:
        head, _, rest = id.partition(self.SEP)
        return self._tag(int(head), self.children[int(head)].score(rest))

    def image(self, id: str, page_number: int) -> Tensor:
        child, rest = self._route(id)
        return child.image(rest, page_number)

    def image_path(self, id: str, page_number: int) -> Path:
        child, rest = self._route(id)
        return child.image_path(rest, page_number)

    def records(self, id: str, first_bar: int, last_bar: int) -> list[str] | None:
        child, rest = self._route(id)
        return child.records(rest, first_bar, last_bar)
