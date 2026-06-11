"""Tests for the rehearsal MixedSource interleaver/router."""

from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from torch import Tensor

from sheetmusic import MixedSource, Score


class FakeSource:
    """Minimal Source whose id-keyed methods echo their inputs, so a caller can
    assert which child (and which stripped id) a MixedSource routed to."""

    def __init__(self, tag: str, n: int) -> None:
        self.tag = tag
        self.n = n

    def scores(self) -> Iterator[Score]:
        for i in range(self.n):
            yield Score(id=f"{self.tag}-{i}", pages=[])

    def score(self, id: str) -> Score:
        return Score(id=id, pages=[])

    def image(self, id: str, page_number: int) -> Tensor:
        return torch.tensor([float(page_number)])

    def image_path(self, id: str, page_number: int) -> Path:
        return Path(self.tag) / id / str(page_number)

    def records(self, id: str, first_bar: int, last_bar: int) -> list[str] | None:
        return [f"{self.tag}:{id}:{first_bar}:{last_bar}"]


def _heads(scores: list[Score]) -> Counter[str]:
    return Counter(s.id.split(MixedSource.SEP, 1)[0] for s in scores)


def test_scores_are_namespaced_by_child() -> None:
    mixed = MixedSource(FakeSource("pdmx", 4), FakeSource("ks", 4), mix=0.5)
    ids = [s.id for s in mixed.scores()]
    assert all(i.startswith(("0::", "1::")) for i in ids)


def test_primary_walked_once_secondary_cycles() -> None:
    # primary=10 once, secondary=2 oversampled to hold the 0.5 share.
    mixed = MixedSource(FakeSource("pdmx", 10), FakeSource("ks", 2), mix=0.5)
    scores = list(mixed.scores())
    heads = _heads(scores)
    assert heads["0"] == 10  # every primary score, exactly once
    sec_ids = [s.id for s in scores if s.id.startswith("1::")]
    assert len(sec_ids) == 10  # oversampled to match
    assert max(Counter(sec_ids).values()) > 1  # ...by cycling (repeats)


@pytest.mark.parametrize("mix", [0.25, 0.5, 0.75])
def test_emitted_fraction_tracks_mix(mix: float) -> None:
    mixed = MixedSource(FakeSource("pdmx", 400), FakeSource("ks", 7), mix=mix)
    heads = _heads(list(mixed.scores()))
    frac = heads["1"] / (heads["0"] + heads["1"])
    assert abs(frac - mix) < 0.01


def test_records_route_and_strip_prefix() -> None:
    mixed = MixedSource(FakeSource("pdmx", 2), FakeSource("ks", 2), mix=0.5)
    assert mixed.records("0::pdmx-1", 3, 4) == ["pdmx:pdmx-1:3:4"]
    assert mixed.records("1::ks-0", 5, 6) == ["ks:ks-0:5:6"]


def test_image_path_routes_to_owning_child() -> None:
    mixed = MixedSource(FakeSource("pdmx", 2), FakeSource("ks", 2), mix=0.5)
    assert mixed.image_path("1::ks-1", 2) == Path("ks") / "ks-1" / "2"


def test_score_reload_stays_namespaced() -> None:
    # The returned Score must keep its prefix so any later use still routes.
    mixed = MixedSource(FakeSource("pdmx", 2), FakeSource("ks", 2), mix=0.5)
    assert mixed.score("1::ks-0").id == "1::ks-0"


def test_mix_out_of_range_rejected() -> None:
    with pytest.raises(AssertionError):
        MixedSource(FakeSource("pdmx", 2), FakeSource("ks", 2), mix=0.0)
    with pytest.raises(AssertionError):
        MixedSource(FakeSource("pdmx", 2), FakeSource("ks", 2), mix=1.0)


def test_empty_secondary_raises() -> None:
    mixed = MixedSource(FakeSource("pdmx", 4), FakeSource("ks", 0), mix=0.5)
    with pytest.raises(ValueError, match="yielded no scores"):
        list(mixed.scores())
