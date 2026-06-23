from kern import KernReader
from sheetmusic import Box, Page, Score, Staff, Status, System
from kernsheet.reviews import STAFF_HEIGHT_TOLERANCE_PX, score_findings


def _page(
    heights: list[int],
    *,
    page_number: int = 1,
    status: Status = Status.PENDING,
    reviewed: list[str] | None = None,
) -> Page:
    """A one-system page whose staves have the given pixel heights (top at 0)."""
    staves = [Staff(box=Box(0, 0, 100, h)) for h in heights]
    system = System(bar_numbers=[1], bars=[0, 100], staves=staves)
    return Page(
        page_number=page_number,
        image_width=200,
        image_height=300,
        systems=[system],
        status=status,
        reviewed=reviewed or [],
    )


def _score(page: Page) -> Score:
    return Score(id="x", pages=[page])


class TestStaffHeight:
    def test_equal_heights_no_finding(self) -> None:
        assert score_findings(_score(_page([50, 50]))) == []

    def test_spread_within_tolerance_no_finding(self) -> None:
        page = _page([50, 50 + STAFF_HEIGHT_TOLERANCE_PX])
        assert score_findings(_score(page)) == []

    def test_spread_over_tolerance_flags(self) -> None:
        page = _page([50, 50 + STAFF_HEIGHT_TOLERANCE_PX + 1])
        findings = score_findings(_score(page))
        assert [f.review for f in findings] == ["staff_height"]
        assert findings[0].page_number == 1
        assert findings[0].score_id == "x"

    def test_single_staff_never_flags(self) -> None:
        assert score_findings(_score(_page([50]))) == []

    def test_message_pinpoints_outlier_systems(self) -> None:
        finding = score_findings(_score(_page([50, 59])))[0]
        assert finding.message == "sys0 59px (large), sys0 50px (short)"
        assert finding.system_index == 0


class TestZeroWidthStaff:
    def _page(self, staves: list[Box]) -> Page:
        system = System(
            bar_numbers=[1], bars=[0, 100], staves=[Staff(box=b) for b in staves]
        )
        return Page(
            page_number=1,
            image_width=200,
            image_height=300,
            systems=[system],
            status=Status.PENDING,
            reviewed=[],
        )

    def test_positive_width_no_finding(self) -> None:
        assert score_findings(_score(self._page([Box(0, 0, 100, 50)]))) == []

    def test_collapsed_box_flags(self) -> None:
        findings = score_findings(_score(self._page([Box(50, 0, 50, 60)])))
        assert [f.review for f in findings] == ["zero_width_staff"]
        assert findings[0].system_index == 0

    def test_left_past_page_width_flags(self) -> None:
        findings = score_findings(_score(self._page([Box(200, 0, 260, 60)])))
        assert [f.review for f in findings] == ["zero_width_staff"]

    def test_one_finding_per_system(self) -> None:
        page = self._page([Box(50, 0, 50, 60), Box(50, 60, 50, 120)])
        findings = [
            f for f in score_findings(_score(page)) if f.review == "zero_width_staff"
        ]
        assert len(findings) == 1


class TestBarNumbers:
    def _page(self, bar_numbers: list[int]) -> Page:
        # VALIDATED on purpose: the point of this review is to flag barless systems
        # that slipped through validation (VALIDATED does not suppress findings).
        staves = [Staff(box=Box(0, 0, 100, 50)), Staff(box=Box(0, 60, 100, 110))]
        system = System(bar_numbers=bar_numbers, bars=[0, 100], staves=staves)
        return Page(
            page_number=1,
            image_width=200,
            image_height=300,
            systems=[system],
            status=Status.VALIDATED,
            reviewed=[],
        )

    def test_system_with_bar_numbers_no_finding(self) -> None:
        assert score_findings(_score(self._page([1])), names=["bar_numbers"]) == []

    def test_barless_system_flags(self) -> None:
        findings = score_findings(_score(self._page([])), names=["bar_numbers"])
        assert [f.review for f in findings] == ["bar_numbers"]
        assert findings[0].page_number == 1
        assert findings[0].score_id == "x"


class TestBarDrift:
    @staticmethod
    def _sys(first: int, n_bars: int, *, numbered: bool = True) -> System:
        """A system starting at bar ``first`` with ``n_bars`` complete bars
        (``n_bars + 1`` barlines). ``numbered=False`` seeds an empty bar_numbers."""
        bars = [i * 100 for i in range(n_bars + 1)]
        return System(
            bar_numbers=[first] if numbered else [],
            bars=bars,
            staves=[Staff(box=Box(0, 0, 100, 50))],
        )

    @classmethod
    def _score(cls, *systems: System, page_number: int = 1) -> Score:
        page = Page(
            page_number=page_number,
            image_width=200,
            image_height=300,
            systems=list(systems),
            status=Status.PENDING,
            reviewed=[],
        )
        return Score(id="x", pages=[page])

    def test_contiguous_chain_no_finding(self) -> None:
        # 1(+3)->4(+2)->6(+5)->11
        score = self._score(self._sys(1, 3), self._sys(4, 2), self._sys(6, 5))
        assert score_findings(score, names=["bar_drift"]) == []

    def test_single_system_anchors_no_finding(self) -> None:
        assert score_findings(self._score(self._sys(1, 3)), names=["bar_drift"]) == []

    def test_break_flags_offending_system(self) -> None:
        # second system should start at 4 (1 + 3) but stored 5
        score = self._score(self._sys(1, 3), self._sys(5, 2))
        findings = score_findings(score, names=["bar_drift"])
        assert [f.review for f in findings] == ["bar_drift"]
        assert "expected 4" in findings[0].message

    def test_break_spans_page_boundary(self) -> None:
        # numbering is continuous across pages: p1 ends expecting 4, p2 starts at 6
        p1 = self._score(self._sys(1, 3)).pages[0]
        p2 = self._score(self._sys(6, 2), page_number=2).pages[0]
        score = Score(id="x", pages=[p1, p2])
        findings = score_findings(score, names=["bar_drift"])
        assert [(f.review, f.page_number) for f in findings] == [("bar_drift", 2)]

    def test_barless_system_skipped(self) -> None:
        # the empty system is the bar_numbers review's job; it doesn't break the
        # chain, which continues 1(+3) -> 4.
        score = self._score(
            self._sys(1, 3), self._sys(0, 0, numbered=False), self._sys(4, 2)
        )
        assert score_findings(score, names=["bar_drift"]) == []


class _Kern(KernReader):
    """A KernReader with a fixed bar/spine total, bypassing the token-file load."""

    def __init__(
        self, bar_count: int, *, bar_zero: bool = False, spine_count: int = 2
    ) -> None:
        self._bar_count = bar_count
        self._bar_zero = bar_zero
        self._spine_count = spine_count

    @property
    def bar_count(self) -> int:
        return self._bar_count

    @property
    def spine_count(self) -> int:
        return self._spine_count

    def has_bar_zero(self) -> bool:
        return self._bar_zero


class TestBarCount:
    # _page([..]) is one system of one bar (bars=[0, 100]); the layout geometry must
    # equal the kern's real-measure count directly (the closing barline is not a bar).
    def test_matching_total_no_finding(self) -> None:
        score = _score(_page([50, 50]))
        assert score_findings(score, names=["bar_count"], kern=_Kern(1)) == []

    def test_mismatch_flags_first_page(self) -> None:
        score = _score(_page([50, 50]))
        findings = score_findings(score, names=["bar_count"], kern=_Kern(5))
        assert [f.review for f in findings] == ["bar_count"]
        assert findings[0].page_number == 1
        assert findings[0].score_id == "x"
        assert "layout has 1 bars" in findings[0].message
        assert "kern has 5" in findings[0].message

    def test_bar_zero_does_not_shift_the_count(self) -> None:
        # The count is pure geometry vs kern total — a pickup bar 0 is already in
        # kern.bar_count and does not add a fudge term either way.
        score = _score(_page([50, 50]))
        assert (
            score_findings(score, names=["bar_count"], kern=_Kern(1, bar_zero=True))
            == []
        )

    def test_no_kern_skips(self) -> None:
        # without the token file the check can't run, even on a mismatch.
        assert score_findings(_score(_page([50, 50])), names=["bar_count"]) == []

    def test_empty_kern_skips(self) -> None:
        score = _score(_page([50, 50]))
        assert score_findings(score, names=["bar_count"], kern=_Kern(0)) == []


class TestSpineCount:
    def test_two_spines_no_finding(self) -> None:
        score = _score(_page([50, 50]))
        assert score_findings(score, names=["spine_count"], kern=_Kern(1)) == []

    def test_extra_spine_flags_first_page(self) -> None:
        score = _score(_page([50, 50]))
        findings = score_findings(
            score, names=["spine_count"], kern=_Kern(1, spine_count=3)
        )
        assert [f.review for f in findings] == ["spine_count"]
        assert findings[0].page_number == 1
        assert findings[0].score_id == "x"
        assert "3 spines" in findings[0].message

    def test_no_kern_skips(self) -> None:
        assert score_findings(_score(_page([50, 50])), names=["spine_count"]) == []


class TestSuppression:
    def _bad(
        self, *, status: Status = Status.PENDING, reviewed: list[str] | None = None
    ) -> Page:
        # spread 40 px, always over tolerance
        return _page([40, 80], status=status, reviewed=reviewed)

    def test_acknowledged_review_suppressed(self) -> None:
        page = self._bad(reviewed=["staff_height"])
        assert score_findings(_score(page)) == []

    def test_rejected_page_suppressed(self) -> None:
        assert score_findings(_score(self._bad(status=Status.REJECTED))) == []

    def test_validated_page_still_flags(self) -> None:
        # validation is the human's stamp, not an acknowledgement of open findings.
        assert score_findings(_score(self._bad(status=Status.VALIDATED)))

    def test_names_filter_unknown_returns_nothing(self) -> None:
        assert score_findings(_score(self._bad()), names=[]) == []
