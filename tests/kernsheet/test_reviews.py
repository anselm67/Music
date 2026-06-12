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
