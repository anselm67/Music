"""Automated layout reviews: cheap geometric/consistency checks over a Score that
flag pages needing human attention in the editor.

Findings are recomputed on demand (never persisted), so they never go stale against
the layout they describe. On a flagged page a human either fixes the layout (the
finding then stops firing), marks the page ``REJECTED`` (``Status.REJECTED``), or
acknowledges the review ("ok") — which records its name in ``Page.reviewed`` so it
stops being flagged. Each check reads only the ``Score`` geometry, so scanning the
whole corpus is a cheap pass over the layout JSON (no images, no model).
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass

from sheetmusic import Page, Score, Status

# A within-page spread of stave heights wider than this many pixels is flagged.
# KernSheet GT carries ~3.5px of within-page annotation jitter (see the
# project_ks_gt_height_noise memory), so the floor sits well above the noise: it
# catches real merges/splits, not manual-annotation wobble.
STAFF_HEIGHT_TOLERANCE_PX = 8


@dataclass(frozen=True)
class Finding:
    """One page-level issue raised by a review."""

    review: str
    score_id: str
    page_number: int
    message: str


Review = Callable[[Score], Iterator[Finding]]
REGISTRY: dict[str, Review] = {}


def register(name: str) -> Callable[[Review], Review]:
    """Register a review. A review must only yield findings whose ``page_number``
    belongs to the score it was given (``score_findings`` looks pages up by it)."""

    def wrap(fn: Review) -> Review:
        assert name not in REGISTRY, f"duplicate review {name!r}"
        REGISTRY[name] = fn
        return fn

    return wrap


def review_names() -> list[str]:
    return list(REGISTRY)


@register("staff_height")
def _staff_height(score: Score) -> Iterator[Finding]:
    """Staves within a page should share a height (one grand-staff geometry). A
    large spread means the detector merged or split a staff, or a box is wrong."""
    for page in score.pages:
        heights = [staff.box.height for sys in page.systems for staff in sys.staves]
        if len(heights) < 2:
            continue
        spread = max(heights) - min(heights)
        if spread > STAFF_HEIGHT_TOLERANCE_PX:
            yield Finding(
                "staff_height",
                score.id,
                page.page_number,
                f"stave heights {sorted(heights)} spread {spread}px "
                f"(> {STAFF_HEIGHT_TOLERANCE_PX}px)",
            )


@register("bar_numbers")
def _bar_numbers(score: Score) -> Iterator[Finding]:
    """Every system needs bar numbers — they pin its transcription target. A system
    with none was added in the editor (``_make_system`` seeds ``bar_numbers=[]``) and
    validated before its bars were ever assigned; it cannot be transcribed and crashes
    the noter dataset build, so it should never have been validated as-is."""
    for page in score.pages:
        barless = sum(1 for sys in page.systems if not sys.bar_numbers)
        if barless:
            yield Finding(
                "bar_numbers",
                score.id,
                page.page_number,
                f"{barless} system(s) with no bar numbers",
            )


@register("bar_drift")
def _bar_drift(score: Score) -> Iterator[Finding]:
    """A system's stored start number must continue where the previous system's
    barlines left off: ``first_bar_number == prev.first_bar_number + prev.bar_count``
    (``bar_count`` is geometry, ``len(bars) - 1``). The editor edits barline geometry
    without ever rewriting ``bar_numbers``, so adding/removing a barline silently
    desyncs the stored numbering from the geometry — and the noter/scorer datasets
    slice the kern by the stale ``first_bar_number``, mislabelling the system. Walks
    systems in reading order across the whole score (numbering is continuous over page
    breaks) and flags each system whose stored start breaks the chain. The first
    numbered system anchors the chain (its start is assumed correct); barless systems
    are skipped — they're the ``bar_numbers`` review's job."""
    expected: int | None = None
    for page in score.pages:
        for system in page.systems:
            if system.staff_count == 0 or not system.bar_numbers:
                continue
            if expected is not None and system.first_bar_number != expected:
                yield Finding(
                    "bar_drift",
                    score.id,
                    page.page_number,
                    f"system starts at bar {system.first_bar_number}, expected "
                    f"{expected} from the preceding systems' barline counts "
                    f"(barlines edited without renumbering)",
                )
            expected = system.first_bar_number + system.bar_count


def _needs_attention(page: Page, review: str) -> bool:
    """A finding needs human attention unless the page is already rejected
    (excluded anyway) or a human has acknowledged this review on it."""
    return page.status != Status.REJECTED and review not in page.reviewed


def score_findings(score: Score, names: list[str] | None = None) -> list[Finding]:
    """All un-suppressed findings for one score, for the given reviews (all if None)."""
    selected = names if names is not None else review_names()
    pages = {p.page_number: p for p in score.pages}
    return [
        finding
        for name in selected
        for finding in REGISTRY[name](score)
        if _needs_attention(pages[finding.page_number], name)
    ]
