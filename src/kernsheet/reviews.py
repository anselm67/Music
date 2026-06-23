"""Automated layout reviews: cheap geometric/consistency checks over a Score that
flag pages needing human attention in the editor.

Findings are recomputed on demand (never persisted), so they never go stale against
the layout they describe. On a flagged page a human either fixes the layout (the
finding then stops firing), marks the page ``REJECTED`` (``Status.REJECTED``), or
acknowledges the review ("ok") — which records its name in ``Page.reviewed`` so it
stops being flagged. Each check reads only the ``Score`` geometry, so scanning the
whole corpus is a cheap pass over the layout JSON (plus, for ``bar_count``, the
small kern token file — still no images, no model).
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass

from kern import KernReader
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
    # The system the finding pins, when it has one (e.g. the outlier staff for
    # ``staff_height``), so the editor can land selection on it; None = page-level.
    system_index: int | None = None


# A review reads the score geometry and, when one is available, the kern token file
# (``bar_count`` needs the expected total; the geometry-only reviews ignore it).
Review = Callable[[Score, KernReader | None], Iterator[Finding]]
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
def _staff_height(score: Score, _kern: KernReader | None) -> Iterator[Finding]:
    """Staves within a page should share a height (one grand-staff geometry). A
    large spread means the detector merged or split a staff, or a box is wrong."""
    for page in score.pages:
        staves = [
            (si, staff.box.height)
            for si, sys in enumerate(page.systems)
            for staff in sys.staves
        ]
        if len(staves) < 2:
            continue
        heights = [h for _, h in staves]
        lo, hi = min(heights), max(heights)
        spread = hi - lo
        if spread > STAFF_HEIGHT_TOLERANCE_PX:
            tall_sys = next(si for si, h in staves if h == hi)
            short_sys = next(si for si, h in staves if h == lo)
            yield Finding(
                "staff_height",
                score.id,
                page.page_number,
                f"sys{tall_sys} {hi}px (large), sys{short_sys} {lo}px (short)",
                system_index=tall_sys,
            )


@register("zero_width_staff")
def _zero_width_staff(score: Score, _kern: KernReader | None) -> Iterator[Finding]:
    """A staff crops to zero width — and crashes the noter encoder
    (``to_padded_tensor: ... non-zero numel``) — when its system's horizontal span is
    degenerate. Staff x is derived from the system's ``bars`` (``bars[0]..bars[-1]``),
    so the crop collapses two ways: a degenerate bar span (``bars[0] == bars[-1]``, a
    single barline with no enclosed bar) or bars that sit entirely past the page's
    right edge (``bars[0] >= image_width``, so ``image_width - left <= 0``). Flag the
    offending system so its bars are fixed or the page rejected. (A system with no bars
    at all is the ``bar_numbers`` review's job and is left to it.)"""
    for page in score.pages:
        for si, sys in enumerate(page.systems):
            if not sys.bars:
                continue
            if sys.bars[-1] - sys.bars[0] <= 0:
                reason = f"bars span {sys.bars[0]}..{sys.bars[-1]} is degenerate"
            elif sys.bars[0] >= page.image_width:
                reason = f"bars start {sys.bars[0]} past page width {page.image_width}"
            else:
                continue
            yield Finding(
                "zero_width_staff",
                score.id,
                page.page_number,
                f"sys{si} {reason} — staves crop to zero width, crash the noter",
                system_index=si,
            )


@register("bar_numbers")
def _bar_numbers(score: Score, _kern: KernReader | None) -> Iterator[Finding]:
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
def _bar_drift(score: Score, _kern: KernReader | None) -> Iterator[Finding]:
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


@register("bar_count")
def _bar_count(score: Score, kern: KernReader | None) -> Iterator[Finding]:
    """The page geometry must account for every bar in the kern: summing each
    system's barline count across the whole score must equal ``kern.bar_count``.
    This is exactly the editor's '/' check (``StaffEditor.check_bar_count``), run as
    a corpus review — a mismatch means barlines are un-split, merged, or spurious, so
    the kern can't be sliced onto the systems. Needs the kern token file, so it is
    skipped when one isn't available or is empty (un-built / corrupt). The whole-score
    total is reported against the first page, where a recount starts."""
    if kern is None or kern.bar_count == 0 or not score.pages:
        return
    # Mirrors check_bar_count: kern.bar_count is the real-measure count (a pickup `=0`
    # counts, the closing `==` does not), which equals the layout's barline geometry.
    count = sum(
        max(system.bar_count, 0)
        for page in score.pages
        for system in page.systems
        if system.staff_count > 0
    )
    if count != kern.bar_count:
        yield Finding(
            "bar_count",
            score.id,
            score.pages[0].page_number,
            f"layout has {count} bars, kern has {kern.bar_count} "
            f"(barlines added/removed without matching the kern)",
        )


# The noter dataset maps one token spine per staff, so a 2-staff scan needs exactly
# two spines; more means a spine is silently dropped.
SPINE_COUNT_MAX = 2


@register("spine_count")
def _spine_count(score: Score, kern: KernReader | None) -> Iterator[Finding]:
    """The noter dataset routes one token spine per staff (``spine_numbers`` from the
    system's staff count), so a token file with more spines than staves silently drops
    the extra spine(s) — the staff then trains against a partial/wrong transcription.
    It happens two ways: a voiced staff (two ``**kern`` spines sharing one printed
    staff) or a krn that declares more staves than the 2-staff scan. Either way the
    score isn't trainable as-is; flag it (reported on its first page) so it surfaces in
    the review worklist instead of only at train time."""
    if kern is None or not score.pages:
        return
    spines = kern.spine_count
    if spines > SPINE_COUNT_MAX:
        yield Finding(
            "spine_count",
            score.id,
            score.pages[0].page_number,
            f"token file has {spines} spines (> {SPINE_COUNT_MAX}); the noter dataset "
            f"maps one spine per staff and drops the rest (voiced staff, or a krn with "
            f"more staves than the 2-staff scan)",
        )


def _needs_attention(page: Page, review: str) -> bool:
    """A finding needs human attention unless the page is already rejected
    (excluded anyway) or a human has acknowledged this review on it."""
    return page.status != Status.REJECTED and review not in page.reviewed


def score_findings(
    score: Score,
    names: list[str] | None = None,
    kern: KernReader | None = None,
) -> list[Finding]:
    """All un-suppressed findings for one score, for the given reviews (all if None).
    ``kern`` is the score's token file when available; ``bar_count`` needs it and is
    silently skipped without it (the geometry-only reviews ignore it)."""
    selected = names if names is not None else review_names()
    pages = {p.page_number: p for p in score.pages}
    return [
        finding
        for name in selected
        for finding in REGISTRY[name](score, kern)
        if _needs_attention(pages[finding.page_number], name)
    ]
