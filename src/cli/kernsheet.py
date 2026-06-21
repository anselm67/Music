#!/usr/bin/env python3
"""Tool to manage the KernSheet dataset.

KernSheet holds real scanned/published piano scores with manually-reviewed layout.
Mirrors the ``pdmx`` CLI but acts on ``~/datasets/KernSheet`` (native ``Score`` layout
under ``layout/``, derived tokens/png under ``build/``).
"""

import logging
import random
import sys
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import click

from kern import KernReader
from kernsheet import KernScore, KernSheet, review_names, score_findings
from kernsheet.reviews import Finding
from utils import log_uncaught_exceptions, print_histogram

HOME = Path("/home/anselm/datasets/KernSheet")


@dataclass
class ClickContext:
    home: Path
    kern_sheet: KernSheet


@click.group()
@click.option(
    "--home",
    "-h",
    type=click.Path(
        dir_okay=True, file_okay=False, exists=True, readable=True, path_type=Path
    ),
    default=HOME,
    show_default=True,
)
@click.option(
    "--log-file",
    type=click.Path(file_okay=True, writable=True, path_type=Path),
    help="Name of kernsheet's log file.",
)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
)
@click.option(
    "--no-excepthook",
    is_flag=True,
    default=False,
    help="Don't route uncaught exceptions through the logger; print the "
    "default traceback to stderr instead.",
)
@click.pass_context
def cli(
    ctx: click.Context,
    home: Path,
    log_file: None | Path,
    log_level: str,
    no_excepthook: bool,
) -> None:
    if not no_excepthook:
        log_uncaught_exceptions()
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        filename=log_file,
        format="%(asctime)s | %(levelname)s | %(module)s.%(funcName)s:%(lineno)d | %(message)s",  # noqa: E501
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.info("Running: %s", " ".join(sys.argv))
    kern_sheet = KernSheet(home)
    ctx.obj = ClickContext(home=home, kern_sheet=kern_sheet)


def _resolve_reviews(review: str | None) -> list[str] | None:
    """Validate a ``--review`` name (None = all registered reviews)."""
    if review is None:
        return None
    if review not in review_names():
        raise click.BadParameter(
            f"unknown review {review!r}; known: {', '.join(review_names())}"
        )
    return [review]


def _scan(
    ks: KernSheet, prefix: str, names: list[str] | None
) -> "list[tuple[str, Finding]]":
    """All (catalog key, Finding) pairs across the corpus for the given reviews."""
    out: list[tuple[str, Finding]] = []
    for key, kern_score in ks.items(prefix, valid=True):
        try:
            score = ks.load_score(kern_score.id)
        except Exception as e:
            logging.error(f"review load {kern_score.id}: {e}")
            continue
        # The bar_count review needs the kern total; other reviews ignore it. An
        # un-built or malformed token file just runs the rest without bar_count,
        # rather than crashing the whole corpus scan on one bad entry.
        kern = None
        tokens = ks.tokens_path(key)
        if tokens.is_file():
            try:
                kern = KernReader(tokens)
            except Exception as e:
                logging.error(f"review kern {kern_score.id}: {e}")
        for finding in score_findings(score, names, kern=kern):
            out.append((key, finding))
    return out


def _review_worklist(
    ks: KernSheet, prefix: str, names: list[str] | None
) -> list[tuple[str, str, int | None]]:
    """One (key, score_id, first-flagged-page) entry per flagged score."""
    seen: dict[str, tuple[str, str, int | None]] = {}
    for key, finding in _scan(ks, prefix, names):
        prev = seen.get(finding.score_id)
        if prev is None or (prev[2] is not None and finding.page_number < prev[2]):
            seen[finding.score_id] = (key, finding.score_id, finding.page_number)
    return list(seen.values())


@click.command()
@click.pass_obj
def make(ctx: ClickContext) -> None:
    """Build the derived cache for every entry: the kern token file (build/tokens,
    tokenised from the .krn) and the per-page images (build/png). Existing outputs
    are left as-is, so it only fills in what's missing."""
    ctx.kern_sheet.make()


@click.command()
@click.argument("prefix", type=str, required=False, default="")
@click.option(
    "--all",
    "edit_all",
    is_flag=True,
    default=False,
    help="Edit all scores, even validated ones.",
)
@click.option(
    "--fast",
    "-f",
    is_flag=True,
    default=False,
    help="Fast mode: auto-save and validate pages on jumps.",
)
@click.option(
    "--review",
    "review",
    type=str,
    default=None,
    help="Only open scores that a review flags, landing on the flagged page. "
    f"One of: {', '.join(review_names())}.",
)
@click.option(
    "--random",
    "shuffle",
    is_flag=True,
    default=False,
    help="Walk the scores in random order (for spot-checking the corpus).",
)
@click.pass_obj
def edit(
    ctx: ClickContext,
    prefix: str,
    edit_all: bool,
    fast: bool,
    review: str | None,
    shuffle: bool,
) -> None:
    """Open the layout editor on every score whose catalog key starts with PREFIX
    (all scores when PREFIX is omitted). Validated scores are skipped unless --all
    is given. With --review, only scores a review flags are opened, on the flagged
    page. With --random, the scores are walked in random order. Press '?' in the
    editor for help."""
    from kernsheet.editor import StaffEditor

    ks = ctx.kern_sheet
    names = _resolve_reviews(review)
    # Materialise the worklist up front: editing a score can delete it (or its
    # whole entry) from the catalog, which would otherwise mutate the dict the
    # items() generator is walking. Re-check existence before opening each one.
    if review:
        worklist = _review_worklist(ks, prefix, names)
        print(f"{len(worklist)} score(s) flagged by {review}.")
    else:
        worklist = [
            (key, score.id, None) for key, score in ks.items(prefix, valid=edit_all)
        ]
    if shuffle:
        random.shuffle(worklist)
    for i, (key, score_id, start_page) in enumerate(worklist, 1):
        if not ks.has_score(score_id):
            continue  # deleted earlier this session (the score or its whole entry)
        if not ks.tokens_path(key).is_file():
            # The editor needs the kern token file (bar counts / 'k' / renumber);
            # skip rather than crash the whole walk on an un-built score.
            click.echo(f"skipping {score_id}: no tokens — run `kernsheet make`")
            continue
        if not StaffEditor(ks, key, score_id).edit(
            fast_mode=fast,
            start_page_number=start_page,
            review_names=names if review else None,
            review_progress=(i, len(worklist)) if review else None,
        ):
            return


@click.command()
@click.argument("prefix", type=str, required=False, default="")
@click.option(
    "--write",
    "-w",
    is_flag=True,
    default=False,
    show_default=True,
    help="Write layout/ files (otherwise a dry-run report only).",
)
@click.option(
    "--width",
    type=int,
    default=1200,
    show_default=True,
    help="Render/processing width for the detector.",
)
@click.pass_obj
def detect(ctx: ClickContext, prefix: str, write: bool, width: int) -> None:
    """Generate a layout via ClassicalStaffer for every catalog score that has none.

    Runs the cv2 projection-profile detector once per source PDF (a PDF shared by
    several entries — an all-in-one edition — is detected once and the layout cloned
    to each score) and writes an UN-validated Score under layout/ for review in the
    editor. Only scores with no existing layout (and a usable pdf + write target) are
    touched; pass PREFIX to restrict to entries whose key starts with it. Dry-run by
    default; pass -w to write.
    """
    from kernsheet import ClassicalStaffer

    ks = ctx.kern_sheet
    staffer = ClassicalStaffer(width=width)
    # Group layout-less scores by source pdf: a pdf shared by many entries yields
    # identical geometry for every score, so detect it once and clone the result
    # (only the embedded id differs) instead of re-running detection per score.
    todo: dict[Path, list[KernScore]] = {}
    for key, entry in ks.catalog.entries.items():
        for score in entry.scores:
            if (
                (not prefix or key.startswith(prefix))
                and not ks.layout_path(score).is_file()
                and score.json_path
                and score.pdf_path
                and ks.pdf_path(score).is_file()
            ):
                todo.setdefault(ks.pdf_path(score), []).append(score)
    candidates = sum(len(scores) for scores in todo.values())
    ok = failed = 0
    for pdf_path, scores in todo.items():
        try:
            base = staffer.detect(pdf_path, scores[0].id)
        except Exception as e:
            failed += len(scores)
            logging.error(f"detect {pdf_path}: {e}")
            continue
        for i, score in enumerate(scores):
            result = base if i == 0 else replace(base, id=score.id)
            print(
                f"  {score.id}: {result.page_count}p "
                f"{result.system_count}sys {result.staff_count}staves"
                + (" (shared pdf)" if i else "")
                + ("" if write else " (dry-run)")
            )
            if write:
                ks.save_score(score.id, result)
            ok += 1
        # png paths are keyed by pdf stem + page, so a shared pdf renders once.
        if write:
            ks.rebuild_images(scores[0], base)
    verb = "written" if write else "detected (dry-run; pass -w to write)"
    print(f"\n{ok} score(s) {verb}, {failed} failed, of {candidates} candidate(s).")


@click.command()
@click.pass_obj
def stats(ctx: ClickContext) -> None:
    """Layout statistics over the migrated KernSheet scores."""
    score_count = page_count = system_count = stave_count = bar_count = 0
    score_failed = valid_page_count = 0
    systems_per_page: Counter = Counter()
    staves_per_system: Counter = Counter()
    bars_per_system: Counter = Counter()
    for _, kern_score in ctx.kern_sheet.items():
        try:
            score = ctx.kern_sheet.load_score(kern_score.id)
        except Exception as e:
            score_failed += 1
            logging.error(f"Failed to load score {kern_score.id}: {e}")
            continue
        score_count += 1
        for page in score.pages:
            page_count += 1
            if page.validated:
                valid_page_count += 1
            systems_per_page[len(page.systems)] += 1
            for system in page.systems:
                system_count += 1
                stave_count += len(system.staves)
                staves_per_system[len(system.staves)] += 1
                bars_per_system[len(system.bar_numbers)] += 1
                bar_count += len(system.bar_numbers)

    print(f"{score_count:,} scores - {score_failed} didn't load:")
    print(f"  Page count: {page_count:,}")
    print(f" Valid pages: {valid_page_count:,} ({page_count - valid_page_count:,} not)")
    print(f"System count: {system_count:,}")
    print(f" Staff count: {stave_count:,}")
    print(f"   Bar count: {bar_count:,}")
    print_histogram(systems_per_page, title="Systems per page:")
    print_histogram(staves_per_system, title="Staves per system:")
    print_histogram(bars_per_system, title="Bars per system:")


@click.command()
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Print each issue as it is found.",
)
@click.pass_obj
def check(ctx: ClickContext, verbose: bool) -> None:
    """Validate catalog integrity against the filesystem."""
    ctx.kern_sheet.check(verbose=verbose)


@click.command()
@click.argument("prefix", type=str, required=False, default="")
@click.option(
    "--review",
    "review",
    type=str,
    default=None,
    help=f"Run one review; default all. One of: {', '.join(review_names())}.",
)
@click.pass_obj
def review(ctx: ClickContext, prefix: str, review: str | None) -> None:
    """Report layout-review findings across the corpus (read-only).

    Recomputes every registered review (or just --review NAME) over each score's
    layout and lists the pages needing attention. Use `edit --review NAME` to walk
    and fix them."""
    names = _resolve_reviews(review)
    findings = _scan(ctx.kern_sheet, prefix, names)
    by_review: dict[str, list[Finding]] = {}
    for _, finding in findings:
        by_review.setdefault(finding.review, []).append(finding)
    for name in sorted(by_review):
        items = by_review[name]
        print(f"\n{name}: {len(items)} page(s)")
        for f in items:
            print(f"  {f.score_id}  page {f.page_number}  {f.message}")
    flagged = len({f.score_id for _, f in findings})
    print(f"\n{len(findings)} finding(s) across {flagged} score(s).")


cli.add_command(make)
cli.add_command(edit)
cli.add_command(detect)
cli.add_command(stats)
cli.add_command(check)
cli.add_command(review)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
