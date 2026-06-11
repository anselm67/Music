#!/usr/bin/env python3
"""Tool to migrate and manage the KernSheet dataset.

KernSheet holds real scanned/published piano scores with manually-reviewed layout.
Mirrors the ``pdmx`` CLI but acts on ``~/datasets/KernSheet`` (native ``Score`` layout
under ``layout/``, derived tokens/png under ``build/``).
"""

import logging
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import click

from kernsheet import KernSheet
from kernsheet import migrate as run_migrate
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


@click.command()
@click.option(
    "--write",
    "-w",
    is_flag=True,
    default=False,
    show_default=True,
    help="Write layout/ files (otherwise a dry-run report only).",
)
@click.option("--limit", "-l", type=int, default=0, help="Process at most N scores.")
@click.pass_obj
def migrate(ctx: ClickContext, write: bool, limit: int) -> None:
    """Rewrite legacy envelope layouts into native Score JSON under layout/."""
    ok, skips = run_migrate(ctx.home, write=write, limit=limit)
    print(f"\nmigrated OK: {ok}   skipped: {len(skips)}")
    by_reason: Counter = Counter()
    for s in skips:
        by_reason[s.reason.split("(")[0].split("bar ")[0].strip()] += 1
    for reason, count in by_reason.most_common():
        print(f"  {count:4d}  {reason}")


@click.command()
@click.pass_obj
def make(ctx: ClickContext) -> None:
    """Pre-render the page-image cache (build/png) for all scores, or one ID."""
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
@click.pass_obj
def edit(ctx: ClickContext, prefix: str, edit_all: bool, fast: bool) -> None:
    """Open the layout editor on every score whose catalog key starts with PREFIX
    (all scores when PREFIX is omitted). Validated scores are skipped unless --all
    is given. Press 'h' in the editor for help."""
    from kernsheet.editor import StaffEditor

    ks = ctx.kern_sheet
    # Materialise the worklist up front: editing a score can delete it (or its
    # whole entry) from the catalog, which would otherwise mutate the dict the
    # items() generator is walking. Re-check existence before opening each one.
    worklist = [(key, score.id) for key, score in ks.items(prefix, valid=edit_all)]
    for key, score_id in worklist:
        if not ks.has_score(score_id):
            continue  # deleted earlier this session (the score or its whole entry)
        if not StaffEditor(ks, key, score_id).edit(fast_mode=fast):
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

    Runs the cv2 projection-profile detector on each PDF and writes an UN-validated
    Score under layout/ for review in the editor. Only scores with no existing layout
    (and a usable pdf + write target) are touched; pass PREFIX to restrict to entries
    whose key starts with it. Dry-run by default; pass -w to write.
    """
    from kernsheet import ClassicalStaffer

    ks = ctx.kern_sheet
    staffer = ClassicalStaffer(width=width)
    todo = [
        (key, score)
        for key, entry in ks.catalog.entries.items()
        for score in entry.scores
        if (not prefix or key.startswith(prefix))
        and not ks.layout_path(score).is_file()
        and score.json_path
        and score.pdf_path
        and ks.pdf_path(score).is_file()
    ]
    ok = failed = 0
    for _, score in todo:
        try:
            result = staffer.detect(ks.pdf_path(score), score.id)
        except Exception as e:
            failed += 1
            logging.error(f"detect {score.id}: {e}")
            continue
        print(
            f"  {score.id}: {result.page_count}p "
            f"{result.system_count}sys {result.staff_count}staves"
            + ("" if write else " (dry-run)")
        )
        if write:
            ks.save_score(score.id, result)
        ok += 1
    verb = "written" if write else "detected (dry-run; pass -w to write)"
    print(f"\n{ok} score(s) {verb}, {failed} failed, of {len(todo)} candidate(s).")


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


cli.add_command(migrate)
cli.add_command(make)
cli.add_command(edit)
cli.add_command(detect)
cli.add_command(stats)
cli.add_command(check)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
