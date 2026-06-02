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

from kernsheet import KernSheetSource
from kernsheet import migrate as run_migrate
from utils import print_histogram

HOME = Path("/home/anselm/datasets/KernSheet")


@dataclass
class ClickContext:
    home: Path
    source: KernSheetSource


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
@click.pass_context
def cli(ctx: click.Context, home: Path, log_file: None | Path, log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        filename=log_file,
        format="%(asctime)s | %(levelname)s | %(module)s.%(funcName)s:%(lineno)d | %(message)s",  # noqa: E501
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.info("Running: %s", " ".join(sys.argv))
    ctx.obj = ClickContext(home=home, source=KernSheetSource(home))


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
@click.argument("id", required=False, default=None)
@click.pass_obj
def make(ctx: ClickContext, id: str | None) -> None:
    """Pre-render the page-image cache (build/png) for all scores, or one ID."""
    source = ctx.source
    scores = [source.score(id)] if id else source.scores()
    rendered = 0
    for score in scores:
        for page in score.pages:
            try:
                source.image(score.id, page.page_number)
                rendered += 1
            except Exception as e:
                logging.error("%s page %d: %s", score.id, page.page_number, e)
    print(f"rendered {rendered} page image(s).")


@click.command()
@click.argument("id")
@click.option(
    "--fast",
    "-f",
    is_flag=True,
    default=False,
    help="Fast mode: auto-save and validate pages on jumps.",
)
@click.pass_obj
def edit(ctx: ClickContext, id: str, fast: bool) -> None:
    """Open the interactive layout editor on score ID (press 'h' for help)."""
    from kernsheet.editor import StaffEditor
    from kernsheet.editor_backend import EditorBackend

    if not (ctx.home / "layout" / f"{id}.json").exists():
        raise click.ClickException(f"no migrated layout for {id!r}")
    StaffEditor(EditorBackend(ctx.home, id)).edit(fast_mode=fast)


@click.command()
@click.pass_obj
def stats(ctx: ClickContext) -> None:
    """Layout statistics over the migrated KernSheet scores."""
    scores = pages = systems = staves = bars = 0
    systems_per_page: Counter = Counter()
    staves_per_system: Counter = Counter()
    bars_per_system: Counter = Counter()
    for score in ctx.source.scores():
        scores += 1
        for page in score.pages:
            pages += 1
            systems_per_page[len(page.systems)] += 1
            for system in page.systems:
                systems += 1
                staves += len(system.staves)
                staves_per_system[len(system.staves)] += 1
                bars_per_system[len(system.bar_numbers)] += 1
                bars += len(system.bar_numbers)

    print(f"{scores:,} scores:")
    print(f"  Page count: {pages:,}")
    print(f"System count: {systems:,}")
    print(f" Staff count: {staves:,}")
    print(f"   Bar count: {bars:,}")
    print_histogram(systems_per_page, title="Systems per page:")
    print_histogram(staves_per_system, title="Staves per system:")
    print_histogram(bars_per_system, title="Bars per system:")


cli.add_command(migrate)
cli.add_command(make)
cli.add_command(edit)
cli.add_command(stats)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
