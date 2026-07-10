#!/usr/bin/env python3
"""End-to-end playback: a scanned score (PDF or images) -> transcription -> MIDI.

Runs the trained ``scorer`` (detect -> crop -> transcribe) over each page, stitches
the per-stave token/articulation output into continuous parts, and renders a MIDI
file via the articulation-aware :mod:`kern.tokens_to_midi` bridge. Optionally plays
or renders the result to audio with ``fluidsynth``.

Supersedes ``scripts/imslp_predict.py``: this is the real end-to-end tool, no longer a
shell-out to ``scorer predict``.

    play score.pdf --play
    play page1.png page2.png -o out.mid
    play score.pdf --wav out.wav
"""

import logging
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import click
import torch
from pdf2image import convert_from_path
from torchvision.io import decode_image
from torchvision.transforms import v2

from kern import Dynamics, Part, render_systems, write_midi
from kernsheet import RENDER_DPI
from noter import Vocab
from sheetmusic import page_transform
from utils import log_uncaught_exceptions

if TYPE_CHECKING:
    from scorer import ScorerModule

KERN_HOME = Path("/home/anselm/datasets/KernSheet")
DEFAULT_SOUNDFONT = Path("/usr/share/sounds/sf2/default-GM.sf2")
TICKS_PER_QUARTER = 480
# Rasterise PDFs at the corpus DPI (kernsheet.RENDER_DPI) so play renders at the exact
# scale the build/png cache was made with. With an antialias=True model the transform
# antialiases the downscale to its canvas, so the DPI is not load-bearing and a raw
# render reads correctly; an antialias=False checkpoint (e.g. tatum-arc-e30) does NOT
# antialias, so it must be retrained under antialias=True to be played reliably.


def load_pages(
    inputs: tuple[Path, ...], page: int | None = None
) -> list[tuple[str, torch.Tensor]]:
    """Expand the CLI inputs into ordered ``(name, RGB CHW uint8 tensor)`` pairs.

    Image files are read with ``decode_image`` — the same call the dataset uses to load
    a corpus page — so play's read path is identical to train/predict. PDFs are
    rasterised with ``pdf2image`` (the KernSheet build's rasteriser) at
    :data:`RENDER_DPI`, each page converted to the same tensor form. The transform's
    antialiased letterbox resize downscales any size to the model canvas, so nothing
    needs resizing here.

    ``page`` (1-based) restricts a PDF to a single page, rendering only that page rather
    than the whole document; it is validated by the caller to a single PDF input.
    """
    pages: list[tuple[str, torch.Tensor]] = []
    for item in inputs:
        if item.suffix.lower() == ".pdf":
            rendered = (
                convert_from_path(str(item), dpi=RENDER_DPI)
                if page is None
                else convert_from_path(
                    str(item), dpi=RENDER_DPI, first_page=page, last_page=page
                )
            )
            for i, image in enumerate(rendered, page or 1):
                pages.append((f"{item.stem}-{i}", v2.functional.pil_to_tensor(image)))
        else:
            pages.append((item.name, decode_image(item.as_posix())))
    return pages


def decode_stave(
    vocab: Vocab, tokens: torch.Tensor, arts: torch.Tensor
) -> list[list[tuple[str, tuple[bool, ...]]]]:
    """One stave's ``(T, mc)`` ids + ``(T, mc, A)`` bits -> a list of timesteps.

    Each timestep is the list of its non-SIL notes as ``(token string, flags)``;
    generation stops at the first EOS row.
    """
    steps: list[list[tuple[str, tuple[bool, ...]]]] = []
    for t in range(tokens.shape[0]):
        if int(tokens[t, 0]) == Vocab.EOS:
            break
        entries: list[tuple[str, tuple[bool, ...]]] = []
        for j in range(tokens.shape[1]):
            tid = int(tokens[t, j])
            if tid in (Vocab.SIL, Vocab.PAD):
                continue
            flags = tuple(bool(x) for x in arts[t, j].tolist())
            entries.append((vocab.decode(tid), flags))
        if entries:
            steps.append(entries)
    return steps


def transcribe_page(
    module: "ScorerModule", vocab: Vocab, image: torch.Tensor
) -> list[list[Part]]:
    """Transcribe one page into its systems, each a list of decoded staves.

    A system's staves are returned top-to-bottom (slot order = instrument); the caller
    concatenates systems across the whole score before rendering, so timing and ties
    are resolved over the full piece.
    """
    boxes, tokens, arts, owners = module.predict(image.unsqueeze(0).to(module.device))
    by_system: dict[int, list[int]] = {}
    for k in range(boxes.shape[0]):
        by_system.setdefault(int(owners[k]), []).append(k)
    return [
        [decode_stave(vocab, tokens[k].cpu(), arts[k].cpu()) for k in by_system[system]]
        for system in sorted(by_system)
    ]


_Row = TypeVar("_Row")


def select_staves(
    systems: list[list[_Row]], staves: tuple[int, ...]
) -> list[list[_Row]]:
    """Keep only the given stave slots (top-to-bottom, 0-based) within each system.

    Slots are the same instrument across systems, so filtering by index plays a
    consistent subset of parts. Indices beyond a system's stave count are ignored.
    """
    return [[row[i] for i in staves if i < len(row)] for row in systems]


def play_or_render(
    midi: Path, soundfont: Path, wav: Path | None, play: bool, gain: float
) -> None:
    """Render the MIDI to ``wav`` and/or play it through ``fluidsynth``."""
    if not soundfont.is_file():
        raise click.ClickException(f"Soundfont not found: {soundfont}")
    gain_args = ["-g", str(gain)]
    if wav is not None:
        subprocess.run(
            [
                "fluidsynth",
                "-ni",
                *gain_args,
                "-F",
                str(wav),
                str(soundfont),
                str(midi),
            ],
            check=True,
        )
        click.echo(f"Rendered audio: {wav}")
    if play:
        # -i (no shell) makes fluidsynth play the file and exit rather than idle.
        subprocess.run(
            ["fluidsynth", "-ni", *gain_args, str(soundfont), str(midi)], check=True
        )


@click.command()
@click.argument(
    "inputs",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--model", default="stravinsky", show_default=True, help="Scorer name.")
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="MIDI output path (default: <first input>.mid).",
)
@click.option("--tempo", type=int, default=90, show_default=True, help="Quarter BPM.")
@click.option(
    "--page",
    type=click.IntRange(1, None),
    default=None,
    help="Play only this page (1-based) from a single-PDF input.",
)
@click.option(
    "--staves",
    "staves_arg",
    default=None,
    help="Comma-separated stave slots to play (0-based, top-down; e.g. '0,1'). "
    "Default: all staves.",
)
@click.option("--play", is_flag=True, default=False, help="Play the result aloud.")
@click.option(
    "--wav",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Also render the audio to this .wav (offline, no speakers needed).",
)
@click.option(
    "--soundfont",
    type=click.Path(dir_okay=False, path_type=Path),
    default=DEFAULT_SOUNDFONT,
    show_default=True,
    help="SoundFont for fluidsynth playback/rendering.",
)
@click.option(
    "--gain",
    type=click.FloatRange(0.0, 10.0),
    default=2.0,
    show_default=True,
    help="fluidsynth output gain (0.0-10.0); scales playback/render volume.",
)
@click.option(
    "--kern-home",
    # Not exists-validated: it is only consulted for the default vocab, so --vocab
    # alone must work on a machine without the corpus.
    type=click.Path(dir_okay=True, file_okay=False, path_type=Path),
    default=KERN_HOME,
    help="Corpus root holding build/vocab.json (default: KernSheet).",
)
@click.option(
    "--vocab",
    "vocab_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Vocab JSON (default: <kern-home>/build/vocab.json).",
)
@click.option(
    "--log-level",
    default="WARNING",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
)
def cli(
    inputs: tuple[Path, ...],
    model: str,
    output: Path | None,
    tempo: int,
    page: int | None,
    staves_arg: str | None,
    play: bool,
    wav: Path | None,
    soundfont: Path,
    gain: float,
    kern_home: Path,
    vocab_path: Path | None,
    log_level: str,
) -> None:
    """Transcribe a score (PDF or page images) and play it back as MIDI.

    INPUTS: one .pdf, or one or more page images (PNG/JPG), in reading order.
    """
    log_uncaught_exceptions()
    logging.basicConfig(level=getattr(logging, log_level.upper()))
    torch.set_float32_matmul_precision("high")

    staves: tuple[int, ...] | None = None
    if staves_arg is not None:
        try:
            staves = tuple(int(s) for s in staves_arg.split(","))
        except ValueError:
            raise click.ClickException(f"Invalid --staves value: {staves_arg!r}")
        if any(s < 0 for s in staves):
            raise click.ClickException("--staves slots must be non-negative.")

    if page is not None and (len(inputs) != 1 or inputs[0].suffix.lower() != ".pdf"):
        raise click.ClickException("--page requires a single PDF input.")

    # Imported here so `--help` stays fast (avoids loading torch-heavy scorer stack).
    from cli.scorer import _load_for_inference

    config, module = _load_for_inference(model)
    vocab_file = vocab_path or kern_home / "build" / "vocab.json"
    if not vocab_file.is_file():
        raise click.ClickException(
            f"Vocab not found: {vocab_file} (pass --vocab or --kern-home)."
        )
    vocab = Vocab.load(vocab_file)
    transform = page_transform(
        config.staffer.image_shape,
        config.staffer.interpolation,
        config.staffer.antialias,
    )

    midi_path = output or inputs[0].with_suffix(".mid")
    pages = load_pages(inputs, page)
    if page is not None and not pages:
        raise click.ClickException(
            f"--page {page} is out of range for {inputs[0].name}."
        )
    click.echo(f"Transcribing {len(pages)} page(s) with '{model}' ...")
    systems: list[list[Part]] = []
    for name, page_image in pages:
        image = transform(page_image).to(module.device)
        page_systems = transcribe_page(module, vocab, image)
        systems.extend(page_systems)
        click.echo(f"  {name}: {sum(len(s) for s in page_systems)} staves")

    if not systems:
        raise click.ClickException("No staves detected — nothing to play.")
    if staves is not None:
        widest = max(len(s) for s in systems)
        systems = select_staves(systems, staves)
        if not any(systems):
            raise click.ClickException(
                f"--staves {staves_arg} selected no staves "
                f"(systems have at most {widest})."
            )
    parts = render_systems(systems, TICKS_PER_QUARTER, Dynamics())
    write_midi(parts, midi_path, tempo=tempo)
    click.echo(f"Wrote MIDI: {midi_path} ({len(parts)} parts)")

    if play or wav is not None:
        play_or_render(midi_path, soundfont, wav, play, gain)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
