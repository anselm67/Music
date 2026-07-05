#!/usr/bin/env python3
"""End-to-end playback: a scanned score (PDF or images) -> transcription -> MIDI.

Runs the trained ``scorer`` (detect -> crop -> transcribe) over each page, stitches
the per-stave token/articulation output into continuous parts, and renders a MIDI
file via the articulation-aware :func:`kern.parts_to_midi` bridge. Optionally plays
or renders the result to audio with ``fluidsynth``.

Supersedes ``scripts/imslp_predict.py``: this is the real end-to-end tool, no longer a
shell-out to ``scorer predict``.

    play score.pdf --play
    play page1.png page2.png -o out.mid
    play score.pdf --wav out.wav
"""

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import click
import cv2
import torch
from torchvision.transforms import v2

from kern import Dynamics, Part, parts_to_midi
from noter import Vocab
from sheetmusic import LetterboxResize, PerImageNormalize
from utils import log_uncaught_exceptions

if TYPE_CHECKING:
    from scorer import ScorerConfig, ScorerModule

HOME = Path("/home/anselm/datasets/PDMX")
DEFAULT_SOUNDFONT = Path("/usr/share/sounds/sf2/default-GM.sf2")


def rasterize_pdf(pdf: Path, out_dir: Path, dpi: int) -> list[Path]:
    """Rasterise every page of ``pdf`` to PNG with ``pdftoppm``."""
    prefix = out_dir / pdf.stem
    subprocess.run(
        ["pdftoppm", "-png", "-r", str(dpi), str(pdf), str(prefix)], check=True
    )
    pages = sorted(out_dir.glob(f"{pdf.stem}-*.png"))
    if not pages:
        raise click.ClickException(f"pdftoppm produced no pages from {pdf}")
    return pages


def resolve_pages(inputs: tuple[Path, ...], out_dir: Path, dpi: int) -> list[Path]:
    """Expand the CLI inputs into a flat, ordered list of page PNGs."""
    pages: list[Path] = []
    for item in inputs:
        if item.suffix.lower() == ".pdf":
            pages.extend(rasterize_pdf(item, out_dir, dpi))
        else:
            pages.append(item)
    return pages


def build_transform(config: "ScorerConfig") -> v2.Compose:
    """The page transform, identical to ScorerDataset's (per-image normalised)."""
    staffer = config.staffer
    return v2.Compose(
        [
            v2.Grayscale(),
            LetterboxResize(
                staffer.image_shape,
                interpolation=staffer.interpolation,
                antialias=staffer.antialias,
                fill=255,
            ),
            v2.ToDtype(torch.float, scale=True),
            PerImageNormalize(),
        ]
    )


def load_page(path: Path, width: int, transform: v2.Compose) -> torch.Tensor:
    """Decode a page and downscale it to the training width before the transform.

    Not about pixel count — a controlled test shows the model reads the KS render
    correctly whether it's 1200px or upscaled to 2481px. It's about stroke
    sharpness: kernsheet builds build/png by ``cv2.resize(INTER_AREA)`` down to
    width 1200, which softens/thickens the staff lines, and that is what the
    scorer trained on. A native high-res render (e.g. a 300-DPI pdftoppm page at
    ~2481px) has hair-thin sharp lines the model never saw, so the staffer
    mis-registers the staff by a line and every pitch reads a third off.

    Uses the exact same ``cv2.resize(INTER_AREA)`` to ``width`` as the kernsheet
    build (``KernSheet._transform``), then hands the transform an RGB CHW tensor —
    the same thing ``decode_image`` yields when the dataset reads that build PNG.
    """
    bgr = cv2.imread(path.as_posix(), cv2.IMREAD_COLOR)
    if bgr is None:
        raise click.ClickException(f"Could not read image: {path}")
    h, w = bgr.shape[:2]
    if w > width:  # only downscale; INTER_AREA degrades to nearest on upscale
        bgr = cv2.resize(bgr, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return transform(torch.from_numpy(rgb).permute(2, 0, 1).contiguous())


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
    module: "ScorerModule", vocab: Vocab, image: torch.Tensor, parts: dict[int, Part]
) -> int:
    """Transcribe one page, appending each stave's timesteps to its part.

    Staves at the same top-to-bottom slot across systems (and pages) are the same
    instrument, so they concatenate into one part. Returns the stave count.
    """
    boxes, tokens, arts, owners = module.predict(image.unsqueeze(0).to(module.device))
    by_system: dict[int, list[int]] = {}
    for k in range(boxes.shape[0]):
        by_system.setdefault(int(owners[k]), []).append(k)
    for system in sorted(by_system):
        for slot, k in enumerate(by_system[system]):
            steps = decode_stave(vocab, tokens[k].cpu(), arts[k].cpu())
            parts.setdefault(slot, []).extend(steps)
    return int(boxes.shape[0])


def play_or_render(midi: Path, soundfont: Path, wav: Path | None, play: bool) -> None:
    """Render the MIDI to ``wav`` and/or play it through ``fluidsynth``."""
    if not soundfont.is_file():
        raise click.ClickException(f"Soundfont not found: {soundfont}")
    if wav is not None:
        subprocess.run(
            ["fluidsynth", "-ni", "-F", str(wav), str(soundfont), str(midi)],
            check=True,
        )
        click.echo(f"Rendered audio: {wav}")
    if play:
        # -i (no shell) makes fluidsynth play the file and exit rather than idle.
        subprocess.run(["fluidsynth", "-ni", str(soundfont), str(midi)], check=True)


@click.command()
@click.argument(
    "inputs",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--model", default="tatum-arc-e30", show_default=True, help="Scorer name."
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="MIDI output path (default: <first input>.mid).",
)
@click.option("--tempo", type=int, default=90, show_default=True, help="Quarter BPM.")
@click.option(
    "--dpi", type=int, default=300, show_default=True, help="PDF rasterisation DPI."
)
@click.option(
    "--width",
    type=int,
    default=1200,
    show_default=True,
    help="Downscale each page to this px width before the model, matching the "
    "kernsheet build (INTER_AREA to 1200) the scorer trained on. Native high-res "
    "renders have sharp thin staff lines the model mis-registers (pitches read a "
    "third off).",
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
    "--pdmx-home",
    # Not exists-validated: it is only consulted for the default vocab, so --vocab
    # alone must work on a machine without PDMX.
    type=click.Path(dir_okay=True, file_okay=False, path_type=Path),
    default=HOME,
    help="Corpus root holding build/vocab.json (default: PDMX).",
)
@click.option(
    "--vocab",
    "vocab_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Vocab JSON (default: <pdmx-home>/build/vocab.json).",
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
    dpi: int,
    width: int,
    play: bool,
    wav: Path | None,
    soundfont: Path,
    pdmx_home: Path,
    vocab_path: Path | None,
    log_level: str,
) -> None:
    """Transcribe a score (PDF or page images) and play it back as MIDI.

    INPUTS: one .pdf, or one or more page images (PNG/JPG), in reading order.
    """
    log_uncaught_exceptions()
    logging.basicConfig(level=getattr(logging, log_level.upper()))
    torch.set_float32_matmul_precision("high")

    # Imported here so `--help` stays fast (avoids loading torch-heavy scorer stack).
    from cli.scorer import _load_for_inference

    config, module = _load_for_inference(model)
    vocab_file = vocab_path or pdmx_home / "build" / "vocab.json"
    if not vocab_file.is_file():
        raise click.ClickException(
            f"Vocab not found: {vocab_file} (pass --vocab or --pdmx-home)."
        )
    vocab = Vocab.load(vocab_file)
    transform = build_transform(config)

    midi_path = output or inputs[0].with_suffix(".mid")
    with tempfile.TemporaryDirectory() as tmp:
        pages = resolve_pages(inputs, Path(tmp), dpi)
        click.echo(f"Transcribing {len(pages)} page(s) with '{model}' ...")
        parts: dict[int, Part] = {}
        for page in pages:
            image = load_page(page, width, transform).to(module.device)
            count = transcribe_page(module, vocab, image, parts)
            click.echo(f"  {page.name}: {count} staves")

    if not parts:
        raise click.ClickException("No staves detected — nothing to play.")
    ordered = [parts[slot] for slot in sorted(parts)]
    parts_to_midi(ordered, midi_path, tempo=tempo, dynamics=Dynamics())
    click.echo(f"Wrote MIDI: {midi_path} ({len(ordered)} parts)")

    if play or wav is not None:
        play_or_render(midi_path, soundfont, wav, play)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
