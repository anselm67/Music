# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Behavioral Guidelines

### 1. Think Before Coding
- State your assumptions explicitly before writing code. If uncertain, ask.
- If multiple interpretations of a task exist, present them—do not pick silently.
- If a simpler approach exists, suggest it and push back when warranted.
- Stop and name what is confusing before guessing through it.

### 2. Simplicity First
- No features beyond what was explicitly asked.
- No abstractions or "flexibility" for single-use code.
- No error handling for impossible scenarios.
- Ask: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes
- Touch only what the user's request requires.
- Do not "improve," reformat, or refactor adjacent code or comments.
- Match existing repository style exactly.
- Remove imports, variables, or functions that YOUR changes made unused; do not touch pre-existing dead code.
- When you change the semantics of a variable, function, or field, rename it to match the new meaning. Stale names that contradict what something now represents are bugs waiting to happen.

### 4. Goal-Driven Execution
- Transform tasks into verifiable goals (e.g., "Write a test that reproduces the bug, then make it pass").
- For multi-step tasks, state a brief plan with explicit verification steps before proceeding.
- Strong success criteria let you loop independently; weak criteria require constant interruption.

## Goal

End-to-end Optical Music Recognition pipeline:
**scan → layout detection → transcription → Kern → MIDI → playback**

Ultimate target: a phone app that scans physical sheet music and plays it back.
Architectural north star: a single fully end-to-end model where detection feeds
transcription in one forward pass.

---

## Commands

```bash
# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/kern/test_tokenizer.py

# Run tests with coverage
uv run pytest --cov=src

# Lint
uv run ruff check src tests

# Format
uv run ruff format src tests

# Type check
uv run mypy src tests
```

**Package management:** always use `uv add <pkg>` / `uv remove <pkg>` — never `pip install`.

### CLI entry points

Dataset roots: PDMX lives at `/home/anselm/datasets/PDMX`, KernSheet at `~/datasets/KernSheet`.
The dataset CLIs (`pdmx`, `kernsheet`) take a single `--home`; the model CLIs (`staffer`,
`noter`, `scorer`) consume **both** corpora and take `--pdmx-home` *and* `--kern-home` (no
`--home`). All CLIs also accept `--csv` (subset CSV, default varies), `--log-level`, `--log-file`.

```bash
# Build all PDMX assets (svg, png, layout, krn, tokens) from a raw download
pdmx make

# Query/filter the dataset into a subset CSV.
# Subset CSVs are per-machine and MUST be built with --valid, or training later dies with
# "token file not found" (at train time, not load time — hard to diagnose).
pdmx query -o Staff16.csv 'index==index' --score 'pages.*.staff_count < 16' --valid
pdmx query -o System2.csv 'index==index' --score 'pages.0.systems.0.staff_count <= 2' --valid

# Dataset stats
pdmx --csv subset.csv stats

# Train the staffer layout detector
staffer --log-file logs/staffer/<model_name>.log train -e 12 --use-sampler <model_name>

# Run staffer inference on images
staffer predict <model_name> <img1.png> [img2.png ...]

# Plot training curves
staffer logs <model_name>

# Build vocab from a PDMX subset, then inspect the noter dataset
noter --csv System2.csv vocab
noter --csv System2.csv show
noter --csv System2.csv stats

# End-to-end scorer (merge of a trained staffer + noter)
scorer eval <model_name> --size 500 [--rerank]
scorer predict <model_name> <img.png> [--rerank]

# Full pipeline playback: score (PDF or images) → transcription → MIDI → audio
play <score.pdf|page1.png ...> [-o out.mid] [--play] [--wav out.wav]

# KernSheet real-scan corpus
kernsheet make                  # pre-render the page-image cache
kernsheet detect [prefix]       # cv2-seed un-validated layouts for scores with none
kernsheet edit [--review NAME]  # editor / review worklist
kernsheet review [--review NAME]

# Humdrum Kern + kern→MIDI utilities
kern --help
```

---

## Repo layout

```
src/
  cli/         # Click entry points: kern.py, pdmx.py, staffer.py, noter.py, scorer.py, kernsheet.py, play.py
  sheetmusic/  # Source protocol + layout types (Score, Page, System, Staff, Box), shared by all datasets
  pdmx/        # PDMX index, asset builder, stats
  kernsheet/   # Real-scan corpus (IMSLP-derived): index, asset builder, editor, reviews framework
  staffer/     # StafferModel (stave-primary detector), loss, module, dataset, datamodule
  noter/       # NoterModel/Config, module, dataset, datamodule, vocab
  scorer/      # end-to-end model: staffer+noter joined in one forward pass (see docs/architecture.html)
  kern/        # Humdrum Kern parser, tokenizer, kern→MIDI converter
  midi/        # Binary MIDI I/O (input, output, typing)
  utils/       # Walker async pool, json_query, misc helpers
  verovio/     # Verovio wrapper, SVG scraper (LayoutExtractor), binaries
tests/         # pytest, mirrors src/ structure
scripts/       # one-off diagnostics (run, not imported; outside the package)
```

`scripts/` holds standalone tools run via `uv run python scripts/<name>.py` — they
import from `src/` but are not part of the installable package (not type-checked
under `mypy src tests`, may use private APIs). All scripts are documented in
`docs/scripts.html` — **keep that file up to date when adding or removing scripts**.

`docs/` holds the project's prose — read it for context, write to it when work lands:
- `journal.html` — design history / reverse-chron ADRs (the "why" behind decisions)
- `architecture.html` — the end-to-end `scorer` design
- `scripts.html` — index of everything in `scripts/`
- `{staffer,noter,scorer}-training.html` — authoritative eval tables; read for the current best
  checkpoint, write after every run

---

## Pipeline overview

### Stage 1 — Layout detection: `staffer`

Detector with two query streams where **staves are primary and systems are derived** (the
`stave-primary-grid` architecture; staves march at near-constant pitch, a stronger prior than
variable-gap system tops).

**Model (`StafferModel`):** ViT backbone (`nn.Conv2d` patch embedding, D=256, H=8, mlp=1024,
16×16 patches) → `StafferDecoder` with two parallel query streams. The **stave** stream is
independent (no group attention — staves never inherit from system queries) and its head predicts
`(top, bottom)` only, as a residual to a **frozen per-slot vertical anchor grid** `(i+0.5)/M`;
x is inherited from the parent system. The **system** stream predicts only the system-level
horizontal extent `(left, right)` + objectness; the system's vertical extent is **derived** as
the hull of its staves. Boxes are normalised ltrb.

**Dataset (`StafferDataset`):** a page image paired with its GT system/stave boxes; drawn from a
PDMX subset (`Staff16`, <16 staves/page) and, for fine-tuning, the KernSheet real-scan corpus.

**Loss (`staffer_loss.py`):** stave top/bottom L1 + system GIoU + BCE objectness. GT staves are
routed to queries by **optimal assignment** (`assign_staves`, `scipy.linear_sum_assignment` over a
cost) — free / non-contiguous, not index-positional — so the anchored slots train at ~uniform
frequency.

**Module (`StafferModule`):** AdamW + warmup + cosine schedule; `WeightedRandomSampler`
(`bottom_bias`, last-page oversampling). Hardware: RTX 4060/5060 Ti 16 GB.

For the current best checkpoint and full run history, see `docs/staffer-training.html` (the
authoritative eval table) — not this file. NB: `val/stave_l1` is **not** a clean cross-architecture
metric (logging-key collision in `StafferModule._step`, logged twice under the same key); use the
clean `val/stave_l1_px`.

---

### Stage 2 — Transcription: `noter`

**Model (`NoterModel`):** encoder-decoder Transformer.
- Encoder: staff image (1×H×W) → `SourceEmbedding` (Conv2d patch projection + positional) → Transformer encoder
- Decoder: target sequence (B, T, max_chords=8) → `TargetEmbedder` (embedding + chord projection) → Transformer decoder → logits (B, T, max_chords, vocab_size)
- Input shape: (64, 768) px staff crop; patch_width=4, patch_height=64 → 192 patches
- Variable-width source handled via `make_src_padding_mask` from actual pixel widths

**Dataset (`NoterDataset`):** built from a PDMX subset (≤2 staves/system). Each sample is a cropped staff image paired with its Kern token sequence for the corresponding bar range. Spine ordering for 2-staff systems: `[1, 0]` (treble=spine 1, bass=spine 0 in tokens file).

**Vocabulary (`Vocab`):** flat tokens — each unique token string (`C/4`, `C/8:1`, `clef-G`, `4/4`, etc.) gets its own integer ID. ~5k tokens built from observed `.tokens` files. Special tokens: PAD=0, UNK=1, SOS=2, EOS=3, SIL=4 (chord padding). Bar numbers are stripped before lookup (`=5` → `=`). Built per corpus and saved as JSON at `<corpus>/build/vocab.json`.

**Output format:** tokens derived from Humdrum **Kern** (single-spine canonical form; chord notes sorted; one spine per instrument staff). `kern/tokenizer.py` translates native Kern to simplified token notation.

**Lightning module:** `NoterModule` — cross-entropy loss over (B×T×max_chords, vocab_size), ignoring PAD; reports token accuracy metric.

For the current best checkpoint and full run history, see `docs/noter-training.html` (the
authoritative eval table) — not this file.

---

### Stage 3 — End-to-end: `scorer`

The architectural north star, landed and in production: `staffer` (detector) and `noter`
(transcriber) fused into **one forward pass** (scan → boxes → transcription), the design in
`docs/architecture.html`.

**Model (`ScorerModel`):** an intact `StafferModel` + `NoterModel` joined by a **differentiable
`grid_sample` crop bridge** (`crop()`) — NOT roi_align (which doesn't backprop to box coords).
Both standalone checkpoints transfer wholesale; the bridge adds 0 params. Decoupled — no shared
patch embedding. The bridge crops each detected stave (1:1, un-stretched, 64×768 px) so the
transcription loss reaches the detector's box-position gradients.

**Module (`ScorerModule`):** joint loss `λ_det·det + λ_tr·tr`, GT-routed teacher forcing, a
staffer-freeze warmup, `load_from_checkpoints` to merge two standalone runs. Inference:
`predict`/`_generate` with beam search and an optional per-system cross-stave barline-agreement
reranker (`_generate_rerank`, `--rerank`; see the constraint-layer work).

**CLI (`scorer`):** `check` / `train` / `logs` / `predict` / `eval`.

**Models are named after composers, worst-first.** Current production end-to-end model and full
run history: see `docs/scorer-training.html` — not this file.

---

### Kern → MIDI

A from-scratch converter (`midi/`): manages the binary MIDI format directly (no library),
handles spine splits / merges / exchanges, channel allocation via `deque`. The playback tail of
the pipeline, downstream of transcription.

---

## Data pipelines

Everything under `build/` (svg/png/layout/krn/tokens/vocab.json) is **gitignored, derived, and
mtime-gated** — a tokenizer or kern change does NOT refire the builders, so `tokens`/`vocab.json`
go silently stale. After any such change, wipe the affected `build/` outputs and re-run
`pdmx make` / `kernsheet make` before trusting them.

### PDMX assets (synthetic, ~500k pages)

The `pdmx make` command builds per-entry assets under `build/`:
- `svg` — Verovio SVG rendering
- `png` — rasterised via `rsvg-convert`
- `layout` — bounding boxes scraped from SVG by `LayoutExtractor`
- `krn` — Kern export via Verovio
- `tokens` — simplified token sequences via `kern.tokenize`

`PDMX` class (DataFrame-backed) manages all path resolution (`get_path`, `get_page_path`).
Flexible JSON query/filter system: `compile_query` / `compile_filter` with wildcard support.
Multi-page scores store per-page PNGs; single-page scores use a flat filename.

#### Verovio notes

- CLI options in regular use: `--mnum-interval`, `-a`, `-l`
- SVG parsing: SMuFL glyph IDs for multi-rest counts; `measure.nummodulus` in MEI
- Known issue: drumset `R`-tokens cause a large SVG chunk to be dropped — valid data, but be aware when debugging ground truth extraction
- Known issue: invisible bars at start of SVG inflate the layout bar count (`LayoutExtractor` counts them but shouldn't)

### KernSheet corpus (real scans)

The second dataset: **real scanned editions** (IMSLP-derived PDFs) with hand-validated
layouts — the real-scan counterpart to PDMX's synthetic Verovio renders, used to fine-tune
every stage across the domain gap. `~/datasets/KernSheet` is itself a **git repo** (deletes
are recoverable); PDFs are **shared across entries**, so entry deletion must not unlink them.
Assets: `layout` is per-score, `krn`/`tokens` per-entry.

```bash
kernsheet make              # pre-render the page-image cache (build/png)
kernsheet detect [prefix]   # cv2 ClassicalStaffer seeds an UN-validated layout for scores with none
kernsheet edit [--review N] # editor: validate/fix layouts, set page status, walk review worklists
kernsheet review [--review N]  # read-only report of layout-review findings across the corpus
kernsheet stats / check
```

Pages carry a `Status` enum (`pending`/`validated`/`rejected`) plus a `reviewed` list; the
training/eval data path consumes **validated pages only** at page granularity (via
`Source.pages`), whereas PDMX yields all pages. `kernsheet/reviews.py` is an extensible
registry of cheap geometry checks (first: `staff_height`) recomputed on demand, never persisted.

---

## Conventions

- `src/` layout, `hatchling` build backend
- `uv` for all package management — never `pip install` directly into the venv
- Type hints throughout; `mypy --disallow-untyped-defs`
- Tests in `tests/` with pytest; `asyncio_mode = "auto"`
- CLI entry points via Click
- Git: tag checkpoints before major architectural changes
- Definition of done: `uv run ruff check src tests` clean, `uv run mypy src tests` clean, and
  `uv run pytest` green.

### Committing

- **Always ask before committing.** Never stage and commit without explicitly prompting the user first.
- Before committing, run `uv run ruff format src tests` to format, then `uv run ruff check src tests` to verify no lint errors remain.
- Then spawn the `code-reviewer` agent to review the staged changes. Skip the review for purely cosmetic or documentation-only changes.

### Running a training (all stages: `staffer` / `noter` / `scorer`)

A run is a multi-hour GPU job. Launch it in the background (or hand the user the command if they
prefer to run it themselves). Same process for every stage; only the CLI binary and
`docs/<stage>-training.html` differ:

1. Make sure all tests pass, then commit a `train/<model-name>` tag so the run is reproducible.
   Use a descriptive name; if the model derives from a previous one, include its name (e.g.
   `enhanced2-vflip` = `enhanced2` plus vflip augmentation).
2. Run the command (or give it to the user), with the log file routed under the stage:
   ```bash
   staffer --log-file logs/staffer/<model-name>.log train OPTIONS...
   noter   --log-file logs/noter/<model-name>.log   train OPTIONS...
   scorer  --log-file logs/scorer/<model-name>.log  train OPTIONS...
   ```
3. Metrics land at `logs/<stage>/<model-name>/metrics.csv`.
4. After the run, update the training log (`docs/<stage>-training.html`) with its eval results
   and metrics.

The live worklist of open items (what's next across staffer/scorer/noter/data) lives in the
session memory index, not here — this file documents the architecture, not point-in-time status.
