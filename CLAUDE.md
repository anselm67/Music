# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
uv run mypy src
```

**Package management:** always use `uv add <pkg>` / `uv remove <pkg>` — never `pip install`.

### CLI entry points

All CLIs accept `--home` (PDMX root, default `/home/anselm/datasets/PDMX`),
`--csv` (subset CSV, default varies), `--log-level`, and `--log-file`.

```bash
# Build all PDMX assets (svg, png, layout, krn, tokens) from a raw download
pdmx make

# Query/filter the dataset into a subset CSV
pdmx query -o Staff16.csv 'index==index' --score 'pages.*.staff_count < 16' --valid
pdmx query -o System2.csv 'index==index' --score 'pages.0.systems.0.staff_count <= 2' --valid

# Dataset stats
pdmx --csv subset.csv stats

# Train the staffer layout detector
staffer --log-file logs/staffer.log train -e 12 --use-sampler <model_name>

# Run staffer inference on images
staffer predict <model_name> <img1.png> [img2.png ...]

# Plot training curves
staffer logs <model_name>

# Build vocab from a PDMX subset, then inspect the noter dataset
noter --csv System2.csv vocab
noter --csv System2.csv show
noter --csv System2.csv stats

# Kern / MIDI tools
kern ...   # Kern file utilities
```

---

## Code review

Before creating any git commit, always spawn the `code-reviewer` agent to review the staged changes first.

---

## Repo layout

```
src/
  cli/         # Click entry points: kern.py, pdmx.py, staffer.py, noter.py
  dataset/     # PDMX index, StafferDataset/DataModule, layout types (Score, Box)
  kern/        # Humdrum Kern parser, tokenizer, kern→MIDI converter
  models/      # staffer_model.py (HierarchicalDETR), staffer_module.py, staffer_loss.py
  noter/       # noter_model.py (NoterModel/Config), noter_module.py, noter_dataset.py, noter_vocab.py
  utils/       # Walker async pool, json_query, misc helpers
  verovio/     # Verovio wrapper, SVG scraper (LayoutExtractor), binaries
tests/         # pytest, mirrors src/ structure
```

---

## Pipeline overview

### Stage 1 — Layout detection: `staffer`

Hierarchical detector: **systems → staves**.

**Backbone:** ViT with `nn.Conv2d` patch embedding (D=256, 16×16 patches).

**Decoder:** `HierarchicalDecoder` with parallel query streams:
- System queries → system bounding boxes
- Stave queries → grouped under parent system via cross-attention

**Key architectural insight:** staves only need to predict y-coordinates (x inherited from parent system). Right now, stave queries generate a full Box.

**Loss (`HierarchicalLoss` / `staffer_loss.py`):** L1 + GIoU box loss, BCE objectness,
cross-entropy assignment, containment, alignment — returns a `LossDict`.

**Matching:** index-based positional matching (top-to-bottom sort), not Hungarian.

**Box format:** normalised cxcywh.

**Training:**
- AdamW + warmup + cosine schedule
- `WeightedRandomSampler` (sqrt-inverse frequency + last-page oversampling)
- Teacher forcing at structural level: GT stave embeddings injected during
  training with curriculum decay schedule (fixes ~10px systematic bar offset)
- Hardware: RTX 4060/5060 Ti 16 GB

Here are the steps for running a training:
- Make sure all tests pass, and that a tag is created with a descrptive name for the new model; If the new model derives from a previous one, it should include its name, eg enhanced2-vflip is the enhanced2 model plus data augmentation vflip
- The actual training command shouhld look like this:
```bash
staffer --log-file logs/staffer/<model-name>.log train OPTIONS...
noter --log-file logs/noter/<model-name>.log train OPTIONS...
```
- Training metrics will be available - eg for staffer - as logs/staffer/<model-name>/metrics.csv


**Current checkpoint status:** system IoU >0.95, stave IoU ~0.93 (tagged in Git
before bar detection work began).

---

### Stage 2 — Transcription: `noter`

**Model (`NoterModel`):** encoder-decoder Transformer.
- Encoder: staff image (1×H×W) → `SourceEmbedding` (Conv2d patch projection + positional) → Transformer encoder
- Decoder: target sequence (B, T, max_chords=8) → `TargetEmbedder` (embedding + chord projection) → Transformer decoder → logits (B, T, max_chords, vocab_size)
- Input shape: (64, 768) px staff crop; patch_width=4, patch_height=64 → 192 patches
- Variable-width source handled via `make_src_padding_mask` from actual pixel widths

**Dataset (`NoterDataset`):** built from a PDMX subset (≤2 staves/system). Each sample is a cropped staff image paired with its Kern token sequence for the corresponding bar range. Spine ordering for 2-staff systems: `[1, 0]` (treble=spine 1, bass=spine 0 in tokens file).

**Vocabulary (`Vocab`):** token encoding packs pitch-base into low 16 bits, duration into high 16 bits. Special tokens: PAD=0, UNK=1, SOS=2, EOS=3, SIL=4 (chord padding). Saved as JSON at `build/vocab.json`.

**Output format:** tokens derived from Humdrum **Kern** (single-spine canonical form; chord notes sorted; one spine per instrument staff). `kern/tokenizer.py` translates native Kern to simplified token notation.

**Lightning module:** `NoterModule` — cross-entropy loss over (B×T×max_chords, vocab_size), ignoring PAD; reports token accuracy metric.

---

### Stage 3 — Kern → MIDI

Custom converter built from scratch:
- Manages binary MIDI format directly (no library)
- Handles spine splits / merges / exchanges
- Channel allocation via `deque`

---

## Data pipelines

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

### Verovio notes

- CLI options in regular use: `--mnum-interval`, `-a`, `-l`
- SVG parsing: SMuFL glyph IDs for multi-rest counts; `measure.nummodulus` in MEI
- Known issue: drumset `R`-tokens cause a large SVG chunk to be dropped — valid data, but be aware when debugging ground truth extraction
- Known issue: invisible bars at start of SVG inflate the layout bar count (`LayoutExtractor` counts them but shouldn't)

---

## Conventions

- `src/` layout, `hatchling` build backend
- `uv` for all package management — never `pip install` directly into the venv
- Type hints throughout; `mypy --disallow-untyped-defs`
- Tests in `tests/` with pytest; `asyncio_mode = "auto"`
- CLI entry points via Click
- Git: tag checkpoints before major architectural changes
- make sure a train/xxx tag is commited before training a model, so we can get back to it if needed.

---

## Current focus areas

1. Aligning stave systems to Kern spines in the dataset — mostly done, but bar number mismatches remain when Score/page/system computed bar numbers diverge from Verovio output.
2. `NoterModel` — simple staff-image → tokens model using scores with ≤2 staves for easy staff↔token matching. `NoterDataset` implements this pairing.
3. [longer term] Multi-stave processing: `SystemTransformer` + `SharedSpineDecoder` + RoIAlign feature extraction from `staffer` detections.
4. Single end-to-end model (detection + transcription in one forward pass).

### Known pending fixes

- Tokenizer should inspect first-bar lengths against the metric to determine where bar number 1 falls.
- `mxl/14/10/QmWAGX...`: rendering missing first few bars → bar sync off.
- `mxl/3/6/Qmd7UQ...`: bar count mismatch due to invisible leading bars in SVG.
- `NoterModel` output: consider predicting pitch and duration separately, or rework `Vocab.encode`.
- Staffer: stave output should become two y-coordinates only; derive full box from parent system.
