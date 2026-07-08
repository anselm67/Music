#!/usr/bin/env bash
#
# Train a full scorer from scratch: PDMX base + rehearsal-mix fine-tune for both
# the staffer and the noter, then merge the two fine-tuned branches and fine-tune
# the scorer on the same mix. Given one base NAME the five sub-models are named:
#
#   staffer/<NAME>           noter/<NAME>           (PDMX bases, shared name)
#   staffer/<NAME>-mixed     noter/<NAME>-mixed     (rehearsal-mix fine-tunes)
#   scorer/<SCORER_NAME>     (the merged end-to-end model; defaults to <NAME>)
#
# Fine-tuning is a PDMX-primary REHEARSAL MIX (`--kern-sheet <KS> $MIX`), not a
# KernSheet-only pass: PDMX stays primary and KernSheet is mixed in at a $MIX
# fraction. This fixes the catastrophic forgetting that KS-only FT caused (the
# unfrozen detector/decoder drifting on PDMX) — a strict win at every site,
# largest at the joint scorer. See memory `project-mixed-finetune`.
#
# Run from the repo root. Each stage is fail-fast and skipped if its checkpoint
# already exists, so the pipeline is resumable. Set DRYRUN=1 to print the commands
# without running, FORCE=1 to retrain stages whose checkpoint already exists.
#
#   scripts/train_scorer_pipeline.sh <NAME> [SCORER_NAME]
#   DRYRUN=1 scripts/train_scorer_pipeline.sh my-model
#
# FINE-TUNE-ONLY re-runs (BASE=<existing>): when you only changed the KernSheet
# data (e.g. a layout fix) the PDMX bases and the vocab are unaffected, so reuse
# them instead of retraining. Set BASE to the existing base name; stages 1 & 3
# are then skipped (the bases are reused via --init-from) and only the three
# rehearsal-mix FT stages run, under the new <NAME> lineage. No asset rebuild is
# needed — layout edits don't touch build/png or build/tokens.
#
#   BASE=mahler scripts/train_scorer_pipeline.sh mahler-sh   # reuse mahler bases
#
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

NAME="${1:?usage: $0 <name> [scorer_name]}"
SCORER_NAME="${2:-$NAME}"
BASE="${BASE:-$NAME}"   # PDMX bases to fine-tune FROM; defaults to NAME (train them)

# ---------------------------------------------------------------------------
# Config — these are the documented best-run recipes; edit to taste. The
# fine-tune split sizes are over the MIXED epoch (PDMX-primary + KernSheet at
# $MIX); KernSheet contributes mix*train_len items so its share stays <= the KS
# dataset (staffer 2678 pages w/ systems, scorer 2670 usable pages, noter 15079
# usable systems — one item per system since the cross-stave change, NOT per
# staff), and the noter/scorer need >= 250 train batches (their
# val_check_interval; the staffer clamps so any split works). NB: the raw
# "validated pages" count (~6463) badly overstates usable data — ~3976 are
# empty shared-PDF spillover (a per-work score spans the whole shared PDF). Size
# train_len to the usable ceiling: `kernsheet stats` reports it (scorer pages +
# noter systems), as does scripts/scorer_usable_ks_pages.py; past total ~=
# usable/$MIX the MixedSource just recycles the same KS pages.
# ---------------------------------------------------------------------------
KS=/home/anselm/datasets/KernSheet
VOCAB="$KS/build/vocab.json"
MIX=0.67                    # KernSheet share of each fine-tune epoch (rest is PDMX rehearsal)

STAFFER_BASE_EPOCHS=7
STAFFER_FT_EPOCHS=30
NOTER_BASE_EPOCHS=16
NOTER_FT_EPOCHS=30   # cap, not a fixed count: stage 4 runs `-s 2.0` early-stopping.
                     # 12 under-trained the FT (tatum-arc-mixed-e30 stopped at 21,
                     # a strict win: KS art micro-F1 68.9->73.2, gap -16.3->-10.6pp).
SCORER_EPOCHS=20

# Per-stage fine-tune LR / warmup. The staffer FT ran 10x lower than the noter FT;
# the scorer rides the ScorerConfig defaults (lr 1e-4, warmup 500, freeze 500) — it
# passes no --lr/--warmup-steps/--freeze-staffer-steps, matching the ravel run.
STAFFER_FT_LR=1e-5; STAFFER_FT_WARMUP=200
NOTER_FT_LR=1e-4;   NOTER_FT_WARMUP=200

STAFFER_FT_TRAIN=3700;  STAFFER_FT_VALID=550    # mixed epoch; KS share ~2847 covers 2678 pages
NOTER_FT_TRAIN=20400;   NOTER_FT_VALID=2300     # KS share ~15209 covers 15079 usable systems
SCORER_FT_TRAIN=3500;   SCORER_FT_VALID=500     # KS share ~2680 covers 2670 usable (>= 250 batches at bs 8)
# ---------------------------------------------------------------------------

run() {
  echo "+ $*"
  [[ "${DRYRUN:-0}" == 1 ]] || "$@"
}

# tag <tag-name>: mark the current commit so the run is reproducible (per CLAUDE.md).
tag() {
  if git rev-parse -q --verify "refs/tags/$1" >/dev/null; then
    echo "  tag $1 already exists — skipping"
  else
    run git tag "$1"
  fi
}

# done_ckpt <branch> <name>: true if a finished checkpoint exists (and not FORCE).
done_ckpt() {
  [[ "${FORCE:-0}" != 1 && -f "checkpoints/$1/$2/last.ckpt" ]]
}

stage() {  # stage "<title>"
  echo
  echo "======================================================================"
  echo "== $1"
  echo "======================================================================"
}

# --- prerequisite: shared KernSheet vocab (token-only; unaffected by geometry) ---
if [[ ! -f "$VOCAB" ]]; then
  stage "Build shared KernSheet vocab (extends the PDMX vocab)"
  run noter extend-vocab "$KS"
fi

# --- Stage 1: staffer PDMX base ---
stage "1/5  staffer PDMX base -> checkpoints/staffer/$BASE"
if [[ "$BASE" != "$NAME" ]]; then
  echo "  reusing base checkpoints/staffer/$BASE (BASE=$BASE) — not training"
  [[ -f "checkpoints/staffer/$BASE/last.ckpt" ]] || { echo "  missing base checkpoint" >&2; exit 1; }
elif done_ckpt staffer "$NAME"; then
  echo "  exists — skipping (FORCE=1 to retrain)"
else
  tag "train/$NAME"
  run staffer --log-file "logs/staffer/$NAME.log" \
    train -e "$STAFFER_BASE_EPOCHS" --use-sampler "$NAME"
fi

# --- Stage 2: staffer rehearsal-mix fine-tune ---
stage "2/5  staffer rehearsal-mix fine-tune -> checkpoints/staffer/$NAME-mixed"
if done_ckpt staffer "$NAME-mixed"; then
  echo "  exists — skipping (FORCE=1 to retrain)"
else
  tag "train/$NAME-mixed"
  run staffer --log-file "logs/staffer/$NAME-mixed.log" \
    train --init-from "checkpoints/staffer/$BASE/last.ckpt" \
    --kern-sheet "$KS" "$MIX" \
    --train-len "$STAFFER_FT_TRAIN" --valid-len "$STAFFER_FT_VALID" \
    --lr "$STAFFER_FT_LR" --warmup-steps "$STAFFER_FT_WARMUP" \
    -e "$STAFFER_FT_EPOCHS" --use-sampler \
    "$NAME-mixed"
fi

# --- Stage 3: noter PDMX base (trains directly on the KernSheet vocab) ---
stage "3/5  noter PDMX base -> checkpoints/noter/$BASE"
if [[ "$BASE" != "$NAME" ]]; then
  echo "  reusing base checkpoints/noter/$BASE (BASE=$BASE) — not training"
  [[ -f "checkpoints/noter/$BASE/last.ckpt" ]] || { echo "  missing base checkpoint" >&2; exit 1; }
elif done_ckpt noter "$NAME"; then
  echo "  exists — skipping (FORCE=1 to retrain)"
else
  tag "train/$NAME"
  run noter --log-file "logs/noter/$NAME.log" \
    train --vocab "$VOCAB" \
    -e "$NOTER_BASE_EPOCHS" \
    "$NAME"
fi

# --- Stage 4: noter rehearsal-mix fine-tune (mix carries KS exposure) ---
stage "4/5  noter rehearsal-mix fine-tune -> checkpoints/noter/$NAME-mixed"
if done_ckpt noter "$NAME-mixed"; then
  echo "  exists — skipping (FORCE=1 to retrain)"
else
  tag "train/$NAME-mixed"
  run noter --log-file "logs/noter/$NAME-mixed.log" \
    train --init-from "checkpoints/noter/$BASE/last.ckpt" \
    --kern-sheet "$KS" "$MIX" --vocab "$VOCAB" \
    --train-len "$NOTER_FT_TRAIN" --valid-len "$NOTER_FT_VALID" \
    --lr "$NOTER_FT_LR" --warmup-steps "$NOTER_FT_WARMUP" \
    -e "$NOTER_FT_EPOCHS" -s 2.0 \
    "$NAME-mixed"
fi

# --- Stage 5: scorer merge + rehearsal-mix fine-tune ---
stage "5/5  scorer rehearsal-mix fine-tune -> checkpoints/scorer/$SCORER_NAME"
if done_ckpt scorer "$SCORER_NAME"; then
  echo "  exists — skipping (FORCE=1 to retrain)"
else
  tag "train/$SCORER_NAME"
  # --vocab is REQUIRED here: the mix is PDMX-primary so the scorer would default
  # to the PDMX vocab (4785), but the mixed branches carry the extended KernSheet
  # vocab (5492) — without it the noter state_dict mismatches and load crashes.
  # LR/warmup/freeze ride the ScorerConfig defaults, as the ravel run did.
  # --batch-size 4: the default 4x4 noter (3072 encoder patches) overflows a 16GB
  # GPU at the default batch 8. And expandable_segments reclaims the ~3GB CUDA
  # reserved-but-unallocated fragmentation that OOMs batch 4 too (tatum-arc-e30).
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  run scorer --log-file "logs/scorer/$SCORER_NAME.log" \
    train --staffer "$NAME-mixed" --noter "$NAME-mixed" \
    --kern-sheet "$KS" "$MIX" --vocab "$VOCAB" \
    --batch-size 4 \
    --train-len "$SCORER_FT_TRAIN" --valid-len "$SCORER_FT_VALID" \
    -e "$SCORER_EPOCHS" \
    "$SCORER_NAME"
fi

echo
echo "Done. Final scorer: checkpoints/scorer/$SCORER_NAME/last.ckpt"
echo "Update docs/{staffer,noter,scorer}-training.html with eval results."
