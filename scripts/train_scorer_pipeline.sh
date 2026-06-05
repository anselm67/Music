#!/usr/bin/env bash
#
# Train a full scorer from scratch: PDMX base + KernSheet fine-tune for both the
# staffer and the noter, then merge the two fine-tuned branches and fine-tune the
# scorer on KernSheet. Given one base NAME the five sub-models are named:
#
#   staffer/<NAME>              noter/<NAME>              (PDMX bases, shared name)
#   staffer/<NAME>-kernsheet    noter/<NAME>-kernsheet    (KernSheet fine-tunes)
#   scorer/<SCORER_NAME>        (the merged end-to-end model; defaults to <NAME>)
#
# Run BY THE USER from the repo root (this orchestrator launches training; Claude
# does not). Each stage is fail-fast and skipped if its checkpoint already exists,
# so the pipeline is resumable. Set DRYRUN=1 to print the commands without running,
# FORCE=1 to retrain stages whose checkpoint already exists.
#
#   scripts/train_scorer_pipeline.sh <NAME> [SCORER_NAME]
#   DRYRUN=1 scripts/train_scorer_pipeline.sh my-model
#
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

NAME="${1:?usage: $0 <name> [scorer_name]}"
SCORER_NAME="${2:-$NAME}"

# ---------------------------------------------------------------------------
# Config — these are the documented best-run recipes; edit to taste. The
# KernSheet fine-tune split sizes must sum to <= the dataset (staffer/scorer
# 2321 pages, noter 26166 staff items) and give the noter/scorer >= 250 train
# batches (their val_check_interval; the staffer clamps so any split works).
# ---------------------------------------------------------------------------
KS=/home/anselm/datasets/KernSheet
VOCAB="$KS/build/vocab.json"

STAFFER_BASE_EPOCHS=7
STAFFER_FT_EPOCHS=30
NOTER_BASE_EPOCHS=12
NOTER_FT_EPOCHS=12
SCORER_EPOCHS=20

NOTER_JITTER=0.5             # PDMX base only; KernSheet boxes are a thirds approx (no jitter)

# Per-stage fine-tune LR / warmup. The staffer FT ran 10x lower than the noter FT;
# the scorer rides the ScorerConfig defaults (lr 1e-4, warmup 500, freeze 500) — it
# passes no --lr/--warmup-steps/--freeze-staffer-steps, matching the ravel run.
STAFFER_FT_LR=1e-5; STAFFER_FT_WARMUP=200
NOTER_FT_LR=1e-4;   NOTER_FT_WARMUP=200

STAFFER_FT_TRAIN=2000; STAFFER_FT_VALID=300     # 2321 pages
NOTER_FT_TRAIN=23400;  NOTER_FT_VALID=2600      # 26166 staff items
SCORER_FT_TRAIN=2000;  SCORER_FT_VALID=300      # 2321 pages (>= 250 batches at bs 8)
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
stage "1/5  staffer PDMX base -> checkpoints/staffer/$NAME"
if done_ckpt staffer "$NAME"; then
  echo "  exists — skipping (FORCE=1 to retrain)"
else
  tag "train/$NAME"
  run staffer --log-file "logs/staffer/$NAME.log" \
    train -e "$STAFFER_BASE_EPOCHS" --use-sampler "$NAME"
fi

# --- Stage 2: staffer KernSheet fine-tune ---
stage "2/5  staffer KernSheet fine-tune -> checkpoints/staffer/$NAME-kernsheet"
if done_ckpt staffer "$NAME-kernsheet"; then
  echo "  exists — skipping (FORCE=1 to retrain)"
else
  tag "train/$NAME-kernsheet"
  run staffer --kern-home "$KS" --log-file "logs/staffer/$NAME-kernsheet.log" \
    train --init-from "checkpoints/staffer/$NAME/last.ckpt" \
    --train-len "$STAFFER_FT_TRAIN" --valid-len "$STAFFER_FT_VALID" \
    --lr "$STAFFER_FT_LR" --warmup-steps "$STAFFER_FT_WARMUP" \
    -e "$STAFFER_FT_EPOCHS" --use-sampler \
    "$NAME-kernsheet"
fi

# --- Stage 3: noter PDMX base (trains directly on the KernSheet vocab) ---
stage "3/5  noter PDMX base -> checkpoints/noter/$NAME"
if done_ckpt noter "$NAME"; then
  echo "  exists — skipping (FORCE=1 to retrain)"
else
  tag "train/$NAME"
  run noter --log-file "logs/noter/$NAME.log" \
    train --vocab "$VOCAB" --jitter "$NOTER_JITTER" -e "$NOTER_BASE_EPOCHS" \
    "$NAME"
fi

# --- Stage 4: noter KernSheet fine-tune (no jitter) ---
stage "4/5  noter KernSheet fine-tune -> checkpoints/noter/$NAME-kernsheet"
if done_ckpt noter "$NAME-kernsheet"; then
  echo "  exists — skipping (FORCE=1 to retrain)"
else
  tag "train/$NAME-kernsheet"
  run noter --kern-home "$KS" --log-file "logs/noter/$NAME-kernsheet.log" \
    train --init-from "checkpoints/noter/$NAME/last.ckpt" \
    --train-len "$NOTER_FT_TRAIN" --valid-len "$NOTER_FT_VALID" \
    --lr "$NOTER_FT_LR" --warmup-steps "$NOTER_FT_WARMUP" \
    -e "$NOTER_FT_EPOCHS" -s 2.0 \
    "$NAME-kernsheet"
fi

# --- Stage 5: scorer merge + KernSheet fine-tune ---
stage "5/5  scorer KernSheet fine-tune -> checkpoints/scorer/$SCORER_NAME"
if done_ckpt scorer "$SCORER_NAME"; then
  echo "  exists — skipping (FORCE=1 to retrain)"
else
  tag "train/$SCORER_NAME"
  # No --vocab/--lr/--warmup-steps/--freeze-staffer-steps: ride the ScorerConfig
  # defaults (the default vocab path under --kern-home is the KernSheet vocab), as
  # the ravel run did.
  run scorer --kern-home "$KS" --log-file "logs/scorer/$SCORER_NAME.log" \
    train --staffer "$NAME-kernsheet" --noter "$NAME-kernsheet" \
    --train-len "$SCORER_FT_TRAIN" --valid-len "$SCORER_FT_VALID" \
    -e "$SCORER_EPOCHS" \
    "$SCORER_NAME"
fi

echo
echo "Done. Final scorer: checkpoints/scorer/$SCORER_NAME/last.ckpt"
echo "Update docs/{staffer,noter,scorer}-training.html with eval results."
