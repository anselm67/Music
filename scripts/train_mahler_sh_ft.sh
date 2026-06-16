#!/usr/bin/env bash
#
# One-off: re-run ONLY the three rehearsal-mix fine-tune stages after the
# KernSheet staff_height layout fix. That fix touches KernSheet box geometry
# only — the PDMX bases (staffer/mahler, noter/mahler) and the token vocab are
# unaffected — so we reuse the existing bases via --init-from and write the FTs
# to a fresh `mahler-sh` lineage. No `kernsheet make` / asset rebuild is needed
# (layout edits don't change build/png or build/tokens).
#
# Run BY THE USER from the repo root. Fail-fast; each stage is skipped if its
# checkpoint already exists. DRYRUN=1 prints the commands, FORCE=1 retrains
# stages whose checkpoint already exists.
#
#   scripts/train_mahler_sh_ft.sh
#   DRYRUN=1 scripts/train_mahler_sh_ft.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

KS=/home/anselm/datasets/KernSheet
VOCAB="$KS/build/vocab.json"
MIX=0.67

BASE=mahler          # reuse these PDMX bases (unchanged by the geometry fix)
NAME=mahler-sh       # new fine-tune lineage

run() {
  echo "+ $*"
  [[ "${DRYRUN:-0}" == 1 ]] || "$@"
}

tag() {
  if git rev-parse -q --verify "refs/tags/$1" >/dev/null; then
    echo "  tag $1 already exists — skipping"
  else
    run git tag "$1"
  fi
}

done_ckpt() {  # done_ckpt <branch> <name>
  [[ "${FORCE:-0}" != 1 && -f "checkpoints/$1/$2/last.ckpt" ]]
}

stage() {
  echo
  echo "======================================================================"
  echo "== $1"
  echo "======================================================================"
}

# --- prerequisites: the reused bases and the shared KernSheet vocab must exist ---
for f in "checkpoints/staffer/$BASE/last.ckpt" "checkpoints/noter/$BASE/last.ckpt" "$VOCAB"; do
  [[ -f "$f" ]] || { echo "missing prerequisite: $f" >&2; exit 1; }
done

# One tag covers all three FT runs (they share a commit).
tag "train/$NAME"

# --- Stage 1/3: staffer rehearsal-mix fine-tune ---
stage "1/3  staffer FT -> checkpoints/staffer/$NAME-mixed"
if done_ckpt staffer "$NAME-mixed"; then
  echo "  exists — skipping (FORCE=1 to retrain)"
else
  run staffer --log-file "logs/staffer/$NAME-mixed.log" \
    train --init-from "checkpoints/staffer/$BASE/last.ckpt" \
    --kern-sheet "$KS" "$MIX" \
    --train-len 3000 --valid-len 450 \
    --lr 1e-5 --warmup-steps 200 \
    -e 30 --use-sampler \
    "$NAME-mixed"
fi

# --- Stage 2/3: noter rehearsal-mix fine-tune ---
stage "2/3  noter FT -> checkpoints/noter/$NAME-mixed"
if done_ckpt noter "$NAME-mixed"; then
  echo "  exists — skipping (FORCE=1 to retrain)"
else
  run noter --log-file "logs/noter/$NAME-mixed.log" \
    train --init-from "checkpoints/noter/$BASE/last.ckpt" \
    --kern-sheet "$KS" "$MIX" --vocab "$VOCAB" \
    --train-len 23400 --valid-len 2600 \
    --lr 1e-4 --warmup-steps 200 \
    -e 12 -s 2.0 \
    "$NAME-mixed"
fi

# --- Stage 3/3: scorer merge + rehearsal-mix fine-tune ---
stage "3/3  scorer FT -> checkpoints/scorer/$NAME"
if done_ckpt scorer "$NAME"; then
  echo "  exists — skipping (FORCE=1 to retrain)"
else
  run scorer --log-file "logs/scorer/$NAME.log" \
    train --staffer "$NAME-mixed" --noter "$NAME-mixed" \
    --kern-sheet "$KS" "$MIX" --vocab "$VOCAB" \
    --train-len 2000 --valid-len 300 \
    -e 20 \
    "$NAME"
fi

echo
echo "Done. Final scorer: checkpoints/scorer/$NAME/last.ckpt"
echo "Next: scorer eval $NAME --seed 0 [--rerank]; update docs/{staffer,noter,scorer}-training.html."
