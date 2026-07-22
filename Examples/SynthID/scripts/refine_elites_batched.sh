#!/bin/bash
# Batched replacement for refine_elites.sh: refines ALL basin-search elites
# simultaneously as one [B]-batched lane-parallel Adam trajectory
# (elites x jittered restarts, B >= 32 per the parallel-lanes spec decision),
# instead of one serial 1400-epoch trajectory per elite.
#
# Selection is per-lane CPU mrstft; winners are re-scored with the serial
# production loss and reported as a ratio against the seed's canonical COLD
# baseline (never an elite's own start). See BATCH_REFINE_FINDING.md for the
# measured comparison against the serial schedule (~7x per lane-step; equal
# or better final ratios in ~1/3 the steps).
#
# Usage: refine_elites_batched.sh <search-out-dir> <target.wav> <cold-baseline-loss> [out-dir]
set -euo pipefail

SEARCH_DIR="$1"
TARGET="$2"
BASELINE="$3"
OUT="${4:-$SEARCH_DIR/refine_batched}"
BIN=".build/release/SynthID"

"$BIN" batch-refine --mode polish \
  --target "$TARGET" \
  --elites "$SEARCH_DIR/elites" \
  --restarts 6 --jitter 0.05 \
  --smooth-steps 150 --steps 350 \
  --baseline "$BASELINE" --log-every 25 \
  --out "$OUT"
