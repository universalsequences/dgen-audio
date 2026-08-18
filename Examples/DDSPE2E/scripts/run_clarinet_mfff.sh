#!/usr/bin/env bash
# Clarinet-only mf/ff control experiment for the R8 reference-conditioning gate.
set -euo pipefail
BIN="${BIN:-./.build/debug/DDSPE2E}"
CACHE="${CACHE:-.ddsp_cache_clarinet_mfff}"
RUN_PREFIX="${RUN_PREFIX:-clarinet_mfff_solo}"
STEPS_A="${STEPS_A:-3000}"
STEPS_B="${STEPS_B:-800}"
STEPS_C="${STEPS_C:-400}"

[[ -f "$CACHE/manifest.json" ]] || { echo "Missing $CACHE/manifest.json" >&2; exit 1; }
[[ -x "$BIN" ]] || swift build -c debug --product DDSPE2E

common=(
  --cache "$CACHE" --mode m2 --split train --instruments 1
  --reference-conditioning false
  --shuffle true --fixed-batch false --seed 1 --batch-size 1 --grad-accum-steps 1
  --grad-clip 1 --clip-mode element --normalize-grad-by-frames false --mse-weight 0
  --spectral-weight 1 --spectral-logmag-weight 1 --spectral-loss-mode l1
  --spectral-log-epsilon 1e-3 --spectral-windows 64,128,256,512,1024
  --spectral-hop-divisor 4 --spectral-warmup-steps 0 --spectral-ramp-steps 0
  --model-hidden 128 --model-layers 2 --harmonics 64 --harmonic-head-mode exp-sigmoid
  --noise-filter true --noise-filter-mode fir --decoder-backbone transformer
  --transformer-d-model 64 --transformer-layers 2 --transformer-ff-multiplier 2
  --transformer-causal true --transformer-positional-encoding true --control-smoothing off
  --loudness-loss-mode db-l1 --best-metric spectral --best-eval-every 100
  --best-eval-chunks 7 --checkpoint-every 500 --log-every 50
)
A="${RUN_PREFIX}_stageA"; B="${RUN_PREFIX}_stageB"; C="${RUN_PREFIX}_stageC"
"$BIN" train "${common[@]}" --steps "$STEPS_A" \
  --lr 3e-4 --lr-schedule exp --lr-half-life 2000 --lr-min 1e-4 \
  --loudness-weight 0 --loudness-weight-end .05 --loudness-warmup-steps 10 --loudness-ramp-steps 40 \
  --run-name "$A"
"$BIN" train "${common[@]}" --steps "$STEPS_B" \
  --lr 3e-5 --lr-schedule exp --lr-half-life 120 --lr-min 3e-6 --loudness-weight .02 \
  --init-checkpoint "runs/$A/checkpoints/model_best.json" --run-name "$B"
"$BIN" train "${common[@]}" --steps "$STEPS_C" \
  --lr 1e-5 --lr-schedule exp --lr-half-life 80 --lr-min 1e-6 --loudness-weight 0 \
  --init-checkpoint "runs/$B/checkpoints/model_best.json" --run-name "$C"
echo "Final checkpoint: runs/$C/checkpoints/model_best.json"
