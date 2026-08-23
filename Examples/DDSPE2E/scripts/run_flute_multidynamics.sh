#!/usr/bin/env bash
# R6 recipe scaled to chromatic C4-C7 TinySOL flute at pp/mf/ff.
set -euo pipefail

BIN="${BIN:-./.build/debug/DDSPE2E}"
CACHE="${CACHE:-.ddsp_cache_flute_multidynamics}"
RUN_PREFIX="${RUN_PREFIX:-flute_multidynamics}"
STEPS_A="${STEPS_A:-10000}"
STEPS_B="${STEPS_B:-3000}"
STEPS_C="${STEPS_C:-2000}"

if [[ ! -f "$CACHE/manifest.json" ]]; then
  echo "Missing $CACHE/manifest.json; run Examples/DDSPE2E/scripts/prepare_flute_multidynamics.sh first" >&2
  exit 1
fi
if [[ ! -x "$BIN" ]]; then
  swift build -c debug --product DDSPE2E
fi

common=(
  --cache "$CACHE" --mode m2 --split train
  --shuffle true --fixed-batch false --seed 1
  --batch-size 1 --grad-accum-steps 1
  --grad-clip 1.0 --clip-mode element --normalize-grad-by-frames false
  --mse-weight 0
  --spectral-weight 1.0 --spectral-logmag-weight 1.0
  --spectral-loss-mode l1 --spectral-log-epsilon 1e-3
  --spectral-windows 64,128,256,512,1024 --spectral-hop-divisor 4
  --spectral-warmup-steps 0 --spectral-ramp-steps 0
  --model-hidden 128 --harmonics 64 --harmonic-head-mode exp-sigmoid
  --noise-filter true --noise-filter-mode fir --model-layers 2
  --decoder-backbone transformer --transformer-d-model 64 --transformer-layers 2
  --transformer-ff-multiplier 2 --transformer-causal true
  --transformer-positional-encoding true --control-smoothing off
  --loudness-loss-mode db-l1 --best-metric spectral
  --best-eval-every 25 --best-eval-chunks 12
)

stage_a="${RUN_PREFIX}_stageA"
stage_b="${RUN_PREFIX}_stageB"
stage_c="${RUN_PREFIX}_stageC"

"$BIN" train "${common[@]}" \
  --steps "$STEPS_A" \
  --lr 3e-4 --lr-schedule exp --lr-half-life 2000 --lr-min 1e-4 \
  --loudness-weight 0 --loudness-weight-end 0.05 \
  --loudness-warmup-steps 10 --loudness-ramp-steps 40 \
  --checkpoint-every 500 --log-every 50 --run-name "$stage_a"

"$BIN" train "${common[@]}" \
  --steps "$STEPS_B" \
  --lr 3e-5 --lr-schedule exp --lr-half-life 120 --lr-min 3e-6 \
  --loudness-weight 0.02 \
  --init-checkpoint "runs/$stage_a/checkpoints/model_best.json" \
  --checkpoint-every 250 --log-every 50 --run-name "$stage_b"

"$BIN" train "${common[@]}" \
  --steps "$STEPS_C" \
  --lr 1e-5 --lr-schedule exp --lr-half-life 80 --lr-min 1e-6 \
  --loudness-weight 0 \
  --init-checkpoint "runs/$stage_b/checkpoints/model_best.json" \
  --checkpoint-every 250 --log-every 50 --run-name "$stage_c"

echo "Final checkpoint: runs/$stage_c/checkpoints/model_best.json"
echo "Render validation:"
echo "$BIN render-checkpoint-batch --cache $CACHE --split val --batch-size 12 --init-checkpoint runs/$stage_c/checkpoints/model_best.json --output runs/${RUN_PREFIX}_val_renders"
