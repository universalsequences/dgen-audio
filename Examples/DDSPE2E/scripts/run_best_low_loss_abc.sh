#!/usr/bin/env bash
set -euo pipefail

# Best-known low-loss A/B/C training chain for DDSPE2E transformer runs.
#
# Default stages:
#   A) db-l1 loudness schedule (0 -> 0.05)
#   B) low-LR continuation with light loudness (0.02)
#   C) spectral-only polish (loudness off)
#   P) spectral-only probe from Stage C best checkpoint
#
# Usage:
#   bash Examples/DDSPE2E/scripts/run_best_low_loss_abc.sh
#
# Common overrides:
#   CACHE=.ddsp_cache_overfit1 RUN_PREFIX=my_run BUILD=1 \
#   STEPS_A=500 STEPS_B=300 STEPS_C=300 \
#   bash Examples/DDSPE2E/scripts/run_best_low_loss_abc.sh
#
# Skip stages (must provide/ensure checkpoints exist):
#   RUN_STAGE_A=0 STAGE_A_CHECKPOINT=runs/existing/checkpoints/model_best.json ...

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

BIN="${BIN:-./.build/debug/DDSPE2E}"
CACHE="${CACHE:-.ddsp_cache_overfit1}"
SEED="${SEED:-1}"
BUILD="${BUILD:-1}"

RUN_PREFIX="${RUN_PREFIX:-best_low_loss_$(date -u +%Y%m%d_%H%M%S)}"
A_RUN="${A_RUN:-${RUN_PREFIX}_stageA}"
B_RUN="${B_RUN:-${RUN_PREFIX}_stageB}"
C_RUN="${C_RUN:-${RUN_PREFIX}_stageC}"
P_RUN="${P_RUN:-${RUN_PREFIX}_probe}"

RUN_STAGE_A="${RUN_STAGE_A:-1}"
RUN_STAGE_B="${RUN_STAGE_B:-1}"
RUN_STAGE_C="${RUN_STAGE_C:-1}"
RUN_PROBE="${RUN_PROBE:-1}"

STEPS_A="${STEPS_A:-500}"
STEPS_B="${STEPS_B:-300}"
STEPS_C="${STEPS_C:-300}"

mkdir -p runs

if [[ "$BUILD" == "1" ]]; then
  echo "[build] swift build -c debug"
  swift build -c debug
fi

if [[ ! -x "$BIN" ]]; then
  echo "[error] DDSPE2E binary not found at: $BIN"
  echo "Run with BUILD=1 or set BIN=... to a valid executable."
  exit 1
fi

run_train() {
  local run_name="$1"
  shift
  local log_path="runs/${run_name}.log"
  echo "[train] run=${run_name}"
  "$BIN" train "$@" --run-name "$run_name" 2>&1 | tee "$log_path"
}

TRAIN_COMMON=(
  --cache "$CACHE"
  --mode m2
  --split train
  --shuffle false
  --fixed-batch true
  --seed "$SEED"
  --batch-size 1
  --grad-accum-steps 1
  --grad-clip 1.0
  --clip-mode element
  --normalize-grad-by-frames false
  --mse-weight 0
  --spectral-weight 1.0
  --spectral-logmag-weight 1.0
  --spectral-loss-mode l1
  --spectral-windows 64,128,256,512,1024
  --spectral-hop-divisor 4
  --spectral-warmup-steps 0
  --spectral-ramp-steps 0
  --model-hidden 128
  --harmonics 64
  --harmonic-head-mode exp-sigmoid
  --noise-filter true
  --model-layers 2
  --decoder-backbone transformer
  --transformer-d-model 64
  --transformer-layers 2
  --transformer-ff-multiplier 2
  --transformer-causal true
  --transformer-positional-encoding true
  --control-smoothing off
  --loudness-loss-mode db-l1
  --dump-controls-every 0
)

if [[ "$RUN_STAGE_A" == "1" ]]; then
  run_train "$A_RUN" \
    "${TRAIN_COMMON[@]}" \
    --steps "$STEPS_A" \
    --lr 3e-4 \
    --lr-schedule exp \
    --lr-half-life 2000 \
    --lr-warmup-steps 0 \
    --lr-min 1e-4 \
    --log-every 25 \
    --loudness-weight 0.0 \
    --loudness-weight-end 0.05 \
    --loudness-warmup-steps 10 \
    --loudness-ramp-steps 40
fi

A_CKPT="${STAGE_A_CHECKPOINT:-runs/${A_RUN}/checkpoints/model_best.json}"
if [[ ! -f "$A_CKPT" ]]; then
  echo "[error] Stage A checkpoint missing: $A_CKPT"
  exit 1
fi

if [[ "$RUN_STAGE_B" == "1" ]]; then
  run_train "$B_RUN" \
    "${TRAIN_COMMON[@]}" \
    --steps "$STEPS_B" \
    --lr 3e-5 \
    --lr-schedule exp \
    --lr-half-life 120 \
    --lr-warmup-steps 0 \
    --lr-min 3e-6 \
    --log-every 30 \
    --loudness-weight 0.02 \
    --init-checkpoint "$A_CKPT"
fi

B_CKPT="${STAGE_B_CHECKPOINT:-runs/${B_RUN}/checkpoints/model_best.json}"
if [[ ! -f "$B_CKPT" ]]; then
  echo "[error] Stage B checkpoint missing: $B_CKPT"
  exit 1
fi

if [[ "$RUN_STAGE_C" == "1" ]]; then
  run_train "$C_RUN" \
    "${TRAIN_COMMON[@]}" \
    --steps "$STEPS_C" \
    --lr 1e-5 \
    --lr-schedule exp \
    --lr-half-life 80 \
    --lr-warmup-steps 0 \
    --lr-min 1e-6 \
    --log-every 30 \
    --loudness-weight 0.0 \
    --init-checkpoint "$B_CKPT"
fi

C_CKPT="${STAGE_C_CHECKPOINT:-runs/${C_RUN}/checkpoints/model_best.json}"
if [[ ! -f "$C_CKPT" ]]; then
  echo "[error] Stage C checkpoint missing: $C_CKPT"
  exit 1
fi

if [[ "$RUN_PROBE" == "1" ]]; then
  run_train "$P_RUN" \
    "${TRAIN_COMMON[@]}" \
    --steps 1 \
    --lr 1e-9 \
    --lr-schedule none \
    --loudness-weight 0.0 \
    --log-every 1 \
    --init-checkpoint "$C_CKPT"
fi

echo
echo "[summary]"
echo "A checkpoint: $A_CKPT"
echo "B checkpoint: $B_CKPT"
echo "C checkpoint: $C_CKPT"
if [[ "$RUN_PROBE" == "1" ]]; then
  PROBE_SUMMARY="runs/${P_RUN}/logs/train_summary.json"
  if [[ -f "$PROBE_SUMMARY" ]]; then
    PROBE_LOSS="$(jq -r '.finalLoss' "$PROBE_SUMMARY")"
    echo "probe run: $P_RUN"
    echo "probe loss: $PROBE_LOSS"
  else
    echo "probe run: $P_RUN (summary missing)"
  fi
fi
