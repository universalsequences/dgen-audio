#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

STEPS="${STEPS:-40}"
DUMP_CONTROLS_EVERY="${DUMP_CONTROLS_EVERY:-10}"
COMPARE_STEPS="${COMPARE_STEPS:-}"
COMPARE_FRAME="${COMPARE_FRAME:-32}"
RUN_NAME="${RUN_NAME:-flute40_controls_$(date -u +%Y%m%d_%H%M%S)}"

echo "[run] steps=${STEPS} dump_controls_every=${DUMP_CONTROLS_EVERY} run_name=${RUN_NAME}"

if [[ ! -x "./.build/debug/DDSPE2E" ]]; then
  echo "[run] Building DDSPE2E (debug)..."
  swift build -c debug
fi

TRAIN_LOG="$(mktemp -t ddspe2e_flute_train.XXXXXX.log)"
echo "[run] Capturing train output to ${TRAIN_LOG}"

./.build/debug/DDSPE2E train \
  --cache .ddsp_cache_tinysol_large256 \
  --mode m2 \
  --split train \
  --steps "${STEPS}" \
  --shuffle false \
  --seed 1 \
  --lr 0.1 \
  --lr-schedule exp \
  --lr-half-life 200 \
  --lr-min 4e-2 \
  --batch-size 8 \
  --grad-clip 1.0 \
  --normalize-grad-by-frames false \
  --mse-weight 0 \
  --spectral-weight 1.0 \
  --spectral-windows 512,1024 \
  --spectral-hop-divisor 4 \
  --spectral-warmup-steps 0 \
  --spectral-ramp-steps 0 \
  --log-every 1 \
  --model-hidden 128 \
  --model-layers 2 \
  --harmonics 64 \
  --noise-filter false \
  --render-every 10 \
  --render-wav /Users/alecresende/Downloads/rendered_flute.wav \
  --dump-controls-every "${DUMP_CONTROLS_EVERY}" \
  --run-name "${RUN_NAME}" \
  --kernel-dump "test_kernel_32_batch.metal" | tee "${TRAIN_LOG}"

RUN_DIR="$(grep -m1 '\[DDSPE2E\] Run directory:' "${TRAIN_LOG}" | sed 's/.*Run directory: //')"
if [[ -z "${RUN_DIR}" ]]; then
  RUN_DIR="${ROOT_DIR}/runs/${RUN_NAME}"
fi

echo "[run] Plotting controls for run: ${RUN_DIR}"
if [[ -z "${COMPARE_STEPS}" ]]; then
  # Default comparison steps every 10 steps (bounded by STEPS-1).
  COMPARE_LIST=()
  for s in 0 10 20 30 40; do
    if (( s < STEPS )); then
      COMPARE_LIST+=("${s}")
    fi
  done
  if [[ ${#COMPARE_LIST[@]} -eq 0 ]]; then
    COMPARE_LIST=("0")
  fi
  COMPARE_STEPS="$(IFS=,; echo "${COMPARE_LIST[*]}")"
fi

echo "[run] compare_steps=${COMPARE_STEPS} compare_frame=${COMPARE_FRAME}"
python3 Examples/DDSPE2E/scripts/plot_controls.py \
  --run "${RUN_DIR}" \
  --steps "${COMPARE_STEPS}" \
  --compare-frame "${COMPARE_FRAME}"

echo "[done] Plot written under: ${RUN_DIR}/logs/controls/plots/"
