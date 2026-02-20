#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

STEPS="${STEPS:-100}"
DUMP_CONTROLS_EVERY="${DUMP_CONTROLS_EVERY:-20}"
COMPARE_FRAME="${COMPARE_FRAME:-32}"
RUN_MODE="${RUN_MODE:-ab}"  # a|b|ab
STAMP="$(date -u +%Y%m%d_%H%M%S)"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-train4_ab_${STEPS}_${STAMP}}"

if (( DUMP_CONTROLS_EVERY <= 0 )); then
  echo "DUMP_CONTROLS_EVERY must be > 0"
  exit 1
fi

case "${RUN_MODE}" in
  a|b|ab) ;;
  *)
    echo "RUN_MODE must be one of: a, b, ab"
    exit 1
    ;;
esac

BUILD_BEFORE_RUN="${BUILD_BEFORE_RUN:-1}"
echo "[run] steps=${STEPS} dump_controls_every=${DUMP_CONTROLS_EVERY} run_mode=${RUN_MODE} run_name_prefix=${RUN_NAME_PREFIX}"
echo "[run] build_before_run=${BUILD_BEFORE_RUN}"

if [[ "${BUILD_BEFORE_RUN}" == "1" ]]; then
  echo "[run] Building DDSPE2E (debug)..."
  swift build -c debug
elif [[ ! -x "./.build/debug/DDSPE2E" ]]; then
  echo "[run] DDSPE2E binary missing; building (debug)..."
  swift build -c debug
fi

COMPARE_LIST=()
for ((s = 0; s < STEPS; s += DUMP_CONTROLS_EVERY)); do
  COMPARE_LIST+=("${s}")
done
if [[ ${#COMPARE_LIST[@]} -eq 0 ]]; then
  COMPARE_LIST=("0")
fi
COMPARE_STEPS="$(IFS=,; echo "${COMPARE_LIST[*]}")"

run_variant() {
  local variant="$1"
  local label lr mse_weight spectral_logmag_weight clip_mode noise_filter render_wav run_name train_log run_dir
  local softmax_temp softmax_temp_end softmax_temp_warmup_steps softmax_temp_ramp_steps
  local softmax_amp_floor softmax_gain_min_db softmax_gain_max_db
  local harmonic_entropy_weight harmonic_entropy_weight_end
  local harmonic_entropy_warmup_steps harmonic_entropy_ramp_steps
  local harmonic_concentration_weight harmonic_concentration_weight_end
  local harmonic_concentration_warmup_steps harmonic_concentration_ramp_steps
  local early_stop_patience early_stop_min_delta
  local lr_schedule lr_half_life lr_warmup_steps lr_min
  local -a init_checkpoint_arg=()

  case "${variant}" in
    a)
      label="A (baseline)"
      lr="0.02"
      mse_weight="0"
      spectral_logmag_weight="${A_SPECTRAL_LOGMAG_WEIGHT:-0.0}"
      clip_mode="element"
      noise_filter="true"
      softmax_temp="1.0"
      softmax_temp_end="1.0"
      softmax_temp_warmup_steps="0"
      softmax_temp_ramp_steps="0"
      softmax_amp_floor="0.0"
      softmax_gain_min_db="-50"
      softmax_gain_max_db="6"
      harmonic_entropy_weight="0.0"
      harmonic_entropy_weight_end="0.0"
      harmonic_entropy_warmup_steps="0"
      harmonic_entropy_ramp_steps="0"
      harmonic_concentration_weight="0.0"
      harmonic_concentration_weight_end="0.0"
      harmonic_concentration_warmup_steps="0"
      harmonic_concentration_ramp_steps="0"
      early_stop_patience="0"
      early_stop_min_delta="0.0"
      lr_schedule="exp"
      lr_half_life="2000"
      lr_warmup_steps="0"
      lr_min="1e-4"
      render_wav="/Users/alecresende/Downloads/rendered_A.wav"
      run_name="${RUN_NAME_PREFIX}_A"
      ;;
    b)
      label="B (anti-collapse decay)"
      lr="${B_LR:-0.01}"
      mse_weight="${B_MSE_WEIGHT:-0.0}"
      spectral_logmag_weight="${B_SPECTRAL_LOGMAG_WEIGHT:-0.0}"
      clip_mode="${B_CLIP_MODE:-element}"
      noise_filter="${B_NOISE_FILTER:-true}"
      softmax_temp="${B_SOFTMAX_TEMP:-1.6}"
      softmax_temp_end="${B_SOFTMAX_TEMP_END:-${softmax_temp}}"
      softmax_temp_warmup_steps="${B_SOFTMAX_TEMP_WARMUP_STEPS:-0}"
      softmax_temp_ramp_steps="${B_SOFTMAX_TEMP_RAMP_STEPS:-0}"
      softmax_amp_floor="${B_SOFTMAX_AMP_FLOOR:-0.0}"
      softmax_gain_min_db="${B_SOFTMAX_GAIN_MIN_DB:--50}"
      softmax_gain_max_db="${B_SOFTMAX_GAIN_MAX_DB:-6}"
      harmonic_entropy_weight="${B_ENTROPY_WEIGHT:-0.03}"
      harmonic_entropy_weight_end="${B_ENTROPY_WEIGHT_END:-0.0}"
      harmonic_entropy_warmup_steps="${B_ENTROPY_WARMUP_STEPS:-10}"
      harmonic_entropy_ramp_steps="${B_ENTROPY_RAMP_STEPS:-50}"
      harmonic_concentration_weight="${B_CONCENTRATION_WEIGHT:-0.0}"
      harmonic_concentration_weight_end="${B_CONCENTRATION_WEIGHT_END:-${harmonic_concentration_weight}}"
      harmonic_concentration_warmup_steps="${B_CONCENTRATION_WARMUP_STEPS:-0}"
      harmonic_concentration_ramp_steps="${B_CONCENTRATION_RAMP_STEPS:-0}"
      early_stop_patience="${B_EARLY_STOP_PATIENCE:-0}"
      early_stop_min_delta="${B_EARLY_STOP_MIN_DELTA:-0.0}"
      lr_schedule="${B_LR_SCHEDULE:-exp}"
      lr_half_life="${B_LR_HALF_LIFE:-2000}"
      lr_warmup_steps="${B_LR_WARMUP_STEPS:-0}"
      lr_min="${B_LR_MIN:-1e-4}"
      render_wav="${B_RENDER_WAV:-/Users/alecresende/Downloads/rendered_B.wav}"
      run_name="${RUN_NAME_PREFIX}_B"
      ;;
    *)
      echo "Unknown variant: ${variant}"
      exit 1
      ;;
  esac

  if [[ "${variant}" == "b" && -n "${B_INIT_CHECKPOINT:-}" ]]; then
    init_checkpoint_arg=(--init-checkpoint "${B_INIT_CHECKPOINT}")
  fi

  train_log="$(mktemp -t ddspe2e_train4_${variant}.XXXXXX.log)"
  echo "[run:${variant}] ${label}"
  echo "[run:${variant}] lr=${lr} mse_weight=${mse_weight} spectral_logmag_weight=${spectral_logmag_weight} clip_mode=${clip_mode} noise_filter=${noise_filter}"
  echo "[run:${variant}] softmax_temp=${softmax_temp}->${softmax_temp_end} temp_warmup=${softmax_temp_warmup_steps} temp_ramp=${softmax_temp_ramp_steps} softmax_amp_floor=${softmax_amp_floor} softmax_gain_min_db=${softmax_gain_min_db} softmax_gain_max_db=${softmax_gain_max_db} entropy_weight=${harmonic_entropy_weight}->${harmonic_entropy_weight_end} entropy_warmup=${harmonic_entropy_warmup_steps} entropy_ramp=${harmonic_entropy_ramp_steps} concentration_weight=${harmonic_concentration_weight}->${harmonic_concentration_weight_end} concentration_warmup=${harmonic_concentration_warmup_steps} concentration_ramp=${harmonic_concentration_ramp_steps} early_stop=${early_stop_patience}/${early_stop_min_delta} lr_sched=${lr_schedule} hl=${lr_half_life} warmup=${lr_warmup_steps} lr_min=${lr_min}"
  if [[ ${#init_checkpoint_arg[@]} -gt 0 ]]; then
    echo "[run:${variant}] init_checkpoint=${B_INIT_CHECKPOINT}"
  fi
  echo "[run:${variant}] Capturing train output to ${train_log}"

  local -a train_cmd=(
    ./.build/debug/DDSPE2E train
    --cache .ddsp_cache_overfit1
    --mode m2
    --split train
    --steps "${STEPS}"
    --shuffle false
    --seed 1
    --lr "${lr}"
    --harmonic-head-mode softmax-db
    --softmax-temp "${softmax_temp}"
    --softmax-temp-end "${softmax_temp_end}"
    --softmax-temp-warmup-steps "${softmax_temp_warmup_steps}"
    --softmax-temp-ramp-steps "${softmax_temp_ramp_steps}"
    --softmax-amp-floor "${softmax_amp_floor}"
    --softmax-gain-min-db "${softmax_gain_min_db}"
    --softmax-gain-max-db "${softmax_gain_max_db}"
    --harmonic-entropy-weight "${harmonic_entropy_weight}"
    --harmonic-entropy-weight-end "${harmonic_entropy_weight_end}"
    --harmonic-entropy-warmup-steps "${harmonic_entropy_warmup_steps}"
    --harmonic-entropy-ramp-steps "${harmonic_entropy_ramp_steps}"
    --harmonic-concentration-weight "${harmonic_concentration_weight}"
    --harmonic-concentration-weight-end "${harmonic_concentration_weight_end}"
    --harmonic-concentration-warmup-steps "${harmonic_concentration_warmup_steps}"
    --harmonic-concentration-ramp-steps "${harmonic_concentration_ramp_steps}"
    --early-stop-patience "${early_stop_patience}"
    --early-stop-min-delta "${early_stop_min_delta}"
    --lr-schedule "${lr_schedule}"
    --lr-half-life "${lr_half_life}"
    --lr-warmup-steps "${lr_warmup_steps}"
    --lr-min "${lr_min}"
    --batch-size 1
    --grad-clip 1.0
    --clip-mode "${clip_mode}"
    --normalize-grad-by-frames false
    --mse-weight "${mse_weight}"
    --spectral-weight 1.0
    --spectral-logmag-weight "${spectral_logmag_weight}"
    --spectral-windows 64,128,256,512,1024
    --spectral-hop-divisor 4
    --spectral-warmup-steps 0
    --spectral-ramp-steps 0
    --log-every 1
    --model-hidden 128
    --noise-filter "${noise_filter}"
    --render-wav "${render_wav}"
    --harmonics 64
    --model-layers 2
    --dump-controls-every "${DUMP_CONTROLS_EVERY}"
    --run-name "${run_name}"
    --kernel-dump test_kernel_yes_gemm.metal
  )
  if [[ ${#init_checkpoint_arg[@]} -gt 0 ]]; then
    train_cmd+=("${init_checkpoint_arg[@]}")
  fi
  "${train_cmd[@]}" | tee "${train_log}"

  run_dir="$(grep -m1 '\[DDSPE2E\] Run directory:' "${train_log}" | sed 's/.*Run directory: //')"
  if [[ -z "${run_dir}" ]]; then
    run_dir="${ROOT_DIR}/runs/${run_name}"
  fi

  local compare_steps_effective=""
  for s in "${COMPARE_LIST[@]}"; do
    local tag
    tag="$(printf "%06d" "$s")"
    if [[ -f "${run_dir}/logs/controls/step_${tag}_control_summary.csv" ]]; then
      if [[ -n "${compare_steps_effective}" ]]; then
        compare_steps_effective+=","
      fi
      compare_steps_effective+="${s}"
    fi
  done
  if [[ -z "${compare_steps_effective}" ]]; then
    compare_steps_effective="$(ls -1 "${run_dir}/logs/controls"/step_*_control_summary.csv 2>/dev/null | sed -E 's/.*step_([0-9]{6})_control_summary\\.csv/\\1/' | sed 's/^0*//' | awk 'NF' | sort -n | paste -sd, -)"
  fi
  if [[ -z "${compare_steps_effective}" ]]; then
    compare_steps_effective="0"
  fi

  echo "[run:${variant}] Plotting controls for run: ${run_dir}"
  echo "[run:${variant}] compare_steps=${compare_steps_effective} compare_frame=${COMPARE_FRAME}"
  python3 Examples/DDSPE2E/scripts/plot_controls.py \
    --run "${run_dir}" \
    --steps "${compare_steps_effective}" \
    --compare-frame "${COMPARE_FRAME}"

  echo "[run:${variant}] Plot written under: ${run_dir}/logs/controls/plots/"
}

if [[ "${RUN_MODE}" == "a" || "${RUN_MODE}" == "ab" ]]; then
  run_variant a
fi
if [[ "${RUN_MODE}" == "b" || "${RUN_MODE}" == "ab" ]]; then
  run_variant b
fi

echo "[done] A/B run complete"
