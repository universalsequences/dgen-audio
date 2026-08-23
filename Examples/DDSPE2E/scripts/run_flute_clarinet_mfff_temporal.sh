#!/usr/bin/env bash
# R8 second attempt: temporal reference encoder on the mf/ff flute+clarinet
# cache. Replaces the averaged-MFCC bottleneck with [16x48] log-mel frames
# through a learned attention-pooled encoder, adds direct z→control residuals,
# and selects checkpoints on reconstruction MINUS the crossed-reference margin
# so a reference-blind model can never be "best".
set -euo pipefail
BIN="${BIN:-./.build/debug/DDSPE2E}"
CACHE="${CACHE:-.ddsp_cache_flute_clarinet_mfff}"
RUN_PREFIX="${RUN_PREFIX:-flute_clarinet_mfff_tref}"
STEPS_A="${STEPS_A:-3000}"
STEPS_B="${STEPS_B:-800}"
STEPS_C="${STEPS_C:-400}"
PRETRAIN_STEPS="${PRETRAIN_STEPS:-800}"
# z→control dynamic range (R8 third attempt): at Z_SCALE=1 the tanh-bounded z
# through 0.1-scale residual weights reaches only ~2 dB of harmonic swing;
# flute↔clarinet H2 contrast needs ~30 dB (±3 exp-sigmoid logits ≈ ±26 dB).
Z_SCALE="${Z_SCALE:-8}"
FILM_GAMMA="${FILM_GAMMA:-1.0}"

[[ -f "$CACHE/manifest.json" ]] || { echo "Missing $CACHE/manifest.json" >&2; exit 1; }
[[ -x "$BIN" ]] || swift build -c debug --product DDSPE2E

common=(
  --cache "$CACHE" --mode m2 --split train --instruments 2
  --reference-conditioning true --reference-encoder temporal
  --reference-time-frames 16 --reference-mel-bins 48 --reference-encoder-hidden 64
  --reference-latent 32 --reference-classification-weight 1.0
  --reference-separation-weight 1.0 --reference-encoder-freeze true
  --reference-z-scale "$Z_SCALE" --reference-film-gamma "$FILM_GAMMA"
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
  --best-eval-chunks 10 --checkpoint-every 500 --log-every 50
)
A="${RUN_PREFIX}_stageA"; B="${RUN_PREFIX}_stageB"; C="${RUN_PREFIX}_stageC"
# Stage A pretrains encoder+classifier discriminatively (joint training lets
# the reconstruction gradient drown the classification gradient); B/C resume
# from A's checkpoint with the encoder kept frozen.
"$BIN" train "${common[@]}" --steps "$STEPS_A" \
  --reference-pretrain-steps "$PRETRAIN_STEPS" --reference-pretrain-lr 3e-3 \
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
echo "Listening gate: $BIN render-reference-triplets --cache $CACHE \\"
echo "  --init-checkpoint runs/$C/checkpoints/model_best.json --split val --count 6 \\"
echo "  --output runs/${RUN_PREFIX}_triplets"
