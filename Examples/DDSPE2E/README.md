# DDSPE2E Example

`DDSPE2E` is an executable example target that demonstrates an end-to-end DDSP-style pipeline:

- preprocess `.wav` audio into cached chunks + features (`f0`, loudness, uv)
- inspect cached chunks
- run training (`dry` scaffold mode or `m2` decoder-only mode)

This example is wired in `Package.swift`, so you can run it with `swift run DDSPE2E ...`.

## Requirements

- macOS with Metal support (training uses DGen Metal backend)
- Swift toolchain matching this repo
- input dataset as `.wav` files (recursive directory scan)

## Quick Start

From repo root:

```bash
swift build
```

1. Dump a default config:

```bash
swift run DDSPE2E dump-config --output ddsp_config.json
```

2. Preprocess a small subset of `.wav` files:

```bash
swift run DDSPE2E preprocess \
  --input datasets/tinysol \
  --cache .ddsp_cache_tinysol \
  --max-files 8 \
  --max-chunks-per-file 8
```

3. Inspect cache:

```bash
swift run DDSPE2E inspect-cache --cache .ddsp_cache_tinysol --limit 5
```

4. Dry run (pipeline sanity):

```bash
swift run DDSPE2E train \
  --cache .ddsp_cache_tinysol \
  --mode dry \
  --steps 20
```

5. Decoder-only training (MSE only):

```bash
swift run DDSPE2E train \
  --cache .ddsp_cache_tinysol \
  --mode m2 \
  --steps 50 \
  --spectral-weight 0
```

6. Decoder-only training with spectral loss + hop gating:

```bash
swift run DDSPE2E train \
  --cache .ddsp_cache_tinysol \
  --mode m2 \
  --steps 50 \
  --spectral-windows 64,128,256 \
  --spectral-weight 1.0 \
  --spectral-hop-divisor 4
```

7. One-step gradient probe from a saved checkpoint (MSE vs spectral):

```bash
# MSE-only contribution at a fixed model state
swift run DDSPE2E train \
  --cache .ddsp_cache_tinysol \
  --mode m2 \
  --steps 1 \
  --lr 1e-9 \
  --init-checkpoint runs/<run>/checkpoints/model_step_00001200.json \
  --mse-weight 1.0 \
  --spectral-weight 0

# Spectral-only contribution at the same model state
swift run DDSPE2E train \
  --cache .ddsp_cache_tinysol \
  --mode m2 \
  --steps 1 \
  --lr 1e-9 \
  --init-checkpoint runs/<run>/checkpoints/model_step_00001200.json \
  --mse-weight 0 \
  --spectral-weight 1.0 \
  --spectral-windows 64,128,256 \
  --spectral-hop-divisor 4 \
  --spectral-warmup-steps 0 \
  --spectral-ramp-steps 0
```

8. Dump decoder controls + synthetic wavetable snapshots for debugging:

```bash
swift run DDSPE2E train \
  --cache .ddsp_cache_tinysol \
  --mode m2 \
  --steps 20 \
  --dump-controls-every 5
```

This writes CSV files under `runs/<run>/logs/controls/`:
- `*_control_summary.csv` (per-frame f0/uv + harmonic/noise gains + harmonic stats)
- `*_b0_f*_harmonics.csv` (harmonic amplitudes for selected frames)
- `*_b0_f*_wavetable.csv` (one-cycle additive wavetable from harmonic amplitudes)
- `*_b0_f*_noise_filter.csv` (when noise filter is enabled)

Quick plot helper:

```bash
python3 Examples/DDSPE2E/scripts/plot_controls.py \
  --run runs/<run-name> \
  --step latest
```

This writes `runs/<run-name>/logs/controls/plots/step_xxxxxx_controls.png`.

Compare control evolution across multiple training steps:

```bash
python3 Examples/DDSPE2E/scripts/plot_controls.py \
  --run runs/<run-name> \
  --steps 0,10,20,50 \
  --compare-frame 32
```

This writes `runs/<run-name>/logs/controls/plots/steps_xxxxxx_xxxxxx_controls_compare.png`.

9. Probe raw vs FIR-smoothed controls (realized tensors, no training):

```bash
swift run DDSPE2E probe-smoothing \
  --cache .ddsp_cache_tinysol \
  --split train \
  --index 0 \
  --output /tmp/ddsp_smoothing_probe
```

Analyze with Python:

```bash
python3 Examples/DDSPE2E/scripts/analyze_smoothing_probe.py \
  --dir /tmp/ddsp_smoothing_probe
```

## Spectral Hop Divisor

For each spectral window `w`, the effective hop is:

`hop = max(1, w / spectralHopDivisor)`

- larger divisor -> smaller hop -> denser spectral eval -> slower, potentially stronger spectral supervision
- smaller divisor -> larger hop -> sparser spectral eval -> faster

Set `--spectral-hop-divisor 1` to effectively disable hop sparsification for that window formula.

## Spectral Log-Magnitude Term

Use `--spectral-logmag-weight <float>` to add an FFT-domain log-magnitude term:

- per-bin term:
  - `--spectral-loss-mode l2` (default): `(log(|X_pred| + eps) - log(|X_tgt| + eps))^2`
  - `--spectral-loss-mode l1`: `|log(|X_pred| + eps) - log(|X_tgt| + eps)|`
- computed in spectral space (not `log(abs(waveform))` in time domain)
- combines with `--spectral-weight` (linear magnitude term) and `--mse-weight`
- for DDSP paper parity, `--spectral-loss-mode l1` is typically the right choice

## Loudness Envelope Loss

Use `--loudness-weight <float>` to add a frame-envelope reconstruction term that supervises decoder gains.

- mode:
  - `--loudness-loss-mode linear-l2` (default): L2 in linear-gain domain
  - `--loudness-loss-mode db-l1`: L1 in normalized dB domain (more robust in current transformer regime)
- optional schedule:
  - `--loudness-weight-end <float>`
  - `--loudness-warmup-steps <int>`
  - `--loudness-ramp-steps <int>`

Needle-moving preset (measured on February 21, 2026):
- `--loudness-weight 0.2 --loudness-loss-mode db-l1`
- overfit probe improved from `3.265924e-4` to `2.5940128e-4` (about `20.6%` lower)

### Best Known Low-Loss Script (A/B/C Staging)

The current best overfit probe was reached with a 3-stage sequence that combines:
- `db-l1` loudness supervision
- scheduled loudness warmup in Stage A
- low-LR continuation in Stage B
- spectral-only polish in Stage C

Reference best probe (lower is better):
- `0.00023486369` from `runs/probe_phasec_from_best_lr1e5/logs/train_summary.json`

Ready-to-run script:

```bash
bash Examples/DDSPE2E/scripts/run_best_low_loss_abc.sh
```

The script supports env overrides (for example `CACHE=...`, `RUN_PREFIX=...`, `STEPS_A=...`) and can skip stages via `RUN_STAGE_A/B/C/PROBE`.

### One-Command Auto A/B/C (Flag)

You can run the same 3-stage chain directly via `train`:

```bash
swift run DDSPE2E train \
  --cache .ddsp_cache_overfit1 \
  --mode m2 \
  --split train \
  --auto-abc true \
  --auto-abc-preset best-low-loss \
  --steps 500 \
  --auto-abc-steps-b 300 \
  --auto-abc-steps-c 300 \
  --run-name autoabc_trial
```

Behavior:
- Stage A/B/C are launched sequentially, each stage initialized from previous stage `model_best.json`.
- Stage-local best checkpoint detection uses early-stop patience + min-delta.
- Writes aggregate summary to `runs/<run-prefix>_auto_abc_summary.json`.
- `--init-checkpoint` (optional) is applied to Stage A start only.
- `--auto-abc-preset best-low-loss` applies the full transformer/spectral/fixed-batch baseline used by the best-known A/B/C chain.

What exactly happens for this command:
1. Stage A runs with the preset baseline plus Stage A overrides.
2. Stage A tracks its own best checkpoint (`model_best.json`) by minimum training loss.
3. Stage B starts from `StageA/model_best.json` (not from Stage A final step).
4. Stage B uses a different objective mix (`loudness-weight=0.02`) and lower LR; because the objective changed, the reported loss can jump higher than where Stage A ended.
5. Stage B still tracks its own best checkpoint and stops early if it stops improving by `patience/min-delta`.
6. Stage C starts from `StageB/model_best.json` (not from Stage B final step).
7. Stage C returns to spectral-only polish (`loudness-weight=0.0`) at very low LR to refine parameters after the Stage B bridge step.
8. Final summary (`runs/<run-prefix>_auto_abc_summary.json`) reports all stage minima and a recommended checkpoint (the lowest `minLoss` across A/B/C).

Why this can break plateau behavior:
- Stage A often finds a good basin quickly but can drift afterward.
- Stage B changes gradient pressure (especially envelope supervision) to move parameters away from that local regime.
- Stage C removes the bridge term and re-optimizes the main spectral target from B's best point.

Run from repo root:

```bash
#!/usr/bin/env bash
set -euo pipefail

BIN="./.build/debug/DDSPE2E"
CACHE=".ddsp_cache_overfit1"

swift build -c debug

# Stage A: main run (db-l1 loudness schedule 0 -> 0.05)
$BIN train \
  --cache "$CACHE" --mode m2 --split train \
  --steps 500 --shuffle false --fixed-batch true --seed 1 \
  --lr 3e-4 --lr-schedule exp --lr-half-life 2000 --lr-warmup-steps 0 --lr-min 1e-4 \
  --batch-size 1 --grad-accum-steps 1 \
  --grad-clip 1.0 --clip-mode element --normalize-grad-by-frames false \
  --mse-weight 0 \
  --spectral-weight 1.0 --spectral-logmag-weight 1.0 --spectral-loss-mode l1 \
  --spectral-windows 64,128,256,512,1024 --spectral-hop-divisor 4 \
  --spectral-warmup-steps 0 --spectral-ramp-steps 0 \
  --model-hidden 128 --harmonics 64 --harmonic-head-mode exp-sigmoid \
  --noise-filter true --model-layers 2 \
  --decoder-backbone transformer --transformer-d-model 64 --transformer-layers 2 \
  --transformer-ff-multiplier 2 --transformer-causal true --transformer-positional-encoding true \
  --control-smoothing off \
  --loudness-loss-mode db-l1 \
  --loudness-weight 0.0 --loudness-weight-end 0.05 \
  --loudness-warmup-steps 10 --loudness-ramp-steps 40 \
  --run-name long_db1_sched005_s500

# Stage B: continue from Stage A best checkpoint at lower LR and lighter loudness
$BIN train \
  --cache "$CACHE" --mode m2 --split train \
  --steps 300 --shuffle false --fixed-batch true --seed 1 \
  --lr 3e-5 --lr-schedule exp --lr-half-life 120 --lr-warmup-steps 0 --lr-min 3e-6 \
  --batch-size 1 --grad-accum-steps 1 \
  --grad-clip 1.0 --clip-mode element --normalize-grad-by-frames false \
  --mse-weight 0 \
  --spectral-weight 1.0 --spectral-logmag-weight 1.0 --spectral-loss-mode l1 \
  --spectral-windows 64,128,256,512,1024 --spectral-hop-divisor 4 \
  --spectral-warmup-steps 0 --spectral-ramp-steps 0 \
  --model-hidden 128 --harmonics 64 --harmonic-head-mode exp-sigmoid \
  --noise-filter true --model-layers 2 \
  --decoder-backbone transformer --transformer-d-model 64 --transformer-layers 2 \
  --transformer-ff-multiplier 2 --transformer-causal true --transformer-positional-encoding true \
  --control-smoothing off \
  --loudness-loss-mode db-l1 --loudness-weight 0.02 \
  --init-checkpoint runs/long_db1_sched005_s500/checkpoints/model_best.json \
  --run-name cont_from_best62_lr3e5_lw002

# Stage C: spectral-only polish from Stage B best checkpoint
$BIN train \
  --cache "$CACHE" --mode m2 --split train \
  --steps 300 --shuffle false --fixed-batch true --seed 1 \
  --lr 1e-5 --lr-schedule exp --lr-half-life 80 --lr-warmup-steps 0 --lr-min 1e-6 \
  --batch-size 1 --grad-accum-steps 1 \
  --grad-clip 1.0 --clip-mode element --normalize-grad-by-frames false \
  --mse-weight 0 \
  --spectral-weight 1.0 --spectral-logmag-weight 1.0 --spectral-loss-mode l1 \
  --spectral-windows 64,128,256,512,1024 --spectral-hop-divisor 4 \
  --spectral-warmup-steps 0 --spectral-ramp-steps 0 \
  --model-hidden 128 --harmonics 64 --harmonic-head-mode exp-sigmoid \
  --noise-filter true --model-layers 2 \
  --decoder-backbone transformer --transformer-d-model 64 --transformer-layers 2 \
  --transformer-ff-multiplier 2 --transformer-causal true --transformer-positional-encoding true \
  --control-smoothing off \
  --loudness-loss-mode db-l1 --loudness-weight 0.0 \
  --init-checkpoint runs/cont_from_best62_lr3e5_lw002/checkpoints/model_best.json \
  --run-name phasec_from_best_lr1e5

# Probe final checkpoint under spectral-only metric
$BIN train \
  --cache "$CACHE" --mode m2 --split train \
  --steps 1 --shuffle false --fixed-batch true --seed 1 \
  --lr 1e-9 --lr-schedule none \
  --batch-size 1 --grad-accum-steps 1 \
  --grad-clip 1.0 --clip-mode element --normalize-grad-by-frames false \
  --mse-weight 0 \
  --spectral-weight 1.0 --spectral-logmag-weight 1.0 --spectral-loss-mode l1 \
  --spectral-windows 64,128,256,512,1024 --spectral-hop-divisor 4 \
  --spectral-warmup-steps 0 --spectral-ramp-steps 0 \
  --model-hidden 128 --harmonics 64 --harmonic-head-mode exp-sigmoid \
  --noise-filter true --model-layers 2 \
  --decoder-backbone transformer --transformer-d-model 64 --transformer-layers 2 \
  --transformer-ff-multiplier 2 --transformer-causal true --transformer-positional-encoding true \
  --control-smoothing off \
  --loudness-weight 0.0 \
  --init-checkpoint runs/phasec_from_best_lr1e5/checkpoints/model_best.json \
  --run-name probe_phasec_from_best_lr1e5

jq -r '.finalLoss' runs/probe_phasec_from_best_lr1e5/logs/train_summary.json
```

Method summary (what got us here):
- Replaced hard noise UV-gating with continuous noise path.
- Added loudness envelope reconstruction as an auxiliary loss.
- Switched loudness reconstruction from `linear-l2` to `db-l1` (normalized dB-domain L1), which gave the largest single jump.
- Found the best Stage A loudness schedule in this regime: `0 -> 0.05` with warmup/ramp.
- Added Stage B low-LR continuation (`3e-5`, `loudness=0.02`) to keep improving after the early minimum.
- Added Stage C spectral-only polish (`1e-5`, `loudness=0`) to avoid late mixed-objective drift.

## Fixed Batch

Use `--fixed-batch true` to reuse the same sampled chunk set every training step.

- with `batch-size = 1`: repeats one chunk each step
- with `batch-size = B`: repeats the same `B` chunk IDs each step
- useful for sanity checks and optimization A/B tests where batch difficulty should not change over time

## Decoder Backbone

By default, training uses the transformer backbone.

- `--decoder-backbone transformer` (default): temporal backbone used for current best results
- `--decoder-backbone mlp` (legacy): older framewise baseline path kept for compatibility and A/B checks

### Architecture At A Glance (Pseudocode)

```text
conditioning features [F,5]
  = [f0_norm, loudness_norm, uv, delta_f0, delta_loudness]
        |
        v
backbone (selected by --decoder-backbone)
  - transformer: input projection + N transformer blocks
  - mlp (legacy): stacked dense+tanh layers
        |
        v
hidden [F,H]
        |
        v
heads (shared for both backbones)
  harm_logits  = hidden @ W_harm  + b_harm   -> harmonicAmps [F,K]
  hgain_logits = hidden @ W_hgain + b_hgain  -> harmonicGain [F,1]
  ngain_logits = hidden @ W_noise + b_noise  -> noiseGain [F,1]
  filter_logits(optional) -> noiseFilter [F,Kf]
        |
        v
activation mapping by harmonic-head-mode
  -> DecoderControls
        |
        v
DDSPSynth.renderSignal(controls, f0, uv, ...)
        |
        v
predicted audio -> training losses -> gradients back through synth, heads, backbone
```

Transformer block used in this project:

```text
input x [F,H]

attn_input = LayerNorm(x)
Q = attn_input @ W_q
K = attn_input @ W_k
V = attn_input @ W_v

scores = (Q @ K^T) / sqrt(H)
scores += causal_or_block_mask (if enabled)
weights = softmax(scores)
attn_out = (weights @ V) @ W_o

x1 = x + attn_out

ff_input = LayerNorm(x1)
ff_hidden = relu(ff_input @ W_ff1 + b_ff1)
ff_out = ff_hidden @ W_ff2 + b_ff2

output y = x1 + ff_out
```

Code map for quick reading:
- Backbone switch + forward entry: `Examples/DDSPE2E/ModelDecoder.swift`
- Transformer forward + blocks + attention + masks + layer norm: `Examples/DDSPE2E/ModelDecoder.swift`
- Head mapping to synth controls: `Examples/DDSPE2E/ModelDecoder.swift`
- Control-to-audio rendering: `Examples/DDSPE2E/Synth.swift`
- Training call path (model forward + synth render + loss): `Examples/DDSPE2E/Trainer.swift`

## Harmonic Head Mode

By default, training uses the legacy harmonic head behavior.

- `--harmonic-head-mode legacy` (default): `sigmoid` harmonic amplitudes + `sigmoid` harmonic gain, with `1/K` harmonic scaling in synthesis
- `--harmonic-head-mode normalized`: `softplus` harmonic amplitudes normalized per frame + `softplus` harmonic gain, without `1/K` harmonic scaling
- `--harmonic-head-mode softmax-db`: `softmax` harmonic amplitudes + bounded harmonic gain in dB (mapped back to linear), without `1/K` harmonic scaling
- `--normalized-harmonic-head <true|false>` remains available as a legacy alias for switching between `legacy` and `normalized`
- `--softmax-temp <float>` controls softmax temperature in `softmax-db` mode (higher = flatter harmonic distributions)
- `--softmax-temp-end <float>`, `--softmax-temp-warmup-steps <int>`, `--softmax-temp-ramp-steps <int>` optionally schedule softmax temperature from start to end during training
- `--softmax-amp-floor <float>` mixes harmonic amplitudes with a uniform floor in `softmax-db` mode: `amps=(1-a)*softmax + a/K`
- `--softmax-gain-min-db <float>`, `--softmax-gain-max-db <float>` set bounded harmonic gain range in dB for `softmax-db` mode
- `--harmonic-entropy-weight <float>` adds an entropy regularizer in training for `softmax-db` mode (`weight * (log(K)-entropy)`)
- `--harmonic-entropy-weight-end <float>`, `--harmonic-entropy-warmup-steps <int>`, `--harmonic-entropy-ramp-steps <int>` optionally schedule entropy weight from start to end during training (useful to avoid early collapse without constraining late optimization)
- `--harmonic-concentration-weight <float>` adds a concentration regularizer in training for `softmax-db` mode (`weight * max(mean(sum(p^2)) - 1/K, 0)`)
- `--harmonic-concentration-weight-end <float>`, `--harmonic-concentration-warmup-steps <int>`, `--harmonic-concentration-ramp-steps <int>` optionally schedule concentration penalty from start to end

## Control Smoothing

- `--control-smoothing fir` (default): frame-domain FIR smoothing (`pad + conv2d`) before sampling controls at audio rate
- `--control-smoothing off`: no smoothing

## Commands

- `dump-config --output <path>`
- `preprocess --input <wav-dir> --cache <cache-dir> [--config <json>] [overrides]`
- `inspect-cache --cache <cache-dir> [--split train|val] [--limit N]`
- `train --cache <cache-dir> [--runs-dir <dir>] [--run-name <name>] [--steps N] [--split train|val] [--mode dry|m2] [--config <json>] [overrides]`

## CLI Overrides (Config)

- `--sample-rate <float>` (default: `16000`)
- `--chunk-size <int>` (default: `16384`)
- `--chunk-hop <int>` (default: `8192`)
- `--frame-size <int>` (default: `1024`)
- `--frame-hop <int>` (default: `256`)
- `--min-f0 <float>` (default: `50`)
- `--max-f0 <float>` (default: `1000`)
- `--silence-rms <float>` (default: `0.0005`)
- `--voiced-threshold <float>` (default: `0.3`)
- `--normalize-to <float>` (default: `0.99`)
- `--train-split <float>` (default: `0.9`)
- `--seed <uint64>` (default: `1337`)
- `--max-files <int>` (default: unset)
- `--max-chunks-per-file <int>` (default: unset)
- `--shuffle <true|false>` (default: `true`)
- `--fixed-batch <true|false>` (default: `false`)
- `--model-hidden <int>` (default: `32`)
- `--model-layers <int>` (default: `1`)
- `--decoder-backbone <transformer|mlp>` (default: `transformer`; `mlp` is legacy)
- `--transformer-d-model <int>` (default: `64`)
- `--transformer-layers <int>` (default: `2`)
- `--transformer-ff-multiplier <int>` (default: `2`)
- `--transformer-causal <true|false>` (default: `true`)
- `--transformer-positional-encoding <true|false>` (default: `true`)
- `--harmonics <int>` (default: `16`)
- `--harmonic-head-mode <legacy|normalized|softmax-db|exp-sigmoid>` (default: `legacy`)
- `--control-smoothing <fir|off>` (default: `fir`)
- `--normalized-harmonic-head <true|false>` (default: `false`)
- `--softmax-temp <float>` (default: `1.0`)
- `--softmax-temp-end <float>` (default: unset; falls back to `--softmax-temp`)
- `--softmax-temp-warmup-steps <int>` (default: `0`)
- `--softmax-temp-ramp-steps <int>` (default: `0`)
- `--softmax-amp-floor <float>` (default: `0.0`)
- `--softmax-gain-min-db <float>` (default: `-50`)
- `--softmax-gain-max-db <float>` (default: `6`)
- `--harmonic-entropy-weight <float>` (default: `0.0`)
- `--harmonic-entropy-weight-end <float>` (default: unset; falls back to `--harmonic-entropy-weight`)
- `--harmonic-entropy-warmup-steps <int>` (default: `0`)
- `--harmonic-entropy-ramp-steps <int>` (default: `0`)
- `--harmonic-concentration-weight <float>` (default: `0.0`)
- `--harmonic-concentration-weight-end <float>` (default: unset; falls back to `--harmonic-concentration-weight`)
- `--harmonic-concentration-warmup-steps <int>` (default: `0`)
- `--harmonic-concentration-ramp-steps <int>` (default: `0`)
- `--lr <float>` (default: `0.001`)
- `--grad-clip <float>` (default: `1.0`)
- `--early-stop-patience <int>` (default: `0`, disabled)
- `--early-stop-min-delta <float>` (default: `0.0`)
- `--spectral-windows <csv-int-list>` (default: empty)
- `--spectral-weight <float>` (default: `0.0`)
- `--spectral-logmag-weight <float>` (default: `0.0`)
- `--spectral-loss-mode <l2|l1>` (default: `l2`)
- `--spectral-hop-divisor <int>` (default: `4`)
- `--spectral-warmup-steps <int>` (default: `100`)
- `--spectral-ramp-steps <int>` (default: `200`)
- `--loudness-weight <float>` (default: `0.0`)
- `--loudness-loss-mode <linear-l2|db-l1>` (default: `linear-l2`)
- `--loudness-weight-end <float>` (default: unset; falls back to `--loudness-weight`)
- `--loudness-warmup-steps <int>` (default: `0`)
- `--loudness-ramp-steps <int>` (default: `0`)
- `--mse-weight <float>` (default: `1.0`)
- `--log-every <int>` (default: `10`)
- `--checkpoint-every <int>` (default: `100`)
- `--kernel-dump [path]` (train only; use `true` to write to `<run-dir>/kernels.metal`)
- `--init-checkpoint <model-checkpoint-json>` (train only; initializes model weights from a saved checkpoint)
- `--dump-controls-every <int>` (train only; default: `0`, disabled)
- `--auto-abc <true|false>` (train only; default: `false`, runs staged A->B->C orchestration in one command)
- `--auto-abc-steps-a <int>` (train only; default: `--steps`)
- `--auto-abc-steps-b <int>` (train only; default: `300`)
- `--auto-abc-steps-c <int>` (train only; default: `300`)
- `--auto-abc-patience-a <int>` (train only; default: `40`)
- `--auto-abc-patience-b <int>` (train only; default: `40`)
- `--auto-abc-patience-c <int>` (train only; default: `40`)
- `--auto-abc-min-delta <float>` (train only; default: `max(1e-7, --early-stop-min-delta)`)
- `--auto-abc-preset <baseline|best-low-loss>` (train only; default: `baseline`)

## Outputs

Each `train` run writes a run directory (`runs` by default):

- `resolved_config.json`
- `run_meta.json`
- `logs/`
- `renders/`
- `checkpoints/` (periodic snapshots)
