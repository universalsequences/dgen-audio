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

## Spectral Hop Divisor

For each spectral window `w`, the effective hop is:

`hop = max(1, w / spectralHopDivisor)`

- larger divisor -> smaller hop -> denser spectral eval -> slower, potentially stronger spectral supervision
- smaller divisor -> larger hop -> sparser spectral eval -> faster

Set `--spectral-hop-divisor 1` to effectively disable hop sparsification for that window formula.

## Spectral Log-Magnitude Term

Use `--spectral-logmag-weight <float>` to add an FFT-domain log-magnitude term:

- per-bin term: `(log(|X_pred| + eps) - log(|X_tgt| + eps))^2`
- computed in spectral space (not `log(abs(waveform))` in time domain)
- combines with `--spectral-weight` (linear magnitude term) and `--mse-weight`

## Fixed Batch

Use `--fixed-batch true` to reuse the same sampled chunk set every training step.

- with `batch-size = 1`: repeats one chunk each step
- with `batch-size = B`: repeats the same `B` chunk IDs each step
- useful for sanity checks and optimization A/B tests where batch difficulty should not change over time

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
- `--harmonics <int>` (default: `16`)
- `--harmonic-head-mode <legacy|normalized|softmax-db>` (default: `legacy`)
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
- `--spectral-hop-divisor <int>` (default: `4`)
- `--spectral-warmup-steps <int>` (default: `100`)
- `--spectral-ramp-steps <int>` (default: `200`)
- `--mse-weight <float>` (default: `1.0`)
- `--log-every <int>` (default: `10`)
- `--checkpoint-every <int>` (default: `100`)
- `--kernel-dump [path]` (train only; use `true` to write to `<run-dir>/kernels.metal`)
- `--init-checkpoint <model-checkpoint-json>` (train only; initializes model weights from a saved checkpoint)
- `--dump-controls-every <int>` (train only; default: `0`, disabled)

## Outputs

Each `train` run writes a run directory (`runs` by default):

- `resolved_config.json`
- `run_meta.json`
- `logs/`
- `renders/`
- `checkpoints/` (periodic snapshots)
