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

## Spectral Hop Divisor

For each spectral window `w`, the effective hop is:

`hop = max(1, w / spectralHopDivisor)`

- larger divisor -> smaller hop -> denser spectral eval -> slower, potentially stronger spectral supervision
- smaller divisor -> larger hop -> sparser spectral eval -> faster

Set `--spectral-hop-divisor 1` to effectively disable hop sparsification for that window formula.

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
- `--model-hidden <int>` (default: `32`)
- `--harmonics <int>` (default: `16`)
- `--lr <float>` (default: `0.001`)
- `--grad-clip <float>` (default: `1.0`)
- `--spectral-windows <csv-int-list>` (default: empty)
- `--spectral-weight <float>` (default: `0.0`)
- `--spectral-hop-divisor <int>` (default: `4`)
- `--spectral-warmup-steps <int>` (default: `100`)
- `--spectral-ramp-steps <int>` (default: `200`)
- `--mse-weight <float>` (default: `1.0`)
- `--log-every <int>` (default: `10`)
- `--checkpoint-every <int>` (default: `100`)
- `--kernel-dump [path]` (train only; use `true` to write to `<run-dir>/kernels.metal`)

## Outputs

Each `train` run writes a run directory (`runs` by default):

- `resolved_config.json`
- `run_meta.json`
- `logs/`
- `renders/`
- `checkpoints/` (periodic snapshots)
