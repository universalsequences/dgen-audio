# HarmonicE2E

`HarmonicE2E` is a deliberately small end-to-end training example:

- preprocess WAV files into cached chunks
- extract `f0`, loudness, and voiced/unvoiced features
- train a tiny MLP decoder
- render a harmonic synth plus a learned FIR-shaped noise branch
- save checkpoints and WAV snapshots

It is not a full DDSP implementation. It is the smallest honest demo of:

`audio -> features -> neural controls -> differentiable synth -> loss -> checkpoint/render`

## Known-Good Demo

Recommended first dataset:

- `datasets/tinysol/Keyboards/Accordion/ordinario`

Recommended demo path:

```bash
swift build --target HarmonicE2E
swift run HarmonicE2E preprocess \
  --input datasets/tinysol/Keyboards/Accordion/ordinario \
  --cache .harmonic_cache_accordion_24k \
  --max-files 8 \
  --max-chunks-per-file 4

swift run HarmonicE2E train \
  --cache .harmonic_cache_accordion_24k \
  --steps 150 \
  --batch-size 4 \
  --run-name harmonic_accordion_demo \
  --render-every 50
```

Outputs are written under `runs/<run-name>/`:

- `resolved_config.json`
- `checkpoints/model_step_*.json`
- `checkpoints/model_best.json`
- `renders/step_*.wav`
- `logs/noise_filter_step_*.csv`
- `logs/noise_filter_summary_step_*.csv`
- `logs/train_summary.json`
- `logs/train_log.csv`

## Success Criteria

This example is a success if:

- preprocessing and caching run from raw WAV files
- training runs end-to-end without graph/runtime issues
- loss decreases on a small monophonic dataset
- renders land in the timbral neighborhood of the source
- the learned noise branch is active and inspectable through the FIR summaries

The intended claim is:

`HarmonicE2E` shows that DGen can support a small batched differentiable audio training pipeline in Swift.

It is not intended to claim:

- paper-faithful DDSP reproduction
- realistic resynthesis across broad instrument families
- production-quality timbral realism

## Useful Knobs

- `--num-harmonics 32`
- `--noise-filter-size 63`
- `--sample-rate 24000`
- `--batch-size 4`
- `--harmonic-path-scale 0`
- `--noise-path-scale 1`

## Limitations

- harmonic-only tonal path plus a learned FIR noise residual is still a simple model
- renders can be recognizable without sounding realistic
- the FIR noise branch is flexible, but its behavior is not a literal `cutoff` control
- this example is intentionally optimized for readability and finishability, not maximum quality
