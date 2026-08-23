# ModalDrum

Closed-form modal-bank + filtered-noise recovery and real-snare fitting from
`docs/MODAL_SNARE_SPEC.md`.

## Corpus preparation

Prepare a directory of WAV files before fitting real snares:

```bash
Examples/ModalDrum/scripts/prepare_snares.sh path/to/raw-snares datasets/snares-prepared
```

The script recursively processes WAVs (PCM, IEEE float, and
`WAVE_FORMAT_EXTENSIBLE`), writes accepted clips as 0.75-second
44.1 kHz mono WAVs, and records every acceptance or rejection in
`manifest.json`. Re-running the command replaces the output deterministically.
The manifest includes source format, original peak, onset/crop offset,
normalization gain, and rejection reasons. Files with sustained RMS re-rise
after 300 ms, DC offset, clipping, or silence are rejected.

The output directory is replaced wholesale on every run, so it must be a
dedicated path outside the corpus (the script refuses an output directory that
contains the input).

## M0 synthetic recovery

```bash
swift run ModalDrum fdcheck --out runs/modal_m0_fd
swift run ModalDrum train --out runs/modal_m0 --steps 300
```

The training target is rendered by the synth itself with a fixed seed/noise
sequence. Frequencies are a fixed 120 Hz–14 kHz log grid. Gains are sigmoid
parameters, modal decays are log-mapped to 5 ms–2 s, and noise decay is
log-mapped to 10–250 ms. The modal oscillator is `deterministicPhasor`, so it
has no history cells and is frame-parallel; only the FIR noise buffer is
stateful.

Artifacts are written below the selected run directory:

- `loss.csv`
- `target.wav` and `previews/*.wav`
- `checkpoints/model_best.json` and periodic checkpoints
- `true_params.json`, `recovered_params.json`, and `summary.json`

Use `--kernel-dump PATH` on `train` to retain generated kernels for the
frame-parallel audit. `--no-loudness` disables the small 256-sample frame-RMS
L1 auxiliary if a configuration makes it unstable.

## M1 real-snare sweep

Choose two different accepted files from the prepared corpus and run:

```bash
swift run ModalDrum fit-real \
  --target datasets/snares-prepared/snare-a.wav \
  --wrong-snare datasets/snares-prepared/snare-b.wav \
  --k 32,64,128 --steps 300 --out runs/modal_m1
```

M1 enforces the spec's fixed 0.75-second, 44.1 kHz input contract. For every K
it runs modal-only and modal+noise fits. Gains are warm-started from the target
spectral envelope, modal decay starts at 150 ms, FIR taps are lowpass-like, and
noise decay starts at 80 ms. `--high-mode-l1` controls the gain budget above
6 kHz (default `1e-3`); noise decay remains hard-bounded to 10–250 ms.

`calibration.json` records target-vs-self, target-vs-silence,
target-vs-RMS-matched white noise, and target-vs-wrong-snare CPU MR-STFT
scores. Since exact self distance is zero, the automatic numeric gate requires
a fit to be 1.5x closer than the nearest negative control. Silence is one of
those controls on purpose: on a 0.75 s percussion window most frames are
decayed tail, so silence scores only ~1.6x worse than a wrong snare and a gate
derived from the loud controls alone would be passable by rendering nothing. Use `--gate SCORE` to record a stricter externally
chosen calibration gate.

Each `kNNN_modal_only` / `kNNN_modal_noise` directory contains `loss.csv`, a
best checkpoint, periodic previews, `full.wav`, `modal.wav`, `noise.wav`, the
best patch, wall-clock time, and branch energy fractions. The top-level
`summary.json` records the smallest passing K and whether the noise branch beat
modal-only at every K. The listen gate is intentionally left `PENDING`: A/B
`full.wav` against `target.wav`, then audition the solo branches before M2.

Measured results for the first target — calibration table, K sweep, noise
ablation, lambda sweep, listen gate and the residual diagnosis — are recorded
in `M1_FINDING.md`.
