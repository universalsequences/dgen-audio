# ModalDrum M0

Closed-form modal-bank + filtered-noise synthetic recovery from
`docs/MODAL_SNARE_SPEC.md` M0.

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
