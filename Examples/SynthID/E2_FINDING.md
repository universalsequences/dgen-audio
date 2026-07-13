# SynthID E2 Independent Renderer and PolyBLEP Equivalence Finding

## Status: PASS — 5/5 seeds

Completed 2026-07-13. E0 and E1 remain PASS. E2 validates both the complete
independent subtractive-voice renderer and the narrower claim that the
training oscillator is the deployment PolyBLEP oscillator. E3 was not started.

## Gate verdict

The full-voice gate is maximum absolute sample error `< 1e-3` between DGen and
an independent NumPy renderer. The oscillator gate is log-magnitude MR-STFT
distance `< 0.00308`, frozen before execution as 5% of the E3 additive
baseline's learned distance (`0.0616`). The oscillator's sample-domain error
is also required to remain below `1e-3`.

| Seed | Full voice max abs | Oscillator max abs | Oscillator MR-STFT | Verdict |
|---:|---:|---:|---:|:---:|
| 1 | 1.162291e-6 | 5.960465e-8 | 1.594136e-7 | PASS |
| 2 | 7.905997e-6 | 5.960465e-8 | 1.731169e-7 | PASS |
| 3 | 7.122755e-6 | 5.960465e-8 | 1.595571e-7 | PASS |
| 4 | 1.635402e-6 | 7.450581e-8 | 1.647094e-7 | PASS |
| 5 | 3.053658e-4 | 5.960465e-8 | 1.777016e-7 | PASS |

**Verdict: PASS (5/5).** The worst complete-voice error is 3.27 times inside
its gate. The worst deployment-oscillator distance is about 17,300 times
inside its spectral gate. The latter is numerical noise, not merely
"small compared with" a fit residual.

## What is independent

`scripts/render_reference.py` constructs the voice sample by sample in NumPy.
It does not import DGen, consume a DGen render, or call the Swift executable.
For `subtractive-bass` it independently implements:

- the pre-update float32 phasor accumulator;
- the eseq `polyblep`, `polyblep_saw`, and `polyblep_pulse` formulas, including
  strict comparison gates and the wrapped falling edge;
- the per-sample time-varying low-pass coefficients and filter history;
- the filter and amplitude envelopes, softsign drive, and output gain.

The renderer intentionally uses float32 because DGen's C/CPU deployment target
does not support float64 and the equivalence claim is about deployable math.
The spectral comparator is separately independent: `compare_polyblep.py`
uses `compare.py`'s NumPy FFT implementation in host precision on raw WAVs,
with windows 256/512/1024/2048 and log epsilon `1e-3`.

The training-side oscillator WAV is rendered directly from
`SubtractiveBassVoice.buildOscillator`; the deployment-side WAV comes from the
literal NumPy transcription. Consequently this check would detect a change in
phasor convention, pulse-width wrap, gate boundary, BLEP sign, or blend rule
without relying on the full filter/envelope comparison to localize it.

## Exact reproduction

Build and run the hard-gated, renderer-only E2 command:

```sh
swift build -c release --product SynthID

.build/release/SynthID rung2 \
  --profile subtractive-bass \
  --seeds 1,2,3,4,5 \
  --verify-only \
  --out output/e2_subtractive_renderer
```

Each seed directory contains `renderer_equivalence.json`,
`oscillator_equivalence.json`, `polyblep_equivalence.json`, and both oscillator
WAVs. The command stops on the first failed full-voice or oscillator gate.

## Ladder status

E2 is complete and PASS. This licenses the central deployment claim: recovered
`shape` and `pw` control the same band-limited oscillator graph in training and
at runtime. E2 itself no longer blocks E3, but the subsequent fresh-seed E1
policy audit fails 0/2; E3 was therefore not started.
