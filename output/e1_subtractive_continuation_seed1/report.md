# SynthID Report

- Rung: 1
- Pass: no
- Init loss: 1.286480
- Final loss: 1.122344
- Loss ratio: 0.872415

Note: the subtractive topology is scored on its declared invariants: filter-EG endpoints, effective output drive, resonance, and sampled VCA-envelope levels. Individual knobs along compensating ridges are diagnostic only.

| Parameter | True | Recovered | Abs err | Rel err | Tol | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fEnv(0) | 3164.94 | 3137.57 | 27.3667 | 0.86% | 10.00% | yes |
| fEnv(infinity) | 3089.95 | 3041.38 | 48.5696 | 1.57% | 10.00% | yes |
| res | 2.73627 | 2.74457 | 0.0083003 | 0.30% | 10.00% | yes |
| drive*outGain | 1.52555 | 1.43437 | 0.0911736 | 5.98% | 10.00% | yes |
| aEnv(10ms) | 0.589025 | 0.619129 | 0.0301045 | 5.11% | 10.00% | yes |
| aEnv(300ms) | 0.636212 | 0.681354 | 0.0451419 | 7.10% | 10.00% | yes |
| aEnv(700ms) | 0.0155547 | 0.0165888 | 0.00103406 | 6.65% | 10.00% | yes |

## Effective Gain Products

| Product | True | Recovered | Rel err |
| --- | ---: | ---: | ---: |
| shape (unscored knob) | 0.546593 | 0.545998 | 0.11% |
| pw (unscored knob) | 0.661724 | 0.64189 | 3.00% |
| fBase (unscored knob) | 3089.95 | 3041.38 | 1.57% |
| fAmt (unscored knob) | 74.9852 | 96.188 | 28.28% |
| fDecay (unscored knob) | 0.0791555 | 0.585784 | 640.04% |
| res (unscored knob) | 2.73627 | 2.74457 | 0.30% |
| attackTime (unscored knob) | 0.0107891 | 0.0100206 | 7.12% |
| decayTime (unscored knob) | 0.169453 | 0.212233 | 25.25% |
| sustain (unscored knob) | 0.561579 | 0.578933 | 3.09% |
| releaseTime (unscored knob) | 0.0280024 | 0.0281637 | 0.58% |
| drive (unscored knob) | 2.15572 | 2.0499 | 4.91% |
| outGain (unscored knob) | 0.707675 | 0.69973 | 1.12% |
