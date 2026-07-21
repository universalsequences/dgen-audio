# SynthID Report

- Rung: 1
- Pass: no
- Init loss: 4.295578
- Final loss: 1.088307
- Loss ratio: 0.253355

Note: the subtractive topology is scored on its declared invariants: filter-EG endpoints, effective output drive, resonance, and sampled VCA-envelope levels. Individual knobs along compensating ridges are diagnostic only.

| Parameter | True | Recovered | Abs err | Rel err | Tol | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fEnv(0) | 2097.82 | 2088.22 | 9.59253 | 0.46% | 10.00% | yes |
| fEnv(infinity) | 1995.03 | 1985.24 | 9.79248 | 0.49% | 10.00% | yes |
| res | 2.01405 | 2.10732 | 0.0932665 | 4.63% | 10.00% | yes |
| drive*outGain | 0.776446 | 0.608266 | 0.16818 | 21.66% | 10.00% | no |
| aEnv(10ms) | 0.509399 | 0.533588 | 0.0241889 | 4.75% | 10.00% | yes |
| aEnv(300ms) | 0.824791 | 0.865449 | 0.0406572 | 4.93% | 10.00% | yes |
| aEnv(700ms) | 0.219379 | 0.232355 | 0.0129762 | 5.91% | 10.00% | yes |

## Effective Gain Products

| Product | True | Recovered | Rel err |
| --- | ---: | ---: | ---: |
| shape (unscored knob) | 0.452019 | 0.540748 | 19.63% |
| pw (unscored knob) | 0.758204 | 0.747103 | 1.46% |
| fBase (unscored knob) | 1995.03 | 1985.24 | 0.49% |
| fAmt (unscored knob) | 102.782 | 102.982 | 0.19% |
| fDecay (unscored knob) | 0.0642665 | 0.0781998 | 21.68% |
| res (unscored knob) | 2.01405 | 2.10732 | 4.63% |
| attackTime (unscored knob) | 0.0138311 | 0.0129759 | 6.18% |
| decayTime (unscored knob) | 0.239436 | 0.596627 | 149.18% |
| sustain (unscored knob) | 0.810735 | 0.778013 | 4.04% |
| releaseTime (unscored knob) | 0.0991403 | 0.102852 | 3.74% |
| drive (unscored knob) | 2.01939 | 1.51933 | 24.76% |
| outGain (unscored knob) | 0.384495 | 0.400351 | 4.12% |
