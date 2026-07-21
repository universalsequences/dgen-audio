# SynthID Report

- Rung: 1
- Pass: no
- Init loss: 7.199857
- Final loss: 1.449859
- Loss ratio: 0.201373

Note: the subtractive topology is scored on its declared invariants: filter-EG endpoints, effective output drive, resonance, and sampled VCA-envelope levels. Individual knobs along compensating ridges are diagnostic only.

| Parameter | True | Recovered | Abs err | Rel err | Tol | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fEnv(0) | 767.609 | 487.591 | 280.018 | 36.48% | 10.00% | no |
| fEnv(infinity) | 762.003 | 439.395 | 322.609 | 42.34% | 10.00% | no |
| res | 2.19516 | 1.60756 | 0.587603 | 26.77% | 10.00% | no |
| drive*outGain | 0.420004 | 0.625895 | 0.205891 | 49.02% | 10.00% | no |
| aEnv(10ms) | 0.562973 | 0.614087 | 0.0511136 | 9.08% | 10.00% | yes |
| aEnv(300ms) | 0.711888 | 0.82535 | 0.113461 | 15.94% | 10.00% | no |
| aEnv(700ms) | 0.196592 | 0.291149 | 0.0945567 | 48.10% | 10.00% | no |

## Effective Gain Products

| Product | True | Recovered | Rel err |
| --- | ---: | ---: | ---: |
| shape (unscored knob) | 0.229415 | 0.199977 | 12.83% |
| pw (unscored knob) | 0.631793 | 0.444932 | 29.58% |
| fBase (unscored knob) | 762.003 | 439.395 | 42.34% |
| fAmt (unscored knob) | 5.60617 | 48.1968 | 759.71% |
| fDecay (unscored knob) | 0.0304446 | 0.103499 | 239.96% |
| res (unscored knob) | 2.19516 | 1.60756 | 26.77% |
| attackTime (unscored knob) | 0.0115827 | 0.0101077 | 12.73% |
| decayTime (unscored knob) | 0.108385 | 0.0773668 | 28.62% |
| sustain (unscored knob) | 0.73043 | 0.909591 | 24.53% |
| releaseTime (unscored knob) | 0.100025 | 0.132734 | 32.70% |
| drive (unscored knob) | 1.19335 | 2.00787 | 68.25% |
| outGain (unscored knob) | 0.351953 | 0.311721 | 11.43% |
