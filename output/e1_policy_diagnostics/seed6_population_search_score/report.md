# SynthID Report

- Rung: 1
- Pass: no
- Init loss: 2.502261
- Final loss: 0.903712
- Loss ratio: 0.361158

Note: the subtractive topology is scored on its declared invariants: filter-EG endpoints, resonance, effective output-envelope levels, and scale-free envelope ratios. Individual knobs along compensating ridges are diagnostic only.

| Parameter | True | Recovered | Abs err | Rel err | Tol | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fEnv(0) | 93.6257 | 169.797 | 76.1717 | 81.36% | 10.00% | no |
| fEnv(infinity) | 86.4364 | 97.3171 | 10.8806 | 12.59% | 10.00% | no |
| res | 3.04147 | 3.4479 | 0.406436 | 13.36% | 10.00% | no |
| effective output envelope (10ms) | 0.956455 | 0.503503 | 0.452951 | 47.36% | 10.00% | no |
| effective output envelope (75ms) | 1.0456 | 0.539171 | 0.506428 | 48.43% | 10.00% | no |
| effective output envelope (300ms) | 0.857682 | 0.348135 | 0.509547 | 59.41% | 10.00% | no |
| effective output envelope (700ms) | 0.00654314 | 0.00425599 | 0.00228715 | 34.95% | 10.00% | no |
| aEnv(75ms)/aEnv(10ms) | 1.0932 | 1.07084 | 0.0223632 | 2.05% | 10.00% | yes |
| aEnv(300ms)/aEnv(10ms) | 0.89673 | 0.691426 | 0.205305 | 22.89% | 10.00% | no |
| aEnv(700ms)/aEnv(300ms) | 0.00762887 | 0.0122251 | 0.00459625 | 60.25% | 10.00% | no |

## Effective Gain Products

| Product | True | Recovered | Rel err |
| --- | ---: | ---: | ---: |
| shape (unscored knob) | 0.667872 | 0.770597 | 15.38% |
| pw (unscored knob) | 0.464674 | 0.429636 | 7.54% |
| fBase (unscored knob) | 86.4364 | 97.3171 | 12.59% |
| fAmt (unscored knob) | 7.18924 | 72.4803 | 908.18% |
| fDecay (unscored knob) | 0.123715 | 0.0205203 | 83.41% |
| res (unscored knob) | 3.04147 | 3.4479 | 13.36% |
| attackTime (unscored knob) | 0.00524861 | 0.00759753 | 44.75% |
| decayTime (unscored knob) | 0.38587 | 0.131806 | 65.84% |
| sustain (unscored knob) | 0.54562 | 0.425836 | 21.95% |
| releaseTime (unscored knob) | 0.0214146 | 0.023434 | 9.43% |
| drive (unscored knob) | 2.04441 | 1.56385 | 23.51% |
| outGain (unscored knob) | 0.556075 | 0.459194 | 17.42% |
