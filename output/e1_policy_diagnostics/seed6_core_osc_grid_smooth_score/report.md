# SynthID Report

- Rung: 1
- Pass: no
- Init loss: 2.502261
- Final loss: 0.721602
- Loss ratio: 0.288380

Note: the subtractive topology is scored on its declared invariants: filter-EG endpoints, resonance, effective output-envelope levels, and scale-free envelope ratios. Individual knobs along compensating ridges are diagnostic only.

| Parameter | True | Recovered | Abs err | Rel err | Tol | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fEnv(0) | 93.6257 | 95.2459 | 1.62019 | 1.73% | 10.00% | yes |
| fEnv(infinity) | 86.4364 | 90.4678 | 4.0314 | 4.66% | 10.00% | yes |
| res | 3.04147 | 2.91686 | 0.124603 | 4.10% | 10.00% | yes |
| effective output envelope (10ms) | 0.956455 | 1.45966 | 0.503202 | 52.61% | 10.00% | no |
| effective output envelope (75ms) | 1.0456 | 1.52346 | 0.477858 | 45.70% | 10.00% | no |
| effective output envelope (300ms) | 0.857682 | 1.20141 | 0.343729 | 40.08% | 10.00% | no |
| effective output envelope (700ms) | 0.00654314 | 0.00754553 | 0.00100239 | 15.32% | 10.00% | no |
| aEnv(75ms)/aEnv(10ms) | 1.0932 | 1.04371 | 0.0494945 | 4.53% | 10.00% | yes |
| aEnv(300ms)/aEnv(10ms) | 0.89673 | 0.823078 | 0.0736525 | 8.21% | 10.00% | yes |
| aEnv(700ms)/aEnv(300ms) | 0.00762887 | 0.00628056 | 0.00134831 | 17.67% | 10.00% | no |

## Effective Gain Products

| Product | True | Recovered | Rel err |
| --- | ---: | ---: | ---: |
| shape (unscored knob) | 0.667872 | 0.143355 | 78.54% |
| pw (unscored knob) | 0.464674 | 0.542103 | 16.66% |
| fBase (unscored knob) | 86.4364 | 90.4678 | 4.66% |
| fAmt (unscored knob) | 7.18924 | 4.77802 | 33.54% |
| fDecay (unscored knob) | 0.123715 | 0.148945 | 20.39% |
| res (unscored knob) | 3.04147 | 2.91686 | 4.10% |
| attackTime (unscored knob) | 0.00524861 | 0.00486175 | 7.37% |
| decayTime (unscored knob) | 0.38587 | 0.288908 | 25.13% |
| sustain (unscored knob) | 0.54562 | 0.546049 | 0.08% |
| releaseTime (unscored knob) | 0.0214146 | 0.0205093 | 4.23% |
| drive (unscored knob) | 2.04441 | 3.39538 | 66.08% |
| outGain (unscored knob) | 0.556075 | 0.500648 | 9.97% |
