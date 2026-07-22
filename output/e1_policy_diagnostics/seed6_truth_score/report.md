# SynthID Report

- Rung: 1
- Pass: yes
- Init loss: 2.502261
- Final loss: 0.000001
- Loss ratio: 0.000000

Note: the subtractive topology is scored on its declared invariants: filter-EG endpoints, resonance, effective output-envelope levels, and scale-free envelope ratios. Individual knobs along compensating ridges are diagnostic only.

| Parameter | True | Recovered | Abs err | Rel err | Tol | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fEnv(0) | 93.6257 | 93.6257 | 0 | 0.00% | 10.00% | yes |
| fEnv(infinity) | 86.4364 | 86.4364 | 0 | 0.00% | 10.00% | yes |
| res | 3.04147 | 3.04147 | 0 | 0.00% | 10.00% | yes |
| effective output envelope (10ms) | 0.956455 | 0.956455 | 0 | 0.00% | 10.00% | yes |
| effective output envelope (75ms) | 1.0456 | 1.0456 | 0 | 0.00% | 10.00% | yes |
| effective output envelope (300ms) | 0.857682 | 0.857682 | 0 | 0.00% | 10.00% | yes |
| effective output envelope (700ms) | 0.00654314 | 0.00654314 | 0 | 0.00% | 10.00% | yes |
| aEnv(75ms)/aEnv(10ms) | 1.0932 | 1.0932 | 0 | 0.00% | 10.00% | yes |
| aEnv(300ms)/aEnv(10ms) | 0.89673 | 0.89673 | 0 | 0.00% | 10.00% | yes |
| aEnv(700ms)/aEnv(300ms) | 0.00762887 | 0.00762887 | 0 | 0.00% | 10.00% | yes |

## Effective Gain Products

| Product | True | Recovered | Rel err |
| --- | ---: | ---: | ---: |
| shape (unscored knob) | 0.667872 | 0.667872 | 0.00% |
| pw (unscored knob) | 0.464674 | 0.464674 | 0.00% |
| fBase (unscored knob) | 86.4364 | 86.4364 | 0.00% |
| fAmt (unscored knob) | 7.18924 | 7.18924 | 0.00% |
| fDecay (unscored knob) | 0.123715 | 0.123715 | 0.00% |
| res (unscored knob) | 3.04147 | 3.04147 | 0.00% |
| attackTime (unscored knob) | 0.00524861 | 0.00524861 | 0.00% |
| decayTime (unscored knob) | 0.38587 | 0.38587 | 0.00% |
| sustain (unscored knob) | 0.54562 | 0.54562 | 0.00% |
| releaseTime (unscored knob) | 0.0214146 | 0.0214146 | 0.00% |
| drive (unscored knob) | 2.04441 | 2.04441 | 0.00% |
| outGain (unscored knob) | 0.556075 | 0.556075 | 0.00% |
