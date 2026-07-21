# SynthID Report

- Rung: 1
- Pass: no
- Init loss: 2.502261
- Final loss: 0.327534
- Loss ratio: 0.130895

Note: the subtractive topology is scored on its declared invariants: filter-EG endpoints, resonance, effective output-envelope levels, and scale-free envelope ratios. Individual knobs along compensating ridges are diagnostic only.

| Parameter | True | Recovered | Abs err | Rel err | Tol | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fEnv(0) | 93.6257 | 102.873 | 9.24707 | 9.88% | 10.00% | yes |
| fEnv(infinity) | 86.4364 | 81.3505 | 5.08597 | 5.88% | 10.00% | yes |
| res | 3.04147 | 2.26219 | 0.779279 | 25.62% | 10.00% | no |
| effective output envelope (10ms) | 0.956455 | 0.728692 | 0.227763 | 23.81% | 10.00% | no |
| effective output envelope (75ms) | 1.0456 | 1.21177 | 0.166168 | 15.89% | 10.00% | no |
| effective output envelope (300ms) | 0.857682 | 0.893209 | 0.0355271 | 4.14% | 10.00% | yes |
| effective output envelope (700ms) | 0.00654314 | 0.00637754 | 0.000165597 | 2.53% | 10.00% | yes |
| aEnv(75ms)/aEnv(10ms) | 1.0932 | 1.66294 | 0.569733 | 52.12% | 10.00% | no |
| aEnv(300ms)/aEnv(10ms) | 0.89673 | 1.22577 | 0.32904 | 36.69% | 10.00% | no |
| aEnv(700ms)/aEnv(300ms) | 0.00762887 | 0.00714004 | 0.000488832 | 6.41% | 10.00% | yes |

## Effective Gain Products

| Product | True | Recovered | Rel err |
| --- | ---: | ---: | ---: |
| shape (unscored knob) | 0.667872 | 0.789829 | 18.26% |
| pw (unscored knob) | 0.464674 | 0.465646 | 0.21% |
| fBase (unscored knob) | 86.4364 | 81.3505 | 5.88% |
| fAmt (unscored knob) | 7.18924 | 21.5223 | 199.37% |
| fDecay (unscored knob) | 0.123715 | 0.0196812 | 84.09% |
| res (unscored knob) | 3.04147 | 2.26219 | 25.62% |
| attackTime (unscored knob) | 0.00524861 | 0.0133076 | 153.55% |
| decayTime (unscored knob) | 0.38587 | 0.245345 | 36.42% |
| sustain (unscored knob) | 0.54562 | 0.481547 | 11.74% |
| releaseTime (unscored knob) | 0.0214146 | 0.0211952 | 1.02% |
| drive (unscored knob) | 2.04441 | 2.54474 | 24.47% |
| outGain (unscored knob) | 0.556075 | 0.553467 | 0.47% |
