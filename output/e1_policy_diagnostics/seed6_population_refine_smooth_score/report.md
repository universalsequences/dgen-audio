# SynthID Report

- Rung: 1
- Pass: no
- Init loss: 2.502261
- Final loss: 0.387583
- Loss ratio: 0.154893

Note: the subtractive topology is scored on its declared invariants: filter-EG endpoints, resonance, effective output-envelope levels, and scale-free envelope ratios. Individual knobs along compensating ridges are diagnostic only.

| Parameter | True | Recovered | Abs err | Rel err | Tol | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fEnv(0) | 93.6257 | 121.295 | 27.6694 | 29.55% | 10.00% | no |
| fEnv(infinity) | 86.4364 | 82.869 | 3.56747 | 4.13% | 10.00% | yes |
| res | 3.04147 | 2.43725 | 0.604223 | 19.87% | 10.00% | no |
| effective output envelope (10ms) | 0.956455 | 0.642294 | 0.31416 | 32.85% | 10.00% | no |
| effective output envelope (75ms) | 1.0456 | 1.11405 | 0.0684505 | 6.55% | 10.00% | yes |
| effective output envelope (300ms) | 0.857682 | 0.830331 | 0.0273504 | 3.19% | 10.00% | yes |
| effective output envelope (700ms) | 0.00654314 | 0.00599785 | 0.000545291 | 8.33% | 10.00% | yes |
| aEnv(75ms)/aEnv(10ms) | 1.0932 | 1.73449 | 0.641282 | 58.66% | 10.00% | no |
| aEnv(300ms)/aEnv(10ms) | 0.89673 | 1.29276 | 0.396028 | 44.16% | 10.00% | no |
| aEnv(700ms)/aEnv(300ms) | 0.00762887 | 0.00722344 | 0.000405427 | 5.31% | 10.00% | yes |

## Effective Gain Products

| Product | True | Recovered | Rel err |
| --- | ---: | ---: | ---: |
| shape (unscored knob) | 0.667872 | 0.780327 | 16.84% |
| pw (unscored knob) | 0.464674 | 0.463735 | 0.20% |
| fBase (unscored knob) | 86.4364 | 82.869 | 4.13% |
| fAmt (unscored knob) | 7.18924 | 38.4261 | 434.49% |
| fDecay (unscored knob) | 0.123715 | 0.0107681 | 91.30% |
| res (unscored knob) | 3.04147 | 2.43725 | 19.87% |
| attackTime (unscored knob) | 0.00524861 | 0.0141045 | 168.73% |
| decayTime (unscored knob) | 0.38587 | 0.2485 | 35.60% |
| sustain (unscored knob) | 0.54562 | 0.491237 | 9.97% |
| releaseTime (unscored knob) | 0.0214146 | 0.0212234 | 0.89% |
| drive (unscored knob) | 2.04441 | 2.36863 | 15.86% |
| outGain (unscored knob) | 0.556075 | 0.544874 | 2.01% |
