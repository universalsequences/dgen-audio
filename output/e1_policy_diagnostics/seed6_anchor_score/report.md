# SynthID Report

- Rung: 1
- Pass: no
- Init loss: 4.741158
- Final loss: 1.358517
- Loss ratio: 0.286537

Note: the subtractive topology is scored on its declared invariants: filter-EG endpoints, resonance, effective output-envelope levels, and scale-free envelope ratios. Individual knobs along compensating ridges are diagnostic only.

| Parameter | True | Recovered | Abs err | Rel err | Tol | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fEnv(0) | 93.6257 | 100.235 | 6.6091 | 7.06% | 10.00% | yes |
| fEnv(infinity) | 86.4364 | 96.7834 | 10.347 | 11.97% | 10.00% | no |
| res | 3.04147 | 2.00374 | 1.03773 | 34.12% | 10.00% | no |
| effective output envelope (10ms) | 0.956455 | 2.14452 | 1.18807 | 124.22% | 10.00% | no |
| effective output envelope (75ms) | 1.0456 | 2.01713 | 0.971528 | 92.92% | 10.00% | no |
| effective output envelope (300ms) | 0.857682 | 1.40728 | 0.549594 | 64.08% | 10.00% | no |
| effective output envelope (700ms) | 0.00654314 | 0.00791308 | 0.00136994 | 20.94% | 10.00% | no |
| aEnv(75ms)/aEnv(10ms) | 1.0932 | 0.940596 | 0.152607 | 13.96% | 10.00% | no |
| aEnv(300ms)/aEnv(10ms) | 0.89673 | 0.656219 | 0.240511 | 26.82% | 10.00% | no |
| aEnv(700ms)/aEnv(300ms) | 0.00762887 | 0.00562298 | 0.00200589 | 26.29% | 10.00% | no |

## Effective Gain Products

| Product | True | Recovered | Rel err |
| --- | ---: | ---: | ---: |
| shape (unscored knob) | 0.667872 | 0.230431 | 65.50% |
| pw (unscored knob) | 0.464674 | 0.327592 | 29.50% |
| fBase (unscored knob) | 86.4364 | 96.7834 | 11.97% |
| fAmt (unscored knob) | 7.18924 | 3.45134 | 51.99% |
| fDecay (unscored knob) | 0.123715 | 0.107658 | 12.98% |
| res (unscored knob) | 3.04147 | 2.00374 | 34.12% |
| attackTime (unscored knob) | 0.00524861 | 0.00436129 | 16.91% |
| decayTime (unscored knob) | 0.38587 | 0.182497 | 52.71% |
| sustain (unscored knob) | 0.54562 | 0.471127 | 13.65% |
| releaseTime (unscored knob) | 0.0214146 | 0.0199927 | 6.64% |
| drive (unscored knob) | 2.04441 | 4.96833 | 143.02% |
| outGain (unscored knob) | 0.556075 | 0.494049 | 11.15% |
