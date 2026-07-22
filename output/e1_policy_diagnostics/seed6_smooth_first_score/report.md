# SynthID Report

- Rung: 1
- Pass: no
- Init loss: 4.741158
- Final loss: 1.418890
- Loss ratio: 0.299271

Note: the subtractive topology is scored on its declared invariants: filter-EG endpoints, resonance, effective output-envelope levels, and scale-free envelope ratios. Individual knobs along compensating ridges are diagnostic only.

| Parameter | True | Recovered | Abs err | Rel err | Tol | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fEnv(0) | 93.6257 | 97.7604 | 4.13474 | 4.42% | 10.00% | yes |
| fEnv(infinity) | 86.4364 | 89.3971 | 2.96071 | 3.43% | 10.00% | yes |
| res | 3.04147 | 4.21226 | 1.17079 | 38.49% | 10.00% | no |
| effective output envelope (10ms) | 0.956455 | 1.32962 | 0.373162 | 39.02% | 10.00% | no |
| effective output envelope (75ms) | 1.0456 | 1.21393 | 0.168327 | 16.10% | 10.00% | no |
| effective output envelope (300ms) | 0.857682 | 0.984334 | 0.126652 | 14.77% | 10.00% | no |
| effective output envelope (700ms) | 0.00654314 | 0.00701872 | 0.000475579 | 7.27% | 10.00% | yes |
| aEnv(75ms)/aEnv(10ms) | 1.0932 | 0.91299 | 0.180214 | 16.48% | 10.00% | no |
| aEnv(300ms)/aEnv(10ms) | 0.89673 | 0.740314 | 0.156416 | 17.44% | 10.00% | no |
| aEnv(700ms)/aEnv(300ms) | 0.00762887 | 0.00713043 | 0.000498443 | 6.53% | 10.00% | yes |

## Effective Gain Products

| Product | True | Recovered | Rel err |
| --- | ---: | ---: | ---: |
| shape (unscored knob) | 0.667872 | 0.227909 | 65.88% |
| pw (unscored knob) | 0.464674 | 0.290017 | 37.59% |
| fBase (unscored knob) | 86.4364 | 89.3971 | 3.43% |
| fAmt (unscored knob) | 7.18924 | 8.36326 | 16.33% |
| fDecay (unscored knob) | 0.123715 | 0.161195 | 30.29% |
| res (unscored knob) | 3.04147 | 4.21226 | 38.49% |
| attackTime (unscored knob) | 0.00524861 | 0.00281098 | 46.44% |
| decayTime (unscored knob) | 0.38587 | 0.156913 | 59.34% |
| sustain (unscored knob) | 0.54562 | 0.652396 | 19.57% |
| releaseTime (unscored knob) | 0.0214146 | 0.0205506 | 4.03% |
| drive (unscored knob) | 2.04441 | 2.44858 | 19.77% |
| outGain (unscored knob) | 0.556075 | 0.57121 | 2.72% |
