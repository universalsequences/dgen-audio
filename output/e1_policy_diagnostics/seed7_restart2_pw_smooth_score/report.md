# SynthID Report

- Rung: 1
- Pass: no
- Init loss: 19.828100
- Final loss: 2.192974
- Loss ratio: 0.110599

Note: the subtractive topology is scored on its declared invariants: filter-EG endpoints, resonance, effective output-envelope levels, and scale-free envelope ratios. Individual knobs along compensating ridges are diagnostic only.

| Parameter | True | Recovered | Abs err | Rel err | Tol | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fEnv(0) | 2535.82 | 3148.64 | 612.818 | 24.17% | 10.00% | no |
| fEnv(infinity) | 2347.84 | 3115.78 | 767.934 | 32.71% | 10.00% | no |
| res | 1.12015 | 0.502102 | 0.618044 | 55.18% | 10.00% | no |
| effective output envelope (10ms) | 0.33282 | 0.224003 | 0.108817 | 32.70% | 10.00% | no |
| effective output envelope (75ms) | 1.23243 | 0.719432 | 0.512994 | 41.62% | 10.00% | no |
| effective output envelope (300ms) | 1.34098 | 0.759827 | 0.581149 | 43.34% | 10.00% | no |
| effective output envelope (700ms) | 0.203625 | 0.134107 | 0.0695183 | 34.14% | 10.00% | no |
| aEnv(75ms)/aEnv(10ms) | 3.70298 | 3.21171 | 0.491269 | 13.27% | 10.00% | no |
| aEnv(300ms)/aEnv(10ms) | 4.02913 | 3.39204 | 0.637087 | 15.81% | 10.00% | no |
| aEnv(700ms)/aEnv(300ms) | 0.151849 | 0.176497 | 0.0246482 | 16.23% | 10.00% | no |

## Effective Gain Products

| Product | True | Recovered | Rel err |
| --- | ---: | ---: | ---: |
| shape (unscored knob) | 0.422881 | 0.56986 | 34.76% |
| pw (unscored knob) | 0.182047 | 0.171873 | 5.59% |
| fBase (unscored knob) | 2347.84 | 3115.78 | 32.71% |
| fAmt (unscored knob) | 187.978 | 32.8623 | 82.52% |
| fDecay (unscored knob) | 0.0819173 | 0.231058 | 182.06% |
| res (unscored knob) | 1.12015 | 0.502102 | 55.18% |
| attackTime (unscored knob) | 0.0397011 | 0.0359812 | 9.37% |
| decayTime (unscored knob) | 0.371793 | 0.0630085 | 83.05% |
| sustain (unscored knob) | 0.818405 | 0.805689 | 1.55% |
| releaseTime (unscored knob) | 0.0603691 | 0.064532 | 6.90% |
| drive (unscored knob) | 3.24289 | 1.2168 | 62.48% |
| outGain (unscored knob) | 0.463181 | 0.781043 | 68.63% |
