# SynthID Report

- Rung: 1
- Pass: no
- Init loss: 6.829750
- Final loss: 6.126420
- Loss ratio: 0.897020

Note: the subtractive topology is scored on its declared invariants: filter-EG endpoints, resonance, effective output-envelope levels, and scale-free envelope ratios. Individual knobs along compensating ridges are diagnostic only.

| Parameter | True | Recovered | Abs err | Rel err | Tol | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fEnv(0) | 2535.82 | 2841.36 | 305.539 | 12.05% | 10.00% | no |
| fEnv(infinity) | 2347.84 | 2801.31 | 453.466 | 19.31% | 10.00% | no |
| res | 1.12015 | 1.38843 | 0.268285 | 23.95% | 10.00% | no |
| effective output envelope (10ms) | 0.33282 | 0.0713031 | 0.261517 | 78.58% | 10.00% | no |
| effective output envelope (75ms) | 1.23243 | 0.224036 | 1.00839 | 81.82% | 10.00% | no |
| effective output envelope (300ms) | 1.34098 | 0.227695 | 1.11328 | 83.02% | 10.00% | no |
| effective output envelope (700ms) | 0.203625 | 0.0482285 | 0.155397 | 76.32% | 10.00% | no |
| aEnv(75ms)/aEnv(10ms) | 3.70298 | 3.14202 | 0.560965 | 15.15% | 10.00% | no |
| aEnv(300ms)/aEnv(10ms) | 4.02913 | 3.19335 | 0.835785 | 20.74% | 10.00% | no |
| aEnv(700ms)/aEnv(300ms) | 0.151849 | 0.211811 | 0.0599627 | 39.49% | 10.00% | no |

## Effective Gain Products

| Product | True | Recovered | Rel err |
| --- | ---: | ---: | ---: |
| shape (unscored knob) | 0.422881 | 0.836341 | 97.77% |
| pw (unscored knob) | 0.182047 | 0.641008 | 252.11% |
| fBase (unscored knob) | 2347.84 | 2801.31 | 19.31% |
| fAmt (unscored knob) | 187.978 | 40.0518 | 78.69% |
| fDecay (unscored knob) | 0.0819173 | 0.0500581 | 38.89% |
| res (unscored knob) | 1.12015 | 1.38843 | 23.95% |
| attackTime (unscored knob) | 0.0397011 | 0.0309797 | 21.97% |
| decayTime (unscored knob) | 0.371793 | 0.115481 | 68.94% |
| sustain (unscored knob) | 0.818405 | 0.879214 | 7.43% |
| releaseTime (unscored knob) | 0.0603691 | 0.0754605 | 25.00% |
| drive (unscored knob) | 3.24289 | 0.707059 | 78.20% |
| outGain (unscored knob) | 0.463181 | 0.369393 | 20.25% |
