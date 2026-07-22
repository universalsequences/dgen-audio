# SynthID Report

- Rung: 1
- Pass: no
- Init loss: 19.828098
- Final loss: 3.810113
- Loss ratio: 0.192157

Note: the subtractive topology is scored on its declared invariants: filter-EG endpoints, resonance, effective output-envelope levels, and scale-free envelope ratios. Individual knobs along compensating ridges are diagnostic only.

| Parameter | True | Recovered | Abs err | Rel err | Tol | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fEnv(0) | 2535.82 | 620.961 | 1914.86 | 75.51% | 10.00% | no |
| fEnv(infinity) | 2347.84 | 613.955 | 1733.89 | 73.85% | 10.00% | no |
| res | 1.12015 | 3.61082 | 2.49068 | 222.35% | 10.00% | no |
| effective output envelope (10ms) | 0.33282 | 0.603188 | 0.270368 | 81.24% | 10.00% | no |
| effective output envelope (75ms) | 1.23243 | 1.72193 | 0.4895 | 39.72% | 10.00% | no |
| effective output envelope (300ms) | 1.34098 | 1.72928 | 0.388309 | 28.96% | 10.00% | no |
| effective output envelope (700ms) | 0.203625 | 0.494895 | 0.29127 | 143.04% | 10.00% | no |
| aEnv(75ms)/aEnv(10ms) | 3.70298 | 2.85471 | 0.848271 | 22.91% | 10.00% | no |
| aEnv(300ms)/aEnv(10ms) | 4.02913 | 2.86691 | 1.16222 | 28.85% | 10.00% | no |
| aEnv(700ms)/aEnv(300ms) | 0.151849 | 0.286185 | 0.134336 | 88.47% | 10.00% | no |

## Effective Gain Products

| Product | True | Recovered | Rel err |
| --- | ---: | ---: | ---: |
| shape (unscored knob) | 0.422881 | 0.211057 | 50.09% |
| pw (unscored knob) | 0.182047 | 0.262158 | 44.01% |
| fBase (unscored knob) | 2347.84 | 613.955 | 73.85% |
| fAmt (unscored knob) | 187.978 | 7.00579 | 96.27% |
| fDecay (unscored knob) | 0.0819173 | 1.50186 | 1733.38% |
| res (unscored knob) | 1.12015 | 3.61082 | 222.35% |
| attackTime (unscored knob) | 0.0397011 | 0.0247383 | 37.69% |
| decayTime (unscored knob) | 0.371793 | 0.087636 | 76.43% |
| sustain (unscored knob) | 0.818405 | 1 | 22.19% |
| releaseTime (unscored knob) | 0.0603691 | 0.101573 | 68.25% |
| drive (unscored knob) | 3.24289 | 4.52267 | 39.46% |
| outGain (unscored knob) | 0.463181 | 0.402303 | 13.14% |
