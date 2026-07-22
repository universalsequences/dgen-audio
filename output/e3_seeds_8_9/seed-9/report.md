# SynthID Report

- Rung: 1
- Pass: no
- Init loss: 2.669606
- Final loss: 0.772825
- Loss ratio: 0.289490

Note: the subtractive topology is scored on its declared invariants: filter-EG endpoints, resonance, effective output-envelope levels, and scale-free envelope ratios. Individual knobs along compensating ridges are diagnostic only.

| Parameter | True | Recovered | Abs err | Rel err | Tol | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fEnv(0) | 907.352 | 993.365 | 86.0131 | 9.48% | 10.00% | yes |
| fEnv(infinity) | 195.696 | 225.727 | 30.0315 | 15.35% | 10.00% | no |
| res | 0.885983 | 0.95138 | 0.065397 | 7.38% | 10.00% | yes |
| effective output envelope (10ms) | 0.13575 | 0.173504 | 0.0377541 | 27.81% | 10.00% | no |
| effective output envelope (75ms) | 0.501053 | 0.66872 | 0.167667 | 33.46% | 10.00% | no |
| effective output envelope (300ms) | 0.527079 | 0.713743 | 0.186663 | 35.41% | 10.00% | no |
| effective output envelope (700ms) | 0.170462 | 0.212804 | 0.0423419 | 24.84% | 10.00% | no |
| aEnv(75ms)/aEnv(10ms) | 3.691 | 3.8542 | 0.163206 | 4.42% | 10.00% | yes |
| aEnv(300ms)/aEnv(10ms) | 3.88272 | 4.11369 | 0.230974 | 5.95% | 10.00% | yes |
| aEnv(700ms)/aEnv(300ms) | 0.323409 | 0.298152 | 0.0252568 | 7.81% | 10.00% | yes |

## Effective Gain Products

| Product | True | Recovered | Rel err |
| --- | ---: | ---: | ---: |
| shape (unscored knob) | 0.627654 | 0.484648 | 22.78% |
| pw (unscored knob) | 0.664957 | 0.675235 | 1.55% |
| fBase (unscored knob) | 195.696 | 225.727 | 15.35% |
| fAmt (unscored knob) | 711.657 | 767.638 | 7.87% |
| fDecay (unscored knob) | 0.0369407 | 0.036252 | 1.86% |
| res (unscored knob) | 0.885983 | 0.95138 | 7.38% |
| attackTime (unscored knob) | 0.0480353 | 0.0452499 | 5.80% |
| decayTime (unscored knob) | 0.086954 | 0.167059 | 92.12% |
| sustain (unscored knob) | 0.756021 | 0.829746 | 9.75% |
| releaseTime (unscored knob) | 0.11917 | 0.110433 | 7.33% |
| drive (unscored knob) | 1.87499 | 2.16567 | 15.50% |
| outGain (unscored knob) | 0.398523 | 0.410043 | 2.89% |
