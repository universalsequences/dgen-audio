# SynthID Report

- Rung: 1
- Pass: no
- Init loss: 2.650582
- Final loss: 0.515013
- Loss ratio: 0.194302

Note: the subtractive topology is scored on its declared invariants: filter-EG endpoints, effective output drive, resonance, and sampled VCA-envelope levels. Individual knobs along compensating ridges are diagnostic only.

| Parameter | True | Recovered | Abs err | Rel err | Tol | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fEnv(0) | 179.128 | 231.413 | 52.2856 | 29.19% | 10.00% | no |
| fEnv(infinity) | 172.265 | 220.043 | 47.7772 | 27.73% | 10.00% | no |
| res | 1.40724 | 1.57169 | 0.164446 | 11.69% | 10.00% | no |
| drive*outGain | 1.58095 | 1.04176 | 0.539187 | 34.11% | 10.00% | no |
| aEnv(10ms) | 0.242913 | 0.224667 | 0.0182458 | 7.51% | 10.00% | yes |
| aEnv(300ms) | 0.804191 | 0.72717 | 0.0770214 | 9.58% | 10.00% | yes |
| aEnv(700ms) | 0.279568 | 0.267013 | 0.0125546 | 4.49% | 10.00% | yes |

## Effective Gain Products

| Product | True | Recovered | Rel err |
| --- | ---: | ---: | ---: |
| shape (unscored knob) | 0.420738 | 0.430345 | 2.28% |
| pw (unscored knob) | 0.666018 | 0.707469 | 6.22% |
| fBase (unscored knob) | 172.265 | 220.043 | 27.73% |
| fAmt (unscored knob) | 6.86215 | 11.3706 | 65.70% |
| fDecay (unscored knob) | 0.027017 | 0.0123334 | 54.35% |
| res (unscored knob) | 1.40724 | 1.57169 | 11.69% |
| attackTime (unscored knob) | 0.0351013 | 0.0376422 | 7.24% |
| decayTime (unscored knob) | 0.1418 | 0.0801992 | 43.44% |
| sustain (unscored knob) | 0.872807 | 0.809585 | 7.24% |
| releaseTime (unscored knob) | 0.132645 | 0.141028 | 6.32% |
| drive (unscored knob) | 2.97996 | 2.39643 | 19.58% |
| outGain (unscored knob) | 0.530525 | 0.434712 | 18.06% |
