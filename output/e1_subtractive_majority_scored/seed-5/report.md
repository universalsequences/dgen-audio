# SynthID Report

- Rung: 1
- Pass: no
- Init loss: 2.650579
- Final loss: 0.515013
- Loss ratio: 0.194302

Note: the subtractive topology is scored on its declared invariants: filter-EG endpoints, resonance, effective output-envelope levels, and scale-free envelope ratios. Individual knobs along compensating ridges are diagnostic only.

| Parameter | True | Recovered | Abs err | Rel err | Tol | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fEnv(0) | 179.128 | 231.413 | 52.2856 | 29.19% | 10.00% | no |
| fEnv(infinity) | 172.265 | 220.043 | 47.7772 | 27.73% | 10.00% | no |
| res | 1.40724 | 1.57169 | 0.164446 | 11.69% | 10.00% | no |
| effective output envelope (10ms) | 0.384032 | 0.234049 | 0.149983 | 39.05% | 10.00% | no |
| effective output envelope (75ms) | 1.2967 | 0.776855 | 0.519848 | 40.09% | 10.00% | no |
| effective output envelope (300ms) | 1.27138 | 0.757535 | 0.513847 | 40.42% | 10.00% | no |
| effective output envelope (700ms) | 0.441981 | 0.278163 | 0.163818 | 37.06% | 10.00% | no |
| aEnv(75ms)/aEnv(10ms) | 3.37655 | 3.3192 | 0.0573492 | 1.70% | 10.00% | yes |
| aEnv(300ms)/aEnv(10ms) | 3.31061 | 3.23665 | 0.0739605 | 2.23% | 10.00% | yes |
| aEnv(700ms)/aEnv(300ms) | 0.347638 | 0.367195 | 0.0195566 | 5.63% | 10.00% | yes |

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
