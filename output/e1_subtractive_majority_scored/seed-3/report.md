# SynthID Report

- Rung: 1
- Pass: yes
- Init loss: 7.199862
- Final loss: 0.025997
- Loss ratio: 0.003611

Note: the subtractive topology is scored on its declared invariants: filter-EG endpoints, resonance, effective output-envelope levels, and scale-free envelope ratios. Individual knobs along compensating ridges are diagnostic only.

| Parameter | True | Recovered | Abs err | Rel err | Tol | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fEnv(0) | 767.609 | 765.458 | 2.15173 | 0.28% | 10.00% | yes |
| fEnv(infinity) | 762.003 | 744.958 | 17.0449 | 2.24% | 10.00% | yes |
| res | 2.19516 | 2.1865 | 0.00866508 | 0.39% | 10.00% | yes |
| effective output envelope (10ms) | 0.236451 | 0.23733 | 0.000878826 | 0.37% | 10.00% | yes |
| effective output envelope (75ms) | 0.361003 | 0.360424 | 0.000579506 | 0.16% | 10.00% | yes |
| effective output envelope (300ms) | 0.298996 | 0.299557 | 0.00056076 | 0.19% | 10.00% | yes |
| effective output envelope (700ms) | 0.0825696 | 0.0831763 | 0.000606671 | 0.73% | 10.00% | yes |
| aEnv(75ms)/aEnv(10ms) | 1.52676 | 1.51866 | 0.00809526 | 0.53% | 10.00% | yes |
| aEnv(300ms)/aEnv(10ms) | 1.26452 | 1.2622 | 0.00231981 | 0.18% | 10.00% | yes |
| aEnv(700ms)/aEnv(300ms) | 0.276156 | 0.277664 | 0.00150827 | 0.55% | 10.00% | yes |

## Effective Gain Products

| Product | True | Recovered | Rel err |
| --- | ---: | ---: | ---: |
| shape (unscored knob) | 0.229415 | 0.229507 | 0.04% |
| pw (unscored knob) | 0.631793 | 0.631815 | 0.00% |
| fBase (unscored knob) | 762.003 | 744.958 | 2.24% |
| fAmt (unscored knob) | 5.60617 | 20.4994 | 265.66% |
| fDecay (unscored knob) | 0.0304446 | 2 | 6469.31% |
| res (unscored knob) | 2.19516 | 2.1865 | 0.39% |
| attackTime (unscored knob) | 0.0115827 | 0.0115211 | 0.53% |
| decayTime (unscored knob) | 0.108385 | 0.104087 | 3.97% |
| sustain (unscored knob) | 0.73043 | 0.733354 | 0.40% |
| releaseTime (unscored knob) | 0.100025 | 0.100351 | 0.33% |
| drive (unscored knob) | 1.19335 | 1.19153 | 0.15% |
| outGain (unscored knob) | 0.351953 | 0.352878 | 0.26% |
