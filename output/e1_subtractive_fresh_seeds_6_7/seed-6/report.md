# SynthID Report

- Rung: 1
- Pass: no
- Init loss: 2.502263
- Final loss: 1.359184
- Loss ratio: 0.543182

Note: the subtractive topology is scored on its declared invariants: filter-EG endpoints, resonance, effective output-envelope levels, and scale-free envelope ratios. Individual knobs along compensating ridges are diagnostic only.

| Parameter | True | Recovered | Abs err | Rel err | Tol | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fEnv(0) | 93.6257 | 473.133 | 379.508 | 405.35% | 10.00% | no |
| fEnv(infinity) | 86.4364 | 265.974 | 179.537 | 207.71% | 10.00% | no |
| res | 3.04147 | 2.06194 | 0.979527 | 32.21% | 10.00% | no |
| effective output envelope (10ms) | 0.956455 | 0.146391 | 0.810063 | 84.69% | 10.00% | no |
| effective output envelope (75ms) | 1.0456 | 0.26708 | 0.77852 | 74.46% | 10.00% | no |
| effective output envelope (300ms) | 0.857682 | 0.193828 | 0.663853 | 77.40% | 10.00% | no |
| effective output envelope (700ms) | 0.00654314 | 0.000714553 | 0.00582859 | 89.08% | 10.00% | no |
| aEnv(75ms)/aEnv(10ms) | 1.0932 | 1.82442 | 0.73122 | 66.89% | 10.00% | no |
| aEnv(300ms)/aEnv(10ms) | 0.89673 | 1.32404 | 0.427312 | 47.65% | 10.00% | no |
| aEnv(700ms)/aEnv(300ms) | 0.00762887 | 0.00368653 | 0.00394234 | 51.68% | 10.00% | no |

## Effective Gain Products

| Product | True | Recovered | Rel err |
| --- | ---: | ---: | ---: |
| shape (unscored knob) | 0.667872 | 0.989325 | 48.13% |
| pw (unscored knob) | 0.464674 | 0.517514 | 11.37% |
| fBase (unscored knob) | 86.4364 | 265.974 | 207.71% |
| fAmt (unscored knob) | 7.18924 | 207.16 | 2781.53% |
| fDecay (unscored knob) | 0.123715 | 0.0146501 | 88.16% |
| res (unscored knob) | 3.04147 | 2.06194 | 32.21% |
| attackTime (unscored knob) | 0.00524861 | 0.0151689 | 189.01% |
| decayTime (unscored knob) | 0.38587 | 0.304472 | 21.09% |
| sustain (unscored knob) | 0.54562 | 0.404633 | 25.84% |
| releaseTime (unscored knob) | 0.0214146 | 0.018875 | 11.86% |
| drive (unscored knob) | 2.04441 | 0.92524 | 54.74% |
| outGain (unscored knob) | 0.556075 | 0.334169 | 39.91% |
