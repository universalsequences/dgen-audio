# SynthID Report

- Rung: 1
- Pass: yes
- Init loss: 3.006388
- Final loss: 0.000420
- Loss ratio: 0.000140

Note: the subtractive topology is scored on its declared invariants: filter-EG endpoints, resonance, effective output-envelope levels, and scale-free envelope ratios. Individual knobs along compensating ridges are diagnostic only.

| Parameter | True | Recovered | Abs err | Rel err | Tol | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fEnv(0) | 1163.82 | 1163.9 | 0.0748291 | 0.01% | 10.00% | yes |
| fEnv(infinity) | 1025.91 | 1025.92 | 0.00146484 | 0.00% | 10.00% | yes |
| res | 1.39326 | 1.39325 | 1.26362e-05 | 0.00% | 10.00% | yes |
| effective output envelope (10ms) | 0.138427 | 0.138423 | 3.78489e-06 | 0.00% | 10.00% | yes |
| effective output envelope (75ms) | 0.190332 | 0.190338 | 5.61774e-06 | 0.00% | 10.00% | yes |
| effective output envelope (300ms) | 0.164566 | 0.164562 | 3.32296e-06 | 0.00% | 10.00% | yes |
| effective output envelope (700ms) | 0.0291597 | 0.0291603 | 6.81728e-07 | 0.00% | 10.00% | yes |
| aEnv(75ms)/aEnv(10ms) | 1.37497 | 1.37505 | 7.83205e-05 | 0.01% | 10.00% | yes |
| aEnv(300ms)/aEnv(10ms) | 1.18883 | 1.18884 | 8.58307e-06 | 0.00% | 10.00% | yes |
| aEnv(700ms)/aEnv(300ms) | 0.177192 | 0.177199 | 7.7337e-06 | 0.00% | 10.00% | yes |

## Effective Gain Products

| Product | True | Recovered | Rel err |
| --- | ---: | ---: | ---: |
| shape (unscored knob) | 0.582953 | 0.582953 | 0.00% |
| pw (unscored knob) | 0.573662 | 0.573662 | 0.00% |
| fBase (unscored knob) | 1025.91 | 1025.92 | 0.00% |
| fAmt (unscored knob) | 137.908 | 137.982 | 0.05% |
| fDecay (unscored knob) | 0.0160516 | 0.0160345 | 0.11% |
| res (unscored knob) | 1.39326 | 1.39325 | 0.00% |
| attackTime (unscored knob) | 0.0103783 | 0.0103854 | 0.07% |
| decayTime (unscored knob) | 0.0783669 | 0.0782502 | 0.15% |
| sustain (unscored knob) | 0.710416 | 0.710097 | 0.04% |
| releaseTime (unscored knob) | 0.0650679 | 0.0650682 | 0.00% |
| drive (unscored knob) | 0.704448 | 0.70479 | 0.05% |
| outGain (unscored knob) | 0.329187 | 0.32918 | 0.00% |
