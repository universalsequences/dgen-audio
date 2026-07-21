# SynthID Report

- Rung: 1
- Pass: yes
- Init loss: 10.403641
- Final loss: 0.062538
- Loss ratio: 0.006011

Note: the subtractive topology is scored on its declared invariants: filter-EG endpoints, resonance, effective output-envelope levels, and scale-free envelope ratios. Individual knobs along compensating ridges are diagnostic only.

| Parameter | True | Recovered | Abs err | Rel err | Tol | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fEnv(0) | 3164.94 | 3130.71 | 34.2344 | 1.08% | 10.00% | yes |
| fEnv(infinity) | 3089.95 | 3055.79 | 34.1667 | 1.11% | 10.00% | yes |
| res | 2.73627 | 2.72945 | 0.00681305 | 0.25% | 10.00% | yes |
| effective output envelope (10ms) | 0.898585 | 0.890394 | 0.00819135 | 0.91% | 10.00% | yes |
| effective output envelope (75ms) | 1.28512 | 1.28358 | 0.00153363 | 0.12% | 10.00% | yes |
| effective output envelope (300ms) | 0.970572 | 0.970323 | 0.000248969 | 0.03% | 10.00% | yes |
| effective output envelope (700ms) | 0.0237295 | 0.0239263 | 0.00019679 | 0.83% | 10.00% | yes |
| aEnv(75ms)/aEnv(10ms) | 1.43016 | 1.44159 | 0.0114346 | 0.80% | 10.00% | yes |
| aEnv(300ms)/aEnv(10ms) | 1.08011 | 1.08977 | 0.00965703 | 0.89% | 10.00% | yes |
| aEnv(700ms)/aEnv(300ms) | 0.024449 | 0.0246581 | 0.000209084 | 0.86% | 10.00% | yes |

## Effective Gain Products

| Product | True | Recovered | Rel err |
| --- | ---: | ---: | ---: |
| shape (unscored knob) | 0.546593 | 0.546606 | 0.00% |
| pw (unscored knob) | 0.661724 | 0.661724 | 0.00% |
| fBase (unscored knob) | 3089.95 | 3055.79 | 1.11% |
| fAmt (unscored knob) | 74.9852 | 74.9176 | 0.09% |
| fDecay (unscored knob) | 0.0791555 | 0.520428 | 557.47% |
| res (unscored knob) | 2.73627 | 2.72945 | 0.25% |
| attackTime (unscored knob) | 0.0107891 | 0.0109722 | 1.70% |
| decayTime (unscored knob) | 0.169453 | 0.165214 | 2.50% |
| sustain (unscored knob) | 0.561579 | 0.564135 | 0.46% |
| releaseTime (unscored knob) | 0.0280024 | 0.0280306 | 0.10% |
| drive (unscored knob) | 2.15572 | 2.15383 | 0.09% |
| outGain (unscored knob) | 0.707675 | 0.709426 | 0.25% |
