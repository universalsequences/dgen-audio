# SynthID Report

- Rung: 1
- Pass: yes
- Init loss: 10.403644
- Final loss: 0.062538
- Loss ratio: 0.006011

Note: the subtractive topology is scored on its declared invariants: filter-EG endpoints, effective output drive, resonance, and sampled VCA-envelope levels. Individual knobs along compensating ridges are diagnostic only.

| Parameter | True | Recovered | Abs err | Rel err | Tol | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fEnv(0) | 3164.94 | 3130.71 | 34.2344 | 1.08% | 10.00% | yes |
| fEnv(infinity) | 3089.95 | 3055.79 | 34.1667 | 1.11% | 10.00% | yes |
| res | 2.73627 | 2.72945 | 0.00681305 | 0.25% | 10.00% | yes |
| drive*outGain | 1.52555 | 1.52798 | 0.00243258 | 0.16% | 10.00% | yes |
| aEnv(10ms) | 0.589025 | 0.582726 | 0.0062986 | 1.07% | 10.00% | yes |
| aEnv(300ms) | 0.636212 | 0.635037 | 0.00117582 | 0.18% | 10.00% | yes |
| aEnv(700ms) | 0.0155547 | 0.0156588 | 0.000104028 | 0.67% | 10.00% | yes |

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
