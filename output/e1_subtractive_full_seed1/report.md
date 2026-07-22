# SynthID Report

- Rung: 1
- Pass: no
- Init loss: 10.403644
- Final loss: 1.286480
- Loss ratio: 0.123657

Note: the subtractive topology is scored on its declared invariants: filter-EG endpoints, effective output drive, resonance, and sampled VCA-envelope levels. Individual knobs along compensating ridges are diagnostic only.

| Parameter | True | Recovered | Abs err | Rel err | Tol | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fEnv(0) | 3164.94 | 3118.92 | 46.0149 | 1.45% | 10.00% | yes |
| fEnv(infinity) | 3089.95 | 3026.31 | 63.6409 | 2.06% | 10.00% | yes |
| res | 2.73627 | 2.7128 | 0.0234666 | 0.86% | 10.00% | yes |
| drive*outGain | 1.52555 | 1.32877 | 0.196778 | 12.90% | 10.00% | no |
| aEnv(10ms) | 0.589025 | 0.627936 | 0.0389108 | 6.61% | 10.00% | yes |
| aEnv(300ms) | 0.636212 | 0.729149 | 0.092937 | 14.61% | 10.00% | no |
| aEnv(700ms) | 0.0155547 | 0.0176068 | 0.00205206 | 13.19% | 10.00% | no |

## Effective Gain Products

| Product | True | Recovered | Rel err |
| --- | ---: | ---: | ---: |
| shape (unscored knob) | 0.546593 | 0.544209 | 0.44% |
| pw (unscored knob) | 0.661724 | 0.641834 | 3.01% |
| fBase (unscored knob) | 3089.95 | 3026.31 | 2.06% |
| fAmt (unscored knob) | 74.9852 | 92.611 | 23.51% |
| fDecay (unscored knob) | 0.0791555 | 0.555988 | 602.40% |
| res (unscored knob) | 2.73627 | 2.7128 | 0.86% |
| attackTime (unscored knob) | 0.0107891 | 0.00989797 | 8.26% |
| decayTime (unscored knob) | 0.169453 | 0.416553 | 145.82% |
| sustain (unscored knob) | 0.561579 | 0.472425 | 15.88% |
| releaseTime (unscored knob) | 0.0280024 | 0.0290086 | 3.59% |
| drive (unscored knob) | 2.15572 | 2.36757 | 9.83% |
| outGain (unscored knob) | 0.707675 | 0.561237 | 20.69% |
