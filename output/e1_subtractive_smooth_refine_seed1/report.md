# SynthID Report

- Rung: 1
- Pass: no
- Init loss: 0.124741
- Final loss: 0.003089
- Loss ratio: 0.024762

Note: the subtractive topology is scored on its declared invariants: filter-EG endpoints, effective output drive, resonance, and sampled VCA-envelope levels. Individual knobs along compensating ridges are diagnostic only.

| Parameter | True | Recovered | Abs err | Rel err | Tol | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fEnv(0) | 3164.94 | 3132.26 | 32.679 | 1.03% | 10.00% | yes |
| fEnv(infinity) | 3089.95 | 3051.82 | 38.1392 | 1.23% | 10.00% | yes |
| res | 2.73627 | 2.72183 | 0.0144358 | 0.53% | 10.00% | yes |
| drive*outGain | 1.52555 | 1.52753 | 0.00198221 | 0.13% | 10.00% | yes |
| aEnv(10ms) | 0.589025 | 0.583844 | 0.00518095 | 0.88% | 10.00% | yes |
| aEnv(300ms) | 0.636212 | 0.635353 | 0.000859201 | 0.14% | 10.00% | yes |
| aEnv(700ms) | 0.0155547 | 0.0156491 | 9.43644e-05 | 0.61% | 10.00% | yes |

## Effective Gain Products

| Product | True | Recovered | Rel err |
| --- | ---: | ---: | ---: |
| shape (unscored knob) | 0.546593 | 0.546733 | 0.03% |
| pw (unscored knob) | 0.661724 | 0.661724 | 0.00% |
| fBase (unscored knob) | 3089.95 | 3051.82 | 1.23% |
| fAmt (unscored knob) | 74.9852 | 80.4454 | 7.28% |
| fDecay (unscored knob) | 0.0791555 | 0.572299 | 623.01% |
| res (unscored knob) | 2.73627 | 2.72183 | 0.53% |
| attackTime (unscored knob) | 0.0107891 | 0.0109384 | 1.38% |
| decayTime (unscored knob) | 0.169453 | 0.165263 | 2.47% |
| sustain (unscored knob) | 0.561579 | 0.564467 | 0.51% |
| releaseTime (unscored knob) | 0.0280024 | 0.0280208 | 0.07% |
| drive (unscored knob) | 2.15572 | 2.15298 | 0.13% |
| outGain (unscored knob) | 0.707675 | 0.709497 | 0.26% |
