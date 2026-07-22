# SynthID Report

- Rung: 1
- Pass: no
- Init loss: 10.403644
- Final loss: 3.108939
- Loss ratio: 0.298832

Note: the subtractive topology is scored on its declared invariants: filter-EG endpoints, effective output drive, resonance, and sampled VCA-envelope levels. Individual knobs along compensating ridges are diagnostic only.

| Parameter | True | Recovered | Abs err | Rel err | Tol | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fEnv(0) | 3164.94 | 3148.33 | 16.6143 | 0.52% | 10.00% | yes |
| fEnv(infinity) | 3089.95 | 3069.42 | 20.5298 | 0.66% | 10.00% | yes |
| res | 2.73627 | 3.00192 | 0.265655 | 9.71% | 10.00% | yes |
| drive*outGain | 1.52555 | 1.20989 | 0.315663 | 20.69% | 10.00% | no |
| aEnv(10ms) | 0.589025 | 0.381417 | 0.207608 | 35.25% | 10.00% | no |
| aEnv(300ms) | 0.636212 | 0.60876 | 0.027452 | 4.31% | 10.00% | yes |
| aEnv(700ms) | 0.0155547 | 0.0313598 | 0.0158051 | 101.61% | 10.00% | no |

## Effective Gain Products

| Product | True | Recovered | Rel err |
| --- | ---: | ---: | ---: |
| shape (unscored knob) | 0.546593 | 0.614519 | 12.43% |
| pw (unscored knob) | 0.661724 | 0.642205 | 2.95% |
| fBase (unscored knob) | 3089.95 | 3069.42 | 0.66% |
| fAmt (unscored knob) | 74.9852 | 78.9007 | 5.22% |
| fDecay (unscored knob) | 0.0791555 | 0.27494 | 247.34% |
| res (unscored knob) | 2.73627 | 3.00192 | 9.71% |
| attackTime (unscored knob) | 0.0107891 | 0.0203323 | 88.45% |
| decayTime (unscored knob) | 0.169453 | 0.405272 | 139.17% |
| sustain (unscored knob) | 0.561579 | 0.252743 | 54.99% |
| releaseTime (unscored knob) | 0.0280024 | 0.0412471 | 47.30% |
| drive (unscored knob) | 2.15572 | 3.44808 | 59.95% |
| outGain (unscored knob) | 0.707675 | 0.350887 | 50.42% |
