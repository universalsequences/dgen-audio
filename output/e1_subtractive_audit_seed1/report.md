# SynthID Report

- Rung: 1
- Pass: no
- Init loss: 24.585949
- Final loss: 2.635568
- Loss ratio: 0.107198

Note: the subtractive topology is scored on its declared invariants: filter-EG endpoints, effective output drive, resonance, and sampled VCA-envelope levels. Individual knobs along compensating ridges are diagnostic only.

| Parameter | True | Recovered | Abs err | Rel err | Tol | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fEnv(0) | 3164.94 | 1360.7 | 1804.23 | 57.01% | 10.00% | no |
| fEnv(infinity) | 3089.95 | 1167.94 | 1922.02 | 62.20% | 10.00% | no |
| res | 2.73627 | 2.32905 | 0.407211 | 14.88% | 10.00% | no |
| drive*outGain | 1.52555 | 2.68419 | 1.15864 | 75.95% | 10.00% | no |
| aEnv(10ms) | 0.589025 | 0.792951 | 0.203926 | 34.62% | 10.00% | no |
| aEnv(300ms) | 0.636212 | 0.879171 | 0.242958 | 38.19% | 10.00% | no |
| aEnv(700ms) | 0.0155547 | 0.0412003 | 0.0256456 | 164.87% | 10.00% | no |

## Effective Gain Products

| Product | True | Recovered | Rel err |
| --- | ---: | ---: | ---: |
| shape (unscored knob) | 0.546593 | 0.554267 | 1.40% |
| pw (unscored knob) | 0.661724 | 0.658706 | 0.46% |
| fBase (unscored knob) | 3089.95 | 1167.94 | 62.20% |
| fAmt (unscored knob) | 74.9852 | 192.766 | 157.07% |
| fDecay (unscored knob) | 0.0791555 | 0.861437 | 988.28% |
| res (unscored knob) | 2.73627 | 2.32905 | 14.88% |
| attackTime (unscored knob) | 0.0107891 | 0.00625241 | 42.05% |
| decayTime (unscored knob) | 0.169453 | 0.295176 | 74.19% |
| sustain (unscored knob) | 0.561579 | 0.810835 | 44.38% |
| releaseTime (unscored knob) | 0.0280024 | 0.0338965 | 21.05% |
| drive (unscored knob) | 2.15572 | 4.14821 | 92.43% |
| outGain (unscored knob) | 0.707675 | 0.647072 | 8.56% |
