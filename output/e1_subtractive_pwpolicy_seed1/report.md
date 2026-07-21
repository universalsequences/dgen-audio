# SynthID Report

- Rung: 1
- Pass: no
- Init loss: 10.403644
- Final loss: 0.666797
- Loss ratio: 0.064093

Note: the subtractive topology is scored on its declared invariants: filter-EG endpoints, effective output drive, resonance, and sampled VCA-envelope levels. Individual knobs along compensating ridges are diagnostic only.

| Parameter | True | Recovered | Abs err | Rel err | Tol | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fEnv(0) | 3164.94 | 2958.71 | 206.229 | 6.52% | 10.00% | yes |
| fEnv(infinity) | 3089.95 | 2876.84 | 213.111 | 6.90% | 10.00% | yes |
| res | 2.73627 | 2.24659 | 0.489673 | 17.90% | 10.00% | no |
| drive*outGain | 1.52555 | 1.53884 | 0.0132957 | 0.87% | 10.00% | yes |
| aEnv(10ms) | 0.589025 | 0.625606 | 0.0365812 | 6.21% | 10.00% | yes |
| aEnv(300ms) | 0.636212 | 0.697418 | 0.0612059 | 9.62% | 10.00% | yes |
| aEnv(700ms) | 0.0155547 | 0.0177331 | 0.00217839 | 14.00% | 10.00% | no |

## Effective Gain Products

| Product | True | Recovered | Rel err |
| --- | ---: | ---: | ---: |
| shape (unscored knob) | 0.546593 | 0.544483 | 0.39% |
| pw (unscored knob) | 0.661724 | 0.6611 | 0.09% |
| fBase (unscored knob) | 3089.95 | 2876.84 | 6.90% |
| fAmt (unscored knob) | 74.9852 | 81.8674 | 9.18% |
| fDecay (unscored knob) | 0.0791555 | 0.607314 | 667.24% |
| res (unscored knob) | 2.73627 | 2.24659 | 17.90% |
| attackTime (unscored knob) | 0.0107891 | 0.00988349 | 8.39% |
| decayTime (unscored knob) | 0.169453 | 0.248952 | 46.92% |
| sustain (unscored knob) | 0.561579 | 0.567969 | 1.14% |
| releaseTime (unscored knob) | 0.0280024 | 0.0287272 | 2.59% |
| drive (unscored knob) | 2.15572 | 2.36485 | 9.70% |
| outGain (unscored knob) | 0.707675 | 0.650716 | 8.05% |
