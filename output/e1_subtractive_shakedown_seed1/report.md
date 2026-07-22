# SynthID Report

- Rung: 1
- Pass: no
- Init loss: 24.585949
- Final loss: 5.639421
- Loss ratio: 0.229376

Note: the subtractive topology is scored on its declared invariants: filter-EG endpoints, effective output drive, resonance, and sampled VCA-envelope levels. Individual knobs along compensating ridges are diagnostic only.

| Parameter | True | Recovered | Abs err | Rel err | Tol | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fEnv(0) | 3164.94 | 935.939 | 2229 | 70.43% | 10.00% | no |
| fEnv(infinity) | 3089.95 | 788.224 | 2301.73 | 74.49% | 10.00% | no |
| res | 2.73627 | 2.6422 | 0.0940697 | 3.44% | 10.00% | yes |
| drive*outGain | 1.52555 | 2.04533 | 0.519781 | 34.07% | 10.00% | no |
| aEnv(10ms) | 0.589025 | 0.798516 | 0.209492 | 35.57% | 10.00% | no |
| aEnv(300ms) | 0.636212 | 0.99985 | 0.363638 | 57.16% | 10.00% | no |
| aEnv(700ms) | 0.0155547 | 0.0504657 | 0.0349109 | 224.44% | 10.00% | no |

## Effective Gain Products

| Product | True | Recovered | Rel err |
| --- | ---: | ---: | ---: |
| shape (unscored knob) | 0.546593 | 0.822227 | 50.43% |
| pw (unscored knob) | 0.661724 | 0.61411 | 7.20% |
| fBase (unscored knob) | 3089.95 | 788.224 | 74.49% |
| fAmt (unscored knob) | 74.9852 | 147.714 | 96.99% |
| fDecay (unscored knob) | 0.0791555 | 0.312478 | 294.77% |
| res (unscored knob) | 2.73627 | 2.6422 | 3.44% |
| attackTime (unscored knob) | 0.0107891 | 0.00624201 | 42.15% |
| decayTime (unscored knob) | 0.169453 | 0.268973 | 58.73% |
| sustain (unscored knob) | 0.561579 | 1 | 78.07% |
| releaseTime (unscored knob) | 0.0280024 | 0.0340753 | 21.69% |
| drive (unscored knob) | 2.15572 | 2.28066 | 5.80% |
| outGain (unscored knob) | 0.707675 | 0.896815 | 26.73% |
