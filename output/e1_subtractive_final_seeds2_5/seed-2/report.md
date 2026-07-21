# SynthID Report

- Rung: 1
- Pass: no
- Init loss: 11.900507
- Final loss: 0.038825
- Loss ratio: 0.003262

Note: the subtractive topology is scored on its declared invariants: filter-EG endpoints, effective output drive, resonance, and sampled VCA-envelope levels. Individual knobs along compensating ridges are diagnostic only.

| Parameter | True | Recovered | Abs err | Rel err | Tol | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fEnv(0) | 1338.4 | 1338.01 | 0.389038 | 0.03% | 10.00% | yes |
| fEnv(infinity) | 712.059 | 712.214 | 0.154541 | 0.02% | 10.00% | yes |
| res | 1.32646 | 1.32512 | 0.00133789 | 0.10% | 10.00% | yes |
| drive*outGain | 0.478348 | 0.545488 | 0.0671407 | 14.04% | 10.00% | no |
| aEnv(10ms) | 0.496629 | 0.430566 | 0.0660627 | 13.30% | 10.00% | no |
| aEnv(300ms) | 0.870295 | 0.760554 | 0.109741 | 12.61% | 10.00% | no |
| aEnv(700ms) | 0.0300246 | 0.026433 | 0.00359161 | 11.96% | 10.00% | no |

## Effective Gain Products

| Product | True | Recovered | Rel err |
| --- | ---: | ---: | ---: |
| shape (unscored knob) | 0.563833 | 0.56389 | 0.01% |
| pw (unscored knob) | 0.663941 | 0.66394 | 0.00% |
| fBase (unscored knob) | 712.059 | 712.214 | 0.02% |
| fAmt (unscored knob) | 626.336 | 625.793 | 0.09% |
| fDecay (unscored knob) | 0.0453753 | 0.0452813 | 0.21% |
| res (unscored knob) | 1.32646 | 1.32512 | 0.10% |
| attackTime (unscored knob) | 0.0143289 | 0.0168851 | 17.84% |
| decayTime (unscored knob) | 0.118242 | 0.060796 | 48.58% |
| sustain (unscored knob) | 0.859201 | 0.758855 | 11.68% |
| releaseTime (unscored knob) | 0.0301307 | 0.0301046 | 0.09% |
| drive (unscored knob) | 0.992184 | 1.13337 | 14.23% |
| outGain (unscored knob) | 0.482116 | 0.4813 | 0.17% |
