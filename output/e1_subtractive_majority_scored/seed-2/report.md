# SynthID Report

- Rung: 1
- Pass: yes
- Init loss: 11.900508
- Final loss: 0.038825
- Loss ratio: 0.003262

Note: the subtractive topology is scored on its declared invariants: filter-EG endpoints, resonance, effective output-envelope levels, and scale-free envelope ratios. Individual knobs along compensating ridges are diagnostic only.

| Parameter | True | Recovered | Abs err | Rel err | Tol | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fEnv(0) | 1338.4 | 1338.01 | 0.389038 | 0.03% | 10.00% | yes |
| fEnv(infinity) | 712.059 | 712.214 | 0.154541 | 0.02% | 10.00% | yes |
| res | 1.32646 | 1.32512 | 0.00133789 | 0.10% | 10.00% | yes |
| effective output envelope (10ms) | 0.237561 | 0.234869 | 0.00269243 | 1.13% | 10.00% | yes |
| effective output envelope (75ms) | 0.444332 | 0.44693 | 0.0025984 | 0.58% | 10.00% | yes |
| effective output envelope (300ms) | 0.416304 | 0.414873 | 0.00143039 | 0.34% | 10.00% | yes |
| effective output envelope (700ms) | 0.0143622 | 0.0144189 | 5.66905e-05 | 0.39% | 10.00% | yes |
| aEnv(75ms)/aEnv(10ms) | 1.87039 | 1.90289 | 0.0325046 | 1.74% | 10.00% | yes |
| aEnv(300ms)/aEnv(10ms) | 1.75241 | 1.7664 | 0.0139986 | 0.80% | 10.00% | yes |
| aEnv(700ms)/aEnv(300ms) | 0.0344993 | 0.0347549 | 0.000255592 | 0.74% | 10.00% | yes |

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
