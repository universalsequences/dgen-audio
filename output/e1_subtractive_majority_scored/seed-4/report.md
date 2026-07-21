# SynthID Report

- Rung: 1
- Pass: no
- Init loss: 4.295578
- Final loss: 1.088307
- Loss ratio: 0.253355

Note: the subtractive topology is scored on its declared invariants: filter-EG endpoints, resonance, effective output-envelope levels, and scale-free envelope ratios. Individual knobs along compensating ridges are diagnostic only.

| Parameter | True | Recovered | Abs err | Rel err | Tol | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fEnv(0) | 2097.82 | 2088.22 | 9.59253 | 0.46% | 10.00% | yes |
| fEnv(infinity) | 1995.03 | 1985.24 | 9.79248 | 0.49% | 10.00% | yes |
| res | 2.01405 | 2.10732 | 0.0932665 | 4.63% | 10.00% | yes |
| effective output envelope (10ms) | 0.395521 | 0.324564 | 0.0709576 | 17.94% | 10.00% | no |
| effective output envelope (75ms) | 0.730012 | 0.586923 | 0.143089 | 19.60% | 10.00% | no |
| effective output envelope (300ms) | 0.640406 | 0.526423 | 0.113983 | 17.80% | 10.00% | no |
| effective output envelope (700ms) | 0.170336 | 0.141334 | 0.0290022 | 17.03% | 10.00% | no |
| aEnv(75ms)/aEnv(10ms) | 1.8457 | 1.80835 | 0.0373509 | 2.02% | 10.00% | yes |
| aEnv(300ms)/aEnv(10ms) | 1.61915 | 1.62194 | 0.00279593 | 0.17% | 10.00% | yes |
| aEnv(700ms)/aEnv(300ms) | 0.265981 | 0.26848 | 0.0024983 | 0.94% | 10.00% | yes |

## Effective Gain Products

| Product | True | Recovered | Rel err |
| --- | ---: | ---: | ---: |
| shape (unscored knob) | 0.452019 | 0.540748 | 19.63% |
| pw (unscored knob) | 0.758204 | 0.747103 | 1.46% |
| fBase (unscored knob) | 1995.03 | 1985.24 | 0.49% |
| fAmt (unscored knob) | 102.782 | 102.982 | 0.19% |
| fDecay (unscored knob) | 0.0642665 | 0.0781998 | 21.68% |
| res (unscored knob) | 2.01405 | 2.10732 | 4.63% |
| attackTime (unscored knob) | 0.0138311 | 0.0129759 | 6.18% |
| decayTime (unscored knob) | 0.239436 | 0.596627 | 149.18% |
| sustain (unscored knob) | 0.810735 | 0.778013 | 4.04% |
| releaseTime (unscored knob) | 0.0991403 | 0.102852 | 3.74% |
| drive (unscored knob) | 2.01939 | 1.51933 | 24.76% |
| outGain (unscored knob) | 0.384495 | 0.400351 | 4.12% |
