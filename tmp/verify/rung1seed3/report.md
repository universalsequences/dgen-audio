# SynthID Report

- Rung: 1
- Pass: no
- Init loss: 0.996006
- Final loss: 0.996006
- Loss ratio: 1.000000

Note: `tanh((bodyAmp·body + clickAmp·click + noiseAmp·noise)·drive)·outGain` depends only on the products `amp·drive` and `outGain`; parameter sets with equal products render identical audio. The products are scored; the factors are listed unscored under Effective Gain Products.

| Parameter | True | Recovered | Abs err | Rel err | Tol | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fStart | 91.345 | 82.049 | 9.29603 | 10.18% | 3.00% | no |
| fEnd | 52.5073 | 52.7933 | 0.285969 | 0.54% | 3.00% | yes |
| pitchDecay | -40.1566 | -25 | 15.1566 | 37.74% | 10.00% | no |
| ampDecay | -10.052 | -7.5 | 2.55205 | 25.39% | 10.00% | no |
| clickFreq | 2126.93 | 1341.64 | 785.293 | 36.92% | 10.00% | no |
| clickDecay | -277.897 | -424.264 | 146.367 | 52.67% | 20.00% | no |
| noiseCutoff | 4437.44 | 2828.43 | 1609.01 | 36.26% | 10.00% | no |
| noiseDecay | -162.532 | -154.919 | 7.6124 | 4.68% | 20.00% | yes |
| outGain | 0.688099 | 0.7 | 0.0119013 | 1.73% | 10.00% | yes |
| bodyHarmonic | 0 | 0 | 0 | 0.00% | 20.00% | yes |
| bodyAmp*drive | 1.30019 | 1.5 | 0.199808 | 15.37% | 10.00% | no |
| clickAmp*drive | 0.301348 | 0.65 | 0.348652 | 115.70% | 20.00% | no |
| noiseAmp*drive | 0.646078 | 0.3 | 0.346078 | 53.57% | 20.00% | no |

## Effective Gain Products

| Product | True | Recovered | Rel err |
| --- | ---: | ---: | ---: |
| bodyAmp*drive*outGain | 0.894661 | 1.05 | 17.36% |
| clickAmp*drive*outGain | 0.207357 | 0.455 | 119.43% |
| noiseAmp*drive*outGain | 0.444565 | 0.21 | 52.76% |
| bodyAmp (unscored factor) | 0.536433 | 0.75 | 39.81% |
| clickAmp (unscored factor) | 0.12433 | 0.325 | 161.40% |
| noiseAmp (unscored factor) | 0.266559 | 0.15 | 43.73% |
| drive (unscored factor) | 2.42377 | 2 | 17.48% |
| bodyAsymmetry (Rung 3 extension; unscored) | 0 | 0 | 0.00% |
