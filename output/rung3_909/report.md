# SynthID Report

- Rung: 3
- Pass: no
- Init loss: 2.100669
- Final loss: 1.812276
- Loss ratio: 0.862714

## Independent Rung 3 Comparison

- Initialization MR-STFT distance: 0.038135
- Learned MR-STFT distance: 0.027011
- Improvement: 29.17%
- Required improvement: 80.00%
- Log-magnitude epsilon: 0.001
- Magnitude normalization: hann coherent gain (sum(window) / 2)
- FFT windows: 256, 512, 1024, 2048
- Capture high-pass: 30 Hz (zero-phase comparator policy)
- Result: fail

## Recovered Patch

| Parameter | Value | Unit |
| --- | ---: | --- |
| fStart | 231.01 | Hz |
| fEnd | 46.8082 | Hz |
| pitchDecay | -47.1805 | 1/s |
| bodyAmp | 0.80536 | lin |
| ampDecay | -12.7337 | 1/s |
| clickFreq | 529.191 | Hz |
| clickAmp | 0.5 | lin |
| clickDecay | -288.059 | 1/s |
| noiseCutoff | 8000 | Hz |
| noiseAmp | 0.00247541 | lin |
| noiseDecay | -20 | 1/s |
| drive | 3.979 | lin |
| outGain | 0.178562 | lin |
| bodyAsymmetry | 0.0324212 | lin |
| bodyHarmonic | -0.555949 | lin |

## Residual Mismatch

The learned patch is constrained to the fixed body, click, and filtered-noise voice. Any remaining difference in `compare.png` or `ab.wav`—especially attack/beater texture and the late decay—is treated as model mismatch rather than hidden lookup data.
