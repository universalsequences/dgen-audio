# SynthID Report

- Rung: 3
- Pass: no
- Init loss: 2.100669
- Final loss: 2.100669
- Loss ratio: 1.000000

## Independent Rung 3 Comparison

- Initialization MR-STFT distance: 0.038135
- Learned MR-STFT distance: 0.038135
- Improvement: 0.00%
- Required improvement: 80.00%
- Log-magnitude epsilon: 0.001
- Magnitude normalization: hann coherent gain (sum(window) / 2)
- FFT windows: 256, 512, 1024, 2048
- Capture high-pass: 30 Hz (zero-phase comparator policy)
- Result: fail

## Recovered Patch

| Parameter | Value | Unit |
| --- | ---: | --- |
| fStart | 267.189 | Hz |
| fEnd | 47.4055 | Hz |
| pitchDecay | -46 | 1/s |
| bodyAmp | 0.525 | lin |
| ampDecay | -14 | 1/s |
| clickFreq | 447.214 | Hz |
| clickAmp | 0.25 | lin |
| clickDecay | -346.41 | 1/s |
| noiseCutoff | 2828.43 | Hz |
| noiseAmp | 0.025 | lin |
| noiseDecay | -54.7723 | 1/s |
| drive | 2.5 | lin |
| outGain | 0.55 | lin |
| bodyAsymmetry | 0 | lin |
| bodyHarmonic | 0 | lin |

## Residual Mismatch

The learned patch is constrained to the fixed body, click, and filtered-noise voice. Any remaining difference in `compare.png` or `ab.wav`—especially attack/beater texture and the late decay—is treated as model mismatch rather than hidden lookup data.
