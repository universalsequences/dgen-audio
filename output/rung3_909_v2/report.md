# SynthID Report

- Rung: 3
- Pass: no
- Init loss: 3.914695
- Final loss: 2.207556
- Loss ratio: 0.563915

## Independent Rung 3 Comparison

- Initialization MR-STFT distance: 0.112018
- Learned MR-STFT distance: 0.037191
- Improvement: 66.80%
- Required improvement: 80.00%
- Log-magnitude epsilon: 0.001
- Magnitude normalization: hann coherent gain (sum(window) / 2)
- FFT windows: 256, 512, 1024, 2048
- Capture high-pass: 30 Hz (zero-phase comparator policy)
- Result: fail

## Recovered Patch

| Parameter | Value | Unit |
| --- | ---: | --- |
| fStart | 335.381 | Hz |
| fEnd | 46.9804 | Hz |
| pitchDecay | -66.1671 | 1/s |
| bodyAmp | 0.905274 | lin |
| ampDecay | -10.306 | 1/s |
| clickFreq | 201.192 | Hz |
| clickAmp | 1.2 | lin |
| clickDecay | -150 | 1/s |
| noiseCutoff | 12660.3 | Hz |
| noiseAmp | 0.0013904 | lin |
| noiseDecay | -12.4442 | 1/s |
| drive | 3.92004 | lin |
| outGain | 0.13279 | lin |
| bodyAsymmetry | -0.021022 | lin |
| bodyHarmonic | -0.341624 | lin |
| ampCurve | -4.58168 | 1/s^2 |

## Residual Mismatch

The learned patch is constrained to the fixed body, click, and filtered-noise voice. Any remaining difference in `compare.png` or `ab.wav`—especially attack/beater texture and the late decay—is treated as model mismatch rather than hidden lookup data.
