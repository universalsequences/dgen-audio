# SynthID Report

- Rung: 3
- Pass: no
- Init loss: 3.932807
- Final loss: 2.282671
- Loss ratio: 0.580418

## Independent Rung 3 Comparison

- Initialization MR-STFT distance: 0.112232
- Learned MR-STFT distance: 0.034797
- Improvement: 69.00%
- Required improvement: 80.00%
- Log-magnitude epsilon: 0.001
- Magnitude normalization: hann coherent gain (sum(window) / 2)
- FFT windows: 256, 512, 1024, 2048
- Capture high-pass: 30 Hz (zero-phase comparator policy)
- Result: fail

## Recovered Patch

| Parameter | Value | Unit |
| --- | ---: | --- |
| fStart | 276.593 | Hz |
| fEnd | 46.9171 | Hz |
| pitchDecay | -52.4121 | 1/s |
| bodyAmp | 0.861252 | lin |
| ampDecay | -11.1345 | 1/s |
| clickFreq | 644.386 | Hz |
| clickAmp | 1.2 | lin |
| clickDecay | -483.615 | 1/s |
| noiseCutoff | 12714.4 | Hz |
| noiseAmp | 0.00105819 | lin |
| noiseDecay | -11.7784 | 1/s |
| drive | 3.51912 | lin |
| outGain | 0.163463 | lin |
| bodyAsymmetry | -0.0139367 | lin |
| bodyHarmonic | -0.52028 | lin |
| ampCurve | -2.87532 | 1/s^2 |

## Residual Mismatch

The learned patch is constrained to the fixed body, click, and filtered-noise voice. Any remaining difference in `compare.png` or `ab.wav`—especially attack/beater texture and the late decay—is treated as model mismatch rather than hidden lookup data.
