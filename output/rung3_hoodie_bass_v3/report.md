# SynthID Report

- Rung: 3
- Pass: no
- Init loss: 2.476320
- Final loss: 1.241752
- Loss ratio: 0.501451

## Independent Rung 3 Comparison

- Initialization MR-STFT distance: 0.163982
- Learned MR-STFT distance: 0.088033
- Improvement: 46.32%
- Required improvement: 80.00%
- Log-magnitude epsilon: 0.001
- Magnitude normalization: hann coherent gain (sum(window) / 2)
- FFT windows: 256, 512, 1024, 2048
- Capture high-pass: 30 Hz (zero-phase comparator policy)
- Result: fail

## Recovered Patch

| Parameter | Value | Unit |
| --- | ---: | --- |
| f0 | 32.7544 | Hz |
| attackTime | 0.0114654 | s |
| decayTime | 0.0622098 | s |
| sustain | 0.3 | lin |
| noteOff | 1.46105 | s |
| releaseTime | 0.0566007 | s |
| brightnessDecay | 6.96473 | 1/s/harmonic |
| drive | 0.709398 | lin |
| outGain | 0.274759 | lin |
| h1s | 0.75 | lin |
| h1c | 0.717563 | lin |
| h2s | 0.345917 | lin |
| h2c | 0.0684673 | lin |
| h3s | 0.75 | lin |
| h3c | 0.742475 | lin |
| h4s | -0.0620197 | lin |
| h4c | -0.0543746 | lin |
| h5s | -0.0222671 | lin |
| h5c | 0.229248 | lin |
| h6s | -0.0868195 | lin |
| h6c | -0.0550715 | lin |
| h7s | -0.0703805 | lin |
| h7c | -0.0215254 | lin |
| h8s | -0.0281712 | lin |
| h8c | -0.00255767 | lin |
| h9s | 0.0103871 | lin |
| h9c | -0.0163445 | lin |
| h10s | 0.0042031 | lin |
| h10c | -0.000931542 | lin |
| h11s | 0.0127887 | lin |
| h11c | 0.0104689 | lin |
| h12s | 0.00230161 | lin |
| h12c | 0.00452093 | lin |
| h13s | -0.00387547 | lin |
| h13c | 0.00782317 | lin |
| h14s | -0.00144935 | lin |
| h14c | 0.00182052 | lin |
| h15s | -0.00183266 | lin |
| h15c | 0.00391064 | lin |
| h16s | -0.00112933 | lin |
| h16c | 0.000628328 | lin |
| bh2s | 0.144601 | lin |
| bh2c | 0.128705 | lin |
| bh3s | -0.266232 | lin |
| bh3c | 0.434875 | lin |
| bh4s | 0.0970797 | lin |
| bh4c | -0.315641 | lin |
| bh5s | 0.212216 | lin |
| bh5c | 0.279395 | lin |
| bh6s | -0.163046 | lin |
| bh6c | -0.0126265 | lin |
| bh7s | -0.206866 | lin |
| bh7c | -0.0220691 | lin |
| bh8s | -0.0186833 | lin |
| bh8c | 0.0549444 | lin |
| bh9s | -0.0562313 | lin |
| bh9c | 0.384 | lin |
| bh10s | 0.0295112 | lin |
| bh10c | 0.0236149 | lin |
| bh11s | -0.18455 | lin |
| bh11c | 0.229599 | lin |
| bh12s | 0.014744 | lin |
| bh12c | -0.0315158 | lin |
| bh13s | -0.15137 | lin |
| bh13c | 0.143146 | lin |
| bh14s | 0.0149182 | lin |
| bh14c | -0.0139369 | lin |
| bh15s | -0.0141654 | lin |
| bh15c | -0.0454825 | lin |
| bh16s | 0.00134457 | lin |
| bh16c | -0.00845571 | lin |

## Residual Mismatch

The learned patch is constrained to the fixed body, click, and filtered-noise voice. Any remaining difference in `compare.png` or `ab.wav`—especially attack/beater texture and the late decay—is treated as model mismatch rather than hidden lookup data.
