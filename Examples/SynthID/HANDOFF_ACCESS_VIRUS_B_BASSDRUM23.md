# Access Virus B BassDrum_23 — SynthID handoff (2026-09-03)

Target: eseq sample-library `BassDrum_23.wav`, tags `Access`,
`Access Virus - B`, `bass`, `drums`, `kick`. The canonical lossless asset is
`Assets/access-virus-b-bassdrum-23.wav`, copied from hash
`32cee493358b8dd6e60a5e82761c21b2f7e42445f9488b522bb805668d5579cf`.
The library also contains hash `d584b7daf13436d65d602fac23754242c81829576d582ea83576d452d329d0a2`
with the same title/tags at 44.1 kHz. Resampling the native 32.5 kHz file to
44.1 kHz gives correlation 0.999999993 and max difference 6.0e-4, so the
second entry is a resampled duplicate, not a different hit.

## Measurements

`Examples/SynthID/scripts/analysis/analyze_access_virus_b_bassdrum23.py`
records the complete measurement output.

- 16-bit stereo, 32,500 Hz, 14,987 frames / 461.14 ms. Channels are not
  identical (max L-R 0.07294); fitting uses the host's mono decode. Peak
  -0.10 dBFS, DC -7.6e-5, final-20-ms RMS -59.5 dBFS.
- -40 dB support 1.05..357.42 ms; -60 dB support reaches 452.62 ms.
- The voice is a pitched swept kick. Positive zero crossings show roughly
  1.6 kHz at the first measurable period, 524 Hz at 11 ms, 214 Hz at 23 ms,
  108 Hz at 35 ms, and a stable 48.2 Hz tail. A single exponential fit gives
  approximately fStart 1.8 kHz, fEnd 48.5 Hz, pitchDecay -113/s. The old
  2048-point autocorrelation path smeared this unusually broad sweep and
  clipped fStart to 500 Hz; the profile now uses a deterministic,
  linearly-interpolated positive-zero-crossing contour.
- The amplitude rises through the first 30-50 ms, then decays at about
  -6.3/s (80-160 ms) and -19/s (160-450 ms). The voice therefore adds one
  profile-scoped, normalized one-pole attack scalar and retains the existing
  log-quadratic decay scalar.
- The target is dominated by the swept body. Early harmonics follow the
  sweep, but a legal 20-scalar harmonic/envelope probe improved independent
  distance by only 0.00083, so that capacity was rejected rather than shipped.
- The native Nyquist is 16.25 kHz. There is no meaningful target energy above
  it and the recovered noise path is tiny, so no fitted output EQ/FIR was added.

## Voice and profile

Profile: `access-virus-b-kick`. It uses the compact kick topology:
closed-form exponential swept sine, profile-scoped normalized attack,
log-quadratic decay, attack-localized H2, shared-envelope H3/H5, decaying sine
click, filtered deterministic noise, and tanh output. All parameters are
ordinary bounded scalars. No waveform samples, lookup/residual tables, target
arrays, or learned filter taps enter the patch.

The profile adds one trainable scalar (`attackTime`) and a pitch-search profile.
Renderer parity for random profile parameters is 3.67e-5 max abs (threshold
1e-3). `attackTime` fdcheck is 1.11e-2 relative error at transformed epsilon
0.002. A two-exponential pitch experiment was rejected: it improved the gate
by only 0.0007 and its new pitch gradients failed fdcheck, so none of that
capacity remains in the implementation.

## Run history

| Run | Independent result | Diagnosis |
|---|---:|---|
| `output/access_virus_b_bassdrum23_v1` | 59.72%, 0.172009 -> 0.069277 | Autocorrelation pitch fit clipped fStart to 500 Hz; click compensated at ~1.1 kHz. Root cause fixed. |
| `output/access_virus_b_bassdrum23_v2` | 63.78%, 0.167315 -> 0.060608 | Correct zero-crossing pitch fit; unnormalized attack created gain/attack degeneracy and pinned bodyAmp/attack. Root cause fixed. |
| `output/access_virus_b_bassdrum23_v3` | **68.35%, 0.181475 -> 0.057440** | Normalized attack; two restarts x 300 epochs, 100 pitch-refine epochs, deterministic coordinate refinement. Final compact result. |

The 80% relative gate is not passed. The absolute result and failed gate are
intentional and documented: legal harmonic and second-sweep probes did not
close the remaining 0.0211 distance, while memorizing the machine/capture
texture would violate the SynthID contract. The v3 A/B is the final ear gate.

## Final scalars

- fStart 1799.3625 Hz; fEnd 48.148235 Hz; pitchDecay -111.90619/s
- bodyAmp 1.9785769; attackTime 0.12940475 s
- ampDecay -16.818668/s; ampCurve -4.7149763/s^2
- bodyAsymmetry -0.032858804; bodyHarmonic -0.11710036
- clickFreq 758.0678 Hz; clickAmp 0.038429655; clickDecay -100/s
- noiseCutoff 3477.9265 Hz; noiseAmp 0.002707424; noiseDecay -11.298755/s
- drive 1.0728197; outGain 1.2074164

## eseq deployment and parity

Factory instrument: `content/instruments/Drums/Virus B BassDrum 23/` plus
`content/instruments/Drums/Virus B BassDrum 23.presets` in the eseq checkout.
At 44.1 kHz and host pitch 385.18588 Hz, its DGenLisp render matches v3
`learned.wav` with max abs 3.9965e-5, zero samples above 1e-3, RMS difference
7.47e-6. Independent distance is 0.0574485 versus learned.wav's 0.0574403.

Listen to:

`output/access_virus_b_bassdrum23_v3/ab.wav`
