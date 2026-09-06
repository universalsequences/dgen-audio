# R-8 MkII '808Clap' Identification — Handoff (2026-09-03)

Target: `Assets/808-clap-r8.wav` (the eseq sample-library file
`808Clap.wav`, sha256 9a4dcc0f…, tagged Roland R-8 MKII). 16-bit stereo,
channels identical, **26,040 Hz** native, 434 ms, peak −0.3 dBFS, no DC.
Fitted at 48 kHz (FFT resample; the source is band-limited at 13 kHz).
Deployed as `content/instruments/Drums/808 Clap` in eseq. All work here
uncommitted.

## Measurements (`scripts/analysis/analyze_808_clap.py`)

- Four bursts at 0.5 / 10.1 / 21.4 / 29.7 ms (spacings 9.7 / 11.3 / 8.3 ms,
  irregular), each with a −10 dB sub-peak ~3.5 ms later; per-burst decay
  ~−15 dB in 2.5 ms. Tail from ~32 ms: two stages, −36..−41/s to 80 ms then
  −15/s (T60 ≈ 450 ms). −60 dB end at 355 ms; noise floor dead (−97 dB).
- Spectrum: broad band 600–2000 Hz (peak bins ~1.3 kHz early, ~0.9–1.2 kHz in
  the tail), a 2 kHz bump in the bursts, −3 dB/oct above 1.3 kHz, nothing
  above 13 kHz (R-8 output rate).

## Voice (`scripts/fit_clap.py`, NumPy; mirrored 1:1 in the eseq dsp.lisp)

One DGen xorshift noise stream → bandpass fc1/q1 + g2·bandpass fc2/q2 (+ an
optional highpass component, fitted to ~0) → four bursts (onsets 0, sp1,
sp1+sp2, sp1+sp2+sp3; levels 1, l2, l3, l4; exp bDecay; sub-burst at
subDelay×subGain) → tail from the last onset: tA1·exp(d1 t) on the band
source + tA2·exp(d2 t) on a lowpassed copy → tanh(drive)·outGain → fixed
4th-order 12 kHz lowpass (the R-8 output stage, not fitted) → fitted highpass
outHp. 26 scalars, documented bounds, no target-derived tables.

## Method notes (why this differs from the kick runs)

- The gate metric (compare.py MR-STFT) has a **noise floor** for a noise
  voice: the same patch rendered with a different noise seed scores 0.46
  against the target. Report *excess over that floor*, not the 80% gate — the
  gate cannot be met by any noise-driven patch and the relative number is
  dominated by the midpoint baseline.
- Per-bin log-magnitude loss is swamped by Rayleigh variance and pulls the
  optimizer into a smeared 4 ms flam (v1/v2). Training on **band-pooled log
  power** (32 log bands, windows 64…2048) recovers the flam timing
  (v3: 9.0/11.8/8.2 ms vs measured 9.7/11.3/8.3). The gate is still
  compare.py, unchanged.
- 40% of v3's residual was >12 kHz, where the render was 4–8 dB too loud: the
  target is band-limited by the R-8's 26 kHz output. A fixed 12 kHz output
  lowpass (machine property) removed it (v4).
- DGen/dgenlisp `biquad` hardcodes 2π/44100 in its coefficients; the port
  scales cutoffs by 44100/samplerate. dgenlisp `accum` reset lags the fit's
  n/sr ramp by one sample; the port adds 1/sr back. Verified: dgenlisp noise
  == `render_reference.dgen_noise`, biquad == RBJ @ 44.1 kHz to 2e-6.

## Runs

| Run | train (pooled) | gate learned | floor | excess | notes |
|---|---|---|---|---|---|
| v1 | — | 0.5564 | 0.4116 | 0.145 | per-bin loss, flam collapsed to 4 ms |
| v2 | 0.988 | 0.5643 | 0.4117 | 0.153 | + HP component, fine windows; still smeared |
| v3 | 1.336 | 0.7677 | 0.5045 | 0.263 | pooled loss: flam recovered; HF residual |
| v4 | 1.120 | 0.4903 | 0.4660 | 0.024 | + fixed 12 kHz output LPF, wider bounds |
| v5 | 1.053 | 0.4818 | 0.4650 | 0.017 | + outHp scalar |
| **v6** | 1.050 | **0.4813** | 0.4608 | 0.021 | widened outHp/l2; converged, deployed |

v6 params: fc1 955 Hz q 1.83; fc2 1823 Hz q 0.62 ×2.08; flam 8.97 / 11.87 /
8.48 ms; bDecay −369/s; levels 1 / 1.66 / 1.02 / 0 (the 4th burst is the
tail onset); sub-burst 1.52 ms ×0.86; tail 0.57·e^(−47 t) + 0.48·e^(−15 t);
drive 2.32; outGain 1.79 (×0.4693 peak normalisation folded into the
instrument's out_gain); outHp 655 Hz. Tail decays: 40–80 ms −40.0/s
(target −40.7), 160–300 ms −16.1/s (target −15.0).

Deployment parity: the eseq instrument's 48 kHz render matches
`output/clap_v6/learned.wav` to 5e-4 max abs and scores the identical gate
distance. `ab.wav` = target / learned / target / learned.
