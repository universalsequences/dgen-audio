# TR-909 Open Hat (HHOD0) Identification — Handoff (2026-09-03)

Target: `Assets/909-open-hat.wav` (the eseq sample-library file `HHOD0.WAV`,
sha256 22faaac3…, tagged 909 / hi-hat / open; the ROM hat through the open-hat
VCA at decay 0). 16-bit, **44,100 Hz**, 253 ms, peak −0.0 dBFS, no DC,
6,450 distinct sample values (a resampled recording, not the raw 6-bit ROM).
`HHOD2/4/A.WAV` are the same hit at longer decay settings (331 / 399 / 517 ms,
identical peaks); the user named HHOD0. Fitted at 48 kHz (FFT resample).
Deployed as `content/instruments/Drums/909 Open Hat` in eseq. All work here
uncommitted. Bead: eseq-jd3u.

## Measurements (`scripts/analysis/analyze_909_open_hat.py`)

- Onset: 9 near-silent samples, then the hit opens within 15 samples. 1 ms RMS
  envelope: −4 dB at 0, −1 dB from 4 to 20 ms, then the VCA: −5 dB at 24–32 ms,
  −11 at 64, −17 at 96, −23 at 160, −29 at 192, −36 at 248 ms where the sample
  ends (hard cut at −36 dB; final 10 ms RMS −41.7 dB). Decay 30–240 ms
  −15.7/s (T60 440 ms); 30–80 ms steeper (−24/s) then −12..−17/s.
- Spectrum: broad plateau 3–10 kHz, a −10 dB shelf above 12.5 kHz, energy to
  20 kHz (no machine band-limit; the source is 44.1 kHz). Narrow metal peaks
  9–17 dB above the local mean in the tail: 647, 1486, 1548, 3407, 3941, 4272,
  5023, 5225, 8311, 9207, 12553, 13433 Hz. Each peak's ±60 Hz band decays at
  the VCA rate (−6..−20/s), i.e. the modes ring longer than the sample: the
  ring is shaped by the envelope, not by the modes' own damping.
- Per-band decay (20–120 ms): −18/s below 5 kHz, −22..−24/s above 8 kHz; the
  top loses ~5 dB more than the body over the first 100 ms.
- Onset thump: 0–3 ms has −18 dBFS in 200–400 Hz the tail never has.

## Voice (`scripts/fit_hat.py`, NumPy; mirrored 1:1 in the eseq dsp.lisp)

One DGen xorshift noise stream → **wash**: bandpass fc1/q1 + g2·bandpass
fc2/q2 + gHp·highpass hpFc, gain aB. **Modes**: twelve struck-once metal modes
`g_k · e^{d_k t} · sin(2π f_k t)` (frequency bounded ±5 % of the measured
peak, own ring decay d_k, amplitude g_k). **Envelope** (the open-hat VCA):
attack ramp `min(1, t/atk)`, hold, then `exp(dTail · max(t − hold, 0))` plus a
fast early stage `aFast · exp(dFast t)` on the wash; the modes get the same
ramp/hold with their own dMode. Gain-normalised `tanh(drive·x)/drive · outGain`
→ output highpass outHp. 53 scalars, documented bounds, no target-derived
tables. No fixed output stage: the source is full-band.

## Method notes

- **The gate is blind here.** compare.py's per-bin metric between the same
  patch and a second noise realisation (the metric floor) is 2.35–2.40 for this
  voice, against a midpoint baseline of 2.86: the noise wash dominates every
  bin. Every round from v1 on scores *below* its own floor. The numbers that
  drive the work are the band-pooled training loss and the deficit table.
- Noise-excited high-Q resonators (v1/v2, Q up to 3000) let the fitter chase
  the particular noise realisation inside each resonator: mode frequencies
  walked to their bound edges and one Q pinned at 3000. A real cymbal mode is
  struck once and rings freely, so v3 made the modes deterministic decaying
  sines. That is also what the eseq port needs for exact parity.
- The pooled loss (per-window band count clamp(w/16, 32, 256)) dilutes one 2 %
  band to 1/256 of a window's error and trades a mode for broad fill: v3 lost
  the 1486 Hz mode onto 1548 and collapsed two modes onto 5000 Hz. v4 adds a
  per-bin log-magnitude term at the 4096/8192 windows (500 Hz–15 kHz, weight 1)
  — the analogue of the pitched voices' harmonic-track loss.
- **Port gotcha (new):** summing 1/samplerate into a history sample by sample
  drifts in float32 — 0.35 cycles on a 5 kHz sine by 200 ms, 8.7e-3 max-abs on
  the hat. The hat template keeps an integer sample counter in the history and
  multiplies once (`t = n · (1/sr)`); parity dropped to 2e-5. The kick template
  still uses the summed ramp (its 1.3e-3 tail drift on the Virus kick is this).
- Onset thump: a zero-default lowpassed noise burst (clickAmp/clickDecay/
  clickFc). Under the pooled loss it pinned at its loudest and slowest to fill
  the −47 dBFS rumble under 250 Hz; the pooled bands now start at 250 Hz and
  the click is bounded (amp ≤ 2, decay ≤ −600/s). It still sits on both
  bounds in v7 and 200–400 Hz is 12 dB over the target's −45 dBFS at 8–15 ms
  (masked by −19..−5 dBFS content in the same window; the ear decides).

## Runs (pooled train loss is only comparable within one loss definition)

| Run | loss | v(prev) under this loss | gate | floor | notes |
|---|---|---|---|---|---|
| v1 | 3.091 | — | 2.364 | 2.402 | 8 noise-resonator modes, 96 bands; modes repurposed as broad fill |
| v2 | 2.771 | 3.160 | 2.362 | 2.391 | 12 modes ±5 %, per-window band schedule; 3 modes on bound edges, a Q pinned at 3000 |
| v3 | 2.733 | 3.095 | 2.336 | 2.352 | modes → decaying sines; 1486 lost onto 1548, two modes on 5000 Hz |
| v4 | 3.640 | 3.684 | 2.309 | 2.328 | + per-bin 4096/8192 term (diluted by noise bins, no effect) |
| v5a | 3.638 | — | 2.309 | 2.328 | reseed 1486/5225 + `--only`: coordinate descent walked both back |
| v6 | 3.133 | 3.479 | 2.334 | 2.346 | frequencies pinned ±0.5 %, **mode-track loss** (0.52 → 0.31), pool ≥ 250 Hz, click |
| **v7** | 3.131 | 3.134 | 2.327 | 2.339 | click/q2 bounds; converged (Δ 0.003); deployed |

Why the spectral losses could not place a sine (probe, v4 params): removing
the second 1548 Hz sine costs pooled +0.063; a 1486 Hz sine at the right level
recovers only +0.045 of it, and the per-bin 8192-window term moves by 0.002
either way — the target has 1486 and 1548 at equal strength (+17.7 / +17.8 dB
over the local mean), but one mode is 3 bins of 2,500. The `ModeTracks` class
(32 ms Hann heterodyne per measured mode, 5 ms hop, −60 dB floor) is the
diagnostic and the loss term; `deficit_table.py --fit-module` prints its table.

v7 mode table (synth − target, dB) is flat to ±3 dB from 25 ms on except
3941 Hz (+7 at 125 ms), 9207 Hz (+9 at 75 ms, −11 at 200 ms) and the 0 ms
column, where the modes are under the attack noise. Deficit table: every band
above 400 Hz within ±2.5 dB from 15 ms on except 800–1500 Hz (+2..+5 short in
the tail); >4 kHz RMS per 50 ms matches to 1 dB throughout.

## v7 params

wash: fc1 8009 Hz q 0.87; fc2 815 Hz q 0.2 ×0.33; hp 14.6 kHz ×0.39; aB 0.99.
env: atk 7.3 ms, hold 8.3 ms, dTail −15.1/s, fast 9.4·e^(−74.7 t), dMode −9.2/s.
click: 2.0·e^(−600 t) LP 815 Hz. out: drive 1.90 (gain-normalised), outGain
0.96, outHp 253 Hz. Modes (Hz / own decay /s / amp): 646/−9.1/0.073,
1483/−10.0/0.080, 1546/−16.0/0.269, 3394/−7.9/0.057, 3960/−7.1/0.185,
4287/−4.0/0.031, 5030/−1.0/0.047, 5208/−4.2/0.049, 8301/−12.0/0.150,
9225/−44.6/0.862 (a 9.2 kHz chick, not a ring), 12543/−4.6/0.044,
13428/−5.7/0.041. Pinned: clickAmp, clickDecay, m7d (5023 Hz has no damping of
its own: it rings at the VCA rate, as measured).

Deployment parity: the eseq instrument's 48 kHz render matches
`output/hat_v7/learned.wav` to 5.5e-4 max abs (0 samples > 1e-3) and scores
the identical gate distance (2.3253). Layout test
`metal_seq_fx_lisp_lays_out_909_open_hat_controls` passes (63 params, no
overlaps). A/Bs: `output/hat_v7/ab.wav` (target / learned ×2),
`output/hat_v7/ab_target_old_new.wav` (target / v4 / v7 ×2),
`output/hat_v7/ab_target_instrument.wav` (target / eseq render ×2).

## Open

- The ear: not yet heard. Character to listen for: the 1486/1548 pair, the
  200–400 Hz excess at 8–15 ms, whether the 9.2 kHz chick reads as the 909's
  attack.
- HHOD2/4/A (longer decays) are the same hit; the `decay` knob at ~1.3/1.6/2.0
  should approximate them, unverified.

## The swish (v8, 2026-09-04)

User on v7: "still missing the swish of the real 909 — close, very close".
Measured what the band tables cannot see: the target's 6–12 kHz envelope has
5.6× more slow (5–30 Hz) modulation energy than v7 and *less* fast (30–150 Hz)
flutter; above 12 kHz its detrended envelope wobbles 3.1 dB vs 1.6; its >2 kHz
centroid drifts 7.5→8.9→8.2→9.1 kHz through the tail. That is dense cymbal
partials beating — flat filtered noise only flutters fast.

Block: the high wash (bp1 + hp) × `exp(swAmp · slow(noise, swRate))`, where
`slow` is two cascaded one-pole lowpasses (k = 1 − e^(−2π fc/sr)). Zero-default.
`SwishStats` (per band 2–4/4–6/6–12/12–20 kHz: detrended 1 ms dB-envelope std,
log modulation energy 5–30 and 30–150 Hz) is the realisation-free statistic;
the pooled/gate losses would zero any modulator. Protocol: `--only
swRate,swAmp,gHp,hpFc,q1,fc1 --pooled-weight 0.3 --swish-weight 1 --mode-weight 0`
(v8a), then a full round with `--swish-weight 1` (v8), then `--only
swAmp,swRate` after the modulator change (v8b).

| run | pooled+mode+swish | swish stat | mode loss | gate | notes |
|---|---|---|---|---|---|
| v7 | — | 0.505 | 0.303 | 2.327 | no swish |
| v8a | 1.293 | 0.404 | — | 2.369 | swish alone, biquad modulator |
| v8 | 3.440 | 0.211 | 0.352 | 2.417 | full; 5–30 Hz energy matches in 2–6 kHz |
| **v8b** | 3.465 | ≈0.21 | 0.352 | 2.415 | one-pole modulator, swRate 5.05 Hz, swAmp 52.3; deployed |

The gate rose 2.33→2.42, as any texture it cannot phase-match does. hpFc
pinned at 16 kHz in v8 (bound now 20 kHz, not yet refitted). 12–20 kHz wobble
still 2.1 vs 3.1 dB.

**Port gotcha (new):** a biquad lowpass at 4.6 Hz broke parity (5.6e-3, gates
differ): its coefficients come from 1 − cos(w0), one float32 ulp of 1.0 in the
dgen runtime. Slow modulators are one-pole cascades on `make-history`.
Parity v8b: 5.0e-4 max abs, identical gate 2.4137.

A/Bs: `output/hat_v8b/ab_target_old_new.wav` (target / v7 / v8b ×2),
`ab_target_instrument.wav`, `ab_loop_target_new.wav` (8 hits target, 8 hits
v8b at 8th notes, 120 bpm).

## Attack dynamics correction (v13, follow-up to the user's second ear rejection)

The missing character is not only a spectral deficit. Over 0–20 ms the target's
amplitude kurtosis is 2.66, while v8b is 1.23: almost a flattened noise waveform.
Across four independent noise offsets v8b stays 1.21–1.24. Its large early wash
(aFast 16.72) drives the tanh hard. Spectral pooling did not protect transient
shape, and the swish-only correction did not address it.

Added `DynamicsStats` to `scripts/fit_hat.py`: short-window log RMS plus log
kurtosis, phase independent, weighted by `--dynamics-weight`. Three regression
tests in `scripts/test_hat_dynamics.py` cover identity/polarity, clipping with
RMS restored, and gain invariance of kurtosis. The drive bound now permits a
nearly linear stage (0.02 instead of 0.5). No added oscillators, waveform tables,
sample playback, or target-derived arrays in the instrument.

Runs: v9 refit dynamics alone; v10 full refit with dynamics weight 4, swish 1;
v11 raised swish weight from v10 (stuck in that basin); v12 restarted from v8b
with dynamics/swish weights 4; v13 refit v12 with mode/dynamics/swish weights 4.
Do not use the new midpoint improvement percentage to compare with yesterday:
the drive bounds changed. Compare the same target and v8b directly instead.

| Diagnostic (raw WAV, lower is better) | v8b | v13 |
| --- | ---: | ---: |
| DynamicsStats | 1.215 | 0.343 |
| SwishStats | 0.230 | 0.219 |
| ModeTracks | 0.352 | 0.412 |
| 0–20 ms kurtosis (target 2.66) | 1.23 | 2.56 |

v10 was rejected despite good attack dynamics because its swish error doubled
(to 0.472). v13 preserves the swish statistic and reduces the flattening. Its
attack kurtosis is 2.28–2.56 across four noise offsets. The correction is a
substantial reduction in early wash drive (aFast 4.61), with a rebalanced output
highpass, metal gains, wash filters and slow modulation. The scalar topology
and all departure controls are unchanged.

**Limitations, not a claim of an exact match:** mode-track error rises by about
0.6 dB on average; the attack's first 5 ms remains about 2 dB quiet; 12–20 kHz
slow undulation is still short. The existing 12-mode/noise model is an
approximation of a sampled cymbal. This candidate needs the user's ear before
closing eseq-jd3u. No subjective listening verdict was fabricated from metrics.

Deployed `output/hat_v13/recovered_params.json` to eseq's
`content/instruments/Drums/909 Open Hat/dsp.lisp`. Template ranges permit the
linear drive bound and round the 5048.2155 Hz mode bound outward to accommodate
the rendered scalar literal. Skill notes now require attack dynamics checks.

Validation:
- Compiled port parity max abs 3.90e-4, zero samples above 1e-3; independent
  gate instrument/reference both 2.2932 (v8b deployed gate 2.4137).
- Existing host `instrument_probe` binary, explicitly using the pinned compiler:
  48 kHz / 48,000 frames, peak 0.9000, RMS 0.07764, no nonfinite samples/state.
- Actual compiled DSP at 44.1 and 48 kHz, repeated triggers, defaults plus
  decay 2.4 / 0.35, swish 0, drive 3: finite, peaks below 0.967, decaying tails.
- Fusion checker clean for both sample-rate builds. Three Python tests pass.
- No UI or Rust changes, hence no UI capture or broad Rust test suite.

Listen: `/tmp/909-open-hat-ab.wav`, also durable in
`output/hat_v13/ab_target_old_new.wav`: target / v8b / compiled v13, repeated
three times, each hit RMS-matched to 0.15 for comparison. Actual unnormalised
compiled playback: `output/hat_v13/instrument.wav` and
`output/hat_v13/retriggered_instrument.wav`. Reload the factory instrument to
pick up the new source/defaults; previously saved instances may retain theirs.

## Body restoration (v15, user: too clear/high-frequency noise, missing body)

The user's rejection of v13 exposed the bad dynamics tradeoff: output highpass
had climbed to 1926 Hz. In the first 20 ms, 250–1000 Hz accounts for -17.2 dB
of target energy but only -25.8 dB in v13. This is not a uniform treble excess;
the low-mid body is missing, and simply lowpassing the output would disguise
rather than repair it. The deficit table also shows 200–400 Hz missing by
25–29 dB through much of v13.

Added `BandBalanceStats` and `--balance-weight` to fit_hat.py. It compares
integrated, Hann-windowed band-energy shares in five time regions (0–10,
10–30, 30–60, 60–120, 120–240 ms), with edges 200/400/800/1500/3000/6000/
10000/16000/22000 Hz. Global gain cannot improve it, and quiet low-mid energy
no longer disappears under the main objective's per-bin log floor. Two new
unit tests prove gain/polarity invariance and detection of body loss even
when total RMS is restored; all five objective tests pass.

v14 started from v13 but remained in its highpass/large-mode-gain basin (loss
8.10 after five passes); stopped it rather than spend further time there.
v15 starts from the fuller-bodied v10 (initial loss 7.66) and converges to 6.447:

```
python3 Examples/SynthID/scripts/fit_hat.py --out output/hat_v15 \
  --start output/hat_v10/recovered_params.json --restarts 0 --keep 1 \
  --passes 7 --steps 11 --final-passes 4 \
  --mode-weight 2 --swish-weight 2 --dynamics-weight 2 --balance-weight 4
```

Key changes: output HP 1926 → 167 Hz; extra highpassed noise gain 0.142 → 0;
low/body wash gain 0.352 → 0.574; mode levels rebalanced instead of pushing a
large low sine through a high output HP. Output drive 1.848 → 0.610. The
first-20-ms body share is now -18.7 dB: about 7.1 dB restored, 1.5 dB short
of the sample instead of 8.6 dB short. No new sound layer or sample playback.

| Same diagnostics, raw WAV | v13 | v15 |
| --- | ---: | ---: |
| BandBalanceStats | 1.192 | 0.269 |
| DynamicsStats | 0.343 | 0.467 |
| SwishStats | 0.219 | 0.275 |
| ModeTracks | 0.412 | 0.372 |

**Tradeoffs remain explicit:** attack/swish statistics are slightly worse than
v13, while broad balance and mode tracking improve. The first 5 ms remains
quiet and 400–800 Hz at 60–120 ms is still about 7 dB short. This is another
ear candidate, not a claim of perfect sample reproduction.

Because the fit needs g_hp=0, the old BRIGHT implementation would be a dead
knob. It now also shifts the main wash bandpass by `2^(bright-1)` (unity at
its default 1). All biquad frequencies are clamped to 20 Hz..0.45*samplerate
before DGen's 44.1k coefficient conversion, so brightness/tune departures
cannot push the filters past Nyquist. No added parameters or UI changes.
The template header now accurately says successive noise hits are not identical.

Validation of shipped v15:
- Port parity max 6.77e-4; no samples >1e-3. Instrument/reference gate
  2.1320/2.1321 (v13 instrument 2.2932).
- Host instrument_probe using the pinned compiler: peak .9000, RMS .08285,
  no nonfinite samples or state, 48k frames at 48 kHz.
- Actual compiled DSP at 44.1/48 kHz: defaults, decay 2.4/.35, swish 0,
  drive 3, bright 0/4, tune -24/+24 and combined bright4/tune24/swish4, with
  retriggers. All finite with decaying tails; BRIGHT changes the waveform at
  both extremes despite g_hp=0. Peaks up to 1.123 under these departures,
  so leave normal floating-point mix headroom. Fusion checks clean.
- Five Python objective tests pass. No Rust/UI structure changes.

A/B `/tmp/909-open-hat-body-ab.wav`, durable copy
`output/hat_v15/ab_target_old_new.wav`: target / compiled v13 / compiled v15,
repeated three times, each hit RMS-matched to .15. `instrument.wav` and
`retriggered_instrument.wav` contain actual unnormalised compiled output.
Factory source/defaults deployed; user listening verdict still pending on
eseq-jd3u. Reload the factory instrument for existing saved instances.

## Ear-approved factory default: SWISH OFF

User confirmed the crumbling-paper texture disappears with SWISH muted and
called the result "super duper close" to the sample. Do not refit the body again
based on the modulation statistic: the synthetic swish was the perceptual error.

Changed only the `swish` default from 1 to 0 in factory dsp.lisp and the hat
template, plus explanatory comments. All other fitted values, gain, DSP routing,
and controls are untouched. The swish remains an optional effect; existing
Swishy/No Swish presets still explicitly select their respective values.

The factory default is now an intentional EAR VOICING of v15, not the original
v15 optimiser output. For original-fit parity explicitly render with `swish=1`;
for factory-default parity use the saved compiled `no_swish.wav` in
`output/hat_v15_texture_isolation` (or render the Python model with swAmp=0,
retaining the v15 output-gain normalization). Do not run the old unmodified
synthid_port_check command and interpret its default-vs-fit mismatch as a
compiler regression, or re-enable swish merely to improve that score.

Verified fresh compiled defaults against the previously auditioned no-swish
render: max difference 5.54e-5 (PCM quantization). Explicit swish=1 still matches
the prior current render to 5.39e-5. Fusion check clean. Pinned host probe passes:
peak .9461, RMS .09017 at 48 kHz/48,000 frames, no nonfinite samples or state.
Existing loaded/saved synths may keep their old value: set SWISH to zero or reload
the factory source/defaults. No other sonic adjustments were made in this step.


