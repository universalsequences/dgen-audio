# TR-808 Low Tom Fitting — Handoff (2026-09-01)

Target: `Assets/808-tom-low.wav` (lossless 16-bit stereo, channels identical,
44.1 kHz, 432 ms, peak −11.0 dBFS, no DC). Goal: the same rung-3 recovery as
the 808 kick / 909 kick, deployed as `content/instruments/drums/synthid-808-tom`
in the eseq repo. All work here is **uncommitted** on `main`.

## Measurements (`scripts/analysis/analyze_808_tom.py`)

- Effective length 431 ms (−60 dB), 349 ms (−40 dB). Fit length 20,480 frames.
- Pitch: zero-crossing contour 108.6 → 93 Hz, flat after ~120 ms. Exponential
  fit fStart 113.3, fEnd 92.9, pitchDecay −20.5/s, residual 0.53 Hz RMS.
- Amplitude: one exponential, −13.5 / −12.8 / −12.2 s⁻¹ over 20–100 / 100–300 /
  300–600 ms — no steepening, so `ampCurve` stays pinned.
- Near-pure sine: H2 −40 dB, H3 −33 dB at onset. H2/H4 *persist* and rise
  relative to H1 in the tail (H2 −36 dB at 300 ms).
- Attack: waveform starts at a trough (cosine phase), no separate click ridge;
  1–15 kHz band 70 dB below the fundamental. Noise floor dead.

## Profile `808-tom` (implemented, 808/909 numerics untouched)

Same TR-808 tanh voice (`Patch.swift` default path); only bounds move:
fStart 80–220 log, fEnd 60–130 log, pitchDecay −80..−5 logneg, bodyAmp
0.2–1, ampDecay −25..−3, clickFreq 300–3000 log, clickAmp 0–1,
clickDecay −1600..−100 logneg, noise as 808, drive 1–3, outGain 0.1–1,
asymmetry/harmonic ±0.5/±1, ampCurve pinned ±0.001.

Touch points: `Params.swift` (`tr808Tom`, `all`), `Config.swift` profile
list, `Sources/DGenTrainProtocol/PitchTrack.swift` (`PitchSearchProfile.tr808Tom`
+ `forSynthIDProfile(_:)`, replacing the `909 ? : 808` ternaries in
`Trainer.swift`), `Trainer.swift` (capture-floor restart 4 now only for
`"808"`; the pitchDecay init standoff is now derived from the active table's
bounds — identical −76..−17 for the 808 table), `scripts/refine_rung3.py`
(`BOUNDS_808_TOM`), `scripts/score_params.py` / `render_reference.py` choices.

Lesson: rung 3 does **not** peak-normalize the target (preprocessing
`normalizationScale: 1`), so a −11 dBFS capture needs bodyAmp/outGain bounds
that reach below the kick table's 0.5 × 0.4 floor — the first shakedown
pinned both at their lower bound.

## Runs

| Run | Result | Notes |
|---|---|---|
| `output/rung3_808tom_shake` | 72.73%, learned 0.02664 | 1 restart × 40 epochs, bodyAmp/outGain pinned low (bounds since widened) |
| `output/rung3_808tom_v1` | **78.07%**, learned **0.014963** (baseline 0.068242) | 5 restarts × 600 epochs + refine; restart 1 won by CPU metric; gate **not** passed |

Recovered (v1): fStart 128.40, fEnd 93.00, pitchDecay −25.77, bodyAmp 0.318,
ampDecay −12.96 (T60 533 ms), bodyAsymmetry 0.017, bodyHarmonic 0.048,
clickFreq 429.6, clickAmp 0.191, clickDecay −1049, noiseCutoff 1561,
noiseAmp 3.4e-4, noiseDecay −4.59, drive 2.206, outGain 0.375, ampCurve
+0.00095 (at its pinned bound; negligible).

Absolute learned distance sits between the accepted 808 kick (0.0116) and
909 kick (0.0223). The relative gate misses because the tom profile's
midpoint baseline (0.068) is already close to the target — a narrower-bounds
profile has a smaller denominator.

## Residual diagnosis (why it was left at 78%)

A NumPy probe (render_reference + compare metric) of a zero-default
*persistent* second harmonic (sin 2φ on the body envelope, amplitude 0.005–0.08,
four phases, with/without extra decay) changed the metric by < 1e-5: the
target's −40 dB harmonics sit at the comparator's 1e-3 log epsilon, so the
metric cannot see them. An H4 term only hurt. The playbook's "probe in NumPy
before adding capacity" rule says stop here; the remaining distance is in the
150–400 Hz band spread across the whole note and is not reachable by a
rule-legal scalar term under this metric.

## Deployment

`content/instruments/drums/synthid-808-tom/{dsp,ui}.lisp` in eseq is the
synthid-808 port with the tom defaults, `body_harmonic` exposed, and the
profile bounds. Audition at `--pitch 743.996 --sr 44100` (the family's /8
pitch mapping; `--sr` matters, the harness defaults to 48 kHz) reproduces
`output/rung3_808tom_v1/learned.wav` to 4e-5 max abs.
