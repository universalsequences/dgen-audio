# TR-909 Kick Fitting — Session Handoff (2026-07-11)

Goal: apply the SynthID rung-3 approach (see `SPEC.md`, paper at `output/pdf/synthid_three_rung_paper.tex`)
to a real TR-909 kick at `Assets/909kick.wav`, keeping the 808 experiment fully usable.
All work is **uncommitted** on branch `train-kick808-example`.

## State at handoff

- The corrected-protocol round-3 run is complete at `output/rung3_909_v3`.
  It selected restart 1 by the independent CPU metric and finished at
  **69.00% honest improvement** (`0.112232 -> 0.034797`), below the 80% gate.
- The structured-capacity v4 continuation is complete at `output/rung3_909_v4` and
  **passes the honest gate at 80.10%** (`0.112232 -> 0.022340`). The learned WAV was
  rendered by Swift; the independent NumPy renderer differs by only `3.33e-6` max abs,
  and the 808 reference render remains bit-identical (`0.0` max abs regression).

- Two full runs exist: `output/rung3_909` (v1) and `output/rung3_909_v2` (v2), each with
  target/initial/learned/ab WAVs, compare.json, recovered_params.json, run logs at
  `output/rung3_909_run.log` / `output/rung3_909_v2_run.log`.

## Headline finding — protocol bug affecting the published 808 result

The rung-3 gate (`1 − learnedDistance/initialDistance ≥ 0.80`, scripts/compare.py) was measured
against **the winning restart's own cold start**, and restarts 1–4 use aggressive fStart/pitchDecay
bracket scaling. When an aggressive restart wins, the baseline is worse and the relative gate inflates.

- **808 (published 84.55%)**: the accepted run's `initial_params.json` (in `/tmp/synthid-rung3-complete`)
  is byte-identical to **restart-4's** cold start (capture-floor + randomized init; it won GPU-loss
  selection 1.1775 vs restart-0's 1.6010). Rescored against the deterministic restart-0 init
  (distance 0.051751 vs winner's 0.075252), the improvement is **77.53% — below the 80% gate**.
  Learned distance itself is unchanged and good (0.011627). The paper's headline number rests on a
  soft denominator; docs/paper have NOT been edited. This needs a user decision.
- **909 v2 (reported 66.80%)**: same artifact — measured against a fixed cold start, honest
  improvement is ~31%, and absolute learned distance was flat across v1/v2 (0.0362→0.0372 on
  comparable 22528-frame footing).
- **Corrected protocol (implemented, rung-3 only)**: `initial.wav`/`initial_params.json`/gate baseline
  are now always the deterministic `restartInitial(pitchFit, restartIndex: 0)`; the winner's cold
  start is preserved in `winner_initial_params.json`; restart selection now uses the independent CPU
  metric (per-restart table `restart index | GPU loss | CPU distance` printed in-run) instead of GPU
  loss, with GPU-loss fallback on python failure.
- **Remaining baseline caveat**: the reference init = spec-table midpoints + pitch fit, so widening
  bounds moves the baseline (v2's widened click bounds moved 909 midpoint clickAmp 0.25→0.6, which is
  why the tiny-run deterministic baseline scores ~0.112, not v1's ~0.054). The *absolute* learned
  MR-STFT distance is the honest cross-run signal; report it alongside the relative gate.

## 909 target measurements (from `Assets/909kick.wav`, mono-averaged)

peak −10.86 dBFS, no clipping, negligible DC/sub-20 Hz junk (no capture high-pass battle; the 30 Hz
symmetric HP in compare.py stays, harmless). Effective length 460 ms (21412 resampled frames).
Pitch fit: **fStart≈247 Hz, fEnd≈46.6 Hz, pitchDecay≈−44.2 s⁻¹** (residual RMS 1.7 Hz).
Amplitude decay **steepens**: −3.3 s⁻¹ (20–80 ms) → −12.4 s⁻¹ (150–450 ms) — motivated the
log-quadratic envelope. Click ≈400–500 Hz, decay ≈−375 s⁻¹, −10..−20 dB rel peak.
H2 −15..−32 dB, H3 −25..−32 dB (lightly-shaped sine; H3 persists at 200 ms).
Noise floor is quantization (dead).

## What was implemented (all 808-inert; verified)

1. **`--profile 909` system**: `SynthIDConfig.profile` (default "808"), `KickParamSpecs.activeProfile`
   selecting `tr808`/`tr909` tables (Params.swift), profile threading through Trainer/main/Rung3,
   `--profile` passed to `scripts/refine_rung3.py` (BOUNDS_808/BOUNDS_909 mirrors).
2. **`bodyHarmonic` (15th scalar)**: odd harmonics `sin(6πφ)/9 + sin(10πφ)/25` on the body envelope
   (Patch.swift), zero-default inert, in ampOpt group, pinned (0,0) in rung-1/2 sampling,
   `decodeIfPresent ?? 0` back-compat, mirrored in `render_reference.py`. Backward pass verified
   (relErr 1.47e-4 at bodyHarmonic=0.3).
3. **`ampCurve` (16th scalar)**: body-family envelope is now `exp(ampDecay·t + ampCurve·t²)`
   (steepening decay; sum-of-exponentials cannot steepen). tr909 bounds −60..0; tr808 pinned
   ±0.001. In decayOpt group. fdcheck relErr 2.03e-3 (needs `--fd-eps 0.1`; default 1e-2 is FD-noisy).
4. **Fit at real target length**: rung3 shrinks `config.frames` to target length rounded up to a
   multiple of the largest STFT window (21412→22528; logged "fit length: 32768 -> 22528 frames",
   `fittedFrames` in preprocessing.json). Shrink-only; 808 (longer target) unchanged. Removed a
   measured 4.66pp padded-tail artifact where training regressed chasing digital silence.
5. **PitchTrack profile** (`PitchSearchProfile.tr808/.tr909`): 909 contour 50–450 Hz, high-ridge
   threshold 420 Hz, plus a **real bug fix**: `fit(points:)` had a hardcoded 80–180 Hz fStart clamp
   (silently capped the 909's 247 Hz onset regardless of spec bounds); now `fStartRange` per profile.
6. **tr909 bounds** (Params.swift + refine_rung3.py BOUNDS_909, kept mirrored): fStart 150–400 log,
   fEnd 35–60 log, pitchDecay −80..−20 logNeg, bodyAmp .05–1, ampDecay −25..−3, ampCurve −60..0,
   clickFreq 200–1000 log, clickAmp 0–1.2, clickDecay −800..−150 logNeg, noiseCutoff 1000–18000 log,
   noiseAmp 0–0.05, noiseDecay −150..−5 logNeg, drive 1–6, outGain .1–1, bodyAsymmetry ±0.5,
   bodyHarmonic ±1.
7. **Round-3 protocol fixes** (see headline section): deterministic reference baseline,
   `winner_initial_params.json`, CPU-metric restart selection via new `scripts/score_params.py`
   (renders via render_reference + compare.py metric; parity verified: 0.037190 vs compare's
   0.0371907, 808 diff ~1.4e-6) and `Rung3IndependentScorer` (Rung3.swift), `--score-script` flag.
   Gentler 909 restart brackets: pdScales [1.00,1.15,0.85,1.00], fStartScales [1.00,1.06,0.94,1.00],
   restart-3 diversity moved to clickAmp×0.5/clickDecay×1.5 midpoints; 808 arrays untouched.
   Capture-floor restart-4 already gated off for 909 (dead noise floor).

**808 regression status**: preprocessed `target.wav` byte-identical across all changes; NumPy renderer
equivalence 4.81e-6 max abs err (gate 1e-3); rung-1/2 sampling pins new scalars to 0.

## Run history and diagnosis

| Run | Reported | Honest/absolute | Notes |
|---|---|---|---|
| v1 (`output/rung3_909`) | 29.17% | learned 0.0270 @32768 padded (0.0368 cropped) | clickAmp/drive/noiseCutoff/noiseDecay pinned at bounds; residual 59% in 10–150 ms, biggest cell 150–400 Hz |
| v2 (`output/rung3_909_v2`) | 66.80% | ~31% vs fixed baseline; learned 0.0372 @22528 | restart-3 (aggressive bracket) won → inflated denominator; click params all pinned (201 Hz/1.2/−150): click body-patching a pitch-curve undershoot (fitted f(t) 10–20% low at 20–45 ms; raw pitch fit tracks better than any bracket) |
| v3 (`output/rung3_909_v3`) | 69.00% | learned 0.034797 @22528 | corrected protocol; restart 1 won independent CPU selection (0.038771 pre-refine); confirms the remaining gap is not the v2 denominator/restart artifact |
| v4 (`output/rung3_909_v4`) | **80.10% pass** | learned 0.022340 @22528 | 909-only softsign output plus pruned multiscale harmonic correction bank; honest deterministic v3 restart-0 baseline retained |

## Round-4 feasibility audit (2026-07-11)

The v3 result was followed by independent-metric probes before changing the documented patch:

- A global differential-evolution search over all 16 documented scalars (~9,700 evaluations)
  did not beat the coordinate-refined basin (best 0.03446 versus v3's 0.03480).
- Dense joint pitch search, the measured full zero-crossing pitch contour, pitch curvature,
  body/click phase, a trainable even-harmonic decay, independent upper harmonics, a VCA attack,
  extra fixed and swept resonators, and a cubic amplitude correction each improved less than
  about 0.001 in isolation. These are not credible routes from 0.0348 to the required 0.02245.
- Alternative scalar waveshapers helped more, but still missed decisively: a softsign output plus
  ten-pass coordinate refinement reached 0.03106 (72.27%).
- The checked-in WAV is a conversion of `Assets/909kick.mp3` (320 kbps MP3), not a lossless source.
  Codec/capture texture therefore contributes to the phase-blind log-spectral floor, while the
  SPEC explicitly forbids residual tables, target-seeded tables, and learned EQ FIRs that could
  memorize that texture.

The audit established that the original scalar topology had a ~69-72% ceiling. The implemented v4
extension crosses the gate without softening the denominator or adding target-derived residuals:

- The 909 output stage uses a biased softsign (the 808 tanh path is unchanged).
- An initially overcomplete harmonic/envelope basis was backward-eliminated to 40 ordinary scalar
  coefficients spanning harmonics 2-16 and four fixed physical decay rates (0, 15, 60, 240/s).
  These are trainable Fourier/envelope terms, not waveform samples, lookup tables, FIR taps, or
  target-seeded residuals.
- Production refinement deterministically migrates old tanh checkpoints, refines the original
  scalars, then performs broad and fine harmonic coordinate passes. Starting from the ordinary v3
  winner, this path independently crossed 80%; the final v4 Swift artifact reaches 80.10%.

v2 recovered params: fStart 335.4, fEnd 47.0, pitchDecay −66.2, bodyAmp .905, ampDecay −10.31,
ampCurve −4.58, bodyAsymmetry −0.021, bodyHarmonic −0.342, clickFreq 201, clickAmp 1.2,
clickDecay −150, noiseAmp .0014, noiseCutoff 12660, noiseDecay −12.4, drive 3.92, outGain .133.
Good news: ampCurve active (steepening confirmed), noise/drive interior (v1 pins resolved).

## Ranked remaining ideas (from round-2 diagnosis)

1. Run v3 (brackets + honest selection) — restart pile-up on the pitch curve was root cause of the
   click pathology; expect click params to relax toward the true click (400–500 Hz, ~−375 s⁻¹).
2. Unhardcode the even-harmonic `exp(−17t)` decay (Patch.swift) into a trainable scalar — H2 deficit
   −6..−11 dB at 80–200 ms; est. 1–2pp only.
3. Click reshape (own pitch sweep, or split fast-click + body-patch term) — only if v3 still pins click params.
4. Pitch-sweep curvature: tried on 808 and **regressed the full run** (RUNG3_STATUS.md:161-167);
   don't retry before restart/selection is stable.
5. Honest-gate reality check: with baseline ≈0.112 (current midpoints) the gate needs learned ≤0.0224;
   with the tighter old midpoints (≈0.054) it needs ≤0.0108 (≈808's final absolute level). If the model
   can't get there, the decision is more capacity vs. adjusting/reporting the gate for this target —
   user decision, alongside how to handle the 808 77.53% provenance finding.

## Gotchas for the next agent

- `--no-refine` is missing from `parseOptions`' boolean-flag set: it eats the NEXT token as a value.
  Put `--allow-fail` (or any flag) BEFORE `--no-refine`.
- fdcheck `fStart` relErr ≈0.70 is PRE-EXISTING on both profiles (documented swept-phasor pitch-group
  discrepancy; see memory/RUNG1 notes), not a regression signal.
- Tensor/graph lifecycle rules in CLAUDE.md apply (create Tensors after `LazyGraphContext.reset()`).
- Analysis scripts (reusable) in scratchpad
  `/private/tmp/claude-501/-Users-alecresende-code-swift-dgen/906bd09c-078b-493d-9375-d903d4ec7b39/scratchpad/`:
  `analyze_kick.py`, `pitch.py`, `amp_decay.py`, `harmonics.py`, `attack.py`, `full_analysis.py`,
  `residual_grid.py` (+`_v2`), `attack_envelope_harm.py`, `harmonics_check.py` — note scratchpads are
  session-scoped and may be gone; they import `compare` from `Examples/SynthID/scripts`.

## Files touched (uncommitted; `git diff` for detail)

`Examples/SynthID/`: Config.swift, Params.swift, Patch.swift, PitchTrack.swift, Trainer.swift,
main.swift, Rung3.swift, scripts/refine_rung3.py, scripts/render_reference.py,
scripts/score_params.py (new), this file (new).
