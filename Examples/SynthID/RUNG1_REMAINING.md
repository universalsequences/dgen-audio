# SynthID Rung 1: Remaining Work

## Current status (updated)

Three root causes of the earlier "improved but stuck" plateau have been found and
addressed. The earlier framing (parameter identifiability as the primary blocker)
was partly wrong: the loss floor itself was broken.

### 1. Log-magnitude loss floor (FIXED — library)

`spectralLossFFT(useLogMagnitude: true)` used a hardcoded `eps = 1e-8` inside
`log(|X| + eps)`. For peak-normalized signals, spectrally empty bins sit at
float-noise magnitudes (~1e-7), so two signals that match to 1e-7 in the waveform
still produce O(1) per-bin log differences. Measured on seed 7 with the full
config: loss at EXACT true params was 1.32 vs init 6.72 — a floor at 20% of init,
making the `<= 0.02` gate unattainable in principle. The training plateau of
0.131 was already below the floor of what truth itself could score.

Fix: `DGenSpectralConfig.logMagnitudeEpsilon` (Sources/DGen/GradientConfig.swift),
default 1e-8 (unchanged library behavior), set to 1e-4 (~-80 dBFS) by SynthID via
`Config.spectralLogEpsilon` / `--log-eps`. After the fix, loss at truth = 0.0135
vs init 1.62 → floor ratio 0.008, and fdcheck for bodyAmp/drive/clickAmp agrees
with autograd to <0.5% (they disagreed or were meaningless before).

Diagnostic that found it: `swift run SynthID probe --target t.wav --params p.json`
compares loss(tensor,tensor), loss(synth,synth), loss(synth,tensor). With the old
eps, two IDENTICAL synth graphs scored 2.56 against each other (waveform diff
1e-5 from the transform round-trip).

### 2. gradPhasor is wrong for swept frequency (WORKED AROUND — library bug open)

`Gradients.swift` case `.phasor` uses `d(phase)/d(freq) = frameIdx / sampleRate`
(Emit+Tensor.swift `gradPhasor`). That rule is only correct for a CONSTANT
frequency input. For time-varying freq, `phase[n] = Σ_{k≤n} f[k]/sr`, so the true
per-sample gradient is a SUFFIX SUM: `∂L/∂f[k] = (1/sr)·Σ_{n≥k} gradPhase[n]`.
Measured effect on the pitch sweep: fStart autograd 12x too small, fEnd ~25% too
large, pitchDecay 5x too small (fdcheck vs central differences).

Workaround in Patch.swift: the body phase is now the closed-form integral of the
pitch envelope built from the `accum` time ramp
(`fEnd·t + (fStart-fEnd)/pd·(e^{pd·t}-1)`), and the click phase is `clickFreq·t`.
All ops in those expressions have correct gradients; fdcheck for the pitch params
now agrees in magnitude (residual ~20% is fd noise from L1 kinks — fd wobbles
across eps while autograd is stable).

Library follow-up (separate task): implement suffix-sum gradPhasor (needs a
kernel boundary so gradOutput is materialized across frames; same pattern as
`spectralLossFFTGradRead`). Note `accum`'s gradient has the same truncation
(passes gradOutput through locally).

### 3. drive is exactly redundant (RESOLVED — spec updated intentionally)

`tanh((bodyAmp·body + clickAmp·click + noiseAmp·noise)·drive)·outGain` depends
only on the products `amp·drive` and `outGain`. This is an exact algebraic
degeneracy: parameter sets with equal products render bit-identical audio, so no
loss can separate the factors. Observed empirically: best checkpoints matched all
three products to <2% while the factors were 22–28% off (bodyAmp pinned at its
bound).

SPEC.md §7.1 and Report.swift now score the products (bodyAmp·drive at 10%,
clickAmp·drive and noiseAmp·drive at 20%) and list the factors unscored. This is
the equivalence-class documentation the previous version of this file required.

### 4. noiseCutoff gradient has the WRONG SIGN (OPEN — library bug)

fdcheck at initial params, full config: fd = +0.498, autograd = -0.469. The
biquad coefficient path (cutoff → coefficients → recursion) backpropagates with
inverted sign, which explains noiseCutoff diverging from truth in every filtered
run. `--no-noise-filter` remains the honest rung-1 configuration until this is
fixed (see docs/BIQUAD_BPTT_GRADIENT_BUG.md for the related BPTT history).
noiseAmp/noiseDecay gradients THROUGH the biquad are fine (fdcheck relErr ~1e-3).

### 5. Learning rates were 10x below spec

Config defaults had amp/decay 3e-3, tone 1e-3 vs spec 3e-2/3e-2/1e-2. With grad
clip 1.0 and 400 epochs, drive (raw domain, range 1–3) could not travel more than
~0.4 from midpoint. Config defaults now match the spec values.

### 6. pitchDecay reparam + pitch coupling (FIXED)

pitchDecay was `raw` reparam in the small-LR pitch group: at scale ~20 it could
travel ~0.4% per run and froze at init, and fStart equilibrated at a compensating
wrong value (all restarts hit fStart≈124 vs true 119 and plateaued). Now
`log(-x)` like the other decays (spec §2 table updated). Two trainer additions:
cosine LR decay (`--no-lr-decay` to disable) and a pitch-only refinement stage
(`--pitch-refine-epochs`, default 200) that descends the 3-D pitch subspace from
the best checkpoint — the loss is far sharper in pitch than anything else, so a
focused final descent buys ~10x loss.

### 7. Spectral floor at -60 dB (log-eps 1e-3, not 1e-4)

Even at eps 1e-4, the loss log-amplifies inaudible bins (1e-4..1e-2 magnitude):
seed-7 recovery with every parameter within 0.05% of truth still scored ratio
0.034 — substituting the exact true pitch trio (ppm-level differences!) dropped
loss 6.7x. The gate was measuring depth into the spectral noise floor. Default
`spectralLogEpsilon` is now 1e-3 (-60 dBFS); SPEC §4 documents the floor and its
coupling to the §7.1 gate. At 1e-3 the gate retains ~13-20x headroom between
truth (ratio ~0.001) and the 0.02 threshold, so it still demands sub-percent
recovery.

### 8. Optimization additions that closed the last seeds

- Pitch-fit tail anchor: `PitchTrack.tailFEnd` measures fEnd from the long
  quasi-stationary tail (16384-sample autocorr windows); the swept-contour fit
  is confined to ±0.25 Hz of it. fEnd inits went from 2-9% error to <=0.55%.
  The old 1024-sample tracker windows could not correlate 35-60 Hz periods at
  all (it reported 299 Hz for a 53.8 Hz tail).
- Deterministic pitch-bracket restarts (Trainer.restartInitial): window
  smearing underestimates |pitchDecay| and fStart together, so restarts 0-3
  scale (pd, fStart) by (1.0,1.0), (1.45,1.10), (0.75,0.92), (1.45,1.22) on
  midpoint amps; restarts 4+ randomize amps. Never initialize a param ON its
  trainable bound (projected Adam + compensation forms a sticky local minimum
  there).
- Cross-restart recombination + clickFreq line search (rung1 in main.swift):
  restarts often solve different subspaces (one nails pitch, another the
  click). Greedily stitch subspaces across restarts and line-search clickFreq
  on a 24-point log grid, judged by the same audio loss, then fine-tune. This
  is restart-style mitigation: selection is by audio loss only, no
  target-derived scores.

## STATUS: RUNG 1 ACCEPTANCE PASSED (2026-07-07)

`swift run SynthID rung1 --seeds 1,2,3,4,5 --out <dir> --epochs 600
--restarts 5 --pitch-refine-epochs 300 --frames 32768 --no-noise-filter`
(these values are now the config defaults) → exit 0, 4/5 seeds (spec gate: >=4).

| seed | pass | loss ratio | param failures | worst scored param |
| ---- | ---- | ---------- | -------------- | ------------------ |
| 1 | yes | 0.0173 | none | clickAmp·drive 0.03% |
| 2 | no  | 0.0241 | none (!) | fStart 0.61% |
| 3 | yes | 0.0062 | none | clickAmp·drive 0.22% |
| 4 | yes | 0.0153 | none | outGain 0.04% |
| 5 | yes | 0.0117 | none | fStart 2.14% |

Seed 2 (the allowed failure) recovers every parameter within tolerance and
misses only the loss-ratio gate by 20% (0.0241 vs 0.02).

## Remaining / follow-ups

1. Library tasks to file: suffix-sum gradPhasor (+accum has the same
   truncation), biquad cutoff gradient sign. Both have minimal repros via
   `SynthID train --fdcheck`.
2. Rung 2's numpy renderer must mirror the closed-form phase convention in
   Patch.swift and omit the noise filter until the biquad gradient is fixed.
3. Optional: chase seed 2's last 20% of loss ratio (pure polish — all params
   already within tolerance).

## Do not paper over

- The product-scoring change is an exact equivalence class, documented in SPEC §7.1
  — not a tolerance loosening. Do not extend it to any non-degenerate parameter.
- Do not select by a target-derived parameter score; training/selection stays on
  the audio loss.
- Do not move to rung 2 until rung 1 passes the acceptance command (now including
  `--no-noise-filter` until the biquad cutoff gradient is fixed; rung 2's numpy
  renderer must then also omit the filter or the fix must land first).
