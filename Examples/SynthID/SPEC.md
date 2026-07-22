# SynthID — Synthesizer Parameter Recovery via Autograd

## Purpose

Produce the first unambiguous, end-to-end demonstration that DGen's autograd can do
something useful: **recover the parameters of a synthesizer patch from audio alone**
(analysis-by-synthesis / system identification).

This spec exists because two prior attempts failed for task-framing reasons, not
autograd reasons:

- **TrainKick808** targeted an E-MU sample that is out of the model class, and used
  phase-exact losses (waveform MSE, zero-crossing gates) that a parametric synth
  cannot win. Only sample-like residual tables ever crossed the goal — see
  `Examples/TrainKick808/EXPERIMENTS.md`. The `fullres4` and residual-oracle runs
  already proved the gradient/optimizer/checkpoint chain works.
- **DDSPE2E** is a research-scale training problem (data, capacity, wall-clock);
  failure there is ambiguous between "library bug" and "undertrained model".

SynthID replaces both with a ladder of tasks where each rung has an objective
pass/fail criterion and the target is **known to be inside the model class**.

## Non-Goals

- Do NOT match arbitrary sampled/layered drums (that was TrainKick808's mistake).
- Do NOT use waveform MSE, slope loss, or zero-crossing counts as training losses
  or gates. Phase-blind losses only.
- Do NOT learn f0/pitch contours by gradient descent from scratch (known non-convex
  pathology; DDSP itself conditions on an externally extracted f0). Pitch is
  extracted deterministically and at most *refined* by gradient.
- Do NOT add residual tables, target-seeded tables, learned EQ FIRs, or any layer
  that reads the target waveform into a parameter. These are oracle cheats and
  invalidated the previous experiment's honesty. The ONLY target-derived inputs
  allowed are: (a) the extracted pitch contour, (b) scalar normalization/level, and
  (c) parameter *initialization ranges* (not values copied from the target signal).

## The Ladder

| Rung | Target audio | Proves | Pass criterion |
| --- | --- | --- | --- |
| 1 | Rendered by DGen itself with hidden params | Gradients are correct through phasors, envelopes, tanh, noise, filters | Parameter recovery table (see §7.1) |
| 2 | Rendered by a standalone numpy script implementing the same math | Not a self-consistency artifact | Same table + spectral distance threshold |
| 3 | A real TR-808 kick recording (analog bridged-T circuit ≈ the model class) | Useful on real audio | A/B audio + spectrogram overlay + MR-STFT distance (see §7.3) |

Implement in order. Rung 1 must fully pass before starting rung 2, etc. Each rung is
a subcommand of the same executable.

## 1. Deliverable Layout

New executable target `SynthID`, wired into `Package.swift` like the existing
examples.

```
Examples/SynthID/
  SPEC.md              (this file)
  main.swift           (CLI dispatch)
  Config.swift         (all knobs + JSON round-trip; embedded in checkpoints)
  Patch.swift          (the KickVoice synth graph, shared by render + train)
  Params.swift         (parameter set: names, bounds, reparameterization, true/recovered)
  PitchTrack.swift     (CPU pitch-contour extraction from a target wav)
  Losses.swift         (multi-resolution log-mag STFT loss builder)
  Trainer.swift        (staged training loop, checkpointing, logging)
  Report.swift         (recovery table, JSON report writer)
  scripts/
    render_reference.py   (rung 2: numpy renderer, same math as Patch.swift)
    compare.py            (spectrogram overlay PNG + MR-STFT distance between two wavs)
  targets/               (checked-in or downloaded target wavs for rung 3)
```

CLI subcommands:

```
swift run SynthID render   --params <json> --out <wav> [--frames N]      # render a patch
swift run SynthID train    --target <wav> --out <dir> [--rung 1|2|3] [--seed N] [flags]
swift run SynthID rung1    --seed <N> --out <dir>     # end-to-end: sample hidden params, render, recover, report
swift run SynthID rung2    --out <dir> [--seeds 1,2,3,4,5]
swift run SynthID rung2    --target <wav-from-numpy> --params <json> --out <dir>  # single external target
swift run SynthID rung3    --target <real-808-wav> --out <dir>
```

`rung1` is fully self-contained: it samples ground-truth parameters from documented
ranges (seeded RNG), renders the target with `realize(frames:)` +
`exportToWav` (`Sources/DGenLazy/AudioFile.swift`), resets the graph, rebuilds the
patch with *different* initial parameter values, trains, and writes the recovery
report. It must be runnable in CI-like fashion: one command, exit code 0 iff the
pass criteria in §7.1 hold.

## 2. The Patch: `KickVoice`

A single synth topology used for all three rungs, modeled on the TR-808 kick voice
(bridged-T resonator ≈ exponentially decaying, pitch-swept sine + click transient +
tone-shaped noise). The base voice uses a compact scalar parameter set. The 909
profile may additionally use a pruned bank of scalar Fourier/envelope corrections
at documented harmonic numbers and fixed decay rates. No tensors of hundreds of
points; expressiveness comes from structure, not waveform or residual tables.

Signal graph (all built with DGenLazy `Signal` ops; see `Examples/TrainKick808/main.swift`
for API patterns):

```
pitchEnv(t)  = f_end + (f_start - f_end) * exp(pitchDecay * t)      # Hz, per-sample
bodyPhase    = statefulPhasor(pitchEnv)                              # or accum-based phasor
body         = sin(2π * bodyPhase) * exp(ampDecay * t) * bodyAmp
click        = click() -> excite a short decaying sine burst:
               sin(2π * phasor(clickFreq)) * exp(clickDecay * t) * clickAmp
noiseBurst   = noise() -> biquad(cutoff=noiseCutoff, mode=LP)        # Signal.biquad, mode 0
               * exp(noiseDecay * t) * noiseAmp
mix          = body + click + noiseBurst
out          = tanh(mix * drive) * outGain
```

Notes:

- `t` is seconds; build `exp(k * t)` as an envelope from `Signal.accum` (a
  per-sample time ramp) — do NOT use per-epoch host-side envelopes. Everything must
  be differentiable inside one DGen graph.
- Use `Signal.param(value, min:max:)` for every scalar (bounds already give clamping;
  see `Sources/DGenLazy/Signal.swift:137`).
- `Signal.click()` requires the fixes noted in project memory (cell init + scalar
  block placement) — these are already in the library; just use it.
- If `biquad` gradient support turns out to be missing/unstable, fall back to a
  one-pole lowpass built from `history()` or drop the noise filter for rung 1 and
  add it in a follow-up. **Verify biquad has a backward rule before relying on it**
  (check `Sources/DGen/Gradients.swift`); if not, that is a scoped library task to
  surface, not silently work around.

### Parameter set (canonical names, used in reports)

| Name | Unit | True-value sampling range (rung 1) | Reparam |
| --- | --- | --- | --- |
| fStart | Hz | 80 – 180 | log |
| fEnd | Hz | 35 – 60 | log |
| pitchDecay | 1/s | -80 – -15 | log(-x) |
| bodyAmp | lin | 0.5 – 1.0 | raw |
| ampDecay | 1/s | -12 – -3 | raw |
| clickFreq | Hz | 600 – 3000 | log |
| clickAmp | lin | 0.05 – 0.6 | raw |
| clickDecay | 1/s | -900 – -200 | log(-x) |
| noiseCutoff | Hz | 1000 – 8000 | log |
| noiseAmp | lin | 0.0 – 0.3 | raw |
| noiseDecay | 1/s | -400 – -60 | log(-x) |
| drive | lin | 1.0 – 3.0 | raw |
| outGain | lin | 0.4 – 1.0 | raw |
| bodyAsymmetry | lin | 0 (Rungs 1–2; learned only for real targets) | raw |

Rung 3 may widen the optimizer-only click/noise bounds to cover a real capture
(`clickAmp ≤ 1.5`, `clickDecay ≥ -1600/s`, `noiseCutoff ≤ 20 kHz`, and
`noiseDecay` as slow as `-0.001/s`). `PatchValues.sample` and the generic midpoint retain
the table's original Rung 1–2 distributions, so the synthetic acceptance tasks
do not silently change.

Reparameterization: optimize in the transformed domain (e.g., the trainable scalar
is `log(fStart)`), so Adam steps are scale-appropriate. The TrainKick808 log
documents the Adam scale-mismatch failure; also use **per-group Adam optimizers**
(freq group, amp group, decay group) with separate learning rates, exactly like the
multi-optimizer pattern at `Examples/TrainKick808/main.swift:1585-1641`.

Sample rate: 44100. Duration: 0.75 s (frames = 33075, round to 32768 if the FFT
plumbing prefers it). Both are `Config` fields.

## 3. Pitch Handling (critical)

Gradient descent on oscillator frequency under spectral loss is non-convex and was
the single biggest sinkhole in TrainKick808. Rules:

1. `PitchTrack.swift` extracts a per-frame f0 contour from the target on the CPU
   (autocorrelation or YIN over 1024-sample windows, hop 256, search band
   30–300 Hz, parabolic peak interpolation). Pure Swift, no graph involvement.
2. Fit `(fStart, fEnd, pitchDecay)` to that contour with a **closed-form/CPU
   least-squares fit** (it's a 3-parameter exponential — a simple Gauss-Newton or
   grid+refine on the CPU is fine). This becomes the *initialization*.
3. During graph training, the pitch parameters are trainable but start at the
   fitted values and use a learning rate 10–100× smaller than amp/decay groups.
   Provide `--freeze-pitch` to lock them entirely; the rung-1 pass criteria must be
   achievable with pitch trainable, but if instability appears, freezing is an
   acceptable documented fallback for rung 3 only.
4. Everything else (amps, decays, cutoff, drive, gain) is learned by gradient from
   generic initialization (midpoint of range) — no target-derived values.

## 4. Loss

Multi-resolution log-magnitude STFT only, via the existing
`spectralLossFFT(_:_:windowSize:useHannWindow:useLogMagnitude:lossMode:hop:normalize:)`
(`Sources/DGenLazy/Functions.swift:880`):

- windows: `[256, 512, 1024, 2048]` at 44.1 kHz (2048 gives 21.5 Hz resolution —
  required to separate fStart from fEnd; see CLAUDE.md frequency-resolution table)
- per window: `useLogMagnitude: true, lossMode: .l1, hop: windowSize/4, normalize: true`
- total loss = sum over windows (equal weights to start; a `--window-weights` flag
  for experimentation)
- optionally add a linear-magnitude term (`useLogMagnitude: false`) at weight 0.1 —
  DDSP-paper parity — behind a flag, default on.
- log-magnitude epsilon (spectral floor): `log(|X| + 1e-3)`, i.e. −60 dBFS for
  peak-normalized signals (`--log-eps`). The floor defines what the loss gate in
  §7.1 measures: audible spectral structure. Smaller epsilons make the loss
  log-amplify inaudible noise-floor bins — at 1e-8 the loss at EXACT parameter
  match is ~20% of init; even at 1e-4, parts-per-million pitch errors carry most
  of the residual. Do not tighten the gate and loosen the floor independently.

Explicitly banned: waveform MSE, first-difference/slope losses, zero-crossing
metrics, band-energy "gates". Time-domain metrics may appear in *reports* for
curiosity but must never influence training or model selection.

Target signal: load the wav on CPU, normalize peak to 0.9, feed as a non-trainable
input (same mechanism TrainKick808 uses for its target). Same normalization applied
to rendered output before comparison in reports.

## 5. Training Loop

Standard DGenLazy tinygrad-style loop (see CLAUDE.md "DGenLazy Training Loop"):
rebuild the loss graph each epoch, `loss.backward(frames:)`, `opt.step()`,
`opt.zeroGrad()`, capture metrics before `zeroGrad()`.

Hard-won project rules that MUST be followed:

- Create every `Tensor`/`Signal` AFTER `LazyGraphContext.reset()` (stale-nodeId
  aliasing; see CLAUDE.md "Testing: Tensor/Signal Creation Order Matters").
- Do not interleave `realize()` renders of candidate snapshots with `backward()`
  in the same process — TrainKick808 found this unsafe. Render candidates by
  re-invoking the executable (`SynthID render --params <checkpoint>`), or render
  only before training starts / after it ends.
- Checkpoints (`checkpoint.json`) must embed the full resolved `Config` so a
  checkpoint is self-describing (reproducibility failure documented in
  EXPERIMENTS.md). Write one every N epochs and at best-loss.
- Zero-amplitude paths get zero gradients ("dead start" trap in EXPERIMENTS.md):
  every amp parameter initializes strictly > 0 (use range midpoints).

Schedule (defaults; all flags):

- epochs: 400, Adam per group — pitch lr 1e-3 (transformed domain), amp lr 3e-2,
  decay lr 3e-2, tone (cutoff/drive/gain) lr 1e-2. Grad clip 1.0.
- Selection: lowest training loss (the loss IS the objective here — no external
  evaluator needed on rungs 1–2, since parameter recovery is the metric).
- Log every 10 epochs: loss, current parameter values in natural units.
- Expected runtime: TrainKick808 ran ~12–50 ms/epoch at 4096 frames; at 32768
  frames with 4 spectral windows expect a few hundred ms/epoch — a full run should
  stay under ~5 minutes on Metal. If it's 10× that, something is wrong; profile
  before proceeding.

Multi-restart: `rung1` runs `--restarts 3` (different init seeds, same target) and
takes the best final loss. Spectral loss on decaying sweeps can still have local
minima; restarts are the honest mitigation (not target-seeded inits).

## 6. Rung Details

### Rung 1 — Self-inversion

`swift run SynthID rung1 --seed 7 --out /tmp/synthid-rung1`

1. Sample true params uniformly from the table in §2 (seeded).
2. Render target wav with DGen; save `target.wav` + `true_params.json`.
3. `LazyGraphContext.reset()`; rebuild patch with midpoint inits (pitch group from
   the CPU pitch fit of `target.wav` — yes, even on rung 1, to exercise the same
   path as rung 3).
4. Train per §5, write `learned.wav`, `recovered_params.json`, `report.json`,
   `report.md` (human-readable table: name, true, recovered, abs/rel error, pass).
5. Run over 5 seeds (`--seeds 1,2,3,4,5`) in the acceptance run.

### Rung 2 — External renderer

`scripts/render_reference.py` implements the §2 math independently in NumPy. It
uses the same pre-update float32 `accum(1/sampleRate)` time convention as
`Patch.swift`, evaluates the closed-form body phase and click phase at that time,
then reproduces DGen's deterministic noise, low-pass biquad, tanh placement, and
output gain.

`rung2` samples each seed's hidden parameters, invokes the Python renderer, and
hard-gates training on max absolute sample error `< 1e-3` against an independent
DGen render of the same parameters. `--verify-only` runs just this inexpensive
gate. With no `--seed` or `--seeds`, the command runs the five-seed acceptance set;
the explicit `--target/--params` form remains available for one externally
prepared target. Targets are normalized only after equivalence is established,
and the corresponding truth `outGain` is adjusted exactly as in rung 1.

After equivalence passes, rung 2 runs the same best-of-restarts, cross-restart
recombination, click-frequency search, reporting, and artifact generation as
rung 1 against the NumPy-rendered target. The Python renderer writes IEEE
float32 WAV; PCM16 is not acceptable here because quantization noise creates an
artificial log-STFT floor in otherwise empty bins.

### Rung 3 — Real TR-808

Targets: 2–4 real TR-808 kick recordings (user supplies wavs into
`Examples/SynthID/targets/`; the implementer should stub the directory and document
the expected format: mono wav, ≥ 44.1 kHz, trimmed to onset, < 1 s). No ground-truth
params exist, so the criteria change (§7.3). Pitch init comes from `PitchTrack`.
Expect to need the noise/click paths; the 808 click is where model mismatch will
show up — that is fine and should be reported honestly, not patched with tables.

## 7. Acceptance Criteria

### 7.1 Rung 1 (hard gate — this is the autograd proof)

For at least 4 of 5 seeds, best-of-3-restarts recovery must satisfy ALL of:

- fStart, fEnd: within 3% relative error
- pitchDecay, ampDecay: within 10% relative error
- outGain: within 10% relative
- bodyAmp·drive: within 10% relative; clickAmp·drive, noiseAmp·drive: within 20%
  relative. The individual factors bodyAmp, clickAmp, noiseAmp, drive are NOT
  scored: `tanh((bodyAmp·body + clickAmp·click + noiseAmp·noise)·drive)·outGain`
  depends only on these products and outGain, so parameter sets with equal
  products render bit-identical audio — the factors are unidentifiable in
  principle (exact algebraic degeneracy, not a loss limitation). Reports list
  the factors unscored for transparency.
- clickFreq, noiseCutoff: within 10% relative
- clickDecay, noiseDecay: within 20% relative
- final MR-STFT loss ≤ 0.02 × loss-at-init (two orders of magnitude reduction)

If a specific parameter systematically fails, that is a *finding about the
gradient path for that op* — investigate (finite-difference check that parameter:
perturb ±ε on CPU, re-render, compare loss delta vs autograd grad) before moving on.
Include a `--fdcheck <paramName>` mode in the trainer that does exactly this; it is
the debugging tool for the whole example.

### 7.2 Rung 2

Same parameter thresholds as 7.1 on ≥ 3 of 5 seeds, plus the
renderer-equivalence assertion in §6. The two-renderer loss has a small nonzero
floor even at the hidden truth because NumPy and Metal transcendental functions
are not bit-identical. Rung 2 therefore applies the same 0.02 reduction gate to
the *reducible* loss:

`max(0, finalLoss - rendererFloor) / max(initLoss - rendererFloor, 1e-12) ≤ 0.02`

`rendererFloor` is the DGen training loss evaluated at the hidden parameters and
is written to the report alongside the unadjusted loss ratio. It is never used
by optimization, initialization, checkpoint selection, restart selection, or
parameter scoring. The command exits nonzero when equivalence fails or fewer
than 3 of the default 5 seeds pass.

### 7.3 Rung 3

No ground truth, so:

- MR-STFT distance (computed by `scripts/compare.py`, independent of the training
  code) between `learned.wav` and target improves ≥ 80% from init.
- For real captures, the comparator applies the declared zero-phase 30 Hz
  capture high-pass equally to target, initialization, and learned audio and
  records the cutoff in `compare.json`. This excludes capture-chain baseline
  motion below the modeled 808 body without changing the stored WAV artifacts.
- A deterministic post-training refinement may search the documented scalar
  parameters against this declared metric. It must write pre/post audit
  artifacts, must rerender the final patch through DGen, and remains subject to
  every target-data prohibition in the Non-Goals section.
- Spectrogram overlay PNG (compare.py) shows matching pitch sweep and decay
  envelope by inspection.
- A/B listening artifact: write `ab.wav` = 1 s target, 0.5 s silence, 1 s learned.
- Report documents residual mismatch honestly (expected: click/beater texture).

### Deliverable artifacts per run

`<out>/target.wav`, `learned.wav`, `ab.wav`, `checkpoint.json` (with embedded
config), `report.json`, `report.md`, `loss_curve.csv`, and for rung 3
`compare.png`. `report.md` is the thing the user shows people — make the recovery
table clean.

## 8. Known Pitfalls (read before coding)

All from this repo's history — see CLAUDE.md and `Examples/TrainKick808/EXPERIMENTS.md`:

1. Tensor/Signal creation before `LazyGraphContext.reset()` → silent wrong results.
2. Spectral blocks inheriting tensor ThreadCountScale → non-deterministic loss
   (fixed in library, but if loss is non-deterministic across identical runs, this
   class of bug is the first suspect — check with two identical renders).
3. Metrics must be captured before `zeroGrad()`.
4. Zero-initialized amp params = dead gradient paths.
5. `realize()` interleaved with `backward()` in one process is unsafe.
6. Frequency resolution: window 2048 @ 44.1 kHz = 21.5 Hz. fEnd values 35–60 Hz
   differ by ~1 bin at 2048 — this is why the window list includes 2048 and why
   fStart/fEnd get log-reparam and small LRs. If recovery of fEnd is noisy,
   consider adding window 4096 before concluding the gradient is wrong.
7. Adam scale mismatch across param groups → per-group optimizers, log reparam.

## 9. Milestones (suggested implementation order)

1. `Patch.swift` + `render` subcommand; render a hand-picked param set, listen
   (use the `/listen` skill), sanity-check it sounds like an 808 kick.
2. `PitchTrack.swift` + unit-style check against a rendered known sweep.
3. `Losses.swift` + `Trainer.swift`; overfit ONE parameter (ampDecay) with all
   others frozen at true values — the minimal gradient smoke test.
4. `--fdcheck` finite-difference harness; run it for every parameter once.
5. Full rung 1 with all params trainable + restarts + report.
6. Rung 2 numpy renderer + equivalence check + recovery.
7. Rung 3 on real 808 wavs + compare.py artifacts.

Each milestone is independently demonstrable; do not skip 3 and 4 — they are cheap
and they are precisely what previous attempts skipped.
