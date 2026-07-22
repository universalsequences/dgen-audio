# SynthID New-Target Playbook

How to point SynthID at a new sample and come out the other side with a
compact synth patch (a `Patch.swift` voice topology) plus recovered scalar
parameters that recreate it. Distilled from the TR-808 → TR-909 port
(`HANDOFF_909.md`), which is the reference example of this process end to
end. Written with bass synths as the next target; a bass-specific section is
at the end.

Budget expectation: the 909 took one long day of analysis + implementation
plus several multi-hour training runs. Assume the same or more for a new
family. Runtime math is in `GPU_REFINEMENT.md` — read it before launching
full runs.

## The contract (what keeps results honest)

Everything below serves one claim: *"this patch was recovered by
optimization, not memorized from the target."* The rules that protect it:

- Parameters are ordinary scalars with documented bounds. **No waveform
  samples, residual tables, target-seeded lookup tables, or learned FIR
  taps** ever become patch parameters (SPEC.md). Trainable Fourier/envelope
  coefficients (the 909's harmonic bank) are fine; a table sampled from the
  target is not.
- The acceptance gate is the **independent CPU metric** (`scripts/compare.py`
  MR-STFT, ≥80% improvement), measured against a **deterministic baseline**:
  spec-table midpoints + the CPU pitch fit (`restartInitial(pitchFit,
  restartIndex: 0)`), never a winning restart's own cold start. That
  baseline moves when you change spec bounds (midpoints move), so **always
  report the absolute learned MR-STFT distance alongside the relative gate**
  — it's the only honest cross-run signal.
- The Swift/DGen renderer and the NumPy reference renderer
  (`scripts/render_reference.py`) must agree to ~1e-3 max abs (the pipeline
  checks this). Every new voice feature gets implemented **twice**.

## Phase 0 — Know what the machine does

One rung-3 run = for each restart, DGen autograd trains the voice's scalars
against a multi-resolution STFT loss on GPU; restarts are scored by the
independent CPU metric; a winner is (optionally) stitched with donor
subspaces from other restarts and fine-tuned; then
`scripts/refine_rung3.py` does gradient-free coordinate descent directly on
the acceptance metric. Artifacts land in `--out`: `recovered_params.json`,
`learned.wav`, `ab.wav`, `compare.json`, per-restart checkpoints and loss
curves.

## Phase 1 — Measure the target before touching any code

This phase is cheap and determines everything. For the 909 it predicted the
pitch fit, exposed the steepening envelope (which motivated `ampCurve`), and
showed the noise floor was dead (which gated off a wasted restart). Produce
a written measurement section like `HANDOFF_909.md`'s before proceeding.

Minimum measurements (python + numpy/matplotlib, throwaway scripts are fine
— but check them into `scripts/analysis/` this time; the 909's died with a
session scratchpad):

1. **Housekeeping**: peak dBFS, clipping, DC offset, sub-20 Hz junk, mono vs
   stereo, source provenance. Lossy sources (the 909 wav was a 320 kbps MP3
   conversion) put codec texture into the loss floor that the rules forbid
   you to memorize — prefer lossless captures, and record the provenance.
2. **Effective length**: where does the sound actually end? rung 3 fits at
   real target length (shrink-only, rounded up to the largest STFT window).
   Frames × epochs is your runtime bill.
3. **Pitch contour**: zero-crossing or autocorrelation f(t). Is it a sweep
   (kick), a steady note (bass), stepped, or vibrato-modulated? Fit the
   candidate parametric form and look at the residual in Hz, not just RMS.
4. **Amplitude envelope**: log-amplitude vs t, measured over multiple
   windows. Is it one exponential? Does it steepen (909: −3.3 → −12.4 s⁻¹,
   sum-of-exponentials cannot do that)? Is there a sustain plateau (bass:
   almost certainly — see the bass section)?
5. **Harmonic structure over time**: per-harmonic level tracks (H1..H16) in
   dB relative to the fundamental, early vs late. This tells you the
   oscillator recipe and which harmonics persist vs are attack-localized.
6. **Attack**: first 5–30 ms in isolation — click/transient frequency,
   decay rate, level relative to peak.
7. **Noise floor**: broadband energy after the pitched content decays. Dead
   (909) vs real (analog capture) changes the noise-source bounds and
   whether a capture-floor restart is worth having.

## Phase 2 — Design the voice topology

Decide the smallest set of scalar-parameterized components that the Phase 1
measurements say you need. The kick voice is: closed-form swept-phase body +
even/odd harmonic terms + click + filtered noise burst + waveshaper. A new
family gets its own topology — do not force a bass through the kick voice.

Design rules learned the hard way:

- **Every capacity addition is zero-default and inert** (term × parameter
  that defaults to 0), so existing profiles are numerically unchanged and
  regressions are detectable (the 808's render stayed bit-identical through
  the entire 909 effort).
- **Closed-form phase over stateful phasors** where possible: the swept
  phase is `∫f dt` in closed form (Patch.swift:13-21), which avoids
  phase-wrap discontinuities and a known `gradPhasor` swept-frequency bug.
- **Reparameterize for gradients**: decay rates live in log space
  (`logneg`), frequencies in log space. Never let an initializer sit ON a
  trainable bound — projected Adam sticks there (Trainer.swift:369-371).
- **Waveshaper choice matters more than it looks**: the 909 gained ~3pp
  switching tanh → biased softsign. Audition the output stage against the
  measured knee before burning runs.
- **Expect a capacity ceiling and plan the escape hatch**: the 909's
  16-scalar topology audited out at a 69–72% ceiling; the gate was crossed
  with a backward-eliminated bank of 40 harmonic/envelope coefficients
  (harmonics 2–16 × four fixed decay rates). The pattern generalizes: build
  an *overcomplete but rule-legal* basis (ordinary scalars on fixed,
  physically-motivated shapes), train it, prune to the terms that matter.
  But see the runtime warning below before making a bank a default.

## Phase 3 — Implementation checklist (all touch points)

Add a new profile (e.g. `"bass1"`) rather than mutating an existing one.
`KickParamSpecs.activeProfile` is set once per command via
`SynthIDConfig.applyRuntime()`.

| # | What | Where |
|---|------|-------|
| 1 | Parameter spec table: name, min, max, transform (`log`/`logneg`/linear), midpoint | `Params.swift` (`KickParamSpecs`, mirror `tr909` pattern) |
| 2 | Profile dispatch | `Params.swift` `activeProfile`, `Config.swift` `profile` + `applyCLI` |
| 3 | Voice topology (DGen side) | `Patch.swift` `KickVoice.build` (or a new voice enum) |
| 4 | Voice topology (NumPy mirror, must match to ~1e-3) | `scripts/render_reference.py` |
| 5 | Metric bounds mirror | `scripts/refine_rung3.py` `BOUNDS_*` |
| 6 | Independent scorer profile plumbing | `scripts/score_params.py`, `Rung3.swift` scorer/refiner calls |
| 7 | Pitch extraction profile: contour range, ridge threshold, fStart range | `PitchTrack.swift` `PitchSearchProfile` |
| 8 | Restart bracket scales for the new profile | `Trainer.swift` `restartInitial` |
| 9 | Optimizer group assignment + LRs for new params | `Trainer.swift` `train()` (pitch/amp/decay/noise/tone groups) |
| 10 | Comparator high-pass / windows if the family needs different ones | `scripts/compare.py`, `Config.swift` `spectralWindows` |

Validation gates before any full run (in order):

1. **Renderer equivalence**: DGen vs `render_reference.py` on midpoint
   params and on a few random param draws (the pipeline's
   `verifyRendererEquivalence` gate is 1e-3; the 909 landed at 3.3e-6).
   Confirm the *existing* profiles still render bit-identically.
2. **fdcheck every new parameter**: `SynthID rung3 --target ... --fdcheck
   <name>` (or `train --fdcheck`). relErr ≤ ~1e-2 is healthy; some params
   need `--fd-eps 0.1` (FD noise, e.g. ampCurve at 2e-3). Known
   pre-existing exception: `fStart` relErr ≈ 0.7 on all profiles (swept-
   phasor pitch-group discrepancy) — not a regression signal.
3. **Rung 1/2 on synthetic self-targets**: render a patch from known params
   with the new voice, recover it. If SynthID can't recover its own
   render, a real target is hopeless. Pin new zero-default scalars to 0 in
   rung-1/2 sampling as the existing ones are.
4. **Pitch fit sanity**: run `--prepare-only`, inspect `pitch_fit.json` and
   `pitch_points.json` against your Phase 1 measurement. The 909 needed a
   PitchTrack bug fix (hardcoded 80–180 Hz clamp) before anything worked.

## Phase 4 — Run protocol

```bash
swift build -c release --product SynthID   # debug costs +38 s/restart and 6× redundant pitch fits

.build/release/SynthID rung3 \
  --target Assets/<sample>.wav \
  --out output/rung3_<name>_v1 \
  --profile <profile> \
  2>&1 | tee output/rung3_<name>_v1_run.log
```

- Flag-order gotcha: `--no-refine` is missing from the boolean-flag set and
  eats the next token — write `--no-refine true`, and put `--allow-fail`
  before it.
- First runs: use `--allow-fail` (the gate will fail; you want the
  artifacts), `--restarts 2 --epochs 300` for a cheap shakedown before the
  full config.
- Number your runs (`_v1`, `_v2`, …), keep the tee'd log, and record in a
  running STATUS/HANDOFF doc: reported %, **absolute learned distance**,
  frame count, which restart won and why, which params pinned at bounds.
- **Params pinned at a bound are a diagnosis, not a result** (909 v1:
  clickAmp/drive/noiseCutoff pinned ⇒ the model was patching a pitch-curve
  undershoot with the click). Widen the bound only if the measurement
  supports it; otherwise find what the pinned param is compensating for.
- Runtime realism (see `GPU_REFINEMENT.md`): ~1.04 s/epoch at 22,528 frames
  with the 909's 40-term bank, ~0.27 s/epoch without it; scale linearly
  with frames. A default 5-restart run with a big bank is ~90+ minutes;
  it's fine to let it run, but do shakedown runs first so the long runs are
  spent on a topology that already basically works. If you adopt a
  correction bank for the new family, strongly consider training it
  winner-only (restart search with the bank frozen out of the graph).

## Phase 5 — Iterate on residuals

After each run, before adding capacity:

1. Grid the residual: which time × frequency cells hold the remaining
   distance (the 909 used a `residual_grid.py`; 59% of v1's residual was in
   10–150 ms × 150–400 Hz — that's a pitch-curve problem, not a capacity
   problem).
2. A/B listen (`ab.wav`) and overlay spectrograms (`compare.png`).
3. Probe candidate fixes against the **independent metric in NumPy first**
   (cheap, seconds per idea) before implementing them in DGen. The 909
   round-4 audit killed ~10 plausible ideas (<0.001 each) this way and
   found the two that mattered (softsign, harmonic bank).
4. Only then implement, fdcheck, and rerun.

Expect the loop: run → diagnose pinned/compensating params → fix root cause
(often initialization or pitch fit, not capacity) → run → capacity audit →
add pruned zero-default capacity → final long run.

## Bass synths — specific guidance

Bass is a different animal from a kick in ways that hit almost every phase:

**Envelope, not decay.** A kick is one decaying event; a bass note is
ADSR-shaped with a sustain plateau. `exp(ampDecay·t + ampCurve·t²)` cannot
hold a sustain level. Plan an envelope with attack/decay/sustain/release
scalars from the start (e.g. `sustain + (1−sustain)·exp(decay·t)` shapes
with a short attack ramp and a release triggered at note-off time — note-off
time itself can be a documented scalar). Measure the real envelope first;
many sampled bass patches are decay-only and simpler than feared.

**Pitch is (nearly) steady — and low.** The pitch trio (fStart/fEnd/
pitchDecay) may collapse to one fundamental scalar + optional glide/vibrato.
Two consequences:

- **Spectral resolution** (CLAUDE.md): resolution = sampleRate/windowSize.
  At 44.1 kHz, a 2048 window gives 21.5 Hz bins — useless for placing a
  41 Hz vs 44 Hz fundamental. Add a 4096 (maybe 8192) window to
  `spectralWindows` for bass profiles, and expect the loss to be extremely
  sharp in the fundamental (the kick already showed sub-1% pitch error
  carrying ~90% of residual loss; a steady bass makes this sharper). A good
  CPU pitch fit as initialization matters more than anywhere else.
- **The comparator's 30 Hz high-pass** (`capture_highpass`) sits inside
  bass fundamental territory. For a 41 Hz bass it attenuates real signal the
  model is supposed to match. Decide deliberately per profile (lower it to
  ~20 Hz or make it profile-dependent in `compare.py` + `refine_rung3.py` +
  `score_params.py`, keeping all three mirrored) — and freeze that choice
  before the first gated run, since it redefines the metric.

**Oscillator waveform is the core capacity question.** Saw/square basses
carry dozens of harmonics with a characteristic rolloff (1/n saw, odd-only
1/n square). Options, in increasing generality: (a) parametric classic
waveforms (shape-blend scalar between saw/square/pulse with pulse-width);
(b) a first-class trainable harmonic-amplitude bank sin/cos(k·2πφ) × the
shared envelope — the 909's correction bank promoted to the main oscillator,
which is rule-legal (ordinary scalars) and probably the right default for
bass; per-harmonic detune stays forbidden territory-adjacent — keep
harmonics locked to integer multiples of the fundamental. Watch runtime: a
32-harmonic bank ≈ the 909's 40-term cost (~1 s/epoch at 22k frames).

**Filter movement is often the identity of the patch.** A synth bass's
character is usually the lowpass envelope sweep (cutoff + env amount +
resonance). DGen has a trainable biquad (used for noiseCutoff). The rung-1
biquad cutoff gradient sign issue is **fixed** (commit bbb3769, "Fix SynthID
temporal and biquad gradients" — FDCHECK_FINDING.md confirms the filtered
cutoff check now matches finite difference). Still unexercised: time-varying
cutoff (cutoff driven by an envelope signal instead of a constant) — fdcheck
it explicitly before trusting a filter-EG fit. A resonant sweep may alternatively be approximated in the
harmonic-bank domain (per-harmonic decay = brightness decay), which sidesteps
the filter entirely and matches the "fixed decay rates × trainable
coefficients" pattern that already passed review.

**Detune/unison and chorus**: two detuned oscillators produce beating that a
phase-blind log-magnitude loss sees as amplitude modulation per harmonic.
One detune scalar (symmetric ± cents pair) is rule-legal and cheap; add it
zero-default only if the measurement shows beating.

**Length and cost**: a 1–2 s bass note at 44.1 kHz is 44–88k frames — 2–4×
the kick's per-epoch cost on top of any harmonic bank (roughly: 1 s note +
32-harmonic bank ≈ 2 s/epoch ⇒ a 5-restart × 900-epoch run ≈ 2.5 hours;
fine if the shakedowns were done at small scale first). Consider fitting at
a musically-complete but cropped length (attack + enough sustain to
establish the timbre + release), the same shrink-to-target logic rung 3
already has. Do not fit minutes of a held note.

**Suggested first bass milestone** (keeps risk low): a *synthetic* bass
rung-2 — render a saw + filter-env patch from known params with the new
voice, recover it. That validates envelope, harmonic bank, filter gradients,
and the widened windows before any real-world capture with unknown
provenance enters the picture.

## Failure modes catalog (all observed on 808/909)

| Symptom | Likely cause | Fix |
|---|---|---|
| Great reported %, mediocre A/B | Gate-inflated baseline (winning restart's own cold start) | Corrected protocol is default now; always also report absolute distance |
| Param sits exactly at a bound all run | Init on the bound, or param is compensating for a different error | Never init on bounds; diagnose what it's patching before widening |
| Training regresses in silence at the tail | Target padded to config.frames | Fit at real length (rung 3 does this; keep it for new profiles) |
| Loss improves, gate doesn't | Training-loss vs independent-metric basin mismatch | That's what refine_rung3 is for; probe ideas against the CPU metric |
| Multiply-by-tensor silently becomes no-op | Tensor created before `LazyGraphContext.reset()` (stale nodeId) | CLAUDE.md lifecycle rules |
| fdcheck relErr ~0.7 on pitch params | Known swept-phasor discrepancy | Pre-existing; don't chase it |
| Recovery pinned to wrong pitch basin | CPU pitch fit clamped/biased (909's 80–180 Hz clamp) | Verify `pitch_fit.json` against manual measurement first |
| Loss floor won't go below X | Codec/capture texture in a lossy source | Get a lossless capture; the rules forbid memorizing texture |
