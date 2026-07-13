# SynthID Subtractive Fitting Path — Spec & Experiment Plan

**Status: proposal / to validate.** Nothing below has run yet.

## Thesis

Fit a *subtractive* voice (band-limited oscillator → resonant filter with
envelope → VCA envelope → drive) instead of an additive harmonic bank, so that
**the fit topology is the deployment topology**. The optimizer's output is
then literally the knob settings of a synth the user could patch — no port
step, no spectral-snapshot parameter blob, and near-zero runtime cost.

Two claims to validate, in order of ambition:

1. **Recovery** — on a real bass/pluck sample, a subtractive voice with
   ~10 scalars matches or approaches the additive voice's result at a tiny
   fraction of the parameter count and deployment cost.
2. **Direction-finding** — even when the topology cannot represent the target
   exactly, gradient descent from a user's *existing* patch moves it reliably
   toward the target sound ("train my patch to this sample"). The contract is
   projection onto the model manifold, not exact recovery.

## Motivation (why now)

- The hoodie-bass additive port (`HOODIE_BASS_STATUS.md`) costs ~100 trig
  evaluations per sample per voice: **18% CPU at 6 voices** in the eseq DAW —
  worse than dense 2D physical models. Additive was chosen as the low-risk
  *training* topology; deployment cost was never weighed.
- The blockers that originally justified avoiding subtractive are gone:
  - The biquad cutoff gradient sign bug is **fixed** (commit bbb3769;
    FDCHECK_FINDING.md — filtered cutoff fdcheck matches finite difference).
  - Biquad BPTT (pass-through history writes, selector off-by-one) is fixed.
  - DGen's `phasor` is a 0–1 ramp, so naive saw/pulse need **no new op**
    (see Oscillator section). Only band-limiting/shape needs design.
- The end-state UX: the eseq DAW is dgenlisp; a subtractive profile is the
  first instance of "compile the user's own patch into the trainer."

## Deployment target (eseq polyblep macros)

These already exist in the eseq repo and define what the recovered parameters
drive at runtime:

```lisp
; PolyBLEP transition correction for anti-aliased hard edges.
(defmacro polyblep (phase freq)
  (def dt (clip (/ freq samplerate) 0.000001 0.5))
  (def left_x (/ phase dt))
  (def left (+ (- (* 2.0 left_x) (* left_x left_x)) -1.0))
  (def right_x (/ (- phase 1.0) dt))
  (def right (+ (* right_x right_x) (* 2.0 right_x) 1.0))
  (+ (* (lt phase dt) left)
     (* (gt phase (- 1.0 dt)) right)))

(defmacro polyblep_saw (phase freq)
  (- (scale phase 0 1 -1 1)
     (polyblep phase freq)))

(defmacro polyblep_pulse (phase width freq)
  (def w (clip width 0.01 0.99))
  (def falling_phase (wrap (- phase w) 0 1))
  (+ (scale (lt phase w) 0 1 -1 1)
     (polyblep phase freq)
     (* -1.0 (polyblep falling_phase freq))))
```

The training graph directly transcribes these deployment macros. With `f0`
frozen from the CPU pitch fit, `dt` is constant; `shape` and `pw` differentiate
through ordinary arithmetic, wrap, and the PolyBLEP quadratics. Training and
deployment therefore use the same oscillator topology rather than relying on
a spectral stand-in.

## Training-side oscillator

`phasor` is a 0–1 ramp. The training graph builds `polyblep_saw` and
`polyblep_pulse` exactly as above, with no new ops or custom adjoint:

```text
osc(φ) = (1 − shape)·polyblep_saw(φ, f0)
       + shape·polyblep_pulse(φ, pw, f0)
```

The raw pulse step and the falling-edge BLEP correction have equal and
opposite jumps, so their sum is continuous as `pw` moves. Zero gradients
through the `lt`/`gt` gates are correct: gate boundaries either have a zero
polynomial payload or have discontinuity terms that cancel in the complete
pulse. The remaining `pw` gradient flows through
`fallingPhase = wrap(phase − pw, 0, 1)` into the smooth quadratics, where
DGen's existing `d wrap / d input = 1` adjoint is the required derivative
almost everywhere.

This claim remains subject to E0 fdcheck. If it fails, the rule-legal fallback
is additive evaluation with closed-form harmonic weights (saw: `1/n`; pulse:
`sin(πn·pw)/n`) and the same scalar controls.

## Voice topology (profile: `subtractive-bass`)

```text
φ         = statefulPhasor(f0)                       # f0 frozen from CPU pitch fit
oscOut    = osc(φ, shape, pw)
fEnv      = fBase + fAmt · exp(−t / fDecay)          # filter EG (decay-only, v1)
filtered  = biquad(oscOut, cutoff=fEnv, res, mode=LP)
aEnv      = smooth ADSR (attack, decay, sustain, release, noteOff)
out       = softsign(filtered · aEnv · drive) · gain
```

### Parameters (~12 scalars)

| Param    | Unit  | Bounds        | Scale | Notes |
|----------|-------|---------------|-------|-------|
| shape    | –     | 0 – 1         | lin   | saw ↔ pulse blend |
| pw       | –     | 0.03 – 0.97   | lin   | matches eseq clip bounds |
| fBase    | Hz    | 30 – 8000     | log   | filter cutoff floor |
| fAmt     | Hz    | 0 – 12000     | log-ish (softplus) | env sweep depth |
| fDecay   | s     | 0.005 – 2.0   | log   | filter EG decay |
| res      | –     | 0.5 – 6.0     | log   | biquad Q |
| attack   | s     | 0.001 – 0.5   | log   | |
| decay    | s     | 0.01 – 2.0    | log   | VCA |
| sustain  | –     | 0 – 1         | lin   | |
| release  | s     | 0.01 – 1.0    | log   | |
| drive    | –     | 0.25 – 8.0    | log   | |
| gain     | –     | 0.05 – 2.0    | log   | |

`f0` frozen from the CPU steady-pitch estimate, as in the hoodie-bass profile
(swept-pitch adjoint exception still applies). Optional zero-default add-ons,
only if measurement demands them: one symmetric detune pair (± cents) for
VCO beating; a noise source for analog floor.

### Known degeneracies (score invariant products, per SPEC §7.1 discipline)

| Ridge | Invariant to score |
|---|---|
| drive × level (rung-1 classic) | effective drive·gain product / output RMS |
| cutoff vs shape brightness | spectral centroid of sustain segment |
| fBase vs fAmt | fEnv(0) = fBase + fAmt and fEnv(∞) = fBase (sweep endpoints) |
| VCA decay vs filter decay | joint fit quality only; flag if either hits a bound |

Individual knobs inside a ridge are reported with a "compensating?" flag when
the residual is large (playbook failure-mode table applies).

## Experiments

Ordered by information gained per hour. Each has a hard gate; a failed gate
stops the ladder and files the finding.

### E0 — Gradient prerequisites (fdcheck, hours)

1. **Time-varying cutoff**: envelope signal driving the biquad coefficient
   per-frame. This is the one gradient path *never* exercised by any previous
   rung (the kick's noiseCutoff was constant). fdcheck fBase, fAmt, fDecay
   through the full voice at the real window config.
2. PolyBLEP oscillator params: shape, pw through the STFT loss.
3. res, and re-confirm constant cutoff (regression check on bbb3769).

**Gate**: relative error < 1e-2 on every trained param (same bar as the
hoodie-bass fdchecks), evaluated under a smooth probe objective: multi-window
log-magnitude L2 with the same Hann windows, hops, weights, and log epsilon as
training. This changes only the production loss's intentional binwise L1 kink;
the complete `log(magnitude(STFT(x)) + epsilon)` adjoint remains exercised.
Use the following fixed epsilon-convergence protocol in transformed parameter
space:

1. Sweep `fdEpsilon` over
   `[1e-4, 1.5e-4, 2e-4, 3e-4, 5e-4, 1e-3, 2e-3, 3e-3, 5e-3, 1e-2, 3e-2, 1e-1]`.
2. A parameter passes only if two adjacent epsilon values both have relative
   error < 1e-2 and their finite-difference gradients agree with each other to
   < 1e-2 relative error.
3. Use the first passing adjacent pair in ascending epsilon order. Do not add,
   remove, or retune epsilon values after seeing a parameter's result.
4. A parameter fails if no adjacent pair passes. Stop E0 at that parameter;
   later parameters and experiments remain unrun.

The transformed-coordinate sweep remains the gate for filter parameters. For
waveform-composition parameters (`shape`, `pw`), an actual-voice isolation on
2026-07-12 established that it is not a valid numerical instrument: the
independent float64 objective has enough local curvature that `1e-4` is already
outside its linear neighborhood, while smaller float32 loss differences are
cancellation-limited. The waveform-parameter gate is therefore frozen as the
mathematically equivalent chain-rule decomposition:

1. Full-voice time-domain MSE fdcheck must satisfy the same fixed-grid,
   adjacent-pair, and `1e-2` criteria. This checks `d(signal)/d(param)`.
2. Render the actual centered voice tangent in transformed coordinates with
   fixed `directionEpsilon=1e-4`, then check the smooth spectral objective on
   `x + alpha*v`. Full-voice Metal autograd, directional Metal autograd, and an
   independent NumPy float64 analytic directional derivative must agree pairwise
   to relative error `< 1e-2`.
3. The NumPy float64 central differences at `1e-6` and `2e-6` must each agree
   with its analytic derivative and with each other to `< 1e-2`.

This changes the measurement, not the error bar. For `pw`, PolyBLEP's
piecewise-polynomial derivative can have kinks near the transition width
`dt = f0/sampleRate`; failure of either chain-rule component is still a hard
stop and cannot be excused by proximity to `dt`.

The production log-L1 loss is separately swept along `shape` to document its
local slope breaks, but that intentionally non-smooth objective is supporting
evidence rather than the fdcheck gate. The smooth-probe protocol is frozen for
E0; a failure under it is a library bug to fix first, not a tuning problem.

> **2026-07-12 protocol correction:** an actual-voice directional isolation and
> independent float64 reference showed that the `shape` adjoint is correct and
> the end-to-end float32 central difference is ill-conditioned. The chain-rule
> gate above replaces that instrument once for both waveform parameters. See
> `E0_FINDING.md` for the signed residual and reference tables.
>
> **Final E0 result: PASS.** `fBase`, `fAmt`, `fDecay`, `shape`, `pw`, `res`,
> and the bbb3769 constant-cutoff regression all pass their authoritative
> `< 1e-2` checks. No training run was performed and E1 was not started.

### E1 — Rung 1 self-inversion (synthetic, days)

DGen renders a hidden-param target from the `subtractive-bass` voice itself;
recover it. This is where degeneracies get mapped cheaply and the invariant
products are finalized. Follow the rung-1 recipe (log-eps 1e-3, cosine LR,
never init on bounds, deterministic restarts).

**Gate**: rung-1 acceptance style — invariant products within 10%, audible
match, majority of seeds.

### E2 — Rung 2 independent renderer + polyblep equivalence (days)

1. NumPy reference renderer for the voice (naive or additive-eval oscillator,
   whichever E1 settled on): max abs error ≤ 1e-3 vs Swift, as usual.
2. **New**: polyblep equivalence — render the same params through the eseq
   polyblep macros; gate on log-mag MR-STFT distance between training-osc and
   polyblep renders being ≪ the fit residual (proposed: < 5% of the E3
   learned distance). This is what licenses "the recovered knobs mean the
   same thing at deployment."

### E3 — Rung 3 on hoodie bass G#2 (the headline experiment)

Same target, baseline protocol, and independent comparator as
`HOODIE_BASS_STATUS.md` (`output/rung3_hoodie_bass_gs2_baseline_audit/`
target; deterministic midpoint+pitch-fit baseline; `compare.py`).

The additive result to beat/approach: **70.55%** improvement, absolute
learned distance **0.0616**, with ~200 trainable scalars.

**Gates** (tiered — this is model-mismatch territory, a Monologue is not
exactly one osc + one 2-pole LP + softsign):

- **Pass**: ≥ 60% improvement with ≤ 14 scalars → subtractive path validated
  as a *recovery* method; proceed to E5.
- **Partial**: 40–60% → validated as *direction-finding* only; E4 becomes the
  product story; document the residual (what the topology can't express).
- **Fail**: < 40% or fdcheck-clean params pinned at bounds → analyze
  compensation, consider the additive-eval oscillator fallback or a second
  filter pole before concluding.

Always report absolute distance alongside the gate (corrected-baseline
discipline; the 808's 84.55% → 77.53% lesson).

### E4 — Direction-finding mode (the new capability)

Seed the optimizer from a **user-supplied patch** (knob settings), not
restarts: short run (~100–300 epochs, small LR), report per-param deltas
("cutoff up 2.1×, filter decay 40% faster, shape → 0.7") plus the improvement
% and absolute distance. One background cold restart as a basin check —
if it beats the seeded run decisively, report "wrong neighborhood" instead of
deltas.

Test matrix on hoodie bass G#2: seed from (a) near-truth E3 winner perturbed,
(b) a generic init-patch bass, (c) an adversarially wrong patch (bright pluck
settings). **Gate**: (a) and (b) close ≥ 70% of *their own* remaining
distance to the E3 winner's distance; (c) is correctly flagged by the basin
check.

### E5 — Stretch: foreign target + deployment (Prophet 6 pluck)

A real Prophet 6 pluck sample (capture provenance recorded, playbook Phase 1
measurement first). Fit with the same voice (+ detune pair if beating is
measured). Port = write the recovered scalars into an eseq polyblep patch —
no code generation.

**Gates**: direction-finding-tier improvement (≥ 40%) on a synth the topology
was not designed around; eseq instrument CPU **< 5% at 6 voices** (vs 18%
additive) via the DAW's own meter.

## Non-goals (v1)

- Oscillator sync, ring mod (genuinely hard discontinuities — feasibility doc
  stands).
- Chorus/FX modeling; capture-chain discipline handles coloration, FX beyond
  that are declared residual.
- Per-harmonic anything. The whole point is that the parameter vector is a
  patch sheet.
- Replacing the additive path — it remains the high-capacity fallback (and
  its runtime problem has a separate fix: the three brightness banks compile
  to static single-cycle wavetables).

## Formalization criteria

The approach graduates from experiment to method when: E0–E3 gates pass; the
degeneracy table is complete with invariants scored in `Report.swift` fashion;
the polyblep-equivalence gate is automated; and E4's delta-report format is
stable enough to wire into the eseq UI. At that point this document gets
rewritten as a SPEC.md-style normative spec and the playbook gains a
subtractive profile section.

## Corrections this spec depends on (already applied)

- `NEW_TARGET_PLAYBOOK.md` biquad caution updated: cutoff gradient sign bug
  fixed at bbb3769; only time-varying cutoff remains unverified (→ E0).
- `MONOLOGUE_FEASIBILITY.md` oscillator paragraph corrected: `phasor` is a
  0–1 ramp, not a sine; naive saw/pulse need no new op.
