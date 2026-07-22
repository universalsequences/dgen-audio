# Spec: Batched lowering of dgenlisp voices (v0.1)

Status: DRAFT — 2026-07-21. Target: the "Korg-style" subtractive eseq voice
(saved as the reference example below) runs as a `[B]`-lane batched DGenLazy
graph with working forward + BPTT backward, driven by the existing
batch-refine harness, from unmodified dgenlisp source.

## Problem statement

The SynthID batched refinement system (`Examples/SynthID/BatchRefine.swift`,
`docs/TENSOR_BIQUAD_PARALLEL_LANES_SPEC.md`) trains B independent parameter
candidates in one lane-per-thread graph at ~9x per lane-step vs serial. But
the batched voice is hand-written Swift: every new synth needs a manual port.

The end goal is to batch **any eseq synth**, and eseq synths are dgenlisp
source. The insight that makes this cheap: `LispEvaluator` is a polymorphic
interpreter where `.signalTensor` is already a first-class value with mixed
`Signal × SignalTensor` broadcasting for arithmetic
(`Sources/DGenLisp/LispEvaluator.swift:811-935`) and dual-map
`read-history`/`write-history` dispatch across scalar and tensor bindings
(`:312`, `:357-370`). So batching is a **lowering mode, not a language
change**: inject tensor-ness at the leaves (`param`, `make-history`,
`phasor`) and the existing type dispatch propagates it through the entire
voice body untouched. The synth author keeps writing scalar instruments.

Survey evidence (2026-07-21) that the backend is ready for this: the
lane-parallel dispatch and BPTT consolidation are keyed to **generic
tensor-registered history cells**, not to biquad —
`StatefulTensorParallelPolicy.swift:47-53` (forward eligibility),
`:124-205` (`laneParallelizable` audit), `BlockFormation.swift:494`
(`consolidateTensorBPTTBackwardBlocks`, gated on `tensorGradCarryCells`).
The built-in biquad is itself just 4 tensor history cells
(`Sources/DGen/HigherOps.swift:401-440`). A hand-built Cytomic SVF lowered
through `TensorHistory` produces structurally identical IR and gets the
identical treatment.

## Reference example

The v0.1 acceptance target is the Korg-style subtractive voice (2 VCOs +
sub + noise, polyblep saw/pulse, SVF HP → SVF LP/BP, dual gswitch ADSRs,
LFO + 3 drift oscillators, tanh drive stages, ~30 params). A copy lives at
`Examples/SynthID/voices/korg_subtractive.lisp` (check in with this spec).
Its wavetable macros are defined but unused — wavetables are explicitly out
of scope for v0.1 (see Non-goals).

## Design

Two components: a **rewrite pass** over s-expressions, then a **batch
evaluation mode** of the existing evaluator.

### 1. Rewrite pass (pre macro-expansion)

Operates on the parsed AST, matching macro/op head symbols against a small
registry. v0.1 rewrites:

1. **`adsr` → closed-form envelope.** The gate/trigger state machine exists
   for live retriggering; for fitting a one-shot note whose on/off times are
   known constants (eseq knows the schedule), the closed form is exact and
   strictly better for gradients (every `gswitch` branch in the macro gives
   zero/straight-through grads). Template — the same shapes the batched
   SynthID voice already trains through (`BatchRefine.swift:207-211`):
   - attack: `1 - exp(-t/τ_a)`
   - decay-to-sustain: `sustain + (1-sustain)·exp(-(t-t_a)/τ_d)`
   - release: multiply by logistic `1/(1+exp((t-t_off)/τ_r))`
   Rewrite consumes `(adsr gate trigger a d s r)` where `gate` resolves to a
   known note schedule; emits the closed form parameterized by the same four
   param signals. Error loudly if `gate` is not a schedule the rewriter
   understands (live input → not batchable in v0.1).
2. **`noise` → hoisted shared source.** Per-lane `.noise` inside the
   recurrence block disqualifies lane-parallel dispatch
   (`StatefulTensorParallelPolicy.swift:157`). Rewrite hoists one shared
   scalar noise signal outside the voice; all lanes hear the same hiss.
   Acceptable for fitting (the target heard one noise realization too).
3. **Selector lint.** DGen's backend already constant-folds mode-style
   switches (`svf` mode, `eq mode N` chains) when the selector is a literal
   — but not when it is a `param`. v0.1 does not add folding; it adds a
   **lowering-time error** when a `gswitch`/`eq`-selector chain's selector
   depends on a `param` or `in`, with a message naming the offending symbol.
   The reference voice uses literal modes (2, 0, 1) throughout, so backend
   folding already deletes the dead filter modes. Searching over discrete
   modes (one-hot relaxation) is future work.

No other rewrites in v0.1. `svf`/`ladder`/`polyblep` lower as-is through the
generic path — no special-casing, that's the point. (Optional later:
matching `svf` to dedupe same-state instances — the reference voice's LP and
BP stages share input/cutoff/q and could be one instance — v0.1 just runs 3
SVFs and eats the cost.)

### 2. Batch evaluation mode

A `BatchConfig(laneCount: Int)` handed to the evaluator changes **only leaf
lowering**:

| form | scalar mode (today) | batch mode |
|---|---|---|
| `(param x @default @min @max)` | constant / scalar signal | per-lane trainable `[B]` tensor: `Tensor.param` initialized per-lane (init strategy supplied by harness: jitter around defaults, LHS sample, elite seeds), then `clip` to `[@min, @max]` in-graph. Linear scale in v0.1; log-reparam for Hz/ms params is future work. |
| `(in n @name gate/pitch/…)` | runtime input | shared scalar `Signal` (note schedule / constants supplied by harness); broadcast into lanes by existing mixed arithmetic. `@modulator` inputs default to 0. |
| `(make-history name)` | `Signal.history()` | `TensorHistory(shape: [B])` — read/write dispatch already handles the tensor binding map |
| `(phasor f)` | scalar phasor | `Signal.statefulPhasor(freqTensor)` when `f` is tensor-valued (`Sources/DGenLazy/SignalTensor.swift:86-120` — per-lane freq AND per-lane phase state); scalar phasor stays scalar when `f` is scalar (drift/LFO phasors with constant rate params CAN stay scalar and broadcast — but note a scalar phasor's output feeding tensor math is fine, while scalar *state cells* inside a tensor recurrence block are a lane-parallel disqualifier `StatefulTensorParallelPolicy.swift` scalar-cell check; simplest v0.1 rule: if any param is tensor, ALL phasors lower tensor) |
| `(out sig …)` | output | per-lane render buffer OR directly into the batched spectral loss (mean over lanes — remember the ×B grad rescale, `BatchTrainBench` convention) |

Everything between the leaves — the whole voice body, all macros — evaluates
unmodified through existing dispatch.

### 3. Gap-filling work items (the actual code changes)

DGenLazy `SignalTensor` overloads (UOps + gradients already exist; each is a
small wrapper mirroring the existing `Functions.swift` patterns):

- [ ] `tan(SignalTensor)` (pattern at `Functions.swift:46,113`; grad exists
      `Gradients.swift:397`). Needed for SVF `g = tan(π·fc/fs)` with
      per-lane cutoff.
- [ ] `clip(SignalTensor, lo, hi)` (pattern `Functions.swift:756-785`) — or
      lower as `min(max(…))`, both exist.
- [ ] `gswitch(SignalTensor, …)` (pattern `Functions.swift:491-537`) — or
      lower branchless `cond*a + (1-cond)*b` like `BatchRefine.swift:153,198`.
- [ ] `wrap(SignalTensor, 0, 1)` — lower via `mod` (exists).

Evaluator dispatch:

- [ ] `.signalTensor` cases in `applyUnaryMath`
      (`LispEvaluator.swift:964-…` — currently stops at `.tensor`),
      `evalComparison`, `clip`, `gswitch`, `wrap`, `scale`, `pow`, `min/max`
      where missing. Mechanical; mirror the `.signal` arms.

Gradient plumbing:

- [ ] `TensorHistory.read()` hardcodes `requiresGrad: false`
      (`Sources/DGenLazy/Tensor.swift:344`). Biquad works only because its
      wrapper forces `needsGrad` (`Functions.swift:1163-1176`). Fix: thread
      a `requiresGrad` flag through `TensorHistory` (constructor or
      graph-level "training mode"), mirroring biquad. Without this,
      hand-built filters silently fall off the tape — add a test that FAILS
      today to pin the behavior.

Harness:

- [ ] `evaluateBatched(source:, config:) -> BatchedVoice` API in DGenLisp
      returning the loss-ready `[B]` output tensor + the param tensor map
      (name → tensor, min/max) for the optimizer.
- [ ] `SynthID lisp-batch-refine <voice.lisp>` subcommand wiring the above
      into the existing batch-refine loop (Adam groups from param map,
      spectral loss, per-lane metrics) — mostly plumbing reuse from
      `BatchRefine.swift`.

## Validation ladder (mirrors the SynthID recipe)

1. **Conformance (forward):** render the post-rewrite voice through scalar
   lowering and through batch lowering at B=1 and B=8 (identical params per
   lane); audio must match to ~1e-6 rel (the tensor-biquad forward matched
   serial to 1.8e-7). This is the per-synth safety gate: any eseq voice
   either passes or errors with the offending construct named.
2. **Gradient probe:** `probe-grads`-style comparison of batched vs scalar
   backward on the full voice. Expect agreement ~2e-3, not 2e-5 — polyblep
   /L1 kink noise is shared by both paths and FD cannot arbitrate at rough
   points (established on the SynthID voice; don't chase it).
3. **Hidden-param recovery:** render a target from known params, recover
   from B=64 jittered lanes. Success = the escape-mode bar from
   `BATCH_REFINE_FINDING.md` (majority of params recovered, loss ratio
   competitive with serial single-lane refinement).

## Performance expectation (extrapolated, to be measured)

Anchors: SynthID voice B=64 = 0.0053 s/lane-step at 8,192 frames (8.4x vs
serial 0.0445); B=32 = 0.0077. The reference voice is ~4-6x the per-frame op
count (7 phasors, 5 polybleps, 3 SVFs, 4 tanh stages, per-frame `tan`
coefficient math). Lane-parallel throughput scales with op count, so
expect roughly **0.02-0.03 s/lane-step at 8k frames / B=64**, and the ~8-9x
ratio over serial to hold (same parallelization structure on both paths).
Batch only at B≥32 (occupancy; established in the parallel-lanes spec).

## Risks

- **Multi-cell BPTT scale.** The consolidation pass is validated on one
  4-cell recurrence; this voice consolidates ~10-14 tensor carry cells plus
  several tensor phasor states in one block. The pass is generic by design,
  but carry-cell aliasing has bitten five times — budget a debugging round,
  and run rung 2 at B=2 first (small kernels are readable).
- **RMW form.** Lane-parallel dispatch rejects fused `.historyReadWrite`
  (`StatefulTensorParallelPolicy.swift:42-46`). The lisp `make-history`
  surface uses separate read/write, which lowers to the allowed form —
  verify nothing in HistoryFusion re-fuses tensor cells into RMW.
- **Scalar state inside tensor blocks.** If any scalar phasor/history lands
  in the recurrence block it disqualifies lane-parallelism — hence the
  "all-tensor phasors" rule above. Add a diagnostic that reports WHICH node
  failed `laneParallelizable` instead of silently taking the slow path.
- **Envelope rewrite fidelity.** Closed-form ADSR ≠ the macro's one-pole
  staged envelope (different decay curvature). Fine for v0.1 (the batched
  system trains its own envelope params), but conformance rung 1 must
  compare post-rewrite scalar vs batched — not original vs batched.

## Non-goals (v0.1)

- **Wavetable reads** (`peek` with per-lane index). Cross-lane gather is a
  lane-parallel disqualifier (`StatefulTensorParallelPolicy.swift:174-178`).
  The right future shape is a lane-uniform-table gather op (shared table,
  per-lane index — provably safe) which would also unlock wavetable
  *starting-point search*; big win, explicitly tabled.
- **Discrete mode search** (params as gswitch selectors; one-hot
  relaxation). v0.1 errors on param-dependent selectors.
- **dgenlisp surface changes.** No tensor syntax, no `@batch` attributes.
  Authors write scalar instruments; batching is a lowering concern.
- **Per-lane noise, rank>1 lanes, log-scale param reparams, SVF instance
  dedup, live gate/retrigger fidelity.**

## Milestones

- M1: overloads + evaluator dispatch + `requiresGrad` fix; minimal-voice
  (one phasor → one SVF → out) passes rungs 1-2.
- M2: rewrite pass (adsr, noise-hoist, selector lint); reference voice
  passes rung 1 forward conformance at B=8.
- M3: full reference voice backward at B=64; rung 3 hidden-param recovery;
  measure s/lane-step against the extrapolation above.
