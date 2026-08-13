# Spec: eliminate per-epoch recompilation in `dgenlisp train`

Status: DESIGN — ready to implement. Owner: (agent).
Companion docs: `docs/TRAIN_SUBCOMMAND_NOTES.md` (deviations list, where this
is the top perf item), `CLAUDE.md` ("DGenLazy Training Loop", "Tensor/Signal
Creation Order Matters" — the invariants this design must not break).

## Problem

`DirectionTrainer.runPhase` (Sources/DGenLisp/DirectionTrainer.swift) runs a
correctness-first epoch loop: every epoch does

1. `configureRuntime` + `LazyGraphContext.reset()` (fresh `LazyGraph`),
2. `LispEvaluator().evaluate(nodes: patchPlan.loweredNodes)` (rebuild voice),
3. `updateDataLazily` the learnable param values,
4. build target signal + MR-STFT loss, `loss.backward(frames:)`.

Because `reset()` replaces the `LazyGraph` *instance*, every per-instance
cache is discarded, so each epoch pays full DGen compilation passes AND full
Metal pipeline creation (`MTLLibrary` compile) for a graph whose **topology is
bit-identical every epoch** — only param cell values change.

Measured on the monologue-bass patch (34 params, 38,072 frames, 4-window
log-L1 + linear MR-STFT, M-series, release build):

- current: **~1.2–1.9 s/epoch** (37.9 s wall for a 10-epoch job = 20 actual
  epochs + plan + final render; only ~5 s of it is CPU time)
- production serial trainer at comparable frame counts (Examples/SynthID,
  reset-once + rebuild-loss pattern): **0.27–0.39 s/epoch**
- target: **≥3x** on the standard 15-epoch confirm run, stretch parity with
  the production trainer.

## Where the time goes (verify first — Milestone 0)

Add a per-epoch timing breakdown to stderr behind `DGENLISP_TRAIN_TIMING=1`:
lisp eval / graph build, DGen compile passes, Metal pipeline creation, GPU
execution, optimizer step. One 15-epoch monologue run before and after each
milestone; keep the numbers in the PR description. Expected split based on
the 5 s CPU / 38 s wall measurement: Metal library+pipeline creation
dominates (~1 s/epoch), DGen compile passes ~0.2 s, GPU exec ~0.3 s.

## Design

Two milestones, independently landable. Milestone 1 is low-risk and captures
most of the win; Milestone 2 reaches production parity but touches the
evaluator lifecycle.

### Milestone 1 — kernel/runtime caches survive `LazyGraphContext.reset()`

`LazyGraph.runtimeCacheByKernelHash` (Sources/DGenLazy/LazyGraph.swift:42)
already caches `LazyRuntime` (MTLLibrary + pipeline states) by kernel source
hash and survives graph *clears* — but it is an instance property, so
`reset()` (LazyGraph.swift:~217) discards it.

Change: move the kernel-hash → runtime cache (and only it) to shared storage
that survives instance replacement. Options, in order of preference:

- a static registry on `LazyGraphContext` (single-threaded CLI; no locking
  needed beyond what exists), consulted by whatever code currently reads
  `runtimeCacheByKernelHash`;
- or copy the cache forward inside `reset()` from the outgoing instance.

Also evaluate carrying `fullCompilationCache` (LazyGraph.swift:46, keyed by
node/tensor/frame-count fingerprint) across resets the same way — that skips
the DGen compilation passes too. If its fingerprint is too coarse to be safe
across arbitrary resets, gate the carry-over behind an explicit opt-in that
only `DirectionTrainer` sets (e.g. `LazyGraphContext.preserveCompilationCaches
= true`), so `compile`/tests keep today's behavior.

**Determinism prerequisite**: cache hits require the generated kernel source
to be byte-identical across epochs. Same eval order over the same lowered AST
in a fresh graph should give identical node/cell ids and identical source —
but verify: dump kernel source hashes for 3 consecutive epochs
(`DGENLISP_TRAIN_TIMING=1` can print them) and confirm they repeat. If they
don't, find the nondeterminism (dictionary-order iteration in codegen,
timestamped names leaking into hashed source) and fix that first; without it
this milestone is a no-op.

**Explicitly out of cache scope**: memory *contents*. Param cells, history
cells, and gradient carry cells are re-allocated per fresh graph; only the
compiled artifacts (and pipeline states) are reused. State reset per epoch
must behave exactly as today (each epoch's render starts from silence).

### Milestone 2 — reset once per phase, rebuild only the loss (production pattern)

Adopt the Examples/SynthID `Trainer.swift` loop shape: one
`LazyGraphContext.reset()` per phase, params created once, per-epoch rebuild
of the computed graph only. For lisp patches this needs an evaluator rebuild
mode, because today re-evaluating the AST in the same graph would duplicate
params/cells:

- `LispEvaluator` gains a re-entrant mode: on re-evaluation, `(param name …)`
  returns the *existing* registered `Signal` for `name` (values preserved)
  instead of allocating a new cell; everything else is rebuilt fresh.
- The per-epoch loop becomes: rebuild voice + target + loss via the
  re-entrant evaluator, `backward`, step — matching the tinygrad-style
  contract in CLAUDE.md (graph cleared after backward; params survive via
  lazy nodeId recreation).
- Watch the known hazards from CLAUDE.md / memory: stale-nodeId aliasing
  (never reuse a Signal created before the current graph generation except
  through the param registry), cell-counter growth across epochs (verify
  cells don't accumulate; the History/BPTT carry cells are re-created per
  rebuild), and `DGenSpectralConfig.logMagnitudeEpsilon` being read at
  codegen time (set once per phase before the first build).

Milestone 2 makes `fullCompilationCache` hit naturally (same graph instance,
same fingerprint) and should reach ~0.3 s/epoch. If Milestone 1 alone gets
under ~0.5 s/epoch, Milestone 2 can be deferred — decide on measured numbers.

## What must NOT change

- **Gradient correctness.** `DGENLISP_TRAIN_FDCHECK=all` on the monologue
  patch (see TRAIN_SUBCOMMAND_NOTES for the harness) must produce the same
  autograd values before/after (float tolerance 1e-4 relative). The four
  history reproducers in the notes doc are the regression canaries for the
  history-write BPTT fix — do not regress `Sources/DGen/Gradients.swift`
  history-write-root behavior while touching compilation caching.
- **Trajectory equivalence.** A 15-epoch monologue run before/after must
  produce identical epoch-event loss values within 1e-3 relative (Metal
  nondeterminism allowance). Byte-identical NDJSON except loss float noise.
- **Protocol layer untouched.** No changes to DGenTrainProtocol events,
  job-dir artifacts, or exit-code contract; TrainCLITests / GoldenTranscript
  / mock-host tests must pass unmodified.
- **`dgenlisp compile` behavior.** The compile subcommand and DGenLispTests
  must be unaffected (cache carry-over must be invisible or opt-in).
- Per-epoch **state reset semantics**: every epoch renders the voice from
  zero state (fresh history cells / silence), exactly as today.

## Acceptance criteria

1. `DGENLISP_TRAIN_TIMING=1` breakdown landed (Milestone 0) with before/after
   numbers in the PR.
2. ≥3x wall-clock speedup on: 15-epoch monologue confirm run (today ~38 s
   for the 10-epoch variant) and the 150-epoch full run (today ~5–6 min).
3. `swift test --filter 'TrainPlannerTests|TrainCLITests|TrainE2ETests|DGenTrainProtocolTests'`
   green; `swift test --filter 'BPTTTests|SVFBPTTScratchTests|BPTTBiquadScratchTests|HistoryTensorTests|TensorBiquadTests'`
   green.
4. fdcheck + trajectory equivalence gates above.
5. TrainE2ETests threshold/time budget updated only if the speedup allows
   *raising* epochs, never loosening the improvement gate.

## Pointers

- Loop: `Sources/DGenLisp/DirectionTrainer.swift` (`runPhase`,
  `configureRuntime`), fdcheck harness in the same file.
- Caches: `Sources/DGenLazy/LazyGraph.swift` (`runtimeCacheByKernelHash`,
  `fullCompilationCache`, `LazyGraphContext.reset`), `Sources/DGenLazy/Realize.swift`.
- Production reference loop: `Examples/SynthID/Trainer.swift:72-313`,
  `Examples/SynthID/BatchRefine.swift:377` (reset-once conventions in the
  file header comment).
- Evaluator/param registry: `Sources/DGenLisp/LispEvaluator.swift`
  (`evalParam` ~1382, `definitions`), `Sources/DGenLazy/Signal.swift:137`
  (`Signal.param`, `updateDataLazily`).
- Timing baseline commands: TRAIN_SUBCOMMAND_NOTES "Post-landing findings"
  section; monologue patch + seed live in the shakedown scratchpad — recreate
  from any eseq-style patch with the `svf` macro, or use
  `Tests/DGenLispTests/TrainE2ETests.swift` assets for CI-sized timing.
