# Spec: Gradients through tensor-shaped biquad (batched BPTT)

Status: IMPLEMENTED — validated 2026-07-20. See "Implementation results" at the
bottom for what was built, the test evidence, and the measured (modest) speedup.

## Problem statement

`Graph.biquad` / `SignalTensor.biquad` (forward-only, `elementShape: Shape?`)
was implemented and validated in `docs/TENSOR_BIQUAD_SPEC.md` (2026-07-14).
It gives correct batched `[B]`-shaped filtering — each lane gets independently
sized, tensor-registered history cells — and is proven at production scale in
the `subtractive-bass` E1 basin-search pipeline (23,552 forward evaluations
in 5.3 min, ~13.6 ms/candidate, matching the earlier micro-benchmark).

Gradients through that op were explicitly out of scope and are hard-blocked:

```swift
// Sources/DGenLazy/Functions.swift:1169-1170, 1189-1190
graph.unsupportedGradientNodes[nodeId] =
  "tensor-shaped biquad backward passes are not implemented"
```

`LazyGraph.runBackward` (`Sources/DGenLazy/Backward.swift:159-161`) walks the
loss's dependency graph before compiling and throws `DGenError
.unsupportedGradient` if any visited node carries an entry in
`unsupportedGradientNodes`. So today, any attempt to `backward()` through a
`SignalTensor.biquad` fails loudly and immediately — which is correct
behavior for now, but it's the wall this spec is about removing.

## Why this matters (motivating use case)

The E1 basin-search policy (`output/e1_basin_v2_seed6_run2/`,
`Examples/SynthID/scripts/refine_elites.sh`) runs in two stages:

1. **Batched forward-only search** — cheap, uses the tensor biquad above.
   23,552 candidates in 5.3 minutes.
2. **Serial per-elite Adam refinement** — 12 elites, each independently
   trained for 600 (smooth loss) + 800 (production loss) epochs through the
   scalar `Signal.biquad` path, one elite at a time in a bash `for` loop.
   Measured live on 2026-07-20: ~280 ms/epoch, ~9-10 minutes/elite,
   ~78 minutes total for all 12 — this dominates the wall-clock cost of the
   whole basin-search-to-gate pipeline by roughly 15x over the search stage.

The 12 elites' training trajectories are mutually independent (different
initial params, same target, same loss). If `SignalTensor.biquad` supported
backward, all 12 could run as one `[12]`-batched Adam trajectory instead of
12 sequential single-lane trajectories — collapsing the serial multiplier the
same way batching collapsed it for forward-only search (forward went from
~200 ms serial to ~14 ms/candidate at B>=32, a ~15-20x amortization of
per-invocation compile/dispatch overhead). The expectation (unverified) is
that refinement would drop from ~78 min toward something close to a single
elite's own serial cost (~9-10 min), since Adam step count per trajectory is
unchanged — only the ×12 sequential multiplier goes away.

This does **not** speed up a single trajectory's own epoch-to-epoch cost
(each Adam step still depends on the previous one) — only the fan-out across
independent trajectories (multiple elites, multiple restarts/seeds trained
concurrently, etc).

## Prior art to build on

The scalar biquad BPTT path is implemented and validated — this is not a
from-scratch gradient derivation, it's an audit of whether that machinery
generalizes to vector-width history cells:

- `docs/BIQUAD_BPTT_GRADIENT_BUG.md` — full history of the scalar fix.
- `Sources/DGen/Blocks/Emission/BPTTEmission.swift` — forward/reverse loop
  split (`wrapWithBPTTLoops`), carry-cell read/write emission inside the
  reverse loop.
- `Sources/DGen/HigherOps.swift:401` (`Graph.biquad`) — the pass-through
  `historyWrite` pattern that keeps writes in the dependency graph so their
  `.backward` rule fires and the reverse-time loop activates. Comment at
  line ~432 explains why dangling writes silently truncate the temporal
  gradient.
- Non-obvious scalar-path findings that must be re-verified, not re-derived,
  for the vector case (from `project_spectral_grad_scale_bug` memory /
  `BIQUAD_BPTT_GRADIENT_BUG.md` "Part B resolution"):
  1. `reverseTopologicalOrder` must treat `historyRead` as a target, or
     chained writes (`write(c0, read(c1))`) get pruned and carries are never
     consumed.
  2. Pre-existing Metal forward race: a scalar `historyWrite` whose input
     comes from a parallel block was emitted in a parallel per-frame kernel
     (all threads racing one cell) — fixed by forcing scalar-cell
     historyRead/Write into `findSequentialNodes` + `findFeedbackLoops`.
     **Check whether vector-width history cells have the same or a
     different race condition** — the tensor biquad's forward path already
     had to solve "one sequential kernel, W parallel lanes" for exactly this
     reason (`docs/TENSOR_BIQUAD_SPEC.md` risk section); the backward loop
     needs the equivalent guarantee.
  3. `HistoryFusionPass` must rewire consumers when deleting pass-through
     writes — verify it still does so correctly when the cell is
     tensor-registered.
  4. Selector backward off-by-one (forward selector is 1-indexed,
     option k ↔ mode==k+1, but backward was gated on mode==k) — this bit
     the scalar cutoff gradient by flipping its sign; the biquad's shelf/mode
     coefficient arithmetic still routes through `selector`/`gswitch`
     (`HigherOps.swift` ~line 460+), now over tensor-shaped mode/gain
     operands per the forward spec's Change 2. Confirm `selector`/`gswitch`
     backward broadcasts correctly for tensor operands — the forward spec
     flagged this as "the most likely hidden cost."
  5. Grad carry cells must be registered in `Graph.persistentCells`
     (`Gradients.swift`, `getGradCarryCell`) — any cell touched via raw
     memoryRead/memoryWrite that's missing from `persistentCells` is a
     recurring aliasing bug class in this codebase (4th occurrence
     documented). Vector-width grad carry cells need the same registration,
     sized for W, not just alloc'd at width 1.

## Scope

### In scope

1. Extend the BPTT reverse-loop machinery (`BPTTEmission.swift`,
   `Gradients.swift`) to handle vector-width (tensor-registered) history/
   carry cells for the biquad's four state cells, for both:
   - shared scalar controls (`SignalTensor.biquad(cutoff: Signal, ...)`)
   - per-element controls (`SignalTensor.biquad(cutoff: SignalTensor, ...)`)
2. Remove the `unsupportedGradientNodes` guard for tensor biquad once the
   above is validated — but keep an explicit fallback error for any biquad
   shape/control combination that isn't covered by the test plan below,
   rather than silently producing wrong gradients. Wrong-but-silent adjoints
   are strictly worse than the current hard failure.
3. Whatever `SignalTensor`/`Trainer`-level plumbing is needed to actually run
   a `[B]`-lane Adam trajectory (batched loss, batched grad readback into
   B independent parameter sets) — check what `Examples/SynthID/Trainer.swift`
   and `BatchBench.swift` already assume about per-lane parameter state.
4. Correctness validation against the existing scalar BPTT ground truth
   (`BPTTBiquadScratchTests.swift` has inline CPU double-adjoint references —
   reuse that pattern per-lane).

### Explicitly out of scope

- Any change to the forward-only tensor biquad's numerics, allocation
  pattern, or the scalar `Signal.biquad` path. Both are validated and must
  stay byte-identical.
- Speeding up a single trajectory's own per-epoch cost (the ~280 ms/epoch
  Adam step itself). This spec is about removing the ×N sequential-elites
  multiplier, not about making one gradient step cheaper.
- Rewriting `refine_elites.sh` / the basin-search CLI to actually use batched
  refinement — that's a follow-up once gradients exist and are validated;
  a minimal repro harness (analogous to `BatchBench.swift`) is enough to
  close this spec.
- Any change to production `subtractive-bass` voice/trainer/policy code,
  which keeps using scalar `Signal.biquad` regardless of this work.

## Test plan

Mirror `docs/TENSOR_BIQUAD_SPEC.md`'s test plan, but for gradients, all
compared against the scalar `Signal.biquad` backward path as ground truth
(same pattern as `BPTTBiquadScratchTests.swift`):

1. **B=1 gradient equivalence**: sine target, `SignalTensor.biquad([1])`
   backward vs `Signal.biquad` backward — cutoff/resonance/gain grads match
   to float precision, all 8 modes.
2. **B=8 shared controls**: one shared cutoff/resonance across 8 lanes with
   distinct inputs/targets; each lane's per-parameter gradient matches its
   independent serial-trained equivalent.
3. **Lane-gradient independence**: perturbing lane 0's target must not change
   lane 1..7's gradients (catches carry-cell/grad-cell aliasing across
   lanes — the vector analogue of the forward spec's "lane-state
   independence" test and of the persistent-cells aliasing bug class).
4. **B=8 per-element controls**: 8 distinct cutoffs/resonances, gradients
   for each lane's own cutoff/resonance match serial per-lane biquad grads
   and do not leak into other lanes' parameters.
5. **Time-varying per-element cutoff** (the actual `subtractive-bass` shape):
   `cutoff_i(t) = fBase_i + fAmt_i * exp(-t / fDecay_i)`; gradient w.r.t.
   `fBase`, `fAmt`, `fDecay` per lane matches serial fdcheck.
6. **Multi-step Adam equivalence** (the actual integration goal): run N Adam
   steps on B independent parameter sets via one batched trajectory vs. B
   independent serial trajectories from identical initial params; final
   parameters and loss curves match within numerical tolerance at each step.
7. **fdcheck**: finite-difference check per lane per parameter, using the
   stability methodology already established (`FDCHECK_FINDING.md`: sweep
   fd-eps, avoid the log-mag eps=1e-8 ill-conditioning, prefer linear-
   magnitude loss for FD stability).
8. **Guard regression**: any biquad backward shape/control combination
   outside the validated set still throws `DGenError.unsupportedGradient`
   (not silently wrong output) — keep this explicit rather than deleting the
   guard machinery wholesale.
9. **Scalar regression**: `swift test --filter Biquad` and
   `BPTTBiquadScratchTests` stay green with zero modifications.

## Acceptance summary

Done when tests 1-9 pass, a minimal batched-refinement repro (comparable to
`BatchBench.swift`'s role for the forward spec) demonstrates B independent
elite-refinement trajectories running as one batched Adam loop with measured
wall-clock time and per-step throughput recorded here, and the scalar and
forward-tensor biquad paths are provably untouched (byte-identical kernels
for their existing test suites). If the measured speedup does not
substantially beat 12x serial (accounting for backward's necessarily higher
per-step cost than forward-only), record the actual number and revisit
whether this is worth wiring into `refine_elites.sh` at all — the estimate
in this spec is directional, not a committed target.

## Implementation results (2026-07-20)

### What was built

The scalar BPTT machinery did NOT generalize in place: the same-block
`wrapWithBPTTLoops` path re-emits backward UOps inside a reverse loop of the
same kernel, where tensor element/frame index variables and cached tensor
registers from the forward loop are out of scope. Instead, the vector-width
backward is routed through the detached-BPTT layout unconditionally:

1. **Vector-width grad carry cells** (`Gradients.swift getGradCarryCell`):
   history cells with a tensor registration get `alloc(vectorWidth: W)` carry
   cells, tensor-registered themselves and tracked in a new
   `Graph.tensorGradCarryCells` set. Every new branch below gates on that set,
   so scalar graphs are provably untouched (byte-equivalent kernels verified
   after every stage; note the scalar carry-write emission order is
   nondeterministic run-to-run — pre-existing — so the check is
   order-insensitive line comparison).
2. **Shape inference + emission** for carry `memoryRead`/`memoryWrite`
   (`ShapeInference.swift`, `TensorOutputBindingPass.swift`,
   `Emit+State.swift`): tensor-shaped, pass-through like historyWrite,
   per-element tload/tstore addressing.
3. **Recurrence consolidation** (`BlockFormation.swift
   consolidateTensorBPTTBackwardBlocks` + `tensorBPTTRecurrenceClosure`,
   called from `CompilationPipeline.buildInitialBlocks`): the reverse-time
   recurrence — carry writes' backward ancestors, carry reads, and their
   boundary-stopped descendants — is extracted into ONE sequential block,
   inserted after every block producing one of its dependencies; consumers of
   its outputs (and of deferred backward leftovers from tensor-history
   forward blocks) are moved after it. Closure boundaries: reductions
   (`sum`/`sumAxis`), grad accumulates, and all isolated-pass spectral ops
   (upstream grads are materialized per-frame BEFORE the recurrence and read
   from frame-aware cells inside the reverse loop). The closure walk follows
   only value edges through `seq` nodes — traversing seq ORDERING edges pulls
   arbitrary side-effect chains into the recurrence nondeterministically.
4. **Detached reverse wrap for shape-aware emission** (`BPTTEmission.swift`,
   `BlockEmission.swift`): `blockIsDetachedBPTTBackward` now also applies to
   shape-aware-emitted bodies; out-of-block reads of vector carry cells throw
   `DGenError.unsupportedGradient` instead of silently truncating; Metal
   `setFrameIndex` declares `_frameIndex` once per kernel and reassigns after.
5. **Scheduling** (`FeedbackAnalysis.swift`): vector carry reads/writes are
   frame-serial (same race class as scalar history), and the recurrence
   closure is marked scalar so block formation keeps it together.
6. **Broadcast reduction fix** (`Gradients.swift reduceBroadcastGradient`):
   a tensor-shaped gradient flowing into a scalar operand now reduces via
   `sum` over lanes (previously passed through unreduced — wrong-but-silent
   for shared scalar controls).
7. **Grad-cell aliasing fix** (`GradientSetup.swift`): gradient accumulation
   cells are registered in `persistentCells` when vector carry cells exist —
   block reordering otherwise lets buffer-reuse alias them onto live forward
   tensors (5th occurrence of this bug class).
8. **Guard relaxation** (`Functions.swift`): rank-1 `[B]` element shapes
   (both shared-scalar and per-element control overloads) support backward;
   rank>1 still throws `DGenError.unsupportedGradient` with a precise message.

### Test evidence (Tests/DGenLazyTests/TensorBiquadGradientTests.swift)

All 9 spec tests pass:
1. B=1 equivalence vs scalar backward, all 8 modes (5e-3 rel tolerance).
2. B=8 shared controls: shared cutoff grad == Σ per-lane serial grads (1e-2).
3. Lane-gradient independence: perturbing lane 0's target leaves lanes 1..7
   grads bit-identical.
4. B=8 per-element controls: per-lane cutoff grads match serial (1e-2).
5. Time-varying per-element cutoff (fBase/fAmt/fDecay per lane) match serial.
6. Multi-step Adam equivalence: see BatchTrainBench below.
7. Per-lane fdcheck, eps sweep [1e-3, 3e-3, 1e-2], linear-magnitude loss.
8. Guard regression: rank-2 [2,4] backward still throws.
9. Scalar regression: Biquad + BPTTBiquadScratch suites green with zero test
   modifications; scalar BPTT kernels byte-equivalent to the pre-change
   baseline (order-insensitive).

Single-backward gradient probe (`batch-train-bench --probe-grads`, spectral
loss, 5 params × B lanes): worst |batched − single-lane| rel diff 1.0e-5 at
B=2/4, 1.9e-5 at B=12.

### Batched-refinement repro (Examples/SynthID/BatchTrainBench.swift)

`swift run SynthID batch-train-bench --batch 12 --steps 20 --frames 8192
--lr 0.005 --mode equivalence` — minimal voice (sine → time-varying-cutoff
biquad → gain), 5 log-space `[12]` `Tensor` params, batched spectral loss
(rescaled mean→sum so per-lane grads are batch-size-invariant), stock Adam
(tensor branch is already per-element):

- Equivalence: worst per-step per-lane parameter rel diff **3.6e-6** across
  20 Adam steps vs 12 independent single-lane trajectories; batched loss
  equals 12 × mean of serial losses.
- Timing: batched **0.737 s/step** vs **0.882 s/step** summed serial —
  **1.2x**, far below the directional ~12x estimate.

### Why the speedup is only ~1.2x (and what would fix it)

The consolidated backward runs as ONE single-threaded kernel iterating
`frames × B` (reverse frame loop with an inner element loop), so backward
cost still scales linearly with B — batching currently amortizes only the
parallel parts (loss, upstream spectral grads, accumulates) and per-dispatch
overhead. The forward-only search won its ~15-20x because its per-lane work
ran in parallel threads. To get a comparable win for refinement, the reverse
loop needs "sequential frames, parallel lanes" nested-loop execution (one
threadgroup with W lanes advancing frame-by-frame) — the same TODO already
noted for tensor feedback in `FeedbackAnalysis.swift`. Until then, wiring
batched refinement into `refine_elites.sh` is not worth it; the machinery is
correct and validated, and the performance work is a follow-up.

The follow-up is specced in `docs/TENSOR_BIQUAD_PARALLEL_LANES_SPEC.md`:
relax `StatefulTensorParallelPolicy`'s tensor-history bail-out under a narrow
per-lane-safety predicate and reuse the existing `.fixedWithFrameLoop(W)`
dispatch (thread id = lane) for both the forward biquad block and the
consolidated reverse block.
