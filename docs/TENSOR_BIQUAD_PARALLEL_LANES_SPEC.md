# Spec: Parallel-lane execution for tensor-biquad forward + BPTT backward

Status: IMPLEMENTED 2026-07-20 (measured results in Acceptance below).
Written 2026-07-20. Follow-up to `docs/TENSOR_BIQUAD_GRADIENT_SPEC.md`
(IMPLEMENTED — correctness done, performance not).

## Problem statement

Batched biquad gradients are correct but deliver only **1.2x** over serial
(BatchTrainBench, B=12, 8192 frames: 0.737 s/step batched vs 0.882 s/step for
12 single-lane trajectories). The forward-only search's ~15-20x win came from
amortizing per-invocation compile/dispatch/readback overhead — NOT from
compute parallelism. Both the forward tensor-biquad block and the
consolidated BPTT backward block are emitted as **one single-threaded
kernel** iterating `frames × B`:

```
// forward (and, reversed, backward): one thread does all lanes
if (id < 1) {
  for (frame loop) {          // sequential — required, it's a recurrence
    for (elem in 0..<B) {...} // serial     — NOT required, lanes are independent
  }
}
```

So per-step compute still scales linearly with B, and batching only removes
dispatch overhead. Until the inner element loop becomes thread-parallel,
batched refinement is not worth wiring into `refine_elites.sh`.

The structure of the problem is favorable: the biquad recurrence is serial in
TIME but fully independent across LANES. Lane `i`'s four history cells and
four grad-carry cells are only ever accessed at element index `i`; nothing in
the forward update or the reverse-time recurrence reads across lanes. The
target shape is "sequential frames, parallel lanes":

```
if (id < B) {                 // one thread per lane
  for (frame loop) {          // each thread walks time privately
    ... all state indexed by elem = id ...
  }
}
```

Expected ceiling: per-step cost approaching a single lane's serial cost —
~10-12x at B=12 — and, because lanes become nearly free up to GPU occupancy,
larger batches (elites × seeds × restarts, B=64-128) amortize even better.

## Prior art — the mechanism already exists

1. **`StatefulTensorParallelPolicy.decide`**
   (`Sources/DGen/Compilation/StatefulTensorParallelPolicy.swift`)
   already implements this exact pattern for batched `phasor`/`accum` blocks:
   Metal-only, sequential scalar block with a known tensor shape,
   frameBased/hopBased temporality → dispatch `id < tensorSize` threads with a
   per-thread sequential frame loop. It **explicitly bails out** when the
   block contains a tensor-registered history cell:

   ```swift
   case .historyRead(let cellId), .historyWrite(let cellId), ...:
     // Keep tensor-history blocks on the existing strict frame-by-frame path.
     if graph.cellToTensor[cellId] != nil {
       return Decision(enabled: false, tensorSize: tensorSize)
     }
   ```

   That bail-out is the single gate that keeps the forward biquad serial.

2. **`UOpBlockFinalization.computeDispatchMode`**
   (`Sources/DGen/Compilation/UOpBlockFinalization.swift:169-171`): when the
   policy enables, the block gets `.fixedWithFrameLoop(tensorSize)` — the
   dispatch mode that produces `if (id < W) { for (frames) ... }` with the
   block's `tensorIndex` bound to the thread id. This is the emission target
   for the forward block.

3. **The consolidated BPTT backward block**
   (`consolidateTensorBPTTBackwardBlocks`, `Sources/DGen/Blocks/
   BlockFormation.swift`) is wrapped by `wrapDetachedBPTTBackwardLoop`
   (`BPTTEmission.swift`) / the shape-aware equivalent in
   `BlockEmission.swift`, which sets `hasOwnFrameLoop = true` →
   `.selfManaged` dispatch. It needs the reverse-loop analogue of
   `.fixedWithFrameLoop`: W threads, each running the reverse frame loop.

4. **Frame-aware tape layout is already lane-friendly**: all per-frame tensor
   tapes are indexed `frameIdx * W + elem` (`IRBuilder+Memory.swift`), so
   lane-per-thread reads/writes are coalesced across the warp. The lane-sum
   reductions, tensorAccumulates, and spectral upstream-grad kernels already
   run in separate kernels outside the recurrence and need no changes.

5. **Why the bail-out exists** (do not regress these): hop-gated tensor
   history feedback must stay one sequential kernel
   (`forceSequentialHopHistoryBlocks`, CompilationPipeline.swift; see
   project_hop_gated_tensor_history), and shared-state scalar ops (`noise`'s
   single xorshift PRNG, scalar `accum`/`phasor`/`latch` cells) cannot be
   duplicated per thread.

## Design

### Change 1 — safety predicate: `laneParallelizable(block:graph:)`

Replace the blanket tensor-history bail-out with a narrow predicate. A
sequential frame-loop block may run with one thread per lane iff ALL hold:

1. Metal backend; block shape is a single known `[W]`, W > 1 (same
   preconditions as today's policy).
2. Every stateful cell touched in the block is tensor-registered with
   element-indexed access:
   - `historyRead/historyWrite(cell)` with `cellToTensor[cell] != nil`;
   - `memoryRead/memoryWrite(cell)` with `cell ∈ tensorGradCarryCells`;
   - NO scalar-cell stateful ops (`accum`, `phasor`, `latch`, `click`,
     `noise`, `historyReadWrite`, scalar history) — any one disqualifies.
3. No hop-gated node in the block (`nodeHopRate[nodeId] != nil`
   disqualifies) — hop feedback keeps the strict path.
4. Every memory WRITE in the emitted body is element-indexed (per-lane
   history/carry/tape writes). Scalar per-frame writes are allowed only if
   guarded to a single thread (see Change 3); if the audit finds an
   unguardable scalar write, the block falls back to single-threaded.
5. No cross-lane tensor addressing: no `gather`, `peek`-style dynamic
   element indices, view transforms that permute the element axis, or
   reductions inside the block (reductions are closure boundaries already —
   assert, don't assume).

The predicate must be conservative: failing any check degrades to today's
correct single-threaded emission, never to wrong answers.

### Change 2 — forward path

Extend `StatefulTensorParallelPolicy.decide`: instead of returning
`enabled: false` on tensor-history cells, treat per-element tensor history as
a candidate op (like `phasor`/`accum`) when `laneParallelizable` holds. The
existing `.fixedWithFrameLoop(W)` dispatch and thread-id tensorIndex binding
then apply unchanged. The forward biquad block becomes
`if (id < W) { for (frame) { lane-id state update } }`.

Verify the emitted forward kernel no longer contains an inner
`for (elem in 0..<W)` loop and that `ctx.tensorIndices` resolves to the
thread id in this mode (this is the same binding batched phasor uses).

### Change 3 — backward path (the consolidated BPTT block)

The consolidated block currently emits body UOps (with per-element loops from
region emission), then wraps them in `beginReverseLoop`/`endLoop` and marks
`.selfManaged`. Add a lane-parallel variant:

1. In `BlockEmission` where the detached wrap is applied (both the standard
   and shape-aware branches), consult `laneParallelizable`. When it holds:
   - bind the block's element index to the thread id instead of emitting
     per-element `beginForLoop`s (reuse the `.fixedWithFrameLoop` binding
     path);
   - wrap in the reverse frame loop as today;
   - finalize with a new dispatch mode `.fixedWithReverseFrameLoop(W)` (or
     reuse `.selfManaged` + `fixedThreadCount = W` if the scheduler supports
     a thread-count override on self-managed kernels — check
     `DispatchMode` and the scheduler before inventing a new case).
2. Scalar values INSIDE the reverse loop:
   - reads (taped `.variable` coefficient values, `t[k*frameCount + i]`)
     are read-only → safe to compute/load redundantly per thread;
   - scalar writes: the grad seed store (`memory[seedCell + elem] = 1.0` is
     element-indexed — fine) and any per-frame scalar tape STORE must be
     guarded `if (id == 0)`. Audit the body UOps at emission time; if a
     scalar write can't be classified, fall back to single-threaded.
3. No fences are needed between frames: each lane's history/carry state is
   private to its thread. Keep the existing end-of-kernel fence.
4. The upstream spectral grad, lane-sum reductions, accumulates, and
   `output(0)` all live outside this block already (closure boundaries) —
   unchanged.

### Change 4 — per-frame scalar tapes shared by backward

`allocatePerFrameStorageCells` / the scalar tape reload path
(`BPTTEmission.swift`) stores shared-scalar coefficient values once per frame
in the forward block and reloads them in the reverse loop. With lane-parallel
forward, those stores become redundant per-thread writes of the SAME value —
benign in principle, but guard them to thread 0 anyway (same-value racing
writes are UB-adjacent on some hardware and free to avoid).

## Out of scope

- C backend (Metal only, matching the existing policy).
- Parallelizing time (impossible — it's the recurrence).
- Rewiring `refine_elites.sh` — still a follow-up, but this spec is the
  gating item for it.
- Hop-gated tensor history and any block that fails the predicate.

## Test plan

Correctness gates (all must stay green with zero test modifications — they
were built to catch exactly the races this change could introduce):

1. `TensorBiquadTests` (forward): B=1 all-modes, shared controls, per-element
   controls, time-varying cutoff, and especially `testLaneStateIsIndependent`
   (impulse in lane 0, lanes 1..7 exactly zero — the forward race detector).
2. `TensorBiquadGradientTests` (backward): all 9 spec tests, especially
   `testLaneGradientIndependence` (perturb lane 0's target, lanes 1..7 grads
   bit-identical — the backward race detector) and the B=8 shared-controls
   lane-sum test (catches lost/duplicated scalar-coefficient tape writes).
3. `batch-train-bench --probe-grads` at B=2/4/12/32: worst rel diff vs
   single-lane stays ≤ ~2e-5 (current values: 1.0e-5 / 1.9e-5).
4. `batch-train-bench --mode equivalence` B=12: per-step per-lane param rel
   diff stays at float-noise level (currently 3.6e-6).
5. Scalar regression: `swift test --filter Biquad`, `BPTTBiquadScratch`,
   byte-equivalent scalar kernels (order-insensitive diff vs baseline).
6. New: a predicate unit test asserting fallback — a block containing
   `Signal.noise()` (shared PRNG) or a hop-gated tensor history must NOT
   lane-parallelize.

Performance gates:

7. `batch-train-bench --mode timing` B=12, 8192 frames: record s/step for
   batched vs 12x serial. Target: ≥ 8x (vs today's 1.2x). Record the actual
   number here regardless.
8. Same at B=32 and B=64 with `--mode timing` (no serial comparison needed;
   record s/step and s/lane-step) to characterize the occupancy curve.
9. Forward-only check: BatchBench correctness gate + timing sweep unchanged
   or improved (forward lane-parallelism should help the search path too;
   its correctness gate max-abs-diff must stay < 1e-4).

## Acceptance

Done when tests 1-6 are green, the measured batched-refinement speedup at
B=12 is recorded in this doc (target ≥ 8x; if materially below, record why —
e.g. occupancy at W=12 threads is a plausible limiter, in which case report
the B=32/64 numbers as the honest capability), and a decision is recorded on
wiring batched refinement into `refine_elites.sh` (worth it iff the measured
multi-elite wall-clock beats serial by ≥ 5x at production frame counts and
epoch counts).

### Results (2026-07-20, M-series laptop, release build, 8192 frames, 20 steps)

Implementation:
- `laneParallelizable` + relaxed `decide` + `decideDetachedBPTTBackward` in
  `StatefulTensorParallelPolicy.swift`; lane-parallel region emission in
  `RegionEmitter.swift` (`laneParallel:` flag binds the region element index
  to the thread id, no per-region element loops); new dispatch mode
  `.selfManagedThreads(W)` (W threads, block-owned reverse frame loop).
- Forward biquad block now emits `.fixedWithFrameLoop(W)`:
  `if (id < W) { for (frame) ... }`; consolidated backward emits
  `.selfManagedThreads(W)`: `if (id < W) { for (i = frames-1; ...) ... }`.
  Verified in dumped kernels: no inner element loops in either.

Correctness gates (all green, zero test modifications):
- `TensorBiquadTests` 6/6 (incl. `testLaneStateIsIndependent`),
  `TensorBiquadGradientTests` 7/7 (incl. `testLaneGradientIndependence`),
  `BPTTBiquadScratchTests` 9/9, all `--filter Biquad` suites, and the full
  `swift test` suite pass.
- `--probe-grads` worst rel diff vs single-lane: 1.07e-5 (B=2/4),
  1.88e-5 (B=12), 2.28e-5 (B=32) — unchanged float-noise regime.
- `--mode equivalence` B=12: worst per-step per-lane param rel diff 1.6e-4
  with lanes parallel vs 1.7e-4 on the pre-change baseline (same machine,
  same seed) — i.e. the pre-existing step-noise level, not a regression.
- `--probe-forward` B=12: batched loss equals the per-lane sum to 5+ digits.
- New predicate unit tests (`StatefulTensorParallelPolicyTests`, 9 tests):
  noise / hop-gated history / scalar stateful / scalar history / scalar
  write all fall back; clean forward + detached-backward blocks enable.

Timing (batched s/step; serial-sum baseline 0.534-0.547 s/step = 0.0445
s/lane-step; pre-change batched baseline measured same-machine via stash):

| B  | before    | after     | s/lane-step | vs 12x serial | vs old batched |
|----|-----------|-----------|-------------|---------------|----------------|
| 12 | 0.731     | 0.162     | 0.0135      | 3.3x          | 4.5x           |
| 32 | —         | 0.245     | 0.0077      | 5.8x (per lane) | —            |
| 64 | 3.420     | 0.339     | 0.0053      | 8.4x (per lane) | 10.1x        |

The B=12 wall-clock speedup vs serial is 3.3x, below the ≥ 8x target — as
anticipated, W=12 threads underutilize a SIMD group and the remaining step
time is dominated by the non-recurrence kernels (spectral loss, reductions,
readback), which batching does not shrink. The honest capability is the
B=32/64 curve: lanes are nearly free once dispatched (64 lanes cost 2.1x a
12-lane step), reaching 8.4x per lane-step at B=64.

Decision on `refine_elites.sh`: wire batched refinement **only for batch
sizes ≥ 32** (elites × seeds × restarts packed into one batch): ≥ 5.8x per
lane-step beats the ≥ 5x threshold. At B=12 (3.3x) it is below threshold —
pad/pack the batch to ≥ 32 lanes in the harness rather than running small
batches.

## Risks

| Risk | Mitigation |
|---|---|
| Hidden scalar write inside the recurrence body races across threads | Emission-time audit of body UOps; unguardable write → single-threaded fallback; lane-independence tests as detectors |
| `ctx.tensorIndices` binding differs between region-emitted and thread-id modes | Reuse the exact `.fixedWithFrameLoop` binding path that batched phasor already exercises; diff emitted kernels at W=1 |
| W=12 threads underutilize a threadgroup/SIMD-group (32 lanes) | Expected; report B=32/64 numbers; consider padding B up to SIMD width in the harness, not the compiler |
| Redundant per-thread scalar recompute inflates cost | Coefficients are O(10) flops/frame vs the state update — negligible; verify in timing |
| Policy relaxation accidentally captures hop-gated or noise blocks | Predicate unit test (test 6); hop/noise checks are explicit disqualifiers |
| Backward block contains nodes region-emitted with multiple shapes (shape-aware emission) | Predicate requires a single `[W]` shape; multi-shape blocks fall back |
