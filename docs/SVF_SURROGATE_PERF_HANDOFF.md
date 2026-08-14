# Handoff: make the SVF frequency-sampled surrogate fast (43 s → ~2 s/epoch)

> **STATUS 2026-08-14: DONE.** 42.9 s → **2.30 s/epoch** gpu_execute
> (goal was ≤ ~2.5 s), epoch-1 loss/params bit-identical to the pre-fix
> baseline, Metal fdcheck 0.99990 unchanged, regression filter identical to
> baseline. See "Resolution" at the end of this document for what was done
> and what remains.

Audience: a fresh implementer with no context on this effort. This document
contains everything learned the hard way on 2026-08-13. Companion spec (design
+ math + what's already landed): `docs/SVF_FREQ_SURROGATE_SPEC.md`.

## Mission in one paragraph

`dgenlisp train` replaces the patch's `(svf ...)` filters with a
differentiable STFT surrogate (`svf-freq`: buffer(1024, hop 256) → FFT →
per-bin |H(ω; cutoff, q)| mask → IFFT → overlapAdd) so training avoids
sample-serial BPTT through the filters. This is IMPLEMENTED and CORRECT
(Metal autograd/FD ratio 0.99990) and the monologue benchmark runs end to
end — but at **42.9 s/epoch GPU vs the 1.5 s/epoch BPTT baseline** it
defeats its own purpose. Profiling + reading the generated kernels shows the
time is NOT inherent: hop-rate tensor work is being emitted inside
single-threaded frame-loop kernels, gated by an in-loop `select` that zeroes
values instead of skipping work — i.e. 256× redundant AND serial. The
compiler already emits the correct shape for a sibling region (see "the
target shape" below), so this is block-formation/emission surgery, not new
machinery. Expected result: ~1.5–2.5 s/epoch, at which point the surrogate
beats BPTT and unblocks rung-C validation (synth recovery + real-target
parity).

## State of the branch (dgenlisp-train, all UNCOMMITTED as of writing)

New files:
- `Sources/DGenLazy/FilterSurrogates.swift` — `svfFrequencySampled(...)`.
  Controls go through `hopHold(hop:)` (critical — see Fix 3 below).
- `Sources/DGenLisp/FilterSurrogateLowering.swift` — AST pass rewriting
  `(svf ...)` heads → `(svf-freq ... @window N @hop H)` in
  `TrainPlanner.makePlan` (same pattern as `ExcitationLowering`). The
  patch's `defmacro svf` stays but goes dead. Render/checkpoint subprocesses
  evaluate the unswapped source → audio always uses the real SVF.
- `Tests/DGenLazyTests/SVFFreqSurrogateTests.swift` — COLA/identity test
  (allpass ≡ input delayed N−1, err < 1%) + Metal fdcheck (cutoff grad vs
  central differences; ratio must stay ≈ 1). The C fdcheck is `XCTSkip`
  (pre-existing C-backend spectral-BPTT tape codegen gap — undeclared
  `tape` identifiers; not ours to fix here).
- `Tests/DGenLispTests/FilterSurrogateTests.swift` — lisp op parity +
  lowering snapshot.

Modified (the three fixes that made it run at all — understand these before
touching scheduling):

1. **Hop-sliced frame-aware storage.** Without it the Metal `memory` buffer
   request was ~300 GB (every FFT/mask intermediate, forward + backward
   tape, allocated `tensorSize × 38072 frames`).
   - `Sources/DGen/DGen.swift`: `Graph.frameAwareCellHops: [CellID: Int]` —
     hop of hop-based frame-aware cells; absent ⇒ hop 1.
   - `Sources/DGen/Compilation/Passes/TensorMemoryMaterializationPass.swift`:
     hop-based cells allocate `tensorSize × ceil(frames/hop)` slots and
     register the hop. `DGEN_DEBUG_TENSOR_ALLOC=1` prints per-cell decisions
     incl. `hop=`.
   - `Sources/DGen/IRBuilder+Memory.swift`: `frameAwareOffset` divides
     `frameIdx / hop` for hop-sliced cells. All four
     `frameAwareTensorRead/Write` variants route through it.
   - `Sources/DGen/Compilation/DGenArtifactCache.swift`: fingerprint
     includes the new map.
2. **Hop-gated backward.** The backward was frame-based zero-padding by
   design (`overlapAddGradGather` wrote real grads at hop frames, zeros at
   the other 255, into full per-frame tapes; every downstream backward node
   inherited frame-based temporality).
   - `Sources/DGen/Gradients.swift` (`case .overlapAdd`): gather's grad cell
     is hop-slot allocated; `gatherOp` and the sequenced grad tensorRef are
     tagged `g.nodeHopRate[...] = (hop, counterNode)` found via
     `findUpstreamHopRate` (BFS up from the tensor input to the bufferView
     seq node, which carries the tag). This is what flips the whole backward
     tensor chain to hop-based in TemporalityPass.
   - `Sources/DGen/Gradients.swift` (`case .seq`, bufferView backward): the
     seq node carries `nodeHopRate` directly; tape hop-sliced the same way.
   - `Sources/DGen/Emit+FFT.swift`: gather + `bufferViewGradStore` write
     `(frame/hop)*win + elem`; `bufferViewGradRead` walks the hop grid
     (≤ N/hop+1 windows per sample instead of N).
   - `Sources/DGen/Compilation/Passes/TemporalityPass.swift`:
     `.bufferViewGradRead` added to `isIntrinsicallyFrameBased` — it emits
     the per-sample grad signal and must NEVER be promoted to hop rate
     (propagation would otherwise promote it because its input, the store,
     is hop-based).
3. **Hop-held mask controls.** cutoff/q are frame-rate signals (envelopes),
   and TemporalityPass's contamination rule (any frame-based input ⇒
   frame-based) demoted ALL mask math — forward and backward — to per-frame
   (this was invisible until measured: 896 frame-rate frame-aware cells).
   `svfFrequencySampled` runs `cutoff.clip(...)` and `max(q, ...)` through
   the existing `Signal.hopHold(hop:)` (`Sources/DGen/HigherOps+HopHold.swift`
   creates a wrapping accum counter, tags the trigger with `nodeHopRate`,
   and `graph.latch` propagates the tag). After this: **0** frame-rate
   frame-aware cells, 4,284 hop-sliced, ~2.6 GB, training runs.

Also added (keep): env-gated instrumentation in
`Sources/DGenLisp/DirectionTrainer.swift` + `Sources/DGenLazy/Realize.swift`:
- `DGENLISP_TRAIN_PROFILE=1` → per-kernel GPU ms table after epoch 1
  (`LazyGraph.profileGPU`).
- `DGENLISP_TRAIN_KERNEL_DUMP=<dir>` → writes every kernel's source to
  `<dir>/kernel_<idx>.txt` (`LazyGraph.dumpKernelSources`).

## Reproduce everything

```sh
swift build -c release --product DGenLisp

# The benchmark (surrogate is DEFAULT; --filter-surrogate none = old BPTT path)
DGENLISP_TRAIN_TIMING=1 .build/release/DGenLisp train \
  --patch benchmarks/train_monologue/patch.lisp \
  --target Assets/monologue-bass.wav \
  --seed-params benchmarks/train_monologue/seed.json \
  --job-dir /tmp/train-surrogate --epochs 3 --checkpoint-every 999

# Profile + kernel dump (1 epoch is enough; expect gpu_execute ≈ 43 s today)
DGENLISP_TRAIN_PROFILE=1 DGENLISP_TRAIN_KERNEL_DUMP=/tmp/kernels \
  .build/release/DGenLisp train ... --epochs 1 ...

# Correctness gates
swift test --filter SVFFreqSurrogateTests    # COLA identity + Metal fdcheck
swift test --filter FilterSurrogateTests     # lisp parity + lowering snapshot

# Allocation audit (per-cell decisions incl. hop classification)
DGEN_DEBUG_TENSOR_ALLOC=1 .build/release/DGenLisp train ... 2>&1 | grep TENSOR-ALLOC
# Other useful: DGEN_DEBUG_BLOCKS, DGEN_DEBUG_TEMPORALITY, DGEN_DEBUG_REGIONS
```

Baseline numbers (M-series, crop 38072, N=1024, hop=256, 149 hops):
BPTT path ~1.5 s/epoch; surrogate today ~42.9 s gpu_execute + ~1.9 s
dgen_compile per epoch.

## The diagnosis (from reading the dumped kernels)

Profile of one epoch (453 kernels total; idx from one representative run —
indices shift when the graph changes, identify kernels by content):

| kernel | ms | dispatch | content |
|---|---|---|---|
| 128 | 25095 (58%) | perFrameScaled | mask-backward math (sqrt/div chains) |
| 137, 143 | 4816, 4810 | singleThreaded | overlapAdd backward + hop-row rewindow ×2 paths |
| 131 | 2998 | singleThreaded | scalar grad routing + trailing 1024 scatter |
| 149 | 2122 | singleThreaded | hop-row copy + gather + carry-cell writes |
| 80 | 1487 | singleThreaded | FORWARD synth scalars + whole forward STFT |
| 141, 135 | ~420 each | singleThreaded | per-frame 1024-row reduce → filter param grads |
| 76 | 236 | singleThreaded | pure scalar frame loop — the healthy serial floor |

Three structural failures:

1. **kernel_128 — missing hop guard + single thread.** Despite
   `dispatchMode=perFrameScaled`, the body is `if (id < 1)` and thread 0
   runs a flattened loop `for i < frameCount*1024` (~39M iterations). Every
   index inside is `(_frameIndex/256)*1024 + elem` — only 149 distinct hop
   rows exist, so each row is recomputed identically 256×, serially.
   Necessary work is 149×1024 ≈ 152K element-ops. Three per-frame scalars
   (a cos + two comparisons writing `t[463..465]`) are fused into this
   block; they are presumably what dragged the tensor math into the
   sequential shape.
2. **kernels 137/143/131/149/141/135 — 1024-element hop-row loops inside
   serial frame loops.** Each has a legitimately sequential scalar core
   (overlapAdd 5-window gather, BPTT carry-cell writes) but drags
   1024-element hop-row copies/scatters/reduces through all 38072 frames.
   The hop condition appears only as a data select INSIDE the loop:
   ```c
   float t41936 = t41935 == 0.0;               // frame % 256 == 0
   for (uint t41937 = 0; t41937 < 1024; t41937++) {
     float t41947 = metal::select(0.0, t41946, t41936 > 0.0);
     memory[34980928 + t41950] = t41947;       // loop runs every frame anyway
   }
   ```
   ~78M single-thread iterations per kernel, ~99.9% of it the hop-row loops.
   Some of the copies are hop-row → hop-row aliases that may be eliminable
   outright.
3. **kernel_80 — forward STFT correctly gated but serial.** The forward
   hop work IS inside `if (t[135*frameCount + i] == 0.0) { ... }`, so it
   runs 149×, but on one thread (fused with the recurrent synth scalars /
   ring-buffer writes that genuinely force frame-sequential execution).
   ~5M serial iterations.

## The target shape (proof the compiler can do it)

The FFT-**backward** butterfly kernels (kernel_132 and siblings 136/138/142/
144/148 in that dump — the 130–160 KB ones) are already emitted correctly
and do NOT appear in the hot list:

```c
if (id >= 0 && id < (uint)(frameCount)) {
    if (t[169*frameCount + id] == 0.0) {      // hop guard OUTSIDE the loops
        for (uint t34514 = 0; t34514 < 1024; t34514++) { ... }
```

Per-frame threads, hop guard outside the element loops, so only 149 of
38072 threads do work. These are exactly the nodes that got hop-classified
via the `nodeHopRate` tagging in Fix 2 — meaning hop classification arrives
correctly at emission; the failure is in which BLOCK the nodes land in and
how mixed blocks are rendered.

## The work: make the fused cases emit the target shape

The core question to answer in the code: **why do some hop-classified
tensor regions land in sequential scalar blocks while the FFT butterflies
get their own frame-thread blocks?** Start in:

- `Sources/DGen/Blocks.swift` / block formation (`determineTensorBlocks`,
  `splitReduceBlocks`, `isReductionOp`) — where nodes are grouped into
  blocks and where reduce ops get kernel boundaries.
- `Sources/DGen/Blocks/Emission/ShapeTransitionPlanner.swift` — regions +
  `hopCounter` (looked up from `ctx.hopBasedNodes[nodeId]`, line ~619).
- `Sources/DGen/Blocks/Emission/RegionEmitter.swift` — emits
  `beginHopCheck(counter)` around regions; note it currently wraps the
  region even in sequential kernels, but selects/gating degenerate to the
  in-loop pattern seen above when the region is inside a frame loop.
- `Sources/DGen/Compilation/HopIslandPass.swift` — groups adjacent
  hop-domain blocks into "islands"; understand its `isIslandEligible` /
  `isFrameCarrierEligible` rules, since it may be what merges hop tensor
  work with frame-scalar carriers (`hop gate -> latch` is explicitly kept
  in the same frame loop).
- `Sources/DGen/MetalRuntime.swift` `dispatchMode` /
  `threadCount(frameCount:)` — why kernel_128 dispatches perFrameScaled but
  bodies `if (id < 1)` (a mismatch worth understanding early; the body, not
  the dispatch, is authoritative for what executes).

Suggested order of attack (biggest win first, each independently testable):

1. **kernel_128 (25 s):** find why the mask-backward tensor nodes fused with
   the 3 per-frame scalars into a sequential block. Split: scalars → normal
   per-frame kernel; tensor math → hop-gated element-parallel kernel (the
   butterfly shape, or 149×1024 flat threads). This alone should take the
   epoch to ~18 s.
2. **The hop-row loops in serial kernels (~15 s):** hoist each 1024-element
   copy/scatter/reduce out of the frame-sequential blocks into hop-gated
   parallel kernels, guard OUTSIDE the loop. The serial remainder
   (5-gather, carry cells, scalar chains) is the kernel_76 floor
   (~0.2–0.3 s each). Check whether the pure hop-row → hop-row copies are
   aliases that buffer-reuse/copy-elision can delete instead.
3. **kernel_80 forward STFT (1.5 s, optional):** split the hop-gated FFT
   stages out of the recurrent scalar kernel and dispatch over hops ×
   elements. Only worth it after 1–2.

## Footguns discovered this session (do not relearn these)

- **Slot-clobber hazard:** hop-sliced cells share one slot per hop. A
  hop-row WRITE that executes on non-hop frames (even writing "zero" via
  select) CLOBBERS the hop frame's value — the old full-frame layout was
  robust to this, the sliced layout is not. Any region you re-schedule must
  either be genuinely gated (guard outside) or not write at all off-hop.
  The Metal fdcheck test is the tripwire; run it after every scheduling
  change.
- **Int `min`/`max`/`select` in Metal:** mixing an int-typed var with a
  constant-folded float literal produces `metal::min(int, 1023.0)` →
  ambiguous overload → MTLLibrary compile error. Renderer idiom: do index
  math in float, `cast(..., to: .int)` only at the final memory index.
  (Constant folding turns `intConstant(a) - intConstant(b)` into a float
  literal — that's how it bites.)
- **Temporality contamination:** any frame-based input demotes a node to
  frame-based BEFORE hop propagation runs. If a hop chain goes per-frame
  mysteriously, look for an untagged frame-rate input (this was the mask
  controls). `nodeHopRate` tags on the bridge node are the escape hatch;
  `hopHold` is the packaged version for scalar signals.
- **Per-frame `t` globals are the zero-padding safety net:** hop-gated
  scalar writes into `t[slot*frameCount + i]` leave non-hop frames at their
  memset-zero value, which is exactly what the cross-frame gradient
  reduce needs. Don't replace per-frame globals with single cells for
  hop-gated values consumed by frame-domain reduces (stale-hold ≠ zero).
- **`bufferViewGradRead` must stay frame-based** (it's in
  `isIntrinsicallyFrameBased` now). Same reasoning applies to anything that
  emits a per-sample signal from hop-tape inputs.
- **Kernel hashes alternate between two values across epochs**
  (`full_cache_hit=0` in the timing lines), so every epoch pays ~1.9 s
  recompile. Likely nondeterministic ids from the hopHold counter cells
  across graph rebuilds. Separate fix, big win at 500-epoch scale — see
  `docs/TRAIN_EPOCH_CACHE_SPEC.md` for the caching machinery.
- **Test flakiness that is NOT yours:** `OptimizerTests`,
  `PhaseVocoderTests`, `SpectralLossOrderingScratchTests`,
  `SVFBPTTScratchTests` fail order-dependently in combined `swift test`
  filters and pass in isolation — reproduced identically on the baseline
  without any of these changes. Attribute regressions by stashing
  `Sources/DGen/` and re-running the SAME filter.
- **C backend cannot compile spectral BPTT** (undeclared `tape`,
  scratch redefinitions) — pre-existing, why the C fdcheck is XCTSkip and
  why all validation here runs on Metal.

## Acceptance criteria

1. `SVFFreqSurrogateTests` green: COLA identity (lag N−1, err < 1%) and
   Metal fdcheck ratio within [0.65, 1.35] (it's 0.99990 today — a drop to
   the tolerance edge means you broke something subtle; investigate).
2. Monologue benchmark epoch gpu_execute ≤ ~2.5 s (goal), and in any case
   decisively under the 1.5 s × (BPTT) + margin so the surrogate is a win.
3. Memory: `memory` buffer stays ~2–3 GB at crop 38072 (no regression to
   per-frame tapes).
4. Regression filter
   `'PhaseVocoder|SpectralLoss|OverlapAdd|HopGated|TensorHistory|BufferView|Spectral'`
   no worse than baseline (see flakiness note).
5. Then run rung C (`docs/SVF_FREQ_SURROGATE_SPEC.md` § Validation ladder):
   synth-target recovery ≈ current rung-2 (~94%), real-target loss parity
   ≈ 0.21 after `--polish-epochs`, and record the ms/epoch deliverable.

## Resolution (2026-08-14)

Result: **2.30 s/epoch gpu_execute** (from 42.9 s), stable across epochs;
epoch-1 loss and every param bit-identical to the pre-fix run; Metal fdcheck
still 0.99990; regression filter failure set identical to baseline
(attributed by stashing the four changed files and re-running the same
filter).

Root cause confirmed: the hop-rate tensor chains were swept into the SCALAR
node set by feedback-cluster path analysis (`findSequentialNodes` marks all
feedback-cluster members scalar; the mask-backward chain sits on read→write
paths via seq/ordering edges), so `partitionIntoBlocks` put them in
sequential runs and no later pass could rescue them. Verified with the
env-gated `DGEN_DEBUG_SCALAR_HOP=1` probe (prints tensor-shaped scalar-set
members with feedback membership).

Two mechanisms, both in `Sources/DGen/Blocks/BlockFormation.swift`, both
gated to Metal (`CompilationPipeline` passes `hopBasedNodes: [:]` for C — the
C renderer's SIMD lowering emits undeclared `simdNN` temps for peeled
blocks):

1. **`peelHopTensorRuns`** (fixes kernel_128-class, −25 s; also captured the
   forward STFT mask math in kernel_80, 1.49 s → 0.09 s).
   `TemporalityPass.inferTemporality` now runs BEFORE block formation
   (node temporality never depended on blocks), and `determineTensorBlocks`
   peels maximal contiguous runs of hop-classified pure tensor math out of
   sequential blocks into parallel blocks. Block temporality assignment then
   gives them `hopBased` and emission produces the butterfly target shape.
   Peel predicate excludes anything stateful: history/accum/latch/noise ops,
   raw memory ops, reduces, self-iterating ops, gemm/conv, tensorRef/seq.
   Safe because these chains read/write hop-sliced frame-aware cells and
   recompute identical values on every frame of a span (idempotent), so
   hop-gating them is semantics-preserving; genuine cross-frame state can
   only flow through the excluded ops.
2. **`isIsolatableHopSerialOp`** (fixes kernels 137/143/131/149-class,
   −14 s). `splitOutAcceleratedFFTNodes` generalized: hop-tagged
   self-iterating grad ops (`bufferViewGradStore`, `overlapAddGradGather`,
   …) are isolated into their own single-op sequential blocks, which then
   classify hopBased and get the block-level hop guard OUTSIDE the frame
   loop (they previously ran their 1024-element loops on all 38072 frames
   behind an in-loop select). They exchange data via memory cells /
   per-frame `t` globals, so the new kernel boundaries are safe;
   intrinsically frame-based ops (`bufferViewGradRead`) stay in the frame
   loop.

Footgun hit on the way: with `inferTemporality` moved earlier,
`temporalDependencies` (hop counters, position deps) are populated before
`partitionIntoBlocks`, and the output-node placement walked
`allDependencies` — the output followed its counter temporal-dep into a much
earlier block than its value producer (`insufficientInputs(output)`).
Placement now walks `node.inputs` only, which is exactly the old behavior.

Bonus: kernel hashes now stabilize after epoch 1 (`runtime_cache_hit=1`,
`pipeline_create_ms=0` from epoch 2 on) — the alternating-hash issue in the
footguns list no longer reproduces. Per-epoch `dgen_compile` (~3 s) remains;
that is the `docs/TRAIN_EPOCH_CACHE_SPEC.md` work.

150-epoch monologue benchmark (surrogate default path, post-fix):
`improvement_pct 41.4`, `abs_distance 4.141`, `basin_check ok` — vs the
BPTT baseline reference `26.9` / `4.099` from
`docs/TRAIN_EPOCH_CACHE_SPEC.md`. Param distance at parity, loss improvement
substantially better. Wall clock 27:14 for the full run (2×150 epochs +
renders), now dominated by the ~3 s/epoch `dgen_compile` — that is the
epoch-cache work, not this one.

Remaining floor (2.27 s total GPU): two ~0.43 s kernels that recompute a
loop-invariant hop-row sum-reduce (1024 elems) inside the per-frame loop —
hoistable to hop rate with a hop-sliced scalar + held read-back, but the
consumer chain is genuinely per-frame; diminishing returns. Then ~4 serial
scalar kernels at ~0.23 s each (the kernel_76 floor).
