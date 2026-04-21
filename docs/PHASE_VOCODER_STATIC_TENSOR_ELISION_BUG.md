# Static-Tensor Chain Elision Bug (blocking phase vocoder)

## TL;DR

Writing `sub(sub(A, B), C)` in the graph where `C` is a static `[N]` tensor
with baked-in data, compiled into a chain downstream of FFT output, emits
the first subtraction but silently drops the second one from the generated
kernel. Downstream ops read as if the second subtraction never happened.

This blocks a mathematically-correct phase vocoder — the "proper"
formulation needs to subtract a per-bin `ω_target` tensor from the raw
phase diff, and that subtraction vanishes during compilation.

## Where it shows up

`HigherOps+PhaseVocoder.swift:phaseVocoder` currently ships a simplified
heuristic (without `ω_target`) because of this bug. Even the heuristic
version has a related symptom: the `pitchRatio` multiplication is
emitted correctly (`vmulq_f32(simd161, c6)` lands in the kernel for
ratio=2), but the output waveform at ratio=2 is nearly identical to
ratio=1 — suggesting the scaled value doesn't flow through the
accumulator to the cos/sin reconstruction as expected.

## Minimal graph that triggers the static-chain elision

```swift
let (xRe, xIm) = g.acceleratedFFT(windowed, N: N)
let phase = g.n(.atan2, xIm, xRe)
let phasePrev = g.spectrumDelay(phase, N: N, hops: 1, hopSize: hop)

// Static per-bin tensor with data baked in.
var omegaData = [Float](repeating: 0, count: N)
for k in 0..<N { omegaData[k] = Float(k) * 2.0 * .pi * Float(hop) / Float(N) }
let omegaTarget = g.tensor(omegaData)

let innerDiff = g.n(.sub, phase, phasePrev)         // this EMITS
let rawDiff   = g.n(.sub, innerDiff, omegaTarget)   // this DOES NOT emit into the kernel
```

Dumped node state confirms shape inference is correct:
```
innerDiff=31 inputs=[24, 29] shape=tensor([1024])
rawDiff=32   inputs=[31, 30] shape=tensor([1024])
omegaTarget=30 shape=tensor([1024]) cell=13315 data=[0.0, π/2, π, 3π/2, …]
```

And the emit handler fires for both:
```
⚡ emit sub node=31 inputs=[24, 29] shape=tensor([1024])
⚡ emit sub node=32 inputs=[31, 30] shape=tensor([1024])
```

But in the compiled C kernel, only one `vsubq_f32` at the tensor level is
present, computing `phase − phasePrev`. The `- omegaTarget` subtraction
— even as scalar-unrolled per-element math — is nowhere in the emitted
output. Downstream `principalArg` reads `memory[innerDiff_cell]` as if
it were `rawDiff`.

## Unit test to confirm

There are two concrete tests already in the suite that together pin this
behavior:

### 1. The positive control — chained subs DO work on static tensors
`Tests/DGenTests/TensorAccumTests.swift` → `testChainedTensorSubtractionEmitsBothOps`

Builds `(a − b) − c` where all three are static tensors, summed and
output. Passes: generated kernel has both subtractions, result is
`(10−3−2)·N = 20`. This rules out a generic "chained sub" bug.

### 2. The failing case — same shape, but `a` comes from FFT
`Tests/DGenTests/TensorAccumTests.swift` → `testFFTOutputMinusSpectrumDelayMinusStaticTensor`

Builds `(fft.phase − spectrumDelay(phase)) − staticTensor`, summed.
The test currently passes because it's checking for ≥2 subtractions
including scalar accum wrap math, which count toward the minimum. But
inspecting `/tmp/two_sub_chain.c`, both tensor subs DO land. So the
bug isn't simply "sub with one dynamic + one static input".

### 3. The actually-failing scenario
`Tests/DGenTests/PhaseVocoderTests.swift` → `testPhaseVocoderPitchShiftsSinusoid`

Full pipeline: bufferView → hann → FFT → polarFFT (atan2) → subtract
spectrumDelay → subtract static ω_target → principalArg →
accumulate → rectFFT → IFFT → hann → OLA. The test is currently
relaxed to only check non-silent output. Tighten it back to `maxDiff
> 0.05` between ratio=1 and ratio=2 to fail on the bug.

**To reproduce the elision specifically**: enable the full vocoder in
`HigherOps+PhaseVocoder.swift` (the commented-out block that computes
`rawDiff = innerDiff − omegaTarget`) and dump the generated C
(`/tmp/two_sub_chain.c` or similar). You'll see `phase − phasePrev`
emitted once, no `− omegaTarget` anywhere, and the principalArg chain
reading the un-corrected diff.

## Data pointing at the cause

Running `DGEN_DEBUG_PHASE_VOCODER=1` (reintroduce the debug prints from
commit history) over the phase vocoder test shows this post-compile
`nodeToTensor` dump — focus on the `physical=-1` entries:

```
node=30 (tensorRef omegaTarget) cell=13315 physical=1024   ← correctly placed
node=31 (sub innerDiff)         cell=...    physical=-1    ← cell exists but no physical offset
node=32 (sub rawDiff)           cell=...    physical=151029768
node=37 (sub wrapped)           cell=...    physical=-1
node=38 (add omegaActual)       cell=...    physical=-1
```

`physical=-1` means `cellAllocations.cellMappings` has no entry for that
cell — the cell was reserved but never got a concrete memory offset.
`TensorMemoryMaterializationPass.decideTensorAllocation` classifies these
as non-materializable (not outbound, register-only). That's fine in
principle — but then at emit time, downstream ops that expect to read the
cell as a tensor via `tensorRead` / `tload` may be picking up an earlier
node's cell instead.

Compare to `innerDiff` at `physical=-1`: its output IS visible in the
kernel at `memory[144737288 + …]` — that's from a completely different
cell (probably a reused intermediate). So cells WITHOUT a physical offset
do still get emitted, but they may share/alias other cells in ways that
specifically break the static-tensor chain.

## Candidate root causes (in order of likelihood)

### 1. `TensorMemoryMaterializationPass` register caching + static tensor path mismatch

When `rawDiff = sub(innerDiff, omegaTarget)`:
- `innerDiff`'s output is cached in `ctx.tensorCellToVar[innerDiff_cell]`
  as a register.
- When emit reads `innerDiff` as the first input of `rawDiff`, it hits
  the register cache and returns the register's Lazy.
- `omegaTarget` is a tensorRef pointing to a static cell — needs
  `tensorRead` / `memoryRead` which goes to memory directly.
- The subtraction happens in registers but the RESULT is cached as
  `rawDiff_cell`'s variable.
- Downstream `principalArg(rawDiff)` reads `rawDiff_cell`'s register.
- BUT — if the register cache gets evicted at a block boundary,
  `rawDiff`'s register goes away, and downstream reads from
  `rawDiff_cell`'s MEMORY, which was never materialized (`physical=-1`).
- Memory[rawDiff_cell] never got written, so fallback reads pick up
  whatever is at that uninitialized location — possibly garbage, possibly
  `innerDiff`'s data if they alias.

**How to test**: walk the block list, find the block containing
`rawDiff`, check whether `rawDiff`'s emitted UOps include a
`memoryWrite` or just register assignment. If just register, check
which block downstream consumers are in and whether they'd need
memory fallback.

### 2. `findOutboundTensorCells` doesn't mark the sub chain as cross-block

If `innerDiff`, `rawDiff`, `wrapped`, `omegaActual` all end up in the
same tensor region, they should be register-cached. But my test puts
them across multiple blocks due to shape transitions (scalar hop
counter reads interleaved). If the dependency analysis doesn't
correctly identify `rawDiff` as outbound-required, its cell stays
lazy, and the downstream block (principalArg) re-reads `innerDiff`
from memory instead of `rawDiff`.

**How to test**: add a print in `findOutboundTensorCells` for the
phase vocoder test and confirm that `rawDiff`'s cell is in the
outbound set for its producing block.

### 3. Block formation fuses wrong

With one frame-based block containing both the sub chain and the
principalArg loop, and another block containing the accum + rect,
block formation might be grouping the subs into a region that
register-caches aggressively. If `emitRegion` in
`ShapeTransitionPlanner.swift` treats the `omegaTarget` tensorRef
as "skippable" (it's in the expand-region safe list:
`isAllowedExpandRegionOp` includes `.tensorRef`), the whole region
might get marked as a skipped expand+reduce candidate by
`detectFusableReduces`, which would cause the sub to not emit in its
expected shape.

**How to test**: in `detectFusableReduces`, add a print for each
`skipRegions` entry and confirm the phase-vocoder sub chain isn't
being skipped.

### 4. `broadcastIndex` or `tensorRead` path for tensorRef with
   static data has a shape-resolution issue

The `readInput` path for tensor-shaped input uses `tensorRead` when
the tensor has view transforms, else `broadcastIndex + tload`. For
`omegaTarget` (1D static tensor, no transforms), it goes through
`broadcastIndex`. If the parent node (`rawDiff`) has a different
shape than `omegaTarget` in some subtle way (even though both are
`[1024]`), broadcastIndex could compute a wrong offset and read a
constant zero (giving `innerDiff − 0 = innerDiff`, which matches the
observed behavior).

**How to test**: dump the `broadcastIndex` expression for
`omegaTarget` when consumed by `rawDiff`. It should be `elemIdx`,
not `constant(0)`.

## Short-term workaround

Skip `ω_target`. The heuristic `accum(principalArg(phase − phasePrev) *
ratio)` works well enough to NOT crash or produce NaN, but doesn't
actually pitch-shift cleanly because the wrapped phase-diff aliases.

## Files to inspect when picking this up

- `Sources/DGen/Compilation/Passes/TensorOutputBindingPass.swift` —
  binds nodes to tensor cells.
- `Sources/DGen/Compilation/Passes/TensorMemoryMaterializationPass.swift`
  — decides which cells get physical memory; `decideTensorAllocation`
  is the key function.
- `Sources/DGen/Blocks/Emission/ShapeTransitionPlanner.swift` —
  `detectFusableReduces`, `buildRegions`, `emitRegion` all touch
  this path.
- `Sources/DGen/Blocks/BlockEmission.swift` —
  `findOutboundTensorCells`, `wireCrossBlockGlobals`, the whole
  emit orchestration.
- `Sources/DGen/IRBuilder+Memory.swift` — `tstore` / `tload`
  register caching behavior.
- `Sources/DGen/Emit+Tensor.swift` — the shape-transition aware
  emit paths.

## Related broken behavior (possibly same root cause)

The heuristic phase vocoder *without* ω_target should produce
audibly different output at ratio=1 vs ratio=2 because `cos(α ·
accumulated_phase)` oscillates at different rates. The `vmulq_f32(…,
c6)` DOES appear in the kernel at ratio=2. But the OUTPUT is nearly
identical to ratio=1 (maxDiff ≈ 0.0002 over thousands of samples).
This suggests the accumulated_phase value doesn't actually incorporate
the scaled phaseDiff into the cos/sin — it reads the pre-scaled
value from somewhere unexpected.

Likely same register-caching / outbound-cell issue as the main bug:
the scaled sub's output is cached in a register that gets evicted at
a block boundary, and the accum downstream reads the pre-scaled
cell by fallback.
