# The Biquad "BPTT Bug": Gradient Carry-Cell Aliasing + Truncated IIR Gradients

Status (2026-07-06):
- **Part A — carry-cell memory aliasing: FIXED** (one line in `Gradients.swift`)
- **Part B — truncated temporal gradient through biquad state: OPEN, by design gap**

Discovered by the SynthID `--fdcheck` harness (`Examples/SynthID/FDCHECK_FINDING.md`).
This document explains what actually happened, how to reproduce both parts, and
the solution space for Part B.

## Original symptom

In a synth graph containing `Signal.biquad` with a **trainable cutoff**, gradients
for *unrelated* parameters were wrong or exactly zero:

```text
outGain  finite diff ~=  2.09    autograd =  0.0 (exactly)
bodyAmp  finite diff ~= -5.80    autograd = -0.53
```

Making the cutoff a `Signal.constant` restored every other parameter's gradient.
So the trigger was: **any gradient target behind biquad's internal history chain**.

## Why this was initially called "the BPTT bug" — and why that's wrong

`Signal.biquad` is a macro (`Sources/DGen/HigherOps.swift`, `func biquad`) that
expands into ~100 primitive ops plus four `historyRead`/`historyWrite` cell pairs
for the filter state (`y[n-1]`, `y[n-2]`, `x[n-1]`, `x[n-2]`).

When `computeGradients` (`Sources/DGen/Gradients.swift`) encounters a
`historyRead` node that has an accumulated gradient, it allocates a **gradient
carry cell** (`getGradCarryCell`) and appends a side-effect
`memoryWrite(carryCell, 0, grad)` — the mechanism BPTT uses to pass gradients
backward through time.

Two things were suspected and ruled out by kernel inspection:

1. **BPTT loop wrapping never activates for biquad.** `wrapWithBPTTLoops`
   (`Sources/DGen/Blocks/Emission/BPTTEmission.swift`) is gated by
   `blockHasPassThroughHistoryWriteWithCarry`, which requires the
   `historyWrite`'s *output* to be consumed (the pass-through pattern from
   `Signal.history()`). Biquad's writes are dangling statements
   (`_ = n(.historyWrite(cell), input)`), so the check returns false. The
   emitted Metal contained no reverse loops in either the working or broken
   configuration.
2. The forward/backward kernel structure was otherwise correct — the
   outGain gradient formula (`tanh_value × spectral_grad`) was present and
   properly cross-kernel-ordered.

## Part A (fixed): carry cells aliased live gradient memory

The actual mechanism, confirmed by dumping `cellAllocations.cellMappings` and
reading gradient memory after the run:

```text
param outGain: gradCell=95341 -> physical 91198   value = 0.0
carryCell 95337 (history cell 6) -> physical 91198   ← SAME ADDRESS
carryCell 95340 (history cell 8) -> physical 74814   ← spectral gradTime buffer
```

`getGradCarryCell` allocated carry cells with `alloc()` but did **not** register
them in `Graph.persistentCells`. Carry cells are accessed via raw
`memoryRead`/`memoryWrite`, which the buffer-reuse liveness analysis in
`remapVectorMemorySlots` (Blocks/MemoryAllocation.swift) does not track — the
same bug class previously fixed for `bufferView` and `overlapAdd` ring buffers.
With `enableBufferReuse = true` (the DGenLazy default), the remapper packed the
carry cells onto memory that was still live:

- One carry cell landed on **outGain's grad-accumulation cell**. The per-frame
  carry write (offset 0) stomped the accumulated gradient → exactly `-0.0`.
- Another landed on the **spectral gradTime buffer**, corrupting the
  ∂loss/∂sample values that every other parameter's gradient is built from →
  bodyAmp/drive gradients off by ~10×.

The corruption pattern (some params zero, some partially wrong, none obviously
NaN) is what made this look like a deep BPTT scheduling bug. It was a memory
allocator interaction.

### The fix

`Sources/DGen/Gradients.swift`, `getGradCarryCell`:

```swift
let carryCell = alloc()
gradCarryCells[historyCellId] = carryCell
persistentCells.insert(carryCell)   // ← the fix
return carryCell
```

After the fix, carry cells get exclusive physical slots and (same graph,
trainable cutoff): `outGain.grad = 1.771` vs `1.770` in the constant-cutoff
control. SynthID fdcheck with the noise filter enabled went from
outGain `relErr = 1.0` (exact zero) to `relErr = 0.08`, noiseAmp to `0.08`
(remaining spread is FD conditioning of the log-magnitude loss, documented in
FDCHECK_FINDING.md). `BPTTTests`, `BufferTests`, `HistoryTensorTests`,
`AudioMLTests`, `SpectralGradientMagnitudeTests` all pass.

**General lesson (third occurrence of this bug class):** any cell that is
accessed via raw `memoryRead`/`memoryWrite` across kernels or frames MUST be
added to `Graph.persistentCells` at allocation time. Grep candidates when
adding new gradient machinery: `alloc()` calls in Gradients.swift and
GradientSetup.swift.

## Part B (open): biquad parameter gradients are truncated in time

With the aliasing fixed, other params' gradients are correct — but the
gradient **for the biquad's own parameters** (cutoff/resonance/gain) only
includes the *current-frame* path:

```text
∂y[n]/∂cutoff  via  coefficients(cutoff) applied to x[n], x[n-1], ...
```

and treats the recursive state reads (`y[n-1]`, `y[n-2]` via `historyRead`) as
constants. For an IIR filter most of the output's sensitivity to the
coefficients flows through the recursion, so the truncated gradient is tiny
and can even have the wrong sign. Measured (linear-L2 spectral loss, window
256, 2048 frames): FD magnitude ~2-3, autograd `-0.011` — roughly 300× too
small.

Mechanically, the temporal terms are dropped because:

1. `historyRead.backward` returns nothing (correct — it has no inputs); the
   accumulated grad w.r.t. its output is written to the carry cell.
2. The carry cell is only ever **consumed** by `historyWrite.backward`
   (`grads = [gradOutput + carryGrad]`) — which runs only if the historyWrite
   node is on the loss path. Biquad's writes are dangling → their backward
   never executes → carry cells are written but never read.
3. Even if they were read, correct temporal propagation requires the reverse
   (N-1 → 0) execution order provided by `wrapWithBPTTLoops`, which never
   activates (no pass-through write).

## Reproduction

Both parts reproduce with this graph (Metal backend, 44.1 kHz, 2048 frames,
`spectralLossFFT(windowSize: 256, hop: 64)`):

```swift
let outGain = Signal.param(0.7, min: 0.4, max: 1.0)
let cutoff  = Signal.param(log(2800.0))          // ← the trigger; use
                                                 //   Signal.constant to control
let t = Signal.accum(Signal.constant(1.0/44100), reset: 0, min: 0, max: 0.05)
let body = sin(Signal.statefulPhasor(Signal.constant(120)) * 2 * .pi)
         * exp(Signal.constant(-7) * t) * 0.75
let noise = Signal.noise().biquad(
  cutoff: exp(cutoff), resonance: Signal.constant(0.707),
  gain: Signal.constant(1.0), mode: Signal.constant(0.0))
let student = tanh((body + noise * exp(Signal.constant(-140) * t) * 0.08) * 2.0)
            * outGain
let loss = spectralLossFFT(student, teacher, windowSize: 256,
                           lossMode: .l2, hop: 64, normalize: true)
_ = try loss.backward(frames: 2048)
// Part A (pre-fix): outGain.grad == -0.0 exactly; with cutoff constant: ~1.77
// Part B (still):   cutoff.grad ~ -0.01 while FD says magnitude ~2-3
```

A runnable version with cell-mapping introspection and kernel dumps lives in
`Tests/DGenLazyTests/BPTTBiquadScratchTests.swift`
(`testDumpKernels` prints gradCell → physical mappings; before the fix it shows
the carry cell and the param grad cell sharing an address).

Also end-to-end via SynthID:

```sh
swift run SynthID train --target <wav> --out /tmp/fd --frames 2048 \
  --windows 256 --no-linear-mag --fd-eps 1e-2 --fdcheck outGain
# pre-fix: autograd exactly 0; post-fix: ~8% of FD (log-loss FD noise)
# --no-noise-filter removes the biquad and both parts of the bug entirely
```

## Solution space for Part B

Ordered roughly by leverage:

### B1. Rewire the biquad macro to the pass-through history pattern (preferred)

`Signal.history()`'s BPTT support exists and is tested; biquad predates it.
Rewrite the macro so each state write is pass-through — i.e. the write node's
output feeds the downstream computation instead of dangling:

```swift
// today:  _ = n(.historyWrite(yCell), yNew);  output = yNew
// instead: output = n(.historyWrite(yCell), yNew)   // pass-through
```

Then `historyWrite.backward` runs (adding the carry gradient), and
`blockHasPassThroughHistoryWriteWithCarry` activates `wrapWithBPTTLoops`,
giving the reverse-time loop that temporal correctness requires. Risks to
check: biquad has FOUR interacting state cells (the x-history writes chain
`historyRead → historyWrite` between cells), and the BPTT machinery has only
been exercised with single-cell `Signal.history()` feedback; the per-frame
tape (`allocatePerFrameStorageCells`) must capture every forward intermediate
the four backward chains need. Verify with `--fdcheck`-style FD comparison
using a **linear-L2** loss (log-L1 FD is too noisy to validate against — see
FDCHECK_FINDING.md).

### B2. Dedicated adjoint-IIR gradient op

The adjoint of an IIR filter is the same filter run backward in time on the
incoming gradient. A hand-written `biquadGrad` LazyOp could:

1. run the adjoint filter over ∂L/∂y (reverse loop, like the spectral
   backward ops already do with their own kernels),
2. accumulate ∂L/∂coefficient = Σₙ adjoint-weighted taps,
3. chain-rule into cutoff/resonance/gain through the coefficient formulas.

More work, but self-contained, numerically standard, and doesn't stress the
general BPTT machinery. Mirrors how `spectralLossFFT` implements its own
backward instead of relying on generic autograd.

### B3. Frequency-sampling surrogate filter for training

For training use cases only: replace the time-domain biquad with a
differentiable frequency-domain approximation (multiply the noise spectrum by
the biquad's magnitude response, which is a closed-form differentiable
function of cutoff/resonance). DDSP does exactly this (FIR noise filtering via
`buffer → conv2d`, already used in DDSPE2E/TrainKick808). Zero library work —
this is the current **SynthID workaround**: `--no-noise-filter`, or model the
noise tone with the existing FIR path.

### B4. Accept truncated gradients, but loudly

Cheapest and worst: document that biquad params get truncated-BPTT-0
gradients. Not recommended silently — the failure mode (tiny, sometimes
wrong-sign gradients that Adam happily follows) is exactly the kind of quiet
wrongness the fdcheck harness exists to catch. If chosen, `computeGradients`
should at minimum emit a warning when it creates carry cells for history
cells whose writes are dangling (that condition is precisely "temporal
gradient will be dropped").

### Guardrail worth adding regardless

A debug assertion or compile-time warning: **if `gradCarryCells` is non-empty
but no block passed `blockHasPassThroughHistoryWriteWithCarry`, the temporal
gradient is being silently truncated.** Today that condition is exactly the
biquad case and costs nothing to detect.

## Timeline / cross-references

- Found while validating SynthID rung 1 (`Examples/SynthID/SPEC.md`); the
  spec's milestone 4 (`--fdcheck` before full training) is what surfaced both
  this and the spectral-gradient scale bug.
- `Examples/SynthID/FDCHECK_FINDING.md` — full investigation log, including
  the (N·numBins)^1.5 spectral scale bug fixed the same day, and the
  fdcheck methodology notes (log-loss conditioning, FD epsilon sweeps).
- Prior instances of the persistent-cell bug class: `bufferView` /
  `overlapAdd` (see MemoryAllocation notes in project memory / git history).
