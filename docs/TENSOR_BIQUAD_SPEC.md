# Spec: Tensor-shaped biquad (batched state + per-element controls)

Status: IMPLEMENTED — validated 2026-07-14.
Written 2026-07-14, motivated by the batched candidate-evaluation benchmark
(`Examples/SynthID/BatchBench.swift`, `SynthID batch-bench`). That benchmark
originally proved [B]-shaped batched evaluation of the `subtractive-bass`
voice with the filter bypassed. With this implementation the filter-active
path is accurate to 6.17e-5 versus serial and reaches 14.1-15.1 ms/candidate
at B=32-256 (239k-255k evals/hour), versus the ~200 ms serial baseline. See
the implementation results below and `Examples/SynthID/E1_POLICY_AUDIT_FINDING.md`
for why the batched population-search path matters.

- **Blocker A (bug)**: `SignalTensor.biquad` produces numerically wrong output
  for tensor-shaped audio — even at B=1 with scalar controls. Output is ~700x
  too small versus the equivalent `Signal.biquad`, and rounds to exactly 0.0
  at some sample rates. Repro: `.build/release/SynthID batch-bench --probe-only`.
- **Blocker B (missing API)**: there is no per-batch-element cutoff/resonance.
  `SignalTensor.biquad(cutoff: Signal, ...)` (`Sources/DGenLazy/Functions.swift:1163`)
  broadcasts one scalar control across all lanes, which defeats the purpose of
  batching candidates that differ in filter parameters.

## Root cause of Blocker A

`Graph.biquad` (`Sources/DGen/HigherOps.swift:401`) is shape-agnostic
elementwise arithmetic except for its state. It allocates four history cells
with plain `alloc()` — width 1, no tensor registration:

```swift
let history0Cell = alloc()   // HigherOps.swift:405-408
let history1Cell = alloc()
let history2Cell = alloc()
let history3Cell = alloc()
```

The engine already has a unified tensor-aware history mechanism, and the
biquad simply doesn't participate in it:

- `ShapeInference.swift:60` — `.historyRead(cellId)` infers `.tensor(shape)`
  only when `graph.cellToTensor[cellId]` maps to a registered tensor;
  otherwise `.scalar`.
- `Emit+State.swift:119,174` — `historyWrite`/`historyRead` emission likewise
  dispatches on `cellToTensor[cellId]`: tensor path if registered, scalar
  slot otherwise.

So with a [B]-shaped input: `x0PassThrough = historyWrite(history2Cell, in1)`
stores a [B] value into a width-1 cell, every `historyRead` infers `.scalar`,
the feedback terms mix scalar garbage into [B] arithmetic, and all lanes
share/clobber one state slot. This is the same bug class documented in
CLAUDE.md ("Memory Allocation & Cell IDs": unsized cells default to width 1
in `remapVectorMemorySlots`, causing overlap).

The correct pattern already exists in the tensor `statefulPhasor`
(`Sources/DGenLazy/SignalTensor.swift:86-121`):

```swift
let stateWidth = Swift.max(1, freqs.shape.reduce(1, *))
let cellId = graph.alloc(vectorWidth: stateWidth)
```

and in `makeHistory` (`HigherOps.swift:~737-750`), which additionally
registers a `Tensor` entry and `cellToTensor[cell] = tensorId` so the
history read/write emitters take the tensor path.

## Design

### Change 1 — shape-aware history cells in `Graph.biquad` (fixes Blocker A)

Extend `Graph.biquad` with an element-count parameter, defaulting to the
current scalar behavior:

```swift
public func biquad(
  _ in1: NodeID, _ cutoff: NodeID, _ resonance: NodeID,
  _ gain: NodeID, _ mode: NodeID,
  elementShape: Shape? = nil        // nil => scalar, byte-identical today
) -> NodeID
```

When `elementShape` is non-nil (including a one-element tensor), for each of
the four history cells with element count W:

1. `alloc(vectorWidth: W)` instead of `alloc()`;
2. create a `Tensor` entry with shape `elementShape` backed by that cell and
   register `cellToTensor[cell] = tensorId`, following the `makeHistory`
   registration (no view transforms needed — plain per-element state).

No changes to the filter arithmetic: every node op in the body (`mul`, `add`,
`selector`, `gswitch`, `cos`, `sin`, ...) is elementwise and already
broadcasts scalar coefficients against tensor-shaped audio once the history
reads/writes carry the right shape.

The `SignalTensor.biquad` wrappers (`Functions.swift:1163,1172`) pass
`elementShape: shape`. The scalar `Signal.biquad` wrappers pass nothing.

**Scalar-path invariant**: with `elementShape == nil` the generated graph,
cell allocations, and kernels must be byte-identical to today. The existing
biquad test suite (27 tests, `swift test --filter Biquad`) plus the E1/E2
regression evidence all ride on the scalar path; do not disturb it.

### Change 2 — per-element controls overload (fixes Blocker B)

Add to `Sources/DGenLazy/Functions.swift`, next to the existing overload:

```swift
public func biquad(
  cutoff: SignalTensor, resonance: SignalTensor,
  gain: Signal, mode: Signal
) -> SignalTensor {
  precondition(cutoff.shape == shape && resonance.shape == shape,
    "per-element biquad controls must match audio shape")
  let nodeId = graph.graph.biquad(
    self.nodeId, cutoff.nodeId, resonance.nodeId,
    gain.nodeId, mode.nodeId, elementShape: shape)
  ...
}
```

Mixed forms (tensor cutoff + scalar resonance) can be provided by promoting
the scalar with a broadcast multiply against a ones tensor at the wrapper
level; do not add combinatorial overloads to the graph layer.

The coefficient arithmetic (cos/sin/div/selector on `cutoff`/`resonance`
nodes) becomes tensor-shaped automatically via shape inference. Verify
`selector` and `gswitch` shape inference broadcast correctly for mixed
scalar/tensor operands — if either rejects the mix, that's in scope to fix,
and is the most likely hidden cost in this spec.

### Explicitly out of scope

- Batched/tensor biquad **gradients**. The population search needs forward
  only. The BPTT carry-cell machinery (`docs/BIQUAD_BPTT_GRADIENT_BUG.md`,
  pass-through writes, reverse-time loop) was built and validated for scalar
  cells; auditing it for vector cells is a separate follow-up (needed later
  for *batched* elite Adam polish — serial polish works today). Until then,
  requesting gradients through a tensor-shaped biquad must **fail loudly**
  (throw/precondition), never silently truncate or emit wrong adjoints.
  Add an explicit guard + test for this.
- Any change to production SynthID voice/trainer/policy code. The
  `subtractive-bass` serial path keeps using scalar `Signal.biquad`.
- The benchmark's compiled-graph-reuse `.sum`/`.output` accumulation issue
  (noted in the batch-bench report) — separate cleanup.

## Test plan

New tests (suggested: `Tests/.../TensorBiquadTests.swift`), all comparing
against the scalar `Signal.biquad` path as ground truth:

1. **B=1 equivalence (the Blocker-A repro, formalized)**: sine through
   `SignalTensor.biquad([1]-shaped)` vs `Signal.biquad`, identical scalar
   controls, all 8 modes, a few sample rates (incl. 44.1 kHz and one that
   currently yields exact-zero output). Max abs diff < 1e-6 over >= 4096
   frames. This test must FAIL before Change 1 and pass after.
2. **B=8 shared controls**: 8 distinct input signals (different frequencies),
   one shared cutoff/resonance; each lane matches its serial render.
3. **Lane-state independence**: impulse in lane 0, silence elsewhere; lanes
   1..7 output exactly 0 for all frames (catches shared/overlapping state,
   the failure mode of the current bug).
4. **B=8 per-element controls** (Change 2): same input signal in all lanes,
   8 different cutoffs spanning 80 Hz–8 kHz and varied resonance; lane i
   matches `Signal.biquad(cutoff_i, res_i)`.
5. **Time-varying per-element cutoff**: the subtractive-voice shape,
   `cutoff_i(t) = fBase_i + fAmt_i * exp(-t / fDecay_i)` as a [B]
   SignalTensor; lane-wise match vs serial time-varying scalar biquad.
6. **Gradient guard**: building a backward pass through a tensor-shaped
   biquad throws the explicit unsupported error.
7. **Scalar regression**: `swift test --filter Biquad` stays 27/27 with zero
   modifications to those tests.

Integration acceptance (the actual goal):

8. Re-run the batch-bench correctness gate **with the filter enabled**
   (`SynthID batch-bench`, B=8, per-candidate filter params from seed-6
   true/initial/recovered + perturbations): per-element max abs diff vs
   serial renders < 1e-4, per-candidate losses matching serial `score`.
9. Re-run the timing sweep (B = 1..256) with the filter active and record
   per-candidate ms — expect some regression vs the 6 ms filter-broken
   number since the sequential biquad kernel now does real per-lane work,
   but the win only has to stay >> serial 200 ms to keep population search
   cheap. Update the numbers in this doc's motivation paragraph if they
   move materially.

## Risks / things to verify while implementing

- **Block formation & feedback analysis**: `FeedbackAnalysis.swift` and
  `Blocks/BlockFormation.swift` special-case history cells for the sequential
  feedback kernel. Verify vector-width history cells still land the biquad in
  one sequential-over-time kernel with W parallel lanes (like the batched
  phasor), not W serialized filters and not a broken parallel split.
  CLAUDE.md's Metal rules apply: the recurrence is over time; lanes are
  independent.
- **Memory remapping**: confirm `remapVectorMemorySlots` /
  `cellAllocationSizes` see width W for all four cells (CLAUDE.md documents
  the default-to-1 overlap failure). Inspect `cellAllocations.cellMappings`
  in a debug run.
- **`historyWrite` pass-through shape**: shape inference for `historyWrite`
  returns its input's shape (`ShapeInference.swift:66`) — fine — but confirm
  the emitted tensor write covers all W elements per frame, not element 0.
- **Existing `TensorHistory` users**: Changes touch shared history emission
  only via new registrations, not modified code paths, but run the spectral /
  hop-gated history tests to be safe.

## Acceptance summary

Done when tests 1–7 pass, integration gates 8–9 are recorded (B=8
filter-enabled correctness < 1e-4; timing table updated), the scalar path is
untouched (27/27 + byte-identical kernels for a scalar render), and gradient
requests through tensor biquads fail loudly.

## Implementation results (2026-07-14)

Implemented and validated:

- Four independently sized and tensor-registered biquad history cells, with
  the nil/scalar graph-construction path unchanged.
- Shared-control and per-element cutoff/resonance `SignalTensor` wrappers.
- Tensor-aware broadcasting/emission for `selector`, tensor-history fusion
  exclusion, pass-through output aliasing, and sequential-over-time history
  scheduling.
- Explicit `DGenError.unsupportedGradient` rejection when a backward loss
  reaches a tensor-shaped biquad.
- Focused coverage in `TensorBiquadTests`: B=1/all modes/two sample rates,
  B=8 shared controls, lane isolation, B=8 per-element controls, time-varying
  per-element cutoff, allocation registration, and the gradient guard.

Filter-enabled `SynthID batch-bench` result (32,768 frames, 44.1 kHz,
20 timing iterations, with the existing first-five steady-state metric;
seed-6 true/initial/recovered plus five per-filter-parameter perturbations):

| B | Compile (s) | Eval (s) | Per candidate (ms) | Evals/hour |
|---:|---:|---:|---:|---:|
| 1 | 0.089 | 0.062 | 61.52 | 58,521 |
| 8 | 0.129 | 0.158 | 19.71 | 182,609 |
| 32 | 0.515 | 0.483 | 15.08 | 238,725 |
| 128 | 1.819 | 1.821 | 14.23 | 252,985 |
| 256 | 3.538 | 3.607 | 14.09 | 255,493 |

The B=8 correctness gate passed with overall max absolute audio difference
`6.170e-5` (< `1e-4`). CPU proxy losses agreed lane by lane; the largest
absolute loss difference was below `3e-8`. Machine-readable results are in
`output/batch_bench/batch_bench_report.json`.
