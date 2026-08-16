# Spec: Frequency-sampled SVF surrogate for `dgenlisp train` (v0.1)

Status: FUNCTIONAL, PERF-BLOCKED — 2026-08-13. Implemented (layers 1-5) and
training runs end-to-end with fdcheck-verified gradients (Metal autograd/FD
ratio 0.99990), but at ~43 s/epoch vs the 1.5 s BPTT baseline. See
"Implementation findings" at the bottom: three landed compiler fixes
(hop-sliced tensor storage, hop-gated backward, hop-held mask controls) and
the remaining kernel-scheduling work item that gates production use.

## Problem statement

The `dgenlisp train` monologue benchmark runs ~1.5 s/epoch, dominated by BPTT
through the patch's three SVF calls (`benchmarks/train_monologue/patch.lisp:305-311`).
Each `svf` macro expansion creates two `make-history` cells (`ic1eq`/`ic2eq`,
patch.lisp:57), so the forward pass is a sample-serial loop and the backward
pass replays it in reverse with a tape — six history cells' worth of
per-sample gradient work per epoch, on every epoch of a 500+-epoch phase.

## Approach: DDSP-style frequency sampling (not free FIR)

We do NOT train free FIR taps and decode them back to SVF parameters
afterward. Two reasons:

1. `lp_cut` is time-varying (`cutoff + filter_env_amount·filt_env·velocity +
   lfo·lfo_filter_amount + keytrack·pitch`, patch.lisp:308) — a static FIR
   cannot represent the sweep, and the sweep is the point of the patch.
2. Unconstrained taps drift off the 2-pole manifold; the projection back
   (Prony fit → biquad → SVF) loses exactly what training gained.

Instead, keep the SVF *parameterization* (cutoff, q, mode) and replace only
the *recurrence*: per STFT frame, evaluate the SVF's closed-form transfer
magnitude |H(ω; g, k)| at the FFT bin frequencies and multiply the input
frame's spectrum by it (zero-phase). Gradients flow directly to `cutoff`,
`filter_env_amount`, `lp_q`, etc. — no decode step exists because we never
leave SVF parameter space. This is the standard trick from DDSP (Engel et
al. 2020) and Nercessian's frequency-sampling differentiable biquads.

The time-varying cutoff is handled for free: each hop gets its own H from
that frame's cutoff value. Nonlinearities between filter stages (`hp_sat`
sits between the HP and LP stages) are unaffected: each surrogate returns a
time-domain signal via IFFT + overlap-add, so `sat` still applies in the
time domain between stages.

**Handoff:** because parameters are shared (not decoded), training can run
the cheap surrogate for the bulk of the epochs, then optionally polish for a
handful of epochs with the true BPTT SVF. Rendering/checkpoints always use
the real SVF (the render subprocess evaluates the unswapped patch).

## Math

The Cytomic/TPT SVF is the bilinear transform of the analog prototype
`H(s) = {1, s, s²}/(s² + k·s + 1)` with prewarped `g = tan(π·fc/fs)`,
`k = 1/Q`. Evaluating on the unit circle at frequency f:

```
w(f)   = tan(π·f/fs) / g                     (prewarped normalized freq)
|D|²   = (1 − w²)² + k²·w²                   (denominator, s = j·w)

|H_lp|²    = 1        / |D|²                 (mode 0, lp = v2)
|H_bp|²    = w²       / |D|²                 (mode 1, bp = v1; peak gain Q)
|H_hp|²    = w⁴       / |D|²                 (mode 2, hp = in − k·v1 − v2)
|H_notch|² = (1 − w²)² / |D|²                (mode 3, lp + hp)
|H_peak|²  = (1 + w²)² / |D|²                (mode 4, lp − hp)
|H_ap|     = 1                               (mode 5)
```

Sanity identity: `H_lp + k·H_bp + H_hp = 1` (matches the macro's
`hp = input − k·v1 − v2`).

Bin mapping — the mask is applied to the **full** N-point complex FFT of a
real frame, so it must be symmetric to keep the IFFT real:

```
f_bin(i) = min(i, N − i) · fs / N,   i = 0..N−1
f_bin clamped to [f_min, 0.499·fs]   (avoid tan singularity at Nyquist;
                                      mirror of the macro's safe_cutoff clip)
```

The mask multiplies re and im by `|H| = sqrt(|H_mode|²)` — zero-phase
filtering. The multi-resolution spectral loss is magnitude-only
(phase-blind), so discarding the SVF's phase response costs nothing the
loss can see.

Approximation errors (accepted, bounded by the validation ladder):

- **Frame-LTI**: cutoff is held constant within a frame; the envelope sweep
  is sampled at hop rate. Error shrinks with hop size.
- **Ring-out**: a high-Q filter rings across frame boundaries; zero-phase
  frame-wise filtering smears that ring's phase but preserves per-frame
  magnitude, which is all the loss measures.
- **Peak undersampling**: a very sharp resonance between bin centers is
  undersampled by the bin grid. Mitigate with N ≥ 1024 (43 Hz bins at
  44.1 kHz); the response is smooth in ω so this only matters at high Q.

## Build order (per discussion): DGen core → DGenLazy tests → DGenLisp → swap

### Layer 1 — DGenLazy composite `svfFrequencySampled` (the core lift)

Substrate audit (2026-08-13) — the surrogate should be a **composite of
existing IR ops**, no new op kinds expected:

| Piece | Status |
|---|---|
| `Signal.buffer(size:hop:)` → SignalTensor | exists (`Signal.swift:387`), backward exists (`bufferViewGradStore/Read`, Gradients.swift:794-819) |
| `signalTensorFFT` / `signalTensorIFFT` | exist (`Functions.swift:807-843`); built from pure tensor view + arithmetic ops, so differentiable through existing elementwise/view grads. (`acceleratedFFT` is the non-differentiable one — do not use.) |
| `SignalTensor.overlapAdd(hop:)` | exists (`Functions.swift:851`), backward exists (`overlapAddGradStore/Gather`, Gradients.swift:1268-1288) |
| Scalar `Signal` broadcast over bin tensor | `SignalTensor.lift(_:shape:)` (`Promotion.swift:30`) + shape-inference broadcast |
| `tan` on SignalTensor | exists (`Promotion.swift:37`) |
| Mode blend | `eq` + arithmetic, same shape as the macro's mode select |

Proposed signature (Functions.swift or a new `FilterSurrogates.swift`):

```swift
/// Frequency-sampled SVF: STFT → per-frame |H(ω; cutoff, q, mode)| mask →
/// ISTFT. Differentiable w.r.t. input, cutoff, and q. Zero-phase; magnitude
/// matches the time-domain Cytomic SVF per frame.
public func svfFrequencySampled(
    _ input: Signal, cutoff: Signal, q: Signal, mode: Signal,
    window N: Int, hop: Int, sampleRate: Float
) -> Signal {
    // 1. frames = input.buffer(size: N, hop: hop) .* hann(N)   (analysis window)
    // 2. (re, im) = signalTensorFFT(frames, N: N)
    // 3. g = tan(pi * clip(cutoff, 1, 0.49*fs) / fs)           (Signal, per hop)
    //    k = 1 / max(q, 0.001)
    //    w = SignalTensor.lift(tan(pi * fBins / fs), ...) / g  (broadcast over bins)
    //    mask = sqrt(blend-by-mode of the |H|² formulas above)
    // 4. (re, im) = (re * mask, im * mask)
    // 5. out = signalTensorIFFT(re, im, N: N).overlapAdd(hop: hop) / colaGain
}
```

Notes and known risk points for this layer:

- **Broadcast backward**: the gradient of a lifted scalar Signal combined
  with an [N]-shaped tensor must **sum over bins** back to the scalar at the
  hop instant. `lift` is a view, so this rides on DGen's existing
  scalar-vs-tensor broadcast grad. Verify with fdcheck first; if the reduce
  is missing, that is the one genuinely new gradient path in this project.
- **Hop gating**: everything downstream of `buffer(hop:)` must execute once
  per hop. The mask math consumes `cutoff` (audio-rate) inside the hop-gated
  region, which gives sample-and-hold at hop instants for free — and the
  backward deposits cutoff's gradient at those instants only. If mask ops
  end up outside the hop gate (per-sample), that's a correctness+perf bug to
  catch in the kernel dump.
- **Kernel boundaries**: the composed chain crosses shape transitions
  (scalar → [N] → scalar) both forward and backward. All the historical
  failure modes documented in CLAUDE.md apply: reduction ops need kernel
  splits, tensors created before `LazyGraphContext.reset()` alias stale
  nodes, `cellAllocationSizes` must cover lazy cells. Budget debugging time
  here — this composition (bufferViewGrad + tensor-FFT grads + a trainable
  mask + overlapAddGrad in ONE graph) has never been exercised.
- **COLA**: hann analysis window with hop = N/4 (or N/2); divide the
  overlap-add output by the constant hann overlap sum. No synthesis window
  in v0.1 (WOLA/sqrt-hann is a refinement if masking artifacts show up in
  the loss).
- **Latency**: buffer→overlapAdd imposes a constant output delay (≈ N − hop
  samples; measure it with the identity test below). A magnitude-STFT loss
  DOES see envelope time-shift, so compensate: delay the target (or advance
  the synth crop) by the measured constant when the surrogate is active.
  Since all three svf calls share (N, hop), the delay is uniform.

### Layer 2 — dedicated DGenLazy tests (before any lisp work)

`Tests/DGenLazyTests/SVFFreqSurrogateTests.swift`:

1. **Identity/COLA**: mode=allpass (mask ≡ 1) → output equals input delayed
   by the constant latency, mid-signal error < 1e-4. Pins down latency and
   COLA gain.
2. **Static magnitude parity**: white-noise input through time-domain `svf`
   (Signal history version) vs surrogate, compare steady-state magnitude
   spectra over a grid: fc ∈ {100, 440, 2k, 8k}, Q ∈ {0.55, 1, 4, 10},
   mode ∈ {lp, bp, hp}. Tolerance loose at high Q (ring-out smearing), tight
   at low Q.
3. **fdcheck cutoff/q**: scalar trainable cutoff and q, spectral loss
   against a target rendered at different (fc, Q); autograd vs central
   differences, sign match + magnitude within 2x (project fdcheck
   convention).
4. **fdcheck through input**: trainable oscillator frequency upstream of the
   surrogate — verifies bufferViewGrad + FFT-composite backward into the
   time-domain input.
5. **Time-varying cutoff**: linearly swept trainable `envAmount` modulating
   cutoff; fdcheck the sweep-amount gradient (this is the case BPTT-SVF
   handles natively and frequency sampling approximates — the gradient must
   at minimum sign-match).
6. **Recovery micro-rung**: saw → surrogate LP; train (cutoff, q) to a
   target rendered with the *time-domain* SVF at known params. Must recover
   within a few percent — proves the surrogate's optimum sits at the real
   filter's parameters, which is the entire premise.

### Layer 3 — DGenLisp op `svf-freq`

New evaluator case (`LispEvaluator.swift`, `evaluateOperator` dispatch):

```lisp
(svf-freq input cutoff q mode @window 1024 @hop 256)
```

Thin wrapper over `svfFrequencySampled`; `@window`/`@hop` optional with
defaults from this spec. Parity test: lisp op output matches the DGenLazy
call bit-for-bit on a fixed patch.

### Layer 4 — the symbol swap (lowering pass)

`svf` is a `defmacro` in the patch itself and macro dispatch is a plain
name lookup (`LispEvaluator.swift:143`), so the swap is an AST rewrite
before evaluation — the same pattern as `ExcitationLowering` /
`stripModulation`, which already run in `TrainPlanner.makePlan`
(TrainPlanner.swift:72-76) and land in `patchPlan.loweredNodes` (re-evaluated
every epoch, serialized to `lowered.lisp` for auditability).

New `FilterSurrogateLowering.swift`:

```swift
/// Rewrite (svf input cutoff q mode) call heads to
/// (svf-freq input cutoff q mode @window N @hop H). The patch's
/// `defmacro svf` stays in the AST but becomes dead. Recursive descent,
/// same shape as ExcitationLowering.rewrite.
static func lower(nodes: [ASTNode], window: Int, hop: Int) -> [ASTNode]
```

Wiring:

- `TrainOptions`: `--filter-surrogate freq|none` (default `freq` for the
  train/direction phases), `--surrogate-window`, `--surrogate-hop`.
- `TrainPlanner.makePlan` applies the pass after excitation lowering when
  enabled; `lowered.lisp` then shows `svf-freq` calls — the audit trail.
- **Render path untouched**: `renderViaSubprocess` and final patch export
  evaluate the original (or excitation-lowered-only) source, so checkpoints
  and the final render always use the real time-domain SVF driven by the
  trained parameters.
- Latency compensation from Layer 1 applies only when the pass is active.
- Out of scope: the `ladder` macro (unused by the monologue patch; its
  in-loop feedback tanh breaks the LTI-per-stage assumption — needs its own
  treatment, e.g. linearized G⁴ response, later).

### Layer 5 — trainer integration + polish phase

- `DirectionTrainer.runPhase` needs no structural change: the surrogate
  arrives via `patchPlan.loweredNodes`.
- Add `--polish-epochs N` (default 0): after the surrogate phase, rebuild
  the plan with `--filter-surrogate none` and run N epochs of true-BPTT
  refinement from the surrogate's best-z. Free handoff — same parameters.
- Epoch runtime caching (TRAIN_EPOCH_CACHE_SPEC) applies unchanged; the
  surrogate graph is stable across epochs so kernel hashes hit.

## Validation ladder (top level)

- **Rung A** = Layer 2 tests (DGenLazy).
- **Rung B**: lisp parity + lowering-pass snapshot test on the monologue
  patch (`lowered.lisp` golden).
- **Rung C** (the benchmark): monologue synth-target recovery with the
  surrogate must match the current rung-2 result (~94%); real-target loss
  must land ≈ 0.21 parity (current v2 plateau) after the polish phase.
  Record epoch timings via `DGENLISP_TRAIN_TIMING=1` — the deliverable
  number is surrogate ms/epoch vs the current ~1500 ms/epoch.

## Performance expectations (honest version)

Removing the three SVFs removes six history cells from the BPTT tape, but
the patch still has sequential state (envelope one-poles, phasors, LFO), so
the sample-serial loop does not disappear — it gets much lighter, and the
surrogate's own work (3 × FFT/IFFT at hop rate + bin math) is
frame-parallel on Metal. Measure, don't assume: if the residual serial loop
dominates, the win is smaller than "SVFs were 90% of the epoch" implies.
The timing instrumentation in `DirectionTrainer.emitTiming` already breaks
out gpu_execute vs compile vs graph-build.

## Open questions

1. Does the scalar-broadcast backward reduce over tensor elements, or is
   that the one missing gradient path? (First fdcheck answers this.)
2. N/hop defaults: 1024/256 proposed (43 Hz bins, 5.8 ms envelope sampling
   at 44.1 kHz). The rung-C sweep should try 512/128 for speed.
3. Does zero-phase filtering interact badly with `overlap-add` edge frames
   at note onset (first N samples)? The loss already down-weights nothing —
   if onset transients dominate the residual (cf. the SynthID rung-3
   finding), consider cropping the first window from the loss when the
   surrogate is active.
4. Metal kernel count: 3 surrogates × (FFT + IFFT) × forward+backward may
   push kernel counts up enough that dispatch overhead matters at small N.
   The epoch cache amortizes compilation; dispatch is the thing to watch.

## Implementation findings (2026-08-13)

Layers 1-5 were implemented (FilterSurrogates.swift, svf-freq lisp op,
FilterSurrogateLowering, TrainOptions --filter-surrogate/--surrogate-window/
--surrogate-hop/--polish-epochs, default `freq`). Getting the monologue
benchmark (crop 38072, N=1024, hop=256) to actually run took three compiler
fixes; the perf ceiling needs a fourth, not yet done.

### Fix 1 — hop-sliced frame-aware storage (landed)

First run failed with `MetalError 4: Failed to create buffer memory`: 2,096
frame-aware tensor cells (FFT butterfly + mask intermediates, forward AND
backward tape) each demanded `1024 × 38072 × 4B` ≈ 156 MB → ~300 GB total.
Hop-rate cells only produce a value every `hop` frames, so:

- `Graph.frameAwareCellHops[cellId]` records the hop of hop-based
  frame-aware cells; allocation becomes `tensorSize × ceil(frames/hop)`
  (TensorMemoryMaterializationPass).
- `frameAwareOffset` (IRBuilder+Memory) indexes hop-sliced cells with
  `frameIdx / hop` — all accesses happen on hop boundaries, so slots are
  dense and consistent. Non-hop cells are unchanged (hop=1 path).
- The artifact-cache fingerprint includes the new map.

### Fix 2 — hop-gated backward (landed)

The backward was designed frame-based with zero-padding: overlapAddGradGather
wrote real grads at hop frames and zeros at the other 255, into full
per-frame tapes, and every backward butterfly inherited frame-based
temporality (correct — the Metal fdcheck passed on the old design at toy
scale — but unscalable). Changes:

- overlapAdd backward tags `gatherOp` and the sequenced grad tensor with the
  forward chain's `nodeHopRate` (found via `findUpstreamHopRate` walking up
  from the tensor input), so TemporalityPass schedules the whole backward
  tensor chain hop-based; its grad cell is hop-slot allocated.
- bufferView backward reads the seq node's own `nodeHopRate` and hop-slices
  its tape the same way.
- `bufferViewGradRead` gained a hop-grid loop: sample p sums the ≤ N/hop+1
  hop windows covering it instead of all N frames (float index math +
  final int cast — int min/max trips metal overload ambiguity).
- `.bufferViewGradRead` added to `isIntrinsicallyFrameBased` — it emits the
  per-sample grad signal and must never be promoted to hop rate.

### Fix 3 — hop-held mask controls (landed)

cutoff/q are frame-rate signals (envelope-driven), which demoted the entire
mask construction — and its backward — to per-frame execution and per-frame
storage (896 cells). `svfFrequencySampled` now runs both through the
existing `hopHold(hop:)` primitive: semantically this IS the spec's
"controls sampled once per hop", and it restores hop classification
end-to-end (post-fix: 0 frame-rate frame-aware cells, all 4,284 hop-sliced,
~2.6 GB, training runs).

### Remaining work — kernel scheduling (NOT done; gates production use)

Measured 42.9 s/epoch GPU (DGENLISP_TRAIN_PROFILE=1 prints a per-kernel
table; DGENLISP_TRAIN_KERNEL_DUMP=<dir> writes kernel sources). Root cause,
from reading the hot kernels: hop-rate tensor work fused into
frame-sequential scalar kernels, gated by an in-loop `select` that zeroes
values instead of skipping work:

- kernel_128 (25 s, 58%): mask-backward math emitted as a single-thread
  `frameCount × 1024` flattened loop with NO hop guard — 256× redundant and
  serial. Three per-frame scalars fused into the block are what dragged the
  tensor math into this shape.
- kernels 137/143/131/149/141/135 (~15 s): serial frame loops (legitimate
  carry-cell/overlap-add-gather cores) each dragging 1024-element hop-row
  copies/scatters/reduces through all 38072 frames.
- kernel_80 (1.5 s): forward FFT correctly hop-gated but single-threaded.

The compiler already produces the correct shape for the FFT backward
butterflies (per-frame threads, hop guard OUTSIDE the element loops, only
149 threads do work) — the fix is block-formation/emission surgery so the
fused cases get the same treatment: split per-frame scalars from hop tensor
regions, hoist hop-row loops into hop-gated element-parallel kernels with
the guard outside the loop. Post-fix estimate ~1.5-2.5 s/epoch (the serial
scalar floor is ~0.2-0.3 s per irreducible carry-cell kernel, cf.
kernel_76).

Also observed: kernel hashes alternate between two values across epochs
(full_cache_hit=0), so per-epoch recompiles (~1.9 s) persist — likely
nondeterministic ids from the hopHold cells; worth fixing alongside the
epoch-cache work.

### Validation status

- `SVFFreqSurrogateTests`: COLA/identity (lag N-1, err < 1%), Metal fdcheck
  cutoff grad ratio 0.99990. C fdcheck = XCTSkip (known spectral-BPTT tape
  codegen gap in the C backend).
- Spectral/hop regression suites pass; OptimizerTests, PhaseVocoderTests,
  SpectralLossOrderingScratchTests, SVFBPTTScratchTests show order-dependent
  failures in combined runs that reproduce identically WITHOUT these changes
  (pre-existing flakiness, all pass in isolation).
- Rung B (lisp parity + lowering snapshot) passes; rung C (synth-recovery /
  real-target parity + timing deliverable) NOT run — blocked on perf.
