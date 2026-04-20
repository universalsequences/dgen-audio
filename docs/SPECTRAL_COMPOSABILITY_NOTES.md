# Spectral Processing — Composability Roadmap

Working notes on refactoring the all-in-one `partitionedConvolve` into a
modular-synth-style set of nodes that can be rewired, modulated, and combined
into spectral effects beyond plain convolution reverb.

## What's really inside `partitionedConvolve`

The patch-editor node is a convenience wrapper around this runtime graph:

```
dry signal
  └► bufferView(N, hop)           # slides a window every sample;
                                  # downstream gated to hop rate
  └► * hann                       # analysis window (reshape + tensor mul)
  └► acceleratedFFT(N)  →  (xRe, xIm)    # live spectrum, per hop
                                │
                                ▼
                partitionedSpectralMACCall   ← THE only bespoke UOp
                                │
                                │  Maintains a mirror-layout [2K, N] ring of
                                │  the last K input spectra (one new row per
                                │  hop, plus a mirror write so reads are
                                │  wrap-free). Per hop:
                                │    Y = Σ_{k=0..K-1} H[k] · X[n-k]
                                │  implemented as K calls to `vDSP_zvma`.
                                ▼
                           (yRe, yIm)
  └► acceleratedIFFT(N)  →  time-domain, per hop
  └► * hann                       # synthesis window
  └► overlapAdd(N, hop)           # back to frame rate
  └► * gain                       # 8·hop / (3·N²) by default (Hann² COLA + 1/N)
```

### Offline at graph-build

- Read IR samples (from `loadIR` or any static-tensor source).
- Chop into K = ⌈L_ir / hop⌉ partitions of P=hop samples each.
- Zero-pad each partition to length N.
- FFT each (via `vDSP_fft_zip` through `radix2FFTInPlace`).
- Stack into two static tensors `irRe[K, N]`, `irIm[K, N]`.
- Everything above is pure Swift at compile-time.

### The only new UOp

`partitionedSpectralMACCall` in `DGen/UOps.swift` / rendered in
`CRenderer.swift`. Renders as:

```c
int p = (int)memory[partitionIdxCell];
memset(&memory[reOut], 0, N * sizeof(float));
memset(&memory[imOut], 0, N * sizeof(float));
DSPSplitComplex Y = { .realp=&memory[reOut], .imagp=&memory[imOut] };
for (int k = 0; k < K; k++) {
  int ring_off = (p + K - k) * N;
  DSPSplitComplex X = { .realp=&memory[ringRe + ring_off],
                        .imagp=&memory[ringIm + ring_off] };
  DSPSplitComplex H = { .realp=&memory[irRe + k*N],
                        .imagp=&memory[irIm + k*N] };
  vDSP_zvma(&X, 1, &H, 1, &Y, 1, &Y, 1, N);
}
```

Everything outside `partitionedSpectralMACCall` is pre-existing building
blocks (bufferView, fft, ifft, mul, overlapAdd) wired together by
`GenPartitionedConvolveOperator.createGraphNode`.

## The 4-piece decomposition (start here)

Break the all-in-one into four patch-editor nodes. Each is wireable on its own.

### 1. `partitionIR  @N  @hopSize`
Inlets: 1 (IR samples — `.matrix` or static tensor nodeId).
Outlets: 2 — `irRe` and `irIm`, both static `[K, N]` tensors.

Does the chop / zero-pad / offline FFT at graph-build time. Pure
compile-time op; just emits two `graph.tensor(shape:[K,N], data:...)`
nodes. Factor the logic out of
`GenPartitionedConvolveOperator.createGraphNode` into this.

### 2. `partitionedSpectralMAC  @N`
Inlets: 4 — `xRe`, `xIm` (live `[N]` spectrum at hop rate), `irRe`, `irIm`
(static `[K, N]`).
Outlets: 2 — `yRe`, `yIm` (live `[N]` at hop rate).

This is the bespoke UOp exposed directly. Infers K from the shape of the
irRe input. Hop rate is inherited via temporality propagation from `xRe`.

### 3. `complexMul`
Inlets: 4 — `aRe`, `aIm`, `bRe`, `bIm`.
Outlets: 2 — `Re`, `Im`.

Just `(aRe·bRe − aIm·bIm, aRe·bIm + aIm·bRe)`. Ergonomic wrapper for
tensor-tensor complex multiplication. Pure compose of existing
mul/add/sub; no new UOp.

### 4. `spectrumHistory  @N  @K`
Inlets: 2 — `re`, `im` (at hop rate).
Outlets: 2 — `histRe`, `histIm` (both `[K, N]` — row 0 = most recent).

Exposes the ring-buffer portion of `partitionedSpectralMAC`. Under the
hood: either reuse `partitionedSpectralMACCall`'s ring logic as a
separate UOp, or write a new small `spectrumRingWrite` UOp that does just
the mirror-write pass. The read surface is just a `[K, N]` static-shape
view of the ring cell (need to think about how to expose it as a
view-chained tensor).

### Backward compat

Keep `partitionedConvolve` as sugar that internally instantiates
`partitionIR` + `bufferView` + `fft` + `partitionedSpectralMAC` +
`ifft` + `overlapAdd` + gain. No user-visible breakage.

## Effects this decomposition unlocks

| Effect | Patch sketch |
|---|---|
| **Spectral freeze** | `spectrumHistory` → `selectRow @k` (latched by a gate) → `complexMul` with live spectrum → `ifft` + OLA |
| **IR length modulation** | `partitionIR` → multiply each row of `irRe`/`irIm` by a mask `[K]` modulated by an envelope / LFO → `partitionedSpectralMAC` |
| **IR morphing** | Two `partitionIR`s → `gswitch` or lerp per partition between (irRe_A, irIm_A) and (irRe_B, irIm_B) → `partitionedSpectralMAC` |
| **Paulstretch-lite** | `fft` → `polarFFT` (re/im → mag/phase) → randomize / interpolate phase → `rectFFT` → `ifft` → OLA |
| **Frequency shift** | `fft` → shift rows of spectrum tensor (roll / asStrided) → `ifft` |
| **Spectral gate** | `fft` → compute magnitude → threshold mask → mul into (re, im) → `ifft` |
| **Cross-synthesis / vocoder** | Two input streams each bufferView+fft; combine their magnitudes and phases with arithmetic → `ifft` |
| **Impulse swap** | `partitionIR` of IR-A and IR-B, crossfade the `[K,N]` tensors → MAC |

## Follow-on ops to add after the decomposition

- [DONE] **`polarFFT`** `(re, im) → (mag, phase)`. `sqrt(re² + im²)` and
  `atan2(im, re)`. Two outlets, pure composition.
- [DONE] **`rectFFT`** `(mag, phase) → (re, im)`. `mag · cos(phase)`, `mag · sin(phase)`.
- [DONE] **`complexConj`** `(re, im) → (re, -im)`.
- [DONE] **`hopHold @hopSize`** — sample-and-hold that latches a frame-rate
  scalar (phasor, envelope, noise, LFO) at hop boundaries and emits the held
  value every frame. Tags its output as hop-producing via `nodeHopRate`, so
  downstream spectral ops stay hop-gated instead of demoting the whole chain
  to sample rate. Required for patterns like `phasor → * pi → hopHold →
  rectFFT.phase` — without it the per-sample modulation forces FFT/IFFT into
  the frame loop. Scalar-per-frame execution is enforced because the
  underlying `.latch` op is flagged `isInherentlyScalar`.
- **`spectrumRoll  @bins`** — circular shift the spectrum along the last dim.
  Frequency shifting / spectral slide effects.
- **`spectrumDisplay`** — visualization node. Not a DSP op; a patch-editor
  UI piece that reads a spectrum cell and plots it.

## Deeper / bigger bets

- **Partitioned MAC with modulated K** — dynamically shortening the IR at
  runtime by zeroing out "active k" beyond some index. Already achievable
  via the IR-mask-modulation trick above, but a dedicated op could be
  cheaper (skip trailing partitions entirely).
- **Partitioned-conv-by-live-signal** — the "live H" case. Feed both
  `irRe`/`irIm` and `xRe`/`xIm` from live bufferView+fft chains. The op
  already supports it since inputs are just tensor references; just needs
  the patch-editor op to accept dynamic IR inputs.
- **Hop-rate arithmetic on spectra** — normal `mul`, `add`, `sub` on
  `[N]` tensors at hop rate already composes. Things like multiplying
  two live spectra, summing them, etc., just work.
- **Phase vocoder** — time-stretching via phase interpolation between
  consecutive hops. Needs `spectrumHistory` + `polarFFT` + per-bin phase
  difference + scaling.

## Implementation order (my plan)

1. [DONE] **Split `partitionedConvolve`** → `partitionIR` +
   `partitionedSpectralMAC` + `complexMul`. All-in-one kept as sugar.
2. [DONE] **`complexMul`** — 4-in / 2-out wrapper, no new UOp.
3. [DONE] **`polarFFT` / `rectFFT` / `complexConj`** — pure compositions.
4. [DONE] **`hopHold`** — frame-rate → hop-rate bridge for modulators.
   Unblocks `phasor → * pi → hopHold → rectFFT.phase` patterns.
5. [NEXT] **`spectrumHistory`** — needs a new UOp (`spectrumRingWrite`) or
   extraction of the ring logic from `partitionedSpectralMACCall`. Design
   the tensor-view surface carefully so `selectRow`/`shrink` across the
   K-axis produces a valid `[N]` tensor at any row. Unlocks spectral
   freeze, phase vocoder, anything that needs to look at recent spectra.
6. **Bigger bets** (modulated K, phase vocoder, etc.) once the primitives
   are in place.

### Block-formation fix that landed alongside hopHold

`bufferView`'s scalar write block previously ran a dead inner
`for (int t1 = 0; t1 < N; t1++) memory[ring + pos] = in[i];` loop —
writing the same per-frame scalar N times because the block had inherited
its tensorRef's shape but contained only scalar memoryWrite / tensorRef
ops. Fixed by `clearWastedTensorLoopMetadata` in the sequential-path
branch of `determineTensorBlocks` (BlockFormation.swift:664). Restricted
to sequential blocks so the parallel tensor-suffix keeps its shape — the
shape is what blocks `determineVectorPlan` from promoting
FFT/IFFT-emitting blocks to SIMD-4. Observed on the paulstretch-ish patch:
~23% → ~8% CPU at N=1024 hop=256.

## Files to touch for step 5 (`spectrumHistory`)

- `Sources/DGen/LazyOp.swift` — add `.spectrumRingWrite(K, N, ringReCell,
  ringImCell, counterCell)`. Mark `emitsInternalIteration` so the
  block-wrap strip keeps working.
- `Sources/DGen/Emit+PartitionedConvolution.swift` or a new
  `Emit+SpectrumHistory.swift` — port the mirror-write half of
  `partitionedSpectralMACCall` into a standalone emit.
- `Sources/DGen/HigherOps+SpectrumHistory.swift` — `graph.spectrumHistory(
  reInput, imInput, K, N, hopSize, hopCounterNode) -> (histReNodeId,
  histImNodeId)` where the returned nodes are tensorRefs over `[K, N]`
  views of the ring cells. Tricky bit: expose a *readable* `[K, N]` tensor
  whose row 0 = most recent hop, without requiring a separate
  materialization pass. Options: mirror-row read path (row i = `(p − i + 2K)
  mod K`) via a sliding-window-style transform, OR zero copy by reading
  the mirror half directly.
- `Sources/DGen/Analysis/ShapeInference.swift` — shape rule:
  `.tensor([K, N])` for the output tensor.
- patch-editor: `GenSpectrumHistory.swift` (1 inlet pair → 2 outlet pair).
- Test: `Tests/DGenTests/SpectrumHistoryTests.swift` — verify that after
  feeding N frames, reading row 0 matches the most recent hop spectrum
  (re/im) byte-for-byte, and row K-1 matches N-K hops ago.
