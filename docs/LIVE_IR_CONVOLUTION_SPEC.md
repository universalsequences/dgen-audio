# Live IR Convolution — Spec

Convolution where the impulse response (IR) is itself a live signal, not a
static WAV file. "Freeze what you're hearing into a space" — one input
becomes the room; another input gets colored by that room, every hop.

Two flavors, sharing the same MAC kernel:

- **Short live IR** — IR ≤ N samples. Pure compose of existing primitives.
  Already shippable today, no new DGen code. Just a sugar operator.
- **Long live IR** — IR spans many hops (seconds of tail). Needs one new
  primitive: `spectrumHistory`, producing a rolling `[K, N]` ring of
  recent spectra so `partitionedSpectralMAC` can run against a live H as
  well as a live X.

## Motivation

Convolution reverb has always assumed a static IR. The web-audio
convolution node (`ConvolverNode`) makes this especially rigid — you hand
it a buffer once and it's baked for the lifetime of the node. Replacing
the IR means re-allocating and re-initializing internal state, which
audibly pops.

Live IR flips the frame: the IR is a signal. It can be

- another live input (e.g. a second mic),
- a sustained pad's captured spectrum,
- a gated version of the input itself (auto-impulse-resampling),
- a modulated or morphed version of a static IR.

Users have wanted this in web-audio land for a decade. DGen already has
every piece except one.

---

## Short live IR — patch recipe (no new code)

IR ≤ N samples. Single-FFT pipeline, no partitioning, no history buffer.

```
in1 (dry) ─► bufferView(N, hop) ─► *hann ─► fft ──┐
                                                   ├─► complexMul ─► ifft ─► *hann ─► OLA ─► out
in2 (IR)  ─► bufferView(N, hop) ─► *hann ─► fft ──┘
```

Every hop: compute two live spectra, multiply them bin-by-bin, inverse
FFT, synthesis-window, overlap-add back to sample rate. Identical to
partitioned convolution with K=1.

**Latency**: one hop.
**Quality**: fine for snare-length / vocal-snippet IRs.
**Memory**: two ring buffers of size N each. That's it.

**Patch**:

```
in 1 ─┐
      ├─► liveIRConvolve @N 1024 @hopSize 256 ─► out
in 2 ─┘
```

### What `liveIRConvolve` wraps (K=1 path)

```swift
let buf1 = graph.bufferView(in1, size: N, hopSize: hop)
let buf2 = graph.bufferView(in2, size: N, hopSize: hop)
let w1   = graph.n(.mul, try graph.reshape(buf1, to: [N]), hann)
let w2   = graph.n(.mul, try graph.reshape(buf2, to: [N]), hann)
let (aRe, aIm) = graph.acceleratedFFT(w1, N: N)
let (bRe, bIm) = graph.acceleratedFFT(w2, N: N)
let (yRe, yIm) = graph.complexMul(aRe, aIm, bRe, bIm)  // existing op
let t   = graph.acceleratedIFFT(yRe, yIm, N: N)
let wOut = graph.n(.mul, t, hann)
let out  = graph.overlapAdd(wOut, windowSize: N, hopSize: hop)
return graph.n(.mul, out, graph.n(.constant(gain)))
```

All primitives already exist. The operator is ~40 lines of Swift wrapping
the above. No DGen changes.

**File to add**: `patch-editor/Sources/Engine/operators/gen/GenLiveIRConvolve.swift`

---

## Long live IR — the real feature

IR spans many hops (K > 1). Need a rolling `[K, N]` ring buffer of the IR
stream's spectra so each hop can run the partitioned MAC:

```
Y[n] = Σ_{k=0..K-1} H[n−k] · X[n−k]
```

where *both* `X` and `H` are live spectra coming off bufferView+FFT
chains. Today `partitionedSpectralMAC` reads H from a static `[K, N]`
tensor; we need the same MAC to accept a live `[K, N]` tensor whose rows
shift every hop.

### Design: `spectrumHistory @K @N`

One new DGen primitive. Maintains a `[2K, N]` mirror-layout ring of the
last K spectra, exactly like `partitionedSpectralConvolve`'s internal
input ring — just exposed as a user-facing tensor output.

```
live (re, im)  ──► spectrumHistory @K @N ──► (histRe, histIm)  : both [K, N]
                                                                 row 0 = most recent
                                                                 row k = k hops ago
```

- Hop-gated (same gating as `partitionedSpectralConvolve`).
- Two persistent ring cells, each size `2·K·N`.
- Mirror-write: every hop, copy new spectrum into row `p` and row `p+K`
  of each ring, then advance `p = (p+1) mod K`.
- Mirror means downstream reads at row `(p + K − k)` for k ∈ [0, K−1]
  always land in [1, 2K) — no runtime modulo.
- Output tensors are zero-copy views over the ring cells, shape `[K, N]`,
  with a strided transform that indexes from row `p+K` backward.

### End-to-end long live-IR patch

```
dry ─► bufferView ─► *hann ─► fft ───────────────┐
                                                  ├─► partitionedSpectralMAC @N ─► ifft ─► *hann ─► OLA ─► out
ir  ─► bufferView ─► *hann ─► fft ─► spectrumHistory @K @N ─┘
                                      (produces live [K, N] IR stack)
```

### Memory cost (ballpark)

| K   | N    | hop  | IR seconds @ 44.1kHz | Memory    |
|-----|------|------|----------------------|-----------|
| 4   | 1024 | 256  | 0.023s               | 64 KB     |
| 32  | 1024 | 256  | 0.186s               | 512 KB    |
| 172 | 1024 | 256  | 1.0s                 | 2.75 MB   |
| 344 | 1024 | 256  | 2.0s                 | 5.5 MB    |

(per live spectrum: `2 · 2 · K · N · 4 bytes` — two channels (re/im),
two copies (mirror), K rows of N floats each.)

Same cost as a partitioned-static reverb of the same length, which is
the apples-to-apples baseline.

### Warm-up latency

Until the ring has seen K hops of input, the "tail" is silence. First
audible tail arrives one hop in; full IR energy at K hops in. Same as
any partitioned convolver.

---

## Implementation plan

### 1. DGen primitive — `spectrumHistory`

**Files to add / modify**:

| File | Change |
|---|---|
| `Sources/DGen/LazyOp.swift` | Add `case spectrumHistory(K:Int, N:Int, ringReCell:CellID, ringImCell:CellID, counterCell:CellID)`. Mark as side-effect, `emitsInternalIteration`. |
| `Sources/DGen/HigherOps+SpectrumHistory.swift` | **New**. Factory `graph.spectrumHistory(reInput, imInput, K, N) -> (histReNodeId, histImNodeId)`. Allocates the two ring cells (size `2·K·N`) and counter cell, registers `persistentCells`, emits the side-effect op, and exposes `[K, N]` tensorRef outputs chained after it. |
| `Sources/DGen/Emit+SpectrumHistory.swift` | **New**. Emit handler: `if (hopCounter == 0) { for n in 0..<N { ring[p*N+n] = re_in[n]; ring[(p+K)*N+n] = re_in[n]; } p = (p+1) mod K; }`. Mirrors the write-half of `partitionedSpectralMACCall`. |
| `Sources/DGen/Analysis/ShapeInference.swift` | Shape rule: both output tensors are `.tensor([K, N])`. The side-effect op itself is `.scalar`. |
| `Sources/DGen/Compilation/Passes/TemporalityPass.swift` | Add `.spectrumHistory` to `opEmitsFullHopGate` — the body self-gates via hop counter check. |
| `Sources/DGen/Blocks/FeedbackAnalysis.swift` | Mark `.spectrumHistory` as inherently scalar so its write loop stays sequential. |
| `Sources/DGen/Gradients.swift` | Stub: `.spectrumHistory` raises "backward unsupported" (reverb is inference-only for now). |

### 2. Expose `[K, N]` output as a readable tensor

The tricky part. Row 0 of the *user-visible* output should be the most
recent hop, but physically the ring is indexed mod K. Two options:

**Option A — strided view into the mirror half**:

The output tensor is a view over ring-cell memory, shape `[K, N]`, with
a dynamic row-base equal to `(p + K) − 0 = p + K`, decreasing with k.
Expose via an `asStrided`-like transform parameterized by the dynamic
counter. This is zero-copy but requires the view machinery to support
a dynamic row origin.

**Option B — materialize to a plain tensor**:

On each hop, after the write, also emit a K-row copy into a flat `[K, N]`
output cell where row 0 = ring row `p+K−0`, row 1 = ring row `p+K−1`,
etc. Costs one extra `K·N` copy per hop but doesn't need dynamic views.
Still cheap — K·N is a fraction of the subsequent MAC's work.

**Recommendation: start with Option B**. Ship the feature; optimize to
zero-copy later if profiling shows the extra copy matters. At N=1024,
K=172, hop=256, one extra 172K-float copy per hop costs ~172µs every
5.8ms — a few percent, not a dealbreaker.

### 3. `partitionedSpectralMAC` already accepts this

`partitionedSpectralMAC`'s inlets for `irRe` / `irIm` today take any
`[K, N]` tensor — the op reads them as `tensorRead(..., k*N + n)`. The
MAC doesn't care whether those tensors are baked static data or a live
ring surface; it just needs the right shape every hop. No changes
needed there.

### 4. Patch-editor operators

| Operator | Role |
|---|---|
| **`liveIRConvolve @N @hopSize`** (new) | Sugar for the short (K=1) case. Two inlets (dry, ir), one outlet. Pure compose of existing primitives. |
| **`spectrumHistory @K @N`** (new) | User-facing handle on the new DGen primitive. Two inlets (re, im), two outlets (histRe, histIm). |
| **`liveIRConvolveLong @N @hopSize @K`** (new) | Sugar for the long case. Internally: two bufferView+FFT chains, one `spectrumHistory` on the IR side, one `partitionedSpectralMAC`, shared IFFT/OLA tail. |

Files:

- `patch-editor/Sources/Engine/operators/gen/GenLiveIRConvolve.swift` (short)
- `patch-editor/Sources/Engine/operators/gen/GenSpectrumHistory.swift`
- `patch-editor/Sources/Engine/operators/gen/GenLiveIRConvolveLong.swift`
- `patch-editor/Sources/Engine/TestUtilities.swift` — register all three

### 5. Tests

`Tests/DGenTests/SpectrumHistoryTests.swift` — new:

- **Basic write / read**: feed a known sequence of `[N]` spectra at hop
  rate, after T hops verify `histRe[row k]` equals the spectrum fed T−k
  hops ago. Both for k=0 (current) and k=K−1 (oldest).
- **Warm-up**: first K−1 hops, rows beyond seen-count must read as
  zero (cell init).
- **Integration**: spectrumHistory + partitionedSpectralMAC against a
  known live-IR pattern; compare against time-domain reference
  computed with `directConv(x, h)` where both x and h are captured
  per-hop streams.

`Tests/DGenTests/LiveIRConvolutionTests.swift` — new:

- **Short path**: K=1 live IR, feed known dry + known IR, compare OLA
  output to time-domain `conv(dry[:L], ir[:L])` on the first L samples.
- **Long path**: K=8 or 16 live IR, same reference comparison.
- **Equivalence**: feed a *constant* IR signal (static `[N]` tensor
  played every hop) and confirm output matches existing
  `partitionedConvolve` within 1e-4 relative.

### 6. Documentation

- Append a "Live IR" section to `docs/SPECTRAL_COMPOSABILITY_NOTES.md`
  once shipped.
- Update `loadIR` docstring to say "for live IR (signal-rate), use
  `liveIRConvolve` or `liveIRConvolveLong`".

---

## Expected user patches

### "Auto-impulse" — use the dry signal as its own IR

```
in 1 ─► gate(envelope) ─► liveIRConvolve @N 1024 @hopSize 256 ─► out
   └───────────────────────┘
```

Classic freeze-and-color trick: the envelope gate captures one
"impulse" shape from the input and convolves everything that follows
with it.

### "Living room" — two-mic convolution

```
mic 1 ─┐
       ├─► liveIRConvolveLong @N 1024 @hopSize 256 @K 172 ─► out
mic 2 ─┘
```

Mic 2 captures ambient room tone; mic 1 (dry source) gets convolved
against the continuously-refreshed acoustic fingerprint.

### "Morphing IR" — crossfade between a static and a live IR

```
liveSpectrum ─► spectrumHistory @K @N ──┐
                                         ├─► lerp(t) ─► partitionedSpectralMAC ─► ifft ─► OLA
partitionIR  ────────────────────────────┘
```

Any `[K, N]` tensor arithmetic works between the live history and a
baked partition — lerp, mask, EQ, spectral bit-crush, etc.

---

## Open questions (resolve during implementation)

- **Dynamic-origin view (Option A)**: if we eventually want it, how does
  `asStrided` express a per-hop base row? Need to inspect `ViewTransform`
  to see if it supports a runtime offset cell rather than a compile-time
  constant.
- **Hop-rate mul of two `[K, N]` tensors**: the morphing-IR patches lean
  on this. Today a live `[K, N]` × static `[K, N]` should Just Work
  through `n(.mul, …)`, but worth verifying with a small test before
  advertising it as a pattern.
- **Gain calibration**: with a live IR whose total energy varies over
  time, static auto-gain is wrong. Either expose @gain as a user control
  (simplest) or add a running-norm node that tracks `Σ|H|²` and
  compensates. Defer to user control for v1.

---

## Out of scope (for this spec)

- Backward pass / trainable live IRs — reverb is inference-only.
- Stereo / multichannel — spec assumes mono. Stereo is K ring pairs per
  channel; straightforward extension.
- Fractional-hop IR delays — if we ever want sub-hop IR shifting. Not
  needed for the natural "live-IR reverb" use case.
