# DGenLisp Spectral and Tensor Upgrade Spec

Status: draft

Goal: make DGenLisp able to express the same spectral and tensor workflows that are currently comfortable in patch-editor gen patches, while keeping Lisp code compact enough for real instrument/effect authoring.

## Design Principles

- Prefer ordinary Lisp forms over new one-off syntax.
- Multi-output DSP operators should return first-class values that can be destructured with `def`.
- Anything patch-editor can build from spectral gen operators should have a direct, readable DGenLisp equivalent.
- Tensor operations should work for both static `tensor` and per-frame `signalTensor` values wherever DGen supports that.
- File-backed audio/tensor assets should share the existing DGenLisp manifest model used by `wavetable` and `wavetable-param`.

## Current Gap Summary

DGenLisp already has useful tensor and spectral pieces:

- `wavetable`, `wavetable-param`, `peek`, `peek-row`, `sample`, `to-signal`
- `zeros`, `ones`, `full`, `randn`, `tensor-param`
- `reshape`, `transpose`, `shrink`, `pad`, `expand`, `repeat`
- `matmul`, `conv2d`
- `sum`, `mean`, `max-axis`, `sum-axis`, `mean-axis`, `softmax`
- `buffer`, `fft`, `ifft`, `overlap-add`

The main missing surfaces are:

- destructuring `def`
- first-class multi-output results
- polar/rect spectral helpers
- Hann/window helpers
- complex helpers
- hop-rate helpers
- spectral delay, phase vocoder, and partitioned convolution wrappers
- `conv1d`, `windows`, and tensor history
- audio-file-backed tensors for IR and morphable filter/wavetable workflows

## Destructuring `def`

Current syntax:

```lisp
(def name expr)
```

Add destructuring syntax:

```lisp
(def (name1 name2 ...) expr)
```

Example:

```lisp
(def frame (reshape (buffer input 1024 256) @shape [1024]))
(def (re im) (fft frame @N 1024))
(def (mag phase) (polar-fft re im))
```

Rules:

- The first `def` operand may be an atom or a list of atoms.
- A destructuring `def` requires the right-hand expression to evaluate to a tuple value.
- Arity must match exactly.
- Bound names are installed in `definitions` exactly like scalar `def`.
- Tensor manifest naming should only auto-name a tensor when a non-destructuring `def` binds a single tensor. Tuple outputs should not silently rename all manifest entries unless explicitly specified by the producing op.
- Macro scoping must treat every atom inside a destructuring `def` target as a local def name.

Implementation note:

```swift
enum EvalResult {
    case signal(Signal)
    case tensor(Tensor)
    case signalTensor(SignalTensor)
    case tuple([EvalResult])
    case float(Float)
    case none
}
```

## Multi-Output Operators

Operators with multiple logical outputs should return `.tuple`.

Required tuple-returning forms:

```lisp
(fft input @N 1024)
(polar-fft re im)
(rect-fft mag phase)
(complex-mul ar ai br bi)
(complex-conj re im)
(phase-vocoder re im ratio @N 1024 @hop 256)
(partition-ir ir @N 1024 @hop 256)
(partitioned-spectral-mac xre xim irre irim @N 1024)
```

Compatibility:

- Existing `(fft input)` behavior that writes `__fft_re` / `__fft_im` may remain temporarily, but should be deprecated.
- Preferred code is always:

```lisp
(def (re im) (fft input @N 1024))
```

## FFT

Preferred syntax:

```lisp
(fft input)
(fft input @N 1024)
(fft input 1024) ; legacy positional form remains accepted

(ifft re im)
(ifft re im @N 1024)
(ifft re im 1024) ; legacy positional form remains accepted
```

Return types:

- `fft tensor -> (tensor tensor)`
- `fft signalTensor -> (signalTensor signalTensor)`
- `ifft tensor tensor -> tensor`
- `ifft signalTensor signalTensor -> signalTensor`

Backend:

```lisp
(fft input @N 1024 @backend tensor)
(fft input @N 1024 @backend accelerated)
```

- `tensor`: current pure graph FFT path.
- `accelerated`: DGen Accelerate-backed FFT/IFFT, C backend only.
- Default can remain `tensor` for portability, but spectral effect examples should prefer `accelerated` once wired.

## Spectral Coordinate Helpers

```lisp
(polar-fft re im)   ; returns (mag phase)
(rect-fft mag phase) ; returns (re im)
```

Equivalent math:

```lisp
mag   = sqrt(re*re + im*im)
phase = atan2(im re)
re    = mag * cos(phase)
im    = mag * sin(phase)
```

Also add:

```lisp
(atan2 y x)
(log10 x)
```

These already exist in lower DGen layers and should be exposed in DGenLisp for spectral math and dB workflows.

## Complex Helpers

```lisp
(complex-mul ar ai br bi) ; returns (re im)
(complex-conj re im)      ; returns (re -im)
```

`complex-mul`:

```text
re = ar*br - ai*bi
im = ar*bi + ai*br
```

Example:

```lisp
(def (xre xim) (fft frame @N 1024))
(def (hre him) (fft ir @N 1024))
(def (yre yim) (complex-mul xre xim hre him))
(def y (ifft yre yim @N 1024))
```

## Windows

Add:

```lisp
(hann 1024)
(window @type hann @N 1024)
```

Initial implementation can support only Hann. `hann` should return a static 1D tensor `[N]` using the periodic form:

```text
w[n] = 0.5 - 0.5*cos(2*pi*n/N)
```

Example:

```lisp
(def win (hann 1024))
(def frame (* (reshape (buffer input 1024 256) @shape [1024]) win))
```

## Hop-Rate Helpers

Add:

```lisp
(hop-hold value hop)
(noise)
(noise @size 1024)
(noise @size 1024 @hop 256)
```

Rules:

- Scalar `(noise)` keeps existing behavior.
- `(noise @size N)` returns a `[N]` `signalTensor`.
- `(noise @size N @hop H)` returns hop-rate tensor noise suitable for randomized spectral phase/masks.
- `hop-hold` should work on `signal`, `tensor`, and `signalTensor` if the underlying DGen op supports it.

Example randomized phase:

```lisp
(def (re im) (fft frame @N 1024))
(def (mag phase) (polar-fft re im))
(def rand-phase (* (noise @size 1024 @hop 256) pi))
(def (out-re out-im) (rect-fft mag rand-phase))
```

## Spectrum Delay

Expose DGen spectrum delay primitives:

```lisp
(spectrum-delay spectrum @N 1024 @hops 4 @hop 256)
(spectrum-delay-mod spectrum delay @N 1024 @max-hops 32 @hop 256)
```

Return type:

- input `signalTensor [N] -> signalTensor [N]`

Examples:

```lisp
(def delayed-phase (spectrum-delay phase @N 1024 @hops 1 @hop 256))
(def phase-delta (- phase delayed-phase))
```

```lisp
(def sweep (* (hop-hold (phasor 0.05) 256) 24))
(def smeared-mag (spectrum-delay-mod mag sweep @N 1024 @max-hops 24 @hop 256))
```

## Phase Vocoder

Expose:

```lisp
(phase-vocoder re im ratio @N 1024 @hop 256)
```

Return:

```lisp
(re im)
```

Example pitch shifter:

```lisp
(def input (in 1 @name input))
(def win (hann 1024))
(def frame (* (reshape (buffer input 1024 256) @shape [1024]) win))
(def (re im) (fft frame @N 1024 @backend accelerated))
(def (pre pim) (phase-vocoder re im 1.5 @N 1024 @hop 256))
(def td (ifft pre pim @N 1024 @backend accelerated))
(out (overlap-add (* td win) 256) 1 @name audio)
```

## IR and Audio-Backed Tensors

Current DGenLisp already supports JSON-backed `wavetable` and `wavetable-param`. Add audio-backed tensor forms for IR and filter/wavetable morphing workflows.

### Audio Tensor Loading

```lisp
(audio-tensor @file "irs/hall.wav")
(audio-tensor @file "irs/hall.wav" @mono true)
(audio-tensor @file "irs/hall.wav" @channel 0)
(audio-tensor @file "irs/hall.wav" @start 0.0 @end 1.5)
(audio-tensor @file "irs/hall.wav" @normalize peak)
```

Return:

- Static 1D `tensor [samples]` for mono.
- Optional future: 2D `[samples channels]` when `@mono false`.

Manifest:

- Add entries to `tensors` with `kind: "audio"` or `kind: "ir"`.
- Preserve `sourceFile`, `shape`, `mutable`, and initial data semantics like `wavetable`.

### IR Alias

```lisp
(ir @file "irs/room.wav")
```

Alias for mono audio tensor loading with `kind: "ir"`.

## Partitioned Convolution

Expose both low-level and high-level APIs.

Low-level:

```lisp
(partition-ir ir @N 1024 @hop 256) ; returns (ir-re ir-im)
(partitioned-spectral-mac xre xim irre irim @N 1024) ; returns (yre yim)
```

High-level:

```lisp
(partitioned-convolve input ir @N 1024 @hop 256)
(partitioned-convolve input ir @N 1024 @hop 256 @gain 1.0)
```

High-level expansion:

```lisp
(def win (hann N))
(def frame (* (reshape (buffer input N hop) @shape [N]) win))
(def (xre xim) (fft frame @N N @backend accelerated))
(def (irre irim) (partition-ir ir @N N @hop hop))
(def (yre yim) (partitioned-spectral-mac xre xim irre irim @N N))
(def td (ifft yre yim @N N @backend accelerated))
(overlap-add (* td win gain) hop)
```

Filter IR wavetable morphing should be expressible by combining `wavetable`/`sample` with FFT:

```lisp
(def bank (wavetable @shape [16 2048] @file "filters/morph_bank.json"))
(def pos (* (phasor 0.05) 15))
(def ir-frame (sample bank pos))
(def wet (partitioned-convolve input ir-frame @N 1024 @hop 256))
```

Open issue: partitioned convolution currently assumes static partitioned IR tensors. For continuously morphing IRs, we need either:

- a smaller non-partitioned FFT convolution path for short morphable filters, or
- runtime-updatable partition tensors with controlled update rate.

## Tensor Operations

Add `conv1d`:

```lisp
(conv1d input kernel)
```

Return:

- `tensor -> tensor`
- `signalTensor -> signalTensor`

Expose window extraction:

```lisp
(windows tensor @shape [kH kW])
```

This maps to DGenLazy `windows(_:)` and returns the im2col/as-strided view.

Optional lower-level future:

```lisp
(as-strided tensor @shape [..] @strides [..])
```

This should be considered advanced/unsafe and can wait until there is a clear authoring need.

## Tensor History

Add tensor state buffers for physical models and recurrent tensor DSP.

Syntax:

```lisp
(make-tensor-history name @shape [H W])
(make-tensor-history name @shape [H W] @data [ ... ])
(read-tensor-history name)
(write-tensor-history name value)
```

Return behavior:

- `read-tensor-history` returns a `signalTensor`.
- `write-tensor-history` returns the written value, mirroring scalar `write-history`, so it can sit in the graph.

Example membrane:

```lisp
(make-tensor-history prev @shape [32 32])
(make-tensor-history curr @shape [32 32])

(def lap-k (tensor @shape [3 3] @data [0 1 0 1 -4 1 0 1 0]))
(def curr-state (read-tensor-history curr))
(def prev-state (read-tensor-history prev))
(def lap (conv2d (pad curr-state @padding [1:1 1:1]) lap-k))
(def next (+ (- (* curr-state 2) prev-state) (* lap 0.01)))

(write-tensor-history prev curr-state)
(write-tensor-history curr next)
(out (sum next) 1 @name audio)
```

Implementation details:

- Add `tensorHistoryBindings` next to existing scalar `historyBindings`.
- Macro local-name discovery/scoping must include `make-tensor-history`.
- `write-tensor-history` should accept `tensor` and `signalTensor`.

## Suggested Implementation Order

1. Add `EvalResult.tuple` and destructuring `def`.
2. Convert `fft` to return tuples while keeping temporary `__fft_re` / `__fft_im` compatibility.
3. Add `atan2`, `log10`, `polar-fft`, `rect-fft`, `complex-mul`, `complex-conj`, `hann`.
4. Add `hop-hold` and tensor/hop noise.
5. Add `conv1d` and `windows`.
6. Add tensor history forms.
7. Add `audio-tensor` / `ir` manifest-backed loading.
8. Add `spectrum-delay`, `spectrum-delay-mod`, `phase-vocoder`.
9. Add `partition-ir`, `partitioned-spectral-mac`, and `partitioned-convolve`.

## Acceptance Tests

Minimum tests:

- `(def (re im) (fft ...))` binds both names and no longer requires `__fft_im`.
- Destructuring arity mismatch throws a clear parse/type error.
- Macro-local destructuring names are scoped independently across macro calls.
- `polar-fft` followed by `rect-fft` round-trips a simple tensor spectrum.
- `hann` returns expected first/middle values.
- `(noise @size 8 @hop 4)` returns a signalTensor with shape `[8]`.
- `conv1d` matches known small convolution cases.
- `windows` returns expected shape/data for a 2D tensor.
- Tensor history read/write compiles in a small feedback graph.
- `audio-tensor` loads a test WAV into tensor manifest data.
- `partitioned-convolve` compiles a simple identity IR case.

