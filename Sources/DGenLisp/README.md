# DGenLisp

A Lisp-to-dylib compiler for DGen. Write DSP patches as S-expressions, compile to optimized native shared libraries with a JSON manifest.

## Usage

```
dgenlisp compile [<file.lisp>] [options]
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `-o`, `--output <dir>` | Output directory | `.` |
| `--name <name>` | Output file name (without extension) | `patch` |
| `--sample-rate <rate>` | Sample rate in Hz | `44100` |
| `--max-frames <count>` | Maximum frame count per process call | `4096` |
| `--debug` | Print debug information to stderr | off |
| `-` | Read from stdin | default if no file |

### Output

- `<name>.dylib` — Compiled shared library exporting the DGen host ABI v1 entry points
- `<name>.json` — Manifest with params, I/O, memory layout (also printed to stdout)

## Parameter Training

Fit bounded continuous patch parameters to a target WAV with `dgenlisp train`.
The seed file has the form `{"params":{"cutoff":1200,"gain":0.5}}`.
Only reachable parameters declaring `@min` and `@max` participate.

```bash
dgenlisp train \
  --patch patch.lisp \
  --target target.wav \
  --seed-params seed.json \
  --job-dir runs/my-fit
```

Training currently supports the `direction` mode and requires the Metal
backend. The target is loaded as mono, peak-normalized to 0.9, and cropped or
zero-padded to the planned frame count. Pitch and gate duration are estimated
from the target unless supplied explicitly.

Before training, the planner:

1. parses the patch and applies the training excitation/lowering policy;
2. classifies bounded, reachable parameters as learnable or frozen;
3. emits a typed `plan` event and writes `lowered.lisp` and `render.lisp`;
4. trains in normalized `[0,1]` coordinates, using logarithmic coordinates for
   wide positive ranges;
5. compares renders with a multi-resolution spectral objective; and
6. writes the selected parameters and final render to the job directory.

### Training options

| Flag | Description | Default |
|------|-------------|---------|
| `--patch <file>` | DGenLisp patch to fit | required |
| `--target <file>` | Target WAV; multichannel files are mono-summed | required |
| `--seed-params <file>` | Initial parameter JSON | required |
| `--job-dir <dir>` | Directory for all trainer artifacts | required |
| `--mode direction` | Training mode; only `direction` is currently supported | `direction` |
| `--epochs <N>` | Adam epochs for each full local trajectory | `300` |
| `--gate-frames <N>` | Excitation gate duration in samples | inferred from target |
| `--pitch-hz <value>` | Excitation pitch | estimated from target |
| `--checkpoint-every <N>` | Preview WAV cadence | `25` |
| `--report-every <N>` | NDJSON epoch-event cadence | `10` |
| `--backend metal\|c` | Compute backend; full training currently requires Metal | `metal` |
| `--plan-only` | Emit the lowering verdict and exit without training | off |
| `--filter-surrogate none\|freq` | Replace SVFs with the experimental frequency-sampled training surrogate | `none` |
| `--surrogate-window <N>` | Power-of-two surrogate FFT window | `1024` |
| `--surrogate-hop <N>` | Surrogate hop, which must divide the window | `256` |
| `--polish-epochs <N>` | True-SVF Adam epochs after surrogate training | `0` |

The command writes progress as typed NDJSON events to stdout. Diagnostics go
to stderr, so consumers can parse every stdout line as a train event. A normal
run emits `plan`, `stage`, incremental `epoch`, `optimization_progress`, and
`checkpoint` events, followed by terminal `result`. `optimization_progress`
reports `current`, `total`, and losses: CMA-ES emits up to five best-first
candidate losses per completed generation, while batched Adam emits its mean
lane loss per completed epoch. `--plan-only` stops after `plan`.

General artifacts under `--job-dir` include:

- `lowered.lisp` — the patch actually used by the training graph
- `render.lisp` — the exact patch used for previews and final rendering
- `seeded.wav` — render of the supplied seed
- `epochNNNN.wav` — optional checkpoint renders
- `final.wav` — selected final render
- `result.json` — terminal result event

### Search modes

The default `--search legacy` runs one Adam trajectory from the supplied seed
and a second trajectory from the transformed midpoint. The midpoint trajectory
is a basin check; legacy mode still returns the user's seeded trajectory.

`--search cma-es` performs tensor-batched, gradient-free global search and can
optionally run an independent local fallback, short top-K Adam refinement, and
a long winner-only Adam continuation. Every handoff uses the independent
forward spectral score, and an Adam result that regresses keeps its input.

### CMA-ES options

| Flag | Description | Default |
|------|-------------|---------|
| `--search legacy\|cma-es` | Select the global-search policy | `legacy` |
| `--cma-generations <N>` | Maximum CMA generations | `12` |
| `--cma-population <N>` | Candidates per generation; `0` selects `max(32, 4 + floor(3*log(D)))` | `0` (automatic) |
| `--cma-sigma <value>` | Initial standard deviation as a fraction of each transformed parameter range | `0.20` |
| `--cma-seed <N>` | Deterministic optimizer RNG seed | `1` |
| `--cma-forward-batch <N>` | Tensor lanes per render chunk; `0` uses the population size | `0` (automatic) |
| `--local-epochs <N>` | Independent Adam fallback from the user's seed; `0` disables it | value of `--epochs` |
| `--cma-continue <K>` | Diverse global candidates eligible for short Adam refinement | `3` |
| `--cma-refine-epochs <N>` | Short Adam epochs per continued candidate; `0` disables top-K refinement | value of `--epochs` |
| `--cma-refine-mode auto\|scalar\|batched` | Top-K refinement execution mode; `auto` uses the lane-parallel batched path only when `--cma-continue` >= 8, scalar otherwise (resolved mode is logged and reported as `refine_mode`) | `auto` |
| `--cma-final-epochs <N>` | Long scalar Adam continuation of the independently selected global winner | `0` |

### CMA-ES pipeline modes

All stages are independently configurable. These recipes map directly to
useful product/GUI modes:

```bash
# CMA only: no gradients
--search cma-es --local-epochs 0 \
--cma-refine-epochs 0 --cma-final-epochs 0

# CMA + local fallback
--search cma-es --local-epochs 300 \
--cma-refine-epochs 0 --cma-final-epochs 0

# CMA + short top-K polish
--search cma-es --local-epochs 0 \
--cma-continue 8 --cma-refine-epochs 5 \
--cma-refine-mode batched --cma-final-epochs 0

# CMA + short top-K polish + long winner training
--search cma-es --local-epochs 0 \
--cma-continue 8 --cma-refine-epochs 5 \
--cma-refine-mode batched --cma-final-epochs 300
```

The full pipeline is:

```text
optional local seed Adam ───────────────────────────────┐
                                                        ├─ final selection
CMA search → optional top-K Adam → select global winner │
                                  → optional final Adam ┘
```

The pre-Adam candidate is retained at both refinement boundaries. The local
fallback competes only at final selection and never changes CMA sampling.

For CMA-only fitting without gradient refinement:

```bash
dgenlisp train \
  --patch patch.lisp \
  --target target.wav \
  --seed-params seed.json \
  --job-dir runs/cma-only \
  --search cma-es \
  --local-epochs 0 \
  --cma-refine-epochs 0 \
  --cma-final-epochs 0
```

CMA runs write these artifacts under `--job-dir`:

- `final.wav` — independently selected final render
- `cma_es_report.json` — generation trace and separate local/global outcomes
- `cma_es_state.json` — serialized optimizer state
- `cma_best_generation_*.json` — best candidate retained from each generation

The complete algorithm, coordinate transformation, fitness, and artifact
schema are documented in [`docs/DGENLISP_CMA_ES_SPEC.md`](../../docs/DGENLISP_CMA_ES_SPEC.md).

### Experimental multistart baseline

The older one-shot tensor-lane multistart remains available as an experimental
baseline when `--search legacy` is selected:

| Flag | Description | Default |
|------|-------------|---------|
| `--multistart-candidates <N>` | Initial stratified candidates; `0` disables multistart | `0` |
| `--multistart-lanes <N>` | Diverse candidates retained for batched Adam | `64` |
| `--multistart-batch <N>` | Forward-render chunk size | `256` |
| `--multistart-steps <N>` | Short batched Adam horizon | `30` |
| `--multistart-seed <N>` | Deterministic population seed | `1` |

A multistart run writes `multistart_report.json`. `--multistart-candidates`
must be at least `--multistart-lanes` when enabled. CMA-ES takes precedence if
`--search cma-es` and multistart flags are both supplied.

Implementation caveats and protocol details are tracked in
[`docs/TRAIN_SUBCOMMAND_NOTES.md`](../../docs/TRAIN_SUBCOMMAND_NOTES.md).

## Language Reference

### Comments

```lisp
; line comment
# also a line comment
```

### Atoms

Numbers, symbols, and named constants:

```lisp
440           ; integer
3.14159       ; float
freq          ; symbol (must be defined with def or param)
pi            ; π
twopi         ; 2π (alias: tau)
e             ; Euler's number
true          ; 1.0
false         ; 0.0
```

### Special Forms

#### def — bind a name

```lisp
(def name expr)
(def osc (sin (* (phasor 440) twopi)))
(def (x y z) (tuple 1 2 3))
```

Destructuring `def` binds each name from a tuple-producing expression. Built-in multi-output
operators like `fft` return tuples, and macros can return explicit tuples with `(tuple ...)`.

#### defmacro — define a reusable macro

```lisp
(defmacro name (params...) body...)

(defmacro ap (sig g d)
  (make-history h)
  (def ds (delay (read-history h) d))
  (def v (+ sig (* g ds)))
  (write-history h v)
  (- ds (* g v)))

(defmacro multi (a b c)
  (tuple (* a 2) (* a b) (* b c)))
```

Local `def` and `make-history` bindings inside macros are automatically scoped — multiple calls to the same macro won't collide.

#### History feedback

```lisp
(make-history name)         ; create a feedback cell
(read-history name)         ; read previous frame's value
(write-history name expr)   ; write current frame's value (returns expr)
```

### I/O

#### param — host-controllable parameter

```lisp
(param name @default value @min value @max value @unit string
       @group group-name @env env-name @role attack|decay|sustain|release)

(param freq @default 440 @min 20 @max 20000 @unit Hz)
(param gain @default 0.5 @min 0 @max 1)
(param cutoff @default 2400 @min 60 @max 12000 @unit Hz @mod true @mod-mode additive)
(param amp-attack @group amp @env amp-env @role attack @default 0.01)
```

The name becomes a symbol you can use in expressions. Parameters appear in the manifest with their physical memory cell ID for host-side control.
Modulatable params generate one hidden active flag plus one hidden depth param per declared modulator, and expose those cells through `modDestinations` metadata in the manifest.

UI metadata attributes are optional and do not affect DSP behavior. `@group` places a param in a generated UI group. `@env` marks a param as part of an envelope, and requires a valid `@role`. Params in the same envelope cannot duplicate roles or declare conflicting groups.

#### in — audio input channel

```lisp
(in channel @name string)

(in 1 @name signal)     ; channel 1 (1-indexed)
(in 5 @name mod1 @modulator 1)
```

`@modulator <slot>` marks an input as a host-visible modulation source.

#### out — audio output channel

```lisp
(out expr channel @name string)

(out (sin (* (phasor 440) twopi)) 1 @name audio)
(out (phasor 0.25) 2 @name macro-a @modulator 1)
```

At least one `out` is required. Channel numbers are 1-indexed.
`@modulator <slot>` marks an output as a host-visible modulation output.

### Arithmetic

Binary operators auto-nest for 3+ arguments: `(+ a b c)` becomes `(+ (+ a b) c)`.

```lisp
(+ a b)      ; addition
(- a b)      ; subtraction
(- a)        ; negation
(* a b)      ; multiplication
(/ a b)      ; division
```

All arithmetic respects type promotion:

| Left | Right | Result |
|------|-------|--------|
| signal | signal | signal |
| tensor | tensor | tensor |
| signal | tensor | signalTensor |
| signalTensor | signal | signalTensor |
| signalTensor | tensor | signalTensor |
| any | float | promotes float |

### Math Functions

#### Unary

```lisp
(sin x)      (cos x)      (tan x)      (tanh x)
(exp x)      (log x)      (sqrt x)     (abs x)
(sign x)     (floor x)    (ceil x)     (round x)
(relu x)     (sigmoid x)
```

Work on signal, tensor, signalTensor, and float.

#### Binary

```lisp
(pow base exponent)
(min a b)
(max a b)
(% a b)
(mse prediction target)    ; mean squared error
```

`pow`, `min`, and `max` follow the same type-promotion rules as arithmetic operators.
`min` and `max` auto-nest like arithmetic operators.

### Comparison

Return 1.0 for true, 0.0 for false:

```lisp
(gt a b)     ; a > b
(lt a b)     ; a < b
(gte a b)    ; a >= b
(lte a b)    ; a <= b
(eq a b)     ; a == b
```

### Signal Generators

```lisp
(phasor freq)              ; ramp 0→1 at freq Hz
(phasor freq reset)        ; with reset trigger
(stateful-phasor freq)     ; forced stateful variant
(noise)                    ; white noise
(click)                    ; impulse: 1.0 on frame 0, then 0.0
```

`phasor` with a tensor frequency returns a signalTensor (one phasor per element).
Tensor phasors are **stateful**: each element gets its own persistent accumulator,
so they stay continuous across process-block boundaries.

### Stateful Operations

```lisp
(accum increment)                       ; accumulate, default range [0,1]
(accum increment reset min max)         ; with reset trigger and bounds
(latch value trigger)                   ; sample-and-hold
(mix a b t)                             ; linear interpolation: a*(1-t) + b*t
```

### Audio Effects

#### biquad — IIR filter

```lisp
(biquad signal cutoff q gain mode)

; or with attributes:
(biquad signal @cutoff 1000 @q 0.707 @gain 0 @mode 0)
```

Modes: 0=lowpass, 1=highpass, 2=bandpass, 3=notch, 4=allpass, 5=peaking, 6=lowshelf, 7=highshelf.

#### compressor

```lisp
(compressor signal ratio threshold knee attack release)

; or with attributes:
(compressor signal @ratio 4 @threshold -20 @knee 6 @attack 0.01 @release 0.1)

; with sidechain (7-arg positional):
(compressor signal ratio threshold knee attack release sidechain)

; with explicit isSidechain control (8-arg positional — isSidechain can be a modulatable signal):
(compressor signal ratio threshold knee attack release isSidechain sidechain)

; with sidechain (attribute style — sidechain must be a variable reference):
(compressor signal @ratio 4 @threshold -20 @knee 6 @attack 0.01 @release 0.1 @sidechain sc)
```

Works on both signal and signalTensor. When a sidechain is provided, level detection uses the sidechain signal instead of the main input.

#### delay

```lisp
(delay signal time_in_samples)
```

### Conditional

```lisp
(gswitch condition true_value false_value)
(selector mode option1 option2 ...)
```

`selector` is 1-based: `mode <= 0` returns `0`, `1` returns `option1`, `2` returns `option2`, and so on.

### Modulation

```lisp
(param cutoff @default 2400 @min 60 @max 12000 @unit Hz @mod true @mod-mode additive)
(def mod1 (in 5 @name mod1 @modulator 1))
(def filtered (biquad sig (mod cutoff) 0.8 1 0))
```

`(mod name)` resolves the generated modulated value for a parameter declared with `@mod true`.
Supported modulation modes are `additive`, `multiplicative`, and `semitone`.

### Utility

```lisp
(scale sig inMin inMax outMin outMax)  ; linear rescale
(triangle phase)                       ; phasor (0..1) → triangle (-1..1)
(wrap sig min max)                     ; wrap value to range
(clip sig min max)                     ; clamp value to range
```

### Tensor Creation

`tensor` is the single constructor for buffer-shaped data. It takes `@shape` plus
one source of contents (inline data, a JSON asset, or nothing = zeros):

```lisp
(tensor @shape [4 2] @data [0 0 0.5 0.5 1 1 0.5 0.5])   ; inline data
(tensor @shape [512 32] @file "waves/factory.json")     ; JSON asset
(tensor @shape [48000 2])                               ; zero-filled buffer
(tensor-param @shape [512 32] @name wave @default-file "waves/init.json")
```

- **`@data`** — inline float list; its length must equal the product of `@shape`.
- **`@file`** — JSON loaded relative to the compiled source file, or relative to
  `--asset-base` when that flag is provided. JSON may be a flat numeric array,
  nested numeric arrays, or an object with `shape` and `data`.
- **neither** — a zero-filled buffer. Record into it at runtime with `poke`.
- **`tensor-param`** — same surface, but host-writable (the host can push new
  contents by `@name`); `@default-file` seeds the initial contents.

The other constructors build tensors from a fill rather than from assets:

```lisp
(zeros [d1,d2,...])          ; zero-filled tensor
(zeros d1 d2)                ; same, with individual dims
(ones [d1,d2,...])           ; all-ones tensor
(full [d1,d2,...] value)     ; filled with constant
(randn [d1,d2,...])          ; random normal
```

The DGenLisp read convention is `(peek tensor index channel)`, so 2D buffers use
shape `[samples channels]` (a wavetable bank is `[samples waves]`). Flat data
should store each channel/wave contiguously. Fractional `index` and fractional
`channel` values are interpolated, so 2D reads are bilinear across sample
position and wave position.

### Tensor Operations

```lisp
(matmul a b)                           ; matrix multiply
(peek tensor index)                    ; interpolated scalar at raw index
(peek tensor index channel)            ; bilinear scalar at (index, channel)
(sample tensor phase channel)          ; scalar at normalized phase 0..1 (wrapped)
(peek-row tensor rowIndex)             ; read row at index → signalTensor
(to-signal tensor)                     ; 1D tensor → signal via playback
(to-signal tensor @max-frames 4096)    ; with explicit frame limit
```

The read family:

| Op | Reads | Index space |
| --- | --- | --- |
| `peek` | scalar | raw index in `[0, shape[0])`, interpolated |
| `sample` | scalar | normalized phase `0..1`, wrapped, scaled by `shape[0]` |
| `peek-row` | whole row → signalTensor | row index |
| `sampleRow` (Swift only) | whole row → signalTensor | fractional row index, cross-row blend |

`sample` is the gen-style, shape-aware read: `(sample t phase ch)` is exactly
`(peek t (* (wrap phase 0 1) N) ch)` where `N` is the tensor's compile-time
`shape[0]`. `channel` may be omitted (defaults to 0), but the 2D convention is
`[samples channels]` so it is normally supplied. There is deliberately no lisp
binding for the whole-row `sampleRow` read — it is a Swift/training-path API.

**Naming rule:** nouns are tensor-driven (`tensor`, `tensor-param`, `@shape`);
verbs follow Max/MSP gen (`peek`, `poke`, `sample`).

### Tensor Shape Operations

```lisp
(reshape tensor @shape [d1,d2,...])
(transpose tensor)                     ; reverse axes
(transpose tensor @axes [1,0])         ; specific axis permutation
(shrink tensor @ranges [0:2,1:3])      ; slice sub-tensor
(pad tensor @padding [1:1,0:0])        ; zero-pad (before:after per axis)
(expand tensor @shape [4,3])           ; broadcast expand
(repeat tensor @repeats [2,3])         ; tile/repeat
(conv2d input kernel)                  ; 2D convolution
```

### Reductions

```lisp
(sum tensor)                 ; sum all → scalar tensor
(sum tensor @axis 0)         ; sum along axis
(mean tensor)                ; mean all → scalar tensor
(mean tensor @axis 1)        ; mean along axis
(sum-axis tensor @axis 0)    ; explicit axis reduce
(mean-axis tensor @axis 0)
(max-axis tensor @axis 0)
(softmax tensor @axis -1)    ; softmax (tensor only)
```

### FFT

```lisp
(fft input)                  ; FFT, returns real part
(fft input N)                ; with explicit size
(ifft real imag)             ; inverse FFT
(ifft real imag N)           ; with explicit size
```

After `(fft x)`, the imaginary part is available as `__fft_im` and real as `__fft_re`.

### Windowing

```lisp
(buffer signal size)          ; ring buffer → [1, size] signalTensor
(buffer signal size hop)      ; with hop size
(overlap-add signalTensor hop) ; scatter-add into output signal
```

## Type System

DGenLisp has four value types:

| Type | Description |
|------|-------------|
| **float** | Compile-time constant (never hits the graph) |
| **signal** | Per-frame scalar (audio sample) |
| **tensor** | Static multi-dimensional array |
| **signalTensor** | Per-frame tensor (tensor that varies each audio frame) |

Floats are promoted automatically when combined with graph types. Signals and tensors produce signalTensors when combined.

## Manifest Format

```json
{
  "version": 1,
  "dylib": "patch.dylib",
  "sampleRate": 44100,
  "maxFrameCount": 4096,
  "totalMemorySlots": 256,
  "params": [{
    "name": "freq",
    "cellId": 84,
    "default": 440,
    "min": 20,
    "max": 20000,
    "unit": "Hz"
  }],
  "groups": [{"name": "amp"}],
  "envelopes": [{
    "name": "amp-env",
    "group": "amp",
    "roles": {
      "attack": "amp-attack",
      "decay": "amp-decay",
      "sustain": "amp-sustain",
      "release": "amp-release"
    }
  }],
  "inputs": [{"channel": 0, "name": "signal"}],
  "outputs": [{"channel": 0, "name": "audio"}],
  "modOutputs": [{
    "slot": 1,
    "channel": 1,
    "name": "macro-a",
    "range": "unipolar"
  }],
  "tensors": [{
    "name": "waves",
    "cellOffset": 100,
    "shape": [512, 32],
    "kind": "wavetable",
    "mutable": false,
    "sourceFile": "waves/factory.json"
  }],
  "tensorInitData": [{"offset": 100, "data": [0.5, ...]}]
}
```

- `cellId` values are **physical** memory offsets (after remapping), ready for direct indexing into the memory buffer
- `groups` and `envelopes` are derived from param UI metadata in first-reference order
- `tensors` gives named metadata for tensor-backed assets and editable tensor slots
- `tensorInitData` entries must be written to the memory buffer before the first `dgen_process_v1()` call
- `totalMemorySlots` is the required memory buffer size (in floats)

### Host Integration

The dylib includes `dgen_runtime.h` and exports exactly:

```c
void dgen_process_v1(
    const float *const *inputs,
    float *const *outputs,
    uint32_t frame_count,
    void *state,
    const DGenProcessContextV1 *context,
    const DGenHostServicesV1 *host
);

void dgen_set_param_value_v1(int32_t cell_id, float value);
```

`DGenProcessContextV1` supplies the runtime sample rate.
`DGenHostServicesV1` supplies only block-level FFT setup, forward/inverse FFT,
and complex multiply-accumulate callbacks. A host must initialize both
structures with ABI version 1 and their full structure sizes. See
`toolchain/include/dgen_runtime.h` and
`toolchain/harness/toolchain_harness.c` for the normative declaration and
reference host.

## Examples

### Simple oscillator

```lisp
(param freq @default 440 @min 20 @max 20000 @unit Hz)
(out (sin (* (phasor freq) twopi)) 1 @name audio)
```

### Stereo

```lisp
(def phase (phasor 440))
(out (sin (* phase twopi)) 1 @name left)
(out (cos (* phase twopi)) 2 @name right)
```

### Allpass reverb with macros

```lisp
(defmacro ap (sig g d)
  (make-history h)
  (def ds (delay (read-history h) d))
  (def v (+ sig (* g ds)))
  (write-history h v)
  (- ds (* g v)))

(def input (in 1 @name signal))
(out (ap (ap input 0.7 11) 0.7 17) 1 @name audio)
```

### Filtered noise

```lisp
(param cutoff @default 1000 @min 100 @max 10000 @unit Hz)
(param q @default 2 @min 0.5 @max 20)
(out (biquad (noise) cutoff q 0 0) 1 @name audio)
```

### Compressor on input

```lisp
(def input (in 1 @name signal))
(out (compressor input @ratio 4 @threshold -20 @knee 6 @attack 0.01 @release 0.1) 1 @name audio)
```

### Sidechain compressor

```lisp
(def input (in 1 @name signal))
(def side (in 2 @name sidechain @modulator 1))
(param threshold @min -40 @max -2 @default -20)
(param ratio @min 1 @max 20 @default 10)
(out (compressor input ratio threshold 6 0.01 0.1 1 side) 1 @name audio)
```

### AM synthesis

```lisp
(param carrier @default 440 @min 20 @max 2000 @unit Hz)
(param modfreq @default 5 @min 0.1 @max 100 @unit Hz)
(param depth @default 0.5 @min 0 @max 1)

(def mod (+ 1 (* depth (sin (* (phasor modfreq) twopi)))))
(def osc (sin (* (phasor carrier) twopi)))
(out (* osc mod) 1 @name audio)
```
