# C execution gates

`(block-gate condition body)` returns `body` where `condition > 0`, and zero
elsewhere. `body` may be a scalar or a tuple of scalars. Unlike `gswitch`, its
exclusively owned scalar DSP is not executed in a process call where the
condition is non-positive for every frame. Its state freezes during skipped
calls. Re-enabling resumes that state; reset is not implied.

For a frame-invariant parameter condition, this is conventional conditional
execution. Audio-rate conditions have an explicit block contract: if even one
frame is enabled, the body advances for the whole call, and its output is
masked per frame. Thus a fade can finish before its section stops executing.
The phase at re-enable for audio-rate conditions can depend on process-call
boundaries. Use a parameter condition when block-independent freezing matters.

```
(param enabled @default 1 @min 0 @max 1)
(out (block-gate enabled (sin (* (phasor 220) twopi))) 1)
```

Gates preserve shared consumers. A producer used by two gates executes if
either needs it; a producer with an ungated consumer always executes. The
analysis follows implicit history read/write connections, so it cannot skip
only half of a recurrence. Predicate dependencies are always evaluated before
choosing regions; in particular, a shared calculation needed to decide whether
another gate is enabled cannot itself wait for that downstream gate.

The first implementation supports scalar DSP on the C backend. Tensor/hop
regions and arbitrary memory side effects in a gate's dependency cone are
rejected. Metal rejects explicit `block-gate`; eager `gswitch` and `selector`
retain their existing behavior on both backends. A gate is not a declaration
that all DSP influencing an output is disposable: shared producers and
predicate preparation remain unconditional. Feedback state still requires
normal initialization and note-on/retrigger handling.

## Per-frame gates

`(sample-gate condition body)` is the per-frame counterpart. On every frame
where `condition > 0` the body runs; on the others its owned state holds and
the result is zero. The condition is tested inside the sample loop, so it may
read state the body writes: a voice gated on its own envelope reads the
envelope the body wrote on the previous frame.

```
(def trig (eq (accum 1 0 0 4800) 0))
(make-history h_env)
(def alive (or trig (gt (read-history h_env) 0)))
(out (sample-gate alive
       (do
         (def env (gswitch trig 1 (* (read-history h_env) 0.999)))
         (write-history h_env (gswitch (lt env 0.0001) 0 env))
         (* env (sin (* twopi (phasor 440))))))
     1)
```

Ownership is lexical. A history, delay line or stateful op (phasor, latch,
noise, ...) created while evaluating the body belongs to the gate and holds
while it is closed. State created outside every sample-gate keeps running
even if only gated code reads it. A nested sample-gate owns its own state.
Shared stateless producers run whenever any consumer needs them; arithmetic
wanted by many alternative gates (say, by each of 32 voices) runs under the
conditions the alternatives share instead of testing them all. `(do e1 ... en)`
evaluates its forms in order and returns the last, so a body can hold
definitions and history writes.

Gated bodies of one feedback region stay in its sample loop as `if` blocks.
The scheduler keeps each gate's nodes together, and values passed between
fragments of one loop are scalars, not scratch buffers. C backend only.

## One sample loop

`(scalar-loop)` compiles the whole patch as a single sample-by-sample loop.
Patches that are mostly feedback (granular processors, reverbs, physical
models) otherwise split into many small frame loops, with values crossing them
through scratch buffers. A patch that mostly computes tensors (spectral
processing, convolution) should not use it.

## Unassigned modulation

The C compiler lowers `modulatedParam` into a conditional arithmetic region.
When the host's destination-active cell is zero, modulation contributions and
their clamping/scaling are skipped, and the base value is selected unchanged.
Assigning a source activates the full audio-rate path on the next process call.
Additive, multiplicative, and semitone modes keep their existing semantics.

This lowering is scoped to compilation: the original graph operator survives
for differentiation, subsequent compilation, and Metal. Parameter and depth
cell identities remain host-addressable. The implementation gates unused work;
it does not yet specialize the entire downstream graph to a different rate
when a modulation assignment changes.

## Validation

`swift test --filter 'ExecutionGateTests|ModulationTests|CodegenPerfPassesTests'`
with the configured hermetic stage exercises freeze/resume at 1/8/64 frames,
audio-rate masking, nested and shared gates, predicate ordering, compilation
restoration, assignment changes in all three modes, and 1/12 voice compilation.

Heat application and measurements live in eseq's `tools/heat` directory. Never
replace an exponential fade by a raw enable predicate: a never-exactly-zero
fade never permits skipping, and testing the raw enable cuts the fade short.
Heat uses a finite 2 ms enable ramp; other continuous controls keep their
existing smoothing.
