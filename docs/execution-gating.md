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
