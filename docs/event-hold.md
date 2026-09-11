# Event-rate coefficient graphs

`(event-hold value trigger)` samples scalar or tensor values whenever the scalar
trigger is positive. Pure calculations derived from those values run on the
same event frames. Unlike `hop-hold`, the trigger can be irregular, including
adjacent frames and events between periodic control ticks. Uses with the same
trigger expression share one clock.

Use ordinary `latch` to bring each final coefficient back to the audio rate:

```lisp
(def onset (in 1))
(def tick (max onset (eq (accum 1 0 0 16) 0)))
(def cutoff (event-hold (in 2) tick))
(def pole (latch (exp (/ (* -1 twopi cutoff) samplerate)) tick))
(make-history previous)
(def filtered (mix (in 3) (read-history previous) pole))
(write-history previous filtered)
(out filtered 1)
```

The expensive exponential runs only on ticks; filter state advances every
sample. Holding the input with an ordinary `latch` alone does not schedule the
subsequent exponential at event rate.

As with periodic hop tensors, an event-rate tensor or derived event expression
is defined only on its event frames. Reading it directly at audio rate yields
zero between events. The scalar `event-hold` latch itself retains its sampled
value, but this does not make its derived expression an audio-rate coefficient.
Always use the explicit final `latch` when a continuous held value is required.
Before the first event the held value is zero. Combining independent clocks
requires latching each branch back to audio rate first; otherwise compilation
fails instead of choosing an arbitrary clock.

The trigger and producer still run at their original rates. This operator
schedules pure consumers, not an arbitrary stateful computation. It currently
supports C forward rendering only. Metal and automatic differentiation reject
graphs using it with a compile-time diagnostic. Calibration can keep its eager
coefficient graph while the forward instrument uses event scheduling.

Mixed-rate feedback fragments share one sample loop in the execution schedule.
The renderer and buffer allocator consume that same region, so scratch values
remain live across every fragment of the loop. Event scratch uses dense frame
slots because adjacent events cannot share a compressed periodic-hop slot.
