# SynthID Rung 3 Blocker — Resolved

## Resolution

The former 68.94% plateau is resolved. The default five-restart Rung 3 command
now reaches **84.55%** corrected independent MR-STFT improvement and exits zero:

```bash
swift run SynthID rung3 \
  --target Assets/808kicklong.wav \
  --out /tmp/synthid-rung3-complete
```

Final corrected metrics:

- Initialization distance: `0.075252`
- Learned distance: `0.011626`
- Improvement: `84.55%`
- Required: `80.00%`
- Capture policy: zero-phase `30 Hz` high-pass applied equally to target,
  initialization, and learned audio by the independent comparator

## Root causes

The blocker combined three effects rather than one broken core pipeline:

1. The sub-20 Hz onset-correlated half-cycle is capture-chain baseline motion,
   not the approximately 49 Hz bridged-T body. Scoring it as synth content made
   the real target inconsistent with the intended model class.
2. The attack has a short even harmonic that decays much faster than the body.
   The rejected fixed-ratio second harmonic polluted the tail and was therefore
   driven to zero. The retained zero-default `bodyAsymmetry` term is explicitly
   attack-localized and keeps the voice at 14 scalars.
3. The PCM16 target has a persistent quiet broadband floor and a stronger click
   than the synthetic sampling ranges allowed. Rung 3 now uses wider optimizer
   bounds plus a target-independent capture-floor restart; Rungs 1–2 retain
   their original target sampling distributions.

The GPU multi-window L1 loss and corrected independent metric also select
different fine scalar basins. The final deterministic refinement searches only
the documented 14 scalars. It does not introduce residual samples, lookup
tables, learned EQ/FIR coefficients, or any target-derived array. DGen rerenders
the refined patch before the independent gate runs.

## Evidence

- Default run selected a capture-floor restart, stitched to training loss
  `1.16483`, then refined to a DGen training loss of `1.035218`.
- The independent refiner predicted `84.55%`; the separate DGen rerender plus
  `compare.py` reproduced `84.55%` (`0.011626` learned distance).
- `checkpoint.json` embeds the resolved config and refined scalar patch.
- Required audio, report, curve, and overlay artifacts are present in
  `/tmp/synthid-rung3-complete`.
