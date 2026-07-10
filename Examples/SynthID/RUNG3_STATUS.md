# SynthID Rung 3 Status

## Corrected real-target baseline (2026-07-10)

Target: `Assets/808kicklong.wav`

- Source: mono PCM16, 32.5 kHz, 37,759 frames (1.1618 s)
- Detected onset: frame 3 (`0.0923 ms`) at the default `-40 dB` threshold
- Training copy: windowed-sinc resampled to 44.1 kHz, onset aligned,
  peak-normalized, and cropped to 32,768 frames (0.7430 s)
- Conversion metadata: `/tmp/synthid-rung3-808-full/preprocessing.json`

Command:

```bash
swift run SynthID rung3 \
  --target Assets/808kicklong.wav \
  --out /tmp/synthid-rung3-808-full
```

The command ran the default five restarts, pitch refinement, cross-restart
subspace stitching, and click-frequency search. It exited nonzero because the
independent Rung 3 acceptance gate did not pass, while still writing the full
diagnostic artifact set.

### Optimization result

| Candidate | Final training loss |
| --- | ---: |
| Restart 1 | 1.666765 |
| Restart 2 | 1.659171 |
| Restart 3 | 1.703847 |
| Restart 4 | 1.656744 |
| Restart 5 | 1.703751 |
| Stitched + tuned | 1.649934 |

The selected cold-start loss was `2.127566`, so the internal training-loss ratio
was `0.775503`.

### Comparator correction

The first report incorrectly applied the fixed `1e-3` epsilon to raw FFT
magnitudes. Raw magnitudes grow with window size, so the claimed `-60 dBFS`
floor was neither referenced to full scale nor consistent across windows. The
independent comparator now divides each spectrum by the Hann window's coherent
gain (`sum(window) / 2`) before applying the epsilon.

The corrected independent result is:

- Initialization distance: `0.056483`
- Learned distance: `0.017543`
- Improvement: `68.94%`
- Required improvement: `80.00%`
- Result: **fail**

| FFT window | Improvement |
| ---: | ---: |
| 256 | 78.88% |
| 512 | 71.64% |
| 1024 | 53.15% |
| 2048 | 31.28% |

The previously reported `19.70%` result is invalid and must not be used.

### Finding

The learned RMS envelope is already close to the target throughout the sound:
within about 2% from 0.10–0.50 s and within about 5% in the final 0.24 s. The
largest residual instead appears at longer FFT windows in every time segment.
This points to fine pitch/spectral-trajectory mismatch rather than a missing
gross decay envelope or attack-only problem.

A local `fEnd` sweep confirms that the recovered pitch endpoint is already at
the narrow independent optimum: `49.200 Hz` gives `68.95%`, essentially the
same as the trained `49.204 Hz`. Moving to the CPU pitch fit's `49.491 Hz`
reduces improvement to `49.14%`; replacing the full pitch trio reduces it to
`31.34%`. The remaining 11-point gap is therefore not hidden behind a simple
pitch-freeze fallback.

Artifacts:

- `/tmp/synthid-rung3-808-full/compare.png`
- `/tmp/synthid-rung3-808-full/ab.wav`
- `/tmp/synthid-rung3-808-full/report.md`
- `/tmp/synthid-rung3-808-full/compare.json`

## Negative follow-up experiments

These experiments were evaluated and deliberately not retained in the voice:

- A zero-default asymmetric body shaper improved the matched 120+60 epoch pilot
  from `61.01%` to only `61.98%`.
- A zero-default fixed second harmonic scored `60.61%`; its learned coefficient
  returned to approximately zero.
- Coherent-gain normalization inside the training loss improved the matched
  short pilot to `62.97%`, but the full five-restart run scored only `67.72%`,
  below the corrected `68.94%` baseline. Its normalized forward/backward math
  passed gradient tests, so this was an optimization-result failure rather than
  an autograd implementation failure. Full artifacts are in
  `/tmp/synthid-rung3-808-normalized-full`.
- A zero-default second-exponential pitch-curvature term cleared the short-pilot
  gate (`64.65%` versus `61.01%`) and passed finite differences away from the
  zero-point L1 cusp (`1.79%` relative gradient error). The full five-restart
  run nevertheless scored only `59.30%`. Its selected cold start was already
  closer to the target, but the absolute learned distance also lost to the
  retained baseline (`0.017713` versus `0.017543`), so the term was removed.
  Full artifacts are in `/tmp/synthid-rung3-pitch-curve-full`.

## Status and next model step

The Rung 3 harness and corrected independent evaluator are implemented. Rung 3
itself is **not complete** because the corrected score remains below 80%.

The next model experiment should test a zero-default initial body-phase offset.
The target's RMS envelope and recovered pitch endpoint are already close, while
the synthesized body is forced to start at sine phase zero after threshold-based
onset alignment. A generic phase offset can address finite-window leakage without
adding target-specific data or changing the pitch contour. Validate it first
with an independent phase grid, then the matched short pilot and finite
differences; do not run the full schedule unless both diagnostics improve.
