# SynthID Rung 3 Status

## First real-target baseline (2026-07-10)

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

### Independent acceptance result

The NumPy comparator independently computes multi-resolution log-magnitude STFT
distance with windows `[256, 512, 1024, 2048]` and `log epsilon = 1e-3`.

- Initialization distance: `0.992795`
- Learned distance: `0.797225`
- Improvement: `19.70%`
- Required improvement: `80.00%`
- Result: **fail**

Per-window improvement was `43.57%`, `28.25%`, `13.40%`, and `6.22%`
respectively from shortest to longest window. The fit improves the transient and
short-time spectrum much more than the resolved harmonic structure.

### Finding

The recovered body follows the target's approximately 55-to-49 Hz early contour
and decay, but the target retains a stronger upper-harmonic ladder than the
learned voice. The overlay shows substantial agreement at the fundamental and
cyan/magenta separation above it. The current symmetric `tanh` body is too smooth
to reproduce that even/upper-harmonic structure. The recovered click amplitude
also reaches its upper bound (`0.6`), while filtered noise falls nearly to zero,
so more restarts or noise tuning are not credible fixes.

Artifacts:

- `/tmp/synthid-rung3-808-full/compare.png`
- `/tmp/synthid-rung3-808-full/ab.wav`
- `/tmp/synthid-rung3-808-full/report.md`
- `/tmp/synthid-rung3-808-full/compare.json`

## Status and next model step

The Rung 3 harness is implemented and the first real-target baseline is complete.
Rung 3 itself is **not complete**.

The next model experiment should add a small trainable asymmetric body-shaping
term (zero by default so Rungs 1 and 2 retain their existing signal path) to
represent even harmonics without target-specific tables. Then rerun this target
and only broaden the click path if the independent overlay still isolates the
attack as the dominant residual.
