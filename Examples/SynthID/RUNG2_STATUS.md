# SynthID Rung 2 Status

## Seed-1 pilot passed (2026-07-09)

Command:

```bash
swift run SynthID rung2 --seed 1 --out /tmp/synthid-rung2-seed1-float
```

This is the full default configuration: 32,768 frames, five restarts, 600 main
epochs, 300 pitch-refinement epochs, and the filtered noise path enabled.

### Renderer gate

- Max absolute NumPy/DGen sample error: `4.87268e-6`
- RMS error: `7.41604e-7`
- Required max error: `< 1e-3`
- Result: pass

The NumPy renderer writes IEEE float32 WAV. PCM16 was explicitly rejected after
an early pilot showed that quantization noise created a large log-STFT floor in
otherwise empty bins.

### Recovery gate

- Initial loss: `0.791955`
- Final loss: `0.021444`
- Measured renderer loss floor: `0.021327`
- Raw loss ratio: `0.027077`
- Floor-adjusted loss ratio: `0.000152` (required: `<= 0.02`)
- Result: pass

Every scored parameter passed. Representative relative errors:

- `fStart`, `fEnd`: below `0.001%`
- `pitchDecay`, `ampDecay`: below `0.004%`
- `clickFreq`: `0.01%`
- `noiseCutoff`: below `0.003%`
- `noiseDecay`: `0.01%`
- `outGain`: `0.01%`
- Effective `body`, `click`, and `noise` gain products: at most `0.04%`

The individual amplitude and drive factors remain intentionally unscored because
only their products are identifiable.

## Five-seed acceptance passed (2026-07-09/10)

Command:

```bash
swift run SynthID rung2 --out /tmp/synthid-rung2-acceptance
```

The command exited successfully. Four of five seeds passed, exceeding the
required three-seed threshold.

| Seed | Result | Initial loss | Final loss | Renderer floor | Floor-adjusted ratio |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | pass | 0.627485 | 0.021498 | 0.021327 | 0.000283 |
| 2 | fail | 0.650968 | 0.072777 | 0.014787 | 0.091153 |
| 3 | pass | 0.957651 | 0.007624 | 0.007106 | 0.000545 |
| 4 | pass | 1.464444 | 0.019768 | 0.019587 | 0.000125 |
| 5 | pass | 0.868312 | 0.012137 | 0.005621 | 0.007552 |

All five external-renderer equivalence checks passed. The worst maximum absolute
sample error was `5.11855e-6`, well below the required `1e-3` threshold.

Every scored parameter passed on seeds 1, 3, 4, and 5. Seed 2 missed the loss
ratio gate and recovered `fStart` with `5.27%` relative error against the `3%`
tolerance; its remaining scored parameters passed.

## Status

Rung 2 is complete. The next implementation milestone is Rung 3.
