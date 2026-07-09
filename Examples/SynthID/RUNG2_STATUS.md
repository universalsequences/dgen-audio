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

## Remaining hard gate

Run the default five-seed command and pass at least three seeds:

```bash
swift run SynthID rung2 --out /tmp/synthid-rung2-acceptance
```

Do not claim Rung 2 complete until that command exits successfully.
