# SynthID fdcheck Finding

## Status: RESOLVED

> **Update 2026-07-06 (later same day):** Bug 2 was root-caused as gradient
> carry-cell memory aliasing under buffer reuse — NOT a BPTT scheduling bug —
> and fixed (one line: carry cells now registered in `persistentCells`).
> The remaining biquad-parameter truncation was fixed on 2026-07-09 by running
> detached gradient-carry blocks in reverse frame order. Full analysis:
> `docs/BIQUAD_BPTT_GRADIENT_BUG.md`. The "Bug 2" section below is the
> original isolation record.

The fdcheck mismatch was caused by two independent library bugs plus two
methodology issues in how the check was run. Investigated 2026-07-06.

## Bug 1 (FIXED): spectral gradients attenuated by (N·numBins)^1.5

`spectralLossFFT`'s backward path contained two ad-hoc normalizations, neither
mathematically justified:

1. `spectralLossFFTGradIFFT` scaled the IFFT result by `1/(windowSize * numBins)`.
   The unnormalized positive-twiddle IFFT of the gradient spectrum IS the exact
   transpose-DFT scatter, so no scale is needed. The factor originated in
   `spectralLossFFTGradInline`, whose comment said "less aggressive to allow
   learning" — a hand-tuned damping factor that the FFT path then copied for
   consistency.
2. `spectralLossFFTGradRead`/`GradRead2` (and batched variants) divided the
   summed cross-window gradient by `sqrt(numBins * windowSize)`.

Combined attenuation: `(N·numBins)^{-3/2}` — ×97,000 at window 64, ×6M at
window 256, ×800M at window 2048 (SynthID's config). Because the factor is
uniform, gradient DIRECTION was preserved: all existing direction-based tests
passed, and Adam's scale invariance let training appear to work.

**Fix**: removed all six normalization sites in
`Sources/DGen/Emit+SpectralLoss.swift` (scalar + batched GradIFFT, scalar +
batched GradRead/GradRead2, GradInline).

**Verification**: `Tests/DGenLazyTests/SpectralGradientMagnitudeTests.swift`
(new, permanent) asserts FD/autograd magnitude agreement — the coverage gap
that let this bug survive. After the fix, ratios are 1.000 ± 0.002% across
window sizes 64–256, frame counts 256–1024, hop 1–16, linear/log, l1/l2.

**Side effects**: spectral gradients are now ~10^5–10^9× larger than before.
Any code with learning rates tuned against the old scale needs retuning
(`OptimizerTests.testSignalParamOnepoleSpectral` lr was recalibrated
0.05 → 4e-6). SynthID LRs will need retuning downward similarly.

## Bug 2 (FIXED): trainable param behind history feedback zeroed other gradients

`Signal.biquad` is a macro that expands into primitive ops plus four
`historyRead`/`historyWrite` state cells. When any **gradient target sits
behind that history chain** (e.g. SynthID's trainable `noiseCutoff`),
`computeGradients` creates gradient carry cells for the history feedback,
which activates the BPTT loop-splitting machinery — and gradients for OTHER
params in the graph become wrong or exactly zero.

Minimal reproducer (window 256, frames 2048, log-l1 spectral loss, Metal):

```swift
let outGain = Signal.param(0.7, min: 0.4, max: 1.0)
let cutoff = Signal.param(log(2800.0))   // ← the trigger
let noise = Signal.noise().biquad(
  cutoff: exp(cutoff), resonance: Signal.constant(0.707),
  gain: Signal.constant(1.0), mode: Signal.constant(0.0))
let student = tanh((body + noise * env) * 2.0) * outGain
// cutoff CONSTANT   → outGain.grad = 1.77  (matches FD)
// cutoff TRAINABLE  → outGain.grad = 0.0 exactly; cutoff.grad = -3.2
```

With `--no-noise-filter` (biquad removed), SynthID fdcheck agrees with FD for
every parameter: outGain 0.2%, bodyAmp 1.2%, drive 3.2%, noiseAmp 0.4%,
ampDecay 4.7% (at `--fd-eps 1e-2`), clickFreq 7% (at `--fd-eps 3e-3`, and FD
converges toward autograd as eps shrinks — the frequency loss surface is
oscillatory, so large eps leaves the linear regime).

The carry-cell aliasing fix restored unrelated gradients. A second scheduling
fix now handles the biquad's own parameters under multi-kernel losses: a
backward-only block that reads and writes gradient carry cells executes in
reverse frame order. The filtered SynthID cutoff check now matches finite
differences to about 0.1% on the 2048-frame/256-window repro.

## Methodology issues in the original fdcheck runs

1. **Never fdcheck at the true parameters.** The original repro passed
   `--params true_params.json`, evaluating gradients at the global minimum of
   an L1-type loss — where the true gradient is a subgradient near 0 and
   central differences straddle |·| kinks. Use the midpoint init (omit
   `--params`) or any point away from the minimum.
2. **The log-magnitude loss with eps=1e-8 is ill-conditioned.** Near-empty
   bins get `1/(mag+1e-8)` ≈ 10^8 gradient amplification from the FFT noise
   floor. FD of this loss is unstable across epsilon (measured 15.0 / 0.68 /
   2.6 for eps 3e-3/1e-2/3e-2 on the same config) and autograd becomes
   slightly nondeterministic (float atomics × 10^8). For fdcheck gating,
   prefer the linear-magnitude term (FD-stable to 5 digits) or sweep
   `--fd-eps` and require FD stability before trusting the comparison.
   Longer-term library improvement: make the log-eps configurable (1e-4-ish)
   instead of the hardcoded 1e-8.
3. **`fdEpsilon = 1e-3` hits float32 loss-readback resolution** for
   small-gradient params (ampDecay showed 49% error at 1e-3, 4.7% at 1e-2).
   Sweep eps; require stability.

## Reproduction of the fixed state

```sh
swift test --filter SpectralGradientMagnitudeTests   # library-level FD magnitude gate

swift run SynthID train \
  --target /tmp/synthid-smoke/target.wav \
  --out /tmp/synthid-fd-body \
  --frames 2048 --windows 256 --no-linear-mag --no-noise-filter \
  --fd-eps 1e-2 --fdcheck bodyAmp
# → fd=-8.10e-01 autograd=-8.00e-01 relErr=1.2e-02
```
