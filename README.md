# DGen

A Swift DSP compiler inspired by [Max/MSP's Gen~](https://docs.cycling74.com/max8/vignettes/gen_overview) with automatic differentiation. Compiles computation graphs to optimized **Metal** (forward + backward) and **C** (forward) backends.

## Quick Start

```swift
import DGenLazy

// Create a sine wave oscillator
let freq = Signal.param(440.0)  // Learnable frequency
let osc = Signal.phasor(freq)
let wave = sin(osc * 2.0 * Float.pi)

// Run forward pass
let samples = try wave.realize(frames: 1024)
```

## Examples

### Tensor Operations

```swift
let a = Tensor([1, 2, 3, 4])
let b = Tensor([4, 5, 6, 7])
let result = ((a + b) * 2.0).relu().sum()
let values = try result.realize()  // [28.0]
```

### Audio DSP with Gradients

```swift
// Learnable parameters
let freq = Signal.param(300.0)
let lfoFreq = Signal.param(5.0)

// Modulated sine wave
let carrier = sin(Signal.phasor(freq) * 2.0 * Float.pi)
let lfo = Signal.phasor(lfoFreq)
let output = carrier * lfo

// Target signal
let target = sin(Signal.phasor(440.0) * 2.0 * Float.pi) * Signal.phasor(10.0)

// Train with MSE loss
let loss = mse(output, target)
try loss.backward(frameCount: 4096)
```

### Bank of Oscillators

```swift
let frequencies = Tensor([440, 880, 1320, 1760])
let phases = Signal.phasor(frequencies)
let waves = sin(phases * 2.0 * Float.pi)
let mix = waves.sum()  // Sum to mono
```

### End-to-End Training with Matmul

Learn harmonic amplitudes via matmul to match a target timbre:

```swift
import DGenLazy

// Matmul maps input to 4-harmonic amplitudes
let input = Tensor([[1.0]])                         // [1, 1]
let weights = Tensor.param([1, 4])                  // Learnable amps
let freqs = Tensor([100, 200, 300, 400])            // Harmonic series

let opt = Adam(params: [weights], lr: 0.05)
for _ in 0..<100 {
    // Rebuild graph each iteration (cleared after backward)
    let amps = input.matmul(weights)
    let phases = Signal.phasor(freqs)
    let waves = sin(phases * 2.0 * Float.pi) * amps.reshape([4])
    let output = waves.sum()

    // Target: 1st + 3rd harmonic
    let target = sin(Signal.phasor(100.0) * 2.0 * Float.pi)
               + sin(Signal.phasor(300.0) * 2.0 * Float.pi)

    let loss = mse(output, target)
    try loss.backward(frames: 1024)
    opt.step()
    opt.zeroGrad()
}
// weights converges to ~[1, 0, 1, 0]
```

### Hop-Gated FFT Pipeline

When processing audio in overlapping windows, most of the heavy computation (FFT, spectral processing, IFFT) only needs to run every *hop* frames — not every sample. DGen handles this automatically via **hop gating**.

```swift
let sig = sin(Signal.phasor(440.0) * 2.0 * Float.pi)

// buffer(hop:) collects samples into a sliding window.
// With N=1024 and hop=256, the FFT only runs every 256 frames.
let buf = sig.buffer(size: 1024, hop: 256)

// Everything downstream of a hop-gated buffer is also hop-gated:
// these tensor ops (FFT butterflies, multiplies) run once per hop,
// not once per frame.
let flat = buf.reshape([1024])
let (re, im) = signalTensorFFT(flat, N: 1024)
// ... spectral processing ...
let recon = signalTensorIFFT(re, im, N: 1024)

// overlapAdd converts back to frame-based (one sample per frame).
// It scatter-adds each hop's window into a ring buffer and emits
// one sample every frame — this is where hop-gated → frame-based.
let out = recon.overlapAdd(hop: 256)

let samples = try out.realize(frames: 4096)
```

The execution timeline looks like this:

```
Frame:    0    1    2   ...  255  256  257  ...  511  512 ...
          │    │    │         │    │    │         │    │
buffer:   ✓    ✓    ✓    ✓    ✓    ✓    ✓    ✓    ✓    ✓   ← writes every frame
FFT:      ✓                        ✓                   ✓   ← runs every 256 frames
IFFT:     ✓                        ✓                   ✓
overlap:  ✓    ✓    ✓    ✓    ✓    ✓    ✓    ✓    ✓    ✓   ← emits every frame
```

The `buffer(hop:)` call creates an internal hop counter. On each frame, it writes the input sample to a circular buffer. Every *hop* frames the counter resets, triggering the downstream tensor blocks (FFT, IFFT). The `overlapAdd` node runs every frame regardless — it reads from its own ring buffer, returning the accumulated result of all previous overlapping windows.

## Backend Support

| Backend | Forward | Backward | Notes |
|---------|---------|----------|-------|
| Metal   | ✓       | ✓        | GPU-accelerated, full autodiff |
| C       | ✓       | —        | SIMD-optimized for Apple Silicon |

## Building

```bash
swift build
swift test
```
