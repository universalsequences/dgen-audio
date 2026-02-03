# DGenLazy Module Specification

A tinygrad-inspired lazy execution frontend for DGen, providing an ergonomic API for tensor operations, audio signal processing, and differentiable DSP.

## Design Philosophy

- **Lazy by default**: Operations build a computation graph; execution happens on `realize()`
- **Implicit graph management**: No explicit `Graph()` creation required
- **Type promotion**: Mixing `Signal` and `Tensor` produces `SignalTensor` (frame-varying tensor)
- **Tinygrad-compatible patterns**: `requiresGrad`, `backward()`, `.grad` property

---

## Core Types

### Tensor

Static multi-dimensional array. Does not vary per frame.

```swift
// Creation
let t1 = Tensor([1, 2, 3, 4])                          // From 1D array
let t2 = Tensor([[1, 2], [3, 4]])                      // From 2D array
let t3 = Tensor.zeros([3, 4])                          // Shape-based, filled with 0
let t4 = Tensor.ones([3, 4])                           // Shape-based, filled with 1
let t5 = Tensor.randn([64, 32])                        // Random normal distribution
let t6 = Tensor.full([3, 4], value: 0.5)               // Shape-based, custom fill

// Learnable tensors
let w = Tensor([0.1, 0.2, 0.3], requiresGrad: true)    // Explicit data
let w2 = Tensor.randn([64, 32], requiresGrad: true)    // Random init, learnable

// Properties
t1.shape      // [4]
t1.requiresGrad  // false
w.grad        // Optional Tensor, populated after backward()
```

### Signal

Frame-based scalar value. Varies per audio frame (sample).

```swift
// Creation
let s1 = Signal.constant(440.0)                        // Static scalar signal
let s2 = Signal.param(440.0)                           // Learnable (shorthand)
let s3 = Signal(440.0, requiresGrad: true)             // Learnable (explicit)
let s4 = Signal.input(0)                               // Audio input channel

// Audio generators
let osc = Signal.phasor(freq)                          // 0..1 ramp at freq Hz
let n = Signal.noise()                                 // White noise

// Properties
s2.requiresGrad  // true
s2.grad          // Optional Signal, populated after backward()
```

### SignalTensor

Frame-varying tensor. Each frame has a tensor of values. Created implicitly when mixing `Signal` and `Tensor`.

```swift
// Implicit creation via type promotion
let freqs = Tensor([440, 880, 1320])                   // [3] static
let phases = Signal.phasor(freqs)                      // [3] SignalTensor

let t = Tensor([0, 1, 1, 0])                           // [4] static
let s = Signal.phasor(440)                             // scalar signal
let st = t * s                                         // [4] SignalTensor

// Reducing to Signal
let mixed: Signal = st.sum()                           // Sum tensor dims -> scalar

// Properties
phases.shape     // [3]
phases.isFrameVarying  // true
```

---

## Type Promotion Rules

When mixing types, the result is promoted to the "stronger" (more dynamic) type:

| Operation | Left | Right | Result |
|-----------|------|-------|--------|
| `+`, `-`, `*`, `/` | `Tensor` | `Float` | `Tensor` |
| `+`, `-`, `*`, `/` | `Signal` | `Float` | `Signal` |
| `+`, `-`, `*`, `/` | `Tensor` | `Signal` | `SignalTensor` |
| `+`, `-`, `*`, `/` | `SignalTensor` | `Signal` | `SignalTensor` |
| `+`, `-`, `*`, `/` | `SignalTensor` | `Tensor` | `SignalTensor` |

**Hierarchy**: `Float` < `Tensor` < `Signal` < `SignalTensor`

**Broadcasting**: Standard numpy-style broadcasting applies for shape mismatches.

---

## Operations

### Arithmetic (all types)

```swift
// Operators
a + b, a - b, a * b, a / b
-a                                                     // Negation

// Functions (both global and methods)
abs(x)      // x.abs()
sign(x)     // x.sign()
floor(x)    // x.floor()
ceil(x)     // x.ceil()
round(x)    // x.round()
mod(x, y)   // x.mod(y)
min(x, y)   // x.min(y)
max(x, y)   // x.max(y)
clamp(x, lo, hi)  // x.clamp(lo, hi)
```

### Math Functions (all types)

```swift
exp(x)      // x.exp()
log(x)      // x.log()
log10(x)    // x.log10()
sqrt(x)     // x.sqrt()
pow(x, y)   // x.pow(y)
sin(x)      // x.sin()
cos(x)      // x.cos()
tan(x)      // x.tan()
tanh(x)     // x.tanh()
atan2(y, x) // y.atan2(x)
```

### Tensor Operations

```swift
// Shape operations
x.reshape([2, 3])
x.transpose()                                          // 2D transpose
x.transpose(axes: [1, 0, 2])                           // Arbitrary axes

// Reductions
x.sum()                                                // Sum all elements
x.sum(axis: 0)                                         // Sum along axis
x.mean()
x.mean(axis: 1)

// Linear algebra
matmul(a, b)    // a.matmul(b) or a @ b
conv1d(x, kernel, stride: 1)
conv2d(x, kernel, stride: 1)

// Indexing
x[0]                                                   // First element/row
x.peekRow(index)                                       // Dynamic row indexing (for SignalTensor)
```

### Activation Functions

```swift
relu(x)         // x.relu()
sigmoid(x)      // x.sigmoid()
softmax(x)      // x.softmax()
logSoftmax(x)   // x.logSoftmax()
```

### Signal-Specific Operations

```swift
// Oscillators
Signal.phasor(freq)                                    // Ramp oscillator
Signal.phasor(freq, reset: trigger)                    // With reset trigger

// Filters
signal.biquad(cutoff: 1000, q: 0.707, mode: .lowpass)
signal.onepole(cutoff: 0.5)                            // Simple lowpole

// State operations
signal.history()                                       // Previous frame's value
signal.latch(trigger)                                  // Sample & hold
signal.accum()                                         // Running sum

// Utilities
signal.delta()                                         // Difference from previous
signal.change()                                        // Sign of difference
```

### Loss Functions

```swift
mse(pred, target)           // Mean squared error
spectralLoss(pred, target, windowSize: 1024)  // FFT-based spectral loss
```

---

## Execution

### realize()

Compiles the computation graph and executes it, returning raw data.

```swift
// Tensor (static)
let values: [Float] = tensor.realize()

// Signal (frame-based)
let samples: [Float] = signal.realize()                // Uses DGen.defaultFrameCount
let samples2: [Float] = signal.realize(frames: 1024)   // Explicit frame count

// SignalTensor (frame-varying tensor)
let data: [Float] = signalTensor.realize(frames: 64)   // Flat array: frameCount * tensorSize
// Data layout: [frame0_elem0, frame0_elem1, ..., frame1_elem0, frame1_elem1, ...]
```

### State Management

Audio signals have internal state (oscillator phase, filter history).

```swift
// Training: implicit reset per realize() call
for _ in 0..<100 {
    let loss = mse(filtered, target)
    loss.backward()
    opt.step()
    // State is reset before next forward pass
}

// Inference: preserve state across calls
let out1 = signal.realize(frames: 512, preserveState: true)
let out2 = signal.realize(frames: 512, preserveState: true)  // Continues from out1
```

---

## Training

### backward()

Computes gradients for all tensors/signals with `requiresGrad: true`.

```swift
let w = Tensor.randn([64, 32], requiresGrad: true)
let pred = relu(matmul(input, w))
let loss = mse(pred, target)

loss.backward()                                        // Compute gradients

// Access gradients
let gradValues: [Float] = w.grad!.realize()            // Gradient is also lazy
```

### Optimizers

```swift
// Create optimizer with explicit parameter list
let opt = Adam(params: [w1, w2, b1, b2], lr: 0.001)
let opt2 = SGD(params: [w], lr: 0.01, momentum: 0.9)

// Training loop
for epoch in 0..<100 {
    let loss = computeLoss()
    loss.backward()
    opt.step()                                         // Update parameters
    opt.zeroGrad()                                     // Clear gradients
}
```

#### Available Optimizers

| Optimizer | Parameters |
|-----------|------------|
| `SGD` | `lr`, `momentum`, `weightDecay`, `nesterov` |
| `Adam` | `lr`, `beta1`, `beta2`, `eps` |

---

## Configuration

Global configuration via static properties:

```swift
DGen.backend = .metal                                  // .metal (default) or .cpu
DGen.sampleRate = 44100.0                              // Audio sample rate (default: 44100)
DGen.defaultFrameCount = 1024                          // Default frames for realize()
```

---

## Error Handling

- **Shape mismatches**: Runtime error on `realize()` or graph construction
- **Type errors**: Compile-time where Swift type system catches them
- **Invalid operations**: Runtime error with descriptive message

```swift
// Example errors
let t1 = Tensor([1, 2, 3])
let t2 = Tensor([1, 2])
let _ = matmul(t1, t2)  // Runtime error: incompatible shapes for matmul
```

---

## Complete Example: Audio Filter Training

```swift
import DGenLazy

// Configure
DGen.sampleRate = 44100.0
DGen.defaultFrameCount = 1024

// Create learnable filter parameters
let cutoff = Signal.param(1000.0)
let resonance = Signal.param(0.5)

// Build audio graph
let input = Signal.input(0)
let filtered = input.biquad(cutoff: cutoff, q: resonance, mode: .lowpass)
let target = Signal.input(1)

// Loss
let loss = mse(filtered, target)

// Optimizer
let opt = Adam(params: [cutoff, resonance], lr: 0.001)

// Training loop
for epoch in 0..<1000 {
    loss.backward()
    opt.step()
    opt.zeroGrad()

    if epoch % 100 == 0 {
        let lossValue = loss.realize(frames: 1024).reduce(0, +) / 1024
        print("Epoch \(epoch): loss = \(lossValue)")
    }
}

// Inference
let output = filtered.realize(frames: 4096, preserveState: true)
```

---

## Complete Example: Neural Audio Synthesis

```swift
import DGenLazy

// MLP weights
let w1 = Tensor.randn([1, 64], requiresGrad: true)
let b1 = Tensor.zeros([64], requiresGrad: true)
let w2 = Tensor.randn([64, 16], requiresGrad: true)
let b2 = Tensor.zeros([16], requiresGrad: true)

// Time input (control rate)
let time = Tensor((0..<32).map { Float($0) / 31.0 }).reshape([32, 1])

// MLP -> harmonic amplitudes
let h1 = tanh(matmul(time, w1) + b1)
let amps = sigmoid(matmul(h1, w2) + b2)  // [32, 16] - 32 time steps, 16 harmonics

// Harmonic frequencies
let f0: Float = 100.0
let harmonics = Tensor((1...16).map { f0 * Float($0) })  // [16]

// Audio-rate playhead
let playhead = Signal.phasor(DGen.sampleRate / Float(DGen.defaultFrameCount))
let frameIdx = playhead * 31.0  // 0..31

// Get amplitudes at current time
let ampsAtTime = amps.peekRow(frameIdx)  // [16] SignalTensor

// Generate harmonics
let phases = Signal.phasor(harmonics)     // [16] SignalTensor
let sines = sin(phases * 2 * .pi)         // [16] SignalTensor
let output = (sines * ampsAtTime).sum()   // Signal (mixed down)

// Train against target
let target = Signal.input(0)
let loss = spectralLoss(output, target, windowSize: 2048)

let opt = Adam(params: [w1, b1, w2, b2], lr: 0.001)

for epoch in 0..<500 {
    loss.backward()
    opt.step()
    opt.zeroGrad()
}
```

---

## Module Structure

```
Sources/DGenLazy/
├── DGenLazy.swift          # Public API, configuration
├── Tensor.swift            # Tensor type and operations
├── Signal.swift            # Signal type and operations
├── SignalTensor.swift      # SignalTensor type (internal promotion)
├── Operators.swift         # Operator overloads (+, -, *, /, etc.)
├── Functions.swift         # Global functions (sin, cos, relu, etc.)
├── Graph.swift             # Implicit graph management
├── Realize.swift           # Compilation and execution
├── Backward.swift          # Gradient computation
├── Optimizers.swift        # Adam, SGD
└── SPEC.md                 # This specification
```

---

## Relationship to Existing DGen

DGenLazy is a **thin wrapper** around the existing DGen infrastructure:

| DGenLazy | DGen (underlying) |
|----------|-------------------|
| `Tensor` | `Graph.tensor()`, `TensorParameter` |
| `Signal` | `Graph.n(.constant)`, `GraphParameter` |
| `backward()` | `Graph.computeGradients()` |
| `realize()` | `CompilationPipeline.compile()` + `MetalCompiledKernel.run()` |
| `Optimizer` | `GraphAdam`, `GraphSGD` |

The lazy graph builds `NodeID` references internally, then delegates to existing compilation and execution infrastructure.
