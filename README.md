# DGen

A Swift-based audio DSP compiler that generates optimized code for real-time audio processing. DGen compiles computation graphs to Metal (GPU) and C (SIMD) backends with automatic gradient computation for machine learning.

Inspired by:
- **tinygrad** - automatic differentiation and kernel fusion approach
- **Gen (Max MSP)** - high-level DSP primitives and graph-based audio processing

The C backend generates high-performance SIMD code heavily optimized for Apple Silicon, while the Metal backend leverages GPU parallelism for maximum throughput.

## Features

- **Multi-backend compilation**: Generate optimized Metal shaders or high-performance C code with SIMD intrinsics
- **Apple Silicon optimization**: Forward-pass C kernels use SIMD intrinsics optimized for Apple Silicon processors
- **Automatic differentiation**: All operations have backward passes defined for gradient-based learning
- **Intelligent scheduling**: Automatic detection of feedback loops and stateful operations requiring sequential execution
- **High-level DSP primitives**: Gen-inspired operations including biquad filters, delays, compressors, phasors, and more
- **Spectral loss functions**: Train algorithms in the frequency domain using FFT-based loss

## How It Works

### 1. Graph Construction

Build audio processing graphs using a node-based API:

```swift
let g = Graph()
let input = g.n(.constant(1.0))
let freq = g.n(.constant(440.0))
let phase = g.n(.phasor(g.alloc()), freq, g.n(.constant(0.0)))
let output = g.n(.mul, phase, input)
_ = g.n(.output(0), output)
```

### 2. Compilation Pipeline

The compiler transforms the graph through several stages:

1. **Feedback loop detection** - Identifies cycles through history read/write operations
2. **Topological sorting** - Orders operations respecting dependencies and temporal constraints
3. **Block scheduling** - Groups operations into SIMD (parallel) or scalar (sequential) blocks
4. **Code generation** - Emits optimized Metal shaders or C functions

### 3. Feedback Loop Detection

The algorithm (Sources/DGen/Blocks.swift:16) detects operations that depend on their own previous outputs:

1. **Dependency chain analysis**: Traces forward and backward through history read/write pairs that create implicit cycles
2. **Reachability computation**: Finds all nodes participating in feedback paths using graph traversal
3. **Cluster formation**: Groups connected feedback nodes that must execute together
4. **Corridor assignment**: Creates sequential execution zones for operations with temporal dependencies

Key insight: Any operation depending on its own previous output (directly or transitively) must execute sample-by-sample rather than in parallel.

### 4. SIMD vs Scalar Blocks

After dependency analysis, the scheduler (Sources/DGen/Blocks.swift:452) assigns operations to execution blocks:

**SIMD Blocks**
- Stateless operations that can process multiple samples simultaneously
- Use vector instructions for parallel computation
- Metal: Dispatched as parallelized kernels across GPU threadgroups
- C: Compiled to loops with explicit SIMD intrinsics or auto-vectorization

**Scalar Blocks**
- Stateful operations (accumulators, phasors, latches, history cells)
- All nodes participating in feedback loops
- Must execute sequentially, one sample at a time
- Metal: Scalar kernels with single-thread dispatch
- C: Simple scalar for-loops

The scheduler groups consecutive nodes of the same kind to minimize cross-block data transfers (buffer copies in Metal, register spills in C) and maximize fusion opportunities.

## Training with Automatic Differentiation

DGen supports gradient-based learning of DSP parameters using spectral loss functions. Here's an example that learns two frequencies by matching sine waves with amplitude modulation:

```swift
let g = Graph()

// Learnable parameters
let freqParam = Parameter(graph: g, value: 237.0, name: "frequency")
let lfoFreqParam = Parameter(graph: g, value: 8.5, name: "lfo-frequency")

// Target values
let targetFreq = g.n(.constant(300.0))
let targetLFOFreq = g.n(.constant(10.0))
let reset = g.n(.constant(0.0))

// Generate learnable signal: sine wave with LFO modulation
let phase1 = g.n(.phasor(g.alloc()), freqParam.node(), reset)
let lfo1 = g.n(.phasor(g.alloc()), lfoFreqParam.node(), reset)
let twoPi = g.n(.constant(2.0 * Float.pi))
let sine1 = g.n(.sin, g.n(.mul, phase1, twoPi))
let sig1 = g.n(.mul, sine1, lfo1)

// Generate target signal
let phase2 = g.n(.phasor(g.alloc()), targetFreq, reset)
let lfo2 = g.n(.phasor(g.alloc()), targetLFOFreq, reset)
let sine2 = g.n(.sin, g.n(.mul, phase2, twoPi))
let sig2 = g.n(.mul, sine2, lfo2)

// Combined loss: spectral + L2
let spectralLoss = g.spectralLoss(sig1, sig2, windowSize: 64)
let l2Loss = g.n(.mse, sig1, sig2)
let loss = g.n(.add,
    g.n(.mul, g.n(.constant(100.0)), spectralLoss),
    g.n(.mul, g.n(.constant(0.003)), l2Loss))

_ = g.n(.output(0), loss)

// Compile with backward pass
let result = try CompilationPipeline.compile(
    graph: g,
    backend: .metal,
    options: .init(frameCount: 4096, backwards: true)
)

let runtime = try MetalCompiledKernel(
    kernels: result.kernels,
    cellAllocations: result.cellAllocations,
    context: result.context,
    frameCount: 4096
)

// Training context
let ctx = TrainingContext(
    parameters: [freqParam, lfoFreqParam],
    optimizer: SGD(lr: 0.03),
    lossNode: loss
)
ctx.initializeMemory(
    runtime: runtime,
    cellAllocations: result.cellAllocations,
    context: result.context,
    frameCount: 4096
)

// Training loop
for iteration in 0..<120 {
    ctx.zeroGrad()

    // Forward + backward pass
    runtime.runWithMemory(
        outputs: outputPtr,
        inputs: inputPtr,
        memory: ctx.getMemory(),
        frameCount: 4096
    )

    // Update parameters
    ctx.step()

    if iteration % 10 == 0 {
        print("Iteration \(iteration): freq=\(freqParam.value) Hz, " +
              "lfoFreq=\(lfoFreqParam.value) Hz, loss=\(outputBuffer.last!)")
    }
}

// Result: freq converges to ~300 Hz, lfoFreq to ~10 Hz
```

This example demonstrates:
- Multiple learnable parameters optimized simultaneously
- Spectral loss for frequency-domain matching
- Combining multiple loss functions
- Real-time audio DSP with gradient computation

## Available Operations

- **Arithmetic**: add, sub, mul, div, abs, sign, floor, mod, pow
- **Comparisons**: gt, lt, gte, lte, eq
- **Trigonometry**: sin, cos, tan, asin, acos, atan
- **Exponentials**: exp, log, log10, sqrt
- **DSP Primitives**: phasor, accum, latch, delay, biquad, compressor
- **Memory**: historyRead, historyWrite, memoryRead, memoryWrite
- **Control Flow**: gswitch (conditional), selector (multi-way switch), seq (ordering)
- **Loss Functions**: mse, spectralLoss (FFT-based magnitude difference)

## Building

```bash
swift build
swift test
```

## Targets

- **macOS 10.15+** (Metal and C backends)
- **Metal**: GPU-accelerated execution via Metal Performance Shaders
- **C**: CPU execution with SIMD intrinsics

## Project Structure

- `Sources/DGen/DGen.swift` - Graph construction API
- `Sources/DGen/Blocks.swift` - Feedback loop detection and block scheduling
- `Sources/DGen/Operators.swift` - Operation definitions and lowering
- `Sources/DGen/Renderer.swift` - Code generation for Metal/C
- `Sources/DGen/Training.swift` - Gradient descent and parameter optimization
- `Tests/DGenTests/` - Compilation tests and training examples
