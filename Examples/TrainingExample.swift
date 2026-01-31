import Foundation
import DGen

/// Example: Training a biquad filter to match a target audio signal
/// This demonstrates how to use the learnable parameter API

func trainingExample() throws {
    print("🎓 Starting training example...")

    // MARK: - 1. Define Graph with Learnable Parameters

    let g = Graph()

    // Input signal
    let input = g.n(.input(0))

    // Learnable parameters (these are what we'll optimize)
    let learnableCutoff = Parameter(graph: g, value: 1000.0, name: "cutoff")
    let learnableResonance = Parameter(graph: g, value: 0.5, name: "resonance")

    // Fixed parameters
    let gain = g.n(.constant(0.0))
    let mode = g.n(.constant(0.0))  // 0 = lowpass

    // Apply biquad filter with learnable parameters
    let filtered = g.biquad(
        input,
        learnableCutoff.node(),
        learnableResonance.node(),
        gain,
        mode
    )

    // Define loss: we'll try to match a target signal
    let target = g.n(.input(1))  // Target signal on input channel 1
    let loss = g.n(.mse, filtered, target)

    // Output the filtered result
    _ = g.n(.output(0), filtered)

    // MARK: - 2. Compile with Backwards Pass Enabled

    let frameCount = 128
    let result = try CompilationPipeline.compile(
        graph: g,
        backend: .metal,
        options: .init(
            frameCount: frameCount,
            debug: true
        )
    )

    let runtime = try MetalCompiledKernel(
        kernels: result.kernels,
        cellAllocations: result.cellAllocations,
        context: result.context
    )

    print("✅ Compiled graph with \(result.context.gradients.count) gradient nodes")
    print("📊 Parameters:")
    print("   - cutoff: GradID = \(result.context.gradients[learnableCutoff.nodeId] ?? -1)")
    print("   - resonance: GradID = \(result.context.gradients[learnableResonance.nodeId] ?? -1)")

    // MARK: - 3. Create Training Context

    // Pass the loss node so it gets marked as a seed gradient
    let trainingContext = TrainingContext(
        parameters: [learnableCutoff, learnableResonance],
        optimizer: Adam(lr: 0.01, beta1: 0.9, beta2: 0.999),
        lossNode: loss  // 🌱 This marks the loss as the seed for backprop
    )

    trainingContext.initializeMemory(
        runtime: runtime,
        cellAllocations: result.cellAllocations,
        context: result.context,
        frameCount: frameCount
    )

    print("✅ Training context initialized")

    // MARK: - 4. Prepare Training Data

    // Create input signal (white noise)
    var inputBuffer = [Float](repeating: 0.0, count: frameCount)
    for i in 0..<frameCount {
        inputBuffer[i] = Float.random(in: -1...1)
    }

    // Create target signal (the same input, but filtered at 440Hz Q=0.7)
    // In a real scenario, this would be your ground truth
    var targetBuffer = [Float](repeating: 0.0, count: frameCount)
    for i in 0..<frameCount {
        // Simulate filtered target (simplified - in reality use actual DSP)
        targetBuffer[i] = inputBuffer[i] * 0.5  // Placeholder
    }

    var outputBuffer = [Float](repeating: 0.0, count: frameCount)

    // MARK: - 5. Training Loop

    let numEpochs = 100
    var bestLoss: Float = .infinity

    for epoch in 0..<numEpochs {
        // Zero gradients before backward pass
        trainingContext.zeroGrad()

        // Forward pass
        inputBuffer.withUnsafeBufferPointer { inputPtr in
            outputBuffer.withUnsafeMutableBufferPointer { outputPtr in
                runtime.runWithMemory(
                    outputs: outputPtr.baseAddress!,
                    inputs: inputPtr.baseAddress!,
                    memory: trainingContext.getMemory(),
                    frameCount: frameCount
                )
            }
        }

        // Compute loss manually (MSE)
        var loss: Float = 0.0
        for i in 0..<frameCount {
            let diff = outputBuffer[i] - targetBuffer[i]
            loss += diff * diff
        }
        loss /= Float(frameCount)

        // Track best loss
        if loss < bestLoss {
            bestLoss = loss
        }

        // Backward pass (already done automatically by the runtime)
        // Gradients are now in the gradient buffer

        // Update parameters
        trainingContext.step()

        // Print progress
        if epoch % 10 == 0 {
            print(String(format: "Epoch %3d: loss = %.6f | cutoff = %6.1f Hz | Q = %.3f",
                epoch, loss,
                learnableCutoff.value,
                learnableResonance.value))
        }
    }

    // MARK: - 6. Results

    print("\n🎉 Training complete!")
    print(String(format: "Final loss: %.6f", bestLoss))
    print(String(format: "Learned cutoff: %.1f Hz", learnableCutoff.value))
    print(String(format: "Learned resonance: %.3f", learnableResonance.value))
}

/// Simpler example: Learning a scalar parameter
func simpleScalarExample() throws {
    print("\n🎯 Simple scalar learning example...")

    // MARK: - Setup

    let g = Graph()

    // Input signal
    let input = g.n(.input(0))

    // Learnable gain parameter (we'll try to learn gain = 0.5)
    let learnableGain = Parameter(graph: g, value: 0.1, name: "gain")  // Start at wrong value

    // Apply gain
    let scaled = g.n(.mul, input, learnableGain.node())

    // Target: input * 0.5
    let targetGain = g.n(.constant(0.5))
    let target = g.n(.mul, input, targetGain)

    // Loss
    let loss = g.n(.mse, scaled, target)

    // Output
    _ = g.n(.output(0), scaled)

    // MARK: - Compile & Train

    let frameCount = 64
    let result = try CompilationPipeline.compile(
        graph: g,
        backend: .metal,
        options: .init(frameCount: frameCount)
    )

    let runtime = try MetalCompiledKernel(
        kernels: result.kernels,
        cellAllocations: result.cellAllocations,
        context: result.context
    )

    let ctx = TrainingContext(
        parameters: [learnableGain],
        optimizer: SGD(lr: 0.01),
        lossNode: loss
    )
    ctx.initializeMemory(
        runtime: runtime,
        cellAllocations: result.cellAllocations,
        context: result.context,
        frameCount: frameCount
    )

    // Training data
    var inputBuffer = [Float](repeating: 1.0, count: frameCount)
    var outputBuffer = [Float](repeating: 0.0, count: frameCount)

    print("Initial gain: \(learnableGain.value)")
    print("Target gain: 0.5")
    print("\nTraining...")

    for epoch in 0..<50 {
        ctx.zeroGrad()

        inputBuffer.withUnsafeBufferPointer { inPtr in
            outputBuffer.withUnsafeMutableBufferPointer { outPtr in
                runtime.runWithMemory(
                    outputs: outPtr.baseAddress!,
                    inputs: inPtr.baseAddress!,
                    memory: ctx.getMemory(),
                    frameCount: frameCount
                )
            }
        }

        ctx.step()

        if epoch % 10 == 0 {
            print(String(format: "Epoch %2d: gain = %.4f", epoch, learnableGain.value))
        }
    }

    print(String(format: "\n✅ Final gain: %.4f (target: 0.5)", learnableGain.value))
}

// MARK: - Run Examples

// Uncomment to run:
// try trainingExample()
// try simpleScalarExample()
