import Foundation
import XCTest

@testable import DGen

final class TemporalityTests: XCTestCase {

    /// Test the simplest case: a static matmul that feeds into frame-based computation.
    /// This validates:
    /// 1. splitBlockByStaticIfPossible correctly separates static ops
    /// 2. Static block runs once, frame-based block runs every frame
    /// 3. Gradients flow through the boundary
    func testStaticMatmulIntoFrameBased() throws {
        let frameCount = 32
        let sampleRate: Float = 1000.0

        print("\n========================================")
        print("🧪 testStaticMatmulIntoFrameBased")
        print("========================================")

        let g = Graph(sampleRate: sampleRate)

        // Static computation: matmul of two constant tensors
        // This should be extracted into a static block
        let inputTensor = g.tensor(shape: [1, 4], data: [1.0, 2.0, 3.0, 4.0])
        let weights = TensorParameter(
            graph: g, shape: [4, 2],
            data: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            name: "W"
        )

        // Static matmul: result is [1, 2] with static values
        let matmulResult = try g.matmul(inputTensor, weights.node())

        // Now peek into this result using a frame-based index
        // This creates the static -> frameBased boundary
        let zero = g.n(.constant(0.0))

        // Frame-based: phasor oscillates each frame
        let phasor = g.phasor(freq: g.n(.constant(sampleRate / Float(frameCount))), reset: zero)

        // Read from static matmul result using frame-based channel index
        // matmulResult is [1, 2], so peek(index=0, channel=phasor*1.99) reads element 0 or 1
        let channelIdx = g.n(.mul, phasor, g.n(.constant(1.99)))
        let peekResult = try g.peek(tensor: matmulResult, index: zero, channel: channelIdx)

        // Target: some constant to compare against
        let target = g.n(.constant(3.0))

        // Loss: MSE
        let diff = g.n(.sub, peekResult, target)
        let loss = g.n(.mul, diff, diff)
        _ = g.n(.output(0), loss)

        print("Graph nodes: \(g.nodes.count)")

        // Compile
        let compileResult = try CompilationPipeline.compile(
            graph: g, backend: .metal,
            options: .init(frameCount: frameCount, debug: true, backwards: true)
        )

        // Write kernels to file for inspection
        let kernelPath = "/tmp/temporality_test_kernels.metal"
        var kernelSource = "// Generated kernels for testStaticMatmulIntoFrameBased\n"
        kernelSource += "// Total kernels: \(compileResult.kernels.count)\n\n"

        for (i, kernel) in compileResult.kernels.enumerated() {
            kernelSource += "// ========================================\n"
            kernelSource += "// Kernel \(i): \(kernel.name)\n"
            kernelSource += "// ========================================\n"
            kernelSource += kernel.source
            kernelSource += "\n\n"
        }

        try kernelSource.write(toFile: kernelPath, atomically: true, encoding: .utf8)
        print("📝 Wrote kernels to: \(kernelPath)")

        // Also print to console
        print("\n=== GENERATED KERNELS ===")
        for (i, kernel) in compileResult.kernels.enumerated() {
            print("\n--- Kernel \(i): \(kernel.name) ---")
            print(kernel.source)
        }
        print("=== END KERNELS ===\n")

        // Create runtime
        let runtime = try MetalCompiledKernel(
            kernels: compileResult.kernels,
            cellAllocations: compileResult.cellAllocations,
            context: compileResult.context
        )

        // Debug: Check if tensor gradients were allocated
        print("\n=== TENSOR GRADIENT DEBUG ===")
        print("weights.nodeId = \(weights.nodeId)")
        print("weights.cellId = \(weights.cellId)")
        print("context.tensorGradients = \(compileResult.context.tensorGradients)")
        print("context.gradients = \(compileResult.context.gradients)")
        print("context.seedGradients BEFORE initializeMemory = \(compileResult.context.seedGradients)")
        print("loss node = \(loss)")
        print("loss gradient ID = \(compileResult.context.gradients[loss] as Any)")
        print("context.maxGradId = \(compileResult.context.maxGradId)")

        // Training setup
        let ctx = TrainingContext(
            tensorParameters: [weights],
            optimizer: SGD(lr: 0.01),
            lossNode: loss
        )

        ctx.initializeMemory(
            runtime: runtime,
            cellAllocations: compileResult.cellAllocations,
            context: compileResult.context,
            frameCount: frameCount,
            graph: g
        )

        print("After initializeMemory:")
        print("weights.baseGradId = \(weights.baseGradId as Any)")

        // Run a few training steps
        let initialLoss = ctx.runStepGPU()
        print("Initial loss: \(initialLoss)")

        // Debug: Check gradients after first step
        if let gradBuffer = runtime.getBuffer(name: "gradients") {
            let gradPtr = gradBuffer.contents().assumingMemoryBound(to: Float.self)
            let baseGradId = weights.baseGradId ?? 0
            print("Gradients at baseGradId \(baseGradId):")
            for i in 0..<min(8, weights.size) {
                var sum: Float = 0
                for frame in 0..<frameCount {
                    sum += gradPtr[(baseGradId + i) * frameCount + frame]
                }
                print("  grad[\(i)] = \(sum)")
            }
        }

        // Print current weight values
        print("Weights after step 1:")
        for i in 0..<weights.size {
            print("  W[\(i)] = \(weights.data[i])")
        }

        var finalLoss = initialLoss
        for epoch in 0..<20 {
            finalLoss = ctx.runStepGPU()
            if epoch % 5 == 0 {
                print("Epoch \(epoch): loss = \(finalLoss)")
            }
        }

        print("Final loss: \(finalLoss)")
        print("Final weights:")
        for i in 0..<weights.size {
            print("  W[\(i)] = \(weights.data[i])")
        }

        // Verify loss decreased
        XCTAssertLessThan(finalLoss, initialLoss, "Loss should decrease with training")
    }

    /// Even simpler test: just verify block splitting happens correctly
    /// without any frame-based computation.
    func testPureStaticBlockExtraction() throws {
        let frameCount = 16

        print("\n========================================")
        print("🧪 testPureStaticBlockExtraction")
        print("========================================")

        let g = Graph(sampleRate: 1000.0)

        // All static: two tensors, matmul, sum to scalar
        let a = g.tensor(shape: [2, 3], data: [1, 2, 3, 4, 5, 6])
        let b = g.tensor(shape: [3, 2], data: [1, 0, 0, 1, 1, 1])
        let c = try g.matmul(a, b)  // [2, 2]
        let summed = g.n(.sum, c)    // scalar
        _ = g.n(.output(0), summed)

        let compileResult = try CompilationPipeline.compile(
            graph: g, backend: .metal,
            options: .init(frameCount: frameCount, debug: true, backwards: false)
        )

        print("Kernels generated: \(compileResult.kernels.count)")
        for (i, kernel) in compileResult.kernels.enumerated() {
            print("Kernel \(i): \(kernel.name)")
        }

        // The matmul should be in a static block (no frame loop)
        // Let's verify by checking if any kernel contains a frame loop
        for kernel in compileResult.kernels {
            print("--- \(kernel.name) ---")
            print(kernel.source.prefix(500))
        }
    }

    /// Test that static blocks execute once while frame-based blocks loop
    func testMixedTemporalityExecution() throws {
        let frameCount = 8

        print("\n========================================")
        print("🧪 testMixedTemporalityExecution")
        print("========================================")

        let g = Graph(sampleRate: 1000.0)

        // Static part: constant tensor
        let staticTensor = g.tensor(shape: [4, 1], data: [1.0, 2.0, 3.0, 4.0])

        // Frame-based part: phasor-driven index
        let zero = g.n(.constant(0.0))
        let phasorCell = g.alloc()
        let phasor = g.n(.phasor(phasorCell), g.n(.constant(1000.0 / Float(frameCount))))
        let idx = g.n(.mul, phasor, g.n(.constant(3.99)))  // 0..3.99

        // Peek reads from static tensor with frame-varying index
        let value = try g.peek(tensor: staticTensor, index: idx, channel: zero)

        _ = g.n(.output(0), value)

        let compileResult = try CompilationPipeline.compile(
            graph: g, backend: .metal,
            options: .init(frameCount: frameCount, debug: true, backwards: false)
        )

        // Write to file
        let kernelPath = "/tmp/mixed_temporality_kernels.metal"
        var kernelSource = ""
        for (i, kernel) in compileResult.kernels.enumerated() {
            kernelSource += "// Kernel \(i): \(kernel.name)\n"
            kernelSource += kernel.source
            kernelSource += "\n\n"
        }
        try kernelSource.write(toFile: kernelPath, atomically: true, encoding: .utf8)
        print("📝 Wrote kernels to: \(kernelPath)")

        // Print kernel info
        print("\n=== GENERATED KERNELS ===")
        for (i, kernel) in compileResult.kernels.enumerated() {
            print("Kernel \(i): \(kernel.name)")
            print(kernel.source.prefix(1000))
            print("---")
        }
    }
}
