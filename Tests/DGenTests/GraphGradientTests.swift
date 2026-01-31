import XCTest

@testable import DGen
@testable import DGenFrontend

/// Tests for the new graph-based gradient computation (Gradients.swift)
/// This approach builds gradients as LazyOps in the graph, rather than
/// emitting backward IR directly. Forward and backward are unified.
final class GraphGradientTests: XCTestCase {

    /// Simple test: y = x * 2, loss = (y - target)^2
    /// Gradient: d(loss)/dx = 2 * (y - target) * 2 = 4 * (x*2 - target)
    func testSimpleMulGradient() throws {
        let g = Graph()

        // Forward: y = x * 2
        let x = g.n(.constant(3.0))  // x = 3
        let two = g.n(.constant(2.0))
        let y = g.n(.mul, [x, two])  // y = 6

        // Loss: (y - target)^2 where target = 10
        let target = g.n(.constant(10.0))
        let loss = g.n(.mse, [y, target])  // (6 - 10)^2 = 16

        // Compute gradients using the new system
        let grads = g.computeGradients(loss: loss, targets: Set([x]))

        // Verify we got a gradient node for x
        XCTAssertNotNil(grads[x], "Should have gradient for x")

        print("Forward graph nodes: \(g.nodes.count)")
        print("Gradient node for x: \(grads[x]!)")

        // Print the gradient subgraph
        printSubgraph(g, root: grads[x]!, label: "grad_x")
    }

    /// Test: sin(x) gradient
    /// d(sin(x))/dx = cos(x)
    func testSinGradient() throws {
        let g = Graph()

        let x = g.n(.constant(0.5))  // x = 0.5
        let sinX = g.n(.sin, [x])

        // Simple loss: just the sin value itself (gradient seed = 1)
        let grads = g.computeGradients(loss: sinX, targets: Set([x]))

        XCTAssertNotNil(grads[x])
        print("Gradient node for x in sin(x): \(grads[x]!)")
        printSubgraph(g, root: grads[x]!, label: "grad_x")

        // The gradient should be cos(x) * 1 = cos(0.5)
        // Let's verify the structure: should be mul(cos(x), constant(1))
        if let gradNode = g.nodes[grads[x]!] {
            print("Gradient op: \(gradNode.op)")
        }
    }

    /// Test: chain rule with mul and sin
    /// y = sin(x * 2)
    /// dy/dx = cos(x * 2) * 2
    func testChainRuleGradient() throws {
        let g = Graph()

        let x = g.n(.constant(1.0))
        let two = g.n(.constant(2.0))
        let scaled = g.n(.mul, [x, two])  // x * 2
        let y = g.n(.sin, [scaled])  // sin(x * 2)

        let grads = g.computeGradients(loss: y, targets: Set([x]))

        XCTAssertNotNil(grads[x])
        print("\n=== Chain Rule Test ===")
        print("y = sin(x * 2)")
        print("dy/dx = cos(x * 2) * 2")
        printSubgraph(g, root: grads[x]!, label: "grad_x")
    }

    /// Test: multiple paths to same variable
    /// y = x * x (x used twice)
    /// dy/dx = x + x = 2x
    func testMultiplePathsGradient() throws {
        let g = Graph()

        let x = g.n(.constant(3.0))
        let y = g.n(.mul, [x, x])  // x^2

        let grads = g.computeGradients(loss: y, targets: Set([x]))

        XCTAssertNotNil(grads[x])
        print("\n=== Multiple Paths Test ===")
        print("y = x * x")
        print("dy/dx = 2x (two paths combined via add)")
        printSubgraph(g, root: grads[x]!, label: "grad_x")

        // Should have an add node combining both gradient paths
        if let gradNode = g.nodes[grads[x]!] {
            if case .add = gradNode.op {
                print("Correctly combined paths with add")
            }
        }
    }

    /// Simplest end-to-end: y = x * 2, output grad_x only
    func testSimpleEndToEnd() throws {
        let g = Graph()

        // y = x * 2, x = 3 => y = 6
        // dy/dx = 2
        let x = g.n(.constant(3.0))
        let two = g.n(.constant(2.0))
        let y = g.n(.mul, [x, two])

        // Compute gradient
        let grads = g.computeGradients(loss: y, targets: Set([x]))
        guard let gradX = grads[x] else {
            XCTFail("No gradient for x")
            return
        }

        print("\n=== Simple End-to-End ===")
        print("y = x * 2, x = 3")
        print("Expected grad = 2")
        print("Total nodes: \(g.nodes.count)")
        printSubgraph(g, root: gradX, label: "grad_x")

        // Output just the gradient
        _ = g.n(.output(0), [gradX])

        // Compile with Metal
        let frameCount = 64
        let result = try CompilationPipeline.compile(
            graph: g,
            backend: .metal,
            options: .init(frameCount: frameCount, debug: true)
        )

        print("\n--- Generated Metal Code ---")
        for kernel in result.kernels {
            print(kernel.source)
        }

        // Run with Metal
        let runtime = try MetalCompiledKernel(
            kernels: result.kernels,
            cellAllocations: result.cellAllocations,
            context: result.context
        )

        var outputs = [Float](repeating: 0, count: frameCount)
        let inputs = [Float](repeating: 0, count: frameCount)

        outputs.withUnsafeMutableBufferPointer { outPtr in
            inputs.withUnsafeBufferPointer { inPtr in
                runtime.run(
                    outputs: outPtr.baseAddress!,
                    inputs: inPtr.baseAddress!,
                    frameCount: frameCount,
                    volumeScale: 1.0
                )
            }
        }

        print("\nResults:")
        print("  grad[0]: \(outputs[0]) (expected: 2.0)")

        XCTAssertEqual(outputs[0], 2.0, accuracy: 0.001)
    }

    /// Chain rule test: sin(x * 2) end-to-end
    /// d(sin(x*2))/dx = cos(x*2) * 2
    /// At x = 0.5: cos(1.0) * 2 = 0.5403 * 2 = 1.0806
    func testChainRuleEndToEnd() throws {
        let g = Graph()

        let x = g.n(.constant(0.5))
        let two = g.n(.constant(2.0))
        let scaled = g.n(.mul, [x, two])  // 1.0
        let y = g.n(.sin, [scaled])  // sin(1.0) = 0.8414

        let grads = g.computeGradients(loss: y, targets: Set([x]))
        guard let gradX = grads[x] else {
            XCTFail("No gradient for x")
            return
        }

        print("\n=== Chain Rule End-to-End ===")
        print("y = sin(x * 2), x = 0.5")
        print("dy/dx = cos(x*2) * 2 = cos(1.0) * 2 = \(cos(1.0) * 2)")
        printSubgraph(g, root: gradX, label: "grad_x")

        _ = g.n(.output(0), [gradX])

        let frameCount = 64
        let result = try CompilationPipeline.compile(
            graph: g,
            backend: .metal,
            options: .init(frameCount: frameCount, debug: true)
        )

        print("\n--- Generated Metal Code ---")
        for kernel in result.kernels { print(kernel.source) }

        let runtime = try MetalCompiledKernel(
            kernels: result.kernels,
            cellAllocations: result.cellAllocations,
            context: result.context
        )

        var outputs = [Float](repeating: 0, count: frameCount)
        let inputs = [Float](repeating: 0, count: frameCount)

        outputs.withUnsafeMutableBufferPointer { outPtr in
            inputs.withUnsafeBufferPointer { inPtr in
                runtime.run(
                    outputs: outPtr.baseAddress!,
                    inputs: inPtr.baseAddress!,
                    frameCount: frameCount,
                    volumeScale: 1.0
                )
            }
        }

        let expected = Float(cos(1.0) * 2.0)  // ~1.0806
        print("\nResults:")
        print("  grad[0]: \(outputs[0]) (expected: \(expected))")

        XCTAssertEqual(outputs[0], expected, accuracy: 0.001)
    }

    /// MSE gradient test: loss = (x - target)^2
    /// d(loss)/dx = 2 * (x - target)
    /// At x = 3, target = 5: grad = 2 * (3 - 5) = -4
    func testMSEGradient() throws {
        let g = Graph()

        let x = g.n(.constant(3.0))
        let target = g.n(.constant(5.0))
        let loss = g.n(.mse, [x, target])

        let grads = g.computeGradients(loss: loss, targets: Set([x]))
        guard let gradX = grads[x] else {
            XCTFail("No gradient for x")
            return
        }

        print("\n=== MSE Gradient Test ===")
        print("loss = (x - 5)^2, x = 3")
        print("d(loss)/dx = 2*(x-5) = -4")
        printSubgraph(g, root: gradX, label: "grad_x")

        _ = g.n(.output(0), [gradX])

        let frameCount = 64
        let result = try CompilationPipeline.compile(
            graph: g,
            backend: .metal,
            options: .init(frameCount: frameCount, debug: true)
        )

        print("\n--- Generated Metal Code ---")
        for kernel in result.kernels { print(kernel.source) }

        let runtime = try MetalCompiledKernel(
            kernels: result.kernels,
            cellAllocations: result.cellAllocations,
            context: result.context
        )

        var outputs = [Float](repeating: 0, count: frameCount)
        let inputs = [Float](repeating: 0, count: frameCount)

        outputs.withUnsafeMutableBufferPointer { outPtr in
            inputs.withUnsafeBufferPointer { inPtr in
                runtime.run(
                    outputs: outPtr.baseAddress!,
                    inputs: inPtr.baseAddress!,
                    frameCount: frameCount,
                    volumeScale: 1.0
                )
            }
        }

        print("\nResults:")
        print("  grad[0]: \(outputs[0]) (expected: -4.0)")

        XCTAssertEqual(outputs[0], -4.0, accuracy: 0.001)
    }

    /// Multi-parameter gradient test
    /// loss = (a * b - target)^2 where a=2, b=3, target=10
    /// forward: a*b = 6, loss = (6-10)^2 = 16
    /// d(loss)/da = 2*(a*b - target) * b = 2*(6-10)*3 = -24
    /// d(loss)/db = 2*(a*b - target) * a = 2*(6-10)*2 = -16
    func testMultiParameterGradients() throws {
        let g = Graph()

        let a = g.n(.constant(2.0))
        let b = g.n(.constant(3.0))
        let target = g.n(.constant(10.0))

        let product = g.n(.mul, [a, b])  // 6
        let loss = g.n(.mse, [product, target])  // (6-10)^2 = 16

        // Compute gradients for both a and b
        let grads = g.computeGradients(loss: loss, targets: Set([a, b]))

        guard let gradA = grads[a], let gradB = grads[b] else {
            XCTFail("Missing gradients")
            return
        }

        print("\n=== Multi-Parameter Gradients ===")
        print("loss = (a * b - 10)^2, a=2, b=3")
        print("Expected: grad_a = -24, grad_b = -16")
        print("\nGradient graph for a:")
        printSubgraph(g, root: gradA, label: "grad_a")
        print("\nGradient graph for b:")
        printSubgraph(g, root: gradB, label: "grad_b")

        // Store gradients to memory cells so we can read them back
        let gradACellId = g.alloc()
        let gradBCellId = g.alloc()
        let zero = g.n(.constant(0.0))

        // Write gradients to memory (frame 0)
        _ = g.n(.memoryWrite(gradACellId), [zero, gradA])
        _ = g.n(.memoryWrite(gradBCellId), [zero, gradB])

        // We need an output to drive execution
        _ = g.n(.output(0), [loss])

        let frameCount = 1
        let result = try CompilationPipeline.compile(
            graph: g,
            backend: .metal,
            options: .init(frameCount: frameCount, debug: true)
        )

        print("\n--- Generated Metal Code ---")
        for kernel in result.kernels { print(kernel.source) }

        let runtime = try MetalCompiledKernel(
            kernels: result.kernels,
            cellAllocations: result.cellAllocations,
            context: result.context
        )

        var outputs = [Float](repeating: 0, count: frameCount)
        let inputs = [Float](repeating: 0, count: frameCount)

        outputs.withUnsafeMutableBufferPointer { outPtr in
            inputs.withUnsafeBufferPointer { inPtr in
                runtime.run(
                    outputs: outPtr.baseAddress!,
                    inputs: inPtr.baseAddress!,
                    frameCount: frameCount,
                    volumeScale: 1.0
                )
            }
        }

        // Read gradients from memory buffer using remapped cell IDs
        guard let memBuffer = runtime.getBuffer(name: "memory") else {
            XCTFail("No memory buffer")
            return
        }
        let memPtr = memBuffer.contents().assumingMemoryBound(to: Float.self)

        // Get physical (remapped) cell IDs
        let physicalGradA = result.cellAllocations.cellMappings[gradACellId] ?? gradACellId
        let physicalGradB = result.cellAllocations.cellMappings[gradBCellId] ?? gradBCellId

        print("\nDebug:")
        print("  gradACellId: \(gradACellId) -> physical: \(physicalGradA)")
        print("  gradBCellId: \(gradBCellId) -> physical: \(physicalGradB)")

        let computedGradA = memPtr[physicalGradA]
        let computedGradB = memPtr[physicalGradB]

        print("\nResults:")
        print("  loss: \(outputs[0]) (expected: 16.0)")
        print("  grad_a: \(computedGradA) (expected: -24.0)")
        print("  grad_b: \(computedGradB) (expected: -16.0)")

        XCTAssertEqual(outputs[0], 16.0, accuracy: 0.001)
        XCTAssertEqual(computedGradA, -24.0, accuracy: 0.001)
        XCTAssertEqual(computedGradB, -16.0, accuracy: 0.001)
    }

    /// Frame-based test with phasor: sin wave frequency matching
    /// sig1 = sin(2*pi*phasor(freq)), sig2 = sin(2*pi*phasor(targetFreq))
    /// loss = mse(sig1, sig2)
    /// This tests gradient flow through frame-based stateful ops
    func testPhasorFrameBasedGradient() throws {
        let g = Graph()

        // Learnable frequency and target
        let freq = g.n(.constant(450.0))  // start at 450 Hz
        let targetFreq = g.n(.constant(440.0))  // target is 440 Hz
        let reset = g.n(.constant(0.0))
        let twoPi = g.n(.constant(2.0 * Float.pi))

        // Phasors
        let phasorCell1 = g.alloc()
        let phasorCell2 = g.alloc()
        let phase1 = g.n(.phasor(phasorCell1), [freq, reset])
        let phase2 = g.n(.phasor(phasorCell2), [targetFreq, reset])

        // Sine waves
        let angle1 = g.n(.mul, [phase1, twoPi])
        let angle2 = g.n(.mul, [phase2, twoPi])
        let sig1 = g.n(.sin, [angle1])
        let sig2 = g.n(.sin, [angle2])

        // MSE loss (per-frame)
        let loss = g.n(.mse, [sig1, sig2])

        print("\n=== Frame-Based Phasor Gradient Test ===")
        print("sig1 = sin(2*pi*phasor(450Hz))")
        print("sig2 = sin(2*pi*phasor(440Hz))")
        print("loss = mse(sig1, sig2)")

        // Compute gradient for freq
        let grads = g.computeGradients(loss: loss, targets: Set([freq]))
        guard let gradFreq = grads[freq] else {
            XCTFail("No gradient for freq")
            return
        }

        print("\nGradient graph for freq:")
        printSubgraph(g, root: gradFreq, label: "grad_freq", maxDepth: 4)

        // Atomically accumulate per-frame gradients to get total gradient
        let gradCell = g.alloc()
        let zero = g.n(.constant(0.0))
        _ = g.n(.memoryAccumulate(gradCell), [zero, gradFreq])

        // Output loss
        _ = g.n(.output(0), [loss])

        let frameCount = 64
        let result = try CompilationPipeline.compile(
            graph: g,
            backend: .metal,
            options: .init(frameCount: frameCount, debug: true)
        )

        print("\n--- Generated Metal Code ---")
        for kernel in result.kernels {
            print(kernel.source)
        }

        let runtime = try MetalCompiledKernel(
            kernels: result.kernels,
            cellAllocations: result.cellAllocations,
            context: result.context
        )

        var outputs = [Float](repeating: 0, count: frameCount)
        let inputs = [Float](repeating: 0, count: frameCount)

        outputs.withUnsafeMutableBufferPointer { outPtr in
            inputs.withUnsafeBufferPointer { inPtr in
                runtime.run(
                    outputs: outPtr.baseAddress!,
                    inputs: inPtr.baseAddress!,
                    frameCount: frameCount,
                    volumeScale: 1.0
                )
            }
        }

        // Read gradient from memory
        guard let memBuffer = runtime.getBuffer(name: "memory") else {
            XCTFail("No memory buffer")
            return
        }
        let memPtr = memBuffer.contents().assumingMemoryBound(to: Float.self)
        let physicalGradCell = result.cellAllocations.cellMappings[gradCell] ?? gradCell

        print("\nDebug memory:")
        print("  gradCell: \(gradCell) -> physical: \(physicalGradCell)")
        print("  Memory size: \(memBuffer.length / 4) floats")
        print("  First 10 memory values: ", terminator: "")
        for i in 0..<min(10, memBuffer.length / 4) {
            print("\(memPtr[i]) ", terminator: "")
        }
        print()

        let computedGrad = memPtr[physicalGradCell]

        print("\nResults:")
        print("  Loss values (first 5 frames): \(Array(outputs.prefix(5)))")
        print("  Gradient for freq: \(computedGrad)")

        // The gradient should be non-zero since freqs differ
        // Sign should indicate we need to decrease freq (450 -> 440)
        XCTAssertNotEqual(computedGrad, 0.0, "Gradient should be non-zero")
        print(
            "  Gradient sign: \(computedGrad > 0 ? "positive (increase freq)" : "negative (decrease freq)")"
        )
    }

    func testPhasorFrameBasedGradientBiquad() throws {
        let g = Graph()

        // Learnable frequency and target
        let freq = g.n(.constant(450.0))  // start at 450 Hz
        let targetFreq = g.n(.constant(440.0))  // target is 440 Hz
        let reset = g.n(.constant(0.0))
        let twoPi = g.n(.constant(2.0 * Float.pi))

        // Phasors
        let phasorCell1 = g.alloc()
        let phasorCell2 = g.alloc()
        let phase1 = g.n(.phasor(phasorCell1), [freq, reset])
        let phase2 = g.n(.phasor(phasorCell2), [targetFreq, reset])

        // Sine waves
        //let angle1 = g.n(.mul, [phase1, twoPi])
        let angle2 = g.n(.mul, [phase2, twoPi])
        let sig1 = g.biquad(
            phase1, g.n(.constant(1000)), g.n(.constant(9)), g.n(.constant(1)), g.n(.constant(0)))
        let sig2 = g.n(.sin, [angle2])

        // MSE loss (per-frame)
        let loss = g.n(.mse, [sig1, sig2])

        print("\n=== Frame-Based Phasor Gradient Test ===")
        print("sig1 = sin(2*pi*phasor(450Hz))")
        print("sig2 = sin(2*pi*phasor(440Hz))")
        print("loss = mse(sig1, sig2)")

        // Compute gradient for freq
        let grads = g.computeGradients(loss: loss, targets: Set([freq]))
        guard let gradFreq = grads[freq] else {
            XCTFail("No gradient for freq")
            return
        }

        print("\nGradient graph for freq:")
        printSubgraph(g, root: gradFreq, label: "grad_freq", maxDepth: 4)

        // Atomically accumulate per-frame gradients to get total gradient
        let gradCell = g.alloc()
        let zero = g.n(.constant(0.0))
        _ = g.n(.memoryAccumulate(gradCell), [zero, gradFreq])

        // Output loss
        _ = g.n(.output(0), [loss])

        let frameCount = 64
        let result = try CompilationPipeline.compile(
            graph: g,
            backend: .metal,
            options: .init(frameCount: frameCount, debug: true)
        )

        print("\n--- Generated Metal Code ---")
        for kernel in result.kernels {
            print(kernel.source)
        }

        let runtime = try MetalCompiledKernel(
            kernels: result.kernels,
            cellAllocations: result.cellAllocations,
            context: result.context
        )

        var outputs = [Float](repeating: 0, count: frameCount)
        let inputs = [Float](repeating: 0, count: frameCount)

        outputs.withUnsafeMutableBufferPointer { outPtr in
            inputs.withUnsafeBufferPointer { inPtr in
                runtime.run(
                    outputs: outPtr.baseAddress!,
                    inputs: inPtr.baseAddress!,
                    frameCount: frameCount,
                    volumeScale: 1.0
                )
            }
        }

        // Read gradient from memory
        guard let memBuffer = runtime.getBuffer(name: "memory") else {
            XCTFail("No memory buffer")
            return
        }
        let memPtr = memBuffer.contents().assumingMemoryBound(to: Float.self)
        let physicalGradCell = result.cellAllocations.cellMappings[gradCell] ?? gradCell

        print("\nDebug memory:")
        print("  gradCell: \(gradCell) -> physical: \(physicalGradCell)")
        print("  Memory size: \(memBuffer.length / 4) floats")
        print("  First 10 memory values: ", terminator: "")
        for i in 0..<min(10, memBuffer.length / 4) {
            print("\(memPtr[i]) ", terminator: "")
        }
        print()

        let computedGrad = memPtr[physicalGradCell]

        print("\nResults:")
        print("  Loss values (first 5 frames): \(Array(outputs.prefix(5)))")
        print("  Gradient for freq: \(computedGrad)")

        // The gradient should be non-zero since freqs differ
        // Sign should indicate we need to decrease freq (450 -> 440)
        XCTAssertNotEqual(computedGrad, 0.0, "Gradient should be non-zero")
        print(
            "  Gradient sign: \(computedGrad > 0 ? "positive (increase freq)" : "negative (decrease freq)")"
        )
    }

    // MARK: - Helpers

    /// Print a subgraph starting from a root node (with max depth)
    func printSubgraph(_ g: Graph, root: NodeID, label: String, indent: Int = 0, maxDepth: Int = 10)
    {
        if indent > maxDepth {
            let prefix = String(repeating: "  ", count: indent)
            print("\(prefix)...")
            return
        }
        printSubgraphInternal(g, root: root, label: label, indent: indent, maxDepth: maxDepth)
    }

    private func printSubgraphInternal(
        _ g: Graph, root: NodeID, label: String, indent: Int, maxDepth: Int
    ) {
        let prefix = String(repeating: "  ", count: indent)
        guard let node = g.nodes[root] else {
            print("\(prefix)\(label): <missing node \(root)>")
            return
        }

        print("\(prefix)\(label) [id=\(root)]: \(node.op)")
        for (i, input) in node.inputs.enumerated() {
            printSubgraph(
                g, root: input, label: "input\(i)", indent: indent + 1, maxDepth: maxDepth)
        }
    }

    // MARK: - GraphTrainingContext Tests

    /// Test the new GraphTrainingContext with a simple optimization problem
    /// Minimize (x - 3)^2, starting from x = 0
    func testGraphTrainingContextSimple() throws {
        let g = Graph()

        // Create trainable parameter starting at 0
        let x = GraphParameter(graph: g, value: 0.0, name: "x")

        // Target value
        let target = g.n(.constant(3.0))

        // Loss = (x - target)^2
        let loss = g.n(.mse, [x.node(), target])

        // Output loss for inspection
        _ = g.n(.output(0), [loss])

        // Create training context
        let ctx = try GraphTrainingContext(
            graph: g,
            loss: loss,
            parameters: [x],
            optimizer: GraphSGD(),
            learningRate: 0.1,
            frameCount: 1
        )

        print("\n=== GraphTrainingContext Simple Test ===")
        print("Minimizing (x - 3)^2, starting from x = 0")

        // Run training steps
        for step in 0..<20 {
            let lossValue = ctx.trainStep()
            if step % 5 == 0 || step == 19 {
                print("Step \(step): x = \(x.value), loss = \(lossValue), grad = \(x.grad)")
            }
        }

        // x should converge toward 3
        XCTAssertEqual(x.value, 3.0, accuracy: 0.1, "x should converge to 3")
    }

    /// Simpler test: learn mix parameter without history dependency
    /// mix(a, b, t) = a*(1-t) + b*t
    /// Target: t should converge to match target mix ratio
    func testGraphTrainingMixSimple() throws {
        let g = Graph()

        // Learnable mix parameter (target is 0.7)
        let t = GraphParameter(graph: g, value: 0.3, name: "t")
        let targetT = Float(0.7)

        // Two fixed signals
        let a = g.n(.constant(1.0))  // Signal A = 1.0
        let b = g.n(.constant(2.0))  // Signal B = 2.0

        // mix(a, b, t) = a*(1-t) + b*t = 1*(1-t) + 2*t = 1 + t
        // With t=0.3: result = 1.3
        // With t=0.7: result = 1.7 (target)
        let mixResult = g.n(.mix, [a, b, t.node()])

        // Target value: 1.0*(1-0.7) + 2.0*0.7 = 0.3 + 1.4 = 1.7
        let target = g.n(.constant(1.0 * (1 - targetT) + 2.0 * targetT))

        // MSE loss
        let loss = g.n(.mse, [mixResult, target])

        // Output for inspection
        _ = g.n(.output(0), [loss])

        print("\n=== GraphTraining Mix Simple Test ===")
        print("Learning t in mix(1, 2, t) to match target output")
        print("Starting t: 0.3, target t: 0.7")
        print("mix(1, 2, 0.3) = 1.3, mix(1, 2, 0.7) = 1.7")

        // Create training context
        let ctx = try GraphTrainingContext(
            graph: g,
            loss: loss,
            parameters: [t],
            optimizer: GraphSGD(),
            learningRate: 0.1,
            frameCount: 1
        )

        for step in 0..<20 {
            let lossValue = ctx.trainStep()
            if step % 5 == 0 || step == 19 {
                let currentMix = 1.0 * (1 - t.value) + 2.0 * t.value
                print(
                    "Step \(step): t = \(t.value), mix = \(currentMix), loss = \(lossValue), grad = \(t.grad)"
                )
            }
        }

        // t should converge toward 0.7
        print("\nFinal t: \(t.value) (target: 0.7)")
        XCTAssertEqual(t.value, 0.7, accuracy: 0.1, "t should converge to 0.7")
    }

    /// Test onepole gradient against analytical expectation
    /// Simplified: compare ONE filter output against the INPUT (no second filter)
    /// The filter output lags behind input; higher cutoff = more lag
    /// So we want to match input closely -> minimize cutoff
    func testGraphTrainingOnepoleLagMinimize() throws {
        let g = Graph()

        // Learnable cutoff parameter - we want it to converge to 0 (no filtering)
        let cutoff = GraphParameter(graph: g, value: 0.5, name: "cutoff")

        // Use a ramp input
        let freq = g.n(.constant(100.0))
        let reset = g.n(.constant(0.0))
        let phasorCell = g.alloc()
        let phase = g.n(.phasor(phasorCell), [freq, reset])

        // Onepole filter
        let historyCell = g.alloc()
        let history = g.n(.historyRead(historyCell))
        let filtered = g.n(.mix, [phase, history, cutoff.node()])
        _ = g.n(.historyWrite(historyCell), [filtered])

        // Loss: MSE between filtered output and RAW input
        // To minimize this, cutoff should go to 0 (no filtering)
        let loss = g.n(.mse, [filtered, phase])
        _ = g.n(.output(0), [loss])

        print("\n=== Onepole Lag Minimize Test ===")
        print("Goal: match filter output to input -> cutoff should go to 0")
        print("Starting cutoff: 0.5")
        print("With cutoff=0: filtered = input (no lag)")
        print("d(loss)/d(cutoff) should be POSITIVE to decrease cutoff")

        let ctx = try GraphTrainingContext(
            graph: g,
            loss: loss,
            parameters: [cutoff],
            optimizer: GraphSGD(),
            learningRate: 0.1,
            frameCount: 64
        )

        for step in 0..<30 {
            let lossValue = ctx.trainStep()
            if step % 5 == 0 || step == 29 {
                print(
                    "Step \(step): cutoff = \(cutoff.value), loss = \(lossValue), grad = \(cutoff.grad)"
                )
            }
        }

        print("\nFinal cutoff: \(cutoff.value) (target: 0)")
        // Cutoff should decrease toward 0
    }

    /// Debug test: trace gradient values with shared vs separate phasor
    func testDebugSharedPhasorGradient() throws {
        // Test with SHARED phasor - outputs gradient nodes
        let g = Graph()

        let cutoff = GraphParameter(graph: g, value: 0.5, name: "cutoff")
        let targetCutoff = g.n(.constant(0.2))

        // SHARED phasor
        let freq = g.n(.constant(100.0))
        let reset = g.n(.constant(0.0))
        let phasorCell = g.alloc()
        let phase = g.n(.phasor(phasorCell), [freq, reset])

        // Two onepole filters sharing the same phase input
        let historyCell1 = g.alloc()
        let history1 = g.n(.historyRead(historyCell1))
        let mix1 = g.n(.mix, [phase, history1, cutoff.node()])
        _ = g.n(.historyWrite(historyCell1), [mix1])

        let historyCell2 = g.alloc()
        let history2 = g.n(.historyRead(historyCell2))
        let mix2 = g.n(.mix, [phase, history2, targetCutoff])
        _ = g.n(.historyWrite(historyCell2), [mix2])

        let loss = g.n(.mse, [mix1, mix2])

        print("\n=== Debug: Shared Phasor Gradient ===")
        print("Graph nodes before gradient computation:")
        for (id, node) in g.nodes.sorted(by: { $0.key < $1.key }) {
            print("  \(id): \(node.op) inputs=\(node.inputs)")
        }

        // Compute gradients
        let grads = g.computeGradients(loss: loss, targets: Set([cutoff.nodeId]))

        print("\nGradient nodes created:")
        for (nodeId, gradNodeId) in grads.sorted(by: { $0.key < $1.key }) {
            if let gradNode = g.nodes[gradNodeId] {
                print("  grad[\(nodeId)] = node \(gradNodeId): \(gradNode.op)")
            }
        }

        print("\nGradient side effects: \(g.gradientSideEffects)")

        // Check which nodes are on target path
        print("\nTarget path analysis:")
        print("  cutoff.nodeId = \(cutoff.nodeId)")
        print("  loss nodeId = \(loss)")
        print("  mix1 nodeId = \(mix1)")
        print("  mix2 nodeId = \(mix2)")
        print("  phase nodeId = \(phase)")

        // Trace the gradient expression for cutoff
        print("\nGradient expression for cutoff (traced):")
        func traceNode(_ id: NodeID, indent: Int = 0) {
            let prefix = String(repeating: "  ", count: indent)
            guard let node = g.nodes[id] else {
                print("\(prefix)\(id): <missing>")
                return
            }
            print("\(prefix)\(id): \(node.op) inputs=\(node.inputs)")
            if indent < 4 {
                for input in node.inputs {
                    traceNode(input, indent: indent + 1)
                }
            }
        }
        if let gradCutoff = grads[cutoff.nodeId] {
            traceNode(gradCutoff)
        }

        // Check which nodes got gradients assigned
        print("\nNodes with gradients assigned:")
        for (nodeId, gradId) in grads.sorted(by: { $0.key < $1.key }) {
            if let node = g.nodes[nodeId] {
                print("  \(nodeId) (\(node.op)): grad = \(gradId)")
            }
        }

        // Check if mix2's backward was processed (it shouldn't be)
        print("\nmix1 nodeId = \(mix1), mix2 nodeId = \(mix2)")
        print(
            "mix1 inputs gradient assigned: \(grads[phase] != nil), \(grads[history1] != nil), \(grads[cutoff.nodeId] != nil)"
        )
        print(
            "mix2 inputs gradient assigned: history2=\(grads[history2] != nil), targetCutoff=\(grads[targetCutoff] != nil)"
        )
        // Note: phase is shared, so if mix2's backward ran, it would have added to grads[phase]

        // Now compile and run to see actual values
        _ = g.n(.output(0), [loss])

        // Store the gradient to memory so we can read it
        let gradCell = g.alloc()
        let zero = g.n(.constant(0.0))
        if let gradCutoff = grads[cutoff.nodeId] {
            _ = g.n(.memoryAccumulate(gradCell), [zero, gradCutoff])
        }

        let result = try CompilationPipeline.compile(
            graph: g, backend: .metal, options: .init(frameCount: 16, debug: true))

        // Print first kernel source to see gradient computation
        print("\n=== Generated Metal (first SIMD kernel) ===")
        if let simdKernel = result.kernels.first(where: { $0.source.contains("simd") }) {
            // Print just the relevant part
            let lines = simdKernel.source.split(separator: "\n")
            for line in lines.prefix(60) {
                print(line)
            }
        }

        let runtime = try MetalCompiledKernel(
            kernels: result.kernels,
            cellAllocations: result.cellAllocations,
            context: result.context
        )

        // Initialize cutoff value
        if let memBuffer = runtime.getBuffer(name: "memory") {
            let memPtr = memBuffer.contents().assumingMemoryBound(to: Float.self)
            let physicalCutoff = result.cellAllocations.cellMappings[cutoff.cellId] ?? cutoff.cellId
            memPtr[physicalCutoff] = 0.5
        }

        // Run
        runtime.runNoCopy(frameCount: 16)

        // Read gradient
        if let memBuffer = runtime.getBuffer(name: "memory") {
            let memPtr = memBuffer.contents().assumingMemoryBound(to: Float.self)
            let physicalGrad = result.cellAllocations.cellMappings[gradCell] ?? gradCell
            print("\nActual accumulated gradient: \(memPtr[physicalGrad])")
            print("Expected: POSITIVE (to decrease cutoff from 0.5 toward 0.2)")
        }

        // Now run the same thing through GraphTrainingContext to compare
        print("\n=== Testing via GraphTrainingContext ===")
        let g2 = Graph()
        let cutoff2 = GraphParameter(graph: g2, value: 0.5, name: "cutoff")
        let targetCutoff2 = g2.n(.constant(0.2))
        let freq2 = g2.n(.constant(100.0))
        let reset2 = g2.n(.constant(0.0))
        let phasorCell2 = g2.alloc()
        let phase2 = g2.n(.phasor(phasorCell2), [freq2, reset2])

        let historyCell1_2 = g2.alloc()
        let history1_2 = g2.n(.historyRead(historyCell1_2))
        let mix1_2 = g2.n(.mix, [phase2, history1_2, cutoff2.node()])
        _ = g2.n(.historyWrite(historyCell1_2), [mix1_2])

        let historyCell2_2 = g2.alloc()
        let history2_2 = g2.n(.historyRead(historyCell2_2))
        let mix2_2 = g2.n(.mix, [phase2, history2_2, targetCutoff2])
        _ = g2.n(.historyWrite(historyCell2_2), [mix2_2])

        let loss2 = g2.n(.mse, [mix1_2, mix2_2])
        _ = g2.n(.output(0), [loss2])

        let ctx = try GraphTrainingContext(
            graph: g2,
            loss: loss2,
            parameters: [cutoff2],
            optimizer: GraphSGD(),
            learningRate: 0.1,
            frameCount: 16
        )

        // Debug: print cell mappings before running
        print("cutoff2.cellId = \(cutoff2.cellId)")
        print("cutoff2.gradientCell = \(cutoff2.gradientCell ?? -1)")

        // Read memory BEFORE training to see initial state
        ctx.configureMemory { memPtr in
            print("Memory BEFORE trainStep:")
            for i in 0..<10 {
                print("  memory[\(i)] = \(memPtr[i])")
            }
        }

        // Run one training step
        let lossVal = ctx.trainStep()
        print("After 1 step: cutoff = \(cutoff2.value), loss = \(lossVal), grad = \(cutoff2.grad)")
        print("Gradient sign: \(cutoff2.grad > 0 ? "POSITIVE (correct)" : "NEGATIVE (wrong)")")

        // Check all memory values to find where the gradient went
        ctx.configureMemory { memPtr in
            print("Memory dump after trainStep:")
            for i in 0..<30 {
                if memPtr[i] != 0 {
                    print("  memory[\(i)] = \(memPtr[i])")
                }
            }
        }

        // Debug: check outputs
        let outputs = ctx.getOutputs()
        print("Output buffer (loss per frame): \(outputs.prefix(5))...")

        // Print gradient info
        print("cutoff2.gradientCell = \(cutoff2.gradientCell ?? -1)")
        print("cutoff2.cellId = \(cutoff2.cellId)")

        // Run another step WITHOUT zeroing grad to see accumulation
        ctx.zeroGrad()
        _ = ctx.forward()
        print("After forward (no step): grad = \(cutoff2.grad)")

        // Compare with separate phasors
        print("\n=== Same test with SEPARATE phasors ===")
        let g3 = Graph()
        let cutoff3 = GraphParameter(graph: g3, value: 0.5, name: "cutoff")
        let targetCutoff3 = g3.n(.constant(0.2))
        let freq3 = g3.n(.constant(100.0))
        let reset3 = g3.n(.constant(0.0))

        // SEPARATE phasors
        let phasorCell3a = g3.alloc()
        let phase3a = g3.n(.phasor(phasorCell3a), [freq3, reset3])
        let phasorCell3b = g3.alloc()
        let phase3b = g3.n(.phasor(phasorCell3b), [freq3, reset3])

        let historyCell3a = g3.alloc()
        let history3a = g3.n(.historyRead(historyCell3a))
        let mix3a = g3.n(.mix, [phase3a, history3a, cutoff3.node()])
        _ = g3.n(.historyWrite(historyCell3a), [mix3a])

        let historyCell3b = g3.alloc()
        let history3b = g3.n(.historyRead(historyCell3b))
        let mix3b = g3.n(.mix, [phase3b, history3b, targetCutoff3])
        _ = g3.n(.historyWrite(historyCell3b), [mix3b])

        let loss3 = g3.n(.mse, [mix3a, mix3b])
        _ = g3.n(.output(0), [loss3])

        let ctx3 = try GraphTrainingContext(
            graph: g3,
            loss: loss3,
            parameters: [cutoff3],
            optimizer: GraphSGD(),
            learningRate: 0.1,
            frameCount: 16
        )

        _ = ctx3.trainStep()
        print("Separate phasors - grad = \(cutoff3.grad)")
        print("Gradient sign: \(cutoff3.grad > 0 ? "POSITIVE (correct)" : "NEGATIVE (wrong)")")
    }

    /// Test with two SEPARATE phasors to avoid shared input gradient issues
    func testGraphTrainingOnepoleSeparate() throws {
        let g = Graph()

        // Learnable cutoff parameter (target is 0.2)
        let cutoff = GraphParameter(graph: g, value: 0.5, name: "cutoff")
        let targetCutoff = g.n(.constant(0.2))

        // SEPARATE phasors for each filter (same frequency, separate cells)
        let freq = g.n(.constant(440.0))
        let reset = g.n(.constant(0.0))

        let phasorCell1 = g.alloc()
        let phase1 = g.n(.phasor(phasorCell1), [freq, reset])

        let phasorCell2 = g.alloc()
        let phase2 = g.n(.phasor(phasorCell2), [freq, reset])

        // Onepole filter: mix(input, history, cutoff)
        func onepole(_ inputNode: NodeID, _ cutoffNode: NodeID) -> NodeID {
            let cellId = g.alloc()
            let history = g.n(.historyRead(cellId))
            let mix = g.n(.mix, [inputNode, history, cutoffNode])
            _ = g.n(.historyWrite(cellId), [mix])
            return mix
        }

        // sig1: filtered with learnable cutoff (using phase1)
        let sig1 = onepole(phase1, cutoff.node())

        // sig2: filtered with target cutoff (using phase2)
        let sig2 = onepole(phase2, targetCutoff)

        // MSE loss between the two filtered signals
        let loss = g.n(.mse, [sig1, sig2])
        _ = g.n(.output(0), [loss])

        print("\n=== GraphTraining Onepole (Separate Phasors) Test ===")
        print("Starting cutoff: 0.5, target: 0.2")

        let ctx = try GraphTrainingContext(
            graph: g,
            loss: loss,
            parameters: [cutoff],
            optimizer: GraphSGD(),
            learningRate: 0.2,  // Lower LR to avoid overshoot
            frameCount: 256
        )

        for step in 0..<1000 {
            let lossValue = ctx.trainStep()
            if step % 20 == 0 || step == 99 {
                print(
                    "Step \(step): cutoff = \(cutoff.value), loss = \(lossValue), grad = \(cutoff.grad)"
                )
            }
        }

        print("\nFinal cutoff: \(cutoff.value) (target: 0.2)")
        XCTAssertEqual(cutoff.value, 0.2, accuracy: 0.15, "Cutoff should converge toward 0.2")
    }

    /// Test with onepole filter (history read/write) - similar to testHistoryBackward
    /// but using MSE loss and the new graph-based training
    /// NOTE: Uses separate phasors due to gradient accumulation issue with shared inputs
    func testGraphTrainingOnepole() throws {
        let g = Graph()

        // Learnable cutoff parameter (target is 0.2)
        let cutoff = GraphParameter(graph: g, value: 0.5, name: "cutoff")
        let targetCutoff = g.n(.constant(0.2))

        // SEPARATE phasors for each filter (same frequency, separate cells)
        // This avoids gradient accumulation issues with shared input nodes
        let freq = g.n(.constant(440.0))
        let reset = g.n(.constant(0.0))

        let phasorCell1 = g.alloc()
        let phase1 = g.n(.phasor(phasorCell1), [freq, reset])

        let phasorCell2 = g.alloc()
        let phase2 = g.n(.phasor(phasorCell2), [freq, reset])

        // Onepole filter: mix(input, history, cutoff)
        // output = input * (1 - cutoff) + history * cutoff
        func onepole(_ inputNode: NodeID, _ cutoffNode: NodeID) -> NodeID {
            let cellId = g.alloc()
            let history = g.n(.historyRead(cellId))
            let mix = g.n(.mix, [inputNode, history, cutoffNode])
            _ = g.n(.historyWrite(cellId), [mix])
            return mix
        }

        // sig1: filtered with learnable cutoff (using phase1)
        let sig1 = onepole(phase1, cutoff.node())

        // sig2: filtered with target cutoff (using phase2)
        let sig2 = onepole(phase2, targetCutoff)

        // MSE loss between the two filtered signals
        let loss = g.n(.mse, [sig1, sig2])

        // Output for inspection
        _ = g.n(.output(0), [loss])

        print("\n=== GraphTraining Onepole Test ===")
        print("Learning cutoff parameter to match target filter")
        print("Starting cutoff: 0.5, target: 0.2")

        // Create training context
        let ctx = try GraphTrainingContext(
            graph: g,
            loss: loss,
            parameters: [cutoff],
            optimizer: GraphSGD(),
            learningRate: 0.05,  // Lower LR for stability
            frameCount: 256
        )

        var lastLoss: Float = 0
        for step in 0..<4000 {
            lastLoss = ctx.trainStep()
            if step % 20 == 0 || step == 99 {
                print(
                    "Step \(step): cutoff = \(cutoff.value), loss = \(lastLoss), grad = \(cutoff.grad)"
                )
            }
        }

        // Cutoff should converge toward 0.2
        print("\nFinal cutoff: \(cutoff.value) (target: 0.2)")
        XCTAssertEqual(cutoff.value, 0.2, accuracy: 0.05, "Cutoff should converge toward 0.2")
    }

    /// Simple test: tensor -> peekRow -> sum -> loss
    func testTensorPeekRowSumLoss() throws {
        let frameCount = 64
        let numRows = 8
        let numCols = 4

        let g = Graph(sampleRate: 1000.0)

        // Learnable 2D tensor [numRows, numCols]
        let tensorData = (0..<(numRows * numCols)).map { Float($0) * 0.1 }
        let learnableTensor = TensorParameter(
            graph: g, shape: [numRows, numCols], data: tensorData, name: "tensor")

        // Target tensor (what we want to converge to)
        let targetData = (0..<numCols).map { Float($0) + 1.0 }  // [1, 2, 3, 4]
        let targetTensor = g.tensor(shape: [numCols], data: targetData)

        let zero = g.n(.constant(0.0))

        // Phasor to drive row index (cycles through rows)
        let playheadFreq = g.n(.constant(1000.0 / Float(frameCount)))
        let playheadPhasor = g.phasor(freq: playheadFreq, reset: zero)
        let playhead = g.n(.mul, [playheadPhasor, g.n(.constant(Float(numRows - 1)))])

        // peekRow: get row at playhead position -> [numCols]
        let rowAtTime = try g.peekRow(tensor: learnableTensor.node(), rowIndex: playhead)

        // Sum each to scalar
        let sumLearnable = g.n(.sum, [rowAtTime])
        let sumTarget = g.n(.sum, [targetTensor])

        // MSE loss
        let diff = g.n(.sub, [sumLearnable, sumTarget])
        let loss = g.n(.mul, [diff, diff])
        _ = g.n(.output(0), [loss])

        print("\n=== Tensor -> peekRow -> sum -> loss ===")

        let ctx = try GraphTrainingContext(
            graph: g,
            loss: loss,
            tensorParameters: [learnableTensor],
            optimizer: GraphAdam(),
            learningRate: 0.1,
            frameCount: frameCount,
            kernelDebugOutput: "/tmp/tensor_peekrow_sum.metal"
        )

        let initialLoss = ctx.trainStep()
        print("Initial loss: \(initialLoss)")

        var finalLoss = initialLoss
        for i in 0..<20 {
            finalLoss = ctx.trainStep()
            if i % 5 == 0 {
                print("Epoch \(i): loss = \(finalLoss)")
            }
        }

        print("Final loss: \(finalLoss)")
        XCTAssertLessThan(finalLoss, initialLoss, "Loss should decrease")
    }

    /// Simple test: matmul -> sum -> loss (no peekRow, static tensors)
    /// Used to debug kernel scheduling for matmul backward
    func testMatmulSumLoss_Debug() throws {
        let frameCount = 1  // Static - no frame variation
        let m = 4  // rows of A
        let k = 3  // cols of A, rows of B
        let n = 2  // cols of B

        let g = Graph(sampleRate: 1000.0)

        // Input tensor A [m, k] - fixed
        let aData = (0..<(m * k)).map { Float($0) * 0.1 }
        let aTensor = g.tensor(shape: [m, k], data: aData)

        // Learnable weight tensor B [k, n]
        let bData = (0..<(k * n)).map { Float($0 + 1) * 0.1 }
        let bTensor = TensorParameter(
            graph: g, shape: [k, n], data: bData, name: "B")

        // matmul: [m, k] x [k, n] -> [m, n]
        let matmulOut = try g.matmul(aTensor, bTensor.node())

        // Sum all elements to scalar
        let sumOut = g.n(.sum, [matmulOut])

        // Simple loss: (sum - target)^2
        let target = g.n(.constant(10.0))
        let diff = g.n(.sub, [sumOut, target])
        let loss = g.n(.mul, [diff, diff])
        _ = g.n(.output(0), [loss])

        print("\n=== Simple Matmul -> Sum -> Loss Debug ===")
        print("A: [\(m), \(k)], B: [\(k), \(n)], Output: [\(m), \(n)]")

        // Compute gradients manually to see the nodes
        let gradients = g.computeGradients(loss: loss, targets: Set([bTensor.nodeId]))
        print("Gradient nodes created: \(gradients.count)")

        // Check scalar nodes
        let feedbackClusters = findFeedbackLoops(g)
        let scalarSet = scalarNodes(g, feedbackClusters: feedbackClusters)

        print("\nNode analysis:")
        for (nodeId, node) in g.nodes.sorted(by: { $0.key < $1.key }) {
            let isScalar = scalarSet.contains(nodeId)
            let hasGrad = gradients[nodeId] != nil
            let marker = isScalar ? "SCALAR" : "SIMD"
            print("  \(nodeId): \(node.op) [\(marker)]\(hasGrad ? " (has grad)" : "")")
        }

        // Now compile and check blocks
        let result = try CompilationPipeline.compile(
            graph: g,
            backend: .metal,
            options: .init(frameCount: frameCount, debug: false)
        )

        print("\nKernels generated: \(result.kernels.count)")
        for (i, kernel) in result.kernels.enumerated() {
            print(
                "  Kernel \(i): kind=\(kernel.kind) threadGroupSize=\(String(describing: kernel.threadGroupSize)) threadCount=\(String(describing: kernel.threadCount))"
            )
        }

        // Write kernels for inspection
        let allKernels = result.kernels.enumerated().map {
            "// KERNEL \($0.offset)\n// Kind: \($0.element.kind)\n// ThreadGroupSize: \($0.element.threadGroupSize ?? -1)\n\n\($0.element.source)"
        }.joined(separator: "\n\n// ========================================\n\n")
        try? allKernels.write(
            toFile: "/tmp/matmul_sum_debug.metal", atomically: true, encoding: .utf8)
        print("\nWrote kernels to /tmp/matmul_sum_debug.metal")
    }

    /// Test: matmul -> peekRow -> sum -> loss
    func testMatmulPeekRowSumLoss() throws {
        let frameCount = 64
        let numRows = 8
        let numCols = 4
        let inputSize = 2

        let g = Graph(sampleRate: 1000.0)

        // Input tensor [numRows, inputSize] - fixed
        let inputData = (0..<(numRows * inputSize)).map { Float($0) * 0.1 }
        let inputTensor = g.tensor(shape: [numRows, inputSize], data: inputData)

        // Learnable weight tensor [inputSize, numCols]
        let weightData = (0..<(inputSize * numCols)).map { _ in Float.random(in: -0.5...0.5) }
        let weightTensor = TensorParameter(
            graph: g, shape: [inputSize, numCols], data: weightData, name: "weights")

        // matmul: [numRows, inputSize] x [inputSize, numCols] -> [numRows, numCols]
        let matmulOut = try g.matmul(inputTensor, weightTensor.node())

        // Target for sum
        let targetSum = g.n(.constant(5.0))

        let zero = g.n(.constant(0.0))

        // Phasor to drive row index
        let playheadFreq = g.n(.constant(1000.0 / Float(frameCount)))
        let playheadPhasor = g.phasor(freq: playheadFreq, reset: zero)
        let playhead = g.n(.mul, [playheadPhasor, g.n(.constant(Float(numRows - 1)))])

        // peekRow from matmul output
        let rowAtTime = try g.peekRow(tensor: matmulOut, rowIndex: playhead)

        // Sum to scalar
        let sumOut = g.n(.sum, [rowAtTime])

        // MSE loss
        let diff = g.n(.sub, [sumOut, targetSum])
        let loss = g.n(.mul, [diff, diff])
        _ = g.n(.output(0), [loss])

        print("\n=== Matmul -> peekRow -> sum -> loss ===")

        let ctx = try GraphTrainingContext(
            graph: g,
            loss: loss,
            tensorParameters: [weightTensor],
            optimizer: GraphAdam(),
            learningRate: 0.1,
            frameCount: frameCount,
            kernelDebugOutput: "/tmp/matmul_peekrow_sum.metal"
        )

        let initialLoss = ctx.trainStep()
        print("Initial loss: \(initialLoss)")

        var finalLoss = initialLoss
        for i in 0..<20 {
            finalLoss = ctx.trainStep()
            if i % 5 == 0 {
                print("Epoch \(i): loss = \(finalLoss)")
            }
        }

        print("Final loss: \(finalLoss)")

        let hasNonZeroGrads = weightTensor.grads.contains { $0 != 0.0 }
        XCTAssertTrue(
            hasNonZeroGrads, "Gradients should flow through matmul -> peekRow -> sum chain")
        XCTAssertLessThan(finalLoss, initialLoss, "Loss should decrease when gradients flow")
    }

    /// MLP -> peekRow -> Harmonic Synth teacher-student test
    /// Tests efficiency of the full pipeline with graph-based gradients
    func testMLPPeekRowHarmonicSynth_TeacherStudent() throws {
        let frameCount = 64
        let controlFrames = 16
        let sampleRate: Float = 2000.0
        let f0: Float = 100.0
        let numHarmonics = 6
        let hiddenSize = 8

        let g = Graph(sampleRate: sampleRate)

        // Control-rate time tensor [controlFrames, 1], normalized 0..1
        let timeData = (0..<controlFrames).map { Float($0) / Float(controlFrames - 1) }
        let timeTensor = g.tensor(shape: [controlFrames, 1], data: timeData)

        func makeArray(_ count: Int, scale: Float, freq: Float, phase: Float, offset: Float = 0.0)
            -> [Float]
        {
            (0..<count).map { i in
                offset + scale * sin(Float(i) * freq + phase)
            }
        }

        // Teacher (fixed) weights
        let teacherW1Data = (0..<hiddenSize).map { i in
            let x = Float(i) / Float(max(1, hiddenSize - 1))
            return 1.1 * sin(x * 3.1 * Float.pi) + 0.5 * cos(x * 2.3 * Float.pi)
        }
        let teacherB1Data = (0..<hiddenSize).map { i in
            let x = Float(i) / Float(max(1, hiddenSize - 1))
            return 0.4 * (x - 0.5) + 0.2 * sin(x * 5.0)
        }
        let teacherW2Data = (0..<(hiddenSize * numHarmonics)).map { i in
            let row = i / numHarmonics
            let col = i % numHarmonics
            let base = Float(row) * 0.6 + Float(col) * 0.25
            let sign: Float = (col % 2 == 0) ? 1.0 : -1.0
            return sign * (0.7 * sin(base) + 0.5 * cos(base * 1.3))
        }
        let teacherB2Data = (0..<numHarmonics).map { i in
            let inv = 0.8 / Float(i + 1)
            let wiggle = 0.1 * sin(Float(i) * 1.7)
            return inv + wiggle
        }

        let teacherW1 = g.tensor(shape: [1, hiddenSize], data: teacherW1Data)
        let teacherB1 = g.tensor(shape: [1, hiddenSize], data: teacherB1Data)
        let teacherW2 = g.tensor(shape: [hiddenSize, numHarmonics], data: teacherW2Data)
        let teacherB2 = g.tensor(shape: [1, numHarmonics], data: teacherB2Data)

        // Student (learnable) weights
        let studentW1 = TensorParameter(
            graph: g, shape: [1, hiddenSize],
            data: makeArray(hiddenSize, scale: 0.12, freq: 0.9, phase: 1.1), name: "W1")
        let studentB1 = TensorParameter(
            graph: g, shape: [1, hiddenSize],
            data: makeArray(hiddenSize, scale: 0.05, freq: 0.6, phase: 0.9), name: "b1")
        let studentW2 = TensorParameter(
            graph: g, shape: [hiddenSize, numHarmonics],
            data: makeArray(hiddenSize * numHarmonics, scale: 0.08, freq: 0.21, phase: 0.7),
            name: "W2")
        let studentB2 = TensorParameter(
            graph: g, shape: [1, numHarmonics],
            data: (0..<numHarmonics).map { _ in 0.1 }, name: "b2")

        // Static MLP: timeTensor -> amplitude tensor [controlFrames, numHarmonics]
        func mlpAmplitudes(time: NodeID, W1: NodeID, b1: NodeID, W2: NodeID, b2: NodeID)
            throws -> NodeID
        {
            let one = g.n(.constant(1.0))
            let h1 = try g.matmul(time, W1)
            let h1b = g.n(.add, [h1, b1])
            let h1a = g.n(.tanh, [h1b])
            let h2 = try g.matmul(h1a, W2)
            let h2b = g.n(.add, [h2, b2])
            let neg = g.n(.mul, [h2b, g.n(.constant(-1.0))])
            let expNeg = g.n(.exp, [neg])
            return g.n(.div, [one, g.n(.add, [one, expNeg])])
        }

        let ampsStudent = try mlpAmplitudes(
            time: timeTensor, W1: studentW1.node(), b1: studentB1.node(),
            W2: studentW2.node(), b2: studentB2.node())
        let ampsTeacher = try mlpAmplitudes(
            time: timeTensor, W1: teacherW1, b1: teacherB1, W2: teacherW2, b2: teacherB2)

        // Reshape to [numHarmonics, controlFrames] then transpose for peekRow
        let ampsStudentView = try g.reshape(ampsStudent, to: [numHarmonics, controlFrames])
        let ampsTeacherView = try g.reshape(ampsTeacher, to: [numHarmonics, controlFrames])
        let ampsStudentT = try g.transpose(ampsStudentView, axes: [1, 0])  // [controlFrames, numHarmonics]
        let ampsTeacherT = try g.transpose(ampsTeacherView, axes: [1, 0])

        let zero = g.n(.constant(0.0))
        let twoPi = g.n(.constant(Float.pi * 2.0))

        // Audio-rate playhead through control frames
        let frameIdx = g.phasor(
            freq: g.n(.constant(sampleRate / Float(frameCount))), reset: zero)
        let playhead = g.n(.mul, [frameIdx, g.n(.constant(Float(controlFrames - 1)))])

        // Vectorized harmonic synthesis
        let freqData = (1...numHarmonics).map { f0 * Float($0) }
        let freqTensor = g.tensor(shape: [numHarmonics], data: freqData)
        let phasesTensor = g.n(.deterministicPhasor, [freqTensor])
        let sinesTensor = g.n(.sin, [g.n(.mul, [twoPi, phasesTensor])])

        let ampsStudentAtTime = try g.peekRow(tensor: ampsStudentT, rowIndex: playhead)
        let ampsTeacherAtTime = try g.peekRow(tensor: ampsTeacherT, rowIndex: playhead)

        let synthStudent = g.n(.sum, [g.n(.mul, [sinesTensor, ampsStudentAtTime])])
        let synthTeacher = g.n(.sum, [g.n(.mul, [sinesTensor, ampsTeacherAtTime])])

        let norm = g.n(.constant(1.0 / Float(numHarmonics)))
        let studentOut = g.n(.mul, [synthStudent, norm])
        let teacherOut = g.n(.mul, [synthTeacher, norm])

        let diff = g.n(.sub, [studentOut, teacherOut])
        let loss = g.n(
            .add, g.spectralLossFFT(studentOut, teacherOut, windowSize: 64),
            g.n(.mul, g.n(.constant(0.001)), g.n(.mul, [diff, diff])))
        _ = g.n(.output(0), [loss])

        print("\n=== MLP -> peekRow -> Harmonic Synth (Teacher-Student) ===")
        print("frameCount: \(frameCount), controlFrames: \(controlFrames)")
        print("numHarmonics: \(numHarmonics), hiddenSize: \(hiddenSize)")
        print(
            "Total learnable params: \(hiddenSize + hiddenSize + hiddenSize*numHarmonics + numHarmonics)"
        )

        // Use GraphTrainingContext with graph-based gradients (no legacy backward IR)
        let compileStart = CFAbsoluteTimeGetCurrent()
        let ctx = try GraphTrainingContext(
            graph: g,
            loss: loss,
            tensorParameters: [studentW1, studentB1, studentW2, studentB2],
            optimizer: GraphSGD(),
            learningRate: 0.000001,
            frameCount: frameCount,
            kernelDebugOutput: "/tmp/mlp_peekrow_harmonic_graph.metal"
        )
        let compileTime = (CFAbsoluteTimeGetCurrent() - compileStart) * 1000
        print("Compile time: \(String(format: "%.2f", compileTime))ms")

        // Warmup
        _ = ctx.trainStep()
        _ = ctx.trainStep()

        let initialLoss = ctx.trainStep()
        print("Initial loss: \(initialLoss)")

        // Training loop with timing
        let epochs = 200
        var finalLoss = initialLoss
        let trainStart = CFAbsoluteTimeGetCurrent()
        for i in 0..<epochs {
            finalLoss = ctx.trainStep()
            print("Epoch \(i): loss = \(String(format: "%.6f", finalLoss))")
            if finalLoss < initialLoss * 0.25 {
                break
            }
        }
        let trainTime = (CFAbsoluteTimeGetCurrent() - trainStart) * 1000
        let timePerEpoch = trainTime / Double(epochs)

        print("\nFinal loss: \(String(format: "%.6f", finalLoss))")
        print("Loss reduction: \(String(format: "%.2f", initialLoss / finalLoss))x")
        print("\n--- Performance ---")
        print("Total train time (\(epochs) epochs): \(String(format: "%.2f", trainTime))ms")
        print("Time per epoch: \(String(format: "%.3f", timePerEpoch))ms")
        print("Epochs per second: \(String(format: "%.1f", 1000.0 / timePerEpoch))")

        XCTAssertLessThan(
            finalLoss, initialLoss * 0.25, "Loss should drop for teacher-student setup")
    }

    /// Test GraphTrainingContext with multiple parameters
    /// Minimize (a*b - 6)^2, finding a=2, b=3 (or any factorization)
    func testGraphTrainingContextMultiParam() throws {
        let g = Graph()

        // Create trainable parameters
        let a = GraphParameter(graph: g, value: 1.0, name: "a")
        let b = GraphParameter(graph: g, value: 1.0, name: "b")

        // Target: a * b = 6
        let target = g.n(.constant(6.0))
        let product = g.n(.mul, [a.node(), b.node()])
        let loss = g.n(.mse, [product, target])

        _ = g.n(.output(0), [loss])

        let ctx = try GraphTrainingContext(
            graph: g,
            loss: loss,
            parameters: [a, b],
            optimizer: GraphAdam(),
            learningRate: 0.1,
            frameCount: 1
        )

        print("\n=== GraphTrainingContext Multi-Param Test ===")
        print("Finding a, b such that a*b = 6")

        var lastLoss: Float = 0
        for step in 0..<100 {
            lastLoss = ctx.trainStep()
            print(
                "Step \(step): a=\(a.value), b=\(b.value), a*b=\(a.value * b.value), loss=\(lastLoss)"
            )
        }

        // Product should be close to 6
        XCTAssertEqual(a.value * b.value, 6.0, accuracy: 0.5, "a*b should converge to 6")
    }
}
