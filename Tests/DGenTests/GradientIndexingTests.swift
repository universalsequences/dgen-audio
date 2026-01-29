import XCTest

@testable import DGen

/// Tests to diagnose gradient indexing issues in multi-layer networks.
/// These tests verify that all tensor elements receive gradients through
/// chains of operations like transpose -> reshape -> sum.
final class GradientIndexingTests: XCTestCase {

    // MARK: - Test Infrastructure

    /// Runs a single forward/backward pass and extracts gradients.
    private func runAndExtractGradients(
        graph g: Graph,
        parameters: [TensorParameter],
        lossNode: NodeID
    ) throws -> [[Float]] {
        let frameCount = 1
        let result = try CompilationPipeline.compile(
            graph: g, backend: .metal,
            options: .init(frameCount: frameCount, backwards: true))

        let runtime = try MetalCompiledKernel(
            kernels: result.kernels,
            cellAllocations: result.cellAllocations,
            context: result.context)

        let ctx = TrainingContext(
            tensorParameters: parameters,
            optimizer: SGD(lr: 0.01),
            lossNode: lossNode)

        ctx.initializeMemory(
            runtime: runtime,
            cellAllocations: result.cellAllocations,
            context: result.context,
            frameCount: frameCount,
            graph: g)
        _ = ctx.runStepGPU()

        return parameters.map { $0.grads }
    }

    // MARK: - Diagnostic Test 1: Gradient Index Tracing

    /// Test that traces gradient indices through reshape -> transpose chain.
    /// This mirrors what happens inside matmul.
    func testGradientIndexTracing() throws {
        let g = Graph()

        // Create a simple chain: tensor -> transpose -> reshape -> sum
        let w = TensorParameter(
            graph: g, shape: [2, 4],
            data: [1, 2, 3, 4, 5, 6, 7, 8], name: "W")

        // Transpose: [2,4] -> [4,2]
        let wT = try g.transpose(w.node(), axes: [1, 0])

        // Reshape: [4,2] -> [1,4,2]
        let wTR = try g.reshape(wT, to: [1, 4, 2])

        // Sum to create gradient flow
        let result = g.n(.sum, wTR)
        let loss = g.n(.mse, result, g.n(.constant(0.0)))
        _ = g.n(.output(0), loss)

        // All 8 W elements should get equal gradients (since sum broadcasts equally)
        let grads = try runAndExtractGradients(graph: g, parameters: [w], lossNode: loss)

        print("Gradient index tracing - W grads: \(grads[0])")

        // Verify ALL elements have non-zero gradients
        for (i, grad) in grads[0].enumerated() {
            XCTAssertNotEqual(
                grad, 0.0, accuracy: 0.001,
                "W[\(i)] should have non-zero gradient")
        }

        // All gradients should be equal (sum broadcasts equally)
        let firstGrad = grads[0][0]
        for (i, grad) in grads[0].enumerated() {
            XCTAssertEqual(
                grad, firstGrad, accuracy: 0.1,
                "All gradients should be equal, but W[\(i)]=\(grad) differs from W[0]=\(firstGrad)")
        }
    }

    // MARK: - Diagnostic Test 2: Isolate transpose backward

    /// Test transpose backward in isolation.
    func testTransposeBackwardIsolated() throws {
        let g = Graph()

        // Simple 2x3 tensor
        let w = TensorParameter(
            graph: g, shape: [2, 3],
            data: [1, 2, 3, 4, 5, 6], name: "W")

        // Transpose: [2,3] -> [3,2]
        let wT = try g.transpose(w.node(), axes: [1, 0])

        // Sum and loss
        let result = g.n(.sum, wT)
        let loss = g.n(.mse, result, g.n(.constant(0.0)))
        _ = g.n(.output(0), loss)

        let grads = try runAndExtractGradients(graph: g, parameters: [w], lossNode: loss)

        print("Transpose backward isolated - W grads: \(grads[0])")

        // All 6 elements should have equal non-zero gradients
        for (i, grad) in grads[0].enumerated() {
            XCTAssertNotEqual(
                grad, 0.0, accuracy: 0.001,
                "W[\(i)] should have non-zero gradient after transpose")
        }
    }

    // MARK: - Diagnostic Test 3: Isolate sumAxis backward with non-contiguous tensor

    /// Test sumAxis backward when the input is a transposed (non-contiguous) tensor.
    func testSumAxisBackwardWithNonContiguousTensor() throws {
        let g = Graph()

        // Create a 2x3 tensor
        let w = TensorParameter(
            graph: g, shape: [2, 3],
            data: [1, 2, 3, 4, 5, 6], name: "W")

        // Transpose to get [3, 2] with strides [1, 3]
        let wT = try g.transpose(w.node(), axes: [1, 0])  // [3, 2]

        // sumAxis on the transposed tensor
        let summed = try g.sum(wT, axis: 0)  // [2]
        let loss = g.n(.sum, summed)
        _ = g.n(.output(0), loss)

        let grads = try runAndExtractGradients(graph: g, parameters: [w], lossNode: loss)

        print("sumAxis with non-contiguous input - W grads: \(grads[0])")

        // All 6 elements of original tensor should get gradients
        for (i, grad) in grads[0].enumerated() {
            XCTAssertNotEqual(
                grad, 0.0, accuracy: 0.001,
                "W[\(i)] should have non-zero gradient through transpose->sumAxis")
        }
    }

    // MARK: - Diagnostic Test 4: Binary broadcast backward

    /// Test binary operation backward with NO broadcasting (same shapes).
    func testBinarySameShapeBackwardIndexing() throws {
        let g = Graph()

        // Same shape test: [2, 2] + [2, 2]
        let x = TensorParameter(
            graph: g, shape: [2, 2],
            data: [1, 2, 3, 4], name: "x")

        let y = TensorParameter(
            graph: g, shape: [2, 2],
            data: [0.1, 0.1, 0.1, 0.1], name: "y")

        // x + y (same shapes, no broadcasting)
        let added = g.n(.add, x.node(), y.node())
        let output = g.n(.sum, added)
        let loss = g.n(.mse, output, g.n(.constant(20.0)))
        _ = g.n(.output(0), loss)

        let grads = try runAndExtractGradients(graph: g, parameters: [x, y], lossNode: loss)

        print("Same shape add - x grads: \(grads[0])")
        print("Same shape add - y grads: \(grads[1])")

        // All elements should have equal gradients
        for (i, grad) in grads[0].enumerated() {
            XCTAssertNotEqual(
                grad, 0.0, accuracy: 0.001,
                "x[\(i)] should have non-zero gradient")
        }
        for (i, grad) in grads[1].enumerated() {
            XCTAssertNotEqual(
                grad, 0.0, accuracy: 0.001,
                "y[\(i)] should have non-zero gradient")
        }
    }

    /// Test binary operation backward with broadcasting.
    func testBinaryBroadcastBackwardIndexing() throws {
        let g = Graph()

        // Simplified test: [2, 2] + [2] broadcasting
        let x = TensorParameter(
            graph: g, shape: [2, 2],
            data: [1, 2, 3, 4], name: "x")

        // Bias tensor: [2] (broadcasts along axis 0)
        let bias = TensorParameter(
            graph: g, shape: [2],
            data: [0.1, 0.1], name: "bias")

        // x + bias with broadcasting
        let added = g.n(.add, x.node(), bias.node())
        let output = g.n(.sum, added)
        let loss = g.n(.mse, output, g.n(.constant(20.0)))
        _ = g.n(.output(0), loss)

        // Compile and print kernel source
        let frameCount = 1
        let result = try CompilationPipeline.compile(
            graph: g, backend: .metal,
            options: .init(frameCount: frameCount, backwards: true))

        print("\n=== Generated Kernel Source ===")
        for kernel in result.kernels {
            print("Kernel: \(kernel.name)")
            print(kernel.source)
        }
        print("=== End Kernel Source ===\n")

        let grads = try runAndExtractGradients(graph: g, parameters: [x, bias], lossNode: loss)

        print("Binary broadcast - x shape: [2, 2], bias shape: [2]")
        print("Binary broadcast - x grads: \(grads[0])")
        print("Binary broadcast - bias grads: \(grads[1])")

        // All x elements should have equal gradients
        for (i, grad) in grads[0].enumerated() {
            XCTAssertNotEqual(
                grad, 0.0, accuracy: 0.001,
                "x[\(i)] should have non-zero gradient")
        }

        // All bias elements should have gradients (accumulated from broadcasts)
        for (i, grad) in grads[1].enumerated() {
            XCTAssertNotEqual(
                grad, 0.0, accuracy: 0.001,
                "bias[\(i)] should have non-zero gradient")
        }
    }

    // MARK: - Diagnostic Test 5: Two-layer matmul gradient coverage

    /// Test that verifies all W1 elements get gradients in a two-layer network.
    func testTwoLayerMatmulGradientCoverage() throws {
        let g = Graph()

        // Input: [4, 2]
        let inputs = g.tensor(
            shape: [4, 2],
            data: [
                0.0, 0.0,
                0.0, 1.0,
                1.0, 0.0,
                1.0, 1.0,
            ])

        // W1: [2, 4] - first layer weights
        let w1 = TensorParameter(
            graph: g, shape: [2, 4],
            data: [
                0.1, 0.2, -0.1, -0.2,
                0.2, 0.1, -0.2, -0.1,
            ], name: "W1")

        // W2: [4, 1] - second layer weights
        let w2 = TensorParameter(
            graph: g, shape: [4, 1],
            data: [0.1, 0.1, 0.1, 0.1], name: "W2")

        // Forward: hidden = inputs @ W1 (no activation), output = hidden @ W2
        let hidden = try g.matmul(inputs, w1.node())  // [4, 4]
        let output = try g.matmul(hidden, w2.node())  // [4, 1]

        // Loss
        let summed = g.n(.sum, output)
        let loss = g.n(.mse, summed, g.n(.constant(10.0)))
        _ = g.n(.output(0), loss)

        let grads = try runAndExtractGradients(graph: g, parameters: [w1, w2], lossNode: loss)

        print("Two-layer matmul - W1 grads: \(grads[0])")
        print("Two-layer matmul - W2 grads: \(grads[1])")

        // Count non-zero gradients for W1
        let w1NonZeroCount = grads[0].filter { abs($0) > 0.001 }.count
        print("W1 non-zero gradient count: \(w1NonZeroCount) / \(grads[0].count)")

        // ALL 8 W1 elements should have non-zero gradients
        XCTAssertEqual(
            w1NonZeroCount, 8,
            "All 8 W1 elements should have non-zero gradients, but only \(w1NonZeroCount) do")

        // Verify each W1 element individually
        for (i, grad) in grads[0].enumerated() {
            XCTAssertNotEqual(
                grad, 0.0, accuracy: 0.001,
                "W1[\(i)] should have non-zero gradient")
        }
    }

    // MARK: - Diagnostic Test 6: Three-layer network

    /// Test gradient flow through a 3-layer network.
    func testThreeLayerNetworkGradientCoverage() throws {
        let g = Graph()

        // Input: [4, 2]
        let inputs = g.tensor(
            shape: [4, 2],
            data: [
                0.0, 0.0,
                0.0, 1.0,
                1.0, 0.0,
                1.0, 1.0,
            ])

        // Layer 1: 2 -> 4 (use all positive values to avoid numerical edge case)
        let w1 = TensorParameter(
            graph: g, shape: [2, 4],
            data: [0.1, 0.2, 0.15, 0.25, 0.2, 0.1, 0.25, 0.15], name: "W1")

        // Layer 2: 4 -> 4
        let w2 = TensorParameter(
            graph: g, shape: [4, 4],
            data: [
                0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1,
            ], name: "W2")

        // Layer 3: 4 -> 1
        let w3 = TensorParameter(
            graph: g, shape: [4, 1],
            data: [0.1, 0.1, 0.1, 0.1], name: "W3")

        // Forward: h1 = tanh(inputs @ W1), h2 = tanh(h1 @ W2), output = h2 @ W3
        let h1_linear = try g.matmul(inputs, w1.node())
        let h1 = g.n(.tanh, h1_linear)

        let h2_linear = try g.matmul(h1, w2.node())
        let h2 = g.n(.tanh, h2_linear)

        let output = try g.matmul(h2, w3.node())

        // Loss
        let summed = g.n(.sum, output)
        let loss = g.n(.mse, summed, g.n(.constant(2.0)))
        _ = g.n(.output(0), loss)

        let grads = try runAndExtractGradients(graph: g, parameters: [w1, w2, w3], lossNode: loss)

        print("Three-layer network - W1 nodeId: \(w1.nodeId), baseGradId: \(w1.baseGradId ?? -1)")
        print("Three-layer network - W2 nodeId: \(w2.nodeId), baseGradId: \(w2.baseGradId ?? -1)")
        print("Three-layer network - W3 nodeId: \(w3.nodeId), baseGradId: \(w3.baseGradId ?? -1)")
        print("Three-layer network - W1 grads: \(grads[0])")
        print("Three-layer network - W2 grads: \(grads[1])")
        print("Three-layer network - W3 grads: \(grads[2])")

        // Count non-zero gradients
        let w1NonZero = grads[0].filter { abs($0) > 0.001 }.count
        let w2NonZero = grads[1].filter { abs($0) > 0.001 }.count
        let w3NonZero = grads[2].filter { abs($0) > 0.001 }.count

        print("W1 non-zero: \(w1NonZero)/\(grads[0].count)")
        print("W2 non-zero: \(w2NonZero)/\(grads[1].count)")
        print("W3 non-zero: \(w3NonZero)/\(grads[2].count)")

        // All layers should have non-zero gradients
        XCTAssertEqual(w1NonZero, 8, "All W1 elements should have gradients")
        XCTAssertEqual(w2NonZero, 16, "All W2 elements should have gradients")
        XCTAssertEqual(w3NonZero, 4, "All W3 elements should have gradients")
    }

    /// Test single matmul backward to isolate W3 issue
    func testSingleMatmulBackward() throws {
        let g = Graph()

        // Simple: [4, 4] @ [4, 1] = [4, 1]
        let h = g.tensor(shape: [4, 4], data: [Float](repeating: 0.5, count: 16))
        let w = TensorParameter(
            graph: g, shape: [4, 1],
            data: [0.1, 0.1, 0.1, 0.1], name: "W")

        let output = try g.matmul(h, w.node())
        let summed = g.n(.sum, output)
        let loss = g.n(.mse, summed, g.n(.constant(2.0)))
        _ = g.n(.output(0), loss)

        let grads = try runAndExtractGradients(graph: g, parameters: [w], lossNode: loss)

        print("Single matmul - W grads: \(grads[0])")
        let nonZero = grads[0].filter { abs($0) > 0.001 }.count
        print("W non-zero: \(nonZero)/\(grads[0].count)")

        XCTAssertEqual(nonZero, 4, "All W elements should have gradients")
    }

    /// Test matmul after tanh (computed tensor input)
    func testMatmulAfterTanhBackward() throws {
        let g = Graph()

        // h_base is a learnable tensor
        let h_base = TensorParameter(
            graph: g, shape: [4, 4],
            data: [Float](repeating: 0.5, count: 16), name: "H")

        // h = tanh(h_base) - computed tensor
        let h = g.n(.tanh, h_base.node())

        // W is learnable
        let w = TensorParameter(
            graph: g, shape: [4, 1],
            data: [0.1, 0.1, 0.1, 0.1], name: "W")

        // output = h @ W
        let output = try g.matmul(h, w.node())
        let summed = g.n(.sum, output)
        let loss = g.n(.mse, summed, g.n(.constant(2.0)))
        _ = g.n(.output(0), loss)

        let grads = try runAndExtractGradients(graph: g, parameters: [h_base, w], lossNode: loss)

        print("Matmul after tanh - H grads: \(grads[0])")
        print("Matmul after tanh - W grads: \(grads[1])")

        let hNonZero = grads[0].filter { abs($0) > 0.001 }.count
        let wNonZero = grads[1].filter { abs($0) > 0.001 }.count

        print("H non-zero: \(hNonZero)/\(grads[0].count)")
        print("W non-zero: \(wNonZero)/\(grads[1].count)")

        XCTAssertGreaterThan(hNonZero, 0, "H should have gradients")
        XCTAssertEqual(wNonZero, 4, "All W elements should have gradients")
    }

    /// Test two matmuls in sequence
    func testTwoMatmulsInSequence() throws {
        let g = Graph()

        let input = g.tensor(shape: [4, 2], data: [Float](repeating: 0.5, count: 8))

        let w1 = TensorParameter(
            graph: g, shape: [2, 4],
            data: [Float](repeating: 0.1, count: 8), name: "W1")
        let w2 = TensorParameter(
            graph: g, shape: [4, 1],
            data: [0.1, 0.1, 0.1, 0.1], name: "W2")

        // Two matmuls without activation
        let h = try g.matmul(input, w1.node())  // [4,4]
        let output = try g.matmul(h, w2.node())  // [4,1]

        let summed = g.n(.sum, output)
        let loss = g.n(.mse, summed, g.n(.constant(2.0)))
        _ = g.n(.output(0), loss)

        let grads = try runAndExtractGradients(graph: g, parameters: [w1, w2], lossNode: loss)

        print("Two matmuls - W1 grads: \(grads[0])")
        print("Two matmuls - W2 grads: \(grads[1])")

        let w1NonZero = grads[0].filter { abs($0) > 0.001 }.count
        let w2NonZero = grads[1].filter { abs($0) > 0.001 }.count

        print("W1 non-zero: \(w1NonZero)/\(grads[0].count)")
        print("W2 non-zero: \(w2NonZero)/\(grads[1].count)")

        XCTAssertEqual(w1NonZero, 8, "All W1 elements should have gradients")
        XCTAssertEqual(w2NonZero, 4, "All W2 elements should have gradients")
    }

    /// Test three matmuls in sequence (no activation)
    func testThreeMatmulsInSequence() throws {
        let g = Graph()

        let input = g.tensor(shape: [4, 2], data: [Float](repeating: 0.5, count: 8))

        let w1 = TensorParameter(
            graph: g, shape: [2, 4],
            data: [Float](repeating: 0.1, count: 8), name: "W1")
        let w2 = TensorParameter(
            graph: g, shape: [4, 4],
            data: [Float](repeating: 0.1, count: 16), name: "W2")
        let w3 = TensorParameter(
            graph: g, shape: [4, 1],
            data: [0.1, 0.1, 0.1, 0.1], name: "W3")

        // Three matmuls without activation
        let h1 = try g.matmul(input, w1.node())  // [4,4]
        let h2 = try g.matmul(h1, w2.node())     // [4,4]
        let output = try g.matmul(h2, w3.node()) // [4,1]

        let summed = g.n(.sum, output)
        let loss = g.n(.mse, summed, g.n(.constant(2.0)))
        _ = g.n(.output(0), loss)

        let grads = try runAndExtractGradients(graph: g, parameters: [w1, w2, w3], lossNode: loss)

        print("Three matmuls - W1 grads: \(grads[0])")
        print("Three matmuls - W2 grads: \(grads[1])")
        print("Three matmuls - W3 grads: \(grads[2])")

        let w1NonZero = grads[0].filter { abs($0) > 0.001 }.count
        let w2NonZero = grads[1].filter { abs($0) > 0.001 }.count
        let w3NonZero = grads[2].filter { abs($0) > 0.001 }.count

        print("W1 non-zero: \(w1NonZero)/\(grads[0].count)")
        print("W2 non-zero: \(w2NonZero)/\(grads[1].count)")
        print("W3 non-zero: \(w3NonZero)/\(grads[2].count)")

        XCTAssertEqual(w1NonZero, 8, "All W1 elements should have gradients")
        XCTAssertEqual(w2NonZero, 16, "All W2 elements should have gradients")
        XCTAssertEqual(w3NonZero, 4, "All W3 elements should have gradients")
    }

    // MARK: - Diagnostic Test 7: Matmul decomposition chain

    /// Test the specific chain of operations inside matmul backward:
    /// dL/dA = dL/dC @ B^T, which involves transpose and broadcasting.
    func testMatmulDecompositionChain() throws {
        let g = Graph()

        // Simulate the backward pass for A in C = A @ B
        // dL/dA = dL/dC @ B^T

        // Gradient from output: dL/dC [4, 1]
        let gradC = TensorParameter(
            graph: g, shape: [4, 1],
            data: [1.0, 1.0, 1.0, 1.0], name: "gradC")

        // B^T [1, 4] (transposed from [4, 1])
        let bT = TensorParameter(
            graph: g, shape: [1, 4],
            data: [0.1, 0.2, 0.3, 0.4], name: "BT")

        // dL/dA = gradC @ bT -> [4, 4]
        let gradA = try g.matmul(gradC.node(), bT.node())

        // Sum and loss (to create gradient flow back to gradC and bT)
        let summed = g.n(.sum, gradA)
        let loss = g.n(.mse, summed, g.n(.constant(10.0)))
        _ = g.n(.output(0), loss)

        let grads = try runAndExtractGradients(graph: g, parameters: [gradC, bT], lossNode: loss)

        print("Matmul decomposition - gradC grads: \(grads[0])")
        print("Matmul decomposition - bT grads: \(grads[1])")

        // All elements should have non-zero gradients
        for (i, grad) in grads[0].enumerated() {
            XCTAssertNotEqual(
                grad, 0.0, accuracy: 0.001,
                "gradC[\(i)] should have non-zero gradient")
        }
        for (i, grad) in grads[1].enumerated() {
            XCTAssertNotEqual(
                grad, 0.0, accuracy: 0.001,
                "bT[\(i)] should have non-zero gradient")
        }
    }

    /// Test 3-layer with mixed positive/negative W1 values.
    ///
    /// NOTE: This test documents an interesting edge case. When W1 has symmetric
    /// positive/negative values like [0.1, 0.2, -0.1, -0.2, ...] and W2 is uniform,
    /// the h1 rows sum to exactly 0, causing h2 = 0 and thus dL/dW3 = 0.
    /// This is mathematically correct, not a bug.
    ///
    /// To test actual gradient flow through 3 layers with mixed weights, we use
    /// non-uniform W2 to break the symmetry.
    func testThreeLayerNegativeW1() throws {
        let g = Graph()

        let input = g.tensor(
            shape: [4, 2],
            data: [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])

        // W1 with mixed positive/negative values
        let w1 = TensorParameter(
            graph: g, shape: [2, 4],
            data: [0.1, 0.2, -0.1, -0.2, 0.2, 0.1, -0.2, -0.1], name: "W1")

        // Use non-uniform W2 to break the h1 row-sum cancellation
        // Different column weights means h1 @ W2 won't be zero even when h1 rows sum to 0
        let w2 = TensorParameter(
            graph: g, shape: [4, 4],
            data: [
                0.1, 0.2, 0.15, 0.25,
                0.2, 0.1, 0.25, 0.15,
                0.15, 0.25, 0.1, 0.2,
                0.25, 0.15, 0.2, 0.1,
            ], name: "W2")
        let w3 = TensorParameter(
            graph: g, shape: [4, 1],
            data: [0.1, 0.1, 0.1, 0.1], name: "W3")

        let h1_linear = try g.matmul(input, w1.node())
        let h1 = g.n(.tanh, h1_linear)
        let h2_linear = try g.matmul(h1, w2.node())
        let h2 = g.n(.tanh, h2_linear)
        let output = try g.matmul(h2, w3.node())

        let summed = g.n(.sum, output)
        let loss = g.n(.mse, summed, g.n(.constant(10.0)))
        _ = g.n(.output(0), loss)

        let grads = try runAndExtractGradients(graph: g, parameters: [w1, w2, w3], lossNode: loss)

        print("3-layer mixed W1, non-uniform W2:")
        print("  W1 grads: \(grads[0])")
        print("  W2 grads: \(grads[1])")
        print("  W3 grads: \(grads[2])")

        let w1NonZero = grads[0].filter { abs($0) > 0.001 }.count
        let w2NonZero = grads[1].filter { abs($0) > 0.001 }.count
        let w3NonZero = grads[2].filter { abs($0) > 0.001 }.count

        print("W1 non-zero: \(w1NonZero)/\(grads[0].count)")
        print("W2 non-zero: \(w2NonZero)/\(grads[1].count)")
        print("W3 non-zero: \(w3NonZero)/\(grads[2].count)")

        // All layers should have non-zero gradients
        XCTAssertEqual(w1NonZero, 8, "All W1 elements should have gradients")
        XCTAssertEqual(w2NonZero, 16, "All W2 elements should have gradients")
        XCTAssertEqual(w3NonZero, 4, "All W3 elements should have gradients")
    }

    /// Test the edge case where symmetric W1 and uniform W2 cause h2 = 0.
    /// This documents that zero gradients for W3 are mathematically correct in this case.
    func testThreeLayerSymmetricCancellation() throws {
        let g = Graph()

        let input = g.tensor(
            shape: [4, 2],
            data: [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])

        // Symmetric W1 values that cause h1 rows to sum to 0
        let w1 = TensorParameter(
            graph: g, shape: [2, 4],
            data: [0.1, 0.2, -0.1, -0.2, 0.2, 0.1, -0.2, -0.1], name: "W1")

        // Uniform W2 - when h1 rows sum to 0, h1 @ W2 = 0
        let w2 = TensorParameter(
            graph: g, shape: [4, 4],
            data: [Float](repeating: 0.1, count: 16), name: "W2")
        let w3 = TensorParameter(
            graph: g, shape: [4, 1],
            data: [0.1, 0.1, 0.1, 0.1], name: "W3")

        let h1_linear = try g.matmul(input, w1.node())
        let h1 = g.n(.tanh, h1_linear)
        let h2_linear = try g.matmul(h1, w2.node())
        let h2 = g.n(.tanh, h2_linear)
        let output = try g.matmul(h2, w3.node())

        let summed = g.n(.sum, output)
        let loss = g.n(.mse, summed, g.n(.constant(10.0)))
        _ = g.n(.output(0), loss)

        let grads = try runAndExtractGradients(graph: g, parameters: [w1, w2, w3], lossNode: loss)

        print("3-layer symmetric cancellation case:")
        print("  W1 grads: \(grads[0])")
        print("  W2 grads: \(grads[1])")
        print("  W3 grads: \(grads[2])")

        // W3 has zero gradients because h2 = 0 (mathematically correct)
        // W1 and W2 still have non-zero gradients because gradient flows through tanh'(0) = 1
        let w1NonZero = grads[0].filter { abs($0) > 0.001 }.count
        let w2NonZero = grads[1].filter { abs($0) > 0.001 }.count
        let w3NonZero = grads[2].filter { abs($0) > 0.001 }.count

        XCTAssertGreaterThan(w1NonZero, 0, "W1 should have gradients (gradient flows through tanh)")
        XCTAssertGreaterThan(w2NonZero, 0, "W2 should have gradients (gradient flows through tanh)")
        // W3 gradients are zero because h2 = 0, this is mathematically correct
        XCTAssertEqual(w3NonZero, 0, "W3 has zero gradients because h2 = 0 (expected)")
    }

    /// Test 3-layer with XOR input data (has zeros)
    func testThreeLayerXORInput() throws {
        let g = Graph()

        // XOR-like input: some values are 0
        let input = g.tensor(
            shape: [4, 2],
            data: [
                0.0, 0.0,
                0.0, 1.0,
                1.0, 0.0,
                1.0, 1.0,
            ])

        let w1 = TensorParameter(
            graph: g, shape: [2, 4],
            data: [Float](repeating: 0.1, count: 8), name: "W1")
        let w2 = TensorParameter(
            graph: g, shape: [4, 4],
            data: [Float](repeating: 0.1, count: 16), name: "W2")
        let w3 = TensorParameter(
            graph: g, shape: [4, 1],
            data: [0.1, 0.1, 0.1, 0.1], name: "W3")

        let h1_linear = try g.matmul(input, w1.node())
        let h1 = g.n(.tanh, h1_linear)
        let h2_linear = try g.matmul(h1, w2.node())
        let h2 = g.n(.tanh, h2_linear)
        let output = try g.matmul(h2, w3.node())

        let summed = g.n(.sum, output)
        let loss = g.n(.mse, summed, g.n(.constant(2.0)))
        _ = g.n(.output(0), loss)

        let grads = try runAndExtractGradients(graph: g, parameters: [w1, w2, w3], lossNode: loss)

        print("3-layer XOR input - W3 grads: \(grads[2])")
        let w3NonZero = grads[2].filter { abs($0) > 0.001 }.count
        print("W3 non-zero: \(w3NonZero)/\(grads[2].count)")
    }

    /// Test 3-layer with non-square W1
    func testThreeLayerNonSquareW1() throws {
        let g = Graph()

        // Match original failing test: input [4,2], W1 [2,4]
        let input = g.tensor(shape: [4, 2], data: [Float](repeating: 0.5, count: 8))

        let w1 = TensorParameter(
            graph: g, shape: [2, 4],
            data: [Float](repeating: 0.1, count: 8), name: "W1")
        let w2 = TensorParameter(
            graph: g, shape: [4, 4],
            data: [Float](repeating: 0.1, count: 16), name: "W2")
        let w3 = TensorParameter(
            graph: g, shape: [4, 1],
            data: [0.1, 0.1, 0.1, 0.1], name: "W3")

        // matmul -> tanh -> matmul -> tanh -> matmul
        let h1_linear = try g.matmul(input, w1.node())  // [4,4]
        let h1 = g.n(.tanh, h1_linear)
        let h2_linear = try g.matmul(h1, w2.node())     // [4,4]
        let h2 = g.n(.tanh, h2_linear)
        let output = try g.matmul(h2, w3.node())        // [4,1]

        let summed = g.n(.sum, output)
        let loss = g.n(.mse, summed, g.n(.constant(2.0)))
        _ = g.n(.output(0), loss)

        let grads = try runAndExtractGradients(graph: g, parameters: [w1, w2, w3], lossNode: loss)

        print("3-layer non-square W1 - W1 grads: \(grads[0])")
        print("3-layer non-square W1 - W2 grads: \(grads[1])")
        print("3-layer non-square W1 - W3 grads: \(grads[2])")

        let w1NonZero = grads[0].filter { abs($0) > 0.001 }.count
        let w2NonZero = grads[1].filter { abs($0) > 0.001 }.count
        let w3NonZero = grads[2].filter { abs($0) > 0.001 }.count

        print("W1 non-zero: \(w1NonZero)/\(grads[0].count)")
        print("W2 non-zero: \(w2NonZero)/\(grads[1].count)")
        print("W3 non-zero: \(w3NonZero)/\(grads[2].count)")

        // Note: This test may fail with current implementation
    }

    /// Test 3-layer with square input
    func testThreeLayerSquareInput() throws {
        let g = Graph()

        // Input [4,4] instead of [4,2]
        let input = g.tensor(shape: [4, 4], data: [Float](repeating: 0.5, count: 16))

        let w1 = TensorParameter(
            graph: g, shape: [4, 4],
            data: [Float](repeating: 0.1, count: 16), name: "W1")
        let w2 = TensorParameter(
            graph: g, shape: [4, 4],
            data: [Float](repeating: 0.1, count: 16), name: "W2")
        let w3 = TensorParameter(
            graph: g, shape: [4, 1],
            data: [0.1, 0.1, 0.1, 0.1], name: "W3")

        // matmul -> tanh -> matmul -> tanh -> matmul
        let h1_linear = try g.matmul(input, w1.node())
        let h1 = g.n(.tanh, h1_linear)
        let h2_linear = try g.matmul(h1, w2.node())
        let h2 = g.n(.tanh, h2_linear)
        let output = try g.matmul(h2, w3.node())

        let summed = g.n(.sum, output)
        let loss = g.n(.mse, summed, g.n(.constant(2.0)))
        _ = g.n(.output(0), loss)

        let grads = try runAndExtractGradients(graph: g, parameters: [w1, w2, w3], lossNode: loss)

        print("3-layer square input - W1 grads: \(grads[0])")
        print("3-layer square input - W2 grads: \(grads[1])")
        print("3-layer square input - W3 grads: \(grads[2])")

        let w1NonZero = grads[0].filter { abs($0) > 0.001 }.count
        let w2NonZero = grads[1].filter { abs($0) > 0.001 }.count
        let w3NonZero = grads[2].filter { abs($0) > 0.001 }.count

        print("W1 non-zero: \(w1NonZero)/\(grads[0].count)")
        print("W2 non-zero: \(w2NonZero)/\(grads[1].count)")
        print("W3 non-zero: \(w3NonZero)/\(grads[2].count)")

        XCTAssertEqual(w1NonZero, 16, "All W1 elements should have gradients")
        XCTAssertEqual(w2NonZero, 16, "All W2 elements should have gradients")
        XCTAssertEqual(w3NonZero, 4, "All W3 elements should have gradients")
    }

    /// Test: tanh output of matmul as input to another matmul
    func testTanhMatmulToMatmul() throws {
        let g = Graph()

        let input = g.tensor(shape: [4, 4], data: [Float](repeating: 0.5, count: 16))

        let w1 = TensorParameter(
            graph: g, shape: [4, 4],
            data: [Float](repeating: 0.1, count: 16), name: "W1")
        let w2 = TensorParameter(
            graph: g, shape: [4, 1],
            data: [0.1, 0.1, 0.1, 0.1], name: "W2")

        // input @ W1 -> tanh -> @ W2
        let h_linear = try g.matmul(input, w1.node())  // [4,4]
        let h = g.n(.tanh, h_linear)                   // [4,4]
        let output = try g.matmul(h, w2.node())        // [4,1]

        let summed = g.n(.sum, output)
        let loss = g.n(.mse, summed, g.n(.constant(2.0)))
        _ = g.n(.output(0), loss)

        let grads = try runAndExtractGradients(graph: g, parameters: [w1, w2], lossNode: loss)

        print("tanh(matmul) -> matmul - W1 grads: \(grads[0])")
        print("tanh(matmul) -> matmul - W2 grads: \(grads[1])")

        let w1NonZero = grads[0].filter { abs($0) > 0.001 }.count
        let w2NonZero = grads[1].filter { abs($0) > 0.001 }.count

        print("W1 non-zero: \(w1NonZero)/\(grads[0].count)")
        print("W2 non-zero: \(w2NonZero)/\(grads[1].count)")

        XCTAssertEqual(w1NonZero, 16, "All W1 elements should have gradients")
        XCTAssertEqual(w2NonZero, 4, "All W2 elements should have gradients")
    }

    /// Test 3 matmuls with tanh between layer 1-2 only (not layer 2-3)
    func testThreeMatmulsOneTanh() throws {
        let g = Graph()

        let input = g.tensor(shape: [4, 2], data: [Float](repeating: 0.5, count: 8))

        let w1 = TensorParameter(
            graph: g, shape: [2, 4],
            data: [Float](repeating: 0.1, count: 8), name: "W1")
        let w2 = TensorParameter(
            graph: g, shape: [4, 4],
            data: [Float](repeating: 0.1, count: 16), name: "W2")
        let w3 = TensorParameter(
            graph: g, shape: [4, 1],
            data: [0.1, 0.1, 0.1, 0.1], name: "W3")

        // matmul -> tanh -> matmul -> matmul (NO tanh before last)
        let h1_linear = try g.matmul(input, w1.node())  // [4,4]
        let h1 = g.n(.tanh, h1_linear)                  // [4,4]
        let h2 = try g.matmul(h1, w2.node())            // [4,4] - no tanh!
        let output = try g.matmul(h2, w3.node())        // [4,1]

        let summed = g.n(.sum, output)
        let loss = g.n(.mse, summed, g.n(.constant(2.0)))
        _ = g.n(.output(0), loss)

        let grads = try runAndExtractGradients(graph: g, parameters: [w1, w2, w3], lossNode: loss)

        print("Three matmuls, one tanh - W1 grads: \(grads[0])")
        print("Three matmuls, one tanh - W2 grads: \(grads[1])")
        print("Three matmuls, one tanh - W3 grads: \(grads[2])")

        let w1NonZero = grads[0].filter { abs($0) > 0.001 }.count
        let w2NonZero = grads[1].filter { abs($0) > 0.001 }.count
        let w3NonZero = grads[2].filter { abs($0) > 0.001 }.count

        print("W1 non-zero: \(w1NonZero)/\(grads[0].count)")
        print("W2 non-zero: \(w2NonZero)/\(grads[1].count)")
        print("W3 non-zero: \(w3NonZero)/\(grads[2].count)")

        XCTAssertEqual(w1NonZero, 8, "All W1 elements should have gradients")
        XCTAssertEqual(w2NonZero, 16, "All W2 elements should have gradients")
        XCTAssertEqual(w3NonZero, 4, "All W3 elements should have gradients")
    }

    /// Test the exact broadcast pattern in matmul: [4,1,4] * [1,1,4]
    func testMatmulBroadcastPattern() throws {
        let g = Graph()

        // Simulate the shapes inside h2 @ W3 matmul
        // a_reshaped: [4, 1, 4], b_reshaped: [1, 1, 4]
        let a = TensorParameter(
            graph: g, shape: [4, 1, 4],
            data: [Float](repeating: 0.5, count: 16), name: "A")
        let b = TensorParameter(
            graph: g, shape: [1, 1, 4],
            data: [0.1, 0.1, 0.1, 0.1], name: "B")

        let product = g.n(.mul, a.node(), b.node())  // [4, 1, 4]
        let summed = g.n(.sum, product)
        let loss = g.n(.mse, summed, g.n(.constant(2.0)))
        _ = g.n(.output(0), loss)

        let grads = try runAndExtractGradients(graph: g, parameters: [a, b], lossNode: loss)

        print("Matmul broadcast pattern [4,1,4] * [1,1,4]")
        print("A grads: \(grads[0])")
        print("B grads: \(grads[1])")

        let aNonZero = grads[0].filter { abs($0) > 0.001 }.count
        let bNonZero = grads[1].filter { abs($0) > 0.001 }.count

        print("A non-zero: \(aNonZero)/\(grads[0].count)")
        print("B non-zero: \(bNonZero)/\(grads[1].count)")

        XCTAssertEqual(aNonZero, 16, "All A elements should have gradients")
        XCTAssertEqual(bNonZero, 4, "All B elements should have gradients")
    }

    /// Test broadcast with symmetric input that causes gradient cancellation
    func testMatmulBroadcastWithSymmetricInput() throws {
        let g = Graph()

        // Simulate symmetric h2 values (positive and negative that sum to 0)
        // This mirrors what happens with mixed positive/negative W1
        // h2 after tanh with symmetric weights creates symmetric activation patterns
        let a = TensorParameter(
            graph: g, shape: [4, 1, 4],
            data: [
                // Row 0: all positive
                0.5, 0.5, 0.5, 0.5,
                // Row 1: half positive, half negative (symmetric)
                0.3, 0.3, -0.3, -0.3,
                // Row 2: half negative, half positive (symmetric)
                -0.3, -0.3, 0.3, 0.3,
                // Row 3: all positive
                0.5, 0.5, 0.5, 0.5,
            ], name: "A")
        let b = TensorParameter(
            graph: g, shape: [1, 1, 4],
            data: [0.1, 0.1, 0.1, 0.1], name: "B")

        let product = g.n(.mul, a.node(), b.node())  // [4, 1, 4]
        let summed = g.n(.sum, product)
        let loss = g.n(.mse, summed, g.n(.constant(2.0)))
        _ = g.n(.output(0), loss)

        let grads = try runAndExtractGradients(graph: g, parameters: [a, b], lossNode: loss)

        print("Broadcast with symmetric A - [4,1,4] * [1,1,4]")
        print("A data: rows sum to different values, columns 0+2 and 1+3 symmetric")
        print("A grads: \(grads[0])")
        print("B grads: \(grads[1])")

        let aNonZero = grads[0].filter { abs($0) > 0.001 }.count
        let bNonZero = grads[1].filter { abs($0) > 0.001 }.count

        print("A non-zero: \(aNonZero)/\(grads[0].count)")
        print("B non-zero: \(bNonZero)/\(grads[1].count)")

        // Key insight: Even with symmetric A, B should have non-zero gradients
        // because dL/dB[k] = sum over all i,j of (gradOut[i,j,k] * A[i,j,k])
        // The gradient only cancels if sum of A values is exactly 0 for that k
        XCTAssertEqual(bNonZero, 4, "All B elements should have gradients")
    }

    /// Test transpose backward with [4,1] shape
    func testTransposeBackward4x1() throws {
        let g = Graph()

        let w = TensorParameter(
            graph: g, shape: [4, 1],
            data: [0.1, 0.2, 0.3, 0.4], name: "W")

        // Transpose [4,1] -> [1,4]
        let wT = try g.transpose(w.node(), axes: [1, 0])

        let summed = g.n(.sum, wT)
        let loss = g.n(.mse, summed, g.n(.constant(2.0)))
        _ = g.n(.output(0), loss)

        let grads = try runAndExtractGradients(graph: g, parameters: [w], lossNode: loss)

        print("Transpose [4,1] -> [1,4] - W grads: \(grads[0])")
        let nonZero = grads[0].filter { abs($0) > 0.001 }.count
        print("W non-zero: \(nonZero)/\(grads[0].count)")

        XCTAssertEqual(nonZero, 4, "All W elements should have gradients")
    }

    /// Test two matmuls with tanh between
    func testTwoMatmulsWithTanh() throws {
        let g = Graph()

        let input = g.tensor(shape: [4, 2], data: [Float](repeating: 0.5, count: 8))

        let w1 = TensorParameter(
            graph: g, shape: [2, 4],
            data: [Float](repeating: 0.1, count: 8), name: "W1")
        let w2 = TensorParameter(
            graph: g, shape: [4, 1],
            data: [0.1, 0.1, 0.1, 0.1], name: "W2")

        // matmul -> tanh -> matmul
        let h1_linear = try g.matmul(input, w1.node())  // [4,4]
        let h1 = g.n(.tanh, h1_linear)                  // [4,4]
        let output = try g.matmul(h1, w2.node())        // [4,1]

        let summed = g.n(.sum, output)
        let loss = g.n(.mse, summed, g.n(.constant(2.0)))
        _ = g.n(.output(0), loss)

        let grads = try runAndExtractGradients(graph: g, parameters: [w1, w2], lossNode: loss)

        print("Two matmuls with tanh - W1 grads: \(grads[0])")
        print("Two matmuls with tanh - W2 grads: \(grads[1])")

        let w1NonZero = grads[0].filter { abs($0) > 0.001 }.count
        let w2NonZero = grads[1].filter { abs($0) > 0.001 }.count

        print("W1 non-zero: \(w1NonZero)/\(grads[0].count)")
        print("W2 non-zero: \(w2NonZero)/\(grads[1].count)")

        XCTAssertEqual(w1NonZero, 8, "All W1 elements should have gradients")
        XCTAssertEqual(w2NonZero, 4, "All W2 elements should have gradients")
    }
}
