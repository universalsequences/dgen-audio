import XCTest

@testable import DGen

final class TensorParameterTests: XCTestCase {

    // MARK: - Test Infrastructure

    /// Result from running a training session
    private struct TrainingResult {
        let initialLoss: Float
        let finalLoss: Float
        let gradients: [[Float]]
    }

    /// Runs a training loop and returns results for verification.
    private func runTrainingLoop(
        graph g: Graph,
        parameters: [TensorParameter],
        lossNode: NodeID,
        optimizer: Optimizer = Adam(lr: 0.1),
        epochs: Int = 200
    ) throws -> TrainingResult {
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
            optimizer: optimizer,
            lossNode: lossNode)

        ctx.initializeMemory(
            runtime: runtime,
            cellAllocations: result.cellAllocations,
            context: result.context,
            frameCount: frameCount,
            graph: g)

        let inputBuffer = [Float](repeating: 0.0, count: frameCount)
        var outputBuffer = [Float](repeating: 0.0, count: frameCount)

        // Run forward/backward to get initial state
        ctx.zeroGrad()
        inputBuffer.withUnsafeBufferPointer { inPtr in
            outputBuffer.withUnsafeMutableBufferPointer { outPtr in
                runtime.runWithMemory(
                    outputs: outPtr.baseAddress!,
                    inputs: inPtr.baseAddress!,
                    memory: ctx.getMemory(),
                    frameCount: frameCount)
            }
        }

        let initialLoss = outputBuffer[0]
        let initialGrads = ctx.extractTensorGradients()

        // Training loop
        for _ in 0..<epochs {
            ctx.step()
            ctx.zeroGrad()
            inputBuffer.withUnsafeBufferPointer { inPtr in
                outputBuffer.withUnsafeMutableBufferPointer { outPtr in
                    runtime.runWithMemory(
                        outputs: outPtr.baseAddress!,
                        inputs: inPtr.baseAddress!,
                        memory: ctx.getMemory(),
                        frameCount: frameCount)
                }
            }
        }

        return TrainingResult(
            initialLoss: initialLoss,
            finalLoss: outputBuffer[0],
            gradients: initialGrads)
    }

    /// Verifies that gradients are non-zero for all parameter tensors.
    private func assertNonZeroGradients(
        _ gradients: [[Float]],
        names: [String],
        file: StaticString = #file,
        line: UInt = #line
    ) {
        for (idx, grads) in gradients.enumerated() {
            let name = idx < names.count ? names[idx] : "param[\(idx)]"
            XCTAssertTrue(
                grads.contains { abs($0) > 0.001 },
                "\(name) gradients should be non-zero",
                file: file,
                line: line)
        }
    }

    // MARK: - Milestone 1: Tensor Gradient Allocation Infrastructure

    func testTensorGradientAllocation() throws {
        let g = Graph()
        let tensor = g.tensor(shape: [2, 3], data: [1, 2, 3, 4, 5, 6])
        let sum = g.n(.sum, tensor)
        _ = g.n(.output(0), sum)

        let result = try CompilationPipeline.compile(
            graph: g, backend: .metal,
            options: .init(frameCount: 1, backwards: true))

        // Verify 6 contiguous GradIDs allocated for the tensor
        XCTAssertEqual(result.context.tensorGradientSizes[tensor], 6)
        XCTAssertNotNil(result.context.tensorGradients[tensor])

        // Verify the base grad ID is valid
        if let baseGradId = result.context.tensorGradients[tensor] {
            XCTAssertGreaterThan(baseGradId, 0)
        }
    }

    // MARK: - Milestone 2: TensorParameter Class

    func testTensorParameterCreation() throws {
        let g = Graph()
        let weights = TensorParameter(graph: g, shape: [2, 3], name: "weights")

        XCTAssertEqual(weights.size, 6)
        XCTAssertEqual(weights.shape, [2, 3])
        XCTAssertEqual(weights.data.count, 6)
        XCTAssertNotNil(g.tensors[weights.tensorId])
    }

    func testTensorParameterWithData() throws {
        let g = Graph()
        let data: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        let weights = TensorParameter(graph: g, shape: [2, 3], data: data, name: "weights")

        XCTAssertEqual(weights.data, data)
        XCTAssertEqual(weights.size, 6)
    }

    func testTensorParameterXavierInit() throws {
        let g = Graph()
        let weights = TensorParameter(graph: g, shape: [100], name: "weights")

        // Xavier init should produce values centered around 0 with reasonable variance
        let mean = weights.data.reduce(0, +) / Float(weights.size)
        let variance = weights.data.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Float(weights.size)

        XCTAssertEqual(mean, 0.0, accuracy: 0.2, "Mean should be close to 0")
        XCTAssertGreaterThan(variance, 0.0, "Variance should be positive")
        XCTAssertLessThan(variance, 1.0, "Variance should be bounded")
    }

    // MARK: - Milestone 3: Sum Backward

    func testSumBackwardBroadcast() throws {
        let g = Graph()
        let tensorParam = TensorParameter(
            graph: g, shape: [4],
            data: [1.0, 2.0, 3.0, 4.0], name: "x")

        // sum([1,2,3,4]) = 10, target = 5
        // Loss = (10-5)^2 = 25
        // dL/d(sum) = 2*(10-5) = 10
        // dL/dx[i] = 10 for all i (broadcast)
        let sum = g.n(.sum, tensorParam.node())
        let loss = g.n(.mse, sum, g.n(.constant(5.0)))
        _ = g.n(.output(0), loss)

        let frameCount = 1
        let result = try CompilationPipeline.compile(
            graph: g, backend: .metal,
            options: .init(frameCount: frameCount, backwards: true))

        let runtime = try MetalCompiledKernel(
            kernels: result.kernels,
            cellAllocations: result.cellAllocations,
            context: result.context)

        let ctx = TrainingContext(
            tensorParameters: [tensorParam],
            optimizer: SGD(lr: 0.01),
            lossNode: loss)

        ctx.initializeMemory(
            runtime: runtime,
            cellAllocations: result.cellAllocations,
            context: result.context,
            frameCount: frameCount)

        // Run forward and backward pass
        let inputBuffer = [Float](repeating: 0.0, count: frameCount)
        var outputBuffer = [Float](repeating: 0.0, count: frameCount)

        ctx.zeroGrad()

        inputBuffer.withUnsafeBufferPointer { inPtr in
            outputBuffer.withUnsafeMutableBufferPointer { outPtr in
                runtime.runWithMemory(
                    outputs: outPtr.baseAddress!,
                    inputs: inPtr.baseAddress!,
                    memory: ctx.getMemory(),
                    frameCount: frameCount)
            }
        }

        // Extract and check gradients
        let grads = ctx.extractTensorGradients()[0]
        print("Sum backward gradients: \(grads)")

        // All gradients should be equal (broadcast from sum)
        for i in 1..<grads.count {
            XCTAssertEqual(grads[0], grads[i], accuracy: 0.5, "All gradients should be equal after sum backward")
        }
    }

    // MARK: - Milestone 4: SumAxis Backward

    func testSumAxisBackward() throws {
        let g = Graph()
        // [[1,2,3], [4,5,6]] sumAxis(1) -> [6, 15]
        let tensorParam = TensorParameter(
            graph: g, shape: [2, 3],
            data: [1, 2, 3, 4, 5, 6], name: "x")

        let summed = try g.sum(tensorParam.node(), axis: 1)
        let total = g.n(.sum, summed)
        let loss = g.n(.mse, total, g.n(.constant(10.0)))
        _ = g.n(.output(0), loss)

        let frameCount = 1
        let result = try CompilationPipeline.compile(
            graph: g, backend: .metal,
            options: .init(frameCount: frameCount, backwards: true))

        let runtime = try MetalCompiledKernel(
            kernels: result.kernels,
            cellAllocations: result.cellAllocations,
            context: result.context)

        let ctx = TrainingContext(
            tensorParameters: [tensorParam],
            optimizer: SGD(lr: 0.01),
            lossNode: loss)

        ctx.initializeMemory(
            runtime: runtime,
            cellAllocations: result.cellAllocations,
            context: result.context,
            frameCount: frameCount)

        // Run forward and backward pass
        let inputBuffer = [Float](repeating: 0.0, count: frameCount)
        var outputBuffer = [Float](repeating: 0.0, count: frameCount)

        ctx.zeroGrad()

        inputBuffer.withUnsafeBufferPointer { inPtr in
            outputBuffer.withUnsafeMutableBufferPointer { outPtr in
                runtime.runWithMemory(
                    outputs: outPtr.baseAddress!,
                    inputs: inPtr.baseAddress!,
                    memory: ctx.getMemory(),
                    frameCount: frameCount)
            }
        }

        let grads = ctx.extractTensorGradients()[0]
        print("SumAxis backward gradients: \(grads)")

        // Row 0 elements should all have same gradient (from sumAxis broadcast)
        // Row 1 elements should all have same gradient
        XCTAssertEqual(grads[0], grads[1], accuracy: 0.1)
        XCTAssertEqual(grads[1], grads[2], accuracy: 0.1)
        XCTAssertEqual(grads[3], grads[4], accuracy: 0.1)
        XCTAssertEqual(grads[4], grads[5], accuracy: 0.1)
    }

    // MARK: - Milestone 5: View Ops Backward

    func testReshapeBackward() throws {
        let g = Graph()
        let tensorParam = TensorParameter(
            graph: g, shape: [2, 3], data: [1, 2, 3, 4, 5, 6], name: "x")
        let reshaped = try g.reshape(tensorParam.node(), to: [3, 2])
        let sum = g.n(.sum, reshaped)
        let loss = g.n(.mse, sum, g.n(.constant(0.0)))
        _ = g.n(.output(0), loss)

        let frameCount = 1
        let result = try CompilationPipeline.compile(
            graph: g, backend: .metal,
            options: .init(frameCount: frameCount, backwards: true))

        let runtime = try MetalCompiledKernel(
            kernels: result.kernels,
            cellAllocations: result.cellAllocations,
            context: result.context)

        let ctx = TrainingContext(
            tensorParameters: [tensorParam],
            optimizer: SGD(lr: 0.01),
            lossNode: loss)

        ctx.initializeMemory(
            runtime: runtime,
            cellAllocations: result.cellAllocations,
            context: result.context,
            frameCount: frameCount)

        ctx.zeroGrad()

        let inputBuffer = [Float](repeating: 0.0, count: frameCount)
        var outputBuffer = [Float](repeating: 0.0, count: frameCount)

        inputBuffer.withUnsafeBufferPointer { inPtr in
            outputBuffer.withUnsafeMutableBufferPointer { outPtr in
                runtime.runWithMemory(
                    outputs: outPtr.baseAddress!,
                    inputs: inPtr.baseAddress!,
                    memory: ctx.getMemory(),
                    frameCount: frameCount)
            }
        }

        let grads = ctx.extractTensorGradients()[0]
        print("Reshape backward gradients: \(grads)")

        // All gradients should be equal (reshape is 1:1 mapping, sum broadcasts)
        for i in 1..<grads.count {
            XCTAssertEqual(grads[0], grads[i], accuracy: 0.5)
        }
    }

    // MARK: - Milestone 7: Full Training Loop

    func testTensorTensorMulBackward() throws {
        let g = Graph()

        // Test: weights * input where both are tensors
        let weights = TensorParameter(
            graph: g, shape: [4],
            data: [1.0, 2.0, 3.0, 4.0], name: "weights")
        let input = g.tensor(shape: [4], data: [1.0, 1.0, 1.0, 1.0])

        // output = sum(weights * input) = sum(weights) when input is all 1s
        let product = g.n(.mul, weights.node(), input)
        let output = g.n(.sum, product)

        // Target: 20 (sum of weights * 1 = 1+2+3+4 = 10, want 20 so weights should double)
        let loss = g.n(.mse, output, g.n(.constant(20.0)))
        _ = g.n(.output(0), loss)

        let frameCount = 1
        let result = try CompilationPipeline.compile(
            graph: g, backend: .metal,
            options: .init(frameCount: frameCount, backwards: true))

        let runtime = try MetalCompiledKernel(
            kernels: result.kernels,
            cellAllocations: result.cellAllocations,
            context: result.context)

        let ctx = TrainingContext(
            tensorParameters: [weights],
            optimizer: Adam(lr: 0.1),
            lossNode: loss)

        ctx.initializeMemory(
            runtime: runtime,
            cellAllocations: result.cellAllocations,
            context: result.context,
            frameCount: frameCount,
            graph: g)

        let inputBuffer = [Float](repeating: 0.0, count: frameCount)
        var outputBuffer = [Float](repeating: 0.0, count: frameCount)

        var losses: [Float] = []
        for epoch in 0..<200 {
            ctx.zeroGrad()

            inputBuffer.withUnsafeBufferPointer { inPtr in
                outputBuffer.withUnsafeMutableBufferPointer { outPtr in
                    runtime.runWithMemory(
                        outputs: outPtr.baseAddress!,
                        inputs: inPtr.baseAddress!,
                        memory: ctx.getMemory(),
                        frameCount: frameCount)
                }
            }

            let currentLoss = outputBuffer[0]
            losses.append(currentLoss)

            ctx.step()
        }

        // Verify convergence: sum(weights) should approach 20
        let finalSum = weights.data.reduce(0, +)
        print("Final sum: \(finalSum) (target: 20)")
        print("Final weights: \(weights.data)")

        XCTAssertEqual(finalSum, 20.0, accuracy: 1.0, "Weights should sum to approximately 20")
        XCTAssertLessThan(losses.last ?? Float.infinity, 1.0, "Loss should be small")
    }

    func testLearnTensorWeights() throws {
        let g = Graph()

        // Learn weights to compute sum = 10
        // Simple test: minimize (sum(weights) - 10)^2
        let weights = TensorParameter(
            graph: g, shape: [4],
            data: [0.1, 0.1, 0.1, 0.1], name: "weights")

        // output = sum(weights)
        let output = g.n(.sum, weights.node())

        // Target: 10 (achieved when weights sum to 10)
        // E.g., weights = [2.5, 2.5, 2.5, 2.5] gives sum = 10
        let loss = g.n(.mse, output, g.n(.constant(10.0)))
        _ = g.n(.output(0), loss)

        let frameCount = 1
        let result = try CompilationPipeline.compile(
            graph: g, backend: .metal,
            options: .init(frameCount: frameCount, backwards: true))

        let runtime = try MetalCompiledKernel(
            kernels: result.kernels,
            cellAllocations: result.cellAllocations,
            context: result.context)

        let ctx = TrainingContext(
            tensorParameters: [weights],
            optimizer: Adam(lr: 0.1),
            lossNode: loss)

        ctx.initializeMemory(
            runtime: runtime,
            cellAllocations: result.cellAllocations,
            context: result.context,
            frameCount: frameCount)

        // Train
        var losses: [Float] = []
        let inputBuffer = [Float](repeating: 0.0, count: frameCount)
        var outputBuffer = [Float](repeating: 0.0, count: frameCount)

        for epoch in 0..<500 {
            ctx.zeroGrad()

            inputBuffer.withUnsafeBufferPointer { inPtr in
                outputBuffer.withUnsafeMutableBufferPointer { outPtr in
                    runtime.runWithMemory(
                        outputs: outPtr.baseAddress!,
                        inputs: inPtr.baseAddress!,
                        memory: ctx.getMemory(),
                        frameCount: frameCount)
                }
            }

            let currentLoss = outputBuffer[0]
            losses.append(currentLoss)

            ctx.step()

            if epoch % 100 == 0 {
                print("Epoch \(epoch): loss = \(currentLoss), weights = \(weights.data)")
            }
        }

        // Verify convergence
        let finalOutput = weights.data.reduce(0, +)
        print("Final output: \(finalOutput) (target: 10.0)")
        print("Final weights: \(weights.data)")
        print("Final loss: \(losses.last ?? -1)")

        XCTAssertEqual(finalOutput, 10.0, accuracy: 1.0, "Weights should sum to approximately 10")
        XCTAssertLessThan(losses.last ?? Float.infinity, 0.1, "Loss should be small")
    }

    func testSimpleSumParameterLearning() throws {
        // Simplest possible test: learn a single tensor element to minimize sum^2
        let g = Graph()

        let weights = TensorParameter(
            graph: g, shape: [2],
            data: [1.0, 1.0], name: "weights")

        // sum([w1, w2]) = w1 + w2, target = 0
        // Loss = (w1 + w2)^2
        // dL/dw1 = 2*(w1 + w2), dL/dw2 = 2*(w1 + w2)
        let sum = g.n(.sum, weights.node())
        let loss = g.n(.mse, sum, g.n(.constant(0.0)))
        _ = g.n(.output(0), loss)

        let frameCount = 1
        let result = try CompilationPipeline.compile(
            graph: g, backend: .metal,
            options: .init(frameCount: frameCount, backwards: true))

        print("=== METAL KERNELS ===")
        for kernel in result.kernels {
            print("--- Kernel: \(kernel.name) ---")
            print(kernel.source.prefix(2000))
        }
        print("=== END KERNELS ===")

        let runtime = try MetalCompiledKernel(
            kernels: result.kernels,
            cellAllocations: result.cellAllocations,
            context: result.context)

        let ctx = TrainingContext(
            tensorParameters: [weights],
            optimizer: SGD(lr: 0.1),
            lossNode: loss)

        ctx.initializeMemory(
            runtime: runtime,
            cellAllocations: result.cellAllocations,
            context: result.context,
            frameCount: frameCount)

        let inputBuffer = [Float](repeating: 0.0, count: frameCount)
        var outputBuffer = [Float](repeating: 0.0, count: frameCount)

        for epoch in 0..<50 {
            ctx.zeroGrad()

            inputBuffer.withUnsafeBufferPointer { inPtr in
                outputBuffer.withUnsafeMutableBufferPointer { outPtr in
                    runtime.runWithMemory(
                        outputs: outPtr.baseAddress!,
                        inputs: inPtr.baseAddress!,
                        memory: ctx.getMemory(),
                        frameCount: frameCount)
                }
            }

            let currentLoss = outputBuffer[0]

            if epoch < 5 || epoch % 10 == 0 {
                let grads = ctx.extractTensorGradients()[0]
                print("Epoch \(epoch): loss = \(currentLoss), weights = \(weights.data), grads = \(grads)")
            }

            ctx.step()
        }

        // Weights should converge towards values that sum to 0
        let finalSum = weights.data.reduce(0, +)
        print("Final sum: \(finalSum), weights: \(weights.data)")
        XCTAssertEqual(finalSum, 0.0, accuracy: 0.5, "Weights should sum close to 0")
    }

    // MARK: - Conv1d Backward Tests

    func testConv1dKernelBackward() throws {
        let g = Graph()

        // Input signal: [1, 2, 3, 4, 5]
        let input = g.tensor(shape: [5], data: [1.0, 2.0, 3.0, 4.0, 5.0])

        // Learnable kernel (size 3)
        // Target: sum(conv_output) = 30
        let kernel = TensorParameter(
            graph: g, shape: [3],
            data: [0.5, 0.5, 0.5], name: "kernel")

        let convResult = g.n(.conv1d(3), input, kernel.node())
        let output = g.n(.sum, convResult)
        let loss = g.n(.mse, output, g.n(.constant(30.0)))
        _ = g.n(.output(0), loss)

        let result = try runTrainingLoop(
            graph: g,
            parameters: [kernel],
            lossNode: loss,
            optimizer: Adam(lr: 0.05),
            epochs: 300)

        print("Conv1d initial loss: \(result.initialLoss)")
        print("Conv1d initial kernel grads: \(result.gradients[0])")
        assertNonZeroGradients(result.gradients, names: ["kernel"])

        print("Conv1d final loss: \(result.finalLoss)")
        print("Conv1d final kernel: \(kernel.data)")
        XCTAssertLessThan(result.finalLoss, result.initialLoss * 0.1, "Loss should decrease significantly")
    }

    func testConv2dKernelBackward() throws {
        let g = Graph()

        // Input: 3x3 grid of ones
        let input = g.ones(shape: [3, 3])

        // Learnable 3x3 kernel
        // Target: sum(conv_output) = 18
        let kernel = TensorParameter(
            graph: g, shape: [3, 3],
            data: [0.1, 0.1, 0.1,
                   0.1, 0.1, 0.1,
                   0.1, 0.1, 0.1], name: "kernel")

        let convResult = g.n(.conv2d([3, 3]), input, kernel.node())
        let output = g.n(.sum, convResult)
        let loss = g.n(.mse, output, g.n(.constant(18.0)))
        _ = g.n(.output(0), loss)

        let result = try runTrainingLoop(
            graph: g,
            parameters: [kernel],
            lossNode: loss,
            optimizer: Adam(lr: 0.05),
            epochs: 300)

        print("Conv2d initial loss: \(result.initialLoss)")
        print("Conv2d initial kernel grads: \(result.gradients[0])")
        assertNonZeroGradients(result.gradients, names: ["kernel"])

        print("Conv2d final loss: \(result.finalLoss)")
        print("Conv2d final kernel: \(kernel.data)")
        XCTAssertLessThan(result.finalLoss, result.initialLoss * 0.1, "Loss should decrease significantly")
    }

    // MARK: - Binary Op Backward Tests (using emitBinaryOpBackward)

    func testTensorTensorAddBackward() throws {
        let g = Graph()

        // Test: a + b where both are learnable tensors
        // output = sum(a + b) = sum(a) + sum(b) = 10 + 2 = 12, target: 20
        let a = TensorParameter(graph: g, shape: [4], data: [1.0, 2.0, 3.0, 4.0], name: "a")
        let b = TensorParameter(graph: g, shape: [4], data: [0.5, 0.5, 0.5, 0.5], name: "b")

        let added = g.n(.add, a.node(), b.node())
        let output = g.n(.sum, added)
        let loss = g.n(.mse, output, g.n(.constant(20.0)))
        _ = g.n(.output(0), loss)

        let result = try runTrainingLoop(graph: g, parameters: [a, b], lossNode: loss)

        print("Add backward - a grads: \(result.gradients[0]), b grads: \(result.gradients[1])")
        assertNonZeroGradients(result.gradients, names: ["a", "b"])

        let finalSum = a.data.reduce(0, +) + b.data.reduce(0, +)
        print("Add backward - final sum: \(finalSum) (target: 20)")
        XCTAssertEqual(finalSum, 20.0, accuracy: 1.0)
    }

    func testTensorTensorSubBackward() throws {
        let g = Graph()

        // Test: a - b where both are learnable tensors
        // output = sum(a - b) = 20 - 4 = 16, target: 10
        let a = TensorParameter(graph: g, shape: [4], data: [5.0, 5.0, 5.0, 5.0], name: "a")
        let b = TensorParameter(graph: g, shape: [4], data: [1.0, 1.0, 1.0, 1.0], name: "b")

        let subbed = g.n(.sub, a.node(), b.node())
        let output = g.n(.sum, subbed)
        let loss = g.n(.mse, output, g.n(.constant(10.0)))
        _ = g.n(.output(0), loss)

        let result = try runTrainingLoop(graph: g, parameters: [a, b], lossNode: loss)

        print("Sub backward - a grads: \(result.gradients[0]), b grads: \(result.gradients[1])")
        assertNonZeroGradients(result.gradients, names: ["a", "b"])

        let finalDiff = a.data.reduce(0, +) - b.data.reduce(0, +)
        print("Sub backward - final diff: \(finalDiff) (target: 10)")
        XCTAssertEqual(finalDiff, 10.0, accuracy: 1.0)
    }

    func testTensorTensorDivBackward() throws {
        let g = Graph()

        // Test: a / b where both are learnable tensors
        // output = sum(a / b) = sum([2, 4, 6, 8]) = 20, target: 10
        let a = TensorParameter(graph: g, shape: [4], data: [4.0, 8.0, 12.0, 16.0], name: "a")
        let b = TensorParameter(graph: g, shape: [4], data: [2.0, 2.0, 2.0, 2.0], name: "b")

        let divided = g.n(.div, a.node(), b.node())
        let output = g.n(.sum, divided)
        let loss = g.n(.mse, output, g.n(.constant(10.0)))
        _ = g.n(.output(0), loss)

        let result = try runTrainingLoop(
            graph: g,
            parameters: [a, b],
            lossNode: loss,
            optimizer: Adam(lr: 0.05),
            epochs: 300)

        print("Div backward - a grads: \(result.gradients[0]), b grads: \(result.gradients[1])")
        assertNonZeroGradients(result.gradients, names: ["a", "b"])

        print("Div backward - initial loss: \(result.initialLoss), final loss: \(result.finalLoss)")
        XCTAssertLessThan(result.finalLoss, result.initialLoss * 0.1, "Loss should decrease significantly")
    }

    // MARK: - Unary Op Backward Tests (using emitUnaryOpBackward)

    func testTensorSinBackward() throws {
        let g = Graph()

        // sin(x) has derivative cos(x)
        // Target: maximize sum(sin(x)) = 4 -> x should converge to pi/2
        let x = TensorParameter(graph: g, shape: [4], data: [0.0, 0.5, 1.0, 1.5], name: "x")

        let sinResult = g.n(.sin, x.node())
        let output = g.n(.sum, sinResult)
        let loss = g.n(.mse, output, g.n(.constant(4.0)))
        _ = g.n(.output(0), loss)

        let result = try runTrainingLoop(graph: g, parameters: [x], lossNode: loss)

        print("Sin backward - grads: \(result.gradients[0])")
        assertNonZeroGradients(result.gradients, names: ["x"])

        print("Sin backward - initial loss: \(result.initialLoss), final loss: \(result.finalLoss)")
        print("Sin backward - final x: \(x.data)")
        XCTAssertLessThan(result.finalLoss, result.initialLoss * 0.5, "Loss should decrease")
    }

    func testTensorCosBackward() throws {
        let g = Graph()

        // Target: sum(cos(x)) = 4 -> x should converge to 0
        let x = TensorParameter(graph: g, shape: [4], data: [1.0, 1.0, 1.0, 1.0], name: "x")

        let cosResult = g.n(.cos, x.node())
        let output = g.n(.sum, cosResult)
        let loss = g.n(.mse, output, g.n(.constant(4.0)))
        _ = g.n(.output(0), loss)

        let result = try runTrainingLoop(graph: g, parameters: [x], lossNode: loss)

        print("Cos backward - grads: \(result.gradients[0])")
        assertNonZeroGradients(result.gradients, names: ["x"])

        print("Cos backward - initial loss: \(result.initialLoss), final loss: \(result.finalLoss)")
        XCTAssertLessThan(result.finalLoss, result.initialLoss * 0.5, "Loss should decrease")
    }

    func testTensorTanhBackward() throws {
        let g = Graph()

        // Target: sum(tanh(x)) = 2 -> x should become positive
        let x = TensorParameter(graph: g, shape: [4], data: [0.0, 0.0, 0.0, 0.0], name: "x")

        let tanhResult = g.n(.tanh, x.node())
        let output = g.n(.sum, tanhResult)
        let loss = g.n(.mse, output, g.n(.constant(2.0)))
        _ = g.n(.output(0), loss)

        let result = try runTrainingLoop(
            graph: g,
            parameters: [x],
            lossNode: loss,
            optimizer: Adam(lr: 0.2))

        print("Tanh backward - grads: \(result.gradients[0])")
        assertNonZeroGradients(result.gradients, names: ["x"])

        print("Tanh backward - initial loss: \(result.initialLoss), final loss: \(result.finalLoss)")
        XCTAssertLessThan(result.finalLoss, result.initialLoss * 0.5, "Loss should decrease")
    }

    func testTensorExpBackward() throws {
        let g = Graph()

        // exp(0) = 1, sum = 4. Target: 8 -> x should increase
        let x = TensorParameter(graph: g, shape: [4], data: [0.0, 0.0, 0.0, 0.0], name: "x")

        let expResult = g.n(.exp, x.node())
        let output = g.n(.sum, expResult)
        let loss = g.n(.mse, output, g.n(.constant(8.0)))
        _ = g.n(.output(0), loss)

        let result = try runTrainingLoop(graph: g, parameters: [x], lossNode: loss)

        print("Exp backward - grads: \(result.gradients[0])")
        assertNonZeroGradients(result.gradients, names: ["x"])

        print("Exp backward - initial loss: \(result.initialLoss), final loss: \(result.finalLoss)")
        XCTAssertLessThan(result.finalLoss, result.initialLoss * 0.1, "Loss should decrease significantly")
    }

    func testTensorLogBackward() throws {
        let g = Graph()

        // log(1) = 0, sum = 0. Target: 4 -> x should increase to e
        let x = TensorParameter(graph: g, shape: [4], data: [1.0, 1.0, 1.0, 1.0], name: "x")

        let logResult = g.n(.log, x.node())
        let output = g.n(.sum, logResult)
        let loss = g.n(.mse, output, g.n(.constant(4.0)))
        _ = g.n(.output(0), loss)

        let result = try runTrainingLoop(
            graph: g,
            parameters: [x],
            lossNode: loss,
            optimizer: Adam(lr: 0.2))

        print("Log backward - grads: \(result.gradients[0])")
        assertNonZeroGradients(result.gradients, names: ["x"])

        print("Log backward - initial loss: \(result.initialLoss), final loss: \(result.finalLoss)")
        XCTAssertLessThan(result.finalLoss, result.initialLoss * 0.5, "Loss should decrease")
    }

    func testTensorSqrtBackward() throws {
        let g = Graph()

        // sqrt(1) = 1, sum = 4. Target: 8 -> x should increase to 4
        let x = TensorParameter(graph: g, shape: [4], data: [1.0, 1.0, 1.0, 1.0], name: "x")

        let sqrtResult = g.n(.sqrt, x.node())
        let output = g.n(.sum, sqrtResult)
        let loss = g.n(.mse, output, g.n(.constant(8.0)))
        _ = g.n(.output(0), loss)

        let result = try runTrainingLoop(
            graph: g,
            parameters: [x],
            lossNode: loss,
            optimizer: Adam(lr: 0.5))

        print("Sqrt backward - grads: \(result.gradients[0])")
        assertNonZeroGradients(result.gradients, names: ["x"])

        print("Sqrt backward - initial loss: \(result.initialLoss), final loss: \(result.finalLoss)")
        XCTAssertLessThan(result.finalLoss, result.initialLoss * 0.1, "Loss should decrease significantly")
    }

    func testTensorAbsBackward() throws {
        let g = Graph()

        // abs([-2,-1,1,2]) = [2,1,1,2], sum = 6. Target: 2 -> reduce magnitudes
        // abs has gradient sign(x)
        let x = TensorParameter(graph: g, shape: [4], data: [-2.0, -1.0, 1.0, 2.0], name: "x")

        let absResult = g.n(.abs, x.node())
        let output = g.n(.sum, absResult)
        let loss = g.n(.mse, output, g.n(.constant(2.0)))
        _ = g.n(.output(0), loss)

        let result = try runTrainingLoop(
            graph: g,
            parameters: [x],
            lossNode: loss,
            optimizer: Adam(lr: 0.2))

        print("Abs backward - grads: \(result.gradients[0])")
        assertNonZeroGradients(result.gradients, names: ["x"])

        print("Abs backward - initial loss: \(result.initialLoss), final loss: \(result.finalLoss)")
        XCTAssertLessThan(result.finalLoss, result.initialLoss * 0.5, "Loss should decrease")
    }
}
