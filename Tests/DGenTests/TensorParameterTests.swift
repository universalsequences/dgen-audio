import XCTest

@testable import DGen

final class TensorParameterTests: XCTestCase {

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

        // Learnable kernel (size 3) - start with identity-ish kernel
        let kernel = TensorParameter(
            graph: g, shape: [3],
            data: [0.5, 0.5, 0.5], name: "kernel")

        // Convolve: output[i] = sum_k(input[i+k-1] * kernel[k])
        // With kernel [0.5, 0.5, 0.5], output ≈ [1.5, 3, 4.5, 6, 7.5] (with padding)
        let convResult = g.n(.conv1d(3), input, kernel.node())
        let output = g.n(.sum, convResult)

        // Target: we want sum(conv_output) = 30
        // With identity kernel [0, 1, 0], output = input, sum = 15
        // We want kernel that doubles: [0, 2, 0] would give sum = 30
        let target = g.n(.constant(30.0))
        let loss = g.n(.mse, output, target)
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
            tensorParameters: [kernel],
            optimizer: Adam(lr: 0.05),
            lossNode: loss)

        ctx.initializeMemory(
            runtime: runtime,
            cellAllocations: result.cellAllocations,
            context: result.context,
            frameCount: frameCount,
            graph: g)

        let inputBuffer = [Float](repeating: 0.0, count: frameCount)
        var outputBuffer = [Float](repeating: 0.0, count: frameCount)

        // Run one step to get initial gradients
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
        let initialGrads = ctx.extractTensorGradients()[0]
        print("Conv1d initial loss: \(initialLoss)")
        print("Conv1d initial kernel grads: \(initialGrads)")

        // Verify gradients are non-zero
        let hasNonZeroGrad = initialGrads.contains { abs($0) > 0.001 }
        XCTAssertTrue(hasNonZeroGrad, "Kernel gradients should be non-zero")

        // Train for a while
        var losses: [Float] = [initialLoss]
        for _ in 0..<300 {
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

            losses.append(outputBuffer[0])
        }

        let finalLoss = losses.last!
        print("Conv1d final loss: \(finalLoss)")
        print("Conv1d final kernel: \(kernel.data)")

        // Verify loss decreased significantly
        XCTAssertLessThan(finalLoss, initialLoss * 0.1, "Loss should decrease significantly")
    }

    func testConv2dKernelBackward() throws {
        let g = Graph()

        // Input: 3x3 grid of ones
        let input = g.ones(shape: [3, 3])

        // Learnable 3x3 kernel - start with small values
        let kernel = TensorParameter(
            graph: g, shape: [3, 3],
            data: [0.1, 0.1, 0.1,
                   0.1, 0.1, 0.1,
                   0.1, 0.1, 0.1], name: "kernel")

        // Convolve
        let convResult = g.n(.conv2d([3, 3]), input, kernel.node())
        let output = g.n(.sum, convResult)

        // Target: we want sum(conv_output) = 18
        // With a center-only kernel [0,0,0, 0,2,0, 0,0,0], sum would = 18 for 3x3 input of ones
        let target = g.n(.constant(18.0))
        let loss = g.n(.mse, output, target)
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
            tensorParameters: [kernel],
            optimizer: Adam(lr: 0.05),
            lossNode: loss)

        ctx.initializeMemory(
            runtime: runtime,
            cellAllocations: result.cellAllocations,
            context: result.context,
            frameCount: frameCount,
            graph: g)

        let inputBuffer = [Float](repeating: 0.0, count: frameCount)
        var outputBuffer = [Float](repeating: 0.0, count: frameCount)

        // Run one step to get initial gradients
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
        let initialGrads = ctx.extractTensorGradients()[0]
        print("Conv2d initial loss: \(initialLoss)")
        print("Conv2d initial kernel grads: \(initialGrads)")

        // Verify gradients are non-zero
        let hasNonZeroGrad = initialGrads.contains { abs($0) > 0.001 }
        XCTAssertTrue(hasNonZeroGrad, "Kernel gradients should be non-zero")

        // Train for a while
        var losses: [Float] = [initialLoss]
        for _ in 0..<300 {
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

            losses.append(outputBuffer[0])
        }

        let finalLoss = losses.last!
        print("Conv2d final loss: \(finalLoss)")
        print("Conv2d final kernel: \(kernel.data)")

        // Verify loss decreased significantly
        XCTAssertLessThan(finalLoss, initialLoss * 0.1, "Loss should decrease significantly")
    }

    // MARK: - Binary Op Backward Tests (using emitBinaryOpBackward)

    func testTensorTensorAddBackward() throws {
        let g = Graph()

        // Test: a + b where both are learnable tensors
        let a = TensorParameter(graph: g, shape: [4], data: [1.0, 2.0, 3.0, 4.0], name: "a")
        let b = TensorParameter(graph: g, shape: [4], data: [0.5, 0.5, 0.5, 0.5], name: "b")

        // output = sum(a + b) = sum(a) + sum(b) = 10 + 2 = 12
        // Target: 20, so we need to increase both a and b
        let added = g.n(.add, a.node(), b.node())
        let output = g.n(.sum, added)
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
            tensorParameters: [a, b],
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

        // Get initial gradients
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

        let grads = ctx.extractTensorGradients()
        print("Add backward - a grads: \(grads[0]), b grads: \(grads[1])")

        // For add: dL/da = dL/d(sum) = gradOutput, dL/db = gradOutput
        // Both should have the same gradient (broadcast from sum)
        XCTAssertTrue(grads[0].contains { abs($0) > 0.001 }, "a gradients should be non-zero")
        XCTAssertTrue(grads[1].contains { abs($0) > 0.001 }, "b gradients should be non-zero")

        // Train and verify convergence
        for _ in 0..<200 {
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

        let finalSum = a.data.reduce(0, +) + b.data.reduce(0, +)
        print("Add backward - final sum: \(finalSum) (target: 20)")
        XCTAssertEqual(finalSum, 20.0, accuracy: 1.0)
    }

    func testTensorTensorSubBackward() throws {
        let g = Graph()

        // Test: a - b where both are learnable tensors
        let a = TensorParameter(graph: g, shape: [4], data: [5.0, 5.0, 5.0, 5.0], name: "a")
        let b = TensorParameter(graph: g, shape: [4], data: [1.0, 1.0, 1.0, 1.0], name: "b")

        // output = sum(a - b) = 20 - 4 = 16
        // Target: 10, so we need to decrease a or increase b
        let subbed = g.n(.sub, a.node(), b.node())
        let output = g.n(.sum, subbed)
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
            tensorParameters: [a, b],
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

        let grads = ctx.extractTensorGradients()
        print("Sub backward - a grads: \(grads[0]), b grads: \(grads[1])")

        // For sub: dL/da = gradOutput, dL/db = -gradOutput
        XCTAssertTrue(grads[0].contains { abs($0) > 0.001 }, "a gradients should be non-zero")
        XCTAssertTrue(grads[1].contains { abs($0) > 0.001 }, "b gradients should be non-zero")

        // Train and verify convergence
        for _ in 0..<200 {
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

        let finalDiff = a.data.reduce(0, +) - b.data.reduce(0, +)
        print("Sub backward - final diff: \(finalDiff) (target: 10)")
        XCTAssertEqual(finalDiff, 10.0, accuracy: 1.0)
    }

    func testTensorTensorDivBackward() throws {
        let g = Graph()

        // Test: a / b where both are learnable tensors
        let a = TensorParameter(graph: g, shape: [4], data: [4.0, 8.0, 12.0, 16.0], name: "a")
        let b = TensorParameter(graph: g, shape: [4], data: [2.0, 2.0, 2.0, 2.0], name: "b")

        // output = sum(a / b) = sum([2, 4, 6, 8]) = 20
        // Target: 10
        let divided = g.n(.div, a.node(), b.node())
        let output = g.n(.sum, divided)
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
            tensorParameters: [a, b],
            optimizer: Adam(lr: 0.05),
            lossNode: loss)

        ctx.initializeMemory(
            runtime: runtime,
            cellAllocations: result.cellAllocations,
            context: result.context,
            frameCount: frameCount,
            graph: g)

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

        let grads = ctx.extractTensorGradients()
        print("Div backward - a grads: \(grads[0]), b grads: \(grads[1])")

        // For div: dL/da = gradOutput / b, dL/db = -gradOutput * a / b^2
        XCTAssertTrue(grads[0].contains { abs($0) > 0.001 }, "a gradients should be non-zero")
        XCTAssertTrue(grads[1].contains { abs($0) > 0.001 }, "b gradients should be non-zero")

        // Train and verify convergence
        let initialLoss = outputBuffer[0]
        for _ in 0..<300 {
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

        let finalLoss = outputBuffer[0]
        print("Div backward - initial loss: \(initialLoss), final loss: \(finalLoss)")
        XCTAssertLessThan(finalLoss, initialLoss * 0.1, "Loss should decrease significantly")
    }

    // MARK: - Unary Op Backward Tests (using emitUnaryOpBackward)

    func testTensorSinBackward() throws {
        let g = Graph()

        // Test: sin(x) where x is learnable
        // sin has derivative cos(x)
        let x = TensorParameter(graph: g, shape: [4], data: [0.0, 0.5, 1.0, 1.5], name: "x")

        let sinResult = g.n(.sin, x.node())
        let output = g.n(.sum, sinResult)
        // Target: maximize sum(sin(x)) -> x should converge to π/2
        let loss = g.n(.mse, output, g.n(.constant(4.0)))  // max sum(sin) for 4 elements = 4
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
            tensorParameters: [x],
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
        print("Sin backward - grads: \(grads)")
        XCTAssertTrue(grads.contains { abs($0) > 0.001 }, "sin gradients should be non-zero")

        // Train
        let initialLoss = outputBuffer[0]
        for _ in 0..<200 {
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

        let finalLoss = outputBuffer[0]
        print("Sin backward - initial loss: \(initialLoss), final loss: \(finalLoss)")
        print("Sin backward - final x: \(x.data)")
        XCTAssertLessThan(finalLoss, initialLoss * 0.5, "Loss should decrease")
    }

    func testTensorCosBackward() throws {
        let g = Graph()

        let x = TensorParameter(graph: g, shape: [4], data: [1.0, 1.0, 1.0, 1.0], name: "x")

        let cosResult = g.n(.cos, x.node())
        let output = g.n(.sum, cosResult)
        // Target: sum(cos(x)) = 4 -> x should be 0
        let loss = g.n(.mse, output, g.n(.constant(4.0)))
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
            tensorParameters: [x],
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
        print("Cos backward - grads: \(grads)")
        XCTAssertTrue(grads.contains { abs($0) > 0.001 }, "cos gradients should be non-zero")

        let initialLoss = outputBuffer[0]
        for _ in 0..<200 {
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

        let finalLoss = outputBuffer[0]
        print("Cos backward - initial loss: \(initialLoss), final loss: \(finalLoss)")
        XCTAssertLessThan(finalLoss, initialLoss * 0.5, "Loss should decrease")
    }

    func testTensorTanhBackward() throws {
        let g = Graph()

        let x = TensorParameter(graph: g, shape: [4], data: [0.0, 0.0, 0.0, 0.0], name: "x")

        let tanhResult = g.n(.tanh, x.node())
        let output = g.n(.sum, tanhResult)
        // Target: sum(tanh(x)) = 2 -> x should be positive
        let loss = g.n(.mse, output, g.n(.constant(2.0)))
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
            tensorParameters: [x],
            optimizer: Adam(lr: 0.2),
            lossNode: loss)

        ctx.initializeMemory(
            runtime: runtime,
            cellAllocations: result.cellAllocations,
            context: result.context,
            frameCount: frameCount,
            graph: g)

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
        print("Tanh backward - grads: \(grads)")
        XCTAssertTrue(grads.contains { abs($0) > 0.001 }, "tanh gradients should be non-zero")

        let initialLoss = outputBuffer[0]
        for _ in 0..<200 {
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

        let finalLoss = outputBuffer[0]
        print("Tanh backward - initial loss: \(initialLoss), final loss: \(finalLoss)")
        XCTAssertLessThan(finalLoss, initialLoss * 0.5, "Loss should decrease")
    }

    func testTensorExpBackward() throws {
        let g = Graph()

        let x = TensorParameter(graph: g, shape: [4], data: [0.0, 0.0, 0.0, 0.0], name: "x")

        let expResult = g.n(.exp, x.node())
        let output = g.n(.sum, expResult)
        // exp(0) = 1, so sum = 4. Target: 8 -> x should increase
        let loss = g.n(.mse, output, g.n(.constant(8.0)))
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
            tensorParameters: [x],
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
        print("Exp backward - grads: \(grads)")
        XCTAssertTrue(grads.contains { abs($0) > 0.001 }, "exp gradients should be non-zero")

        let initialLoss = outputBuffer[0]
        for _ in 0..<200 {
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

        let finalLoss = outputBuffer[0]
        print("Exp backward - initial loss: \(initialLoss), final loss: \(finalLoss)")
        XCTAssertLessThan(finalLoss, initialLoss * 0.1, "Loss should decrease significantly")
    }

    func testTensorLogBackward() throws {
        let g = Graph()

        // Start with positive values for log
        let x = TensorParameter(graph: g, shape: [4], data: [1.0, 1.0, 1.0, 1.0], name: "x")

        let logResult = g.n(.log, x.node())
        let output = g.n(.sum, logResult)
        // log(1) = 0, so sum = 0. Target: 4 -> x should increase to e
        let loss = g.n(.mse, output, g.n(.constant(4.0)))
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
            tensorParameters: [x],
            optimizer: Adam(lr: 0.2),
            lossNode: loss)

        ctx.initializeMemory(
            runtime: runtime,
            cellAllocations: result.cellAllocations,
            context: result.context,
            frameCount: frameCount,
            graph: g)

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
        print("Log backward - grads: \(grads)")
        XCTAssertTrue(grads.contains { abs($0) > 0.001 }, "log gradients should be non-zero")

        let initialLoss = outputBuffer[0]
        for _ in 0..<200 {
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

        let finalLoss = outputBuffer[0]
        print("Log backward - initial loss: \(initialLoss), final loss: \(finalLoss)")
        XCTAssertLessThan(finalLoss, initialLoss * 0.5, "Loss should decrease")
    }

    func testTensorSqrtBackward() throws {
        let g = Graph()

        // Start with positive values for sqrt
        let x = TensorParameter(graph: g, shape: [4], data: [1.0, 1.0, 1.0, 1.0], name: "x")

        let sqrtResult = g.n(.sqrt, x.node())
        let output = g.n(.sum, sqrtResult)
        // sqrt(1) = 1, so sum = 4. Target: 8 -> x should increase to 4
        let loss = g.n(.mse, output, g.n(.constant(8.0)))
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
            tensorParameters: [x],
            optimizer: Adam(lr: 0.5),
            lossNode: loss)

        ctx.initializeMemory(
            runtime: runtime,
            cellAllocations: result.cellAllocations,
            context: result.context,
            frameCount: frameCount,
            graph: g)

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
        print("Sqrt backward - grads: \(grads)")
        XCTAssertTrue(grads.contains { abs($0) > 0.001 }, "sqrt gradients should be non-zero")

        let initialLoss = outputBuffer[0]
        for _ in 0..<200 {
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

        let finalLoss = outputBuffer[0]
        print("Sqrt backward - initial loss: \(initialLoss), final loss: \(finalLoss)")
        XCTAssertLessThan(finalLoss, initialLoss * 0.1, "Loss should decrease significantly")
    }

    func testTensorAbsBackward() throws {
        let g = Graph()

        // abs has gradient sign(x)
        let x = TensorParameter(graph: g, shape: [4], data: [-2.0, -1.0, 1.0, 2.0], name: "x")

        let absResult = g.n(.abs, x.node())
        let output = g.n(.sum, absResult)
        // abs([-2,-1,1,2]) = [2,1,1,2], sum = 6. Target: 2 -> reduce magnitudes
        let loss = g.n(.mse, output, g.n(.constant(2.0)))
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
            tensorParameters: [x],
            optimizer: Adam(lr: 0.2),
            lossNode: loss)

        ctx.initializeMemory(
            runtime: runtime,
            cellAllocations: result.cellAllocations,
            context: result.context,
            frameCount: frameCount,
            graph: g)

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
        print("Abs backward - grads: \(grads)")
        // Gradient should be sign(x) * gradOutput
        XCTAssertTrue(grads.contains { abs($0) > 0.001 }, "abs gradients should be non-zero")

        let initialLoss = outputBuffer[0]
        for _ in 0..<200 {
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

        let finalLoss = outputBuffer[0]
        print("Abs backward - initial loss: \(initialLoss), final loss: \(finalLoss)")
        XCTAssertLessThan(finalLoss, initialLoss * 0.5, "Loss should decrease")
    }
}
