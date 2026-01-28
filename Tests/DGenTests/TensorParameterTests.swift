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
}
