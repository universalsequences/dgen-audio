import XCTest

@testable import DGen

final class TensorOpsTests: XCTestCase {

    // MARK: - Basic Tensor Operations

    func testTensorAddScalar() throws {
        // Test: tensor + scalar broadcasting
        let g = Graph()

        // Create a 2x2 tensor
        let tensorNode = g.tensor(shape: [2, 2])
        let scalar = g.n(.constant(1.0))

        // Add scalar to tensor
        let result = g.n(.add, tensorNode, scalar)

        // Sum the result to get a scalar output
        let sumResult = g.n(.sum, result)
        _ = g.n(.output(0), sumResult)

        // Compile
        let compilationResult = try CompilationPipeline.compile(
            graph: g,
            backend: .c,
            options: .init(frameCount: 1, debug: true)
        )

        print("=== Tensor Add Scalar - Generated Source ===")
        print(compilationResult.source)

        XCTAssertFalse(compilationResult.source.isEmpty)
    }

    func testTensorMulTensor() throws {
        // Test: tensor * tensor element-wise
        let g = Graph()

        // Create two 3x3 tensors
        let t1 = g.tensor(shape: [3, 3])
        let t2 = g.tensor(shape: [3, 3])

        // Element-wise multiply
        let result = g.n(.mul, t1, t2)

        // Sum to scalar
        let sumResult = g.n(.sum, result)
        _ = g.n(.output(0), sumResult)

        // Compile
        let compilationResult = try CompilationPipeline.compile(
            graph: g,
            backend: .c,
            options: .init(frameCount: 1, debug: true)
        )

        print("=== Tensor Mul Tensor - Generated Source ===")
        print(compilationResult.source)

        XCTAssertFalse(compilationResult.source.isEmpty)
    }

    // MARK: - Conv2d Tests

    func testConv2dLaplacian() throws {
        // Test conv2d with a 3x3 Laplacian kernel on a 4x4 input
        let g = Graph()

        // 4x4 input tensor
        let input = g.tensor(shape: [4, 4])

        // 3x3 Laplacian kernel
        let kernel = g.tensor(shape: [3, 3])

        // Conv2d operation
        let convResult = g.n(.conv2d([3, 3]), input, kernel)

        // Sum the result
        let sumResult = g.n(.sum, convResult)
        _ = g.n(.output(0), sumResult)

        // Compile
        let compilationResult = try CompilationPipeline.compile(
            graph: g,
            backend: .c,
            options: .init(frameCount: 1, debug: true)
        )

        print("=== Conv2d Laplacian - Generated Source ===")
        print(compilationResult.source)

        XCTAssertFalse(compilationResult.source.isEmpty)
    }

    // MARK: - Sum Reduce Tests

    func testSumReduce() throws {
        // Test sum reduction of a tensor to scalar
        let g = Graph()

        // Create a 5x5 tensor
        let tensor = g.tensor(shape: [5, 5])

        // Sum to scalar
        let sumResult = g.n(.sum, tensor)
        _ = g.n(.output(0), sumResult)

        // Compile
        let compilationResult = try CompilationPipeline.compile(
            graph: g,
            backend: .c,
            options: .init(frameCount: 1, debug: true)
        )

        print("=== Sum Reduce - Generated Source ===")
        print(compilationResult.source)

        XCTAssertFalse(compilationResult.source.isEmpty)
    }

    // MARK: - Tensor History Tests

    func testTensorHistoryReadWrite() throws {
        // Test tensor history for state across frames
        let g = Graph()

        // Create a history buffer for 4x4 state
        let stateBuffer = g.tensorHistoryBuffer(shape: [4, 4])

        // Read previous state, add increment, write back
        let prevState = g.tensorHistoryRead(stateBuffer)
        let increment = g.n(.constant(0.1))
        let newState = g.n(.add, prevState, increment)
        g.tensorHistoryWrite(stateBuffer, newState)

        // Output the sum
        _ = g.n(.output(0), g.n(.sum, newState))

        // Compile
        let compilationResult = try CompilationPipeline.compile(
            graph: g,
            backend: .c,
            options: .init(frameCount: 4, debug: true)
        )

        print("=== Tensor History - Generated Source ===")
        print(compilationResult.source)

        XCTAssertFalse(compilationResult.source.isEmpty)
    }

    // MARK: - Membrane Simulation Test

    func testMembraneSimulationCompiles() throws {
        // Simplified membrane physical model:
        // state_t+1 = 2*state_t - state_t-1 + c^2 * laplacian(state_t)

        let g = Graph()
        let gridShape: Shape = [4, 4]

        // History buffers for state (these persist across frames, need manual cell allocation)
        let stateBuffer = g.tensorHistoryBuffer(shape: gridShape)
        let prevStateBuffer = g.tensorHistoryBuffer(shape: gridShape)

        // Read current and previous state from history
        let state_t = g.tensorHistoryRead(stateBuffer)
        let state_t_1 = g.tensorHistoryRead(prevStateBuffer)

        // Laplacian kernel - use the clean tensor API with data
        let kernel = g.tensor(shape: [3, 3], data: [
            0,  1, 0,
            1, -4, 1,
            0,  1, 0
        ])

        // Compute laplacian = conv2d(state_t, kernel)
        let laplacian = g.n(.conv2d([3, 3]), state_t, kernel)

        // Compute: state_t+1 = 2*state_t - state_t_1 + c^2 * laplacian
        let two = g.n(.constant(2.0))
        let c_squared = g.n(.constant(0.1))

        let twoState = g.n(.mul, two, state_t)
        let scaledLaplacian = g.n(.mul, c_squared, laplacian)
        let diff = g.n(.sub, twoState, state_t_1)
        let state_t_plus_1 = g.n(.add, diff, scaledLaplacian)

        // Write new state to history (shift: prev <- current, current <- new)
        g.tensorHistoryWrite(prevStateBuffer, state_t)
        g.tensorHistoryWrite(stateBuffer, state_t_plus_1)

        // Output: sum of state (to get a scalar for audio output)
        let sumOutput = g.n(.sum, state_t_plus_1)
        _ = g.n(.output(0), sumOutput)

        // Compile for C
        let cResult = try CompilationPipeline.compile(
            graph: g,
            backend: .c,
            options: .init(frameCount: 8, debug: true)
        )

        print("=== Membrane Simulation - C Source ===")
        print(cResult.source)

        XCTAssertFalse(cResult.source.isEmpty, "C source should not be empty")

        // Compile for Metal
        let mResult = try CompilationPipeline.compile(
            graph: g,
            backend: .metal,
            options: .init(frameCount: 8, debug: true)
        )

        print("=== Membrane Simulation - Metal Kernels ===")
        for kernel in mResult.kernels {
            print("--- Kernel: \(kernel.name) ---")
            print(kernel.source)
        }

        XCTAssertFalse(mResult.kernels.isEmpty, "Metal kernels should not be empty")
    }

    // MARK: - Execution Tests

    func testSumReduceExecution() throws {
        // Test that sum reduce actually computes correct values
        let g = Graph()

        // Create a 2x2 tensor with initial data - this is the clean API!
        let tensorNode = g.tensor(shape: [2, 2], data: [1.0, 2.0, 3.0, 4.0])

        // Sum to scalar
        let sumResult = g.n(.sum, tensorNode)
        _ = g.n(.output(0), sumResult)

        let frameCount = 1

        // Compile
        let cResult = try CompilationPipeline.compile(
            graph: g,
            backend: .c,
            options: .init(frameCount: frameCount, debug: false)
        )

        // Create runtime
        let cRuntime = CCompiledKernel(
            source: cResult.source,
            cellAllocations: cResult.cellAllocations,
            memorySize: cResult.totalMemorySlots
        )
        try cRuntime.compileAndLoad()

        // Allocate memory
        guard let mem = cRuntime.allocateNodeMemory() else {
            XCTFail("Failed to allocate memory")
            return
        }
        defer { cRuntime.deallocateNodeMemory(mem) }

        // Inject tensor data automatically from graph
        injectTensorData(result: cResult, memory: mem.assumingMemoryBound(to: Float.self))

        // Run
        var output = [Float](repeating: 0, count: frameCount)
        let input = [Float](repeating: 0, count: frameCount)

        output.withUnsafeMutableBufferPointer { outPtr in
            input.withUnsafeBufferPointer { inPtr in
                cRuntime.runWithMemory(
                    outputs: outPtr.baseAddress!,
                    inputs: inPtr.baseAddress!,
                    memory: mem,
                    frameCount: frameCount
                )
            }
        }

        // Verify: 1 + 2 + 3 + 4 = 10
        XCTAssertEqual(output[0], 10.0, accuracy: 1e-5, "Sum should be 10.0")
    }

    func testTensorAddScalarExecution() throws {
        // Test tensor + scalar with actual execution
        let g = Graph()

        // Create a 2x2 tensor with data
        let tensorNode = g.tensor(shape: [2, 2], data: [1.0, 2.0, 3.0, 4.0])

        // Add scalar 5.0 to each element
        let scalar = g.n(.constant(5.0))
        let result = g.n(.add, tensorNode, scalar)

        // Sum the result to get scalar output
        let sumResult = g.n(.sum, result)
        _ = g.n(.output(0), sumResult)

        let frameCount = 1

        // Compile
        let cResult = try CompilationPipeline.compile(
            graph: g,
            backend: .c,
            options: .init(frameCount: frameCount, debug: false)
        )

        // Create runtime
        let cRuntime = CCompiledKernel(
            source: cResult.source,
            cellAllocations: cResult.cellAllocations,
            memorySize: cResult.totalMemorySlots
        )
        try cRuntime.compileAndLoad()

        guard let mem = cRuntime.allocateNodeMemory() else {
            XCTFail("Failed to allocate memory")
            return
        }
        defer { cRuntime.deallocateNodeMemory(mem) }

        // Inject tensor data
        injectTensorData(result: cResult, memory: mem.assumingMemoryBound(to: Float.self))

        // Run
        var output = [Float](repeating: 0, count: frameCount)
        let input = [Float](repeating: 0, count: frameCount)

        output.withUnsafeMutableBufferPointer { outPtr in
            input.withUnsafeBufferPointer { inPtr in
                cRuntime.runWithMemory(
                    outputs: outPtr.baseAddress!,
                    inputs: inPtr.baseAddress!,
                    memory: mem,
                    frameCount: frameCount
                )
            }
        }

        // Expected: (1+5) + (2+5) + (3+5) + (4+5) = 6 + 7 + 8 + 9 = 30
        XCTAssertEqual(output[0], 30.0, accuracy: 1e-5, "Sum of (tensor + 5) should be 30.0")
    }

    func testConv2dExecution() throws {
        // Test conv2d with a simple kernel
        let g = Graph()

        // 3x3 input tensor with all 1s
        let inputNode = g.ones(shape: [3, 3])

        // 3x3 identity kernel (center = 1, rest = 0)
        let kernelNode = g.tensor(shape: [3, 3], data: [
            0, 0, 0,
            0, 1, 0,
            0, 0, 0
        ])

        // Conv2d
        let convResult = g.n(.conv2d([3, 3]), inputNode, kernelNode)

        // Sum to scalar output
        let sumResult = g.n(.sum, convResult)
        _ = g.n(.output(0), sumResult)

        let frameCount = 1

        // Compile
        let cResult = try CompilationPipeline.compile(
            graph: g,
            backend: .c,
            options: .init(frameCount: frameCount, debug: false)
        )

        // Create runtime
        let cRuntime = CCompiledKernel(
            source: cResult.source,
            cellAllocations: cResult.cellAllocations,
            memorySize: cResult.totalMemorySlots
        )
        try cRuntime.compileAndLoad()

        guard let mem = cRuntime.allocateNodeMemory() else {
            XCTFail("Failed to allocate memory")
            return
        }
        defer { cRuntime.deallocateNodeMemory(mem) }

        // Inject tensor data
        injectTensorData(result: cResult, memory: mem.assumingMemoryBound(to: Float.self))

        // Run
        var output = [Float](repeating: 0, count: frameCount)
        let input = [Float](repeating: 0, count: frameCount)

        output.withUnsafeMutableBufferPointer { outPtr in
            input.withUnsafeBufferPointer { inPtr in
                cRuntime.runWithMemory(
                    outputs: outPtr.baseAddress!,
                    inputs: inPtr.baseAddress!,
                    memory: mem,
                    frameCount: frameCount
                )
            }
        }

        // With identity kernel (center=1), conv output should equal input
        // Sum of 9 ones = 9.0
        XCTAssertEqual(output[0], 9.0, accuracy: 1e-4, "Conv2d with identity kernel should preserve sum")
    }

    func testTensorHistoryExecution() throws {
        // Test that tensor history persists across frames
        let g = Graph()

        // Create a history buffer for 2x2 state (starts at 0)
        let stateBuffer = g.tensorHistoryBuffer(shape: [2, 2])

        // Read state (should be what was written last frame)
        let state = g.tensorHistoryRead(stateBuffer)

        // Add 1 to each element
        let newState = g.n(.add, state, g.n(.constant(1.0)))

        // Write back to history
        g.tensorHistoryWrite(stateBuffer, newState)

        // Output sum
        _ = g.n(.output(0), g.n(.sum, newState))

        let frameCount = 4  // Run 4 frames

        // Compile
        let cResult = try CompilationPipeline.compile(
            graph: g,
            backend: .c,
            options: .init(frameCount: frameCount, debug: true)
        )

        print("=== C Generated Code ===")
        print(cResult.source)

        // Create runtime
        let cRuntime = CCompiledKernel(
            source: cResult.source,
            cellAllocations: cResult.cellAllocations,
            memorySize: cResult.totalMemorySlots
        )
        try cRuntime.compileAndLoad()

        guard let mem = cRuntime.allocateNodeMemory() else {
            XCTFail("Failed to allocate memory")
            return
        }
        defer { cRuntime.deallocateNodeMemory(mem) }

        // Memory starts at 0 (already zero-initialized)

        // Run
        var output = [Float](repeating: 0, count: frameCount)
        let input = [Float](repeating: 0, count: frameCount)

        output.withUnsafeMutableBufferPointer { outPtr in
            input.withUnsafeBufferPointer { inPtr in
                cRuntime.runWithMemory(
                    outputs: outPtr.baseAddress!,
                    inputs: inPtr.baseAddress!,
                    memory: mem,
                    frameCount: frameCount
                )
            }
        }

        // Frame 0: state=[0,0,0,0], newState=[1,1,1,1], sum=4
        // Frame 1: state=[1,1,1,1], newState=[2,2,2,2], sum=8
        // Frame 2: state=[2,2,2,2], newState=[3,3,3,3], sum=12
        // Frame 3: state=[3,3,3,3], newState=[4,4,4,4], sum=16

        // Each frame adds 1 to each of 4 elements, so sum increases by 4 each frame
        XCTAssertEqual(output[0], 4.0, accuracy: 1e-5, "Frame 0 sum should be 4")
        XCTAssertEqual(output[1], 8.0, accuracy: 1e-5, "Frame 1 sum should be 8")
        XCTAssertEqual(output[2], 12.0, accuracy: 1e-5, "Frame 2 sum should be 12")
        XCTAssertEqual(output[3], 16.0, accuracy: 1e-5, "Frame 3 sum should be 16")
    }
}
