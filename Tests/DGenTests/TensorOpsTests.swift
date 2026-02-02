import XCTest

@testable import DGen

final class CTensorOpsTests: XCTestCase {

        // MARK: - Basic Tensor Operations

        func testTensorAddScalar() throws {
                // Test: tensor + scalar broadcasting
                let g = Graph()

                // Create a 2x2 tensor
                let tensorNode = g.tensor(shape: [2, 2])
                let scalar = g.n(.constant(1.0))
                let scalar2 = g.n(.constant(1.0))

                // Add scalar to tensor and multiply by 2 (scalar)
                let result = g.n(.mul, scalar2, g.n(.add, tensorNode, scalar))

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
                let kernel = g.tensor(
                        shape: [3, 3],
                        data: [
                                0, 1, 0,
                                1, -4, 1,
                                0, 1, 0,
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
                XCTAssertEqual(
                        output[0], 30.0, accuracy: 1e-5, "Sum of (tensor + 5) should be 30.0")
        }

        func testConv2dExecution() throws {
                // Test conv2d with a simple kernel
                let g = Graph()

                // 3x3 input tensor with all 1s
                let inputNode = g.ones(shape: [3, 3])

                // 3x3 identity kernel (center = 1, rest = 0)
                let kernelNode = g.tensor(
                        shape: [3, 3],
                        data: [
                                0, 0, 0,
                                0, 1, 0,
                                0, 0, 0,
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
                XCTAssertEqual(
                        output[0], 9.0, accuracy: 1e-4,
                        "Conv2d with identity kernel should preserve sum")
        }

        func testConv1dExecution() throws {
                // Test conv1d with a simple averaging kernel
                let g = Graph()

                // 5-element input: [1, 2, 3, 4, 5]
                let inputNode = g.tensor(shape: [5], data: [1, 2, 3, 4, 5])

                // 3-element averaging kernel: [1, 1, 1] (will sum neighbors)
                let kernelNode = g.tensor(shape: [3], data: [1, 1, 1])

                // Conv1d with kernel size 3
                let convResult = g.n(.conv1d(3), inputNode, kernelNode)

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

                // With kernel [1,1,1] on input [1,2,3,4,5]:
                // Output[0] = 0+1+2 = 3   (padding on left)
                // Output[1] = 1+2+3 = 6
                // Output[2] = 2+3+4 = 9
                // Output[3] = 3+4+5 = 12
                // Output[4] = 4+5+0 = 9   (padding on right)
                // Sum = 3+6+9+12+9 = 39
                XCTAssertEqual(
                        output[0], 39.0, accuracy: 1e-4,
                        "Conv1d with [1,1,1] kernel on [1,2,3,4,5] should sum to 39")
        }

        func testConv1dIdentityKernel() throws {
                // Test conv1d with identity kernel (center=1)
                let g = Graph()

                // 5-element input: [1, 2, 3, 4, 5]
                let inputNode = g.tensor(shape: [5], data: [1, 2, 3, 4, 5])

                // 3-element identity kernel: [0, 1, 0]
                let kernelNode = g.tensor(shape: [3], data: [0, 1, 0])

                // Conv1d
                let convResult = g.n(.conv1d(3), inputNode, kernelNode)

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

                // Identity kernel should preserve the input
                // Sum of [1,2,3,4,5] = 15
                XCTAssertEqual(
                        output[0], 15.0, accuracy: 1e-4,
                        "Conv1d with identity kernel should preserve sum (15)")
        }

        func testTensorHistoryExecutionC() throws {
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

        func testMembraneSimulationExecute() throws {
                // Simplified membrane physical model with damping:
                // state_t+1 = damping * (2*state_t - state_t-1 + c^2 * laplacian(state_t))
                // Without damping, energy is conserved and the wave oscillates forever.
                // With damping < 1.0, the wave gradually decays.

                let g = Graph()
                let gridShape: Shape = [4, 4]

                // Fixed: 16 values for 4x4 grid
                let excite = g.tensor(
                        shape: [4, 4],
                        data: [
                                0, 0, 0, 0,
                                0, 1, 1, 0,
                                0, 1, 1, 0,
                                0, 0, 0, 0,
                        ])

                // Very slow phasor (0.1 Hz) so it only triggers once during our test
                let phase = g.n(.phasor(g.alloc()), g.n(.constant(0.1)), g.n(.constant(0)))
                let shaped = g.rampToTrig(phase)
                let excitement = g.n(.mul, excite, shaped)

                // History buffers for state (these persist across frames, need manual cell allocation)
                let stateBuffer = g.tensorHistoryBuffer(shape: gridShape)
                let prevStateBuffer = g.tensorHistoryBuffer(shape: gridShape)

                // Read current and previous state from history
                let state_t = g.n(.add, excitement, g.tensorHistoryRead(stateBuffer))
                let state_t_1 = g.tensorHistoryRead(prevStateBuffer)

                // Laplacian kernel - use the clean tensor API with data
                let kernel = g.tensor(
                        shape: [3, 3],
                        data: [
                                0, 1, 0,
                                1, -4, 1,
                                0, 1, 0,
                        ])

                // Compute laplacian = conv2d(state_t, kernel)
                let laplacian = g.n(.conv2d([3, 3]), state_t, kernel)

                // Velocity-proportional damping (correct physical formulation):
                // state_t+1 = (2-d)*state_t - (1-d)*state_t_1 + c²*laplacian
                // where d is the damping coefficient (0 = undamped, higher = more damping)
                let c_squared = g.n(.constant(0.1))
                let d = g.n(.constant(0.03))  // Damping coefficient
                let two = g.n(.constant(2.0))
                let one = g.n(.constant(1.0))

                // Coefficients: (2-d) and (1-d)
                let twoMinusD = g.n(.sub, two, d)  // 1.98
                let oneMinusD = g.n(.sub, one, d)  // 0.98

                let scaledState = g.n(.mul, twoMinusD, state_t)
                let scaledPrev = g.n(.mul, oneMinusD, state_t_1)
                let scaledLaplacian = g.n(.mul, c_squared, laplacian)

                let diff = g.n(.sub, scaledState, scaledPrev)
                let state_t_plus_1 = g.n(.add, diff, scaledLaplacian)

                // Write new state to history (shift: prev <- current, current <- new)
                g.tensorHistoryWrite(prevStateBuffer, state_t)
                g.tensorHistoryWrite(stateBuffer, state_t_plus_1)

                // Output: sum of state (to get a scalar for audio output)
                let sumOutput = g.n(.sum, state_t_plus_1)
                _ = g.n(.output(0), sumOutput)

                // Run frames to observe decay (must be <= SCRATCH_STRIDE which is 512)
                let frameCount = 512

                // Compile for C
                let cResult = try CompilationPipeline.compile(
                        graph: g,
                        backend: .c,
                        options: .init(frameCount: frameCount, debug: true)
                )

                print("=== Membrane Simulation - C Source ===")
                print(cResult.source)

                XCTAssertFalse(cResult.source.isEmpty, "C source should not be empty")

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
                let memPtr = mem.assumingMemoryBound(to: Float.self)
                injectTensorData(result: cResult, memory: memPtr)

                // Get state buffer cell ID for inspection
                let stateCellId = stateBuffer.cellId
                let stateCellPhysical =
                        cResult.cellAllocations.cellMappings[stateCellId] ?? stateCellId

                print(
                        "=== State buffer at cell \(stateCellId) -> physical \(stateCellPhysical) ==="
                )

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

                // Print output values and check for damping
                print("\n=== OUTPUT VALUES (sum of state) ===")
                var maxOutput: Float = 0
                var maxFrame = 0
                for (i, x) in output.enumerated() {
                        // Print first 20, then every 50th, then last 10
                        if i < 20 || i % 50 == 0 || i >= frameCount - 10 {
                                print("frame \(i): output=\(x)")
                        }
                        if abs(x) > maxOutput {
                                maxOutput = abs(x)
                                maxFrame = i
                        }
                }

                // Inspect final state matrix
                print("\n=== Final State Matrix (4x4) ===")
                for row in 0..<4 {
                        var rowStr = "["
                        for col in 0..<4 {
                                let idx = row * 4 + col
                                let val = memPtr[stateCellPhysical + idx]
                                rowStr += String(format: "%8.4f", val)
                                if col < 3 { rowStr += ", " }
                        }
                        rowStr += "]"
                        print(rowStr)
                }

                // Verify damping: later frames should have smaller magnitude than peak
                let lastOutput = abs(output[frameCount - 1])
                print("\n=== Damping Check ===")
                print("Peak output: \(maxOutput) at frame \(maxFrame)")
                print("Final output: \(lastOutput)")

                // After initial excitation and transient, output should decay
                // Allow for some oscillation but expect overall decay trend
                XCTAssertLessThan(
                        lastOutput, maxOutput * 1.5,
                        "Output should not grow unbounded - simulation may be unstable")

                // With 512 frames and d=0.03 damping, should decay significantly
                // Final output should be much smaller than peak
                let decayRatio = lastOutput / maxOutput
                print("Decay ratio (final/peak): \(decayRatio)")
                XCTAssertLessThan(
                        decayRatio, 0.05,
                        "After 512 frames, output should decay to <5% of peak")

                // Check that energy has spread (corners should have some non-zero values after propagation)
                let corner00 = memPtr[stateCellPhysical + 0]
                let corner33 = memPtr[stateCellPhysical + 15]
                print("Corner [0,0]: \(corner00)")
                print("Corner [3,3]: \(corner33)")

                // After 64 frames, wave should have propagated to corners
                let cornerSum = abs(corner00) + abs(corner33)
                print("Corner sum (propagation check): \(cornerSum)")

        }

        // MARK: - Reshape Tests

        func testReshapeBasic() throws {
                let g = Graph()

                // Create a 2x3 tensor with data (use Float literals!)
                let t = g.tensor(shape: [2, 3], data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

                // First test: sum WITHOUT reshape (sanity check)
                // let result = g.n(.sum, t)

                // Reshape to 3x2
                let reshaped = try g.reshape(t, to: [3, 2])

                // Sum to verify the data is preserved
                let result = g.n(.sum, reshaped)
                _ = g.n(.output(0), result)

                let compilationResult = try CompilationPipeline.compile(
                        graph: g,
                        backend: .c,
                        options: .init(frameCount: 1, debug: true)
                )

                print("=== Reshape Basic - Generated Source ===")
                print(compilationResult.source)

                XCTAssertFalse(compilationResult.source.isEmpty)
        }

        func testSumTensorDirect() throws {
                // Baseline test: sum a tensor directly without reshape
                let g = Graph()

                let t = g.tensor(shape: [2, 3], data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                let result = g.n(.sum, t)
                _ = g.n(.output(0), result)

                let compilationResult = try CompilationPipeline.compile(
                        graph: g,
                        backend: .c,
                        options: .init(frameCount: 1, debug: true)
                )

                print("=== Sum Tensor Direct - Generated Source ===")
                print(compilationResult.source)

                XCTAssertFalse(compilationResult.source.isEmpty)
        }

        func testReshapeExecution() throws {
                let g = Graph()

                // Create a 2x3 tensor: [[1,2,3],[4,5,6]]
                let t = g.tensor(shape: [2, 3], data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

                // Reshape to 3x2 and sum
                let reshaped = try g.reshape(t, to: [3, 2])
                let result = g.n(.sum, reshaped)
                _ = g.n(.output(0), result)

                let frameCount = 1

                let cResult = try CompilationPipeline.compile(
                        graph: g,
                        backend: .c,
                        options: .init(frameCount: frameCount, debug: false)
                )

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

                injectTensorData(result: cResult, memory: mem.assumingMemoryBound(to: Float.self))

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

                print("=== Reshape Execution ===")
                print("Sum of reshaped tensor: \(output[0])")

                // Sum of 1+2+3+4+5+6 = 21
                XCTAssertEqual(output[0], 21.0, accuracy: 0.001)
        }

        // MARK: - Transpose Tests

        func testTransposeBasic() throws {
                let g = Graph()

                // Create a 2x3 tensor
                let t = g.tensor(shape: [2, 3], data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

                // Transpose to 3x2
                let transposed = try g.transpose(t)

                // Sum to verify data preserved
                let result = g.n(.sum, transposed)
                _ = g.n(.output(0), result)

                let compilationResult = try CompilationPipeline.compile(
                        graph: g,
                        backend: .c,
                        options: .init(frameCount: 1, debug: true)
                )

                print("=== Transpose Basic - Generated Source ===")
                print(compilationResult.source)

                XCTAssertFalse(compilationResult.source.isEmpty)
        }

        // MARK: - SumAxis Tests

        func testSumAxisBasic() throws {
                let g = Graph()

                // Create a 2x3 tensor
                let t = g.tensor(shape: [2, 3], data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

                // Sum along axis 1 (columns) -> [6, 15]
                let summed = try g.sum(t, axis: 1)

                // Sum the result to get a scalar
                let result = g.n(.sum, summed)
                _ = g.n(.output(0), result)

                let compilationResult = try CompilationPipeline.compile(
                        graph: g,
                        backend: .c,
                        options: .init(frameCount: 1, debug: true)
                )

                print("=== SumAxis Basic - Generated Source ===")
                print(compilationResult.source)

                XCTAssertFalse(compilationResult.source.isEmpty)
        }

        func testSumAxisExecution() throws {
                let g = Graph()

                // Create a 2x3 tensor: [[1,2,3],[4,5,6]]
                let t = g.tensor(shape: [2, 3], data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

                // Sum along axis -1 (last axis, columns) -> [6, 15]
                // Then sum again to get 21
                let summed = try g.sum(t, axis: -1)
                let result = g.n(.sum, summed)
                _ = g.n(.output(0), result)

                let frameCount = 1

                let cResult = try CompilationPipeline.compile(
                        graph: g,
                        backend: .c,
                        options: .init(frameCount: frameCount, debug: false)
                )

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

                injectTensorData(result: cResult, memory: mem.assumingMemoryBound(to: Float.self))

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

                print("=== SumAxis Execution ===")
                print("Sum after sumAxis: \(output[0])")

                // Sum of [[1,2,3],[4,5,6]] along axis -1 gives [6, 15], sum of that is 21
                XCTAssertEqual(output[0], 21.0, accuracy: 0.001)
        }

        // MARK: - Matmul Tests

        func testMatmulExecution() throws {
                let g = Graph()

                // A: 2x3 matrix
                let a = g.tensor(shape: [2, 3], data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

                // B: 3x2 matrix
                let b = g.tensor(shape: [3, 2], data: [7.0, 8.0, 9.0, 10.0, 11.0, 12.0])

                // C = A @ B should be 2x2
                // C[0,0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
                // C[0,1] = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
                // C[1,0] = 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
                // C[1,1] = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
                // Sum = 58 + 64 + 139 + 154 = 415
                let c = try g.matmul(a, b)
                let result = g.n(.sum, c)
                _ = g.n(.output(0), result)

                let frameCount = 1
                let cResult = try CompilationPipeline.compile(
                        graph: g,
                        backend: .c,
                        options: .init(frameCount: frameCount, debug: true)
                )

                print("=== Matmul Execution - Generated Source ===")
                print(cResult.source)

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

                injectTensorData(result: cResult, memory: mem.assumingMemoryBound(to: Float.self))

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

                print("=== Matmul Result ===")
                print("Output: \(output[0]), Expected: 415.0")

                // A @ B = [[58, 64], [139, 154]], sum = 415
                XCTAssertEqual(output[0], 415.0, accuracy: 0.001, "Matmul sum should be 415")
        }

        func testMatmulWithScalarMul() throws {
                // Bug: matmul(tensorA, tensorB * scalar) crashes saying "matmul requires tensor inputs"
                // The result of tensor * scalar should be a tensor!
                let g = Graph()

                // A: 2x3 matrix
                let a = g.tensor(shape: [2, 3], data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

                // B: 3x2 matrix, multiplied by a scalar (simulating phasor output)
                let b = g.tensor(shape: [3, 2], data: [7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
                let scalar = g.n(.constant(2.0))  // Simulate phasor output
                let bScaled = g.n(.mul, b, scalar)  // tensor * scalar should still be tensor

                // This should work: matmul(a, b * scalar)
                let c = try g.matmul(a, bScaled)
                let result = g.n(.sum, c)
                _ = g.n(.output(0), result)

                let frameCount = 1
                let cResult = try CompilationPipeline.compile(
                        graph: g,
                        backend: .c,
                        options: .init(frameCount: frameCount, debug: true)
                )

                print("=== Matmul with Scalar Mul - Generated Source ===")
                print(cResult.source)

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

                injectTensorData(result: cResult, memory: mem.assumingMemoryBound(to: Float.self))

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

                print("=== Matmul with Scalar Mul Result ===")
                print("Output: \(output[0]), Expected: 830.0")  // 415 * 2 = 830

                // (A @ (B * 2)) = 2 * (A @ B) = 2 * 415 = 830
                XCTAssertEqual(
                        output[0], 830.0, accuracy: 0.001, "Matmul with scaled B should be 830")
        }

        // MARK: - Comprehensive Reshape Tests

        func testReshapeToFlat() throws {
                // Reshape 2D to 1D
                let g = Graph()

                // [2,3] -> [6]
                let t = g.tensor(shape: [2, 3], data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                let flat = try g.reshape(t, to: [6])

                // Multiply by weights [1,2,3,4,5,6] to verify order preserved
                let weights = g.tensor(shape: [6], data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                let weighted = g.n(.mul, flat, weights)
                let result = g.n(.sum, weighted)
                _ = g.n(.output(0), result)

                let frameCount = 1
                let cResult = try CompilationPipeline.compile(
                        graph: g, backend: .c,
                        options: .init(frameCount: frameCount, debug: false)
                )

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

                injectTensorData(result: cResult, memory: mem.assumingMemoryBound(to: Float.self))

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

                // 1*1 + 2*2 + 3*3 + 4*4 + 5*5 + 6*6 = 1+4+9+16+25+36 = 91
                XCTAssertEqual(
                        output[0], 91.0, accuracy: 0.001,
                        "Reshape to flat should preserve data order")
        }

        func testReshapeFromFlat() throws {
                // Reshape 1D to 2D, then use sumAxis to verify shape
                let g = Graph()

                // [6] -> [2,3]
                let t = g.tensor(shape: [6], data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                let reshaped = try g.reshape(t, to: [2, 3])

                // sumAxis(1) on [2,3] -> [1+2+3, 4+5+6] = [6, 15]
                let summed = try g.sum(reshaped, axis: 1)

                // Multiply by [1, 2] -> 6*1 + 15*2 = 36
                let weights = g.tensor(shape: [2], data: [1.0, 2.0])
                let weighted = g.n(.mul, summed, weights)
                let result = g.n(.sum, weighted)
                _ = g.n(.output(0), result)

                let frameCount = 1
                let cResult = try CompilationPipeline.compile(
                        graph: g, backend: .c,
                        options: .init(frameCount: frameCount, debug: false)
                )

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

                injectTensorData(result: cResult, memory: mem.assumingMemoryBound(to: Float.self))

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

                // [6, 15] * [1, 2] = 6 + 30 = 36
                XCTAssertEqual(
                        output[0], 36.0, accuracy: 0.001,
                        "Reshape from flat should create correct 2D shape")
        }

        // MARK: - Comprehensive Transpose Tests

        func testTransposeExecution() throws {
                // Transpose [2,3] -> [3,2] and verify with sumAxis
                // This test verifies that transpose correctly reorders elements via strided indexing
                let g = Graph()

                // [2,3]: [[1,2,3], [4,5,6]] stored as [1,2,3,4,5,6]
                // Transposed [3,2]: [[1,4], [2,5], [3,6]] (same memory, different strides)
                let t = g.tensor(shape: [2, 3], data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                let transposed = try g.transpose(t)  // shape [3,2], strides [1, 3]

                // sumAxis(1) on transposed [3,2]: [1+4, 2+5, 3+6] = [5, 7, 9]
                let summed = try g.sum(transposed, axis: 1)

                // Multiply by [1, 2, 3] -> 5*1 + 7*2 + 9*3 = 5 + 14 + 27 = 46
                let weights = g.tensor(shape: [3], data: [1.0, 2.0, 3.0])
                let weighted = g.n(.mul, summed, weights)
                let result = g.n(.sum, weighted)
                _ = g.n(.output(0), result)

                let frameCount = 1
                let cResult = try CompilationPipeline.compile(
                        graph: g, backend: .c,
                        options: .init(frameCount: frameCount, debug: true)
                )

                print("=== Transpose Execution - Generated Source ===")
                print(cResult.source)

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

                injectTensorData(result: cResult, memory: mem.assumingMemoryBound(to: Float.self))

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

                print("=== Transpose Result ===")
                print("Output: \(output[0]), Expected: 46.0")

                // Transpose [3,2]: [[1,4], [2,5], [3,6]] -> sumAxis(1) = [5, 7, 9]
                // Weighted: 5*1 + 7*2 + 9*3 = 46
                // (Compare to reshape which would give [3, 7, 11] -> weighted = 50)
                XCTAssertEqual(
                        output[0], 46.0, accuracy: 0.001,
                        "Transpose should reorder elements correctly via strided indexing")
        }

        // MARK: - Comprehensive SumAxis Tests

        func testSumAxisAxis0() throws {
                // Sum along axis 0 (sum columns)
                let g = Graph()

                // [2,3]: [[1,2,3], [4,5,6]]
                // sumAxis(0) -> [1+4, 2+5, 3+6] = [5, 7, 9]
                let t = g.tensor(shape: [2, 3], data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                let summed = try g.sum(t, axis: 0)

                // Multiply by [1, 2, 3] -> 5*1 + 7*2 + 9*3 = 5 + 14 + 27 = 46
                let weights = g.tensor(shape: [3], data: [1.0, 2.0, 3.0])
                let weighted = g.n(.mul, summed, weights)
                let result = g.n(.sum, weighted)
                _ = g.n(.output(0), result)

                let frameCount = 1
                let cResult = try CompilationPipeline.compile(
                        graph: g, backend: .c,
                        options: .init(frameCount: frameCount, debug: false)
                )

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

                injectTensorData(result: cResult, memory: mem.assumingMemoryBound(to: Float.self))

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

                XCTAssertEqual(output[0], 46.0, accuracy: 0.001, "SumAxis(0) should sum columns")
        }

        // NOTE: 3D tensor sumAxis test skipped due to code gen bug with nested parallel ranges.
        // The 3D case triggers similar SIMD variable scoping issues as matmul.
        // This needs investigation in the C renderer's handling of higher-dimensional tensors.

        func testSumAxisToScalar() throws {
                // SumAxis on 1D tensor reduces to scalar
                let g = Graph()

                let t = g.tensor(shape: [4], data: [1.0, 2.0, 3.0, 4.0])
                let summed = try g.sum(t, axis: 0)  // Only axis, becomes scalar
                _ = g.n(.output(0), summed)

                let frameCount = 1
                let cResult = try CompilationPipeline.compile(
                        graph: g, backend: .c,
                        options: .init(frameCount: frameCount, debug: false)
                )

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

                injectTensorData(result: cResult, memory: mem.assumingMemoryBound(to: Float.self))

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

                XCTAssertEqual(
                        output[0], 10.0, accuracy: 0.001, "SumAxis on 1D should reduce to scalar")
        }

        // MARK: - Broadcasting Tests

        func testBroadcastScalarTensor() throws {
                // Scalar + Tensor broadcasting
                let g = Graph()

                let t = g.tensor(shape: [2, 3], data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                let scalar = g.n(.constant(10.0))
                let added = g.n(.add, t, scalar)
                let result = g.n(.sum, added)
                _ = g.n(.output(0), result)

                let frameCount = 1
                let cResult = try CompilationPipeline.compile(
                        graph: g, backend: .c,
                        options: .init(frameCount: frameCount, debug: false)
                )

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

                injectTensorData(result: cResult, memory: mem.assumingMemoryBound(to: Float.self))

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

                // (1+10) + (2+10) + ... + (6+10) = 21 + 60 = 81
                XCTAssertEqual(output[0], 81.0, accuracy: 0.001, "Scalar + Tensor should broadcast")
        }

        // NOTE: 1D to 2D broadcasting needs proper implementation
        // Currently the type checker supports broadcasting shapes, but code gen
        // doesn't properly handle strided access for broadcast dimensions.
        // Skipping this test until broadcasting code gen is implemented.

        // MARK: - Nested ParallelRange Debug Test

        func testNestedParallelRangeDebug() throws {
                // Test: 4x3 tensor -> sumAxis(1) -> [4] output
                // This triggers SIMD when output size is divisible by 4
                let g = Graph()

                // 4x3 tensor -> sumAxis(1) -> [4] output
                let t = g.tensor(
                        shape: [4, 3],
                        data: [
                                1.0, 2.0, 3.0,  // row 0: sum = 6
                                4.0, 5.0, 6.0,  // row 1: sum = 15
                                7.0, 8.0, 9.0,  // row 2: sum = 24
                                10.0, 11.0, 12.0,  // row 3: sum = 33
                        ])

                let summed = try g.sum(t, axis: 1)  // [4,3] -> [4]

                // Output sum: 6 + 15 + 24 + 33 = 78
                let result = g.n(.sum, summed)
                _ = g.n(.output(0), result)

                let frameCount = 1
                let cResult = try CompilationPipeline.compile(
                        graph: g, backend: .c,
                        options: .init(frameCount: frameCount, debug: true)
                )

                print("=== Nested ParallelRange Debug - Generated Source ===")
                print(cResult.source)

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

                injectTensorData(result: cResult, memory: mem.assumingMemoryBound(to: Float.self))

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

                // [4,3] -> sumAxis(1) -> [6, 15, 24, 33] -> sum -> 78
                XCTAssertEqual(
                        output[0], 78.0, accuracy: 0.001, "Sum of sumAxis(1) on [4,3] should be 78")
        }

        func testReshapeThenSumAxisExecution() throws {
                // Test: reshape changes how sumAxis interprets axes
                // This is a meaningful reshape test because sumAxis behavior depends on shape
                let g = Graph()

                // Tensor with data [1,2,3,4,5,6] (row-major)
                // As [2,3]: [[1,2,3], [4,5,6]]
                //   sumAxis(0) -> [5, 7, 9] (sum columns) -> total 21
                //   sumAxis(1) -> [6, 15] (sum rows) -> total 21
                //
                // As [3,2]: [[1,2], [3,4], [5,6]]
                //   sumAxis(0) -> [9, 12] (sum columns) -> total 21
                //   sumAxis(1) -> [3, 7, 11] (sum rows) -> total 21
                //
                // The totals are always 21, but intermediate results differ!
                // To test reshape, we need to check the intermediate sumAxis result.

                // Test 1: [2,3] sumAxis(1) gives [6, 15], then sum gives 21
                // But if we reshape to [3,2] first, sumAxis(1) gives [3, 7, 11], sum still 21
                // We need a way to distinguish these...

                // Better approach: after sumAxis, multiply by position weights
                // [2,3] -> sumAxis(1) -> [6, 15] -> weighted sum: 6*1 + 15*2 = 36
                // [3,2] -> sumAxis(1) -> [3, 7, 11] -> weighted sum: 3*1 + 7*2 + 11*3 = 50

                let t = g.tensor(shape: [2, 3], data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

                // Reshape to [3,2]: [[1,2], [3,4], [5,6]]
                let reshaped = try g.reshape(t, to: [3, 2])

                // sumAxis(1) on [3,2] sums each row: [1+2, 3+4, 5+6] = [3, 7, 11]
                let summed = try g.sum(reshaped, axis: 1)

                // Multiply by weights [1, 2, 3] and sum: 3*1 + 7*2 + 11*3 = 3 + 14 + 33 = 50
                let weights = g.tensor(shape: [3], data: [1.0, 2.0, 3.0])
                let weighted = g.n(.mul, summed, weights)
                let result = g.n(.sum, weighted)
                _ = g.n(.output(0), result)

                let frameCount = 1

                let cResult = try CompilationPipeline.compile(
                        graph: g,
                        backend: .c,
                        options: .init(frameCount: frameCount, debug: true)
                )

                print("=== Reshape Then SumAxis - Generated Source ===")
                print(cResult.source)

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

                injectTensorData(result: cResult, memory: mem.assumingMemoryBound(to: Float.self))

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

                print("=== Reshape Then SumAxis Result ===")
                print("Output: \(output[0]), Expected: 50.0")

                // If reshape didn't work (still [2,3]):
                //   sumAxis(1) -> [6, 15] (only 2 elements!)
                //   weights [1,2,3] wouldn't match shape
                //
                // With reshape to [3,2]:
                //   sumAxis(1) -> [3, 7, 11] (3 elements)
                //   weighted: 3*1 + 7*2 + 11*3 = 50
                XCTAssertEqual(
                        output[0], 50.0, accuracy: 0.001, "Reshape then sumAxis should give 50.0")
        }

        // MARK: - Stack Tests

        func testStackBasic() throws {
                // Stack 4 scalar constants into a [4] tensor
                let g = Graph()

                let s1 = g.n(.constant(1.0))
                let s2 = g.n(.constant(2.0))
                let s3 = g.n(.constant(3.0))
                let s4 = g.n(.constant(4.0))

                let stacked = try g.stack([s1, s2, s3, s4])

                // Sum the stacked tensor: 1 + 2 + 3 + 4 = 10
                let result = g.n(.sum, stacked)
                _ = g.n(.output(0), result)

                let frameCount = 1
                let cResult = try CompilationPipeline.compile(
                        graph: g, backend: .c,
                        options: .init(frameCount: frameCount, debug: true)
                )

                print("=== Stack Basic - Generated Source ===")
                print(cResult.source)

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

                print("=== Stack Basic Result ===")
                print("Output: \(output[0]), Expected: 10.0")

                XCTAssertEqual(output[0], 10.0, accuracy: 0.001, "Stack sum should be 10.0")
        }

        func testStackWithShape() throws {
                // Stack 4 scalars into a [2, 2] tensor
                let g = Graph()

                let s1 = g.n(.constant(1.0))
                let s2 = g.n(.constant(2.0))
                let s3 = g.n(.constant(3.0))
                let s4 = g.n(.constant(4.0))

                let stacked = try g.stack([s1, s2, s3, s4], shape: [2, 2])

                // Sum along axis 0: [[1,2],[3,4]] -> [4, 6]
                let summed = try g.sum(stacked, axis: 0)

                // Multiply by weights [1, 2]: 4*1 + 6*2 = 16
                let weights = g.tensor(shape: [2], data: [1.0, 2.0])
                let weighted = g.n(.mul, summed, weights)
                let result = g.n(.sum, weighted)
                _ = g.n(.output(0), result)

                let frameCount = 1
                let cResult = try CompilationPipeline.compile(
                        graph: g, backend: .c,
                        options: .init(frameCount: frameCount, debug: true)
                )

                print("=== Stack With Shape - Generated Source ===")
                print(cResult.source)

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

                injectTensorData(result: cResult, memory: mem.assumingMemoryBound(to: Float.self))

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

                print("=== Stack With Shape Result ===")
                print("Output: \(output[0]), Expected: 16.0")

                XCTAssertEqual(
                        output[0], 16.0, accuracy: 0.001, "Stack with shape should give 16.0")
        }

        // MARK: - Shrink Tests

        func testShrinkBasic() throws {
                // Test basic shrink compilation
                let g = Graph()

                // Create a 4x4 tensor
                let t = g.tensor(
                        shape: [4, 4],
                        data: [
                                1.0, 2.0, 3.0, 4.0,
                                5.0, 6.0, 7.0, 8.0,
                                9.0, 10.0, 11.0, 12.0,
                                13.0, 14.0, 15.0, 16.0,
                        ])

                // Shrink to rows 1:3, cols 1:3 -> [[6,7], [10,11]]
                let shrunk = try g.shrink(t, ranges: [(1, 3), (1, 3)])

                // Sum to verify
                let result = g.n(.sum, shrunk)
                _ = g.n(.output(0), result)

                let compilationResult = try CompilationPipeline.compile(
                        graph: g,
                        backend: .c,
                        options: .init(frameCount: 1, debug: true)
                )

                print("=== Shrink Basic - Generated Source ===")
                print(compilationResult.source)

                XCTAssertFalse(compilationResult.source.isEmpty)
        }

        func testShrinkExecution() throws {
                // Test shrink with actual execution
                let g = Graph()

                // Create a 4x4 tensor:
                // [[ 1,  2,  3,  4],
                //  [ 5,  6,  7,  8],
                //  [ 9, 10, 11, 12],
                //  [13, 14, 15, 16]]
                let t = g.tensor(
                        shape: [4, 4],
                        data: [
                                1.0, 2.0, 3.0, 4.0,
                                5.0, 6.0, 7.0, 8.0,
                                9.0, 10.0, 11.0, 12.0,
                                13.0, 14.0, 15.0, 16.0,
                        ])

                // Shrink to rows 1:3, cols 1:3 -> 2x2 submatrix:
                // [[6, 7],
                //  [10, 11]]
                // Sum = 6 + 7 + 10 + 11 = 34
                let shrunk = try g.shrink(t, ranges: [(1, 3), (1, 3)])
                let result = g.n(.sum, shrunk)
                _ = g.n(.output(0), result)

                let frameCount = 1
                let cResult = try CompilationPipeline.compile(
                        graph: g,
                        backend: .c,
                        options: .init(frameCount: frameCount, debug: true)
                )

                print("=== Shrink Execution - Generated Source ===")
                print(cResult.source)

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

                injectTensorData(result: cResult, memory: mem.assumingMemoryBound(to: Float.self))

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

                print("=== Shrink Execution Result ===")
                print("Output: \(output[0]), Expected: 34.0")

                // Shrunk [[6,7],[10,11]] sum = 34
                XCTAssertEqual(output[0], 34.0, accuracy: 0.001, "Shrink sum should be 34.0")
        }

        func testShrinkColumnOnly() throws {
                // Shrink only columns, keep all rows
                let g = Graph()

                // 3x6 tensor
                let t = g.tensor(
                        shape: [3, 6],
                        data: [
                                1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                                7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                                13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
                        ])

                // Shrink to cols 2:5 (keep all rows) -> 3x3 submatrix:
                // [[3, 4, 5],
                //  [9, 10, 11],
                //  [15, 16, 17]]
                // Sum = 3+4+5 + 9+10+11 + 15+16+17 = 12 + 30 + 48 = 90
                let shrunk = try g.shrink(t, ranges: [nil, (2, 5)])
                let result = g.n(.sum, shrunk)
                _ = g.n(.output(0), result)

                let frameCount = 1
                let cResult = try CompilationPipeline.compile(
                        graph: g,
                        backend: .c,
                        options: .init(frameCount: frameCount, debug: false)
                )

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

                injectTensorData(result: cResult, memory: mem.assumingMemoryBound(to: Float.self))

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

                print("=== Shrink Column Only Result ===")
                print("Output: \(output[0]), Expected: 90.0")

                XCTAssertEqual(
                        output[0], 90.0, accuracy: 0.001, "Shrink column-only sum should be 90.0")
        }

        func testShrinkWithScalarOp() throws {
                // Shrink then element-wise op with scalar
                let g = Graph()

                // 4x4 tensor
                let t = g.tensor(
                        shape: [4, 4],
                        data: [
                                1.0, 2.0, 3.0, 4.0,
                                5.0, 6.0, 7.0, 8.0,
                                9.0, 10.0, 11.0, 12.0,
                                13.0, 14.0, 15.0, 16.0,
                        ])

                // Shrink to [[6,7],[10,11]]
                let shrunk = try g.shrink(t, ranges: [(1, 3), (1, 3)])

                // Multiply by 2: [[12,14],[20,22]]
                let scaled = g.n(.mul, shrunk, g.n(.constant(2.0)))

                // Sum = 12 + 14 + 20 + 22 = 68
                let result = g.n(.sum, scaled)
                _ = g.n(.output(0), result)

                let frameCount = 1
                let cResult = try CompilationPipeline.compile(
                        graph: g,
                        backend: .c,
                        options: .init(frameCount: frameCount, debug: false)
                )

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

                injectTensorData(result: cResult, memory: mem.assumingMemoryBound(to: Float.self))

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

                // (6+7+10+11) * 2 = 34 * 2 = 68
                XCTAssertEqual(
                        output[0], 68.0, accuracy: 0.001, "Shrink then scalar mul should be 68.0")
        }

        func testShrinkWithSumAxis() throws {
                // Shrink then sumAxis - tests strided access in sumAxis
                let g = Graph()

                // 4x4 tensor
                let t = g.tensor(
                        shape: [4, 4],
                        data: [
                                1.0, 2.0, 3.0, 4.0,
                                5.0, 6.0, 7.0, 8.0,
                                9.0, 10.0, 11.0, 12.0,
                                13.0, 14.0, 15.0, 16.0,
                        ])

                // Shrink to rows 1:3, cols 0:4 (keep all cols) -> 2x4:
                // [[5, 6, 7, 8],
                //  [9, 10, 11, 12]]
                let shrunk = try g.shrink(t, ranges: [(1, 3), nil])

                // sumAxis(1) -> [5+6+7+8, 9+10+11+12] = [26, 42]
                let summed = try g.sum(shrunk, axis: 1)

                // Multiply by weights [1, 2] -> 26*1 + 42*2 = 26 + 84 = 110
                let weights = g.tensor(shape: [2], data: [1.0, 2.0])
                let weighted = g.n(.mul, summed, weights)
                let result = g.n(.sum, weighted)
                _ = g.n(.output(0), result)

                let frameCount = 1
                let cResult = try CompilationPipeline.compile(
                        graph: g,
                        backend: .c,
                        options: .init(frameCount: frameCount, debug: true)
                )

                print("=== Shrink With SumAxis - Generated Source ===")
                print(cResult.source)

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

                injectTensorData(result: cResult, memory: mem.assumingMemoryBound(to: Float.self))

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

                print("=== Shrink With SumAxis Result ===")
                print("Output: \(output[0]), Expected: 110.0")

                XCTAssertEqual(
                        output[0], 110.0, accuracy: 0.001, "Shrink then sumAxis should be 110.0")
        }

        func testChainedShrink() throws {
                // Test shrinking a shrunk tensor (cumulative offset)
                let g = Graph()

                // 6x6 tensor
                var data = [Float]()
                for i in 0..<36 {
                        data.append(Float(i + 1))
                }
                let t = g.tensor(shape: [6, 6], data: data)

                // First shrink: rows 1:5, cols 1:5 -> 4x4 submatrix
                // Values at positions: offset = 1*6 + 1 = 7
                // [[8,9,10,11], [14,15,16,17], [20,21,22,23], [26,27,28,29]]
                let shrunk1 = try g.shrink(t, ranges: [(1, 5), (1, 5)])

                // Second shrink: rows 1:3, cols 1:3 -> 2x2 submatrix
                // From the 4x4 above, take [[15,16], [21,22]]
                // Sum = 15 + 16 + 21 + 22 = 74
                let shrunk2 = try g.shrink(shrunk1, ranges: [(1, 3), (1, 3)])

                let result = g.n(.sum, shrunk2)
                _ = g.n(.output(0), result)

                let frameCount = 1
                let cResult = try CompilationPipeline.compile(
                        graph: g,
                        backend: .c,
                        options: .init(frameCount: frameCount, debug: true)
                )

                print("=== Chained Shrink - Generated Source ===")
                print(cResult.source)

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

                injectTensorData(result: cResult, memory: mem.assumingMemoryBound(to: Float.self))

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

                print("=== Chained Shrink Result ===")
                print("Output: \(output[0]), Expected: 74.0")

                // Chained shrink should give [[15,16],[21,22]] sum = 74
                XCTAssertEqual(
                        output[0], 74.0, accuracy: 0.001, "Chained shrink sum should be 74.0")
        }

        func testShrinkWithBroadcastOp() throws {
                // Shrink then broadcast multiply with another shrunk tensor
                let g = Graph()

                // 4x4 tensor A
                let a = g.tensor(
                        shape: [4, 4],
                        data: [
                                1.0, 2.0, 3.0, 4.0,
                                5.0, 6.0, 7.0, 8.0,
                                9.0, 10.0, 11.0, 12.0,
                                13.0, 14.0, 15.0, 16.0,
                        ])

                // 4x4 tensor B (all 2s)
                let b = g.tensor(shape: [4, 4], data: [Float](repeating: 2.0, count: 16))

                // Shrink both to 2x2
                let shrunkA = try g.shrink(a, ranges: [(1, 3), (1, 3)])  // [[6,7],[10,11]]
                let shrunkB = try g.shrink(b, ranges: [(0, 2), (0, 2)])  // [[2,2],[2,2]]

                // Element-wise multiply: [[12,14],[20,22]]
                let product = g.n(.mul, shrunkA, shrunkB)

                // Sum = 12 + 14 + 20 + 22 = 68
                let result = g.n(.sum, product)
                _ = g.n(.output(0), result)

                let frameCount = 1
                let cResult = try CompilationPipeline.compile(
                        graph: g,
                        backend: .c,
                        options: .init(frameCount: frameCount, debug: false)
                )

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

                injectTensorData(result: cResult, memory: mem.assumingMemoryBound(to: Float.self))

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

                // [[6,7],[10,11]] * [[2,2],[2,2]] = [[12,14],[20,22]] sum = 68
                XCTAssertEqual(
                        output[0], 68.0, accuracy: 0.001, "Shrink broadcast mul sum should be 68.0")
        }

        func testShrinkThenConv2d() throws {
                // Shrink a tensor, then apply conv2d with identity kernel
                // This tests that conv2d properly uses the shrunk tensor's offset
                let g = Graph()

                // 4x4 tensor where first 2x2 block is zeros, rest is ones
                // [[0, 0, 1, 1],
                //  [0, 0, 1, 1],
                //  [1, 1, 1, 1],
                //  [1, 1, 1, 1]]
                let t = g.tensor(
                        shape: [4, 4],
                        data: [
                                0.0, 0.0, 1.0, 1.0,
                                0.0, 0.0, 1.0, 1.0,
                                1.0, 1.0, 1.0, 1.0,
                                1.0, 1.0, 1.0, 1.0,
                        ])

                // Shrink to bottom-right 2x2 which should be all 1s:
                // [[1, 1],
                //  [1, 1]]
                let shrunk = try g.shrink(t, ranges: [(2, 4), (2, 4)])

                // Identity kernel (just passes through center value)
                let kernel = g.tensor(
                        shape: [3, 3],
                        data: [
                                0.0, 0.0, 0.0,
                                0.0, 1.0, 0.0,
                                0.0, 0.0, 0.0,
                        ])

                // Conv2d with identity kernel on 2x2 all-ones should give 2x2 all-ones
                let convResult = g.n(.conv2d([3, 3]), shrunk, kernel)

                // Sum should be 4.0 (four ones)
                let result = g.n(.sum, convResult)
                _ = g.n(.output(0), result)

                let frameCount = 1
                let cResult = try CompilationPipeline.compile(
                        graph: g,
                        backend: .c,
                        options: .init(frameCount: frameCount, debug: true)
                )

                print("=== Shrink Then Conv2d - Generated Source ===")
                print(cResult.source)

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

                injectTensorData(result: cResult, memory: mem.assumingMemoryBound(to: Float.self))

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

                print("=== Shrink Then Conv2d Result ===")
                print("Output: \(output[0]), Expected: 4.0")

                // If conv2d correctly reads from shrunk region (all 1s), sum = 4.0
                // If conv2d incorrectly reads from start (first 2x2 is zeros), sum would be < 4.0
                XCTAssertEqual(
                        output[0], 4.0, accuracy: 0.001,
                        "Conv2d on shrunk all-ones should sum to 4.0 - offset may not be applied")
        }

        // MARK: - Pad Tests

        func testPadBasic() throws {
                // Test: basic pad operation - metadata only
                let g = Graph()

                // Create a 2x2 tensor with data [1, 2, 3, 4]
                let tensorNode = g.tensor([[1.0, 2.0], [3.0, 4.0]])

                // Pad with 1 on each side: (1, 1) for each axis
                // Result shape should be [4, 4]
                let padded = try g.pad(tensorNode, padding: [(1, 1), (1, 1)])

                // Verify the padded tensor has correct shape
                let paddedTensor = try g.getTensor(padded)
                XCTAssertEqual(paddedTensor.shape, [4, 4], "Padded shape should be [4, 4]")
                XCTAssertNotNil(paddedTensor.padding, "Padded tensor should have padding info")
                XCTAssertEqual(paddedTensor.padding?[0].left, 1)
                XCTAssertEqual(paddedTensor.padding?[0].right, 1)
        }

        func testPadExecution() throws {
                // Test: pad a tensor and sum it - zeros in pad region
                let g = Graph()

                // Create a 2x2 tensor with data [1, 2, 3, 4] (sum = 10)
                let tensorNode = g.tensor([[1.0, 2.0], [3.0, 4.0]])

                // Pad with 1 on each side -> [4, 4] tensor
                // Padded region is zeros, so sum should still be 10
                let padded = try g.pad(tensorNode, padding: [(1, 1), (1, 1)])

                // Sum the padded tensor
                let sumResult = g.n(.sum, padded)
                _ = g.n(.output(0), sumResult)

                let frameCount = 1
                let cResult = try CompilationPipeline.compile(
                        graph: g,
                        backend: .c,
                        options: .init(frameCount: frameCount, debug: true)
                )

                print("=== Pad Execution - Generated Source ===")
                print(cResult.source)

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

                injectTensorData(result: cResult, memory: mem.assumingMemoryBound(to: Float.self))

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

                print("=== Pad Execution Result ===")
                print("Output: \(output[0]), Expected: 10.0")

                // Sum of padded tensor should equal sum of original (zeros don't contribute)
                XCTAssertEqual(
                        output[0], 10.0, accuracy: 0.001,
                        "Sum of padded tensor should equal sum of original data")
        }

        func testPadAsymmetric() throws {
                // Test: asymmetric padding (different left/right)
                let g = Graph()

                // Create a 1D tensor [1, 2, 3]
                let tensorNode = g.tensor([1.0, 2.0, 3.0])

                // Pad with 2 on left, 1 on right -> [0, 0, 1, 2, 3, 0]
                let padded = try g.pad(tensorNode, padding: [(2, 1)])

                let paddedTensor = try g.getTensor(padded)
                XCTAssertEqual(paddedTensor.shape, [6], "Padded shape should be [6]")

                // Sum should still be 6
                let sumResult = g.n(.sum, padded)
                _ = g.n(.output(0), sumResult)

                let frameCount = 1
                let cResult = try CompilationPipeline.compile(
                        graph: g,
                        backend: .c,
                        options: .init(frameCount: frameCount, debug: true)
                )

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

                injectTensorData(result: cResult, memory: mem.assumingMemoryBound(to: Float.self))

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

                print("=== Asymmetric Pad Result ===")
                print("Output: \(output[0]), Expected: 6.0")

                XCTAssertEqual(output[0], 6.0, accuracy: 0.001)
        }

        func testConcatViaPadAndAdd() throws {
                // Test: concat [1, 2] and [3, 4] via padding and addition
                // [1, 2] padded right: [1, 2, 0, 0]
                // [3, 4] padded left:  [0, 0, 3, 4]
                // Sum:                 [1, 2, 3, 4]
                let g = Graph()

                // Create two 1D tensors
                let t1 = g.tensor([1.0, 2.0])  // [1, 2]
                let t2 = g.tensor([3.0, 4.0])  // [3, 4]

                // Pad t1 with 0 left, 2 right -> [1, 2, 0, 0]
                let t1Padded = try g.pad(t1, padding: [(0, 2)])

                // Pad t2 with 2 left, 0 right -> [0, 0, 3, 4]
                let t2Padded = try g.pad(t2, padding: [(2, 0)])

                // Add them together -> [1, 2, 3, 4]
                let concat = g.n(.add, t1Padded, t2Padded)

                // Sum to verify: 1 + 2 + 3 + 4 = 10
                let sumResult = g.n(.sum, concat)
                _ = g.n(.output(0), sumResult)

                let frameCount = 1
                let cResult = try CompilationPipeline.compile(
                        graph: g,
                        backend: .c,
                        options: .init(frameCount: frameCount, debug: true)
                )

                print("=== Concat via Pad+Add - Generated Source ===")
                print(cResult.source)

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

                injectTensorData(result: cResult, memory: mem.assumingMemoryBound(to: Float.self))

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

                print("=== Concat via Pad+Add Result ===")
                print("Output: \(output[0]), Expected: 10.0")

                // Sum of concatenated tensor [1, 2, 3, 4] = 10
                XCTAssertEqual(
                        output[0], 10.0, accuracy: 0.001,
                        "Concat via pad+add should give [1,2,3,4] with sum 10")
        }

        func testConcat2DViaPadAndAdd() throws {
                // Test: concat two 2x2 matrices along axis 0
                // [[1, 2], [3, 4]] concat [[5, 6], [7, 8]] -> [[1,2], [3,4], [5,6], [7,8]]
                let g = Graph()

                // Create two 2x2 tensors
                let t1 = g.tensor([[1.0, 2.0], [3.0, 4.0]])  // shape [2, 2]
                let t2 = g.tensor([[5.0, 6.0], [7.0, 8.0]])  // shape [2, 2]

                // Pad t1 with 0 top, 2 bottom (axis 0) -> shape [4, 2]
                let t1Padded = try g.pad(t1, padding: [(0, 2), (0, 0)])

                // Pad t2 with 2 top, 0 bottom (axis 0) -> shape [4, 2]
                let t2Padded = try g.pad(t2, padding: [(2, 0), (0, 0)])

                // Add them -> [[1,2], [3,4], [5,6], [7,8]]
                let concat = g.n(.add, t1Padded, t2Padded)

                // Sum to verify: 1+2+3+4+5+6+7+8 = 36
                let sumResult = g.n(.sum, concat)
                _ = g.n(.output(0), sumResult)

                let frameCount = 1
                let cResult = try CompilationPipeline.compile(
                        graph: g,
                        backend: .c,
                        options: .init(frameCount: frameCount, debug: true)
                )

                print("=== 2D Concat via Pad+Add - Generated Source ===")
                print(cResult.source)

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

                injectTensorData(result: cResult, memory: mem.assumingMemoryBound(to: Float.self))

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

                print("=== 2D Concat via Pad+Add Result ===")
                print("Output: \(output[0]), Expected: 36.0")

                XCTAssertEqual(
                        output[0], 36.0, accuracy: 0.001,
                        "2D concat via pad+add should sum to 36")
        }

        // MARK: - Tensor + Phasor Tests

        /// Test: phasor() with a tensor of frequencies
        /// Goal: let freqs = tensor(shape: [2,2]) containing frequencies
        ///       let sigs = phasor(freqs) -> should produce a tensor of phasors
        ///       let res = sum(sigs) -> sum of all phasor values
        func testPhasorWithTensorFrequencies() throws {
                let g = Graph()

                // Create a 2x2 tensor of frequencies
                // Each cell will have a different frequency
                let freqs = g.tensor(shape: [2, 2], data: [100.0, 200.0, 300.0, 400.0])

                // Try to create a phasor with the tensor as input
                // This is what we WANT to work: phasor(tensor) -> tensor of phasors
                let cellId = g.alloc()
                let zeroReset = g.n(.constant(0.0))
                let phasorNode = g.n(.phasor(cellId), freqs, zeroReset)

                // Sum all the phasor values
                let result = g.n(.sum, phasorNode)
                _ = g.n(.output(0), result)

                // Try to compile - this should either:
                // 1. Work correctly (if tensor phasors are supported)
                // 2. Fail with a meaningful error
                // 3. Compile but produce incorrect results
                do {
                        let cResult = try CompilationPipeline.compile(
                                graph: g,
                                backend: .c,
                                options: .init(frameCount: 100, debug: true)
                        )

                        print("=== Phasor With Tensor - Generated Source ===")
                        print(cResult.source)

                        // Check what shape the phasor node got
                        if let phasorShape = g.nodes[phasorNode]?.shape {
                                print("Phasor node shape: \(phasorShape)")
                        }

                        // If it compiled, try to run it
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

                        injectTensorData(result: cResult, memory: mem.assumingMemoryBound(to: Float.self))

                        let frameCount = 100
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

                        print("=== Phasor With Tensor Result ===")
                        print("First 10 outputs: \(Array(output.prefix(10)))")

                        // For 4 phasors at 100, 200, 300, 400 Hz:
                        // At 44100 sample rate, increments are:
                        // 100/44100 ≈ 0.00227, 200/44100 ≈ 0.00454, etc.
                        // Sum should increase over time as phasors ramp
                        XCTAssertGreaterThan(output[50], output[0], "Phasors should accumulate")

                } catch {
                        print("=== Phasor With Tensor - Compilation Failed ===")
                        print("Error: \(error)")
                        // This might be expected if tensor phasors aren't supported yet
                        XCTFail("Tensor phasor should compile: \(error)")
                }
        }

        /// Alternative approach: stack individual phasors into a tensor
        /// DEPRECATED: Use testPhasorWithTensorFrequencies instead - phasor now natively supports tensor inputs.
        /// This test is kept to verify compilation works, but has known timing issues.
        /// The stacking approach reads phasor values before they're updated in the same frame.
        func testStackedPhasors() throws {
                let g = Graph()

                // Create 4 individual phasors with different frequencies
                let freq1 = g.n(.constant(100.0))
                let freq2 = g.n(.constant(200.0))
                let freq3 = g.n(.constant(300.0))
                let freq4 = g.n(.constant(400.0))

                let cell1 = g.alloc()
                let cell2 = g.alloc()
                let cell3 = g.alloc()
                let cell4 = g.alloc()

                let zero = g.n(.constant(0.0))
                let phasor1 = g.n(.phasor(cell1), freq1, zero)
                let phasor2 = g.n(.phasor(cell2), freq2, zero)
                let phasor3 = g.n(.phasor(cell3), freq3, zero)
                let phasor4 = g.n(.phasor(cell4), freq4, zero)

                // Stack them into a [2,2] tensor
                let stacked = try g.stack([phasor1, phasor2, phasor3, phasor4], shape: [2, 2])

                // Sum all the phasor values
                let result = g.n(.sum, stacked)
                _ = g.n(.output(0), result)

                let frameCount = 100
                let cResult = try CompilationPipeline.compile(
                        graph: g,
                        backend: .c,
                        options: .init(frameCount: frameCount, debug: true)
                )

                print("=== Stacked Phasors - Generated Source ===")
                print(cResult.source)

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

                print("=== Stacked Phasors Result ===")
                print("First 10 outputs: \(Array(output.prefix(10)))")

                // This test verifies compilation succeeds - the output timing is known to be broken.
                // Use testPhasorWithTensorFrequencies for proper tensor phasor behavior.
                XCTAssertFalse(cResult.source.isEmpty, "Stacked phasors should compile")
        }

        // MARK: - Tensor Accum Tests

        /// Test: accum() with tensor inputs
        /// Each element in the tensor accumulates independently
        func testAccumWithTensorInputs() throws {
                let g = Graph()

                // Create a 2x2 tensor of increment values
                let increments = g.tensor(shape: [2, 2], data: [0.1, 0.2, 0.3, 0.4])

                // Create scalar inputs for reset, min, max (will broadcast)
                let reset = g.n(.constant(0.0))
                let min = g.n(.constant(0.0))
                let max = g.n(.constant(10.0))

                // Create tensor accum
                let cellId = g.alloc()
                let accumNode = g.n(.accum(cellId), increments, reset, min, max)

                // Sum all accumulator values
                let result = g.n(.sum, accumNode)
                _ = g.n(.output(0), result)

                do {
                        let cResult = try CompilationPipeline.compile(
                                graph: g,
                                backend: .c,
                                options: .init(frameCount: 100, debug: true)
                        )

                        print("=== Tensor Accum - Generated Source ===")
                        print(cResult.source)

                        if let accumShape = g.nodes[accumNode]?.shape {
                                print("Accum node shape: \(accumShape)")
                        }

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

                        injectTensorData(result: cResult, memory: mem.assumingMemoryBound(to: Float.self))

                        let frameCount = 100
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

                        print("=== Tensor Accum Result ===")
                        print("First 10 outputs: \(Array(output.prefix(10)))")

                        // With increments [0.1, 0.2, 0.3, 0.4], sum of increments = 1.0 per frame
                        // After N frames, sum should be approximately N * 1.0
                        // (accounting for accumulator returning previous value)
                        XCTAssertGreaterThan(output[50], output[10], "Accum should accumulate over time")

                } catch {
                        print("=== Tensor Accum - Compilation Failed ===")
                        print("Error: \(error)")
                        XCTFail("Tensor accum should compile: \(error)")
                }
        }

        // MARK: - Tensor Latch Tests

        /// Test: latch() with tensor inputs
        /// Each element latches independently based on its condition
        func testLatchWithTensorInputs() throws {
                let g = Graph()

                // Create a 2x2 tensor of values to latch
                let values = g.tensor(shape: [2, 2], data: [1.0, 2.0, 3.0, 4.0])

                // Always trigger (constant 1.0) - this will latch immediately
                let trigger = g.n(.constant(1.0))

                // Create tensor latch
                let latchCell = g.alloc()
                let latchNode = g.n(.latch(latchCell), values, trigger)

                // Sum all latched values
                let result = g.n(.sum, latchNode)
                _ = g.n(.output(0), result)

                do {
                        let cResult = try CompilationPipeline.compile(
                                graph: g,
                                backend: .c,
                                options: .init(frameCount: 10, debug: true)
                        )

                        print("=== Tensor Latch - Generated Source ===")
                        print(cResult.source)

                        if let latchShape = g.nodes[latchNode]?.shape {
                                print("Latch node shape: \(latchShape)")
                        }

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

                        injectTensorData(result: cResult, memory: mem.assumingMemoryBound(to: Float.self))

                        let frameCount = 10
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

                        print("=== Tensor Latch Result ===")
                        print("Outputs: \(output)")

                        // With constant trigger=1, latch should capture values immediately
                        // Frame 0: returns old latched value (0), stores new value (1+2+3+4)
                        // Frame 1+: returns 10 (the sum of latched values)
                        // So output[1] should be 10.0
                        XCTAssertEqual(output[1], 10.0, accuracy: 0.001, "Latch should capture tensor values")

                } catch {
                        print("=== Tensor Latch - Compilation Failed ===")
                        print("Error: \(error)")
                        XCTFail("Tensor latch should compile: \(error)")
                }
        }

        // MARK: - Phasor + Cos Test (Static issue investigation)

        /// Test: cos(phasor(tensor) * twopi) with larger tensor and wider frequency range
        /// Testing for potential aliasing or memory issues
        func testCosPhasorTensorLarge() throws {
                let g = Graph()

                // Create a 4x4 tensor of frequencies - wider range to test edge cases
                let freqs = g.tensor(shape: [4, 4], data: [
                        50.0, 100.0, 150.0, 200.0,
                        250.0, 300.0, 350.0, 400.0,
                        450.0, 500.0, 600.0, 700.0,
                        800.0, 1000.0, 1500.0, 2000.0
                ])

                // Create phasor with tensor input
                let cellId = g.alloc()
                let zeroReset = g.n(.constant(0.0))
                let phasorNode = g.n(.phasor(cellId), freqs, zeroReset)

                // Multiply by 2*pi
                let twopi = g.n(.constant(Float.pi * 2.0))
                let scaled = g.n(.mul, phasorNode, twopi)

                // Apply cos
                let cosNode = g.n(.cos, scaled)

                // Sum all the cos values
                let result = g.n(.sum, cosNode)
                _ = g.n(.output(0), result)

                do {
                        let cResult = try CompilationPipeline.compile(
                                graph: g,
                                backend: .c,
                                options: .init(frameCount: 4096, debug: true)  // 100ms at 44100Hz
                        )

                        print("=== Cos(Phasor(Tensor)) Large - Generated Source ===")
                        print(cResult.source)

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

                        injectTensorData(result: cResult, memory: mem.assumingMemoryBound(to: Float.self))

                        let frameCount = 4096
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

                        print("=== Cos(Phasor(Tensor)) Large Result ===")
                        print("First 20 outputs: \(Array(output.prefix(20)))")
                        print("Last 20 outputs: \(Array(output.suffix(20)))")

                        // Check for NaN/Inf
                        let hasNaN = output.contains { $0.isNaN }
                        let hasInf = output.contains { $0.isInfinite }
                        XCTAssertFalse(hasNaN, "Output should not contain NaN")
                        XCTAssertFalse(hasInf, "Output should not contain Inf")

                        // cos() output should be bounded (sum of 16 cosines: -16 to 16)
                        let minVal = output.min() ?? 0
                        let maxVal = output.max() ?? 0
                        print("Min: \(minVal), Max: \(maxVal)")
                        XCTAssertGreaterThanOrEqual(minVal, -16.1, "Sum of 16 cos values should be >= -16")
                        XCTAssertLessThanOrEqual(maxVal, 16.1, "Sum of 16 cos values should be <= 16")

                        // Check smoothness - max delta between consecutive samples
                        var maxDelta: Float = 0
                        for i in 1..<output.count {
                                let delta = abs(output[i] - output[i-1])
                                maxDelta = max(maxDelta, delta)
                        }
                        print("Max delta between consecutive samples: \(maxDelta)")

                        // With 16 summed cosines at various frequencies, max delta should still be reasonable
                        XCTAssertLessThan(maxDelta, 8.0, "Output should be smooth, not static (max delta: \(maxDelta))")

                } catch {
                        print("=== Cos(Phasor(Tensor)) Large - Compilation Failed ===")
                        print("Error: \(error)")
                        XCTFail("cos(phasor(tensor)) large should compile: \(error)")
                }
        }

        /// Test: cos(phasor(tensor) * twopi)
        /// This is causing "mad static" in practice - let's see what's happening
        func testCosPhasorTensor() throws {
                let g = Graph()

                // Create a 2x2 tensor of frequencies
                let freqs = g.tensor(shape: [2, 2], data: [100.0, 200.0, 300.0, 400.0])

                // Create phasor with tensor input
                let cellId = g.alloc()
                let zeroReset = g.n(.constant(0.0))
                let phasorNode = g.n(.phasor(cellId), freqs, zeroReset)

                // Multiply by 2*pi (twopi ≈ 6.28318)
                let twopi = g.n(.constant(Float.pi * 2.0))
                let scaled = g.n(.mul, phasorNode, twopi)

                // Apply cos
                let cosNode = g.n(.cos, scaled)

                // Sum all the cos values
                let result = g.n(.sum, cosNode)
                _ = g.n(.output(0), result)

                do {
                        let cResult = try CompilationPipeline.compile(
                                graph: g,
                                backend: .c,
                                options: .init(frameCount: 441, debug: true)  // ~10ms at 44100Hz
                        )

                        print("=== Cos(Phasor(Tensor)) - Generated Source ===")
                        print(cResult.source)

                        // Check shapes
                        if let phasorShape = g.nodes[phasorNode]?.shape {
                                print("Phasor node shape: \(phasorShape)")
                        }
                        if let scaledShape = g.nodes[scaled]?.shape {
                                print("Scaled (phasor*twopi) shape: \(scaledShape)")
                        }
                        if let cosShape = g.nodes[cosNode]?.shape {
                                print("Cos node shape: \(cosShape)")
                        }

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

                        injectTensorData(result: cResult, memory: mem.assumingMemoryBound(to: Float.self))

                        let frameCount = 441
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

                        print("=== Cos(Phasor(Tensor)) Result ===")
                        print("First 20 outputs: \(Array(output.prefix(20)))")
                        print("Last 20 outputs: \(Array(output.suffix(20)))")

                        // Check for NaN/Inf (would cause static)
                        let hasNaN = output.contains { $0.isNaN }
                        let hasInf = output.contains { $0.isInfinite }
                        XCTAssertFalse(hasNaN, "Output should not contain NaN")
                        XCTAssertFalse(hasInf, "Output should not contain Inf")

                        // cos() output should be bounded between -4 and 4 (sum of 4 cosines)
                        let minVal = output.min() ?? 0
                        let maxVal = output.max() ?? 0
                        print("Min: \(minVal), Max: \(maxVal)")
                        XCTAssertGreaterThanOrEqual(minVal, -4.1, "Sum of 4 cos values should be >= -4")
                        XCTAssertLessThanOrEqual(maxVal, 4.1, "Sum of 4 cos values should be <= 4")

                        // The output should be smooth oscillations, not random static
                        // Check that consecutive samples don't jump wildly
                        var maxDelta: Float = 0
                        for i in 1..<output.count {
                                let delta = abs(output[i] - output[i-1])
                                maxDelta = max(maxDelta, delta)
                        }
                        print("Max delta between consecutive samples: \(maxDelta)")

                        // For smooth audio, delta shouldn't be huge
                        // At 100-400Hz, with 4 summed cosines, max reasonable delta per sample is ~0.5
                        XCTAssertLessThan(maxDelta, 2.0, "Output should be smooth, not static (max delta: \(maxDelta))")

                } catch {
                        print("=== Cos(Phasor(Tensor)) - Compilation Failed ===")
                        print("Error: \(error)")
                        XCTFail("cos(phasor(tensor)) should compile: \(error)")
                }
        }

        // MARK: - Cell Allocation Inspection Tests

        /// Test that cell mappings for tensor phasor don't overlap
        /// This verifies that:
        /// 1. Input tensor data has its own memory region
        /// 2. Phasor state has its own memory region (expanded to tensor size)
        /// 3. Output tensor has its own memory region
        /// 4. No regions overlap
        func testTensorPhasorCellMappingsNoOverlap() throws {
                let g = Graph()

                // Create a 2x2 tensor of frequencies (4 elements)
                let freqs = g.tensor(shape: [2, 2], data: [100.0, 200.0, 300.0, 400.0])

                // Get the tensor's cellId for the frequency data
                guard let freqTensorId = g.nodeToTensor[freqs],
                      let freqTensor = g.tensors[freqTensorId] else {
                        XCTFail("Frequency tensor not found")
                        return
                }
                let freqCellId = freqTensor.cellId
                print("Frequency tensor cellId: \(freqCellId), shape: \(freqTensor.shape)")

                // Create phasor with tensor input
                let phasorStateCellId = g.alloc()
                print("Phasor state cellId (before compilation): \(phasorStateCellId)")

                let zeroReset = g.n(.constant(0.0))
                let phasorNode = g.n(.phasor(phasorStateCellId), freqs, zeroReset)

                // Sum to output
                let result = g.n(.sum, phasorNode)
                _ = g.n(.output(0), result)

                // Compile
                let cResult = try CompilationPipeline.compile(
                        graph: g,
                        backend: .c,
                        options: .init(frameCount: 100, debug: true)
                )

                // Inspect cell allocations
                let cellMappings = cResult.cellAllocations.cellMappings
                let totalSlots = cResult.totalMemorySlots

                print("\n=== Cell Allocation Inspection ===")
                print("Total memory slots: \(totalSlots)")
                print("Cell mappings: \(cellMappings.sorted(by: { $0.key < $1.key }))")

                // Check what the frequency tensor's cell maps to
                let mappedFreqCell = cellMappings[freqCellId]
                print("Frequency tensor cell \(freqCellId) -> mapped to: \(mappedFreqCell ?? -1)")

                // Check what the phasor state cell maps to
                let mappedPhasorStateCell = cellMappings[phasorStateCellId]
                print("Phasor state cell \(phasorStateCellId) -> mapped to: \(mappedPhasorStateCell ?? -1)")

                // Get the phasor output tensor's cell
                guard let phasorOutputTensorId = g.nodeToTensor[phasorNode],
                      let phasorOutputTensor = g.tensors[phasorOutputTensorId] else {
                        XCTFail("Phasor output tensor not found")
                        return
                }
                let phasorOutputCellId = phasorOutputTensor.cellId
                let mappedPhasorOutputCell = cellMappings[phasorOutputCellId]
                print("Phasor output tensor cell \(phasorOutputCellId) -> mapped to: \(mappedPhasorOutputCell ?? -1)")

                // Check cell allocation sizes
                print("\nCell allocation sizes in graph:")
                print("  Freq tensor cell \(freqCellId): \(g.cellAllocationSizes[freqCellId] ?? 1)")
                print("  Phasor state cell \(phasorStateCellId): \(g.cellAllocationSizes[phasorStateCellId] ?? 1)")
                print("  Phasor output cell \(phasorOutputCellId): \(g.cellAllocationSizes[phasorOutputCellId] ?? 1)")

                // Verify no overlap
                // Each cell should map to a different memory region
                // For a [2,2] tensor, each region needs 4 slots

                guard let freqStart = mappedFreqCell,
                      let stateStart = mappedPhasorStateCell,
                      let outputStart = mappedPhasorOutputCell else {
                        XCTFail("Some cells were not mapped")
                        return
                }

                let tensorSize = 4  // [2,2] = 4 elements
                let freqRange = freqStart..<(freqStart + tensorSize)
                let stateRange = stateStart..<(stateStart + tensorSize)
                let outputRange = outputStart..<(outputStart + tensorSize)

                print("\nMemory ranges:")
                print("  Freq tensor: \(freqRange)")
                print("  Phasor state: \(stateRange)")
                print("  Phasor output: \(outputRange)")

                // Check for overlaps
                let freqStateOverlap = freqRange.overlaps(stateRange)
                let freqOutputOverlap = freqRange.overlaps(outputRange)
                let stateOutputOverlap = stateRange.overlaps(outputRange)

                XCTAssertFalse(freqStateOverlap, "Frequency tensor and phasor state should not overlap! freq:\(freqRange) state:\(stateRange)")
                XCTAssertFalse(freqOutputOverlap, "Frequency tensor and phasor output should not overlap! freq:\(freqRange) output:\(outputRange)")
                XCTAssertFalse(stateOutputOverlap, "Phasor state and phasor output should not overlap! state:\(stateRange) output:\(outputRange)")

                // All ranges should fit within total slots
                XCTAssertLessThanOrEqual(freqRange.upperBound, totalSlots, "Freq range exceeds total slots")
                XCTAssertLessThanOrEqual(stateRange.upperBound, totalSlots, "State range exceeds total slots")
                XCTAssertLessThanOrEqual(outputRange.upperBound, totalSlots, "Output range exceeds total slots")

                print("\n=== All cell mappings verified - no overlaps ===")
        }

        /// Test cell mappings with multiple tensor phasors to catch potential conflicts
        func testMultipleTensorPhasorsCellMappings() throws {
                let g = Graph()

                // Create two different frequency tensors
                let freqs1 = g.tensor(shape: [2, 2], data: [100.0, 200.0, 300.0, 400.0])
                let freqs2 = g.tensor(shape: [2, 2], data: [500.0, 600.0, 700.0, 800.0])

                // Create two phasors with separate state cells
                let phasorCell1 = g.alloc()
                let phasorCell2 = g.alloc()

                let zero = g.n(.constant(0.0))
                let phasor1 = g.n(.phasor(phasorCell1), freqs1, zero)
                let phasor2 = g.n(.phasor(phasorCell2), freqs2, zero)

                // Add them together and sum
                let added = g.n(.add, phasor1, phasor2)
                let result = g.n(.sum, added)
                _ = g.n(.output(0), result)

                // Compile
                let cResult = try CompilationPipeline.compile(
                        graph: g,
                        backend: .c,
                        options: .init(frameCount: 100, debug: true)
                )

                let cellMappings = cResult.cellAllocations.cellMappings
                let totalSlots = cResult.totalMemorySlots

                print("\n=== Multiple Tensor Phasors Cell Inspection ===")
                print("Total memory slots: \(totalSlots)")
                print("All cell mappings: \(cellMappings.sorted(by: { $0.key < $1.key }))")

                // Collect all memory ranges
                var allRanges: [(String, Range<Int>)] = []
                let tensorSize = 4

                // Helper to add a range
                func addRange(name: String, cellId: Int) {
                        if let mapped = cellMappings[cellId] {
                                let size = g.cellAllocationSizes[cellId] ?? 1
                                let range = mapped..<(mapped + max(size, tensorSize))
                                allRanges.append((name, range))
                                print("  \(name): cell \(cellId) -> \(range) (size: \(size))")
                        }
                }

                // Get tensor cellIds
                if let t1 = g.nodeToTensor[freqs1], let tensor1 = g.tensors[t1] {
                        addRange(name: "freqs1 data", cellId: tensor1.cellId)
                }
                if let t2 = g.nodeToTensor[freqs2], let tensor2 = g.tensors[t2] {
                        addRange(name: "freqs2 data", cellId: tensor2.cellId)
                }

                addRange(name: "phasor1 state", cellId: phasorCell1)
                addRange(name: "phasor2 state", cellId: phasorCell2)

                if let t1 = g.nodeToTensor[phasor1], let tensor1 = g.tensors[t1] {
                        addRange(name: "phasor1 output", cellId: tensor1.cellId)
                }
                if let t2 = g.nodeToTensor[phasor2], let tensor2 = g.tensors[t2] {
                        addRange(name: "phasor2 output", cellId: tensor2.cellId)
                }
                if let t = g.nodeToTensor[added], let tensor = g.tensors[t] {
                        addRange(name: "add output", cellId: tensor.cellId)
                }

                // Check all pairs for overlaps
                print("\nChecking for overlaps...")
                var hasOverlap = false
                for i in 0..<allRanges.count {
                        for j in (i+1)..<allRanges.count {
                                let (name1, range1) = allRanges[i]
                                let (name2, range2) = allRanges[j]
                                if range1.overlaps(range2) {
                                        print("  OVERLAP: \(name1) \(range1) overlaps with \(name2) \(range2)")
                                        hasOverlap = true
                                }
                        }
                }

                XCTAssertFalse(hasOverlap, "Found overlapping memory regions!")

                if !hasOverlap {
                        print("  No overlaps found - all memory regions are separate")
                }

                // Verify total slots is sufficient
                let maxUsedSlot = allRanges.map { $0.1.upperBound }.max() ?? 0
                XCTAssertLessThanOrEqual(maxUsedSlot, totalSlots, "Used slots exceed total allocated")
                print("\nMax used slot: \(maxUsedSlot), Total allocated: \(totalSlots)")
        }

        // MARK: - Peek with Frame-Based Tensor Tests

        /// Simple test: peek on a static tensor (should work with existing implementation)
        func testPeekOnStaticTensor() throws {
                let g = Graph()

                // Create a simple 2D tensor
                let data = g.tensor(shape: [3, 1], data: [1.0, 2.0, 3.0])

                let zero = g.n(.constant(0.0))
                let peekResult = try g.peek(tensor: data, index: zero, channel: zero)

                _ = g.n(.output(0), peekResult)

                let cResult = try CompilationPipeline.compile(
                        graph: g,
                        backend: .c,
                        options: .init(frameCount: 10, debug: true)
                )

                print("=== Simple Peek Test - Generated Source ===")
                print(cResult.source)

                XCTAssertFalse(cResult.source.isEmpty)
        }

        /// Test: peek on a frame-based tensor (phasor with tensor input)
        /// This demonstrates that peek NOW properly handles frame-based tensors via lazy evaluation.
        func testPeekOnPhasorTensor() throws {
                let g = Graph()

                // Create a [3,1] tensor of frequencies
                let freqs = g.tensor(shape: [3, 1], data: [100.0, 200.0, 300.0])

                // Create phasor with tensor input - this produces a frame-based tensor
                let phasorCell = g.alloc()
                let zeroReset = g.n(.constant(0.0))
                let phasorTensor = g.n(.phasor(phasorCell), freqs, zeroReset)

                // Peek the first element - this now uses the lazy .peek operation
                let zero = g.n(.constant(0.0))
                let peekResult = try g.peek(tensor: phasorTensor, index: zero, channel: zero)

                // Just output peek result (single output)
                _ = g.n(.output(0), peekResult)

                let frameCount = 100
                let cResult = try CompilationPipeline.compile(
                        graph: g,
                        backend: .c,
                        options: .init(frameCount: frameCount, debug: true)
                )

                print("=== Peek on Phasor Tensor - Generated Source ===")
                print(cResult.source)

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

                injectTensorData(result: cResult, memory: mem.assumingMemoryBound(to: Float.self))

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

                print("=== Peek on Phasor Tensor Results ===")
                print("First 10 outputs: \(Array(output.prefix(10)))")
                print("Outputs 40-50: \(Array(output[40..<50]))")

                // Peek should increase over time as phasor accumulates phase
                XCTAssertGreaterThan(output[50], output[0], "Peek should increase (frame-based tensor)")
                XCTAssertGreaterThan(output[10], 0, "Peek should return non-zero from phasor tensor")
        }

        // MARK: - Static to FrameBased Tests

        func testStaticMatmulIntoFrameBasedPeek() throws {
                // Test that static tensor operations (matmul) correctly flow into frame-based operations
                // This verifies that:
                // 1. Static blocks compute correctly (matmul on constant data)
                // 2. Static -> frameBased boundary works (result read by frame-based op)
                // 3. Tape buffer is properly included for static blocks
                // 4. defineGlobal/loadGlobal don't cause duplicate variable definitions
                let g = Graph()

                // Static: matmul [1,4] x [4,1] -> [1,1] = scalar
                // Result: 1*0.5 + 2*0.5 + 3*0.5 + 4*0.5 = 5.0
                let weights = g.tensor(shape: [1, 4], data: [1.0, 2.0, 3.0, 4.0])
                let bias = g.tensor(shape: [4, 1], data: [0.5, 0.5, 0.5, 0.5])
                let matmulResult = try g.matmul(weights, bias)  // [1, 1] = [[5.0]]

                // Sum to get scalar (still static)
                let summed = g.n(.sum, matmulResult)  // 5.0

                // Frame-based: use audio input to modulate the static result
                // This forces a static -> frameBased boundary
                let inputNode = g.n(.input(0))  // Frame-varying input
                let one = g.n(.constant(1.0))
                let scaledInput = g.n(.add, inputNode, one)  // input + 1 (so we get non-zero even with zero input)

                // Multiply static matmul result by frame-based input
                let output = g.n(.mul, summed, scaledInput)  // 5.0 * (input + 1)

                _ = g.n(.output(0), output)

                let frameCount = 64

                // Compile with C backend
                let cResult = try CompilationPipeline.compile(
                        graph: g,
                        backend: .c,
                        options: .init(frameCount: frameCount, debug: true)
                )

                print("=== Static Matmul into FrameBased - Generated Source ===")
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

                injectTensorData(result: cResult, memory: mem.assumingMemoryBound(to: Float.self))

                var output_ = [Float](repeating: 0, count: frameCount)
                // Input that varies: 0, 0.1, 0.2, ... (representing audio samples)
                var input = (0..<frameCount).map { Float($0) * 0.1 }

                output_.withUnsafeMutableBufferPointer { outPtr in
                        input.withUnsafeMutableBufferPointer { inPtr in
                                cRuntime.runWithMemory(
                                        outputs: outPtr.baseAddress!,
                                        inputs: inPtr.baseAddress!,
                                        memory: mem,
                                        frameCount: frameCount
                                )
                        }
                }

                print("=== Static -> FrameBased Results ===")
                print("First 10 outputs: \(Array(output_.prefix(10)))")
                print("Input was: \(Array(input.prefix(10)))")

                // At frame 0: output = 5.0 * (0.0 + 1.0) = 5.0
                XCTAssertEqual(output_[0], 5.0, accuracy: 0.01, "Frame 0: 5.0 * (0+1) = 5.0")

                // At frame 10: output = 5.0 * (1.0 + 1.0) = 10.0
                XCTAssertEqual(output_[10], 10.0, accuracy: 0.01, "Frame 10: 5.0 * (1.0+1) = 10.0")

                // At frame 20: output = 5.0 * (2.0 + 1.0) = 15.0
                XCTAssertEqual(output_[20], 15.0, accuracy: 0.01, "Frame 20: 5.0 * (2.0+1) = 15.0")

                // Verify output increases (frame-based modulation working)
                XCTAssertGreaterThan(output_[10], output_[0], "Output should increase with input")
        }

}
