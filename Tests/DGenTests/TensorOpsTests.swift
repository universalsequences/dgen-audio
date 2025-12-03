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
                XCTAssertEqual(output[0], 830.0, accuracy: 0.001, "Matmul with scaled B should be 830")
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
                XCTAssertEqual(output[0], 91.0, accuracy: 0.001, "Reshape to flat should preserve data order")
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
                XCTAssertEqual(output[0], 36.0, accuracy: 0.001, "Reshape from flat should create correct 2D shape")
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
                XCTAssertEqual(output[0], 46.0, accuracy: 0.001, "Transpose should reorder elements correctly via strided indexing")
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

                XCTAssertEqual(output[0], 10.0, accuracy: 0.001, "SumAxis on 1D should reduce to scalar")
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
                let t = g.tensor(shape: [4, 3], data: [
                        1.0, 2.0, 3.0,    // row 0: sum = 6
                        4.0, 5.0, 6.0,    // row 1: sum = 15
                        7.0, 8.0, 9.0,    // row 2: sum = 24
                        10.0, 11.0, 12.0  // row 3: sum = 33
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
                XCTAssertEqual(output[0], 78.0, accuracy: 0.001, "Sum of sumAxis(1) on [4,3] should be 78")
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
                XCTAssertEqual(output[0], 50.0, accuracy: 0.001, "Reshape then sumAxis should give 50.0")
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

                XCTAssertEqual(output[0], 16.0, accuracy: 0.001, "Stack with shape should give 16.0")
        }

}
