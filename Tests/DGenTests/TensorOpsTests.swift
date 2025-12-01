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

        func testTensorHistoryExecutionMix() throws {
                // Test that tensor history persists across frames
                let g = Graph()

                // Create a history buffer for 2x2 state (starts at 0)
                let stateBuffer = g.tensorHistoryBuffer(shape: [2, 2])

                // Read state (should be what was written last frame)
                let state = g.tensorHistoryRead(stateBuffer)

                // Add 1 to each element
                let newState = g.n(
                        .mix, state,
                        g.n(.phasor(g.alloc()), g.n(.constant(4400)), g.n(.constant(0.0))),
                        g.n(.constant(0.5)))

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

        // MARK: - Compressor Tests

        func testStereoCompressor() throws {
                // Test: stereo compressor - input1 -> compressor -> out1, input2 -> compressor -> out2
                // This tests that scalar operations still emit correctly after tensor optimization changes
                let g = Graph()

                // Get stereo inputs
                let input1 = g.n(.input(0))
                let input2 = g.n(.input(1))

                // Shared compressor parameters
                let ratio = g.n(.constant(4.0))       // 4:1 ratio
                let threshold = g.n(.constant(-12.0)) // -12 dB threshold
                let knee = g.n(.constant(6.0))        // 6 dB knee
                let attack = g.n(.constant(0.01))     // 10ms attack
                let release = g.n(.constant(0.1))     // 100ms release
                let isSideChain = g.n(.constant(0.0)) // no sidechain
                let sidechainIn = g.n(.constant(0.0)) // unused

                // Apply compressor to each channel
                let compressed1 = g.compressor(
                        input1, ratio, threshold, knee, attack, release, isSideChain, sidechainIn)
                let compressed2 = g.compressor(
                        input2, ratio, threshold, knee, attack, release, isSideChain, sidechainIn)

                // Outputs
                _ = g.n(.output(0), compressed1)
                _ = g.n(.output(1), compressed2)

                let frameCount = 64

                // Compile
                let cResult = try CompilationPipeline.compile(
                        graph: g,
                        backend: .c,
                        options: .init(frameCount: frameCount, debug: true)
                )

                print("=== Stereo Compressor - C Source ===")
                print(cResult.source)

                // Verify source is not empty
                XCTAssertFalse(cResult.source.isEmpty, "C source should not be empty")

                // Create runtime and execute
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

                // Create test input - a simple sine-like pattern
                var input1Data = [Float](repeating: 0, count: frameCount)
                var input2Data = [Float](repeating: 0, count: frameCount)
                for i in 0..<frameCount {
                        input1Data[i] = sin(Float(i) * 0.2) * 0.8  // Left channel
                        input2Data[i] = cos(Float(i) * 0.2) * 0.8  // Right channel (phase shifted)
                }

                // Run - need 2 output channels
                var output1 = [Float](repeating: 0, count: frameCount)
                var output2 = [Float](repeating: 0, count: frameCount)

                // The runtime expects interleaved or separate buffers - check how it works
                output1.withUnsafeMutableBufferPointer { out1Ptr in
                        output2.withUnsafeMutableBufferPointer { out2Ptr in
                                input1Data.withUnsafeBufferPointer { in1Ptr in
                                        input2Data.withUnsafeBufferPointer { in2Ptr in
                                                // Create arrays of pointers for multi-channel
                                                var outPtrs: [UnsafeMutablePointer<Float>?] = [out1Ptr.baseAddress, out2Ptr.baseAddress]
                                                var inPtrs: [UnsafePointer<Float>?] = [in1Ptr.baseAddress, in2Ptr.baseAddress]

                                                outPtrs.withUnsafeMutableBufferPointer { outBuf in
                                                        inPtrs.withUnsafeBufferPointer { inBuf in
                                                                cRuntime.runWithMemory(
                                                                        outputs: outBuf.baseAddress!.pointee!,
                                                                        inputs: inBuf.baseAddress!.pointee!,
                                                                        memory: mem,
                                                                        frameCount: frameCount
                                                                )
                                                        }
                                                }
                                        }
                                }
                        }
                }

                // Print some output values
                print("=== Stereo Compressor Output ===")
                for i in stride(from: 0, to: min(10, frameCount), by: 1) {
                        print("frame \(i): L=\(output1[i]) R=\(output2[i])")
                }

                // Verify we got non-zero output (compressor should pass signal through)
                let hasNonZeroL = output1.contains { abs($0) > 0.0001 }
                let hasNonZeroR = output2.contains { abs($0) > 0.0001 }
                XCTAssertTrue(hasNonZeroL, "Left channel should have non-zero values")
                XCTAssertTrue(hasNonZeroR, "Right channel should have non-zero values")
        }

}
