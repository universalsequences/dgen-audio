import XCTest

@testable import DGen

final class MetalTensorOpsTests: XCTestCase {

    // MARK: - Sum Reduce Execution

    func testSumReduceExecution() throws {
        // Test that sum reduce actually computes correct values on Metal
        let g = Graph()

        // Create a 2x2 tensor with initial data
        let tensorNode = g.tensor(shape: [2, 2], data: [1.0, 2.0, 3.0, 4.0])

        // Sum to scalar
        let sumResult = g.n(.sum, tensorNode)
        _ = g.n(.output(0), sumResult)

        let frameCount = 1

        // Compile for Metal
        let mResult = try CompilationPipeline.compile(
            graph: g,
            backend: .metal,
            options: .init(frameCount: frameCount, debug: true)
        )

        print("=== Metal Sum Reduce - Kernels ===")
        for kernel in mResult.kernels {
            print("--- Kernel: \(kernel.name) kind=\(kernel.kind) ---")
            print(kernel.source)
        }

        // Create Metal runtime
        let metalRuntime = try MetalCompiledKernel(
            kernels: mResult.kernels,
            cellAllocations: mResult.cellAllocations,
            context: mResult.context,
            frameCount: frameCount
        )

        // Inject tensor data into Metal memory buffer
        if let memoryBuffer = metalRuntime.getBuffer(name: "memory") {
            let memPtr = memoryBuffer.contents().assumingMemoryBound(to: Float.self)
            injectTensorData(result: mResult, memory: memPtr)
        }

        // Run
        var output = [Float](repeating: 0, count: frameCount)
        let input = [Float](repeating: 0, count: frameCount)

        output.withUnsafeMutableBufferPointer { outPtr in
            input.withUnsafeBufferPointer { inPtr in
                metalRuntime.run(
                    outputs: outPtr.baseAddress!,
                    inputs: inPtr.baseAddress!,
                    frameCount: frameCount
                )
            }
        }

        // Verify: 1 + 2 + 3 + 4 = 10
        print("Metal output: \(output)")
        XCTAssertEqual(output[0], 10.0, accuracy: 1e-5, "Sum should be 10.0")
    }

    func testTensorAddScalarExecution() throws {
        // Test tensor + scalar with actual Metal execution
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

        // Compile for Metal
        let mResult = try CompilationPipeline.compile(
            graph: g,
            backend: .metal,
            options: .init(frameCount: frameCount, debug: true)
        )

        print("=== Metal Tensor Add Scalar - Kernels ===")
        for kernel in mResult.kernels {
            print("--- Kernel: \(kernel.name) kind=\(kernel.kind) ---")
            print(kernel.source)
        }

        // Create Metal runtime
        let metalRuntime = try MetalCompiledKernel(
            kernels: mResult.kernels,
            cellAllocations: mResult.cellAllocations,
            context: mResult.context,
            frameCount: frameCount
        )

        // Inject tensor data
        if let memoryBuffer = metalRuntime.getBuffer(name: "memory") {
            let memPtr = memoryBuffer.contents().assumingMemoryBound(to: Float.self)
            injectTensorData(result: mResult, memory: memPtr)
        }

        // Run
        var output = [Float](repeating: 0, count: frameCount)
        let input = [Float](repeating: 0, count: frameCount)

        output.withUnsafeMutableBufferPointer { outPtr in
            input.withUnsafeBufferPointer { inPtr in
                metalRuntime.run(
                    outputs: outPtr.baseAddress!,
                    inputs: inPtr.baseAddress!,
                    frameCount: frameCount
                )
            }
        }

        // Expected: (1+5) + (2+5) + (3+5) + (4+5) = 6 + 7 + 8 + 9 = 30
        print("Metal output: \(output)")
        XCTAssertEqual(output[0], 30.0, accuracy: 1e-5, "Sum of (tensor + 5) should be 30.0")
    }

    func testTensorHistoryExecution() throws {
        // Test that tensor history persists across frames on Metal
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

        // Compile for Metal
        let mResult = try CompilationPipeline.compile(
            graph: g,
            backend: .metal,
            options: .init(frameCount: frameCount, debug: true)
        )

        print("=== Metal Kernel Source ===")
        for kernel in mResult.kernels {
            print(kernel.source)
        }

        // Create Metal runtime
        let metalRuntime = try MetalCompiledKernel(
            kernels: mResult.kernels,
            cellAllocations: mResult.cellAllocations,
            context: mResult.context,
            frameCount: frameCount
        )

        // Run with frameCount=4 in single dispatch
        var output = [Float](repeating: 0, count: frameCount)
        let input = [Float](repeating: 0, count: frameCount)

        output.withUnsafeMutableBufferPointer { outPtr in
            input.withUnsafeBufferPointer { inPtr in
                metalRuntime.run(
                    outputs: outPtr.baseAddress!,
                    inputs: inPtr.baseAddress!,
                    frameCount: frameCount
                )
            }
        }
        print("Metal output: \(output)")

        // Frame 0: state=[0,0,0,0], newState=[1,1,1,1], sum=4
        // Frame 1: state=[1,1,1,1], newState=[2,2,2,2], sum=8
        // Frame 2: state=[2,2,2,2], newState=[3,3,3,3], sum=12
        // Frame 3: state=[3,3,3,3], newState=[4,4,4,4], sum=16
        XCTAssertEqual(output[0], 4.0, accuracy: 1e-5, "Frame 0 sum should be 4")
        XCTAssertEqual(output[1], 8.0, accuracy: 1e-5, "Frame 1 sum should be 8")
        XCTAssertEqual(output[2], 12.0, accuracy: 1e-5, "Frame 2 sum should be 12")
        XCTAssertEqual(output[3], 16.0, accuracy: 1e-5, "Frame 3 sum should be 16")
    }

    func testConv2dExecution() throws {
        // Test conv2d with a simple kernel on Metal
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

        // Compile for Metal
        let mResult = try CompilationPipeline.compile(
            graph: g,
            backend: .metal,
            options: .init(frameCount: frameCount, debug: true)
        )

        print("=== Metal Conv2d - Kernels ===")
        for kernel in mResult.kernels {
            print("--- Kernel: \(kernel.name) kind=\(kernel.kind) ---")
            print(kernel.source)
        }

        // Create Metal runtime
        let metalRuntime = try MetalCompiledKernel(
            kernels: mResult.kernels,
            cellAllocations: mResult.cellAllocations,
            context: mResult.context,
            frameCount: frameCount
        )

        // Inject tensor data
        if let memoryBuffer = metalRuntime.getBuffer(name: "memory") {
            let memPtr = memoryBuffer.contents().assumingMemoryBound(to: Float.self)
            injectTensorData(result: mResult, memory: memPtr)
        }

        // Run
        var output = [Float](repeating: 0, count: frameCount)
        let input = [Float](repeating: 0, count: frameCount)

        output.withUnsafeMutableBufferPointer { outPtr in
            input.withUnsafeBufferPointer { inPtr in
                metalRuntime.run(
                    outputs: outPtr.baseAddress!,
                    inputs: inPtr.baseAddress!,
                    frameCount: frameCount
                )
            }
        }

        // With identity kernel (center=1), conv output should equal input
        // Sum of 9 ones = 9.0
        print("Metal output: \(output)")
        XCTAssertEqual(output[0], 9.0, accuracy: 1e-4, "Conv2d with identity kernel should preserve sum")
    }
}
