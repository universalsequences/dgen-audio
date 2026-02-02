import XCTest
import Metal

/// Tests to verify Metal atomic operations behavior
/// These are raw Metal tests - no DGen involved
final class MetalAtomicTests: XCTestCase {

    /// Test 1: Simple atomic add - 100 threads each add 1.0
    /// Expected result: 100.0
    func testSimpleAtomicAdd() throws {
        let device = MTLCreateSystemDefaultDevice()!
        let commandQueue = device.makeCommandQueue()!

        let kernelSource = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void atomic_add_one(
            device float *memory [[buffer(0)]],
            uint id [[thread_position_in_grid]]
        ) {
            atomic_fetch_add_explicit(
                (device atomic<float>*)&memory[0],
                1.0f,
                memory_order_relaxed
            );
        }
        """

        let library = try device.makeLibrary(source: kernelSource, options: nil)
        let function = library.makeFunction(name: "atomic_add_one")!
        let pipeline = try device.makeComputePipelineState(function: function)

        let buffer = device.makeBuffer(length: MemoryLayout<Float>.size, options: .storageModeShared)!

        let threadCount = 100
        var results: [Float] = []

        print("\n=== Test: 100 threads each atomically add 1.0 ===")

        for run in 0..<20 {
            memset(buffer.contents(), 0, buffer.length)

            let commandBuffer = commandQueue.makeCommandBuffer()!
            let encoder = commandBuffer.makeComputeCommandEncoder()!
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(buffer, offset: 0, index: 0)

            let threadsPerGrid = MTLSize(width: threadCount, height: 1, depth: 1)
            let threadsPerGroup = MTLSize(width: min(threadCount, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
            encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerGroup)

            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()

            let result = buffer.contents().assumingMemoryBound(to: Float.self)[0]
            results.append(result)
            print("Run \(run): \(result)")
        }

        let allCorrect = results.allSatisfy { $0 == Float(threadCount) }
        print("\nExpected: \(threadCount).0")
        print("All correct: \(allCorrect)")

        XCTAssertTrue(allCorrect, "All runs should equal \(threadCount)")
    }

    /// Test 2: Each thread adds its thread ID (0, 1, 2, ... 99)
    /// Expected result: 0+1+2+...+99 = 4950
    func testAtomicAddThreadIds() throws {
        let device = MTLCreateSystemDefaultDevice()!
        let commandQueue = device.makeCommandQueue()!

        let kernelSource = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void atomic_add_id(
            device float *memory [[buffer(0)]],
            uint id [[thread_position_in_grid]]
        ) {
            atomic_fetch_add_explicit(
                (device atomic<float>*)&memory[0],
                (float)id,
                memory_order_relaxed
            );
        }
        """

        let library = try device.makeLibrary(source: kernelSource, options: nil)
        let function = library.makeFunction(name: "atomic_add_id")!
        let pipeline = try device.makeComputePipelineState(function: function)

        let buffer = device.makeBuffer(length: MemoryLayout<Float>.size, options: .storageModeShared)!

        let threadCount = 100
        let expected = Float(threadCount * (threadCount - 1) / 2)  // Sum of 0..99
        var results: [Float] = []

        print("\n=== Test: 100 threads add their thread ID (0-99) ===")

        for run in 0..<20 {
            memset(buffer.contents(), 0, buffer.length)

            let commandBuffer = commandQueue.makeCommandBuffer()!
            let encoder = commandBuffer.makeComputeCommandEncoder()!
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(buffer, offset: 0, index: 0)

            let threadsPerGrid = MTLSize(width: threadCount, height: 1, depth: 1)
            let threadsPerGroup = MTLSize(width: min(threadCount, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
            encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerGroup)

            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()

            let result = buffer.contents().assumingMemoryBound(to: Float.self)[0]
            results.append(result)
            print("Run \(run): \(result)")
        }

        let allCorrect = results.allSatisfy { $0 == expected }
        print("\nExpected: \(expected)")
        print("All correct: \(allCorrect)")

        XCTAssertTrue(allCorrect, "All runs should equal \(expected)")
    }

    /// Test 3: Many more threads (10000) to stress test
    func testAtomicAddManyThreads() throws {
        let device = MTLCreateSystemDefaultDevice()!
        let commandQueue = device.makeCommandQueue()!

        let kernelSource = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void atomic_add_one(
            device float *memory [[buffer(0)]],
            uint id [[thread_position_in_grid]]
        ) {
            atomic_fetch_add_explicit(
                (device atomic<float>*)&memory[0],
                1.0f,
                memory_order_relaxed
            );
        }
        """

        let library = try device.makeLibrary(source: kernelSource, options: nil)
        let function = library.makeFunction(name: "atomic_add_one")!
        let pipeline = try device.makeComputePipelineState(function: function)

        let buffer = device.makeBuffer(length: MemoryLayout<Float>.size, options: .storageModeShared)!

        let threadCount = 10000
        var results: [Float] = []

        print("\n=== Test: 10000 threads each atomically add 1.0 ===")

        for run in 0..<20 {
            memset(buffer.contents(), 0, buffer.length)

            let commandBuffer = commandQueue.makeCommandBuffer()!
            let encoder = commandBuffer.makeComputeCommandEncoder()!
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(buffer, offset: 0, index: 0)

            let threadsPerGrid = MTLSize(width: threadCount, height: 1, depth: 1)
            let threadsPerGroup = MTLSize(width: min(threadCount, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
            encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerGroup)

            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()

            let result = buffer.contents().assumingMemoryBound(to: Float.self)[0]
            results.append(result)
            print("Run \(run): \(result)")
        }

        let allCorrect = results.allSatisfy { $0 == Float(threadCount) }
        print("\nExpected: \(threadCount).0")
        print("All correct: \(allCorrect)")

        XCTAssertTrue(allCorrect, "All runs should equal \(threadCount)")
    }

    /// Test 4: Multiple atomic adds per thread (like nested loops)
    /// Each of 100 threads does 10 atomic adds of 1.0
    /// Expected: 1000
    func testAtomicAddMultiplePerThread() throws {
        let device = MTLCreateSystemDefaultDevice()!
        let commandQueue = device.makeCommandQueue()!

        let kernelSource = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void atomic_add_loop(
            device float *memory [[buffer(0)]],
            uint id [[thread_position_in_grid]]
        ) {
            for (int i = 0; i < 10; i++) {
                atomic_fetch_add_explicit(
                    (device atomic<float>*)&memory[0],
                    1.0f,
                    memory_order_relaxed
                );
            }
        }
        """

        let library = try device.makeLibrary(source: kernelSource, options: nil)
        let function = library.makeFunction(name: "atomic_add_loop")!
        let pipeline = try device.makeComputePipelineState(function: function)

        let buffer = device.makeBuffer(length: MemoryLayout<Float>.size, options: .storageModeShared)!

        let threadCount = 100
        let addsPerThread = 10
        let expected = Float(threadCount * addsPerThread)
        var results: [Float] = []

        print("\n=== Test: 100 threads, each does 10 atomic adds ===")

        for run in 0..<20 {
            memset(buffer.contents(), 0, buffer.length)

            let commandBuffer = commandQueue.makeCommandBuffer()!
            let encoder = commandBuffer.makeComputeCommandEncoder()!
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(buffer, offset: 0, index: 0)

            let threadsPerGrid = MTLSize(width: threadCount, height: 1, depth: 1)
            let threadsPerGroup = MTLSize(width: min(threadCount, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
            encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerGroup)

            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()

            let result = buffer.contents().assumingMemoryBound(to: Float.self)[0]
            results.append(result)
            print("Run \(run): \(result)")
        }

        let allCorrect = results.allSatisfy { $0 == expected }
        print("\nExpected: \(expected)")
        print("All correct: \(allCorrect)")

        XCTAssertTrue(allCorrect, "All runs should equal \(expected)")
    }

    /// Test 5: Floating point values that might cause precision issues
    /// Each thread adds a small float that when summed should equal a known value
    func testAtomicAddFloatPrecision() throws {
        let device = MTLCreateSystemDefaultDevice()!
        let commandQueue = device.makeCommandQueue()!

        let kernelSource = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void atomic_add_small(
            device float *memory [[buffer(0)]],
            uint id [[thread_position_in_grid]]
        ) {
            // Each thread adds 0.01
            atomic_fetch_add_explicit(
                (device atomic<float>*)&memory[0],
                0.01f,
                memory_order_relaxed
            );
        }
        """

        let library = try device.makeLibrary(source: kernelSource, options: nil)
        let function = library.makeFunction(name: "atomic_add_small")!
        let pipeline = try device.makeComputePipelineState(function: function)

        let buffer = device.makeBuffer(length: MemoryLayout<Float>.size, options: .storageModeShared)!

        let threadCount = 100
        let expected: Float = 1.0  // 100 * 0.01
        var results: [Float] = []

        print("\n=== Test: 100 threads each add 0.01 (expect ~1.0) ===")

        for run in 0..<20 {
            memset(buffer.contents(), 0, buffer.length)

            let commandBuffer = commandQueue.makeCommandBuffer()!
            let encoder = commandBuffer.makeComputeCommandEncoder()!
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(buffer, offset: 0, index: 0)

            let threadsPerGrid = MTLSize(width: threadCount, height: 1, depth: 1)
            let threadsPerGroup = MTLSize(width: min(threadCount, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
            encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerGroup)

            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()

            let result = buffer.contents().assumingMemoryBound(to: Float.self)[0]
            results.append(result)
            print("Run \(run): \(result)")
        }

        // Check if all results are the same (even if not exactly 1.0 due to FP precision)
        let allSame = results.allSatisfy { abs($0 - results[0]) < 0.0001 }
        let closeToExpected = results.allSatisfy { abs($0 - expected) < 0.01 }

        print("\nExpected: ~\(expected)")
        print("All identical: \(allSame)")
        print("All close to expected: \(closeToExpected)")

        XCTAssertTrue(allSame, "All runs should be identical")
    }
}
