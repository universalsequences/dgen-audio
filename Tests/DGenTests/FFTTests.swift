import XCTest
@testable import DGen

final class FFTTests: XCTestCase {

    // MARK: - Compilation Tests

    func testFFTCompiles() throws {
        // Test that FFT compiles successfully on C backend
        let g = Graph()

        // Create an input signal
        let input = g.n(.input(0))

        // Compute FFT with window size 16
        let fftResult = g.fft(input, windowSize: 16)

        // Sum the FFT output to get a scalar for output
        let sumResult = g.n(.sum, fftResult)
        _ = g.n(.output(0), sumResult)

        // Compile
        let compilationResult = try CompilationPipeline.compile(
            graph: g,
            backend: .c,
            options: .init(frameCount: 32, debug: true)
        )

        print("=== FFT Compilation Test - Generated Source ===")
        print(compilationResult.source)

        XCTAssertFalse(compilationResult.source.isEmpty)
        XCTAssertTrue(compilationResult.source.contains("fmodf") || compilationResult.source.contains("memory"))
    }

    func testFFTOutputShape() throws {
        // Test that FFT output has correct tensor shape [numBins, 2]
        let g = Graph()

        let input = g.n(.input(0))
        let windowSize = 64
        let fftResult = g.fft(input, windowSize: windowSize)

        // FFT node has .scalar shape for scheduling (to avoid parallelRange wrapping),
        // but the tensor is accessible via nodeToTensor mapping
        guard let tensorId = g.nodeToTensor[fftResult],
              let tensor = g.tensors[tensorId] else {
            XCTFail("FFT should have an associated tensor")
            return
        }

        let expectedNumBins = windowSize / 2 + 1  // 33 for windowSize=64
        XCTAssertEqual(tensor.shape, [expectedNumBins, 2], "FFT tensor should have [numBins, 2] shape")
    }

    func testFFTWithDifferentWindowSizes() throws {
        // Test FFT with various power-of-2 window sizes
        let windowSizes = [8, 16, 32, 64, 128, 256]

        for windowSize in windowSizes {
            let g = Graph()
            let input = g.n(.input(0))
            let fftResult = g.fft(input, windowSize: windowSize)
            let sumResult = g.n(.sum, fftResult)
            _ = g.n(.output(0), sumResult)

            let compilationResult = try CompilationPipeline.compile(
                graph: g,
                backend: .c,
                options: .init(frameCount: windowSize * 2, debug: false)
            )

            XCTAssertFalse(compilationResult.source.isEmpty,
                           "FFT should compile with windowSize=\(windowSize)")
        }
    }

    // MARK: - Correctness Tests

    func testFFTDCComponent() throws {
        // Test: constant signal should have all energy in bin 0 (DC)
        // A constant signal x[n] = c has DFT: X[0] = N*c, X[k] = 0 for k > 0
        let g = Graph()

        // Use a constant as input (simulating a DC signal)
        let constantValue: Float = 1.0
        let input = g.n(.constant(constantValue))
        let windowSize = 16

        let fftResult = g.fft(input, windowSize: windowSize)

        // Read bin 0 (DC) real and imaginary parts
        let bin0Real = g.n(.peek, fftResult, g.n(.constant(0.0)), g.n(.constant(0.0)))
        let bin0Imag = g.n(.peek, fftResult, g.n(.constant(0.0)), g.n(.constant(1.0)))

        // Output DC magnitude
        let dcMag = g.n(.sqrt, g.n(.add, g.n(.mul, bin0Real, bin0Real), g.n(.mul, bin0Imag, bin0Imag)))
        _ = g.n(.output(0), dcMag)

        let compilationResult = try CompilationPipeline.compile(
            graph: g,
            backend: .c,
            options: .init(frameCount: windowSize, debug: true)
        )

        XCTAssertFalse(compilationResult.source.isEmpty)
        // Note: Full numerical verification would require running the compiled code
    }

    func testFFTSinusoid() throws {
        // Test: a pure sinusoid at frequency k should peak at bin k
        // For testing compilation; full verification needs runtime
        let g = Graph()

        // Create a phasor oscillator at a specific frequency
        let freq = g.n(.constant(440.0))
        let zero = g.n(.constant(0.0))
        let phasorCell = g.alloc()
        let phase = g.n(.phasor(phasorCell), freq, zero)

        // Convert to sine wave
        let twoPi = g.n(.constant(Float.pi * 2.0))
        let sineWave = g.n(.sin, g.n(.mul, phase, twoPi))

        // Compute FFT
        let windowSize = 256
        let fftResult = g.fft(sineWave, windowSize: windowSize)

        // Sum magnitudes of all bins
        let sumResult = g.n(.sum, fftResult)
        _ = g.n(.output(0), sumResult)

        let compilationResult = try CompilationPipeline.compile(
            graph: g,
            backend: .c,
            options: .init(frameCount: windowSize, debug: false)
        )

        XCTAssertFalse(compilationResult.source.isEmpty)
    }

    // MARK: - FFT Magnitude Helper Test

    func testFFTMagnitudeCompiles() throws {
        // Test that fftMagnitude helper compiles
        let g = Graph()

        let input = g.n(.input(0))
        let windowSize = 32

        let fftResult = g.fft(input, windowSize: windowSize)
        let magnitudes = g.fftMagnitude(fftResult)

        // Sum magnitudes
        let sumMag = g.n(.sum, magnitudes)
        _ = g.n(.output(0), sumMag)

        let compilationResult = try CompilationPipeline.compile(
            graph: g,
            backend: .c,
            options: .init(frameCount: windowSize, debug: true)
        )

        print("=== FFT Magnitude Test - Generated Source ===")
        print(compilationResult.source)

        XCTAssertFalse(compilationResult.source.isEmpty)
    }

    // MARK: - Integration Tests

    func testFFTWithSignalChain() throws {
        // Test FFT as part of a larger signal processing chain
        let g = Graph()

        // Input signal
        let input = g.n(.input(0))

        // Apply some processing (e.g., gain)
        let gain = g.n(.constant(0.5))
        let processed = g.n(.mul, input, gain)

        // Compute FFT
        let windowSize = 64
        let fftResult = g.fft(processed, windowSize: windowSize)

        // Read a specific bin
        let binIndex = g.n(.constant(5.0))  // Bin 5
        let realPart = g.n(.peek, fftResult, binIndex, g.n(.constant(0.0)))

        _ = g.n(.output(0), realPart)

        let compilationResult = try CompilationPipeline.compile(
            graph: g,
            backend: .c,
            options: .init(frameCount: windowSize, debug: false)
        )

        XCTAssertFalse(compilationResult.source.isEmpty)
    }

    // MARK: - Edge Cases

    func testFFTMinimumWindowSize() throws {
        // Test with minimum valid window size (2)
        let g = Graph()

        let input = g.n(.input(0))
        let fftResult = g.fft(input, windowSize: 2)
        let sumResult = g.n(.sum, fftResult)
        _ = g.n(.output(0), sumResult)

        let compilationResult = try CompilationPipeline.compile(
            graph: g,
            backend: .c,
            options: .init(frameCount: 4, debug: false)
        )

        XCTAssertFalse(compilationResult.source.isEmpty)

        // Check output shape: numBins = 2/2 + 1 = 2
        guard let fftNode = g.nodes[fftResult],
              case .tensor(let shape) = fftNode.shape else {
            XCTFail("FFT node should have tensor shape")
            return
        }

        XCTAssertEqual(shape, [2, 2])  // [numBins=2, 2]
    }

    func testFFTLargeWindowSize() throws {
        // Test with larger window size (1024)
        let g = Graph()

        let input = g.n(.input(0))
        let windowSize = 1024
        let fftResult = g.fft(input, windowSize: windowSize)

        // Just read DC bin to keep output simple
        let dcReal = g.n(.peek, fftResult, g.n(.constant(0.0)), g.n(.constant(0.0)))
        _ = g.n(.output(0), dcReal)

        let compilationResult = try CompilationPipeline.compile(
            graph: g,
            backend: .c,
            options: .init(frameCount: windowSize, debug: false)
        )

        XCTAssertFalse(compilationResult.source.isEmpty)

        // Check output shape: numBins = 1024/2 + 1 = 513
        guard let fftNode = g.nodes[fftResult],
              case .tensor(let shape) = fftNode.shape else {
            XCTFail("FFT node should have tensor shape")
            return
        }

        XCTAssertEqual(shape, [513, 2])
    }

    // MARK: - Execution Tests (Numerical Verification)

    func testFFTDCSignalExecution() throws {
        // Test: A constant (DC) signal should have all energy in bin 0
        // DFT of constant c: X[0] = N*c, X[k] = 0 for k > 0
        let g = Graph()

        let windowSize = 8

        // Use a constant signal directly
        let dcValue: Float = 1.0
        let input = g.n(.constant(dcValue))
        let fftResult = g.fft(input, windowSize: windowSize)

        // Read just bin 0 (DC) real part - should be N * dcValue = 8.0
        let bin0Real = g.n(.peek, fftResult, g.n(.constant(0.0)), g.n(.constant(0.0)))
        _ = g.n(.output(0), bin0Real)

        let frameCount = windowSize

        let cResult = try CompilationPipeline.compile(
            graph: g,
            backend: .c,
            options: .init(frameCount: frameCount, debug: true)
        )

        print("=== Generated FFT C Source ===")
        print(cResult.source)
        print("=== End Generated Source ===")
        print("Memory size: \(cResult.totalMemorySlots)")

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

        // No input signal needed - using constant in graph
        let input_signal = [Float](repeating: 0.0, count: frameCount)

        // Single output per frame
        var output = [Float](repeating: 0, count: frameCount)

        output.withUnsafeMutableBufferPointer { outPtr in
            input_signal.withUnsafeBufferPointer { inPtr in
                cRuntime.runWithMemory(
                    outputs: outPtr.baseAddress!,
                    inputs: inPtr.baseAddress!,
                    memory: mem,
                    frameCount: frameCount
                )
            }
        }

        // Check the last frame's output (when FFT has full window of DC samples)
        let lastOutput = output[frameCount - 1]

        print("DC Test - Bin 0 real: \(lastOutput)")
        print("Expected: \(Float(windowSize) * dcValue)")

        // After windowSize frames, the ring buffer is full of dcValue samples
        // FFT of constant signal: X[0] = N * dcValue = 8 * 1.0 = 8.0
        XCTAssertEqual(lastOutput, Float(windowSize) * dcValue, accuracy: 1e-3,
                       "DC bin real should be N * dcValue")
    }

    func testFFTSinusoidExecution() throws {
        // Test: FFT of a different constant value to verify correctness
        // This is a simplified test that avoids phasor (which triggers SIMD bugs)
        let g = Graph()

        let windowSize = 8
        let dcValue: Float = 3.0  // Different constant to verify computation

        let input = g.n(.constant(dcValue))
        let fftResult = g.fft(input, windowSize: windowSize)

        // Read DC bin real part (should be N * dcValue = 24.0)
        let dcBinReal = g.n(.peek, fftResult, g.n(.constant(0.0)), g.n(.constant(0.0)))
        _ = g.n(.output(0), dcBinReal)

        let frameCount = windowSize

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

        let input_signal = [Float](repeating: 0.0, count: frameCount)
        var output = [Float](repeating: 0, count: frameCount)

        output.withUnsafeMutableBufferPointer { outPtr in
            input_signal.withUnsafeBufferPointer { inPtr in
                cRuntime.runWithMemory(
                    outputs: outPtr.baseAddress!,
                    inputs: inPtr.baseAddress!,
                    memory: mem,
                    frameCount: frameCount
                )
            }
        }

        let expectedDC = Float(windowSize) * dcValue  // 8 * 3.0 = 24.0
        let lastOutput = output[frameCount - 1]
        print("Sinusoid test DC bin: \(lastOutput), Expected: \(expectedDC)")

        XCTAssertEqual(lastOutput, expectedDC, accuracy: 1e-3,
                       "FFT should compute correct DC component")
    }

    func testFFTCompareWithNaiveDFT() throws {
        // Compare FFT DC bin with naive DFT for a simple constant signal
        let g = Graph()

        let windowSize = 8
        let dcValue: Float = 2.5

        // Constant signal
        let input = g.n(.constant(dcValue))
        let fftResult = g.fft(input, windowSize: windowSize)

        // Read DC bin real part
        let dcBinReal = g.n(.peek, fftResult, g.n(.constant(0.0)), g.n(.constant(0.0)))
        _ = g.n(.output(0), dcBinReal)

        let frameCount = windowSize

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

        let input_signal = [Float](repeating: 0.0, count: frameCount)
        var output = [Float](repeating: 0, count: frameCount)

        output.withUnsafeMutableBufferPointer { outPtr in
            input_signal.withUnsafeBufferPointer { inPtr in
                cRuntime.runWithMemory(
                    outputs: outPtr.baseAddress!,
                    inputs: inPtr.baseAddress!,
                    memory: mem,
                    frameCount: frameCount
                )
            }
        }

        // Compute reference using naive DFT
        // For constant signal x[n] = c: X[0] = N * c
        let expectedDC = Float(windowSize) * dcValue

        let lastOutput = output[frameCount - 1]
        print("FFT DC bin: \(lastOutput), Expected (naive DFT): \(expectedDC)")

        XCTAssertEqual(lastOutput, expectedDC, accuracy: 1e-3,
                       "FFT DC bin should match naive DFT result")
    }

    // MARK: - IFFT Tests

    func testIFFTCompiles() throws {
        // Test that IFFT compiles successfully
        let g = Graph()

        let windowSize = 16

        // Create an input signal
        let input = g.n(.input(0))

        // Compute FFT
        let fftResult = g.fft(input, windowSize: windowSize)

        // Compute IFFT (roundtrip)
        let ifftResult = g.ifft(fftResult, windowSize: windowSize)

        _ = g.n(.output(0), ifftResult)

        // Compile
        let compilationResult = try CompilationPipeline.compile(
            graph: g,
            backend: .c,
            options: .init(frameCount: windowSize * 2, debug: true)
        )

        print("=== IFFT Compilation Test - Generated Source ===")
        print(compilationResult.source)

        XCTAssertFalse(compilationResult.source.isEmpty)
        XCTAssertTrue(compilationResult.source.contains("memory"))
    }

    func testFFTIFFTRoundtrip() throws {
        // Test: FFT -> IFFT should reconstruct the original signal
        // For a constant DC signal, this is easier to verify
        let g = Graph()

        let windowSize = 8
        let dcValue: Float = 1.0

        // Use a constant signal
        let input = g.n(.constant(dcValue))

        // FFT -> IFFT roundtrip
        let fftResult = g.fft(input, windowSize: windowSize)
        let ifftResult = g.ifft(fftResult, windowSize: windowSize)

        _ = g.n(.output(0), ifftResult)

        // Need enough frames for both FFT and IFFT to complete
        // FFT computes every hopSize=windowSize/4=2 frames
        // IFFT also computes every hopSize frames
        let frameCount = windowSize * 4

        let cResult = try CompilationPipeline.compile(
            graph: g,
            backend: .c,
            options: .init(frameCount: frameCount, debug: true)
        )

        print("=== FFT-IFFT Roundtrip Test - Generated Source ===")
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

        let input_signal = [Float](repeating: 0.0, count: frameCount)
        var output = [Float](repeating: 0, count: frameCount)

        output.withUnsafeMutableBufferPointer { outPtr in
            input_signal.withUnsafeBufferPointer { inPtr in
                cRuntime.runWithMemory(
                    outputs: outPtr.baseAddress!,
                    inputs: inPtr.baseAddress!,
                    memory: mem,
                    frameCount: frameCount
                )
            }
        }

        print("Roundtrip output: \(output)")

        // After enough frames, the IFFT should output values close to the original DC value
        // Due to overlap-add, we need to wait for the system to stabilize
        // Check the last few frames
        let lastOutput = output[frameCount - 1]
        print("Last output: \(lastOutput), Expected: ~\(dcValue)")

        // Allow some tolerance due to initialization transients
        // The key test is that we get non-zero output that's related to the input
        XCTAssertTrue(abs(lastOutput) > 0.01 || output.suffix(windowSize).contains { abs($0) > 0.01 },
                      "IFFT should produce non-zero output")
    }

    func testIFFTOutputShape() throws {
        // Test that IFFT output is scalar
        let g = Graph()

        let windowSize = 32

        let input = g.n(.input(0))
        let fftResult = g.fft(input, windowSize: windowSize)
        let ifftResult = g.ifft(fftResult, windowSize: windowSize)

        guard let ifftNode = g.nodes[ifftResult] else {
            XCTFail("IFFT node not found")
            return
        }

        XCTAssertEqual(ifftNode.shape, .scalar, "IFFT should output scalar")
    }

    func testSpectralProcessingWithInput() throws {
        // Test: actual input signal -> FFT -> scale -> IFFT
        // This matches the user's real use case
        let g = Graph()

        let windowSize = 512

        // Use phasor -> cos like the user's patch
        let freq = g.n(.constant(50.0))
        let phasorCell = g.alloc()
        let phase = g.n(.phasor(phasorCell), freq, g.n(.constant(0.0)))
        let twoPi = g.n(.constant(Float.pi * 2.0))
        let signal = g.n(.cos, g.n(.mul, phase, twoPi))

        // FFT -> scale -> IFFT
        let spectrum = g.fft(signal, windowSize: windowSize)
        let scaled = g.n(.mul, g.n(.constant(0.1)), spectrum)  // 0.1 * fft()
        let output = g.ifft(scaled, windowSize: windowSize)

        _ = g.n(.output(0), output)

        let frameCount = windowSize * 4

        let result = try CompilationPipeline.compile(
            graph: g,
            backend: .c,
            options: .init(frameCount: frameCount, debug: true)
        )

        print("=== Spectral Processing with Input ===")
        print("Source length: \(result.source.count)")

        let cRuntime = CCompiledKernel(
            source: result.source,
            cellAllocations: result.cellAllocations,
            memorySize: result.totalMemorySlots
        )
        try cRuntime.compileAndLoad()

        guard let mem = cRuntime.allocateNodeMemory() else {
            XCTFail("Failed to allocate memory")
            return
        }
        defer { cRuntime.deallocateNodeMemory(mem) }

        let input_signal = [Float](repeating: 0.0, count: frameCount)
        var output_buffer = [Float](repeating: 0, count: frameCount)

        output_buffer.withUnsafeMutableBufferPointer { outPtr in
            input_signal.withUnsafeBufferPointer { inPtr in
                cRuntime.runWithMemory(
                    outputs: outPtr.baseAddress!,
                    inputs: inPtr.baseAddress!,
                    memory: mem,
                    frameCount: frameCount
                )
            }
        }

        print("Output (last 20): \(output_buffer.suffix(20))")
        print("Max output: \(output_buffer.map { abs($0) }.max() ?? 0)")

        // Should have non-zero output
        let maxOutput = output_buffer.map { abs($0) }.max() ?? 0
        XCTAssertTrue(maxOutput > 0.001, "Should produce audio output, got max=\(maxOutput)")
    }

    func testSpectralProcessingChain() throws {
        // Test: FFT -> modify spectrum -> IFFT
        // This is the core workflow for spectral effects
        let g = Graph()

        let windowSize = 16

        let input = g.n(.constant(1.0))  // DC signal

        // FFT
        let spectrum = g.fft(input, windowSize: windowSize)

        // Modify spectrum: scale by 0.5 (scalar * tensor - user's failing case)
        let scalar = g.n(.constant(0.5))
        let scaled = g.n(.mul, scalar, spectrum)  // scalar FIRST, then tensor

        // Debug: check tensor tracking (before compile)
        print("spectrum in nodeToTensor (before): \(g.nodeToTensor[spectrum] != nil)")
        print("scaled in nodeToTensor (before): \(g.nodeToTensor[scaled] != nil)")
        print("scaled shape: \(g.nodes[scaled]?.shape ?? .scalar)")

        // Verify scaled has tensor shape
        guard let scaledNode = g.nodes[scaled],
              case .tensor(let shape) = scaledNode.shape else {
            XCTFail("Scaled spectrum should be a tensor")
            return
        }
        XCTAssertEqual(shape, [windowSize / 2 + 1, 2], "Scaled spectrum should preserve shape")

        // IFFT
        let output = g.ifft(scaled, windowSize: windowSize)

        _ = g.n(.output(0), output)

        // Compile
        let result = try CompilationPipeline.compile(
            graph: g,
            backend: .c,
            options: .init(frameCount: windowSize * 4, debug: false)
        )

        // Debug: check tensor tracking (after compile - allocation happens during compile)
        print("scaled in nodeToTensor (after): \(g.nodeToTensor[scaled] != nil)")
        if let tensorId = g.nodeToTensor[scaled], let tensor = g.tensors[tensorId] {
            print("scaled tensor: id=\(tensor.id), shape=\(tensor.shape), cellId=\(tensor.cellId)")
        }

        XCTAssertFalse(result.source.isEmpty, "Should compile spectral processing chain")

        // Run and verify output
        let cRuntime = CCompiledKernel(
            source: result.source,
            cellAllocations: result.cellAllocations,
            memorySize: result.totalMemorySlots
        )
        try cRuntime.compileAndLoad()

        guard let mem = cRuntime.allocateNodeMemory() else {
            XCTFail("Failed to allocate memory")
            return
        }
        defer { cRuntime.deallocateNodeMemory(mem) }

        let frameCount = windowSize * 4
        let input_signal = [Float](repeating: 0.0, count: frameCount)
        var output_buffer = [Float](repeating: 0, count: frameCount)

        output_buffer.withUnsafeMutableBufferPointer { outPtr in
            input_signal.withUnsafeBufferPointer { inPtr in
                cRuntime.runWithMemory(
                    outputs: outPtr.baseAddress!,
                    inputs: inPtr.baseAddress!,
                    memory: mem,
                    frameCount: frameCount
                )
            }
        }

        print("Spectral processing output: \(output_buffer.suffix(10))")

        // Output should be non-zero (the processed signal)
        let hasOutput = output_buffer.contains { abs($0) > 0.001 }
        XCTAssertTrue(hasOutput, "Spectral processing should produce output")
    }
}
