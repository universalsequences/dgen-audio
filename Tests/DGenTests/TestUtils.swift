import XCTest
@testable import DGen

// ANSI color codes for terminal output
struct ANSIColors {
    static let red = "\u{001B}[31m"
    static let green = "\u{001B}[32m"
    static let yellow = "\u{001B}[33m"
    static let blue = "\u{001B}[34m"
    static let reset = "\u{001B}[0m"
    static let bold = "\u{001B}[1m"
}

/// Represents the output from executing a graph on a specific device
public struct GraphExecutionResult {
    let device: Device
    let deviceName: String
    let outputs: [Float]
    let success: Bool
    let error: Error?
    let kernels: [CompiledKernel]?
    
    init(device: Device, deviceName: String, outputs: [Float], kernels: [CompiledKernel]? = nil) {
        self.device = device
        self.deviceName = deviceName
        self.outputs = outputs
        self.success = true
        self.error = nil
        self.kernels = kernels
    }
    
    init(device: Device, deviceName: String, error: Error, kernels: [CompiledKernel]? = nil) {
        self.device = device
        self.deviceName = deviceName
        self.outputs = []
        self.success = false
        self.error = error
        self.kernels = kernels
    }
}

/// Configuration for graph execution
public struct GraphTestConfig {
    let sampleRate: Int
    let duration: Double
    let frameSize: Int
    let volumeScale: Float
    let maxNodesPerBlock: Int
    let debug: Bool
    let printKernelsOnError: Bool
    
    public init(
        sampleRate: Int = 44100,
        duration: Double = 1.0,
        frameSize: Int = 128,
        volumeScale: Float = 1.0,
        maxNodesPerBlock: Int = 320,
        debug: Bool = false,
        printKernelsOnError: Bool = true
    ) {
        self.sampleRate = sampleRate
        self.duration = duration
        self.frameSize = frameSize
        self.volumeScale = volumeScale
        self.maxNodesPerBlock = maxNodesPerBlock
        self.debug = debug
        self.printKernelsOnError = printKernelsOnError
    }
}

/// Test framework for executing graphs on multiple backends
public class GraphTestFramework {
    
    /// Execute a graph on all available backends and return results
    public static func executeGraph(
        _ graph: Graph,
        config: GraphTestConfig = GraphTestConfig(),
        backends: [(Device, String)] = [(.C, "C"), (.Metal, "Metal")]
    ) -> [GraphExecutionResult] {
        
        var results: [GraphExecutionResult] = []
        
        // Process the graph
        let sorted = topo(graph, debug: config.debug)
        let scalar = scalarNodes(graph)
        let blocks = determineBlocks(sorted: sorted, scalar: scalar, g: graph, 
                                   maxNodesPerBlock: config.maxNodesPerBlock, debug: config.debug)
        let sortedBlockIds = sortBlocksByDependencies(blocks, graph, debug: config.debug)
        
        var sortedBlocks: [Block] = []
        sortedBlockIds.forEach { sortedBlocks.append(blocks[$0]) }
        
        var ctx = IRContext()
        var uopBlocks: [BlockUOps] = []
        
        for blockIdx in sortedBlockIds {
            let block = blocks[blockIdx]
            let uops = emitBlockUOps(ctx: ctx, block: block, blocks: sortedBlocks, g: graph, debug: config.debug)
            uopBlocks.append(BlockUOps(ops: uops, kind: block.kind))
        }
        
        // Test each backend
        for (device, deviceName) in backends {
            do {
                let (outputs, kernels) = try executeOnDevice(
                    device: device,
                    uopBlocks: &uopBlocks,
                    ctx: ctx,
                    config: config
                )
                results.append(GraphExecutionResult(device: device, deviceName: deviceName, outputs: outputs, kernels: kernels))
            } catch {
                // Try to generate kernels even if execution fails, for debugging
                let renderer: Renderer = device == .Metal ? MetalRenderer() : CRenderer()
                let kernels = lowerUOpBlocks(&uopBlocks, renderer: renderer, ctx: ctx, frameCount: config.frameSize)
                results.append(GraphExecutionResult(device: device, deviceName: deviceName, error: error, kernels: kernels))
            }
        }
        
        return results
    }
    
    /// Execute UOp blocks on a specific device
    private static func executeOnDevice(
        device: Device,
        uopBlocks: inout [BlockUOps],
        ctx: IRContext,
        config: GraphTestConfig
    ) throws -> ([Float], [CompiledKernel]) {
        
        let renderer: Renderer = device == .Metal ? MetalRenderer() : CRenderer()
        let kernels = lowerUOpBlocks(&uopBlocks, renderer: renderer, ctx: ctx, frameCount: config.frameSize)
        
        let runtime: CompiledKernelRuntime
        if device == .Metal {
            runtime = try MetalCompiledKernel(kernels: kernels)
        } else {
            let source = kernels.first?.source ?? ""
            let compiled = CCompiledKernel(source: source)
            try compiled.compileAndLoad()
            runtime = compiled
        }
        
        // Calculate number of frames to process
        let totalSamples = Int(config.duration * Double(config.sampleRate))
        let numFrames = totalSamples / config.frameSize
        
        var allOutputs: [Float] = []
        let silentInput = [Float](repeating: 0, count: config.frameSize)
        
        // Process frame by frame
        for _ in 0..<numFrames {
            var outputs = [Float](repeating: 0, count: config.frameSize)
            silentInput.withUnsafeBufferPointer { inputPtr in
                runtime.run(outputs: &outputs, inputs: inputPtr.baseAddress!, frameCount: config.frameSize, volumeScale: config.volumeScale)
            }
            allOutputs.append(contentsOf: outputs)
        }
        
        // Clean up runtime resources to ensure test isolation
        runtime.cleanup()
        
        return (allOutputs, kernels)
    }
    
    /// Assert that all execution results are successful
    public static func assertAllSuccessful(_ results: [GraphExecutionResult], file: StaticString = #file, line: UInt = #line) {
        for result in results {
            XCTAssertTrue(result.success, "\(ANSIColors.red)\(ANSIColors.bold)Execution failed on \(result.deviceName): \(result.error?.localizedDescription ?? "Unknown error")\(ANSIColors.reset)", file: file, line: line)
        }
    }
    
    /// Assert that outputs from different backends match within tolerance
    public static func assertOutputsMatch(
        _ results: [GraphExecutionResult],
        tolerance: Float = 0.001,
        maxErrors: Int = 10,
        printKernelsOnError: Bool = true,
        file: StaticString = #file,
        line: UInt = #line
    ) {
        guard results.count >= 2 else {
            XCTFail("Need at least 2 results to compare", file: file, line: line)
            return
        }
        
        // Filter successful results
        let successfulResults = results.filter { $0.success }
        guard successfulResults.count >= 2 else {
            XCTFail("Need at least 2 successful results to compare", file: file, line: line)
            return
        }
        
        let reference = successfulResults[0]
        var hasErrors = false
        
        for i in 1..<successfulResults.count {
            let other = successfulResults[i]
            let errorCount = countDifferences(reference.outputs, other.outputs, tolerance: tolerance)
            if errorCount > 0 {
                hasErrors = true
            }
            
            assertOutputsEqual(
                reference.outputs,
                other.outputs,
                tolerance: tolerance,
                message: "Outputs don't match between \(reference.deviceName) and \(other.deviceName)",
                maxErrors: maxErrors,
                file: file,
                line: line
            )
        }
        
        // If there were errors and printKernelsOnError is enabled, print the kernels
        if hasErrors && printKernelsOnError {
            printKernelDebugInfo(results)
        }
    }
    
    /// Count differences between two output arrays
    private static func countDifferences(_ outputs1: [Float], _ outputs2: [Float], tolerance: Float) -> Int {
        let minCount = min(outputs1.count, outputs2.count)
        var errorCount = 0
        for i in 0..<minCount {
            if abs(outputs1[i] - outputs2[i]) > tolerance {
                errorCount += 1
            }
        }
        return errorCount
    }
    
    /// Assert that two output arrays are equal within tolerance
    public static func assertOutputsEqual(
        _ outputs1: [Float],
        _ outputs2: [Float],
        tolerance: Float = 0.001,
        message: String = "Outputs don't match",
        maxErrors: Int = 10,
        file: StaticString = #file,
        line: UInt = #line
    ) {
        XCTAssertEqual(outputs1.count, outputs2.count, "\(message): Different output lengths", file: file, line: line)
        
        let minCount = min(outputs1.count, outputs2.count)
        var errorCount = 0
        var firstError: Int?
        var lastError: Int?
        
        for i in 0..<minCount {
            let diff = abs(outputs1[i] - outputs2[i])
            if diff > tolerance {
                errorCount += 1
                if firstError == nil {
                    firstError = i
                }
                lastError = i
                
                if errorCount <= maxErrors {
                    XCTAssertLessThanOrEqual(
                        diff, 
                        tolerance, 
                        "\(ANSIColors.red)\(ANSIColors.bold)\(message): Sample \(i) differs by \(diff) (tolerance: \(tolerance)). \(outputs1[i]) vs \(outputs2[i])\(ANSIColors.reset)",
                        file: file,
                        line: line
                    )
                } else if errorCount == maxErrors + 1 {
                    // Print summary message when we hit the limit
                    XCTFail("\(ANSIColors.red)\(ANSIColors.bold)\(message): Too many differences! Stopping after \(maxErrors) errors. Continuing to count...\(ANSIColors.reset)", file: file, line: line)
                }
            }
        }
        
        // If we had more errors than we printed, show a summary
        if errorCount > maxErrors {
            XCTFail("\(ANSIColors.red)\(ANSIColors.bold)\(message): Total \(errorCount) samples differ (showed first \(maxErrors)). First error at sample \(firstError ?? 0), last at sample \(lastError ?? 0)\(ANSIColors.reset)", file: file, line: line)
        }
    }
    
    /// Print debug information about execution results
    public static func printResultsSummary(_ results: [GraphExecutionResult], sampleCount: Int = 5) {
        print("\n\(ANSIColors.bold)=== Graph Execution Results ===\(ANSIColors.reset)")
        for result in results {
            if result.success {
                print("\(ANSIColors.green)✅ \(result.deviceName):\(ANSIColors.reset) \(result.outputs.count) samples")
                if !result.outputs.isEmpty {
                    let sampleIndices = Array(0..<min(sampleCount, result.outputs.count))
                    let samples = sampleIndices.map { String(format: "%.6f", result.outputs[$0]) }
                    print("   First \(samples.count) samples: [\(samples.joined(separator: ", "))]")
                    
                    if result.outputs.count > sampleCount {
                        let lastIndices = Array((result.outputs.count - sampleCount)..<result.outputs.count)
                        let lastSamples = lastIndices.map { String(format: "%.6f", result.outputs[$0]) }
                        print("   Last \(lastSamples.count) samples: [\(lastSamples.joined(separator: ", "))]")
                    }
                }
            } else {
                print("\(ANSIColors.red)\(ANSIColors.bold)❌ \(result.deviceName): FAILED\(ANSIColors.reset) - \(result.error?.localizedDescription ?? "Unknown error")")
            }
        }
    }
    
    /// Validate outputs are within expected ranges and patterns
    public static func validateOutputRange(
        _ outputs: [Float],
        minValue: Float,
        maxValue: Float,
        deviceName: String,
        file: StaticString = #file,
        line: UInt = #line
    ) {
        for (i, sample) in outputs.enumerated() {
            XCTAssertGreaterThanOrEqual(sample, minValue, "\(deviceName): Sample \(i) below minimum", file: file, line: line)
            XCTAssertLessThanOrEqual(sample, maxValue, "\(deviceName): Sample \(i) above maximum", file: file, line: line)
        }
    }
    
    /// Print kernel debug information
    public static func printKernelDebugInfo(_ results: [GraphExecutionResult]) {
        print("\n\(ANSIColors.yellow)\(ANSIColors.bold)=== KERNEL DEBUG INFO ===\(ANSIColors.reset)")
        
        for result in results {
            guard let kernels = result.kernels else { continue }
            
            print("\n\(ANSIColors.bold)\(result.deviceName) Backend Kernels:\(ANSIColors.reset)")
            print("Total kernels: \(kernels.count)")
            
            for (index, kernel) in kernels.enumerated() {
                print("\n\(ANSIColors.blue)--- Kernel \(index): \(kernel.name) ---\(ANSIColors.reset)")
                print("Thread group size: \(kernel.threadGroupSize)")
                print("Buffers: \(kernel.buffers.joined(separator: ", "))")
                
                // Print kernel source
                print("\(ANSIColors.bold)Source:\(ANSIColors.reset)")
                let lines = kernel.source.split(separator: "\n")
                for (lineNum, line) in lines.enumerated() {
                    print(String(format: "%3d: %@", lineNum + 1, String(line)))
                }
            }
        }
        
        print("\n\(ANSIColors.yellow)=== END KERNEL DEBUG INFO ===\(ANSIColors.reset)\n")
    }
}
