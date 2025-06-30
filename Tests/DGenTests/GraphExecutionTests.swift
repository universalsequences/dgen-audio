import XCTest
@testable import DGen

final class GraphExecutionTests: XCTestCase {
    
    func testSimplePhasorExecution() throws {
        // Create a simple 1Hz phasor graph
        let g = Graph()
        let freq = g.n(.constant(1.0))  // 1Hz frequency
        let reset = g.n(.constant(0.0)) // No reset
        let phasor = g.n(.phasor(0), freq, reset)
        let output = g.n(.output(0), phasor)
        
        let config = GraphTestConfig(duration: 1.0, frameSize: 128)
        let results = GraphTestFramework.executeGraph(g, config: config)
        
        // Assert all backends succeed
        GraphTestFramework.assertAllSuccessful(results)
        
        // Validate phasor output range [0, 1]
        for result in results.filter({ $0.success }) {
            GraphTestFramework.validateOutputRange(
                result.outputs,
                minValue: -0.1,  // Small tolerance for floating point
                maxValue: 1.1,   // Small tolerance for floating point
                deviceName: result.deviceName
            )
            
            // Check that phasor starts near 0
            let firstSamples = Array(result.outputs.prefix(10))
            for (i, sample) in firstSamples.enumerated() {
                XCTAssertLessThanOrEqual(abs(sample), 0.1, 
                    "\(result.deviceName): Sample \(i) should start near 0, got \(sample)")
            }
        }
        
        // Assert outputs match between backends
        GraphTestFramework.assertOutputsMatch(results, tolerance: 0.01)
        
        GraphTestFramework.printResultsSummary(results)
    }
    
    func testComplexAudioGraph() throws {
        // Create the complex graph from main.swift
        let g = Graph()
        let freq1 = g.n(.mul, g.n(.constant(90)), g.n(.constant(2.5)))
        let freq2 = g.n(.mul, g.n(.constant(2)), g.n(.constant(3.5)))
        let ph1 = g.n(.phasor(0), freq1, g.n(.constant(0)))
        let ph2 = g.n(.phasor(1), freq2, g.n(.constant(0)))
        let cond = g.n(.lt, ph2, g.n(.constant(0.1)))
        let latched = g.n(.latch(2), ph1, cond)
        let mult = g.n(.mul, latched, ph1)
        let output = g.n(.output(0), mult)
        
        let config = GraphTestConfig(duration: 0.5, frameSize: 128)  // Shorter duration for complex graph
        let results = GraphTestFramework.executeGraph(g, config: config)
        
        // Assert all backends succeed
        GraphTestFramework.assertAllSuccessful(results)
        
        // Assert outputs match between backends
        GraphTestFramework.assertOutputsMatch(results, tolerance: 0.001)
        
        // Validate output is reasonable (should be positive multiplication result)
        for result in results.filter({ $0.success }) {
            XCTAssertFalse(result.outputs.isEmpty, "\(result.deviceName): Should produce output")
            
            // Most outputs should be positive (phasor * phasor)
            let positiveCount = result.outputs.filter { $0 > 0 }.count
            let totalCount = result.outputs.count
            XCTAssertGreaterThan(Double(positiveCount) / Double(totalCount), 0.5, 
                "\(result.deviceName): Expected mostly positive outputs for phasor multiplication")
        }
        
        GraphTestFramework.printResultsSummary(results)
    }
    
    func testUnconnectedNodesHandling() throws {
        // Test the scenario that was causing issues: unconnected nodes
        let g = Graph()
        
        // Connected component
        let freq1 = g.n(.constant(1.0))
        let reset1 = g.n(.constant(0.0))
        let phasor1 = g.n(.phasor(0), freq1, reset1)
        let output = g.n(.output(0), phasor1)
        
        // Unconnected component - should not affect the output
        let freq2 = g.n(.constant(120.0))
        let reset2 = g.n(.constant(0.0))
        let phasor2 = g.n(.phasor(1), freq2, reset2)
        
        let config = GraphTestConfig(duration: 1.0, frameSize: 128)
        let results = GraphTestFramework.executeGraph(g, config: config)
        
        // Assert all backends succeed
        GraphTestFramework.assertAllSuccessful(results)
        
        // Assert outputs match between backends
        GraphTestFramework.assertOutputsMatch(results, tolerance: 0.01)
        
        // The output should be the same as if the unconnected phasor wasn't there
        // Create reference graph without unconnected node
        let referenceGraph = Graph()
        let refFreq = referenceGraph.n(.constant(1.0))
        let refReset = referenceGraph.n(.constant(0.0))
        let refPhasor = referenceGraph.n(.phasor(0), refFreq, refReset)
        let refOutput = referenceGraph.n(.output(0), refPhasor)
        
        let referenceResults = GraphTestFramework.executeGraph(referenceGraph, config: config)
        GraphTestFramework.assertAllSuccessful(referenceResults)
        
        // Compare with reference (should be identical)
        let successfulResults = results.filter { $0.success }
        let successfulReference = referenceResults.filter { $0.success }
        
        XCTAssertEqual(successfulResults.count, successfulReference.count, "Should have same number of successful results")
        
        for i in 0..<min(successfulResults.count, successfulReference.count) {
            GraphTestFramework.assertOutputsEqual(
                successfulResults[i].outputs,
                successfulReference[i].outputs,
                tolerance: 0.001,
                message: "Unconnected node should not affect output"
            )
        }
        
        GraphTestFramework.printResultsSummary(results)
    }
    
    func testConstantOutputs() throws {
        // Simple test with just constants
        let g = Graph()
        let const1 = g.n(.constant(5.0))
        let const2 = g.n(.constant(3.0))
        let sum = g.n(.add, const1, const2)
        let output = g.n(.output(0), sum)
        
        let config = GraphTestConfig(duration: 0.1, frameSize: 64)
        let results = GraphTestFramework.executeGraph(g, config: config)
        
        // Assert all backends succeed
        GraphTestFramework.assertAllSuccessful(results)
        
        // Assert outputs match between backends
        GraphTestFramework.assertOutputsMatch(results, tolerance: 0.001)
        
        // All outputs should be exactly 8.0 (5 + 3)
        for result in results.filter({ $0.success }) {
            for (i, sample) in result.outputs.enumerated() {
                XCTAssertEqual(sample, 8.0, accuracy: 0.001, 
                    "\(result.deviceName): Sample \(i) should be 8.0, got \(sample)")
            }
        }
        
        GraphTestFramework.printResultsSummary(results)
    }
    
    func testMixedOperations() throws {
        // Test graph with mix of SIMD and scalar operations
        let g = Graph()
        let freq = g.n(.mul, g.n(.constant(2.0)), g.n(.constant(2.0)))  // SIMD: 4.0
        let reset = g.n(.constant(0.0))  // SIMD
        let phasor = g.n(.phasor(0), freq, reset)  // Scalar
        let gain = g.n(.constant(0.5))  // SIMD  
        let scaled = g.n(.mul, phasor, gain)  // SIMD
        let output = g.n(.output(0), scaled)
        
        let config = GraphTestConfig(duration: 0.5, frameSize: 128)
        let results = GraphTestFramework.executeGraph(g, config: config)
        
        // Assert all backends succeed
        GraphTestFramework.assertAllSuccessful(results)
        
        // Assert outputs match between backends
        GraphTestFramework.assertOutputsMatch(results, tolerance: 0.01)
        
        // Validate range [0, 0.5] since we're scaling phasor by 0.5
        for result in results.filter({ $0.success }) {
            GraphTestFramework.validateOutputRange(
                result.outputs,
                minValue: -0.1,  // Small tolerance
                maxValue: 0.6,   // Small tolerance
                deviceName: result.deviceName
            )
        }
        
        GraphTestFramework.printResultsSummary(results)
    }
    
    func testLargeGraph() throws {
        // Test performance with a larger graph
        let g = Graph()
        
        // Create a chain of operations
        var lastNode = g.n(.constant(1.0))
        for i in 0..<10 {
            let multiplier = g.n(.constant(Float(i + 1) * 0.1))
            lastNode = g.n(.mul, lastNode, multiplier)
        }
        
        // Add some phasors
        let phasorFreq = g.n(.constant(1.0))
        let phasorReset = g.n(.constant(0.0))
        let phasor = g.n(.phasor(0), phasorFreq, phasorReset)
        
        // Combine with chain
        let combined = g.n(.add, lastNode, phasor)
        let output = g.n(.output(0), combined)
        
        let config = GraphTestConfig(duration: 0.2, frameSize: 64)  // Smaller duration for performance
        let results = GraphTestFramework.executeGraph(g, config: config)
        
        // Assert all backends succeed
        GraphTestFramework.assertAllSuccessful(results)
        
        // Assert outputs match between backends
        GraphTestFramework.assertOutputsMatch(results, tolerance: 0.01)
        
        GraphTestFramework.printResultsSummary(results)
    }
    
    func testMainSwiftGraphWithDebug() throws {
        // Exactly replicate main.swift graph with detailed debugging
        let g = Graph()
        
        let freq1 = g.n(.mul, g.n(.constant(90)), g.n(.constant(2.5)))
        let freq2 = g.n(.mul, g.n(.constant(2)), g.n(.constant(3.5)))
        let ph1 = g.n(.phasor(0), freq1, g.n(.constant(0)))
        let ph2 = g.n(.phasor(1), freq2, g.n(.constant(0)))
        let cond = g.n(.lt, ph2, g.n(.constant(0.1)))
        let latched = g.n(.latch(2), ph1, cond)
        let mult = g.n(.mul, latched, ph1)
        let out = g.n(.output(0), mult)
        
        // Generate the same compilation path as main.swift
        let sorted = topo(g)
        let scalar = scalarNodes(g)
        let blocks = determineBlocks(sorted: sorted, scalar: scalar, g: g, maxNodesPerBlock: 320)
        let sortedBlockIds = sortBlocksByDependencies(blocks, g)
        var sortedBlocks: [Block] = []
        sortedBlockIds.forEach { sortedBlocks.append(blocks[$0]) }
        
        var ctx = IRContext()
        var uopBlocks: [BlockUOps] = []
        
        for blockIdx in sortedBlockIds {
            let block = blocks[blockIdx]
            uopBlocks.append(BlockUOps(ops: emitBlockUOps(ctx: ctx, block: block, blocks: sortedBlocks, g: g, debug: false), kind: block.kind))
        }
        
        // Test both backends with debug output
        let backends: [(Device, String)] = [(.C, "C"), (.Metal, "Metal")]
        
        for (targetDevice, deviceName) in backends {
            print("\n🧪 Testing \(deviceName) backend with debug...")
            
            let renderer: Renderer = targetDevice == .Metal ? MetalRenderer() : CRenderer()
            let kernels = lowerUOpBlocks(&uopBlocks, renderer: renderer, ctx: ctx, frameCount: 128)
            
            print("📊 Generated \(kernels.count) kernel(s)")
            
            let runtime: CompiledKernelRuntime
            if targetDevice == .Metal {
                runtime = try MetalCompiledKernel(kernels: kernels)
            } else {
                let source = kernels.first?.source ?? ""
                let compiled = CCompiledKernel(source: source)
                try compiled.compileAndLoad()
                runtime = compiled
            }
            
            // Prepare output buffer
            let frameCount = 128
            let outputBuffer = UnsafeMutablePointer<Float>.allocate(capacity: frameCount)
            let inputBuffer = UnsafePointer<Float>(UnsafeMutablePointer<Float>.allocate(capacity: frameCount))
            defer {
                outputBuffer.deallocate()
                inputBuffer.deallocate()
            }
            
            // Initialize input to zeros
            for i in 0..<frameCount {
                UnsafeMutablePointer(mutating: inputBuffer)[i] = 0.0
            }
            
            // Run multiple times to check for non-deterministic behavior
            var allRuns: [[Float]] = []
            for runIndex in 0..<3 {
                // Clear output buffer
                for i in 0..<frameCount {
                    outputBuffer[i] = 0.0
                }
                
                runtime.run(outputs: outputBuffer, inputs: inputBuffer, frameCount: frameCount, volumeScale: 1.0)
                
                let outputs = Array(UnsafeBufferPointer(start: outputBuffer, count: frameCount))
                allRuns.append(outputs)
                
                print("🔍 \(deviceName) Run \(runIndex + 1):")
                print("   First 5: [\(outputs.prefix(5).map { String(format: "%.6f", $0) }.joined(separator: ", "))]")
                print("   Last 5:  [\(outputs.suffix(5).map { String(format: "%.6f", $0) }.joined(separator: ", "))]")
                print("   Min: \(String(format: "%.6f", outputs.min() ?? 0))")
                print("   Max: \(String(format: "%.6f", outputs.max() ?? 0))")
                
                // Check for ridiculous values
                let maxAbsValue = outputs.map { abs($0) }.max() ?? 0
                if maxAbsValue > 1000 {
                    print("⚠️  WARNING: Extremely large values detected! Max absolute: \(maxAbsValue)")
                }
                
                // Check for NaN or infinity
                let hasNaN = outputs.contains { $0.isNaN }
                let hasInf = outputs.contains { $0.isInfinite }
                if hasNaN {
                    print("⚠️  WARNING: NaN values detected!")
                }
                if hasInf {
                    print("⚠️  WARNING: Infinite values detected!")
                }
            }
            
            // Check determinism between runs
            let firstRun = allRuns[0]
            var isDeterministic = true
            for runIndex in 1..<allRuns.count {
                let currentRun = allRuns[runIndex]
                for i in 0..<min(firstRun.count, currentRun.count) {
                    if abs(firstRun[i] - currentRun[i]) > 0.000001 {
                        isDeterministic = false
                        print("⚠️  NON-DETERMINISTIC: Run 1 vs Run \(runIndex + 1), sample \(i): \(firstRun[i]) vs \(currentRun[i])")
                        break
                    }
                }
                if !isDeterministic { break }
            }
            
            if isDeterministic {
                print("✅ \(deviceName): All runs are deterministic")
            } else {
                print("❌ \(deviceName): NON-DETERMINISTIC BEHAVIOR DETECTED!")
            }
            
            // For Metal, read intermediate buffers
            if let metalRuntime = runtime as? MetalCompiledKernel {
                print("📊 Metal intermediate buffers:")
                metalRuntime.debugBufferStates()
            }
        }
    }
}