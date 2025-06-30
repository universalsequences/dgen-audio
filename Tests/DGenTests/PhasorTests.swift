import XCTest
@testable import DGen

final class PhasorTests: XCTestCase {
    
    func testSimplePhasor1Hz() throws {
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
        
        // Assert outputs match between backends
        GraphTestFramework.assertOutputsMatch(results, tolerance: 0.01)
        
        // Validate phasor behavior for each backend
        for result in results.filter({ $0.success }) {
            let tolerance: Float = 0.1
            
            // Check first few samples (should start near 0)
            for i in 0..<10 {
                XCTAssertLessThanOrEqual(abs(result.outputs[i]), tolerance, 
                    "\(result.deviceName): Sample \(i) should be near 0, got \(result.outputs[i])")
            }
            
            // Validate range [0, 1]
            GraphTestFramework.validateOutputRange(
                result.outputs,
                minValue: -tolerance,
                maxValue: 1.0 + tolerance,
                deviceName: result.deviceName
            )
            
            // Check that phasor increases over time (between frames)
            let frameSize = 128
            var foundIncrease = false
            for frameIdx in 1..<min(10, result.outputs.count / frameSize) {
                let currentFrameValue = result.outputs[frameIdx * frameSize]
                let previousFrameValue = result.outputs[(frameIdx - 1) * frameSize]
                if currentFrameValue > previousFrameValue {
                    foundIncrease = true
                    break
                }
            }
            XCTAssertTrue(foundIncrease, "\(result.deviceName): Phasor should increase over time")
        }
        
        GraphTestFramework.printResultsSummary(results)
    }
    
    func testUnconnectedPhasorNode() throws {
        print("test unconnected")
        // Create a graph with a connected phasor AND an unconnected phasor
        let g = Graph()
        let freq = g.n(.constant(1.0))  // 1Hz frequency
        let reset = g.n(.constant(0.0)) // No reset
        let phasor = g.n(.phasor(0), freq, reset)
        let output = g.n(.output(0), phasor)
        
        // Add an UNCONNECTED phasor that doesn't feed into the output
        let unconnectedFreq = g.n(.constant(120.0))  // 120Hz frequency
        let unconnectedReset = g.n(.constant(0.0))   // No reset
        let unconnectedPhasor = g.n(.phasor(1), unconnectedFreq, unconnectedReset)
        
        let config = GraphTestConfig(duration: 1.0, frameSize: 128, printKernelsOnError: true)  // Short test
        let results = GraphTestFramework.executeGraph(g, config: config)
        
        // Assert all backends succeed (this was the main issue)
        GraphTestFramework.assertAllSuccessful(results)
        
        // Assert outputs match between backends
        GraphTestFramework.assertOutputsMatch(results, tolerance: 0.01)
        
    }
}
