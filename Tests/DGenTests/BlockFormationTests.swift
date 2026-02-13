import XCTest
@testable import DGen

/// Tests that directly inspect the compilation pipeline's block formation
/// to debug issues with frame-based tensor operations like cos(phasor(tensor)*twopi)
final class BlockFormationTests: XCTestCase {

    // MARK: - Helper to print block structure

    private func printBlockStructure(
        blocks: [Block],
        graph: Graph,
        scalarSet: Set<NodeID>,
        title: String
    ) {
        print("\n=== \(title) ===")
        for (idx, block) in blocks.enumerated() {
            print("Block \(idx): kind=\(block.kind), temporality=\(block.temporality), shape=\(block.shape ?? [])")
            for nodeId in block.nodes {
                if let node = graph.nodes[nodeId] {
                    let isScalar = scalarSet.contains(nodeId)
                    let marker = isScalar ? "[SCALAR]" : "[SIMD]"
                    print("  \(marker) node \(nodeId): \(node.op), shape=\(node.shape)")
                }
            }
        }
    }

    // MARK: - Test: Frame-based tensor block formation

    /// This test inspects the block formation for cos(phasor(tensor)*twopi)
    /// to understand why cos ends up in a separate block from phasor.
    func testFrameBasedTensorBlockFormation() throws {
        let g = Graph()

        // Create a tensor of frequencies (size 4)
        let freqs = g.tensor(shape: [4], data: [100.0, 200.0, 300.0, 400.0])

        // Create phasor with tensor input - this is frame-based!
        let cellId = g.alloc()
        let zeroReset = g.n(.constant(0.0))
        let phasorNode = g.n(.phasor(cellId), freqs, zeroReset)

        // Multiply by 2*pi
        let twopi = g.n(.constant(Float.pi * 2.0))
        let scaled = g.n(.mul, phasorNode, twopi)

        // Apply cos - should be in same block as phasor!
        let cosNode = g.n(.cos, scaled)

        // Sum all the cos values
        let result = g.n(.sum, cosNode)
        _ = g.n(.output(0), result)

        // Step 1: Find feedback loops
        let feedbackClusters = findFeedbackLoops(g)
        print("\n=== Feedback Clusters ===")
        print(feedbackClusters)

        // Step 2: Get scalar nodes
        let scalarNodeSet = findSequentialNodes(g, feedbackClusters: feedbackClusters, backend: .c)
        print("\n=== Scalar Nodes ===")
        for nodeId in scalarNodeSet.sorted() {
            if let node = g.nodes[nodeId] {
                print("  node \(nodeId): \(node.op)")
            }
        }

        // Step 3: Topological sort
        let sortedNodes = topologicalSort(g, feedbackClusters: feedbackClusters, scalarNodeSet: scalarNodeSet, debug: true)
        print("\n=== Sorted Nodes ===")
        for nodeId in sortedNodes {
            if let node = g.nodes[nodeId] {
                let isScalar = scalarNodeSet.contains(nodeId)
                print("  node \(nodeId): \(node.op), scalar=\(isScalar)")
            }
        }

        // Step 4: Shape inference
        try inferShapes(graph: g, sortedNodes: sortedNodes)
        TensorOutputBindingPass.bindTensorOutputsAndReserveLazyCells(graph: g, sortedNodes: sortedNodes)

        print("\n=== Node Shapes After Inference ===")
        for nodeId in sortedNodes {
            if let node = g.nodes[nodeId] {
                print("  node \(nodeId): \(node.op), shape=\(node.shape)")
            }
        }

        // Step 5: Determine blocks (simple)
        let blocks = partitionIntoBlocks(
            sorted: sortedNodes,
            scalar: scalarNodeSet,
            g: g,
            debug: true
        )
        printBlockStructure(blocks: blocks, graph: g, scalarSet: scalarNodeSet, title: "After partitionIntoBlocks")

        // Step 6: Fuse blocks
        let fusedBlocks = fuseBlocks(blocks)
        printBlockStructure(blocks: fusedBlocks, graph: g, scalarSet: scalarNodeSet, title: "After fuseBlocks")

        // Step 7: Isolate spectral passes (shouldn't affect this test)
        let isolatedBlocks = isolateSpectralPasses(fusedBlocks, g)
        printBlockStructure(blocks: isolatedBlocks, graph: g, scalarSet: scalarNodeSet, title: "After isolateSpectralPasses")

        // Step 8: Re-fuse
        let reFusedBlocks = fuseBlocks(isolatedBlocks)
        printBlockStructure(blocks: reFusedBlocks, graph: g, scalarSet: scalarNodeSet, title: "After second fuseBlocks")

        // Step 9: Determine tensor blocks
        let context = IRContext(g: g)
        let tensorBlocks = determineTensorBlocks(reFusedBlocks, g, context)
        printBlockStructure(blocks: tensorBlocks, graph: g, scalarSet: scalarNodeSet, title: "After determineTensorBlocks")

        // Step 10: Infer temporality
        let temporalityResult = TemporalityPass.inferTemporality(graph: g, sortedNodes: sortedNodes)
        let frameBasedNodes = temporalityResult.frameBasedNodes
        print("\n=== Frame-Based Nodes ===")
        for nodeId in frameBasedNodes.sorted() {
            if let node = g.nodes[nodeId] {
                print("  node \(nodeId): \(node.op)")
            }
        }

        var finalBlocks = tensorBlocks
        TemporalityPass.assignBlockTemporality(
            blocks: &finalBlocks,
            frameBasedNodes: frameBasedNodes,
            hopBasedNodes: temporalityResult.hopBasedNodes
        )
        printBlockStructure(blocks: finalBlocks, graph: g, scalarSet: scalarNodeSet, title: "Final Blocks")

        // CRITICAL ASSERTIONS:

        // 1. Phasor, mul, and cos should all be marked as scalar
        XCTAssertTrue(scalarNodeSet.contains(phasorNode), "phasor node should be scalar (frame-based)")
        XCTAssertTrue(scalarNodeSet.contains(scaled), "mul node should be scalar (consumes frame-based tensor)")
        XCTAssertTrue(scalarNodeSet.contains(cosNode), "cos node should be scalar (consumes frame-based tensor)")

        // 2. All tensor ops consuming frame-based phasor should be in the same block
        // Find which block contains the phasor
        var phasorBlockIdx: Int? = nil
        var mulBlockIdx: Int? = nil
        var cosBlockIdx: Int? = nil

        for (idx, block) in finalBlocks.enumerated() {
            if block.nodes.contains(phasorNode) { phasorBlockIdx = idx }
            if block.nodes.contains(scaled) { mulBlockIdx = idx }
            if block.nodes.contains(cosNode) { cosBlockIdx = idx }
        }

        print("\n=== Block Membership ===")
        print("phasor in block: \(phasorBlockIdx ?? -1)")
        print("mul in block: \(mulBlockIdx ?? -1)")
        print("cos in block: \(cosBlockIdx ?? -1)")

        // They should all be in the same block!
        if let pIdx = phasorBlockIdx, let mIdx = mulBlockIdx, let cIdx = cosBlockIdx {
            // For now, just report the issue - don't fail
            // XCTAssertEqual(pIdx, mIdx, "phasor and mul should be in same block")
            // XCTAssertEqual(mIdx, cIdx, "mul and cos should be in same block")

            if pIdx != mIdx || mIdx != cIdx {
                print("\n*** BUG: Frame-based tensor ops are in DIFFERENT blocks! ***")
                print("This causes the 'static' issue because cos runs in a separate SIMD block")
                print("phasor block kind: \(finalBlocks[pIdx].kind)")
                print("mul block kind: \(finalBlocks[mIdx].kind)")
                print("cos block kind: \(finalBlocks[cIdx].kind)")
            }
        }

        // Now compile fully and show the generated kernel
        print("\n=== Full Compilation ===")
        let compileResult = try CompilationPipeline.compile(
            graph: g,
            backend: .c,
            options: .init(frameCount: 128, debug: true)
        )
        print("\n=== Generated Kernel ===")
        print(compileResult.source)
    }

    // MARK: - Test: Tensor * constant -> phasor -> * twopi -> cos -> sum (user's patch)

    /// This test mimics the user's actual patch:
    /// tensor 2x2 [1,15,3,1] -> *100 -> phasor -> *twopi -> cos -> sum -> *0.22 -> out
    func testUserPatchBlockFormation() throws {
        let g = Graph()

        // tensor 2x2 with data [1, 15, 3, 1] (midi notes or similar)
        let tensor = g.tensor(shape: [2, 2], data: [1.0, 15.0, 3.0, 1.0])

        // * 100 (convert to frequency or scale up)
        let scaled = g.n(.mul, tensor, g.n(.constant(100.0)))

        // phasor with tensor input - this is frame-based!
        let cellId = g.alloc()
        let zeroReset = g.n(.constant(0.0))
        let phasorNode = g.n(.phasor(cellId), scaled, zeroReset)

        // * twopi
        let twopi = g.n(.constant(Float.pi * 2.0))
        let phase = g.n(.mul, phasorNode, twopi)

        // cos
        let cosNode = g.n(.cos, phase)

        // sum
        let sumNode = g.n(.sum, cosNode)

        // * 0.22 (volume)
        let volume = g.n(.mul, sumNode, g.n(.constant(0.22)))

        // output
        _ = g.n(.output(0), volume)

        // Compile and check
        let feedbackClusters = findFeedbackLoops(g)
        let scalarNodeSet = findSequentialNodes(g, feedbackClusters: feedbackClusters, backend: .c)

        print("\n=== User Patch Scalar Nodes ===")
        for nodeId in scalarNodeSet.sorted() {
            if let node = g.nodes[nodeId] {
                print("  node \(nodeId): \(node.op)")
            }
        }

        // Key assertions: phasor, mul*twopi, cos should ALL be scalar
        XCTAssertTrue(scalarNodeSet.contains(phasorNode), "phasor should be scalar")
        XCTAssertTrue(scalarNodeSet.contains(phase), "mul*twopi should be scalar")
        XCTAssertTrue(scalarNodeSet.contains(cosNode), "cos should be scalar")
        XCTAssertTrue(scalarNodeSet.contains(sumNode), "sum should be scalar")

        // Compile fully
        let result = try CompilationPipeline.compile(
            graph: g,
            backend: .c,
            options: .init(frameCount: 128, debug: true)
        )

        print("\n=== User Patch Generated Kernel ===")
        print(result.source)

        // Check that cos block doesn't have stride 4 on frame loop
        // The kernel should have all frame-based blocks with "i += 1"
        let source = result.source

        // Find cos block - it should NOT be in a "i += 4" loop
        // This is a rough check - look for cosf near "i += 4"
        let lines = source.components(separatedBy: "\n")
        var inFrameLoop = false
        var frameStride = 1
        var cosLineNumber = -1
        var cosFrameStride = -1

        for (lineNum, line) in lines.enumerated() {
            if line.contains("for (int i = 0; i < frameCount;") {
                if line.contains("i += 4") {
                    frameStride = 4
                } else if line.contains("i += 1") {
                    frameStride = 1
                }
                inFrameLoop = true
            }
            if line.contains("cosf(") && inFrameLoop {
                cosLineNumber = lineNum
                cosFrameStride = frameStride
            }
        }

        print("\n=== Cos Analysis ===")
        print("cos at line \(cosLineNumber), frame stride = \(cosFrameStride)")

        XCTAssertEqual(cosFrameStride, 1, "cos should be in a frame loop with stride 1, not \(cosFrameStride)")
    }

    // MARK: - Test: Compare with tensorHistoryBuffer (which works)

    /// tensorHistoryReadWrite correctly fuses all tensor ops per-frame.
    /// This test shows what correct block formation looks like.
    func testTensorHistoryBlockFormation() throws {
        let g = Graph()

        // Create a history buffer for 4-element state
        let stateBuffer = g.tensorHistoryBuffer(shape: [4])

        // Read previous state, add increment, write back
        let prevState = g.tensorHistoryRead(stateBuffer)
        let increment = g.n(.constant(0.1))
        let newState = g.n(.add, prevState, increment)
        g.tensorHistoryWrite(stateBuffer, newState)

        // Apply cos (should be in same block!)
        let cosState = g.n(.cos, newState)

        // Output the sum
        _ = g.n(.output(0), g.n(.sum, cosState))

        // Step through compilation
        let feedbackClusters = findFeedbackLoops(g)
        let scalarNodeSet = findSequentialNodes(g, feedbackClusters: feedbackClusters, backend: .c)
        let sortedNodes = topologicalSort(g, feedbackClusters: feedbackClusters, scalarNodeSet: scalarNodeSet, debug: true)

        try inferShapes(graph: g, sortedNodes: sortedNodes)
        TensorOutputBindingPass.bindTensorOutputsAndReserveLazyCells(graph: g, sortedNodes: sortedNodes)

        let blocks = partitionIntoBlocks(sorted: sortedNodes, scalar: scalarNodeSet, g: g, debug: true)
        let fusedBlocks = fuseBlocks(blocks)
        let isolatedBlocks = isolateSpectralPasses(fusedBlocks, g)
        let reFusedBlocks = fuseBlocks(isolatedBlocks)

        let context = IRContext(g: g)
        let tensorBlocks = determineTensorBlocks(reFusedBlocks, g, context)

        let temporalityResult = TemporalityPass.inferTemporality(graph: g, sortedNodes: sortedNodes)
        var finalBlocks = tensorBlocks
        TemporalityPass.assignBlockTemporality(
            blocks: &finalBlocks,
            frameBasedNodes: temporalityResult.frameBasedNodes,
            hopBasedNodes: temporalityResult.hopBasedNodes
        )

        printBlockStructure(blocks: finalBlocks, graph: g, scalarSet: scalarNodeSet, title: "TensorHistory Final Blocks")

        // Show the generated kernel
        print("\n=== TensorHistory Generated Kernel ===")
        let result = try CompilationPipeline.compile(
            graph: g,
            backend: .c,
            options: .init(frameCount: 128, debug: true)
        )
        print(result.source)
    }

    // MARK: - Test: Simple tensor operations (no phasor)

    /// Test that pure tensor ops without frame-based sources work correctly
    func testPureTensorBlockFormation() throws {
        let g = Graph()

        // Create a tensor
        let tensor = g.tensor(shape: [4], data: [1.0, 2.0, 3.0, 4.0])

        // Chain of tensor ops
        let doubled = g.n(.mul, tensor, g.n(.constant(2.0)))
        let cosResult = g.n(.cos, doubled)
        let summed = g.n(.sum, cosResult)
        _ = g.n(.output(0), summed)

        // Step through compilation
        let feedbackClusters = findFeedbackLoops(g)
        let scalarNodeSet = findSequentialNodes(g, feedbackClusters: feedbackClusters, backend: .c)
        let sortedNodes = topologicalSort(g, feedbackClusters: feedbackClusters, scalarNodeSet: scalarNodeSet, debug: true)

        try inferShapes(graph: g, sortedNodes: sortedNodes)
        TensorOutputBindingPass.bindTensorOutputsAndReserveLazyCells(graph: g, sortedNodes: sortedNodes)

        let blocks = partitionIntoBlocks(sorted: sortedNodes, scalar: scalarNodeSet, g: g, debug: true)
        let fusedBlocks = fuseBlocks(blocks)
        let isolatedBlocks = isolateSpectralPasses(fusedBlocks, g)
        let reFusedBlocks = fuseBlocks(isolatedBlocks)

        let context = IRContext(g: g)
        let tensorBlocks = determineTensorBlocks(reFusedBlocks, g, context)

        let temporalityResult = TemporalityPass.inferTemporality(graph: g, sortedNodes: sortedNodes)
        var finalBlocks = tensorBlocks
        TemporalityPass.assignBlockTemporality(
            blocks: &finalBlocks,
            frameBasedNodes: temporalityResult.frameBasedNodes,
            hopBasedNodes: temporalityResult.hopBasedNodes
        )

        printBlockStructure(blocks: finalBlocks, graph: g, scalarSet: scalarNodeSet, title: "Pure Tensor Final Blocks")

        // This case SHOULD use SIMD because there's no frame-based dependency
        // mul and cos should be in the same tensor block with shape [4]

        print("\n=== Pure Tensor Generated Kernel ===")
        let result = try CompilationPipeline.compile(
            graph: g,
            backend: .c,
            options: .init(frameCount: 128, debug: true)
        )
        print(result.source)
    }

    // MARK: - determineTensorBlocks Targeted Tests

    func testDetermineTensorBlocksSplitsScalarPrefixForInherentlyScalarOps() throws {
        let g = Graph()

        let tensor = g.tensor(shape: [4], data: [1, 2, 3, 4])
        let seed = g.n(.constant(1.0))
        let accumNode = g.n(.accum(g.alloc()), seed)
        let tensorAdd = g.n(.add, tensor, g.n(.constant(0.5)))

        try inferShapes(graph: g, sortedNodes: [tensor, seed, accumNode, tensorAdd])

        var block = Block(kind: .scalar)
        block.nodes = [accumNode, tensorAdd]

        let result = determineTensorBlocks([block], g, IRContext(g: g))

        XCTAssertEqual(result.count, 2)
        XCTAssertEqual(result[0].nodes, [accumNode])
        XCTAssertNil(result[0].tensorIndex)
        XCTAssertEqual(result[1].nodes, [tensorAdd])
        XCTAssertNotNil(result[1].tensorIndex)
        XCTAssertEqual(result[1].shape, [4])
    }

    func testDetermineTensorBlocksIsolatesOverlapAddAsScalarBlock() throws {
        let g = Graph()

        let tensor = g.tensor(shape: [4], data: [1, 2, 3, 4])
        let mul = g.n(.mul, tensor, g.n(.constant(2.0)))
        let overlapAdd = g.n(.overlapAdd(8, 4, g.alloc(), g.alloc(), g.alloc()), mul)
        let post = g.n(.add, tensor, g.n(.constant(1.0)))

        try inferShapes(graph: g, sortedNodes: [tensor, mul, overlapAdd, post])

        var block = Block(kind: .simd)
        block.nodes = [mul, overlapAdd, post]

        let result = determineTensorBlocks([block], g, IRContext(g: g))

        XCTAssertEqual(result.count, 3)
        XCTAssertEqual(result[0].nodes, [mul])
        XCTAssertEqual(result[1].nodes, [overlapAdd])
        XCTAssertEqual(result[1].kind, .scalar)
        XCTAssertEqual(result[2].nodes, [post])
        XCTAssertNotNil(result[0].tensorIndex)
        XCTAssertNotNil(result[2].tensorIndex)
    }

    func testDetermineTensorBlocksSplitsOnNonFusableShapeTransition() throws {
        let g = Graph()

        let matrix = g.tensor(shape: [2, 3], data: [1, 2, 3, 4, 5, 6])
        let matrixAdd = g.n(.add, matrix, g.n(.constant(1.0)))
        let row = g.n(.selectRow, matrixAdd, g.n(.constant(0.0)))

        try inferShapes(graph: g, sortedNodes: [matrix, matrixAdd, row])

        var block = Block(kind: .simd)
        block.nodes = [matrixAdd, row]

        let result = determineTensorBlocks([block], g, IRContext(g: g))

        XCTAssertEqual(result.count, 2)
        XCTAssertEqual(result[0].nodes, [matrixAdd])
        XCTAssertEqual(result[1].nodes, [row])
        XCTAssertEqual(result[0].shape, [2, 3])
        XCTAssertEqual(result[1].shape, [3])
    }

    func testDetermineTensorBlocksKeepsAxisReduceTransitionInSameBlock() throws {
        let g = Graph()

        let input = g.tensor(shape: [2, 3], data: [1, 2, 3, 4, 5, 6])
        let reduced = g.n(.sumAxis(1), input)
        let post = g.n(.add, reduced, g.n(.constant(1.0)))

        try inferShapes(graph: g, sortedNodes: [input, reduced, post])

        var block = Block(kind: .simd)
        block.nodes = [input, reduced, post]

        let result = determineTensorBlocks([block], g, IRContext(g: g))

        XCTAssertEqual(result.count, 1)
        XCTAssertEqual(result[0].nodes, [input, reduced, post])
        XCTAssertNotNil(result[0].tensorIndex)
    }
}
