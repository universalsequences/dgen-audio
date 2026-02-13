import XCTest

@testable import DGen

final class ShapeInferenceTests: XCTestCase {

    // MARK: - Shape Inference Tests

    func testScalarConstantInfersScalar() throws {
        let g = Graph()
        _ = g.n(.constant(1.0))

        let shape = try inferShape(op: .constant(1.0), inputs: [], graph: g)
        XCTAssertEqual(shape, .scalar)
    }

    func testTensorRefInfersTensorShape() throws {
        let g = Graph()
        let t = g.tensor(shape: [3, 4], data: nil)

        // Get the tensorRef op from the node
        guard let node = g.nodes[t], case .tensorRef(let tid) = node.op else {
            XCTFail("Expected tensorRef node")
            return
        }

        let shape = try inferShape(op: .tensorRef(tid), inputs: [], graph: g)
        XCTAssertEqual(shape, .tensor([3, 4]))
    }

    func testTensorRefMissingTensorThrows() throws {
        let g = Graph()

        // Try to infer shape for non-existent tensor ID
        XCTAssertThrowsError(try inferShape(op: .tensorRef(999), inputs: [], graph: g)) { error in
            guard case DGenError.missingTensorID = error else {
                XCTFail("Expected missingTensorID error, got \(error)")
                return
            }
        }
    }

    func testAddScalarAndScalarInfersScalar() throws {
        let g = Graph()

        let shape = try inferShape(op: .add, inputs: [.scalar, .scalar], graph: g)
        XCTAssertEqual(shape, .scalar)
    }

    func testAddTensorAndScalarInfersTensor() throws {
        let g = Graph()

        let shape = try inferShape(op: .add, inputs: [.tensor([2, 3]), .scalar], graph: g)
        XCTAssertEqual(shape, .tensor([2, 3]))
    }

    func testAddScalarAndTensorInfersTensor() throws {
        let g = Graph()

        let shape = try inferShape(op: .add, inputs: [.scalar, .tensor([4, 5])], graph: g)
        XCTAssertEqual(shape, .tensor([4, 5]))
    }

    func testAddTensorAndTensorSameShapeInfersTensor() throws {
        let g = Graph()

        let shape = try inferShape(
            op: .add, inputs: [.tensor([2, 2]), .tensor([2, 2])], graph: g)
        XCTAssertEqual(shape, .tensor([2, 2]))
    }

    func testAddTensorAndTensorMismatchedShapeThrows() throws {
        let g = Graph()

        XCTAssertThrowsError(
            try inferShape(op: .add, inputs: [.tensor([2, 3]), .tensor([3, 2])], graph: g)
        ) { error in
            guard case DGenError.shapeMismatch(op: _, shape1: _, shape2: _) = error else {
                XCTFail("Expected shapeMismatch error, got \(error)")
                return
            }
        }
    }

    func testSumReduceAlwaysInfersScalar() throws {
        let g = Graph()

        // Sum of a tensor should be scalar
        let shape = try inferShape(op: .sum, inputs: [.tensor([4, 4])], graph: g)
        XCTAssertEqual(shape, .scalar)
    }

    func testConv2dPreservesInputShape() throws {
        let g = Graph()

        let shape = try inferShape(op: .conv2d([3, 3]), inputs: [.tensor([8, 8])], graph: g)
        XCTAssertEqual(shape, .tensor([8, 8]))
    }

    func testConv2dMissingInputThrows() throws {
        let g = Graph()

        XCTAssertThrowsError(try inferShape(op: .conv2d([3, 3]), inputs: [], graph: g)) { error in
            guard case DGenError.shapeInferenceFailed(op: let op, reason: _) = error else {
                XCTFail("Expected shapeInferenceFailed error, got \(error)")
                return
            }
            XCTAssertEqual(op, "conv2d")
        }
    }

    func testHistoryReadWithTensorCellInfersTensorShape() throws {
        let g = Graph()
        let buffer = g.tensorHistoryBuffer(shape: [4, 4])

        // historyRead with a tensor cell returns tensor shape
        let shape = try inferShape(op: .historyRead(buffer.cellId), inputs: [], graph: g)
        XCTAssertEqual(shape, .tensor([4, 4]))
    }

    func testHistoryReadWithScalarCellInfersScalar() throws {
        let g = Graph()
        let scalarCell = g.alloc()  // Scalar cell, not in cellToTensor

        // historyRead with a scalar cell returns scalar
        let shape = try inferShape(op: .historyRead(scalarCell), inputs: [], graph: g)
        XCTAssertEqual(shape, .scalar)
    }

    func testHistoryWritePassthroughShape() throws {
        let g = Graph()

        let shape = try inferShape(
            op: .historyWrite(0), inputs: [.tensor([3, 3])], graph: g)
        XCTAssertEqual(shape, .tensor([3, 3]))
    }

    func testHistoryWriteMissingInputThrows() throws {
        let g = Graph()

        XCTAssertThrowsError(try inferShape(op: .historyWrite(0), inputs: [], graph: g)) {
            error in
            guard case DGenError.shapeInferenceFailed(op: let op, reason: _) = error else {
                XCTFail("Expected shapeInferenceFailed error, got \(error)")
                return
            }
            XCTAssertEqual(op, "historyWrite")
        }
    }

    // MARK: - Full Graph Shape Inference Tests

    func testInferShapesForSimpleGraph() throws {
        let g = Graph()

        // Create: tensor([2,2]) + scalar -> sum -> output
        let t = g.tensor(shape: [2, 2], data: [1, 2, 3, 4])
        let s = g.n(.constant(1.0))
        let added = g.n(.add, t, s)
        let summed = g.n(.sum, added)
        _ = g.n(.output(0), summed)

        let sortedNodes = [t, s, added, summed]
        try inferShapes(graph: g, sortedNodes: sortedNodes)

        // Verify shapes were assigned
        XCTAssertEqual(g.nodes[t]?.shape, .tensor([2, 2]))
        XCTAssertEqual(g.nodes[s]?.shape, .scalar)
        XCTAssertEqual(g.nodes[added]?.shape, .tensor([2, 2]))
        XCTAssertEqual(g.nodes[summed]?.shape, .scalar)
    }

    func testInferShapesForTensorHistoryGraph() throws {
        let g = Graph()

        // Create: historyRead -> add scalar -> historyWrite
        let buffer = g.tensorHistoryBuffer(shape: [4])
        let state = g.tensorHistoryRead(buffer)
        let increment = g.n(.constant(1.0))
        let newState = g.n(.add, state, increment)
        let writeNode = g.tensorHistoryWrite(buffer, newState)

        let sortedNodes = [state, increment, newState, writeNode]
        try inferShapes(graph: g, sortedNodes: sortedNodes)

        XCTAssertEqual(g.nodes[state]?.shape, .tensor([4]))
        XCTAssertEqual(g.nodes[increment]?.shape, .scalar)
        XCTAssertEqual(g.nodes[newState]?.shape, .tensor([4]))
        XCTAssertEqual(g.nodes[writeNode]?.shape, .tensor([4]))
    }

    // MARK: - Compilation Pipeline Integration Tests

    func testCompilationPipelineReinfersMissingNodeShapes() throws {
        let g = Graph()

        let t = g.tensor(shape: [2, 2], data: [1, 2, 3, 4])
        let s = g.n(.constant(1.0))
        let add = g.n(.add, t, s)
        let summed = g.n(.sum, add)
        let out = g.n(.output(0), summed)

        // Simulate stale/missing shape metadata before compilation.
        for nodeId in [t, s, add, summed, out] {
            guard var node = g.nodes[nodeId] else {
                XCTFail("Missing node \(nodeId)")
                return
            }
            node.shape = nil
            g.nodes[nodeId] = node
        }

        XCTAssertNil(g.nodes[t]?.shape)
        XCTAssertNil(g.nodes[add]?.shape)

        _ = try CompilationPipeline.compile(
            graph: g,
            backend: .c,
            options: .init(frameCount: 1, debug: false)
        )

        XCTAssertEqual(g.nodes[t]?.shape, .tensor([2, 2]))
        XCTAssertEqual(g.nodes[s]?.shape, .scalar)
        XCTAssertEqual(g.nodes[add]?.shape, .tensor([2, 2]))
        XCTAssertEqual(g.nodes[summed]?.shape, .scalar)
        XCTAssertEqual(g.nodes[out]?.shape, .scalar)
    }

    // MARK: - Temporality Inference Tests

    func testPhasorIsFrameBased() {
        XCTAssertTrue(TemporalityPass.isIntrinsicallyFrameBased(.phasor(0)))
    }

    func testConstantIsNotFrameBased() {
        XCTAssertFalse(TemporalityPass.isIntrinsicallyFrameBased(.constant(1.0)))
    }

    func testHistoryReadIsFrameBased() {
        XCTAssertTrue(TemporalityPass.isIntrinsicallyFrameBased(.historyRead(0)))
    }

    func testTemporalityPropagates() {
        let g = Graph()

        // Create: phasor -> mul scalar -> output
        // phasor is frame-based, so mul should also be frame-based
        let freq = g.n(.constant(440.0))
        let phasorCell = g.alloc()
        let phasor = g.n(.phasor(phasorCell), freq)
        let scale = g.n(.constant(0.5))
        let scaled = g.n(.mul, phasor, scale)

        let sortedNodes = [freq, phasor, scale, scaled]
        let temporalityResult = TemporalityPass.inferTemporality(graph: g, sortedNodes: sortedNodes)
        let frameBased = temporalityResult.frameBasedNodes

        XCTAssertFalse(frameBased.contains(freq))  // constant is static
        XCTAssertTrue(frameBased.contains(phasor))  // phasor is frame-based
        XCTAssertFalse(frameBased.contains(scale))  // constant is static
        XCTAssertTrue(frameBased.contains(scaled))  // has frame-based input
    }

    // MARK: - Elementwise Operations Shape Tests

    func testMulInheritsShape() throws {
        let g = Graph()
        let shape = try inferShape(op: .mul, inputs: [.tensor([5, 5]), .scalar], graph: g)
        XCTAssertEqual(shape, .tensor([5, 5]))
    }

    func testSubInheritsShape() throws {
        let g = Graph()
        let shape = try inferShape(op: .sub, inputs: [.scalar, .tensor([3])], graph: g)
        XCTAssertEqual(shape, .tensor([3]))
    }

    func testDivInheritsShape() throws {
        let g = Graph()
        let shape = try inferShape(
            op: .div, inputs: [.tensor([2, 2]), .tensor([2, 2])], graph: g)
        XCTAssertEqual(shape, .tensor([2, 2]))
    }

    func testSinInheritsShape() throws {
        let g = Graph()
        let shape = try inferShape(op: .sin, inputs: [.tensor([10])], graph: g)
        XCTAssertEqual(shape, .tensor([10]))
    }

    func testExpInheritsShape() throws {
        let g = Graph()
        let shape = try inferShape(op: .exp, inputs: [.tensor([3, 3])], graph: g)
        XCTAssertEqual(shape, .tensor([3, 3]))
    }

    // MARK: - Broadcast Helper Tests

    func testBroadcastShapesSameRankSucceeds() {
        XCTAssertEqual(
            broadcastShapes([2, 1, 3], [1, 2, 3]),
            [2, 2, 3]
        )
    }

    func testBroadcastShapesDifferentRankSucceeds() {
        XCTAssertEqual(
            broadcastShapes([5, 1, 4], [1, 4]),
            [5, 1, 4]
        )
    }

    func testBroadcastShapesIncompatibleReturnsNil() {
        XCTAssertNil(broadcastShapes([2, 3], [3, 2]))
    }

    // MARK: - Axis Reduction Shape Tests

    func testSumAxisRemovesSpecifiedAxis() throws {
        let g = Graph()
        let shape = try inferShape(op: .sumAxis(1), inputs: [.tensor([2, 3, 4])], graph: g)
        XCTAssertEqual(shape, .tensor([2, 4]))
    }

    func testSumAxisNegativeAxisIsNormalized() throws {
        let g = Graph()
        let shape = try inferShape(op: .sumAxis(-1), inputs: [.tensor([2, 3, 4])], graph: g)
        XCTAssertEqual(shape, .tensor([2, 3]))
    }

    func testSumAxisSingleDimensionReturnsScalar() throws {
        let g = Graph()
        let shape = try inferShape(op: .sumAxis(0), inputs: [.tensor([8])], graph: g)
        XCTAssertEqual(shape, .scalar)
    }

    func testSumAxisOutOfRangeThrows() {
        let g = Graph()
        XCTAssertThrowsError(try inferShape(op: .sumAxis(2), inputs: [.tensor([2, 3])], graph: g)) {
            error in
            guard case DGenError.shapeInferenceFailed(op: let op, reason: _) = error else {
                XCTFail("Expected shapeInferenceFailed, got \(error)")
                return
            }
            XCTAssertEqual(op, "sumAxis")
        }
    }

    func testMaxAxisRemovesSpecifiedAxis() throws {
        let g = Graph()
        let shape = try inferShape(op: .maxAxis(0), inputs: [.tensor([6, 7, 8])], graph: g)
        XCTAssertEqual(shape, .tensor([7, 8]))
    }

    func testMeanAxisNegativeAxisIsNormalized() throws {
        let g = Graph()
        let shape = try inferShape(op: .meanAxis(-2), inputs: [.tensor([6, 7, 8])], graph: g)
        XCTAssertEqual(shape, .tensor([6, 8]))
    }

    // MARK: - View And Transform Shape Tests

    func testReshapeInfersTargetShape() throws {
        let g = Graph()
        let shape = try inferShape(op: .reshape([3, 2]), inputs: [.tensor([2, 3])], graph: g)
        XCTAssertEqual(shape, .tensor([3, 2]))
    }

    func testAsStridedInfersTargetShape() throws {
        let g = Graph()
        let shape = try inferShape(op: .asStrided([2, 2], [2, 1]), inputs: [.tensor([4])], graph: g)
        XCTAssertEqual(shape, .tensor([2, 2]))
    }

    func testTransposeWithExplicitAxesPermutesShape() throws {
        let g = Graph()
        let shape = try inferShape(op: .transpose([1, 0, 2]), inputs: [.tensor([2, 3, 4])], graph: g)
        XCTAssertEqual(shape, .tensor([3, 2, 4]))
    }

    func testTransposeWithEmptyAxesReversesShape() throws {
        let g = Graph()
        let shape = try inferShape(op: .transpose([]), inputs: [.tensor([2, 3, 4])], graph: g)
        XCTAssertEqual(shape, .tensor([4, 3, 2]))
    }

    func testShrinkInfersSlicedShape() throws {
        let g = Graph()
        let shape = try inferShape(
            op: .shrink([(1, 4), nil, (10, 20)]),
            inputs: [.tensor([10, 20, 30])],
            graph: g
        )
        XCTAssertEqual(shape, .tensor([3, 20, 10]))
    }

    func testPadInfersExpandedShape() throws {
        let g = Graph()
        let shape = try inferShape(
            op: .pad([(1, 2), (3, 4)]),
            inputs: [.tensor([3, 4])],
            graph: g
        )
        XCTAssertEqual(shape, .tensor([6, 11]))
    }

    func testExpandViewInfersTargetShape() throws {
        let g = Graph()
        let shape = try inferShape(op: .expandView([4, 8]), inputs: [.tensor([1, 8])], graph: g)
        XCTAssertEqual(shape, .tensor([4, 8]))
    }

    func testRepeatViewInfersTiledShape() throws {
        let g = Graph()
        let shape = try inferShape(op: .repeatView([4, 5]), inputs: [.tensor([2, 3])], graph: g)
        XCTAssertEqual(shape, .tensor([8, 15]))
    }

    // MARK: - Tensor Select And Peek Shape Tests

    func testPeekAlwaysInfersScalar() throws {
        let g = Graph()
        let shape = try inferShape(op: .peek, inputs: [.tensor([8, 16]), .scalar, .scalar], graph: g)
        XCTAssertEqual(shape, .scalar)
    }

    func testSelectRowInfersRowShape() throws {
        let g = Graph()
        let shape = try inferShape(op: .selectRow, inputs: [.tensor([8, 16]), .scalar], graph: g)
        XCTAssertEqual(shape, .tensor([16]))
    }

    func testSelectRowWithNon2DTensorThrows() {
        let g = Graph()
        XCTAssertThrowsError(try inferShape(op: .selectRow, inputs: [.tensor([16]), .scalar], graph: g)) {
            error in
            guard case DGenError.shapeInferenceFailed(op: let op, reason: _) = error else {
                XCTFail("Expected shapeInferenceFailed, got \(error)")
                return
            }
            XCTAssertEqual(op, "selectRow")
        }
    }

    func testPeekRowInlineInfersNumCols() throws {
        let g = Graph()
        let shape = try inferShape(
            op: .peekRowInline(scratchCell: 99, numRows: 8, numCols: 16),
            inputs: [.tensor([8, 16]), .scalar],
            graph: g
        )
        XCTAssertEqual(shape, .tensor([16]))
    }

    // MARK: - Misc Shape Rules

    func testSeqReturnsLastInputShape() throws {
        let g = Graph()
        let shape = try inferShape(
            op: .seq,
            inputs: [.tensor([2, 2]), .scalar, .tensor([4])],
            graph: g
        )
        XCTAssertEqual(shape, .tensor([4]))
    }

    func testSeqWithNoInputsReturnsScalar() throws {
        let g = Graph()
        let shape = try inferShape(op: .seq, inputs: [], graph: g)
        XCTAssertEqual(shape, .scalar)
    }

    func testTensorAccumulateInfersScalar() throws {
        let g = Graph()
        let shape = try inferShape(op: .tensorAccumulate(42), inputs: [.tensor([8])], graph: g)
        XCTAssertEqual(shape, .scalar)
    }

    func testNegInfersInputShape() throws {
        let g = Graph()
        let shape = try inferShape(op: .neg, inputs: [.tensor([3, 5])], graph: g)
        XCTAssertEqual(shape, .tensor([3, 5]))
    }

    func testExpandInfersTargetShape() throws {
        let g = Graph()
        let shape = try inferShape(op: .expand([4, 8]), inputs: [.scalar], graph: g)
        XCTAssertEqual(shape, .tensor([4, 8]))
    }

    func testExpandAxisInfersTargetShape() throws {
        let g = Graph()
        let shape = try inferShape(op: .expandAxis([4, 8], 1), inputs: [.tensor([4])], graph: g)
        XCTAssertEqual(shape, .tensor([4, 8]))
    }

    func testOverlapAddFamilyInfersScalar() throws {
        let g = Graph()
        let overlapAdd = try inferShape(op: .overlapAdd(512, 128, 1, 2, 3), inputs: [.tensor([512])], graph: g)
        let gradStore = try inferShape(op: .overlapAddGradStore(gradStoreCell: 4), inputs: [.scalar], graph: g)
        let gradGather = try inferShape(
            op: .overlapAddGradGather(windowSize: 512, hopSize: 128, gradStoreCell: 4, gradInputCell: 5),
            inputs: [.scalar],
            graph: g
        )
        XCTAssertEqual(overlapAdd, .scalar)
        XCTAssertEqual(gradStore, .scalar)
        XCTAssertEqual(gradGather, .scalar)
    }

    func testBufferViewGradFamilyInfersScalar() throws {
        let g = Graph()
        let gradStore = try inferShape(op: .bufferViewGradStore(gradCell: 7, windowSize: 256), inputs: [.tensor([256])], graph: g)
        let gradRead = try inferShape(op: .bufferViewGradRead(gradCell: 7, windowSize: 256), inputs: [.scalar], graph: g)
        XCTAssertEqual(gradStore, .scalar)
        XCTAssertEqual(gradRead, .scalar)
    }

    func testElementwiseBroadcastingAcrossRanksInfersBroadcastShape() throws {
        let g = Graph()
        let shape = try inferShape(op: .mul, inputs: [.tensor([3, 1, 5]), .tensor([1, 4, 5])], graph: g)
        XCTAssertEqual(shape, .tensor([3, 4, 5]))
    }

    func testDeterministicPhasorWithTensorInputInheritsTensorShape() throws {
        let g = Graph()
        let shape = try inferShape(op: .deterministicPhasor, inputs: [.tensor([16])], graph: g)
        XCTAssertEqual(shape, .tensor([16]))
    }
}
