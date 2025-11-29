import XCTest

@testable import DGen

final class TypeCheckerTests: XCTestCase {

    // MARK: - Shape Inference Tests

    func testScalarConstantInfersScalar() throws {
        let g = Graph()
        let c = g.n(.constant(1.0))

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

        let shape = try inferShape(op: .add, inputs: [.tensor([2, 2]), .tensor([2, 2])], graph: g)
        XCTAssertEqual(shape, .tensor([2, 2]))
    }

    func testAddTensorAndTensorMismatchedShapeThrows() throws {
        let g = Graph()

        XCTAssertThrowsError(
            try inferShape(op: .add, inputs: [.tensor([2, 3]), .tensor([3, 2])], graph: g)
        ) { error in
            guard case DGenError.shapeMismatch(_, _, _) = error else {
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
            guard case DGenError.shapeInferenceFailed(let op, _) = error else {
                XCTFail("Expected shapeInferenceFailed error, got \(error)")
                return
            }
            XCTAssertEqual(op, "conv2d")
        }
    }

    func testTensorHistoryReadInfersShape() throws {
        let g = Graph()
        let buffer = g.tensorHistoryBuffer(shape: [4, 4])

        let shape = try inferShape(op: .tensorHistoryRead(buffer.cellId), inputs: [], graph: g)
        XCTAssertEqual(shape, .tensor([4, 4]))
    }

    func testTensorHistoryReadMissingCellThrows() throws {
        let g = Graph()

        XCTAssertThrowsError(try inferShape(op: .tensorHistoryRead(999), inputs: [], graph: g)) {
            error in
            guard case DGenError.missingCellID(let cellId) = error else {
                XCTFail("Expected missingCellID error, got \(error)")
                return
            }
            XCTAssertEqual(cellId, 999)
        }
    }

    func testTensorHistoryWritePassthroughShape() throws {
        let g = Graph()

        let shape = try inferShape(
            op: .tensorHistoryWrite(0), inputs: [.tensor([3, 3])], graph: g)
        XCTAssertEqual(shape, .tensor([3, 3]))
    }

    func testTensorHistoryWriteMissingInputThrows() throws {
        let g = Graph()

        XCTAssertThrowsError(try inferShape(op: .tensorHistoryWrite(0), inputs: [], graph: g)) {
            error in
            guard case DGenError.shapeInferenceFailed(let op, _) = error else {
                XCTFail("Expected shapeInferenceFailed error, got \(error)")
                return
            }
            XCTAssertEqual(op, "tensorHistoryWrite")
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

    // MARK: - Temporality Inference Tests

    func testPhasorIsFrameBased() {
        XCTAssertTrue(isIntrinsicallyFrameBased(.phasor(0)))
    }

    func testConstantIsNotFrameBased() {
        XCTAssertFalse(isIntrinsicallyFrameBased(.constant(1.0)))
    }

    func testTensorHistoryReadIsFrameBased() {
        XCTAssertTrue(isIntrinsicallyFrameBased(.tensorHistoryRead(0)))
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
        let frameBased = inferTemporality(graph: g, sortedNodes: sortedNodes)

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
        let shape = try inferShape(op: .div, inputs: [.tensor([2, 2]), .tensor([2, 2])], graph: g)
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
}
