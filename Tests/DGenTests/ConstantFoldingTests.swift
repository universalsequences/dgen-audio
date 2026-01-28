import XCTest

@testable import DGen

final class ConstantFoldingTests: XCTestCase {

    // MARK: - Basic Arithmetic Folding

    func testFoldAddition() throws {
        let g = Graph()
        let a = g.n(.constant(2.0))
        let b = g.n(.constant(3.0))
        let sum = g.n(.add, a, b)
        _ = g.n(.output(0), sum)

        foldConstants(g, options: .init(debug: true))

        // The sum node should now be a constant with value 5.0
        if case .constant(let value) = g.nodes[sum]?.op {
            XCTAssertEqual(value, 5.0, accuracy: 0.0001)
        } else {
            XCTFail("Expected sum node to be folded to constant")
        }
    }

    func testFoldMultiplication() throws {
        let g = Graph()
        let a = g.n(.constant(4.0))
        let b = g.n(.constant(2.5))
        let product = g.n(.mul, a, b)
        _ = g.n(.output(0), product)

        foldConstants(g, options: .init(debug: true))

        if case .constant(let value) = g.nodes[product]?.op {
            XCTAssertEqual(value, 10.0, accuracy: 0.0001)
        } else {
            XCTFail("Expected product node to be folded to constant")
        }
    }

    func testFoldDivision() throws {
        let g = Graph()
        let a = g.n(.constant(10.0))
        let b = g.n(.constant(4.0))
        let quotient = g.n(.div, a, b)
        _ = g.n(.output(0), quotient)

        foldConstants(g, options: .init(debug: true))

        if case .constant(let value) = g.nodes[quotient]?.op {
            XCTAssertEqual(value, 2.5, accuracy: 0.0001)
        } else {
            XCTFail("Expected quotient node to be folded to constant")
        }
    }

    func testFoldDivisionByZeroSkipped() throws {
        let g = Graph()
        let a = g.n(.constant(10.0))
        let b = g.n(.constant(0.0))
        let quotient = g.n(.div, a, b)
        _ = g.n(.output(0), quotient)

        foldConstants(g, options: .init(debug: true))

        // Division by zero should NOT be folded
        if case .div = g.nodes[quotient]?.op {
            // Good - still a div operation
        } else {
            XCTFail("Division by zero should not be folded")
        }
    }

    // MARK: - Comparison Folding

    func testFoldGreaterThan() throws {
        let g = Graph()
        let a = g.n(.constant(5.0))
        let b = g.n(.constant(3.0))
        let cmp = g.n(.gt, a, b)
        _ = g.n(.output(0), cmp)

        foldConstants(g, options: .init(debug: true))

        if case .constant(let value) = g.nodes[cmp]?.op {
            XCTAssertEqual(value, 1.0, accuracy: 0.0001)  // true
        } else {
            XCTFail("Expected comparison to be folded to constant")
        }
    }

    func testFoldEqual() throws {
        let g = Graph()
        let a = g.n(.constant(3.0))
        let b = g.n(.constant(3.0))
        let cmp = g.n(.eq, a, b)
        _ = g.n(.output(0), cmp)

        foldConstants(g, options: .init(debug: true))

        if case .constant(let value) = g.nodes[cmp]?.op {
            XCTAssertEqual(value, 1.0, accuracy: 0.0001)  // true
        } else {
            XCTFail("Expected comparison to be folded to constant")
        }
    }

    // MARK: - Gswitch Folding (Key for Biquad)

    func testFoldGswitchTrue() throws {
        let g = Graph()
        let cond = g.n(.constant(1.0))  // true (> 0)
        let ifTrue = g.n(.constant(100.0))
        let ifFalse = g.n(.constant(200.0))
        let result = g.n(.gswitch, cond, ifTrue, ifFalse)
        _ = g.n(.output(0), result)

        foldConstants(g, options: .init(debug: true))

        if case .constant(let value) = g.nodes[result]?.op {
            XCTAssertEqual(value, 100.0, accuracy: 0.0001)  // should select ifTrue
        } else {
            XCTFail("Expected gswitch to be folded to constant")
        }
    }

    func testFoldGswitchFalse() throws {
        let g = Graph()
        let cond = g.n(.constant(-1.0))  // false (<= 0)
        let ifTrue = g.n(.constant(100.0))
        let ifFalse = g.n(.constant(200.0))
        let result = g.n(.gswitch, cond, ifTrue, ifFalse)
        _ = g.n(.output(0), result)

        foldConstants(g, options: .init(debug: true))

        if case .constant(let value) = g.nodes[result]?.op {
            XCTAssertEqual(value, 200.0, accuracy: 0.0001)  // should select ifFalse
        } else {
            XCTFail("Expected gswitch to be folded to constant")
        }
    }

    // MARK: - Selector Folding (Key for Biquad Mode Selection)

    func testFoldSelectorMode1() throws {
        let g = Graph()
        let mode = g.n(.constant(1.0))  // Select first option
        let opt1 = g.n(.constant(10.0))
        let opt2 = g.n(.constant(20.0))
        let opt3 = g.n(.constant(30.0))
        let result = g.n(.selector, mode, opt1, opt2, opt3)
        _ = g.n(.output(0), result)

        foldConstants(g, options: .init(debug: true))

        if case .constant(let value) = g.nodes[result]?.op {
            XCTAssertEqual(value, 10.0, accuracy: 0.0001)
        } else {
            XCTFail("Expected selector to be folded to constant")
        }
    }

    func testFoldSelectorMode3() throws {
        let g = Graph()
        let mode = g.n(.constant(3.0))  // Select third option
        let opt1 = g.n(.constant(10.0))
        let opt2 = g.n(.constant(20.0))
        let opt3 = g.n(.constant(30.0))
        let result = g.n(.selector, mode, opt1, opt2, opt3)
        _ = g.n(.output(0), result)

        foldConstants(g, options: .init(debug: true))

        if case .constant(let value) = g.nodes[result]?.op {
            XCTAssertEqual(value, 30.0, accuracy: 0.0001)
        } else {
            XCTFail("Expected selector to be folded to constant")
        }
    }

    // MARK: - Transitive Folding

    func testTransitiveFolding() throws {
        // Test that folding propagates: (2 + 3) * 4 should fold to 20
        let g = Graph()
        let a = g.n(.constant(2.0))
        let b = g.n(.constant(3.0))
        let sum = g.n(.add, a, b)  // 5.0
        let c = g.n(.constant(4.0))
        let product = g.n(.mul, sum, c)  // 20.0
        _ = g.n(.output(0), product)

        foldConstants(g, options: .init(debug: true))

        // Both sum and product should be folded
        if case .constant(let sumValue) = g.nodes[sum]?.op {
            XCTAssertEqual(sumValue, 5.0, accuracy: 0.0001)
        } else {
            XCTFail("Expected sum to be folded")
        }

        if case .constant(let productValue) = g.nodes[product]?.op {
            XCTAssertEqual(productValue, 20.0, accuracy: 0.0001)
        } else {
            XCTFail("Expected product to be folded")
        }
    }

    func testComparisonThenGswitch() throws {
        // This mimics the biquad pattern: mode == 1 ? lowpass_coeff : other_coeff
        let g = Graph()
        let mode = g.n(.constant(1.0))
        let targetMode = g.n(.constant(1.0))
        let cmp = g.n(.eq, mode, targetMode)  // 1.0 (true)
        let lowpassCoeff = g.n(.constant(0.5))
        let otherCoeff = g.n(.constant(0.8))
        let result = g.n(.gswitch, cmp, lowpassCoeff, otherCoeff)
        _ = g.n(.output(0), result)

        foldConstants(g, options: .init(debug: true))

        // Comparison should fold to 1.0, then gswitch should fold to lowpassCoeff
        if case .constant(let value) = g.nodes[result]?.op {
            XCTAssertEqual(value, 0.5, accuracy: 0.0001)
        } else {
            XCTFail("Expected gswitch to be folded after comparison folded")
        }
    }

    // MARK: - Non-Constant Inputs Should Not Fold

    func testNonConstantInputsNotFolded() throws {
        let g = Graph()
        let cellId = g.alloc()
        let phasor = g.n(.phasor(cellId), g.n(.constant(440.0)), g.n(.constant(0.0)))
        let constant = g.n(.constant(2.0))
        let product = g.n(.mul, phasor, constant)
        _ = g.n(.output(0), product)

        foldConstants(g, options: .init(debug: true))

        // Product should NOT be folded because phasor is not a constant
        if case .mul = g.nodes[product]?.op {
            // Good - still a mul operation
        } else {
            XCTFail("Non-constant operation should not be folded")
        }
    }

    // MARK: - Unary Math Functions

    func testFoldSin() throws {
        let g = Graph()
        let x = g.n(.constant(0.0))
        let result = g.n(.sin, x)
        _ = g.n(.output(0), result)

        foldConstants(g, options: .init(debug: true))

        if case .constant(let value) = g.nodes[result]?.op {
            XCTAssertEqual(value, 0.0, accuracy: 0.0001)
        } else {
            XCTFail("Expected sin to be folded")
        }
    }

    func testFoldCos() throws {
        let g = Graph()
        let x = g.n(.constant(0.0))
        let result = g.n(.cos, x)
        _ = g.n(.output(0), result)

        foldConstants(g, options: .init(debug: true))

        if case .constant(let value) = g.nodes[result]?.op {
            XCTAssertEqual(value, 1.0, accuracy: 0.0001)
        } else {
            XCTFail("Expected cos to be folded")
        }
    }

    func testFoldSqrt() throws {
        let g = Graph()
        let x = g.n(.constant(16.0))
        let result = g.n(.sqrt, x)
        _ = g.n(.output(0), result)

        foldConstants(g, options: .init(debug: true))

        if case .constant(let value) = g.nodes[result]?.op {
            XCTAssertEqual(value, 4.0, accuracy: 0.0001)
        } else {
            XCTFail("Expected sqrt to be folded")
        }
    }

    func testFoldSqrtNegativeSkipped() throws {
        let g = Graph()
        let x = g.n(.constant(-1.0))
        let result = g.n(.sqrt, x)
        _ = g.n(.output(0), result)

        foldConstants(g, options: .init(debug: true))

        // sqrt of negative should NOT be folded
        if case .sqrt = g.nodes[result]?.op {
            // Good - still a sqrt operation
        } else {
            XCTFail("sqrt of negative should not be folded")
        }
    }

    // MARK: - Integration Test: Folding Count

    func testFoldingCountReported() throws {
        // Create a graph with multiple foldable operations
        let g = Graph()
        let a = g.n(.constant(1.0))
        let b = g.n(.constant(2.0))
        let c = g.n(.constant(3.0))
        let d = g.n(.constant(4.0))
        let ab = g.n(.add, a, b)      // fold 1
        let cd = g.n(.mul, c, d)      // fold 2
        let result = g.n(.sub, ab, cd) // fold 3
        _ = g.n(.output(0), result)

        // Count nodes before
        var constantCountBefore = 0
        for (_, node) in g.nodes {
            if case .constant = node.op {
                constantCountBefore += 1
            }
        }

        foldConstants(g, options: .init(debug: true))

        // Count nodes after
        var constantCountAfter = 0
        for (_, node) in g.nodes {
            if case .constant = node.op {
                constantCountAfter += 1
            }
        }

        // Should have 3 more constants (ab, cd, result all became constants)
        XCTAssertEqual(constantCountAfter - constantCountBefore, 3)

        // Final result should be (1+2) - (3*4) = 3 - 12 = -9
        if case .constant(let value) = g.nodes[result]?.op {
            XCTAssertEqual(value, -9.0, accuracy: 0.0001)
        } else {
            XCTFail("Expected final result to be folded")
        }
    }
}
