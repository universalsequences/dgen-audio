import XCTest
@testable import DGenLazy
@testable import DGen

/// Tests for the C (CPU) backend via DGenLazy realize()
/// Note: C backend requires frameCount divisible by 4 (SIMD constraint)
final class CBackendTests: XCTestCase {

    override func setUp() {
        super.setUp()
        DGenConfig.backend = .c
        DGenConfig.sampleRate = 44100.0
        LazyGraphContext.reset()
    }

    override func tearDown() {
        DGenConfig.backend = .metal
        DGenConfig.sampleRate = 44100.0
        super.tearDown()
    }

    // MARK: - Signal Tests

    func testSignalConstant() throws {
        let s = Signal.constant(42.0)
        let result = try s.realize(frames: 8)

        XCTAssertEqual(result.count, 8)
        for val in result {
            XCTAssertEqual(val, 42.0, accuracy: 1e-5)
        }
    }

    func testSignalArithmetic() throws {
        let a = Signal.constant(10.0)
        let b = Signal.constant(3.0)
        let result = (a + b) * 2.0
        let values = try result.realize(frames: 4)

        // (10 + 3) * 2 = 26
        for val in values {
            XCTAssertEqual(val, 26.0, accuracy: 1e-5)
        }
    }

    func testSignalPhasor() throws {
        DGenConfig.sampleRate = 1000.0
        LazyGraphContext.reset()

        // Phasor at 100 Hz with 1000 Hz sample rate: increment = 0.1 per frame
        let osc = Signal.phasor(100.0)
        let result = try osc.realize(frames: 12)  // divisible by 4

        XCTAssertEqual(result.count, 12)
        XCTAssertEqual(result[0], 0.0, accuracy: 1e-4)
        XCTAssertEqual(result[1], 0.1, accuracy: 1e-4)
        XCTAssertEqual(result[5], 0.5, accuracy: 1e-4)
        XCTAssertEqual(result[9], 0.9, accuracy: 1e-4)
    }

    func testSignalSineWave() throws {
        DGenConfig.sampleRate = 4.0
        LazyGraphContext.reset()

        // 1 Hz sine wave at 4 Hz sample rate
        // Phases: 0, 0.25, 0.5, 0.75
        // sin(2pi * phase): 0, 1, 0, -1
        let osc = Signal.phasor(1.0)
        let wave = sin(osc * 2.0 * Float.pi)
        let result = try wave.realize(frames: 4)

        XCTAssertEqual(result[0], 0.0, accuracy: 1e-4)   // sin(0)
        XCTAssertEqual(result[1], 1.0, accuracy: 1e-4)   // sin(pi/2)
        XCTAssertEqual(result[2], 0.0, accuracy: 1e-4)   // sin(pi)
        XCTAssertEqual(result[3], -1.0, accuracy: 1e-4)  // sin(3pi/2)
    }

    // MARK: - Tensor Tests

    func testTensorRealize() throws {
        let t = Tensor([1, 2, 3, 4]) * 2.0 + 1.0
        let result = try t.realize()

        XCTAssertEqual(result, [3, 5, 7, 9])
    }

    func testTensorAddTensor() throws {
        let a = Tensor([1, 2, 3, 4])
        let b = Tensor([10, 20, 30, 40])
        let result = try (a + b).realize()

        XCTAssertEqual(result, [11, 22, 33, 44])
    }
}
