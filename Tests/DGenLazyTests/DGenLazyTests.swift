import XCTest
@testable import DGenLazy
@testable import DGen

/// Tests for the DGenLazy module
final class DGenLazyTests: XCTestCase {

    override func setUp() {
        super.setUp()
        // Reset to fresh graph before each test
        DGenConfig.sampleRate = 44100.0  // Default sample rate
        LazyGraphContext.reset()
    }

    override func tearDown() {
        // Reset sample rate to default to avoid affecting other tests
        DGenConfig.sampleRate = 44100.0
        super.tearDown()
    }

    // MARK: - Tensor Creation Tests

    func testTensorFrom1DArray() {
        let t = Tensor([1, 2, 3, 4])

        XCTAssertEqual(t.shape, [4])
        XCTAssertEqual(t.size, 4)
        XCTAssertEqual(t.ndim, 1)
        XCTAssertFalse(t.requiresGrad)
    }

    func testTensorFrom2DArray() {
        let t = Tensor([[1, 2, 3], [4, 5, 6]])

        XCTAssertEqual(t.shape, [2, 3])
        XCTAssertEqual(t.size, 6)
        XCTAssertEqual(t.ndim, 2)
    }

    func testTensorZeros() {
        let t = Tensor.zeros([3, 4])

        XCTAssertEqual(t.shape, [3, 4])
        XCTAssertEqual(t.size, 12)
    }

    func testTensorOnes() {
        let t = Tensor.ones([2, 2])

        XCTAssertEqual(t.shape, [2, 2])
    }

    func testTensorRandn() {
        let t = Tensor.randn([64, 32])

        XCTAssertEqual(t.shape, [64, 32])
        XCTAssertEqual(t.size, 64 * 32)
    }

    func testTensorParam() {
        let t = Tensor.param([10, 5])

        XCTAssertEqual(t.shape, [10, 5])
        XCTAssertTrue(t.requiresGrad)
    }

    func testTensorRequiresGrad() {
        let t1 = Tensor([1, 2, 3])
        let t2 = Tensor([1, 2, 3], requiresGrad: true)

        XCTAssertFalse(t1.requiresGrad)
        XCTAssertTrue(t2.requiresGrad)
    }

    // MARK: - Signal Creation Tests

    func testSignalConstant() {
        let s = Signal.constant(440.0)

        XCTAssertFalse(s.requiresGrad)
    }

    func testSignalParam() {
        let s = Signal.param(440.0)

        XCTAssertTrue(s.requiresGrad)
    }

    func testSignalWithRequiresGrad() {
        let s = Signal(440.0, requiresGrad: true)

        XCTAssertTrue(s.requiresGrad)
    }

    func testSignalPhasor() {
        let freq = Signal.constant(440.0)
        let osc = Signal.phasor(freq)

        XCTAssertFalse(osc.requiresGrad)
        XCTAssertNotNil(osc.cellId) // Phasor has state
    }

    func testSignalPhasorWithFloat() {
        let osc = Signal.phasor(440.0)

        XCTAssertNotNil(osc.cellId)
    }

    func testSignalNoise() {
        let n = Signal.noise()

        XCTAssertFalse(n.requiresGrad)
    }

    // MARK: - SignalTensor Creation Tests

    func testSignalTensorFromTensorPhasor() {
        let freqs = Tensor([440, 880, 1320])
        let phases = Signal.phasor(freqs)

        XCTAssertEqual(phases.shape, [3])
    }

    // MARK: - Tensor Operator Tests

    func testTensorAddTensor() {
        let a = Tensor([1, 2, 3])
        let b = Tensor([4, 5, 6])
        let c = a + b

        XCTAssertEqual(c.shape, [3])
        XCTAssertFalse(c.requiresGrad)
    }

    func testTensorAddTensorRequiresGrad() {
        let a = Tensor([1, 2, 3], requiresGrad: true)
        let b = Tensor([4, 5, 6])
        let c = a + b

        XCTAssertTrue(c.requiresGrad) // Propagates from a
    }

    func testTensorSubTensor() {
        let a = Tensor([1, 2, 3])
        let b = Tensor([4, 5, 6])
        let c = a - b

        XCTAssertEqual(c.shape, [3])
    }

    func testTensorMulTensor() {
        let a = Tensor([1, 2, 3])
        let b = Tensor([4, 5, 6])
        let c = a * b

        XCTAssertEqual(c.shape, [3])
    }

    func testTensorDivTensor() {
        let a = Tensor([1, 2, 3])
        let b = Tensor([4, 5, 6])
        let c = a / b

        XCTAssertEqual(c.shape, [3])
    }

    func testTensorAddFloat() {
        let a = Tensor([1, 2, 3])
        let b = a + 1.0

        XCTAssertEqual(b.shape, [3])
    }

    func testFloatAddTensor() {
        let a = Tensor([1, 2, 3])
        let b = 1.0 + a

        XCTAssertEqual(b.shape, [3])
    }

    func testTensorMulFloat() {
        let a = Tensor([1, 2, 3])
        let b = a * 2.0

        XCTAssertEqual(b.shape, [3])
    }

    func testTensorNegation() {
        let a = Tensor([1, 2, 3])
        let b = -a

        XCTAssertEqual(b.shape, [3])
    }

    // MARK: - Signal Operator Tests

    func testSignalAddSignal() {
        let a = Signal.constant(1.0)
        let b = Signal.constant(2.0)
        let c = a + b

        XCTAssertFalse(c.requiresGrad)
    }

    func testSignalMulFloat() {
        let a = Signal.constant(440.0)
        let b = a * 2.0

        XCTAssertFalse(b.requiresGrad)
    }

    func testSignalAddFloat() {
        let a = Signal.constant(440.0)
        let b = a + 100.0

        XCTAssertFalse(b.requiresGrad)
    }

    // MARK: - Type Promotion Tests

    func testTensorAddSignal() {
        let t = Tensor([1, 2, 3])
        let s = Signal.phasor(440)
        let st = t + s

        XCTAssertEqual(st.shape, [3])
        // Result should be SignalTensor
        XCTAssert(type(of: st) == SignalTensor.self)
    }

    func testTensorMulSignal() {
        let t = Tensor([0, 1, 1, 0])
        let s = Signal.phasor(440)
        let st = t * s

        XCTAssertEqual(st.shape, [4])
    }

    func testSignalAddTensor() {
        let s = Signal.phasor(440)
        let t = Tensor([1, 2, 3])
        let st = s + t

        XCTAssertEqual(st.shape, [3])
    }

    // MARK: - Math Function Tests

    func testTensorSin() {
        let t = Tensor([0, 1, 2, 3])
        let s = sin(t)

        XCTAssertEqual(s.shape, [4])
    }

    func testTensorExp() {
        let t = Tensor([0, 1, 2])
        let e = exp(t)

        XCTAssertEqual(e.shape, [3])
    }

    func testTensorRelu() {
        let t = Tensor([-1, 0, 1, 2])
        let r = relu(t)

        XCTAssertEqual(r.shape, [4])
    }

    func testTensorSigmoid() {
        let t = Tensor([-2, -1, 0, 1, 2])
        let s = sigmoid(t)

        XCTAssertEqual(s.shape, [5])
    }

    func testSignalSin() {
        let s = Signal.phasor(440)
        let wave = sin(s * 2.0 * Float.pi)

        XCTAssertFalse(wave.requiresGrad)
    }

    func testSignalTensorSin() {
        let freqs = Tensor([440, 880])
        let phases = Signal.phasor(freqs)
        let waves = sin(phases * 2.0 * Float.pi)

        XCTAssertEqual(waves.shape, [2])
    }

    // MARK: - Method Alias Tests

    func testTensorMethodAliases() {
        let t = Tensor([1, 2, 3])

        _ = t.abs()
        _ = t.exp()
        _ = t.log()
        _ = t.sqrt()
        _ = t.sin()
        _ = t.cos()
        _ = t.tanh()
        _ = t.relu()
        _ = t.sigmoid()
        _ = t.pow(2.0)
    }

    func testSignalMethodAliases() {
        let s = Signal.constant(1.0)

        _ = s.abs()
        _ = s.exp()
        _ = s.sin()
        _ = s.tanh()
        _ = s.relu()
    }

    // MARK: - Reduction Tests

    func testTensorSum() {
        let t = Tensor([1, 2, 3, 4])
        let s = t.sum()

        XCTAssertEqual(s.shape, [1])
    }

    func testTensorMean() {
        let t = Tensor([1, 2, 3, 4])
        let m = t.mean()

        XCTAssertEqual(m.shape, [1])
    }

    func testSignalTensorSum() {
        let freqs = Tensor([440, 880, 1320])
        let phases = Signal.phasor(freqs)
        let summed = phases.sum()

        // Result is a Signal (scalar per frame)
        XCTAssert(type(of: summed) == Signal.self)
    }

    // MARK: - Loss Function Tests

    func testMSETensor() {
        let pred = Tensor([1, 2, 3])
        let target = Tensor([1.1, 2.1, 3.1])
        let loss = mse(pred, target)

        XCTAssertEqual(loss.shape, [1])
    }

    func testMSESignal() {
        let pred = Signal.phasor(440)
        let target = Signal.phasor(450)
        let loss = mse(pred, target)

        XCTAssertFalse(loss.requiresGrad)
    }

    // MARK: - Chained Operations Tests

    func testChainedTensorOps() {
        let t = Tensor([1, 2, 3, 4])
        let result = ((t + 1) * 2).relu().sum()

        XCTAssertEqual(result.shape, [1])
    }

    func testChainedSignalOps() {
        let freq = Signal.param(440.0)
        let osc = Signal.phasor(freq)
        let wave = sin(osc * 2.0 * Float.pi)
        let scaled = wave * 0.5

        XCTAssertTrue(scaled.requiresGrad) // Propagates from freq
    }

    // MARK: - Graph Context Tests

    func testGraphContextReset() {
        let t1 = Tensor([1, 2, 3])
        let graph1 = t1.graph

        LazyGraphContext.reset()

        let t2 = Tensor([4, 5, 6])
        let graph2 = t2.graph

        XCTAssert(graph1 !== graph2) // Different graph instances
    }

    func testTensorsShareGraph() {
        let t1 = Tensor([1, 2, 3])
        let t2 = Tensor([4, 5, 6])
        let t3 = t1 + t2

        XCTAssert(t1.graph === t2.graph)
        XCTAssert(t2.graph === t3.graph)
    }

    // MARK: - Broadcasting Tests

    func testBroadcastShapeSame() {
        let result = broadcastShape([3, 4], [3, 4])
        XCTAssertEqual(result, [3, 4])
    }

    func testBroadcastShapeScalar() {
        let result = broadcastShape([3, 4], [1])
        XCTAssertEqual(result, [3, 4])
    }

    func testBroadcastShapeDifferentDims() {
        let result = broadcastShape([3, 4], [4])
        XCTAssertEqual(result, [3, 4])
    }

    func testBroadcastShapeBothExpand() {
        let result = broadcastShape([1, 4], [3, 1])
        XCTAssertEqual(result, [3, 4])
    }

    // MARK: - Derived Tensor Realize Tests (Element-wise verification)

    func testTensorAddTensorRealize() throws {
        LazyGraphContext.reset()

        let a = Tensor([1, 2, 3])
        let b = Tensor([4, 5, 6])
        let c = a + b
        let result = try c.realize()

        XCTAssertEqual(result, [5, 7, 9])
    }

    func testTensorSubTensorRealize() throws {
        LazyGraphContext.reset()

        let a = Tensor([10, 20, 30])
        let b = Tensor([1, 2, 3])
        let c = a - b
        let result = try c.realize()

        XCTAssertEqual(result, [9, 18, 27])
    }

    func testTensorMulTensorRealize() throws {
        LazyGraphContext.reset()

        let a = Tensor([1, 2, 3])
        let b = Tensor([4, 5, 6])
        let c = a * b
        let result = try c.realize()

        XCTAssertEqual(result, [4, 10, 18])
    }

    func testTensorDivTensorRealize() throws {
        LazyGraphContext.reset()

        let a = Tensor([10, 20, 30])
        let b = Tensor([2, 4, 5])
        let c = a / b
        let result = try c.realize()

        XCTAssertEqual(result[0], 5.0, accuracy: 1e-5)
        XCTAssertEqual(result[1], 5.0, accuracy: 1e-5)
        XCTAssertEqual(result[2], 6.0, accuracy: 1e-5)
    }

    func testTensorAddFloatRealize() throws {
        LazyGraphContext.reset()

        let a = Tensor([1, 2, 3])
        let b = a + 10.0
        let result = try b.realize()

        XCTAssertEqual(result, [11, 12, 13])
    }

    func testTensorMulFloatRealize() throws {
        LazyGraphContext.reset()

        let a = Tensor([1, 2, 3])
        let b = a * 3.0
        let result = try b.realize()

        XCTAssertEqual(result, [3, 6, 9])
    }

    func testTensorNegationRealize() throws {
        LazyGraphContext.reset()

        let a = Tensor([1, -2, 3])
        let b = -a
        let result = try b.realize()

        XCTAssertEqual(result, [-1, 2, -3])
    }

    func testTensorChainedOpsRealize() throws {
        LazyGraphContext.reset()

        // (a + b) * 2 - 1
        let a = Tensor([1, 2, 3])
        let b = Tensor([4, 5, 6])
        let c = (a + b) * 2.0 - 1.0
        let result = try c.realize()

        // (1+4)*2-1=9, (2+5)*2-1=13, (3+6)*2-1=17
        XCTAssertEqual(result, [9, 13, 17])
    }

    func testTensorSinRealize() throws {
        LazyGraphContext.reset()

        let a = Tensor([0, Float.pi / 2, Float.pi])
        let b = sin(a)
        let result = try b.realize()

        XCTAssertEqual(result[0], 0.0, accuracy: 1e-5)      // sin(0) = 0
        XCTAssertEqual(result[1], 1.0, accuracy: 1e-5)      // sin(π/2) = 1
        XCTAssertEqual(result[2], 0.0, accuracy: 1e-4)      // sin(π) ≈ 0
    }

    func testTensorExpRealize() throws {
        LazyGraphContext.reset()

        let a = Tensor([0, 1, 2])
        let b = exp(a)
        let result = try b.realize()

        XCTAssertEqual(result[0], 1.0, accuracy: 1e-5)           // e^0 = 1
        XCTAssertEqual(result[1], Float(M_E), accuracy: 1e-4)    // e^1 = e
        XCTAssertEqual(result[2], Float(M_E * M_E), accuracy: 1e-3)  // e^2
    }

    func testTensorReluRealize() throws {
        LazyGraphContext.reset()

        let a = Tensor([-2, -1, 0, 1, 2])
        let b = relu(a)
        let result = try b.realize()

        XCTAssertEqual(result, [0, 0, 0, 1, 2])
    }

    func testTensor2DRealize() throws {
        LazyGraphContext.reset()

        let a = Tensor([[1, 2], [3, 4]])
        let b = Tensor([[10, 20], [30, 40]])
        let c = a + b
        let result = try c.realize()

        // Row-major: [11, 22, 33, 44]
        XCTAssertEqual(result, [11, 22, 33, 44])
    }

    // MARK: - SignalTensor Derived Realize Tests

    func testSignalTensorDerivedRealize() throws {
        DGenConfig.sampleRate = 1000.0
        LazyGraphContext.reset()

        // Source SignalTensor (phasor with tensor of frequencies)
        let freqs = Tensor([100, 200])
        let phases = Signal.phasor(freqs)

        // Derived SignalTensor (multiply by 2)
        let doubled = phases * 2.0
        let result = try doubled.realize(frames: 4)

        // phases at 100Hz (1000Hz SR): 0, 0.1, 0.2, 0.3 per frame
        // phases at 200Hz (1000Hz SR): 0, 0.2, 0.4, 0.6 per frame
        // doubled: phases * 2
        // Layout: [frame0_elem0, frame0_elem1, frame1_elem0, frame1_elem1, ...]
        XCTAssertEqual(result.count, 8, "Expected 8 values (4 frames * 2 elements)")
        XCTAssertEqual(result[0], 0.0, accuracy: 1e-4)  // frame 0, elem 0: 0*2
        XCTAssertEqual(result[1], 0.0, accuracy: 1e-4)  // frame 0, elem 1: 0*2
        XCTAssertEqual(result[2], 0.2, accuracy: 1e-4)  // frame 1, elem 0: 0.1*2
        XCTAssertEqual(result[3], 0.4, accuracy: 1e-4)  // frame 1, elem 1: 0.2*2
        XCTAssertEqual(result[6], 0.6, accuracy: 1e-4)  // frame 3, elem 0: 0.3*2
        XCTAssertEqual(result[7], 1.2, accuracy: 1e-4)  // frame 3, elem 1: 0.6*2
    }

    func testSignalTensorAddRealize() throws {
        DGenConfig.sampleRate = 1000.0
        LazyGraphContext.reset()

        let freqs = Tensor([100, 200])
        let phases = Signal.phasor(freqs)
        let offset = phases + 0.5

        let result = try offset.realize(frames: 2)

        // frame 0: phases=[0, 0], offset=[0.5, 0.5]
        // frame 1: phases=[0.1, 0.2], offset=[0.6, 0.7]
        XCTAssertEqual(result.count, 4)
        XCTAssertEqual(result[0], 0.5, accuracy: 1e-4)  // frame 0, elem 0
        XCTAssertEqual(result[1], 0.5, accuracy: 1e-4)  // frame 0, elem 1
        XCTAssertEqual(result[2], 0.6, accuracy: 1e-4)  // frame 1, elem 0
        XCTAssertEqual(result[3], 0.7, accuracy: 1e-4)  // frame 1, elem 1
    }

    // MARK: - Realize Tests (Forward Pass)

    func testTensorRealizeSimple() throws {
        LazyGraphContext.reset()

        let t = Tensor([1, 2, 3, 4])
        let result = try t.realize()

        XCTAssertEqual(result.count, 4)
        XCTAssertEqual(result[0], 1.0, accuracy: 1e-5)
        XCTAssertEqual(result[1], 2.0, accuracy: 1e-5)
        XCTAssertEqual(result[2], 3.0, accuracy: 1e-5)
        XCTAssertEqual(result[3], 4.0, accuracy: 1e-5)
    }

    func testTensorRealizeSum() throws {
        LazyGraphContext.reset()

        let t = Tensor([1, 2, 3, 4])
        let s = t.sum()
        let result = try s.realize()

        // Sum is scalar, should be single value
        XCTAssertEqual(result.count, 1)
        XCTAssertEqual(result[0], 10.0, accuracy: 1e-5)
    }

    func testTensorRealizeArithmetic() throws {
        LazyGraphContext.reset()

        let t = Tensor([1, 2, 3, 4])
        let result = ((t + 1) * 2).sum()
        let values = try result.realize()

        // (1+1)*2 + (2+1)*2 + (3+1)*2 + (4+1)*2 = 4 + 6 + 8 + 10 = 28
        XCTAssertEqual(values[0], 28.0, accuracy: 1e-5)
    }

    func testTensorRealize2D() throws {
        LazyGraphContext.reset()

        let t = Tensor([[1, 2], [3, 4]])
        let result = try t.realize()

        XCTAssertEqual(result.count, 4)
        XCTAssertEqual(result[0], 1.0, accuracy: 1e-5)
        XCTAssertEqual(result[1], 2.0, accuracy: 1e-5)
        XCTAssertEqual(result[2], 3.0, accuracy: 1e-5)
        XCTAssertEqual(result[3], 4.0, accuracy: 1e-5)
    }

    func testSignalRealizeConstant() throws {
        LazyGraphContext.reset()

        let s = Signal.constant(42.0)
        let result = try s.realize(frames: 4)

        XCTAssertEqual(result.count, 4)
        for val in result {
            XCTAssertEqual(val, 42.0, accuracy: 1e-5)
        }
    }

    func testSignalRealizePhasor() throws {
        // Must set sample rate BEFORE resetting graph
        DGenConfig.sampleRate = 1000.0  // 1000 Hz sample rate
        LazyGraphContext.reset()

        // Phasor at 100 Hz with 1000 Hz sample rate should increment by 0.1 per frame
        let osc = Signal.phasor(100.0)
        let result = try osc.realize(frames: 10)

        XCTAssertEqual(result.count, 10)
        // Phasor should ramp from 0 to ~0.9
        XCTAssertEqual(result[0], 0.0, accuracy: 1e-4)
        XCTAssertEqual(result[1], 0.1, accuracy: 1e-4)
        XCTAssertEqual(result[5], 0.5, accuracy: 1e-4)
        XCTAssertEqual(result[9], 0.9, accuracy: 1e-4)
    }

    func testSignalRealizeArithmetic() throws {
        LazyGraphContext.reset()

        let a = Signal.constant(10.0)
        let b = Signal.constant(3.0)
        let result = (a + b) * 2.0
        let values = try result.realize(frames: 4)

        // (10 + 3) * 2 = 26
        for val in values {
            XCTAssertEqual(val, 26.0, accuracy: 1e-5)
        }
    }

    func testSignalRealizeSin() throws {
        LazyGraphContext.reset()
        DGenConfig.sampleRate = 1000.0

        // sin(0) = 0, sin(pi/2) = 1
        let phase = Signal.constant(0.0)
        let wave = sin(phase)
        let result = try wave.realize(frames: 4)

        // sin(0) = 0 for all frames
        for val in result {
            XCTAssertEqual(val, 0.0, accuracy: 1e-5)
        }
    }

    func testSignalRealizeSineWave() throws {
        // Must set sample rate BEFORE resetting graph
        DGenConfig.sampleRate = 4.0  // Very low sample rate for easy testing
        LazyGraphContext.reset()

        // 1 Hz sine wave at 4 Hz sample rate
        // Phases: 0, 0.25, 0.5, 0.75
        // sin(2pi * phase): sin(0)=0, sin(pi/2)=1, sin(pi)=0, sin(3pi/2)=-1
        let osc = Signal.phasor(1.0)
        let wave = sin(osc * 2.0 * Float.pi)
        let result = try wave.realize(frames: 4)

        XCTAssertEqual(result[0], 0.0, accuracy: 1e-4)   // sin(0)
        XCTAssertEqual(result[1], 1.0, accuracy: 1e-4)   // sin(pi/2)
        XCTAssertEqual(result[2], 0.0, accuracy: 1e-4)   // sin(pi)
        XCTAssertEqual(result[3], -1.0, accuracy: 1e-4)  // sin(3pi/2)
    }

    func testMSERealize() throws {
        LazyGraphContext.reset()

        let pred = Signal.constant(3.0)
        let target = Signal.constant(5.0)
        let loss = mse(pred, target)
        let result = try loss.realize(frames: 4)

        // MSE = (3 - 5)^2 = 4
        for val in result {
            XCTAssertEqual(val, 4.0, accuracy: 1e-5)
        }
    }

    // MARK: - Backward Tests (Gradient Computation)

    func testTensorBackwardSimple() throws {
        LazyGraphContext.reset()
        DGenConfig.kernelOutputPath = "/tmp/tensor_backward_simple.metal"
        DGenConfig.debug = true

        // Simple case: loss = sum(w * 2)
        // d(loss)/d(w) = 2 for each element
        let w = Tensor([1, 2, 3, 4], requiresGrad: true)
        let loss = (w * 2.0).sum()

        // Use frameCount=1 for static tensor ops (no frame variation)
        try loss.backward(frameCount: 1)
        DGenConfig.kernelOutputPath = nil
        DGenConfig.debug = false

        XCTAssertNotNil(w.grad, "Gradient should be populated after backward")
        if let grad = w.grad {
            let gradValues = try grad.realize()
            XCTAssertEqual(gradValues.count, 4)
            // Each gradient should be 2.0
            for val in gradValues {
                XCTAssertEqual(val, 2.0, accuracy: 1e-4)
            }
        }
    }

    func testTensorBackwardAddition() throws {
        LazyGraphContext.reset()

        // loss = sum(w + 1)
        // d(loss)/d(w) = 1 for each element
        let w = Tensor([1, 2, 3], requiresGrad: true)
        let loss = (w + 1.0).sum()

        try loss.backward(frameCount: 1)

        XCTAssertNotNil(w.grad)
        if let grad = w.grad {
            let gradValues = try grad.realize()
            for val in gradValues {
                XCTAssertEqual(val, 1.0, accuracy: 1e-4)
            }
        }
    }

    func testTensorBackwardChained() throws {
        LazyGraphContext.reset()

        // loss = sum((w + 1) * 2)
        // d(loss)/d(w) = 2 for each element
        let w = Tensor([1, 2, 3, 4], requiresGrad: true)
        let loss = ((w + 1.0) * 2.0).sum()

        try loss.backward(frameCount: 1)

        XCTAssertNotNil(w.grad)
        if let grad = w.grad {
            let gradValues = try grad.realize()
            for val in gradValues {
                XCTAssertEqual(val, 2.0, accuracy: 1e-4)
            }
        }
    }

    func testTensorBackwardSquare() throws {
        LazyGraphContext.reset()

        // loss = sum(w * w)
        // d(loss)/d(w) = 2*w
        let w = Tensor([1, 2, 3], requiresGrad: true)
        let loss = (w * w).sum()

        try loss.backward(frameCount: 1)

        XCTAssertNotNil(w.grad)
        if let grad = w.grad {
            let gradValues = try grad.realize()
            XCTAssertEqual(gradValues[0], 2.0, accuracy: 1e-4)  // 2*1
            XCTAssertEqual(gradValues[1], 4.0, accuracy: 1e-4)  // 2*2
            XCTAssertEqual(gradValues[2], 6.0, accuracy: 1e-4)  // 2*3
        }
    }

    func testTensorBackwardMSE() throws {
        LazyGraphContext.reset()

        // pred with grad, target is constant
        // MSE = sum((pred - target)^2) / n
        // d(MSE)/d(pred) = 2*(pred - target) / n
        let pred = Tensor([3, 5, 7], requiresGrad: true)
        let target = Tensor([1, 2, 3])

        // Compute MSE manually: ((pred - target)^2).mean()
        let diff = pred - target
        let squaredDiff = diff * diff
        let loss = squaredDiff.mean()

        try loss.backward(frameCount: 1)

        XCTAssertNotNil(pred.grad)
        if let grad = pred.grad {
            let gradValues = try grad.realize()
            // d(mse)/d(pred) = 2*(pred-target)/n = 2*(3-1)/3, 2*(5-2)/3, 2*(7-3)/3
            // = 4/3, 6/3, 8/3 = 1.333, 2.0, 2.667
            XCTAssertEqual(gradValues[0], 4.0/3.0, accuracy: 1e-2)
            XCTAssertEqual(gradValues[1], 6.0/3.0, accuracy: 1e-2)
            XCTAssertEqual(gradValues[2], 8.0/3.0, accuracy: 1e-2)
        }
    }

    func testMultipleTensorsBackward() throws {
        LazyGraphContext.reset()

        // loss = sum(a * b)
        // d(loss)/d(a) = b, d(loss)/d(b) = a
        let a = Tensor([1, 2, 3], requiresGrad: true)
        let b = Tensor([4, 5, 6], requiresGrad: true)
        let loss = (a * b).sum()

        try loss.backward(frameCount: 1)

        XCTAssertNotNil(a.grad, "Gradient for 'a' should not be nil")
        XCTAssertNotNil(b.grad, "Gradient for 'b' should not be nil")

        // Debug: Print actual gradient values
        if let gradA = a.grad {
            let gradValues = try gradA.realize()
            print("DEBUG: gradA = \(gradValues)")
            // d/da = b = [4, 5, 6]
            XCTAssertEqual(gradValues[0], 4.0, accuracy: 1e-4)
            XCTAssertEqual(gradValues[1], 5.0, accuracy: 1e-4)
            XCTAssertEqual(gradValues[2], 6.0, accuracy: 1e-4)
        }

        if let gradB = b.grad {
            let gradValues = try gradB.realize()
            print("DEBUG: gradB = \(gradValues)")
            // d/db = a = [1, 2, 3]
            XCTAssertEqual(gradValues[0], 1.0, accuracy: 1e-4)
            XCTAssertEqual(gradValues[1], 2.0, accuracy: 1e-4)
            XCTAssertEqual(gradValues[2], 3.0, accuracy: 1e-4)
        }
    }
}
