import XCTest

@testable import DGenLazy

/// Tests for tensor-based history feedback loops
/// Working toward full membrane simulation
final class HistoryTensorTests: XCTestCase {

  override func setUp() {
    super.setUp()
    LazyGraphContext.reset()
  }

  // MARK: - Simple Tensor Accumulator

  /// Simplest test: tensor that accumulates over frames
  func testTensorAccumulator() throws {
    let size = 3
    let frameCount = 4

    // Increment each frame
    let increment = Tensor([1.0, 2.0, 3.0])

    // Create history buffer using wrapper
    let history = TensorHistory(shape: [size])

    // Read previous value
    let prev = history.read()

    // Add increment (SignalTensor + Tensor = SignalTensor)
    let newValue = prev + increment

    // Write back
    history.write(newValue)

    // Output: sum (SignalTensor.sum() -> Signal)
    let output = newValue.sum()

    // Run with frames
    let result = try output.realize(frames: frameCount)

    print("\n=== Tensor Accumulator ===")
    print("Increment: [1, 2, 3], Frames: \(frameCount)")
    print("Sum per frame: \(result)")

    // Frame 0: 0+1+2+3 = 6 (prev is 0)
    // Frame 1: (1+1)+(2+2)+(3+3) = 12
    // Frame 2: 18
    // Frame 3: 24
    XCTAssertEqual(result[0], 6.0, accuracy: 1e-5, "Frame 0 sum")
    XCTAssertEqual(result[1], 12.0, accuracy: 1e-5, "Frame 1 sum")
    XCTAssertEqual(result[2], 18.0, accuracy: 1e-5, "Frame 2 sum")
    XCTAssertEqual(result[3], 24.0, accuracy: 1e-5, "Frame 3 sum")
  }

  // MARK: - One-Pole Filter with Tensor

  /// One-pole filter: y[n] = a * x[n] + (1-a) * y[n-1]
  func testOnePoleFilter() throws {
    let numChannels = 4
    let frameCount = 8

    // Frequencies for each channel
    let freqs = Tensor([100.0, 200.0, 300.0, 400.0])

    // Phasor per channel (creates SignalTensor)
    let phasors = Signal.phasor(freqs)

    // Filter coefficient
    let alpha: Float = 0.1

    // Create history buffer
    let history = TensorHistory(shape: [numChannels])

    // Read previous output
    let prevOutput = history.read()

    // One-pole: y = alpha * x + (1-alpha) * y_prev
    let scaledInput = phasors * alpha
    let scaledPrev = prevOutput * (1.0 - alpha)
    let newOutput = scaledInput + scaledPrev

    // Write new output to history
    history.write(newOutput)

    // Output: sum of all channels
    let output = newOutput.sum()

    // Run multiple frames
    let result = try output.realize(frames: frameCount)

    print("\n=== One-Pole Filter (Tensor) ===")
    print("Channels: \(numChannels), Frames: \(frameCount)")
    print("Alpha: \(alpha)")
    print("Output per frame: \(result)")

    // Verify we got the right number of outputs
    XCTAssertEqual(result.count, frameCount)

    // Output should show smoothing behavior
    print("First 4 frames: \(Array(result.prefix(4)))")
  }

  // MARK: - Conv2D in Feedback Loop

  /// Feedback loop using proper conv2d with kernel tensor
  /// Demonstrates: history.read() -> conv2d(kernel) -> history.write()
  func testConv2DFeedback() throws {
    // Enable kernel output and debug for analysis
    DGenConfig.kernelOutputPath = "/tmp/conv2d_feedback_kernel.metal"
    DGenConfig.debug = true

    let frameCount = 4

    // 4x4 state with center impulse
    let initialState: [Float] = [
      0.0, 0.0, 0.0, 0.0,
      0.0, 1.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0,
    ]

    // 2x2 averaging kernel (sums to 1.0 to preserve energy)
    let kernel = Tensor([
      [0.25, 0.25],
      [0.25, 0.25],
    ])

    // Create history buffer with initial state
    let history = TensorHistory(shape: [4, 4], data: initialState)

    // Read previous state (SignalTensor [4, 4])
    let state = history.read()

    // Apply proper conv2d with kernel tensor
    // state [4,4] conv kernel [2,2] -> convResult [3,3]
    let convResult = state.conv2d(kernel)

    // Decay the original state and use convResult as output
    // (A full membrane would combine these, but for testing just decay)
    let decayed = state * 0.9

    // Write decayed state back
    history.write(decayed)

    // Output: sum of convolution result
    // The center impulse at [1,1] contributes to windows at [0,0], [0,1], [1,0], [1,1]
    // Each window gets 0.25 * 1.0 = 0.25, so total = 1.0
    let output = convResult.sum()

    let result = try output.realize(frames: frameCount)

    print("\n=== Conv2D Feedback (proper kernel) ===")
    print("Initial: 4x4 with 1.0 at [1,1]")
    print("Kernel: 2x2 averaging (0.25 each)")
    print("Conv sum per frame: \(result)")

    // EXPECTED (with sequential frame execution):
    // Frame 0: impulse at [1,1], conv sum = 1.0 (impulse appears in 4 windows)
    // Frame 1: decayed by 0.9, conv sum = 0.9
    // Frame 2: conv sum = 0.81
    // Frame 3: conv sum = 0.729
    XCTAssertEqual(result[0], 1.0, accuracy: 1e-4, "Frame 0 conv sum")
    XCTAssertEqual(result[1], 0.9, accuracy: 1e-4, "Frame 1 conv sum")
    XCTAssertEqual(result[2], 0.81, accuracy: 1e-4, "Frame 2 conv sum")
    XCTAssertEqual(result[3], 0.729, accuracy: 1e-4, "Frame 3 conv sum")

    // Clean up config
    DGenConfig.kernelOutputPath = nil
    DGenConfig.debug = false
  }

  // MARK: - Conv2D WITHOUT History (Isolation Test)

  /// Same conv2d computation as testConv2DFeedback but with static tensor (no history write)
  /// This isolates whether the [1.0, 0.0, 0.0, 0.0] result comes from history feedback or conv2d/sumAxis
  func testConv2DStatic() throws {
    DGenConfig.kernelOutputPath = "/tmp/conv2d_static_kernel.metal"
    DGenConfig.debug = true

    let frameCount = 4

    // All zeros - should produce [0, 0, 0, 0]
    let zeroData: [Float] = Array(repeating: 0.0, count: 16)

    // Use TensorHistory but DON'T write to it - stays constant
    let history = TensorHistory(shape: [4, 4], data: zeroData)
    let state = history.read()  // SignalTensor (constant across frames)
    // No history.write() - state stays zero every frame

    // 2x2 averaging kernel
    let kernel = Tensor([
      [0.25, 0.25],
      [0.25, 0.25],
    ])

    // Proper conv2d with kernel tensor
    let convResult = state.conv2d(kernel)
    let output = convResult.sum()

    let result = try output.realize(frames: frameCount)

    print("\n=== Conv2D Static (zeros, no history write) ===")
    print("State: 4x4 zeros")
    print("Conv sum per frame: \(result)")

    // All frames should be 0 since input is 0
    for i in 0..<frameCount {
      XCTAssertEqual(result[i], 0.0, accuracy: 1e-4, "Frame \(i) should be 0")
    }

    DGenConfig.kernelOutputPath = nil
    DGenConfig.debug = false
  }

  /// Conv2d with static impulse (no history write) - should produce same value every frame
  func testConv2DStaticImpulse() throws {
    DGenConfig.kernelOutputPath = "/tmp/conv2d_static_impulse_kernel.metal"
    DGenConfig.debug = true

    let frameCount = 4

    // Same impulse as testConv2DFeedback
    let impulseData: [Float] = [
      0.0, 0.0, 0.0, 0.0,
      0.0, 1.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0,
    ]

    // Use TensorHistory - add dummy write to preserve shape
    let history = TensorHistory(shape: [4, 4], data: impulseData)
    let state = history.read()  // SignalTensor (constant across frames)
    // Write same value back (no actual change, but establishes shape)
    history.write(state)

    let kernel = Tensor([
      [0.25, 0.25],
      [0.25, 0.25],
    ])

    // Proper conv2d with kernel tensor
    let convResult = state.conv2d(kernel)
    let output = convResult.sum()

    let result = try output.realize(frames: frameCount)

    print("\n=== Conv2D Static (impulse, no history write) ===")
    print("State: 4x4 with 1.0 at [1,1]")
    print("Conv sum per frame: \(result)")

    // All frames should be 1.0 (same static input, no decay)
    for i in 0..<frameCount {
      XCTAssertEqual(result[i], 1.0, accuracy: 1e-4, "Frame \(i) should be 1.0")
    }

    DGenConfig.kernelOutputPath = nil
    DGenConfig.debug = false
  }

  // MARK: - Simple Feedback with Decay

  /// Tensor with exponential decay feedback
  func testTensorDecay() throws {
    let size = 2
    let frameCount = 10

    // Initial impulse (only frame 0)
    // We'll add it via the history initial data
    let history = TensorHistory(shape: [size], data: [10.0, 20.0])

    // Decay factor
    let decay: Float = 0.8

    // Read and decay
    let prev = history.read()
    let decayed = prev * decay

    // Write back
    history.write(decayed)

    // Output: sum
    let output = decayed.sum()

    let result = try output.realize(frames: frameCount)

    print("\n=== Tensor Decay ===")
    print("Initial: [10, 20], Decay: \(decay)")
    print("Sum per frame: \(result)")

    // Frame 0: (10 + 20) * 0.8 = 24
    // Frame 1: 24 * 0.8 = 19.2
    // Frame 2: 19.2 * 0.8 = 15.36
    XCTAssertEqual(result.count, frameCount, "Should have correct frame count")
    XCTAssertEqual(result[0], 24.0, accuracy: 1e-4, "Frame 0")
    XCTAssertEqual(result[1], 19.2, accuracy: 1e-4, "Frame 1")
    XCTAssertEqual(result[2], 15.36, accuracy: 1e-3, "Frame 2")
  }

  /// Full membrane physical simulation with damping
  /// Implements: state_t+1 = (2-d)*state_t - (1-d)*state_t-1 + c²*laplacian(state_t)
  /// Functionally equivalent to TensorOpsTests.testMembraneSimulationExecute
  func testConv2DMembraneSimulation() throws {
    DGenConfig.kernelOutputPath = "/tmp/membrane_simulation.metal"

    let gridSize = 4
    let frameCount = 64

    // Physical parameters
    let cSquared: Float = 0.1  // Wave speed squared
    let damping: Float = 0.03  // Velocity-proportional damping

    // Coefficients for wave equation
    let twoMinusD: Float = 2.0 - damping  // 1.97
    let oneMinusD: Float = 1.0 - damping  // 0.97

    // Initial excitation: 2x2 block in center
    var exciteData = [Float](repeating: 0.0, count: gridSize * gridSize)
    exciteData[5] = 1.0  // [1,1]
    exciteData[6] = 1.0  // [1,2]
    exciteData[9] = 1.0  // [2,1]
    exciteData[10] = 1.0  // [2,2]
    let excitation = Tensor(exciteData).reshape([gridSize, gridSize])

    // Frame counter: 0, 1, 2, 3... (accum outputs value BEFORE incrementing)
    let frameCounter = Signal.accum(Signal.constant(1.0), min: 0.0, max: 10000.0)
    let isFirstFrame = frameCounter.eq(0.0)  // 1.0 on frame 0, 0.0 after

    // Gate excitation to only fire on first frame
    let gatedExcite = excitation * isFirstFrame  // Tensor * Signal -> SignalTensor

    // Two history buffers for wave equation (need current and previous state)
    let stateHistory = TensorHistory(shape: [gridSize, gridSize])
    let prevStateHistory = TensorHistory(shape: [gridSize, gridSize])

    // Read current and previous state
    let state_t_raw = stateHistory.read()
    let state_t_1 = prevStateHistory.read()

    // Add gated excitation (only on frame 0)
    let state_t = state_t_raw + gatedExcite

    // Laplacian kernel for wave equation
    let laplacianKernel = Tensor([
      [0.0, 1.0, 0.0],
      [1.0, -4.0, 1.0],
      [0.0, 1.0, 0.0],
    ])

    // Pad state for same-size convolution output (Dirichlet boundary = fixed at 0)
    let paddedState = state_t.pad([(1, 1), (1, 1)])  // [gridSize+2, gridSize+2]

    // Apply Laplacian: padded [6,6] conv kernel [3,3] -> [4,4]
    let laplacian = paddedState.conv2d(laplacianKernel)

    // Wave equation: state_t+1 = (2-d)*state_t - (1-d)*state_t-1 + c²*laplacian
    let scaledState = state_t * twoMinusD
    let scaledPrev = state_t_1 * oneMinusD
    let scaledLaplacian = laplacian * cSquared

    let diff = scaledState - scaledPrev
    let state_t_plus_1 = diff + scaledLaplacian

    // Shift state buffers: prev <- current, current <- new
    prevStateHistory.write(state_t)
    stateHistory.write(state_t_plus_1)

    // Output: sum of state (scalar for audio output)
    let output = state_t_plus_1.sum()

    // Run simulation
    let result = try output.realize(frames: frameCount)

    // Find peak and verify decay
    var maxOutput: Float = 0
    var maxFrame = 0
    for (i, val) in result.enumerated() {
      if abs(val) > maxOutput {
        maxOutput = abs(val)
        maxFrame = i
      }
    }
    // Verify we got output values
    XCTAssertEqual(result.count, frameCount)

    // Verify output is non-zero initially (excitation produces wave)
    let earlySum = result.prefix(10).reduce(0, +)
    XCTAssertNotEqual(earlySum, 0, "Early frames should have non-zero output from excitation")

    // Verify damping: late frames should have smaller magnitude than peak
    let lateAvg = result.suffix(10).map { abs($0) }.reduce(0, +) / 10.0
    XCTAssertLessThan(lateAvg, maxOutput, "Output should decay due to damping")

    DGenConfig.kernelOutputPath = nil
    DGenConfig.debug = false
  }
}
