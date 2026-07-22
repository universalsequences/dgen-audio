import DGen
import XCTest

@testable import DGenLazy

final class TensorBiquadTests: XCTestCase {
  override func setUp() {
    super.setUp()
    DGenConfig.backend = .metal
    DGenConfig.sampleRate = 44_100
    DGenConfig.defaultFrameCount = 4096
    DGenConfig.maxFrameCount = 4096
    LazyGraphContext.reset()
  }

  override func tearDown() {
    DGenConfig.backend = .metal
    DGenConfig.sampleRate = 44_100
    DGenConfig.defaultFrameCount = 1024
    DGenConfig.maxFrameCount = 4096
    super.tearDown()
  }

  private func configure(sampleRate: Float, frames: Int) {
    DGenConfig.sampleRate = sampleRate
    DGenConfig.defaultFrameCount = frames
    DGenConfig.maxFrameCount = frames
    LazyGraphContext.reset()
  }

  private func sine(_ frequency: Float, sampleRate: Float) -> Signal {
    let phase = Signal.accum(
      Signal.constant(2 * Float.pi * frequency / sampleRate),
      reset: 0, min: 0, max: 1e9)
    return DGenLazy.sin(phase)
  }

  private func lane(_ flat: [Float], index: Int, batchSize: Int) -> [Float] {
    stride(from: index, to: flat.count, by: batchSize).map { flat[$0] }
  }

  private func maxAbsDiff(_ lhs: [Float], _ rhs: [Float]) -> Float {
    XCTAssertEqual(lhs.count, rhs.count)
    return zip(lhs, rhs).reduce(0) { Swift.max($0, Swift.abs($1.0 - $1.1)) }
  }

  private func renderScalar(
    sampleRate: Float, frames: Int, frequency: Float,
    cutoff: Float, resonance: Float, gain: Float = 1, mode: Int = 0
  ) throws -> [Float] {
    configure(sampleRate: sampleRate, frames: frames)
    return try sine(frequency, sampleRate: sampleRate)
      .biquad(cutoff: cutoff, resonance: resonance, gain: gain, mode: mode)
      .realize(frames: frames)
  }

  private func renderScalarWithParameterControls(
    sampleRate: Float, frames: Int, frequency: Float,
    cutoff: Float, resonance: Float
  ) throws -> [Float] {
    configure(sampleRate: sampleRate, frames: frames)
    return try sine(frequency, sampleRate: sampleRate).biquad(
      cutoff: Signal.param(cutoff), resonance: Signal.param(resonance),
      gain: Signal.constant(1), mode: Signal.constant(0)
    ).realize(frames: frames)
  }

  func testBatchOneMatchesScalarForEveryModeAndSampleRate() throws {
    let frames = 4096
    for sampleRate: Float in [2000, 44_100] {
      let frequency: Float = sampleRate == 2000 ? 170 : 440
      let cutoff: Float = sampleRate == 2000 ? 320 : 1400

      for mode in 0...7 {
        let gain: Float = mode >= 6 ? 1.7 : 1
        let expected = try renderScalar(
          sampleRate: sampleRate, frames: frames, frequency: frequency,
          cutoff: cutoff, resonance: 0.9, gain: gain, mode: mode)

        configure(sampleRate: sampleRate, frames: frames)
        let input = Tensor([Float(1)]) * sine(frequency, sampleRate: sampleRate)
        let actual = try input.biquad(
          cutoff: cutoff, resonance: 0.9, gain: gain, mode: mode
        ).realize(frames: frames)

        XCTAssertLessThan(
          maxAbsDiff(actual, expected), 1e-6,
          "B=1 mismatch for mode \(mode) at \(sampleRate) Hz")
      }
    }
  }

  func testSharedControlsMatchEightDistinctSerialInputs() throws {
    let frames = 1024
    let sampleRate: Float = 44_100
    let frequencies: [Float] = [110, 173, 241, 337, 463, 617, 809, 997]
    let batchSize = frequencies.count
    let expected = try frequencies.map {
      try renderScalar(
        sampleRate: sampleRate, frames: frames, frequency: $0,
        cutoff: 1200, resonance: 0.8)
    }

    configure(sampleRate: sampleRate, frames: frames)
    var batchedInput: SignalTensor?
    for (index, frequency) in frequencies.enumerated() {
      var mask = [Float](repeating: 0, count: batchSize)
      mask[index] = 1
      let component = Tensor(mask) * sine(frequency, sampleRate: sampleRate)
      batchedInput = batchedInput.map { $0 + component } ?? component
    }

    let flat = try XCTUnwrap(batchedInput).biquad(
      cutoff: 1200, resonance: 0.8, gain: 1, mode: 0
    ).realize(frames: frames)

    for index in 0..<batchSize {
      XCTAssertLessThan(
        maxAbsDiff(lane(flat, index: index, batchSize: batchSize), expected[index]),
        1e-5, "shared-control lane \(index)")
    }
  }

  func testLaneStateIsIndependent() throws {
    let frames = 512
    let batchSize = 8
    configure(sampleRate: 44_100, frames: frames)

    var impulseMask = [Float](repeating: 0, count: batchSize)
    impulseMask[0] = 1
    let input = Tensor(impulseMask) * Signal.click()
    let flat = try input.biquad(
      cutoff: 1000, resonance: 0.707, gain: 1, mode: 0
    ).realize(frames: frames)

    XCTAssertGreaterThan(lane(flat, index: 0, batchSize: batchSize).map(abs).max() ?? 0, 0)
    for index in 1..<batchSize {
      XCTAssertTrue(
        lane(flat, index: index, batchSize: batchSize).allSatisfy { $0 == 0 },
        "silent lane \(index) received another lane's filter state")
    }
  }

  func testPerElementControlsMatchSerialFilters() throws {
    let frames = 2048
    let sampleRate: Float = 44_100
    let cutoffs: [Float] = [80, 140, 260, 480, 900, 1700, 3600, 8000]
    let resonances: [Float] = [0.55, 0.7, 0.9, 1.2, 1.7, 2.3, 3.1, 4.0]
    let batchSize = cutoffs.count
    let expected = try zip(cutoffs, resonances).map {
      try renderScalarWithParameterControls(
        sampleRate: sampleRate, frames: frames, frequency: 440,
        cutoff: $0.0, resonance: $0.1)
    }

    configure(sampleRate: sampleRate, frames: frames)
    let input = Tensor([Float](repeating: 1, count: batchSize))
      * sine(440, sampleRate: sampleRate)
    let cutoff = Tensor(cutoffs) * Signal.constant(1)
    let resonance = Tensor(resonances) * Signal.constant(1)
    let filtered = input.biquad(
      cutoff: cutoff, resonance: resonance,
      gain: Signal.constant(1), mode: Signal.constant(0))

    let historyCells = Set(filtered.graph.graph.nodes.values.compactMap { node -> CellID? in
      switch node.op {
      case .historyRead(let cell), .historyWrite(let cell): return cell
      default: return nil
      }
    })
    XCTAssertEqual(historyCells.count, 4)
    for cell in historyCells {
      XCTAssertEqual(filtered.graph.graph.cellAllocationSizes[cell], batchSize)
      let tensorId = try XCTUnwrap(filtered.graph.graph.cellToTensor[cell])
      XCTAssertEqual(filtered.graph.graph.tensors[tensorId]?.shape, [batchSize])
    }

    let flat = try filtered.realize(frames: frames)
    for index in 0..<batchSize {
      XCTAssertLessThan(
        maxAbsDiff(lane(flat, index: index, batchSize: batchSize), expected[index]),
        1e-4, "per-element control lane \(index)")
    }
  }

  func testTimeVaryingPerElementCutoffMatchesSerialFilters() throws {
    let frames = 2048
    let sampleRate: Float = 44_100
    let bases: [Float] = [80, 120, 180, 260, 380, 550, 800, 1200]
    let amounts: [Float] = [200, 300, 450, 650, 900, 1300, 1900, 2800]
    let decays: [Float] = [0.03, 0.05, 0.07, 0.1, 0.14, 0.2, 0.3, 0.45]
    let resonances: [Float] = [0.6, 0.75, 0.9, 1.1, 1.35, 1.7, 2.2, 3.0]
    let batchSize = bases.count

    var expected = [[Float]]()
    for index in 0..<batchSize {
      configure(sampleRate: sampleRate, frames: frames)
      let t = Signal.accum(
        Signal.constant(1 / sampleRate), reset: 0, min: 0,
        max: Float(frames + 1) / sampleRate + 1)
      let cutoff = Signal.constant(bases[index])
        + Signal.constant(amounts[index])
        * DGenLazy.exp(-t / Signal.constant(decays[index]))
      let output = sine(330, sampleRate: sampleRate).biquad(
        cutoff: cutoff, resonance: Signal.constant(resonances[index]),
        gain: Signal.constant(1), mode: Signal.constant(0))
      expected.append(try output.realize(frames: frames))
    }

    configure(sampleRate: sampleRate, frames: frames)
    let t = Signal.accum(
      Signal.constant(1 / sampleRate), reset: 0, min: 0,
      max: Float(frames + 1) / sampleRate + 1)
    let input = Tensor([Float](repeating: 1, count: batchSize))
      * sine(330, sampleRate: sampleRate)
    let base = Tensor(bases) * Signal.constant(1)
    let amount = Tensor(amounts) * Signal.constant(1)
    let decay = Tensor(decays) * Signal.constant(1)
    let cutoff = base + amount * DGenLazy.exp(-t / decay)
    let resonance = Tensor(resonances) * Signal.constant(1)
    let flat = try input.biquad(
      cutoff: cutoff, resonance: resonance,
      gain: Signal.constant(1), mode: Signal.constant(0)
    ).realize(frames: frames)

    for index in 0..<batchSize {
      XCTAssertLessThan(
        maxAbsDiff(lane(flat, index: index, batchSize: batchSize), expected[index]),
        1e-4, "time-varying control lane \(index)")
    }
  }

  func testBackwardThroughTensorBiquadThrowsExplicitError() throws {
    // Rank-1 [B] shapes support backward; anything else must still fail loudly.
    configure(sampleRate: 44_100, frames: 64)
    let input = (Tensor([Float](repeating: 1, count: 4), requiresGrad: true)
      * sine(220, sampleRate: 44_100)).reshape([2, 2])
    let loss = input.biquad(
      cutoff: 1000, resonance: 0.707, gain: 1, mode: 0
    ).sum()

    XCTAssertThrowsError(try loss.backward(frames: 64)) { error in
      guard case DGenError.unsupportedGradient(let reason) = error else {
        return XCTFail("unexpected error: \(error)")
      }
      XCTAssertTrue(reason.contains("tensor-shaped biquad"))
    }
  }
}
