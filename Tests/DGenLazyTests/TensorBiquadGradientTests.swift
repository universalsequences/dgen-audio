import DGen
import Foundation
import XCTest

@testable import DGenLazy

/// Gradient (BPTT) tests for the tensor-shaped biquad, validated against the
/// scalar `Signal.biquad` backward path as ground truth.
/// See docs/TENSOR_BIQUAD_GRADIENT_SPEC.md.
final class TensorBiquadGradientTests: XCTestCase {
  override func setUp() {
    super.setUp()
    DGenConfig.backend = .metal
    DGenConfig.sampleRate = 44_100
    DGenConfig.defaultFrameCount = 1024
    DGenConfig.maxFrameCount = 1024
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
    if let dump = ProcessInfo.processInfo.environment["TENSOR_BIQUAD_GRAD_DUMP"] {
      DGenConfig.kernelOutputPath = dump
    }
    LazyGraphContext.reset()
  }

  private func sine(_ frequency: Float, sampleRate: Float) -> Signal {
    let phase = Signal.accum(
      Signal.constant(2 * Float.pi * frequency / sampleRate),
      reset: 0, min: 0, max: 1e9)
    return DGenLazy.sin(phase)
  }

  /// Scalar ground truth: time-MSE loss through Signal.biquad, cutoff behind
  /// exp(z) so the gradient chain matches the tensor build exactly.
  private func scalarCutoffGrad(
    sampleRate: Float, frames: Int, frequency: Float, teacherFrequency: Float,
    cutoffZ: Float, resonance: Float, gain: Float, mode: Int
  ) throws -> Float {
    configure(sampleRate: sampleRate, frames: frames)
    let z = Signal.param(cutoffZ)
    let y = sine(frequency, sampleRate: sampleRate).biquad(
      cutoff: DGenLazy.exp(z),
      resonance: Signal.constant(resonance),
      gain: Signal.constant(gain),
      mode: Signal.constant(Float(mode)))
    let teacher = sine(teacherFrequency, sampleRate: sampleRate)
    let diff = y - teacher
    let loss = diff * diff
    _ = try loss.backward(frames: frames)
    guard let grad = z.grad?.data else {
      XCTFail("scalar cutoff grad missing")
      return .nan
    }
    return grad
  }

  func testBatchOneCutoffGradMatchesScalarForEveryMode() throws {
    let frames = 1024
    let sampleRate: Float = 44_100
    let cutoffZ = Foundation.log(Float(1500))

    for mode in 0...7 {
      let gain: Float = mode >= 6 ? 1.7 : 1
      let reference = try scalarCutoffGrad(
        sampleRate: sampleRate, frames: frames, frequency: 220, teacherFrequency: 97,
        cutoffZ: cutoffZ, resonance: 0.707, gain: gain, mode: mode)

      configure(sampleRate: sampleRate, frames: frames)
      let z = Signal.param(cutoffZ)
      let input = Tensor([Float(1)]) * sine(220, sampleRate: sampleRate)
      let y = input.biquad(
        cutoff: DGenLazy.exp(z),
        resonance: Signal.constant(0.707),
        gain: Signal.constant(gain),
        mode: Signal.constant(Float(mode)))
      let teacher = Tensor([Float(1)]) * sine(97, sampleRate: sampleRate)
      let diff = y - teacher
      let loss = (diff * diff).sum()
      _ = try loss.backward(frames: frames)
      let batched = z.grad?.data ?? .nan

      XCTAssertFalse(batched.isNaN, "mode \(mode): batched cutoff grad is NaN")
      XCTAssertEqual(
        batched, reference,
        accuracy: Swift.max(Swift.abs(reference) * 5e-3, 1e-6),
        "mode \(mode): batched [1] cutoff grad \(batched) vs scalar \(reference)")
    }
  }

  /// Spec test 2: one shared cutoff across 8 lanes with distinct inputs and
  /// targets. The shared-param gradient must equal the sum of the per-lane
  /// serial gradients.
  func testSharedControlGradIsSumOfPerLaneSerialGrads() throws {
    let frames = 1024
    let sampleRate: Float = 44_100
    let frequencies: [Float] = [110, 173, 241, 337, 463, 617, 809, 997]
    let teacherFrequencies: [Float] = [97, 151, 219, 307, 431, 577, 761, 941]
    let batchSize = frequencies.count
    let cutoffZ = Foundation.log(Float(1200))

    var serialSum: Float = 0
    for i in 0..<batchSize {
      serialSum += try scalarCutoffGrad(
        sampleRate: sampleRate, frames: frames,
        frequency: frequencies[i], teacherFrequency: teacherFrequencies[i],
        cutoffZ: cutoffZ, resonance: 0.8, gain: 1, mode: 0)
    }

    configure(sampleRate: sampleRate, frames: frames)
    let z = Signal.param(cutoffZ)
    var input: SignalTensor?
    var teacher: SignalTensor?
    for i in 0..<batchSize {
      var mask = [Float](repeating: 0, count: batchSize)
      mask[i] = 1
      let inComponent = Tensor(mask) * sine(frequencies[i], sampleRate: sampleRate)
      input = input.map { $0 + inComponent } ?? inComponent
      let tComponent = Tensor(mask) * sine(teacherFrequencies[i], sampleRate: sampleRate)
      teacher = teacher.map { $0 + tComponent } ?? tComponent
    }

    let y = try XCTUnwrap(input).biquad(
      cutoff: DGenLazy.exp(z),
      resonance: Signal.constant(0.8),
      gain: Signal.constant(1),
      mode: Signal.constant(0))
    let diff = y - (try XCTUnwrap(teacher))
    let loss = (diff * diff).sum()
    _ = try loss.backward(frames: frames)
    let batched = z.grad?.data ?? .nan

    XCTAssertFalse(batched.isNaN)
    XCTAssertEqual(
      batched, serialSum,
      accuracy: Swift.max(Swift.abs(serialSum) * 1e-2, 1e-5),
      "shared cutoff grad \(batched) vs sum of serial lane grads \(serialSum)")
  }

  /// Spec test 3: perturbing lane 0's target must not change lanes 1..7's
  /// gradients (catches carry-cell / grad-cell aliasing across lanes).
  func testLaneGradientIndependence() throws {
    let frames = 512
    let sampleRate: Float = 44_100
    let batchSize = 8

    func laneInputGrads(lane0TeacherFrequency: Float) throws -> [Float] {
      configure(sampleRate: sampleRate, frames: frames)
      let gains = Tensor([Float](repeating: 1, count: batchSize), requiresGrad: true)
      let input = gains * sine(220, sampleRate: sampleRate)
      let y = input.biquad(
        cutoff: Signal.constant(1000),
        resonance: Signal.constant(0.707),
        gain: Signal.constant(1),
        mode: Signal.constant(0))
      var teacher: SignalTensor?
      for i in 0..<batchSize {
        var mask = [Float](repeating: 0, count: batchSize)
        mask[i] = 1
        let f: Float = i == 0 ? lane0TeacherFrequency : Float(97 + 40 * i)
        let component = Tensor(mask) * sine(f, sampleRate: sampleRate)
        teacher = teacher.map { $0 + component } ?? component
      }
      let diff = y - (try XCTUnwrap(teacher))
      let loss = (diff * diff).sum()
      _ = try loss.backward(frames: frames)
      let grads = try XCTUnwrap(gains.grad?.getData())
      XCTAssertEqual(grads.count, batchSize)
      return grads
    }

    let base = try laneInputGrads(lane0TeacherFrequency: 97)
    let perturbed = try laneInputGrads(lane0TeacherFrequency: 397)

    XCTAssertNotEqual(base[0], perturbed[0], "lane 0 grad should respond to its own target")
    for i in 1..<batchSize {
      XCTAssertEqual(
        base[i], perturbed[i],
        "lane \(i) grad changed when only lane 0's target moved (cross-lane aliasing)")
    }
  }
}

extension TensorBiquadGradientTests {
  /// Spec test 4: 8 distinct per-element cutoffs; each lane's cutoff gradient
  /// must match its independent serial equivalent and not leak across lanes.
  func testPerElementControlGradsMatchSerialPerLane() throws {
    let frames = 1024
    let sampleRate: Float = 44_100
    let frequencies: [Float] = [110, 173, 241, 337, 463, 617, 809, 997]
    let teacherFrequencies: [Float] = [97, 151, 219, 307, 431, 577, 761, 941]
    let cutoffZs: [Float] = [400, 700, 1000, 1400, 1900, 2500, 3200, 4000].map {
      Foundation.log($0)
    }
    let batchSize = frequencies.count

    var serial = [Float]()
    for i in 0..<batchSize {
      serial.append(
        try scalarCutoffGrad(
          sampleRate: sampleRate, frames: frames,
          frequency: frequencies[i], teacherFrequency: teacherFrequencies[i],
          cutoffZ: cutoffZs[i], resonance: 0.8, gain: 1, mode: 0))
    }

    configure(sampleRate: sampleRate, frames: frames)
    let z = Tensor(cutoffZs, requiresGrad: true)
    let cutoff = DGenLazy.exp(z * Signal.constant(1))
    let resonance = Tensor([Float](repeating: 0.8, count: batchSize)) * Signal.constant(1)
    var input: SignalTensor?
    var teacher: SignalTensor?
    for i in 0..<batchSize {
      var mask = [Float](repeating: 0, count: batchSize)
      mask[i] = 1
      let inComponent = Tensor(mask) * sine(frequencies[i], sampleRate: sampleRate)
      input = input.map { $0 + inComponent } ?? inComponent
      let tComponent = Tensor(mask) * sine(teacherFrequencies[i], sampleRate: sampleRate)
      teacher = teacher.map { $0 + tComponent } ?? tComponent
    }

    let y = try XCTUnwrap(input).biquad(
      cutoff: cutoff, resonance: resonance,
      gain: Signal.constant(1), mode: Signal.constant(0))
    let diff = y - (try XCTUnwrap(teacher))
    let loss = (diff * diff).sum()
    _ = try loss.backward(frames: frames)
    let batched = try XCTUnwrap(z.grad?.getData())
    XCTAssertEqual(batched.count, batchSize)

    for i in 0..<batchSize {
      XCTAssertEqual(
        batched[i], serial[i],
        accuracy: Swift.max(Swift.abs(serial[i]) * 1e-2, 1e-5),
        "lane \(i): per-element cutoff grad \(batched[i]) vs serial \(serial[i])")
    }
  }

  /// Spec test 5: time-varying per-element cutoff (the subtractive-bass
  /// shape): cutoff_i(t) = base_i + amt_i * exp(-t / decay_i). Per-lane
  /// gradients for base/amt/decay must match serial equivalents.
  func testTimeVaryingPerElementCutoffGradsMatchSerial() throws {
    let frames = 1024
    let sampleRate: Float = 44_100
    let bases: [Float] = [200, 300, 450, 650, 900, 1300, 1900, 2800]
    let amounts: [Float] = [400, 550, 700, 900, 1100, 1400, 1700, 2000]
    let decays: [Float] = [0.003, 0.005, 0.007, 0.01, 0.014, 0.02, 0.03, 0.045]
    let frequencies: [Float] = [110, 173, 241, 337, 463, 617, 809, 997]
    let teacherFrequencies: [Float] = [97, 151, 219, 307, 431, 577, 761, 941]
    let batchSize = bases.count

    func timeSignal() -> Signal {
      Signal.accum(
        Signal.constant(1 / sampleRate), reset: 0, min: 0,
        max: Float(frames + 1) / sampleRate + 1)
    }

    var serial = [(base: Float, amt: Float, decay: Float)]()
    for i in 0..<batchSize {
      configure(sampleRate: sampleRate, frames: frames)
      let base = Signal.param(bases[i])
      let amt = Signal.param(amounts[i])
      let decay = Signal.param(decays[i])
      let cutoff = base + amt * DGenLazy.exp(-timeSignal() / decay)
      let y = sine(frequencies[i], sampleRate: sampleRate).biquad(
        cutoff: cutoff, resonance: Signal.constant(0.8),
        gain: Signal.constant(1), mode: Signal.constant(0))
      let teacher = sine(teacherFrequencies[i], sampleRate: sampleRate)
      let diff = y - teacher
      let loss = diff * diff
      _ = try loss.backward(frames: frames)
      serial.append(
        (
          base: base.grad?.data ?? .nan,
          amt: amt.grad?.data ?? .nan,
          decay: decay.grad?.data ?? .nan
        ))
    }

    configure(sampleRate: sampleRate, frames: frames)
    let baseT = Tensor(bases, requiresGrad: true)
    let amtT = Tensor(amounts, requiresGrad: true)
    let decayT = Tensor(decays, requiresGrad: true)
    let cutoff =
      baseT * Signal.constant(1)
      + amtT * Signal.constant(1) * DGenLazy.exp(-timeSignal() / (decayT * Signal.constant(1)))
    let resonance = Tensor([Float](repeating: 0.8, count: batchSize)) * Signal.constant(1)
    var input: SignalTensor?
    var teacher: SignalTensor?
    for i in 0..<batchSize {
      var mask = [Float](repeating: 0, count: batchSize)
      mask[i] = 1
      let inComponent = Tensor(mask) * sine(frequencies[i], sampleRate: sampleRate)
      input = input.map { $0 + inComponent } ?? inComponent
      let tComponent = Tensor(mask) * sine(teacherFrequencies[i], sampleRate: sampleRate)
      teacher = teacher.map { $0 + tComponent } ?? tComponent
    }

    let y = try XCTUnwrap(input).biquad(
      cutoff: cutoff, resonance: resonance,
      gain: Signal.constant(1), mode: Signal.constant(0))
    let diff = y - (try XCTUnwrap(teacher))
    let loss = (diff * diff).sum()
    _ = try loss.backward(frames: frames)
    let baseGrads = try XCTUnwrap(baseT.grad?.getData())
    let amtGrads = try XCTUnwrap(amtT.grad?.getData())
    let decayGrads = try XCTUnwrap(decayT.grad?.getData())

    for i in 0..<batchSize {
      XCTAssertEqual(
        baseGrads[i], serial[i].base,
        accuracy: Swift.max(Swift.abs(serial[i].base) * 2e-2, 1e-5),
        "lane \(i) fBase grad")
      XCTAssertEqual(
        amtGrads[i], serial[i].amt,
        accuracy: Swift.max(Swift.abs(serial[i].amt) * 2e-2, 1e-5),
        "lane \(i) fAmt grad")
      XCTAssertEqual(
        decayGrads[i], serial[i].decay,
        accuracy: Swift.max(Swift.abs(serial[i].decay) * 2e-2, 1e-2),
        "lane \(i) fDecay grad")
    }
  }

  /// Spec test 7: finite-difference check per lane. Time-domain MSE loss
  /// (linear magnitude, FD-stable per FDCHECK_FINDING.md) with an eps sweep;
  /// autograd per-lane cutoff grads must match the stable FD estimate.
  func testPerLaneFDCheck() throws {
    let frames = 512
    let sampleRate: Float = 44_100
    let frequencies: [Float] = [110, 241, 463, 809]
    let teacherFrequencies: [Float] = [97, 219, 431, 761]
    let baseZs: [Float] = [500, 1000, 2000, 3500].map { Foundation.log($0) }
    let batchSize = frequencies.count

    func lossSum(zs: [Float]) throws -> (loss: Float, grads: [Float]) {
      configure(sampleRate: sampleRate, frames: frames)
      let z = Tensor(zs, requiresGrad: true)
      let cutoff = DGenLazy.exp(z * Signal.constant(1))
      let resonance = Tensor([Float](repeating: 0.8, count: batchSize)) * Signal.constant(1)
      var input: SignalTensor?
      var teacher: SignalTensor?
      for i in 0..<batchSize {
        var mask = [Float](repeating: 0, count: batchSize)
        mask[i] = 1
        let inComponent = Tensor(mask) * sine(frequencies[i], sampleRate: sampleRate)
        input = input.map { $0 + inComponent } ?? inComponent
        let tComponent = Tensor(mask) * sine(teacherFrequencies[i], sampleRate: sampleRate)
        teacher = teacher.map { $0 + tComponent } ?? tComponent
      }
      let y = try XCTUnwrap(input).biquad(
        cutoff: cutoff, resonance: resonance,
        gain: Signal.constant(1), mode: Signal.constant(0))
      let diff = y - (try XCTUnwrap(teacher))
      let loss = (diff * diff).sum()
      let lossValues = try loss.backward(frames: frames)
      return (lossValues.reduce(0, +), z.grad?.getData() ?? [])
    }

    let (_, autograd) = try lossSum(zs: baseZs)
    XCTAssertEqual(autograd.count, batchSize)

    for lane in 0..<batchSize {
      var fdEstimates = [Float]()
      for eps in [Float(1e-3), 3e-3, 1e-2] {
        var plus = baseZs
        plus[lane] += eps
        var minus = baseZs
        minus[lane] -= eps
        let fd = (try lossSum(zs: plus).loss - lossSum(zs: minus).loss) / (2 * eps)
        fdEstimates.append(fd)
      }
      // Require FD stability across the two largest eps, then compare autograd.
      let stable = fdEstimates[2]
      XCTAssertEqual(
        fdEstimates[1], stable,
        accuracy: Swift.max(Swift.abs(stable) * 0.05, 1e-4),
        "lane \(lane): FD unstable across eps sweep \(fdEstimates)")
      XCTAssertEqual(
        autograd[lane], stable,
        accuracy: Swift.max(Swift.abs(stable) * 0.05, 1e-4),
        "lane \(lane): autograd \(autograd[lane]) vs FD \(stable) (sweep \(fdEstimates))")
    }
  }

  /// Temporary diagnostic: B=2 with lane 1 silent — shared grad must equal lane 0's serial grad.
  func testDiagSharedB2SilentLane() throws {
    let frames = 1024
    let sampleRate: Float = 44_100
    let cutoffZ = Foundation.log(Float(1200))
    let serial = try scalarCutoffGrad(
      sampleRate: sampleRate, frames: frames, frequency: 110, teacherFrequency: 97,
      cutoffZ: cutoffZ, resonance: 0.8, gain: 1, mode: 0)

    configure(sampleRate: sampleRate, frames: frames)
    let z = Signal.param(cutoffZ)
    let input = Tensor([1, 0]) * sine(110, sampleRate: sampleRate)
    let teacher = Tensor([1, 0]) * sine(97, sampleRate: sampleRate)
    let y = input.biquad(
      cutoff: DGenLazy.exp(z), resonance: Signal.constant(0.8),
      gain: Signal.constant(1), mode: Signal.constant(0))
    let diff = y - teacher
    let loss = (diff * diff).sum()
    _ = try loss.backward(frames: frames)
    let batched = z.grad?.data ?? .nan
    XCTAssertEqual(batched, serial, accuracy: Swift.max(Swift.abs(serial) * 1e-2, 1e-5),
      "B=2 silent lane: \(batched) vs serial \(serial)")
  }
}
