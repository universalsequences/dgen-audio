import XCTest

@testable import DGenLazy

/// Tests for Signal.buffer() — sliding window view over a signal's history.
///
/// buffer(size) writes each frame's signal value into a flat history array
/// and returns a [1, size] tensor view via a slidingWindow transform.
/// Element i at frame f reads history[f - size + 1 + i], with out-of-bounds → 0.
final class BufferTests: XCTestCase {

  override func setUp() {
    super.setUp()
    LazyGraphContext.reset()
  }

  /// Verify buffer sum matches expected sliding window sums.
  /// Counter 0, 1, 2, ... with buffer(4) should sum the last 4 values.
  func testBufferSum() throws {
    let counter = Signal.accum(Signal.constant(1.0), min: 0.0, max: 10000.0)
    let result = try counter.buffer(size: 4).sum().realize(frames: 8)

    //   f=0: [0,0,0,0]→0   f=1: [0,0,0,1]→1   f=2: [0,0,1,2]→3   f=3: [0,1,2,3]→6
    //   f=4: [1,2,3,4]→10  f=5: [2,3,4,5]→14  f=6: [3,4,5,6]→18  f=7: [4,5,6,7]→22
    let expected: [Float] = [0, 1, 3, 6, 10, 14, 18, 22]
    for i in 0..<8 {
      XCTAssertEqual(result[i], expected[i], accuracy: 0.01, "Frame \(i) sum mismatch")
    }
  }

  /// Verify element ORDER using a weighted conv2d kernel [1, 10, 100, 1000].
  /// Each buffer position contributes to a unique decimal digit, so misordering
  /// produces a visibly wrong number (e.g. 1234 vs 4321).
  func testBufferElementOrder() throws {
    let counter = Signal.accum(Signal.constant(1.0), min: 0.0, max: 10000.0)
    let buf = counter.buffer(size: 4)

    let kernel = Tensor([[1, 10, 100, 1000]])
    let result = try buf.conv2d(kernel).sum().realize(frames: 8)

    // Buffer at frame f = [max(f-3,0), max(f-2,0), max(f-1,0), f] (zeros for early frames)
    // Weighted: 1*buf[0] + 10*buf[1] + 100*buf[2] + 1000*buf[3]
    let expected: [Float] = [0, 1000, 2100, 3210, 4321, 5432, 6543, 7654]
    for i in 0..<8 {
      XCTAssertEqual(result[i], expected[i], accuracy: 0.5,
        "Frame \(i): buffer element order is wrong")
    }
  }

  /// buffer composes with conv2d on a real signal (sine wave).
  func testBufferConv2d() throws {
    let sig = sin(Signal.phasor(440.0) * Signal.constant(2.0 * .pi))
    let filtered = sig.buffer(size: 128).conv2d(Tensor([[0.2, 0.2, 0.2, 0.2, 0.2]]))

    let result = try filtered.realize(frames: 200)
    let tensorSize = 124  // 128 - 5 + 1

    let lastFrame = Array(result[(199 * tensorSize)..<(200 * tensorSize)])
    let range = (lastFrame.max() ?? 0) - (lastFrame.min() ?? 0)

    XCTAssertGreaterThan(range, 0.01, "Filtered output should have non-trivial variation")
  }

  /// Gradients flow through buffer -> conv2d to the kernel.
  func testBufferGradient() throws {
    let sig = sin(Signal.phasor(440.0) * Signal.constant(2.0 * .pi))
    let kernel = Tensor.param([1, 5], data: [0.2, 0.2, 0.2, 0.2, 0.2])
    let filtered = sig.buffer(size: 32).conv2d(kernel)

    let loss = (filtered * filtered).sum()
    _ = try loss.backward(frames: 64)

    let gradData = kernel.grad?.getData() ?? []
    XCTAssertFalse(gradData.isEmpty, "Kernel should have gradients")
    XCTAssertTrue(gradData.contains(where: { abs($0) > 1e-6 }), "At least one gradient should be non-zero")
  }

  /// Regression harness for DDSP-style static FIR noise branch using:
  ///   noise.buffer(size).conv2d(kernel).sum()
  ///
  /// Regression: this used to trigger a Metal compile failure
  /// (`use of undeclared identifier`) on a later step.
  func testDDSPStyleStaticFIRNoiseBranchCompileRegression() throws {
    let previousBackend = DGenConfig.backend
    let previousDebug = DGenConfig.debug
    let previousSampleRate = DGenConfig.sampleRate
    let previousMaxFrameCount = DGenConfig.maxFrameCount
    let previousKernelOutputPath = DGenConfig.kernelOutputPath
    defer {
      DGenConfig.backend = previousBackend
      DGenConfig.debug = previousDebug
      DGenConfig.sampleRate = previousSampleRate
      DGenConfig.maxFrameCount = previousMaxFrameCount
      DGenConfig.kernelOutputPath = previousKernelOutputPath
    }

    DGenConfig.backend = .metal
    DGenConfig.debug = false
    DGenConfig.sampleRate = 16_000
    DGenConfig.maxFrameCount = 16_384
    DGenConfig.kernelOutputPath = "/tmp/ddsp_fir_compile_regression.metal"
    LazyGraphContext.reset()

    let frameCount = 16_384
    let featureFrames = 61
    let numHarmonics = 16
    let hiddenSize = 32
    let firSize = 15
    let twoPi = Float.pi * 2.0

    let harmonicIndices = Tensor((0..<numHarmonics).map { Float($0 + 1) })

    func initData(count: Int, scale: Float) -> [Float] {
      (0..<count).map { i in
        let x = Float(i)
        return (Foundation.sin(x * 0.137) + Foundation.cos(x * 0.073)) * 0.5 * scale
      }
    }

    // Minimal decoder MLP params (8 tensors, like DDSP M2 path)
    let w1 = Tensor.param([3, hiddenSize], data: initData(count: 3 * hiddenSize, scale: 0.2))
    let b1 = Tensor.param([1, hiddenSize], data: initData(count: hiddenSize, scale: 0.05))
    let wAmp = Tensor.param(
      [hiddenSize, numHarmonics], data: initData(count: hiddenSize * numHarmonics, scale: 0.15))
    let bAmp = Tensor.param([1, numHarmonics], data: initData(count: numHarmonics, scale: 0.02))
    let wHG = Tensor.param([hiddenSize, 1], data: initData(count: hiddenSize, scale: 0.1))
    let bHG = Tensor.param([1, 1], data: [0.0])
    let wNG = Tensor.param([hiddenSize, 1], data: initData(count: hiddenSize, scale: 0.1))
    let bNG = Tensor.param([1, 1], data: [0.0])
    let optimizer = Adam(params: [w1, b1, wAmp, bAmp, wHG, bHG, wNG, bNG], lr: 1e-3)

    func makeStepConditioning(step: Int) -> (features: Tensor, f0Hz: [Float], uvMask: [Float]) {
      let s = Float(step)
      var f0Hz = [Float]()
      var loudnessDB = [Float]()
      var uvMask = [Float]()
      f0Hz.reserveCapacity(featureFrames)
      loudnessDB.reserveCapacity(featureFrames)
      uvMask.reserveCapacity(featureFrames)

      for i in 0..<featureFrames {
        let t = Float(i) / Float(max(1, featureFrames - 1))
        let f0 = 90.0 + 55.0 * Foundation.sin(t * twoPi * 0.3 + s * 0.21)
        let loud = -48.0 + 10.0 * Foundation.cos(t * twoPi * 0.7 + s * 0.17)
        let uv = max(0.0, min(1.0, 0.82 + 0.18 * Foundation.sin(t * twoPi + s * 0.11)))
        f0Hz.append(f0)
        loudnessDB.append(loud)
        uvMask.append(uv)
      }

      let rows = (0..<featureFrames).map { i -> [Float] in
        let uv = min(1.0, max(0.0, uvMask[i]))
        let safeF0 = max(1.0, f0Hz[i])
        let f0Norm = log2(safeF0 / 440.0)
        let loudNorm = min(1.0, max(0.0, (loudnessDB[i] + 80.0) / 80.0))
        return [f0Norm, loudNorm, uv]
      }
      return (Tensor(rows), f0Hz, uvMask)
    }

    func makeStepTarget(step: Int) -> [Float] {
      let s = Float(step)
      return (0..<frameCount).map { i in
        let t = Float(i) / 16_000.0
        let a = Foundation.sin(twoPi * (220.0 + 4.0 * s) * t) * 0.18
        let b = Foundation.sin(twoPi * (110.0 + 2.0 * s) * t) * 0.08
        return a + b
      }
    }

    func buildLoss(step: Int) -> Signal {
      let cond = makeStepConditioning(step: step)
      let features = cond.features
      let hidden = (features.matmul(w1) + b1).tanh()
      let harmonicAmps = (hidden.matmul(wAmp) + bAmp).sigmoid()
      let harmonicGain = (hidden.matmul(wHG) + bHG).sigmoid()
      let noiseGain = (hidden.matmul(wNG) + bNG).sigmoid()

      let featureMaxIndex = Float(featureFrames - 1)
      let frameDenom = Float(max(1, frameCount - 1))
      let playheadStep = featureMaxIndex / Float(max(1, frameCount - 1))
      let playheadRaw = Signal.accum(
        Signal.constant(playheadStep), reset: 0.0, min: 0.0, max: featureMaxIndex
      )
      let playhead = playheadRaw.clip(0.0, Double(featureMaxIndex - 1e-4))

      _ = frameDenom  // keep shape/order close to DDSPSynth.renderSignal

      let f0 = min(max(Tensor(cond.f0Hz).peek(playhead), 20.0), 500.0)
      let uv = Tensor(cond.uvMask).peek(playhead).clip(0.0, 1.0)

      let ampsAtTime = harmonicAmps.peekRow(playhead)
      let harmonicFreqs = harmonicIndices * f0
      let harmonicPhases = Signal.statefulPhasor(harmonicFreqs)
      let harmonic = (sin(harmonicPhases * twoPi) * ampsAtTime).sum() * uv
      let harmonicOut =
        harmonic * harmonicGain.peek(playhead, channel: Signal.constant(0.0))
        * (1.0 / Float(numHarmonics))

      let firTap = 1.0 / Float(firSize)
      let firKernel = Tensor([Float](repeating: firTap, count: firSize)).reshape([1, firSize])
      let noise = Signal.noise()
      let filteredNoise = noise.buffer(size: firSize).conv2d(firKernel).sum()
      let noiseOut =
        filteredNoise * noiseGain.peek(playhead, channel: Signal.constant(0.0)) * (1.0 - uv)

      let prediction = harmonicOut + noiseOut
      let target = Tensor(makeStepTarget(step: step)).toSignal(maxFrames: frameCount)

      var loss = mse(prediction, target)
      for windowSize in [64, 128, 256] {
        let hop = max(1, windowSize / 4)
        loss = loss + spectralLossFFT(
          prediction,
          target,
          windowSize: windowSize,
          useHannWindow: true,
          hop: hop,
          normalize: true
        )
      }
      return loss
    }

    func sanitizeAndClip(_ params: [Tensor], clip: Float) {
      for p in params {
        guard let grad = p.grad, let data = grad.getData() else { continue }
        let cleaned = data.map { g -> Float in
          if !g.isFinite { return 0.0 }
          if g > clip { return clip }
          if g < -clip { return -clip }
          return g
        }
        grad.updateDataLazily(cleaned)
      }
    }

    var sawCompileError = false
    for step in 0..<5 {
      do {
        _ = try buildLoss(step: step).backward(frames: frameCount)
        sanitizeAndClip([w1, b1, wAmp, bAmp, wHG, bHG, wNG, bNG], clip: 1.0)
        optimizer.step()
        optimizer.zeroGrad()
      } catch {
        let message = String(describing: error).lowercased()
        if message.contains("undeclared identifier") || message.contains("mtllibraryerror") {
          sawCompileError = true
          break
        }
        throw error
      }
    }

    XCTAssertFalse(
      sawCompileError,
      "Unexpected Metal compile error in DDSP-style FIR noise branch")
  }
}
