import XCTest
@testable import DGenLazy
@testable import DGen

final class KernelAnalysisTests: XCTestCase {

  override func setUp() {
    super.setUp()
    DGenConfig.sampleRate = 44100.0
    LazyGraphContext.reset()
  }

  override func tearDown() {
    DGenConfig.sampleRate = 44100.0
    super.tearDown()
  }

  // MARK: - Tensor Analysis

  func testAnalyzeSimpleTensor() throws {
    let a = Tensor([1, 2, 3, 4])
    let b = Tensor([5, 6, 7, 8])
    let c = (a + b).sum()
    let analysis = try c.analyze()

    XCTAssertGreaterThan(analysis.kernelCount, 0)
    XCTAssertGreaterThan(analysis.work, 0)
    XCTAssertGreaterThan(analysis.memoryBytes, 0)
    XCTAssertFalse(analysis.includesBackward)
    print("Simple tensor - Kernels: \(analysis.kernelCount), Work: \(analysis.work), Span: \(analysis.span), Memory: \(analysis.memoryBytes) bytes")
  }

  func testAnalyzeBackward() throws {
    // Forward-only analysis
    let w1 = Tensor([1, 2, 3, 4], requiresGrad: true)
    let fwdLoss = (w1 * 2.0).sum()
    let fwdAnalysis = try fwdLoss.analyze(backward: false)
    XCTAssertFalse(fwdAnalysis.includesBackward)
    XCTAssertGreaterThan(fwdAnalysis.kernelCount, 0)

    // Backward analysis (need fresh graph)
    LazyGraphContext.reset()
    let w2 = Tensor([1, 2, 3, 4], requiresGrad: true)
    let bwdLoss = (w2 * 2.0).sum()
    let bwdAnalysis = try bwdLoss.analyze(backward: true)
    XCTAssertTrue(bwdAnalysis.includesBackward)
    XCTAssertGreaterThan(bwdAnalysis.kernelCount, 0)

    // Backward should have at least as many kernels or more work
    XCTAssertGreaterThanOrEqual(bwdAnalysis.work, fwdAnalysis.work)

    print("Forward - Kernels: \(fwdAnalysis.kernelCount), Work: \(fwdAnalysis.work)")
    print("Backward - Kernels: \(bwdAnalysis.kernelCount), Work: \(bwdAnalysis.work)")
  }

  func testSharedMulConsumedByTwoSumAxisBackward() throws {
    LazyGraphContext.reset()

    let x = Tensor.param([2, 3], data: [0.2, -0.4, 0.7, 1.1, -0.3, 0.5])
    let y = Tensor([[1.5, -2.0, 0.25], [-0.75, 0.5, 3.0]])

    let z = x * y
    let colSums = z.sum(axis: 0)
    let rowSums = z.sum(axis: 1)
    let loss = colSums.sum() + rowSums.sum()

    _ = try loss.backward(frameCount: 1)

    guard let grad = x.grad?.getData() else {
      XCTFail("Expected gradient data for x")
      return
    }
    let yData = y.getData() ?? []
    XCTAssertEqual(grad.count, yData.count)

    // loss = sum(z) + sum(z) => dloss/dx = 2 * y
    for i in 0..<min(grad.count, yData.count) {
      XCTAssertEqual(grad[i], 2.0 * yData[i], accuracy: 1e-4, "Mismatch at index \(i)")
    }
  }

  // MARK: - Signal Analysis

  func testAnalyzeSignal() throws {
    DGenConfig.sampleRate = 1000.0
    let freq = Signal.param(100.0)
    let phase = Signal.phasor(freq)
    let wave = sin(phase * 2 * .pi)
    let analysis = try wave.analyze(frames: 64)

    XCTAssertGreaterThan(analysis.kernelCount, 0)
    XCTAssertGreaterThan(analysis.work, 0)
    XCTAssertGreaterThan(analysis.span, 0)
    XCTAssertFalse(analysis.includesBackward)
    print("Signal - Kernels: \(analysis.kernelCount), Work: \(analysis.work), Span: \(analysis.span)")
  }

  func testDumpTensorPhasorKernel() throws {
    DGenConfig.sampleRate = 1000.0
    let previousKernelPath = DGenConfig.kernelOutputPath
    let kernelPath = "/tmp/tensor_phasor_simple.metal"
    DGenConfig.kernelOutputPath = kernelPath
    defer {
      DGenConfig.kernelOutputPath = previousKernelPath
    }

    LazyGraphContext.reset()

    let freqs = Tensor([100.0, 200.0, 300.0, 400.0])
    let phases = Signal.statefulPhasor(freqs)
    let signal = sin(phases * 2.0 * Float.pi).sum()

    _ = try signal.realize(frames: 64)

    let kernelSource = try String(contentsOfFile: kernelPath, encoding: .utf8)
    XCTAssertTrue(kernelSource.contains("kernel void"))
    // Stateful tensor phasor should dispatch one thread per tensor element
    // and loop sequentially over frames inside that thread.
    XCTAssertTrue(kernelSource.contains("// KERNEL 0\n// Kind: scalar"))
    XCTAssertTrue(kernelSource.contains("if (id >= 0 && id < (uint)(4))"))
    XCTAssertTrue(kernelSource.contains("for (uint i = 0; i < frameCount; i += 1)"))
    print("Wrote tensor-phasor kernel dump to \(kernelPath)")
  }

  func testDumpSignalTensorPhasorKernel() throws {
    DGenConfig.sampleRate = 1000.0
    let previousKernelPath = DGenConfig.kernelOutputPath
    let kernelPath = "/tmp/signal_tensor_phasor_simple.metal"
    DGenConfig.kernelOutputPath = kernelPath
    defer {
      DGenConfig.kernelOutputPath = previousKernelPath
    }

    LazyGraphContext.reset()

    let baseFreq = Signal.accum(
      Signal.constant(1.0),
      reset: 0.0,
      min: 100.0,
      max: 200.0
    )
    let harmonicIndices = Tensor([1.0, 2.0, 3.0, 4.0])
    let freqs = harmonicIndices * baseFreq
    let phases = Signal.statefulPhasor(freqs)
    let signal = sin(phases * 2.0 * Float.pi).sum()

    _ = try signal.realize(frames: 64)

    let kernelSource = try String(contentsOfFile: kernelPath, encoding: .utf8)
    XCTAssertTrue(kernelSource.contains("kernel void"))
    XCTAssertTrue(kernelSource.contains("// KERNEL 0\n// Kind: scalar"))
    XCTAssertTrue(kernelSource.contains("if (id >= 0 && id < (uint)(4))"))
    XCTAssertTrue(kernelSource.contains("for (uint i = 0; i < frameCount; i += 1)"))
    print("Wrote signal-tensor phasor kernel dump to \(kernelPath)")
  }

  func testPeekBackwardABAnalytics() throws {
    struct Metrics {
      let analysis: KernelAnalysis
      let gradL2: Float
      let gradNZ: Int
    }

    func runMode(label: String, dropPeekTensorInputGradient: Bool) throws -> Metrics {
      DGenConfig.sampleRate = 4000.0
      DGenConfig.maxFrameCount = 512
      DGenConfig.useDeterministicPeekGradients = false
      DGenConfig.dropPeekTensorInputGradient = dropPeekTensorInputGradient
      DGenConfig.kernelOutputPath = "/tmp/peek_ab_\(label).metal"

      let frames = 256
      let controlFrames = 32
      let harmonics = 16

      // ---------- Compile-only analysis pass ----------
      LazyGraphContext.reset()
      let timeRows = (0..<controlFrames).map { [Float($0) / Float(max(1, controlFrames - 1))] }
      let timeTensor = Tensor(timeRows)
      let wA = Tensor.param(
        [1, harmonics],
        data: (0..<harmonics).map { i in
          let x = Float(i) / Float(max(1, harmonics - 1))
          return 0.1 * Foundation.sin(x * 3.0 * Float.pi)
        }
      )
      let bA = Tensor.param([1, harmonics], data: [Float](repeating: 0.0, count: harmonics))
      let ampsA = sigmoid(timeTensor.matmul(wA) + bA)
      let playheadA = Signal.phasor(DGenConfig.sampleRate / Float(frames)) * Float(controlFrames - 1)
      var predA = Signal.constant(0.0)
      for h in 0..<harmonics {
        let amp = ampsA.peek(playheadA, channel: Signal.constant(Float(h)))
        let osc = sin(Signal.phasor(Float(80 + h * 20)) * 2.0 * Float.pi)
        predA = predA + amp * osc
      }
      let lossA = mse(predA, Signal.constant(0.0))
      let analysis = try lossA.analyze(backward: true, frames: frames)

      // ---------- Runtime backward pass for gradient magnitude ----------
      LazyGraphContext.reset()
      let timeTensorB = Tensor(timeRows)
      let wB = Tensor.param(
        [1, harmonics],
        data: (0..<harmonics).map { i in
          let x = Float(i) / Float(max(1, harmonics - 1))
          return 0.1 * Foundation.sin(x * 3.0 * Float.pi)
        }
      )
      let bB = Tensor.param([1, harmonics], data: [Float](repeating: 0.0, count: harmonics))
      let ampsB = sigmoid(timeTensorB.matmul(wB) + bB)
      let playheadB = Signal.phasor(DGenConfig.sampleRate / Float(frames)) * Float(controlFrames - 1)
      var predB = Signal.constant(0.0)
      for h in 0..<harmonics {
        let amp = ampsB.peek(playheadB, channel: Signal.constant(Float(h)))
        let osc = sin(Signal.phasor(Float(80 + h * 20)) * 2.0 * Float.pi)
        predB = predB + amp * osc
      }
      let lossB = mse(predB, Signal.constant(0.0))
      _ = try lossB.backward(frames: frames)

      let gradData = wB.grad?.getData() ?? []
      let gradL2 = Foundation.sqrt(gradData.reduce(0.0) { $0 + $1 * $1 })
      let gradNZ = gradData.filter { $0 != 0 && $0.isFinite }.count

      return Metrics(analysis: analysis, gradL2: gradL2, gradNZ: gradNZ)
    }

    let legacy = try runMode(label: "legacy_drop", dropPeekTensorInputGradient: true)
    let fixed = try runMode(label: "fixed_upstream", dropPeekTensorInputGradient: false)

    let legacyScalar = legacy.analysis.kernels.filter { $0.kind == .scalar }
    let fixedScalar = fixed.analysis.kernels.filter { $0.kind == .scalar }
    let legacyScalarWork = legacyScalar.reduce(0) { $0 + $1.work }
    let fixedScalarWork = fixedScalar.reduce(0) { $0 + $1.work }

    print("=== Peek Backward A/B Analytics ===")
    print(
      "legacy(drop=true): kernels=\(legacy.analysis.kernelCount) work=\(legacy.analysis.work) span=\(legacy.analysis.span) "
        + "scalarKernels=\(legacyScalar.count) scalarWork=\(legacyScalarWork) "
        + "gradL2=\(legacy.gradL2) gradNZ=\(legacy.gradNZ)"
    )
    print(
      "fixed(drop=false): kernels=\(fixed.analysis.kernelCount) work=\(fixed.analysis.work) span=\(fixed.analysis.span) "
        + "scalarKernels=\(fixedScalar.count) scalarWork=\(fixedScalarWork) "
        + "gradL2=\(fixed.gradL2) gradNZ=\(fixed.gradNZ)"
    )
    print("legacy kernels: /tmp/peek_ab_legacy_drop.metal")
    print("fixed kernels: /tmp/peek_ab_fixed_upstream.metal")

    XCTAssertGreaterThan(legacy.analysis.kernelCount, 0)
    XCTAssertGreaterThan(fixed.analysis.kernelCount, 0)
  }

  func testPeekRowTensorPhasorKernelAnalytics() throws {
    struct Metrics {
      let analysis: KernelAnalysis
      let gradL2: Float
      let gradNZ: Int
      let kernelPath: String
    }

    let frames = 256
    let controlFrames = 32
    let harmonics = 16
    let twoPi = Float.pi * 2.0
    let harmonicFreqs = (0..<harmonics).map { Float(80 + $0 * 20) }
    let harmonicTensor: DGenLazy.Tensor = Tensor(harmonicFreqs)

    func runVariant(label: String, useTensorPhasor: Bool) throws -> Metrics {
      func buildLoss(
        w: DGenLazy.Tensor,
        b: DGenLazy.Tensor
      ) -> Signal {
        let timeRows = (0..<controlFrames).map { [Float($0) / Float(max(1, controlFrames - 1))] }
        let timeTensor = Tensor(timeRows)  // [controlFrames, 1]
        let amps = sigmoid(timeTensor.matmul(w) + b)  // [controlFrames, harmonics]
        let playhead =
          Signal.phasor(DGenConfig.sampleRate / Float(frames)) * Float(controlFrames - 1)

        let pred: Signal
        if useTensorPhasor {
          let ampsAtTime = amps.peekRow(playhead)  // [harmonics]
          let phases = Signal.statefulPhasor(harmonicTensor)  // [harmonics]
          let sines = sin(phases * twoPi)  // [harmonics]
          pred = (ampsAtTime * sines).sum() * (1.0 / Float(harmonics))
        } else {
          var out = Signal.constant(0.0)
          for h in 0..<harmonics {
            let amp = amps.peek(playhead, channel: Signal.constant(Float(h)))
            let osc = sin(Signal.phasor(harmonicFreqs[h]) * twoPi)
            out = out + amp * osc
          }
          pred = out * (1.0 / Float(harmonics))
        }
        return mse(pred, Signal.constant(0.0))
      }

      DGenConfig.sampleRate = 4000.0
      DGenConfig.maxFrameCount = 512
      DGenConfig.useDeterministicPeekGradients = false
      DGenConfig.dropPeekTensorInputGradient = false

      let kernelPath = "/tmp/peek_tensor_phasor_\(label).metal"
      DGenConfig.kernelOutputPath = kernelPath

      // Compile-only analysis pass
      LazyGraphContext.reset()
      let wA = Tensor.param(
        [1, harmonics],
        data: (0..<harmonics).map { i in
          let x = Float(i) / Float(max(1, harmonics - 1))
          return 0.1 * Foundation.sin(x * 3.0 * Float.pi)
        }
      )
      let bA = Tensor.param([1, harmonics], data: [Float](repeating: 0.0, count: harmonics))
      let lossA = buildLoss(w: wA, b: bA)
      let analysis = try lossA.analyze(backward: true, frames: frames)

      // Runtime backward pass for grad magnitude
      LazyGraphContext.reset()
      let wB = Tensor.param(
        [1, harmonics],
        data: (0..<harmonics).map { i in
          let x = Float(i) / Float(max(1, harmonics - 1))
          return 0.1 * Foundation.sin(x * 3.0 * Float.pi)
        }
      )
      let bB = Tensor.param([1, harmonics], data: [Float](repeating: 0.0, count: harmonics))
      let lossB = buildLoss(w: wB, b: bB)
      _ = try lossB.backward(frames: frames)

      let gradData = wB.grad?.getData() ?? []
      let gradL2 = Foundation.sqrt(gradData.reduce(0.0) { $0 + $1 * $1 })
      let gradNZ = gradData.filter { $0 != 0 && $0.isFinite }.count

      return Metrics(
        analysis: analysis,
        gradL2: gradL2,
        gradNZ: gradNZ,
        kernelPath: kernelPath
      )
    }

    let scalarLoop = try runVariant(label: "scalar_loop", useTensorPhasor: false)
    let tensorPhasor = try runVariant(label: "peekrow_tensor_phasor", useTensorPhasor: true)

    let scalarLoopScalar = scalarLoop.analysis.kernels.filter { $0.kind == .scalar }
    let tensorPhasorScalar = tensorPhasor.analysis.kernels.filter { $0.kind == .scalar }
    let scalarLoopScalarWork = scalarLoopScalar.reduce(0) { $0 + $1.work }
    let tensorPhasorScalarWork = tensorPhasorScalar.reduce(0) { $0 + $1.work }

    print("=== PeekRow + Tensor Phasor Kernel Analytics ===")
    print(
      "scalar-loop: kernels=\(scalarLoop.analysis.kernelCount) work=\(scalarLoop.analysis.work) span=\(scalarLoop.analysis.span) "
        + "scalarKernels=\(scalarLoopScalar.count) scalarWork=\(scalarLoopScalarWork) "
        + "gradL2=\(scalarLoop.gradL2) gradNZ=\(scalarLoop.gradNZ)"
    )
    print(
      "peekRow+tensorPhasor: kernels=\(tensorPhasor.analysis.kernelCount) work=\(tensorPhasor.analysis.work) span=\(tensorPhasor.analysis.span) "
        + "scalarKernels=\(tensorPhasorScalar.count) scalarWork=\(tensorPhasorScalarWork) "
        + "gradL2=\(tensorPhasor.gradL2) gradNZ=\(tensorPhasor.gradNZ)"
    )
    print("scalar-loop kernels: \(scalarLoop.kernelPath)")
    print("peekRow+tensorPhasor kernels: \(tensorPhasor.kernelPath)")

    XCTAssertGreaterThan(scalarLoop.analysis.kernelCount, 0)
    XCTAssertGreaterThan(tensorPhasor.analysis.kernelCount, 0)
    XCTAssertGreaterThan(scalarLoop.gradNZ, 0)
    XCTAssertGreaterThan(tensorPhasor.gradNZ, 0)
  }

  // MARK: - Per-Kernel Breakdown

  func testKernelBreakdown() throws {
    let a = Tensor([1, 2, 3, 4])
    let b = Tensor([5, 6, 7, 8])
    let c = (a + b).sum()
    let analysis = try c.analyze()

    for kernel in analysis.kernels {
      print("\(kernel.name): kind=\(kernel.kind), work=\(kernel.work), span=\(kernel.span), "
            + "arith=\(kernel.arithmeticOps), trans=\(kernel.transcendentalOps), mem=\(kernel.memoryOps)")
    }

    // At least one kernel should have arithmetic ops
    let hasArith = analysis.kernels.contains { $0.arithmeticOps > 0 }
    XCTAssertTrue(hasArith)
  }

  // MARK: - Memory Consistency

  func testMemorySlots() throws {
    let a = Tensor([1, 2, 3, 4])
    let b = Tensor([5, 6, 7, 8])
    let c = (a + b).sum()
    let analysis = try c.analyze()

    XCTAssertEqual(analysis.memoryBytes, analysis.memorySlots * 4)
    XCTAssertGreaterThan(analysis.memorySlots, 0)
  }

  // MARK: - Graph Cleanup

  func testAnalyzeDoesNotPolluteGraph() throws {
    // analyze() should call clearComputationGraph(), so a second
    // analyze on a fresh graph should also work
    let a = Tensor([1, 2, 3])
    let b = a * 2
    let _ = try b.analyze()

    // Build a new graph on the same context — should not crash
    LazyGraphContext.reset()
    let c = Tensor([4, 5, 6])
    let d = c + 1
    let analysis = try d.analyze()
    XCTAssertGreaterThan(analysis.kernelCount, 0)
  }
}
