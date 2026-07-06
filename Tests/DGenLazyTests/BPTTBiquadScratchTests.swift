import XCTest

import DGen

@testable import DGenLazy

/// Scratch: dump kernels for the trainable-biquad gradient corruption repro.
final class BPTTBiquadScratchTests: XCTestCase {
  /// FD vs autograd under LINEAR loss (FD-stable) with trainable biquad cutoff.
  /// outGain should match; cutoff is expected to be a truncated gradient
  /// (temporal recursion terms missing while biquad historyWrites are dangling).
  func testLinearLossFDComparison() throws {
    let frameCount = 2048
    let sampleRate: Float = 44100.0
    let twoPi = Float.pi * 2.0

    func build(cutoffZ: Float, outGainV: Float) -> (loss: Signal, cutoff: Signal, outGain: Signal)
    {
      LazyGraphContext.reset()
      DGenConfig.backend = .metal
      DGenConfig.sampleRate = sampleRate
      DGenConfig.maxFrameCount = frameCount
      let outGain = Signal.param(outGainV)
      let cutoff = Signal.param(cutoffZ)
      let t = Signal.accum(
        Signal.constant(1.0 / sampleRate), reset: 0.0, min: 0.0,
        max: Float(frameCount + 1) / sampleRate)
      let body =
        sin(Signal.statefulPhasor(Signal.constant(120.0)) * twoPi)
        * DGenLazy.exp(Signal.constant(-7.0) * t) * 0.75
      let noise = Signal.noise().biquad(
        cutoff: DGenLazy.exp(cutoff),
        resonance: Signal.constant(0.707),
        gain: Signal.constant(1.0),
        mode: Signal.constant(0.0))
      let noiseBurst = noise * DGenLazy.exp(Signal.constant(-140.0) * t) * 0.08
      let student = DGenLazy.tanh((body + noiseBurst) * 2.0) * outGain
      let teacher = sin(Signal.phasor(Signal.constant(97.0)) * twoPi) * 0.6
      let lossSig = spectralLossFFT(
        student, teacher, windowSize: 256, lossMode: .l2, hop: 64, normalize: true)
      return (lossSig, cutoff, outGain)
    }

    func lossSum(cutoffZ: Float, outGainV: Float) throws -> Float {
      let built = build(cutoffZ: cutoffZ, outGainV: outGainV)
      return try built.loss.backward(frames: frameCount).reduce(0, +)
    }

    let z0 = Foundation.log(Float(2800.0))
    let built = build(cutoffZ: z0, outGainV: 0.7)
    _ = try built.loss.backward(frames: frameCount)
    let autoOut = built.outGain.grad?.data ?? .nan
    let autoCut = built.cutoff.grad?.data ?? .nan

    for eps in [Float(3e-3), 1e-2] {
      let fdOut =
        (try lossSum(cutoffZ: z0, outGainV: 0.7 + eps)
          - lossSum(cutoffZ: z0, outGainV: 0.7 - eps)) / (2 * eps)
      let fdCut =
        (try lossSum(cutoffZ: z0 + eps, outGainV: 0.7)
          - lossSum(cutoffZ: z0 - eps, outGainV: 0.7)) / (2 * eps)
      print("[linfd] eps=\(eps) outGain fd=\(fdOut) auto=\(autoOut) | cutoff fd=\(fdCut) auto=\(autoCut)")
    }
  }

  /// MSE (time-domain) loss keeps forward+backward in one scalar block, so
  /// wrapWithBPTTLoops can activate. Validates the pass-through biquad macro
  /// produces full temporal gradients for cutoff.
  func testMSELossFDComparison() throws {
    let frameCount = 512
    let sampleRate: Float = 44100.0

    func build(cutoffZ: Float) -> (loss: Signal, cutoff: Signal) {
      LazyGraphContext.reset()
      DGenConfig.backend = .metal
      DGenConfig.sampleRate = sampleRate
      DGenConfig.maxFrameCount = frameCount
      let cutoff = Signal.param(cutoffZ)
      let src = sin(Signal.statefulPhasor(Signal.constant(3000.0)) * Float.pi * 2.0)
      let filtered = src.biquad(
        cutoff: DGenLazy.exp(cutoff),
        resonance: Signal.constant(0.707),
        gain: Signal.constant(1.0),
        mode: Signal.constant(0.0))
      let target = sin(Signal.statefulPhasor(Signal.constant(3000.0)) * Float.pi * 2.0) * 0.5
      return (mse(filtered, target), cutoff)
    }

    func lossSum(cutoffZ: Float) throws -> Float {
      let built = build(cutoffZ: cutoffZ)
      return try built.loss.backward(frames: frameCount).reduce(0, +)
    }

    let z0 = Foundation.log(Float(2000.0))
    let built = build(cutoffZ: z0)
    _ = try built.loss.backward(frames: frameCount)
    let autoCut = built.cutoff.grad?.data ?? .nan

    let l1 = try lossSum(cutoffZ: z0)
    let l2 = try lossSum(cutoffZ: z0)
    print("[msefd] determinism check: loss(z0) run1=\(l1) run2=\(l2)")

    var fdAt3em2: Float = .nan
    for eps in [Float(1e-3), 3e-3, 1e-2, 3e-2] {
      let lp = try lossSum(cutoffZ: z0 + eps)
      let lm = try lossSum(cutoffZ: z0 - eps)
      let fdCut = (lp - lm) / (2 * eps)
      if eps == 3e-2 { fdAt3em2 = fdCut }
      print("[msefd] eps=\(eps) l+=\(lp) l-=\(lm) fd=\(fdCut) auto=\(autoCut)")
    }
    XCTAssertEqual(autoCut, fdAt3em2, accuracy: max(abs(fdAt3em2) * 0.1, 1e-4))
  }

  /// Minimal two-cell chain-write repro of the biquad y-recursion:
  ///   y[n] = x[n] - a1*y[n-1] - a2*y[n-2]
  /// with y[n-2] maintained by a chained pass-through write (biquad pattern):
  ///   y1chained = writeC0(readC1); y = ... ; out = writeC1(y)
  /// a1 is the trainable param directly; CPU adjoint reference inline.
  func testTwoCellChainWriteGradient() throws {
    let frameCount = 256
    let sampleRate: Float = 44100.0
    let a1v: Float = 0.5
    let a2v: Float = 0.2

    func build(a1val: Float) -> (loss: Signal, a1: Signal) {
      LazyGraphContext.reset()
      DGenConfig.backend = .metal
      DGenConfig.sampleRate = sampleRate
      DGenConfig.maxFrameCount = frameCount
      let a1 = Signal.param(a1val)
      let src = sin(Signal.statefulPhasor(Signal.constant(3000.0)) * Float.pi * 2.0)
      let h1 = Signal.history()  // y[n-1] cell
      let h0 = Signal.history()  // y[n-2] cell
      let y1chained = h0.write(h1.read)  // pass-through = y[n-1]
      let y = src - a1 * y1chained - Signal.constant(a2v) * h0.read
      let out = h1.write(y)
      let target = sin(Signal.statefulPhasor(Signal.constant(3000.0)) * Float.pi * 2.0) * 0.3
      return (mse(out, target), a1)
    }

    let built = build(a1val: a1v)
    _ = try built.loss.backward(frames: frameCount)
    let auto = built.a1.grad?.data ?? .nan

    // CPU reference (double)
    let fs = 44100.0
    func sig(_ f: Double) -> [Double] {
      var out = [Double](repeating: 0, count: frameCount)
      var ph = 0.0
      for n in 0..<frameCount {
        ph += f / fs
        if ph >= 1 { ph -= 1 }
        out[n] = sin(2 * Double.pi * ph)
      }
      return out
    }
    let x = sig(3000)
    let tgt = sig(3000).map { $0 * 0.3 }
    func fwd(_ a1d: Double) -> (Double, [Double]) {
      var y1 = 0.0, y2 = 0.0, loss = 0.0
      var ys = [Double](repeating: 0, count: frameCount)
      for n in 0..<frameCount {
        let y = x[n] - a1d * y1 - Double(a2v) * y2
        ys[n] = y
        loss += (y - tgt[n]) * (y - tgt[n])
        y2 = y1
        y1 = y
      }
      return (loss, ys)
    }
    let eps = 1e-6
    let fd = (fwd(Double(a1v) + eps).0 - fwd(Double(a1v) - eps).0) / (2 * eps)
    let (_, ys) = fwd(Double(a1v))
    var delta = [Double](repeating: 0, count: frameCount + 2)
    var dA1 = 0.0
    for n in stride(from: frameCount - 1, through: 0, by: -1) {
      delta[n] =
        2 * (ys[n] - tgt[n]) - Double(a1v) * delta[n + 1] - Double(a2v) * delta[n + 2]
      dA1 -= delta[n] * (n >= 1 ? ys[n - 1] : 0.0)
    }
    print("[chain2] cpuFD=\(fd) cpuAdj=\(dA1) gpuAuto=\(auto)")
    XCTAssertEqual(auto, Float(fd), accuracy: Float(max(abs(fd) * 0.02, 1e-4)))
  }

  /// Minimal x-side FIR chain repro (biquad pattern, no feedback):
  ///   out = p*x[n] + p*x[n-1] + p*x[n-2]
  /// with x[n-1]/x[n-2] maintained by chained pass-through writes.
  func testFIRChainWriteGradient() throws {
    let frameCount = 256
    let sampleRate: Float = 44100.0
    let pv: Float = 0.4

    func build(pval: Float) -> (loss: Signal, p: Signal) {
      LazyGraphContext.reset()
      DGenConfig.backend = .metal
      DGenConfig.sampleRate = sampleRate
      DGenConfig.maxFrameCount = frameCount
      let p = Signal.param(pval)
      let src = sin(Signal.statefulPhasor(Signal.constant(3000.0)) * Float.pi * 2.0)
      let h2 = Signal.history()  // x[n-1] cell
      let h3 = Signal.history()  // x[n-2] cell
      let x0 = h2.write(src)  // pass-through = x[n]
      let x1 = h3.write(h2.read)  // pass-through = x[n-1]
      let x2 = h3.read  // x[n-2]
      let out = p * x0 + p * x1 + p * x2
      let target = sin(Signal.statefulPhasor(Signal.constant(3000.0)) * Float.pi * 2.0) * 0.3
      return (mse(out, target), p)
    }

    let built = build(pval: pv)
    _ = try built.loss.backward(frames: frameCount)
    let auto = built.p.grad?.data ?? .nan

    // CPU reference (double)
    let fs = 44100.0
    func sig(_ f: Double) -> [Double] {
      var out = [Double](repeating: 0, count: frameCount)
      var ph = 0.0
      for n in 0..<frameCount {
        ph += f / fs
        if ph >= 1 { ph -= 1 }
        out[n] = sin(2 * Double.pi * ph)
      }
      return out
    }
    let x = sig(3000)
    let tgt = sig(3000).map { $0 * 0.3 }
    func fwd(_ pd: Double) -> Double {
      var x1 = 0.0, x2 = 0.0, loss = 0.0
      for n in 0..<frameCount {
        let y = pd * x[n] + pd * x1 + pd * x2
        loss += (y - tgt[n]) * (y - tgt[n])
        x2 = x1
        x1 = x[n]
      }
      return loss
    }
    let eps = 1e-6
    let fd = (fwd(Double(pv) + eps) - fwd(Double(pv) - eps)) / (2 * eps)
    print("[firchain] cpuFD=\(fd) gpuAuto=\(auto)")
    XCTAssertEqual(auto, Float(fd), accuracy: Float(max(abs(fd) * 0.02, 1e-4)))
  }

  /// Full biquad topology (4 cells, chained writes, feedback + FIR side) with
  /// simple linear coefficients: b0=b1=b2=p, a1=0.5p, a2=0.2p.
  func testFullTopologySimpleCoeffsGradient() throws {
    let frameCount = 256
    let sampleRate: Float = 44100.0
    let pv: Float = 0.4

    func build(pval: Float) -> (loss: Signal, p: Signal) {
      LazyGraphContext.reset()
      DGenConfig.backend = .metal
      DGenConfig.sampleRate = sampleRate
      DGenConfig.maxFrameCount = frameCount
      let p = Signal.param(pval)
      let src = sin(Signal.statefulPhasor(Signal.constant(3000.0)) * Float.pi * 2.0)
      let h1 = Signal.history()  // y[n-1]
      let h0 = Signal.history()  // y[n-2]
      let h2 = Signal.history()  // x[n-1]
      let h3 = Signal.history()  // x[n-2]
      let y1c = h0.write(h1.read)
      let x0 = h2.write(src)
      let x1 = h3.write(h2.read)
      let fir = p * x0 + p * x1 + p * h3.read
      let y = fir - (p * 0.5) * y1c - (p * 0.2) * h0.read
      let out = h1.write(y)
      let target = sin(Signal.statefulPhasor(Signal.constant(3000.0)) * Float.pi * 2.0) * 0.3
      return (mse(out, target), p)
    }

    let built = build(pval: pv)
    _ = try built.loss.backward(frames: frameCount)
    let auto = built.p.grad?.data ?? .nan

    let fs = 44100.0
    func sig(_ f: Double) -> [Double] {
      var out = [Double](repeating: 0, count: frameCount)
      var ph = 0.0
      for n in 0..<frameCount {
        ph += f / fs
        if ph >= 1 { ph -= 1 }
        out[n] = sin(2 * Double.pi * ph)
      }
      return out
    }
    let x = sig(3000)
    let tgt = sig(3000).map { $0 * 0.3 }
    func fwd(_ pd: Double) -> Double {
      var y1 = 0.0, y2 = 0.0, x1 = 0.0, x2 = 0.0, loss = 0.0
      for n in 0..<frameCount {
        let y = pd * x[n] + pd * x1 + pd * x2 - 0.5 * pd * y1 - 0.2 * pd * y2
        loss += (y - tgt[n]) * (y - tgt[n])
        y2 = y1
        y1 = y
        x2 = x1
        x1 = x[n]
      }
      return loss
    }
    let eps = 1e-6
    let fd = (fwd(Double(pv) + eps) - fwd(Double(pv) - eps)) / (2 * eps)
    print("[fulltopo] cpuFD=\(fd) gpuAuto=\(auto)")
    XCTAssertEqual(auto, Float(fd), accuracy: Float(max(abs(fd) * 0.02, 1e-4)))
  }

  /// Full topology + real RBJ lowpass coefficient formulas built from Signal
  /// ops (cos/sin/div), but no selectors/gswitch/abs. Isolates whether the
  /// formula chain (shared cos/sin subexpressions, div backward) breaks under BPTT.
  func testFullTopologyRBJFormulasGradient() throws {
    let frameCount = 256
    let sampleRate: Float = 44100.0
    let q: Float = 0.707

    func build(z: Float) -> (loss: Signal, p: Signal) {
      LazyGraphContext.reset()
      DGenConfig.backend = .metal
      DGenConfig.sampleRate = sampleRate
      DGenConfig.maxFrameCount = frameCount
      let p = Signal.param(z)
      let w0 = abs(DGenLazy.exp(p)) * (2.0 * Float.pi / sampleRate)
      let cosw = cos(w0)
      let sinw = sin(w0)
      let alpha = sinw * 0.5 / q
      let norm = 1.0 / (1.0 + alpha)
      // selector + gswitch layers mimicking the macro's mode dispatch
      // (selector cond = mode+1 = 1 -> first branch)
      let graph = p.graph
      func sel(_ branches: Signal...) -> Signal {
        let cond = graph.node(.constant(1.0), [])
        let nodeId = graph.graph.n(.selector, [cond] + branches.map { $0.nodeId })
        return Signal(nodeId: nodeId, graph: graph, requiresGrad: true)
      }
      let isShelf = Signal.constant(0.0)
      let b0 = gswitch(isShelf, cosw * 0.0, sel((1.0 - cosw) * 0.5, sinw, cosw) * norm)
      let b1 = gswitch(isShelf, cosw * 0.0, sel(1.0 - cosw, sinw, cosw) * norm)
      let b2 = gswitch(isShelf, cosw * 0.0, sel((1.0 - cosw) * 0.5, sinw, cosw) * norm)
      let a1 = gswitch(isShelf, cosw * 0.0, sel(-2.0 * cosw, sinw, cosw) * norm)
      let a2 = gswitch(isShelf, cosw * 0.0, sel(1.0 - alpha, sinw, cosw) * norm)
      let src = sin(Signal.statefulPhasor(Signal.constant(3000.0)) * Float.pi * 2.0)
      let h1 = Signal.history()
      let h0 = Signal.history()
      let h2 = Signal.history()
      let h3 = Signal.history()
      let y1c = h0.write(h1.read)
      let x0 = h2.write(src)
      let x1 = h3.write(h2.read)
      let fir = b0 * x0 + b1 * x1 + b2 * h3.read
      let y = fir - a1 * y1c - a2 * h0.read
      let out = h1.write(y)
      let target = sin(Signal.statefulPhasor(Signal.constant(3000.0)) * Float.pi * 2.0) * 0.5
      return (mse(out, target), p)
    }

    func lossSum(z: Float) throws -> Float {
      try build(z: z).loss.backward(frames: frameCount).reduce(0, +)
    }

    let z0 = Foundation.log(Float(2000.0))
    let built = build(z: z0)
    _ = try built.loss.backward(frames: frameCount)
    let auto = built.p.grad?.data ?? .nan
    var fdBest: Float = .nan
    for eps in [Float(1e-2), 3e-2] {
      let fd = (try lossSum(z: z0 + eps) - lossSum(z: z0 - eps)) / (2 * eps)
      fdBest = fd
      print("[rbjtopo] eps=\(eps) fd=\(fd) auto=\(auto)")
    }
    XCTAssertEqual(auto, fdBest, accuracy: max(abs(fdBest) * 0.1, 1e-4))
  }

  /// Coefficient chain in isolation (no history/BPTT): student = src * b0(z)
  /// with b0 = (1 - cos(exp(z)*2pi/fs)) / 2. Checks the cos/exp backward chain
  /// sign without any temporal recursion.
  func testCoefficientChainFD() throws {
    let frameCount = 512
    let sampleRate: Float = 44100.0

    func build(z: Float) -> (loss: Signal, p: Signal) {
      LazyGraphContext.reset()
      DGenConfig.backend = .metal
      DGenConfig.sampleRate = sampleRate
      DGenConfig.maxFrameCount = frameCount
      let p = Signal.param(z)
      let w0 = DGenLazy.exp(p) * (2.0 * Float.pi / sampleRate)
      let b0 = (1.0 - cos(w0)) * 0.5
      let src = sin(Signal.statefulPhasor(Signal.constant(3000.0)) * Float.pi * 2.0)
      let target = sin(Signal.statefulPhasor(Signal.constant(3000.0)) * Float.pi * 2.0) * 0.5
      return (mse(src * b0, target), p)
    }

    func lossSum(z: Float) throws -> Float {
      try build(z: z).loss.backward(frames: frameCount).reduce(0, +)
    }

    let z0 = Foundation.log(Float(2000.0))
    let built = build(z: z0)
    _ = try built.loss.backward(frames: frameCount)
    let auto = built.p.grad?.data ?? .nan
    let eps: Float = 1e-2
    let fd = (try lossSum(z: z0 + eps) - lossSum(z: z0 - eps)) / (2 * eps)
    print("[coefffd] fd=\(fd) auto=\(auto)")
    XCTAssertEqual(auto, fd, accuracy: max(abs(fd) * 0.05, 1e-4))
  }

  /// Double-precision CPU reference for the MSE test above: RBJ lowpass biquad,
  /// same src/target, FD vs hand-written adjoint (BPTT) gradient for d loss/d z
  /// where z = ln(cutoffHz). Establishes ground-truth sign/magnitude.
  func testCPUReferenceGradient() throws {
    let frameCount = 512
    let fs = 44100.0
    let q = 0.707

    // statefulPhasor: reads state, then advances. sin(2π·phase).
    func makeSignal(freq: Double, count: Int) -> [Double] {
      var out = [Double](repeating: 0, count: count)
      var phase = 0.0
      for n in 0..<count {
        let inc = freq / fs
        phase += inc
        if phase >= 1.0 { phase -= 1.0 }
        out[n] = sin(2.0 * Double.pi * phase)
      }
      return out
    }

    let x = makeSignal(freq: 3000.0, count: frameCount)
    let target = makeSignal(freq: 3000.0, count: frameCount).map { $0 * 0.5 }

    struct Coeffs { var b0 = 0.0, b1 = 0.0, b2 = 0.0, a1 = 0.0, a2 = 0.0 }
    func coeffs(z: Double) -> Coeffs {
      let fc = exp(z)
      let w0 = fc * (2.0 * Double.pi / fs)
      let cosw = cos(w0), sinw = sin(w0)
      let alpha = sinw * 0.5 / q
      let norm = 1.0 / (1.0 + alpha)
      var c = Coeffs()
      c.b0 = (1.0 - cosw) / 2.0 * norm
      c.b1 = (1.0 - cosw) * norm
      c.b2 = (1.0 - cosw) / 2.0 * norm
      c.a1 = -2.0 * cosw * norm
      c.a2 = (1.0 - alpha) * norm
      return c
    }

    func forward(z: Double) -> (loss: Double, ys: [Double]) {
      let c = coeffs(z: z)
      var y1 = 0.0, y2 = 0.0, x1 = 0.0, x2 = 0.0
      var loss = 0.0
      var ys = [Double](repeating: 0, count: frameCount)
      for n in 0..<frameCount {
        let y = c.b0 * x[n] + c.b1 * x1 + c.b2 * x2 - c.a1 * y1 - c.a2 * y2
        ys[n] = y
        let e = y - target[n]
        loss += e * e
        y2 = y1
        y1 = y
        x2 = x1
        x1 = x[n]
      }
      return (loss, ys)
    }

    let z0 = Foundation.log(2000.0)
    let (loss0, ys) = forward(z: z0)

    // FD ground truth
    let eps = 1e-6
    let fdZ = (forward(z: z0 + eps).loss - forward(z: z0 - eps).loss) / (2 * eps)

    // Adjoint: delta[n] = 2(y[n]-t[n]) - a1*delta[n+1] - a2*delta[n+2]
    let c = coeffs(z: z0)
    var delta = [Double](repeating: 0, count: frameCount + 2)
    for n in stride(from: frameCount - 1, through: 0, by: -1) {
      delta[n] = 2.0 * (ys[n] - target[n]) - c.a1 * delta[n + 1] - c.a2 * delta[n + 2]
    }
    var dB0 = 0.0, dB1 = 0.0, dB2 = 0.0, dA1 = 0.0, dA2 = 0.0
    for n in 0..<frameCount {
      let x1 = n >= 1 ? x[n - 1] : 0.0
      let x2 = n >= 2 ? x[n - 2] : 0.0
      let y1 = n >= 1 ? ys[n - 1] : 0.0
      let y2 = n >= 2 ? ys[n - 2] : 0.0
      dB0 += delta[n] * x[n]
      dB1 += delta[n] * x1
      dB2 += delta[n] * x2
      dA1 -= delta[n] * y1
      dA2 -= delta[n] * y2
    }
    // Chain rule to z via FD on the coefficient formulas (smooth scalars)
    let cp = coeffs(z: z0 + eps), cm = coeffs(z: z0 - eps)
    let db0dz = (cp.b0 - cm.b0) / (2 * eps)
    let db1dz = (cp.b1 - cm.b1) / (2 * eps)
    let db2dz = (cp.b2 - cm.b2) / (2 * eps)
    let da1dz = (cp.a1 - cm.a1) / (2 * eps)
    let da2dz = (cp.a2 - cm.a2) / (2 * eps)
    let adjZ = dB0 * db0dz + dB1 * db1dz + dB2 * db2dz + dA1 * da1dz + dA2 * da2dz

    print("[cpuref] loss=\(loss0) fdZ=\(fdZ) adjZ=\(adjZ)")
    print("[cpuref] contributions: b0=\(dB0 * db0dz) b1=\(dB1 * db1dz) b2=\(dB2 * db2dz) a1=\(dA1 * da1dz) a2=\(dA2 * da2dz)")
    XCTAssertEqual(adjZ, fdZ, accuracy: abs(fdZ) * 1e-4)
  }

  func testDumpKernels() throws {
    let frameCount = 2048
    let sampleRate: Float = 44100.0
    let twoPi = Float.pi * 2.0

    for trainableCutoff in [false, true] {
      LazyGraphContext.reset()
      DGenConfig.backend = .metal
      DGenConfig.sampleRate = sampleRate
      DGenConfig.maxFrameCount = frameCount
      DGenConfig.kernelOutputPath =
        trainableCutoff ? "/tmp/bptt_biquad_trainable.metal" : "/tmp/bptt_biquad_constant.metal"

      let outGain = Signal.param(0.7, min: 0.4, max: 1.0)
      let cutoff: Signal =
        trainableCutoff
        ? Signal.param(Foundation.log(Float(2800.0)))
        : Signal.constant(Foundation.log(Float(2800.0)))
      let t = Signal.accum(
        Signal.constant(1.0 / sampleRate), reset: 0.0, min: 0.0,
        max: Float(frameCount + 1) / sampleRate)
      let body =
        sin(Signal.statefulPhasor(Signal.constant(120.0)) * twoPi)
        * DGenLazy.exp(Signal.constant(-7.0) * t) * 0.75
      let noise = Signal.noise().biquad(
        cutoff: DGenLazy.exp(cutoff),
        resonance: Signal.constant(0.707),
        gain: Signal.constant(1.0),
        mode: Signal.constant(0.0))
      let noiseBurst = noise * DGenLazy.exp(Signal.constant(-140.0) * t) * 0.08
      let student = DGenLazy.tanh((body + noiseBurst) * 2.0) * outGain
      let teacher = sin(Signal.phasor(Signal.constant(97.0)) * twoPi) * 0.6
      let lossSig = spectralLossFFT(
        student, teacher, windowSize: 256,
        useLogMagnitude: true, lossMode: .l1, hop: 64, normalize: true)
      // Replicate runBackward with introspection
      let graph = lossSig.graph
      graph.markDirty()
      _ = graph.setupGradients(loss: lossSig.nodeId, frameCount: frameCount)
      if graph.graph.gradientSideEffects.isEmpty {
        _ = graph.graph.n(.output(0), [lossSig.nodeId])
      } else {
        let chained = graph.graph.chainGradientSideEffects(after: lossSig.nodeId)
        _ = graph.graph.n(.output(0), [chained])
        graph.graph.gradientSideEffects = []
      }
      let context = try graph.compile(frameCount: frameCount)
      graph.run(context: context, preserveState: false)

      let registry = graph.parameterRegistry
      let mappings = context.compilationResult.cellAllocations.cellMappings
      print("[introspect] trainableCutoff=\(trainableCutoff)")
      print("  gradCarryCells (history->carry): \(graph.graph.gradCarryCells)")
      for signal in registry.signals {
        guard let gradCell = registry.signalGradCells[signal.nodeId] else { continue }
        let phys = mappings[gradCell] ?? gradCell
        let val = context.runtime.memoryPointer()?[phys] ?? .nan
        print("  param node=\(signal.nodeId) gradCell=\(gradCell) -> phys=\(phys) value=\(val)")
      }
      for (hist, carry) in graph.graph.gradCarryCells {
        let phys = mappings[carry] ?? carry
        print("  carryCell \(carry) (for history \(hist)) -> phys=\(phys)")
      }
      graph.clearComputationGraph()
      DGenConfig.kernelOutputPath = nil
    }
  }
}
