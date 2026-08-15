import DGen
import XCTest

@testable import DGenLazy

/// Scratch localization for BPTT through a hand-rolled two-cell ZDF SVF
/// (the MonologueVoice filter in Examples/SynthID/Patch.swift). The SVF has
/// two mutually-coupled history cells with pass-through writes; gradients
/// for its parameters (and everything upstream) must include the temporal
/// recursion, exactly like the biquad after B1
/// (docs/BIQUAD_BPTT_GRADIENT_BUG.md).
final class SVFBPTTScratchTests: XCTestCase {

  override func setUp() {
    super.setUp()
    DGenConfig.maxFrameCount = 4096
    LazyGraphContext.reset()
  }

  /// ZDF SVF lowpass with pass-through history writes; k=0 sat strength.
  private func svfLowpass(input: Signal, g: Signal, kDamp: Signal, kSat: Signal) -> Signal {
    let a1 = 1.0 / (1.0 + g * (g + kDamp))
    let a2 = g * a1
    let ic1 = Signal.history()
    let ic2 = Signal.history()
    let s1 = ic1.read / (1.0 + DGenLazy.abs(kSat * ic1.read))
    let s2 = ic2.read / (1.0 + DGenLazy.abs(kSat * ic2.read))
    let v3 = input - s2
    let v1 = a1 * s1 + a2 * v3
    let ic1New = ic1.write(2.0 * v1 - s1)
    let v1PassThrough = (ic1New + s1) * 0.5
    let v2 = s2 + g * v1PassThrough
    let ic2New = ic2.write(2.0 * v2 - s2)
    return (ic2New + s2) * 0.5
  }

  private func buildLoss(gValue: Signal) -> Signal {
    let saw = Signal.phasor(55.0) * 2.0 - 1.0
    let student = svfLowpass(
      input: saw, g: gValue,
      kDamp: Signal.constant(1.0 / 1.2), kSat: Signal.constant(0.0))
    let saw2 = Signal.phasor(55.0) * 2.0 - 1.0
    let target = svfLowpass(
      input: saw2, g: Signal.constant(0.06),
      kDamp: Signal.constant(1.0 / 1.2), kSat: Signal.constant(0.0))
    return mse(student, target)
  }

  private func lossValue(g: Float, frames: Int) throws -> Float {
    LazyGraphContext.reset()
    let loss = buildLoss(gValue: Signal.constant(g))
    let values = try loss.realize(frames: frames)
    return values.reduce(0, +)
  }

  // Step 2 of the bisection: time-varying g (envelope-driven cutoff through
  // tan), trainable fBase — mirrors MonologueVoice's filter drive.
  private func buildLossTimeVarying(fBase: Signal, kSat: Signal) -> Signal {
    let sr = Signal.constant(44100.0)
    let t = Signal.accum(Signal.constant(1.0 / 44100.0), reset: 0.0, min: 0.0, max: 1.0)
    let saw = Signal.phasor(55.0) * 2.0 - 1.0
    let cutoff = (fBase + 900.0 * DGenLazy.exp(-t / 0.15)).clip(20.0, 20000.0)
    let g = DGenLazy.tan(Signal.constant(Float.pi) * cutoff / sr)
    let student = svfLowpass(
      input: saw, g: g, kDamp: Signal.constant(1.0 / 1.2), kSat: kSat)

    let t2 = Signal.accum(Signal.constant(1.0 / 44100.0), reset: 0.0, min: 0.0, max: 1.0)
    let saw2 = Signal.phasor(55.0) * 2.0 - 1.0
    let cutoff2 = (Signal.constant(400.0) + 900.0 * DGenLazy.exp(-t2 / 0.15)).clip(20.0, 20000.0)
    let g2 = DGenLazy.tan(Signal.constant(Float.pi) * cutoff2 / sr)
    let target = svfLowpass(
      input: saw2, g: g2, kDamp: Signal.constant(1.0 / 1.2), kSat: Signal.constant(0.0))
    return mse(student, target)
  }

  private func lossValueTV(fBase: Float, kSat: Float, frames: Int) throws -> Float {
    LazyGraphContext.reset()
    let loss = buildLossTimeVarying(
      fBase: Signal.constant(fBase), kSat: Signal.constant(kSat))
    return try loss.realize(frames: frames).reduce(0, +)
  }

  func testTimeVaryingCutoffGradientMatchesFD() throws {
    let frames = 512
    let f0: Float = 300.0
    let eps: Float = 0.5

    LazyGraphContext.reset()
    let param = Signal.param(f0)
    let loss = buildLossTimeVarying(fBase: param, kSat: Signal.constant(0.0))
    _ = try loss.backward(frames: frames)
    let autograd = param.grad?.data

    let fd = (try lossValueTV(fBase: f0 + eps, kSat: 0, frames: frames)
      - (try lossValueTV(fBase: f0 - eps, kSat: 0, frames: frames))) / (2 * eps)
    print("TV-SVF dL/dfBase: autograd=\(autograd ?? .nan) fd=\(fd)")
    let relErr = abs((autograd ?? 0) - fd) / max(abs(fd), 1e-9)
    print("relErr=\(relErr)")
    XCTAssertLessThan(relErr, 0.05)
  }

  func testFiltSatGradientMatchesFD() throws {
    let frames = 512
    let k0: Float = 1.0
    let eps: Float = 1e-2

    LazyGraphContext.reset()
    let param = Signal.param(k0)
    let loss = buildLossTimeVarying(fBase: Signal.constant(300.0), kSat: param)
    _ = try loss.backward(frames: frames)
    let autograd = param.grad?.data

    let fd = (try lossValueTV(fBase: 300, kSat: k0 + eps, frames: frames)
      - (try lossValueTV(fBase: 300, kSat: k0 - eps, frames: frames))) / (2 * eps)
    print("TV-SVF dL/dfiltSat: autograd=\(autograd ?? .nan) fd=\(fd)")
    let relErr = abs((autograd ?? 0) - fd) / max(abs(fd), 1e-9)
    print("relErr=\(relErr)")
    XCTAssertLessThan(relErr, 0.05)
  }

  // Step 3: spectral loss (multi-kernel; exercises detached backward-block
  // reverse scheduling) instead of MSE.
  private func buildSpectralLoss(fBase: Signal) -> Signal {
    let sr = Signal.constant(44100.0)
    let t = Signal.accum(Signal.constant(1.0 / 44100.0), reset: 0.0, min: 0.0, max: 1.0)
    let saw = Signal.phasor(55.0) * 2.0 - 1.0
    let cutoff = (fBase + 900.0 * DGenLazy.exp(-t / 0.15)).clip(20.0, 20000.0)
    let g = DGenLazy.tan(Signal.constant(Float.pi) * cutoff / sr)
    let student = svfLowpass(
      input: saw, g: g, kDamp: Signal.constant(1.0 / 1.2), kSat: Signal.constant(0.5))

    let t2 = Signal.accum(Signal.constant(1.0 / 44100.0), reset: 0.0, min: 0.0, max: 1.0)
    let saw2 = Signal.phasor(55.0) * 2.0 - 1.0
    let cutoff2 = (Signal.constant(400.0) + 900.0 * DGenLazy.exp(-t2 / 0.15)).clip(20.0, 20000.0)
    let g2 = DGenLazy.tan(Signal.constant(Float.pi) * cutoff2 / sr)
    let target = svfLowpass(
      input: saw2, g: g2, kDamp: Signal.constant(1.0 / 1.2), kSat: Signal.constant(0.0))
    return spectralLossFFT(
      student, target, windowSize: 256, lossMode: .l2, hop: 64, normalize: true)
  }

  private func spectralLossValue(fBase: Float, frames: Int) throws -> Float {
    LazyGraphContext.reset()
    let loss = buildSpectralLoss(fBase: Signal.constant(fBase))
    return try loss.realize(frames: frames).reduce(0, +)
  }

  func testSpectralLossCutoffGradientMatchesFD() throws {
    let frames = 2048
    let f0: Float = 300.0
    let eps: Float = 1.0

    LazyGraphContext.reset()
    let param = Signal.param(f0)
    let loss = buildSpectralLoss(fBase: param)
    _ = try loss.backward(frames: frames)
    let autograd = param.grad?.data

    let fd = (try spectralLossValue(fBase: f0 + eps, frames: frames)
      - (try spectralLossValue(fBase: f0 - eps, frames: frames))) / (2 * eps)
    print("SPEC-SVF dL/dfBase: autograd=\(autograd ?? .nan) fd=\(fd)")
    let relErr = abs((autograd ?? 0) - fd) / max(abs(fd), 1e-9)
    print("relErr=\(relErr)")
    XCTAssertLessThan(relErr, 0.05)
  }

  // Step 4: the full MonologueVoice chain (two polyblep VCOs via
  // statefulPhasor, polynomial pre-sat, SVF, VCA envelope, softsign drive).
  private func polyblepOsc(frequency: Signal, shape: Signal, pw: Signal) -> Signal {
    let sr = Signal.constant(44100.0)
    let phase = Signal.statefulPhasor(frequency)
    let dt = (frequency / sr).clip(0.000001, 0.5)
    let blep: (Signal) -> Signal = { p in
      let lX = p / dt
      let l = 2.0 * lX - lX * lX - 1.0
      let rX = (p - 1.0) / dt
      let r = rX * rX + 2.0 * rX + 1.0
      return (p < dt) * l + (p > (1.0 - dt)) * r
    }
    let saw = (phase * 2.0 - 1.0) - blep(phase)
    let clippedWidth = pw.clip(0.01, 0.99)
    let fallingPhase = mod(phase - clippedWidth, 1.0)
    let rawPulse = (phase < clippedWidth) * 2.0 - 1.0
    let pulse = rawPulse + blep(phase) - blep(fallingPhase)
    return (1.0 - shape) * saw + shape * pulse
  }

  private func buildVoiceLoss(fBase: Signal, satA3: Signal) -> Signal {
    func voice(fBaseV: Signal, satA3V: Signal) -> Signal {
      let sr = Signal.constant(44100.0)
      let t = Signal.accum(Signal.constant(1.0 / 44100.0), reset: 0.0, min: 0.0, max: 1.0)
      let osc1 = polyblepOsc(
        frequency: Signal.constant(35.37), shape: Signal.constant(0.6),
        pw: Signal.constant(0.55))
      let osc2 = polyblepOsc(
        frequency: Signal.constant(35.37 - 0.65), shape: Signal.constant(0.6),
        pw: Signal.constant(0.55))
      let mixed = (osc1 + 0.6 * osc2) / 1.6
      let y = 1.8 * mixed + 0.12
      let shapedPre = y + 0.15 * (y * y) + satA3V * (y * y * y)
      let preSat = shapedPre - (0.12 + 0.15 * 0.0144 + satA3V * 0.001728)
      let cutoff = (fBaseV + 900.0 * DGenLazy.exp(-t / 0.15)).clip(20.0, 20000.0)
      let g = DGenLazy.tan(Signal.constant(Float.pi) * cutoff / sr)
      let lp = svfLowpass(
        input: preSat, g: g, kDamp: Signal.constant(1.0 / 1.2),
        kSat: Signal.constant(1.2))
      let attack = 1.0 - DGenLazy.exp(-t / 0.005)
      let decay = DGenLazy.exp(-t / 0.2)
      let release = 1.0 / (1.0 + DGenLazy.exp((t - 0.5) / 0.02))
      let driven = lp * attack * decay * release * 2.5
      let shaped = driven / (1.0 + DGenLazy.abs(driven))
      return shaped * 0.45
    }
    let student = voice(fBaseV: fBase, satA3V: satA3)
    let target = voice(fBaseV: Signal.constant(400.0), satA3V: Signal.constant(0.1))
    return mse(student, target)
  }

  private func voiceLossValue(fBase: Float, satA3: Float, frames: Int) throws -> Float {
    LazyGraphContext.reset()
    let loss = buildVoiceLoss(
      fBase: Signal.constant(fBase), satA3: Signal.constant(satA3))
    return try loss.realize(frames: frames).reduce(0, +)
  }

  func testFullVoiceCutoffGradientMatchesFD() throws {
    let frames = 2048
    let f0: Float = 300.0
    let eps: Float = 1.0

    LazyGraphContext.reset()
    let param = Signal.param(f0)
    let loss = buildVoiceLoss(fBase: param, satA3: Signal.constant(0.2))
    _ = try loss.backward(frames: frames)
    let autograd = param.grad?.data

    let fd = (try voiceLossValue(fBase: f0 + eps, satA3: 0.2, frames: frames)
      - (try voiceLossValue(fBase: f0 - eps, satA3: 0.2, frames: frames))) / (2 * eps)
    print("VOICE dL/dfBase: autograd=\(autograd ?? .nan) fd=\(fd)")
    let relErr = abs((autograd ?? 0) - fd) / max(abs(fd), 1e-9)
    print("relErr=\(relErr)")
    XCTAssertLessThan(relErr, 0.05)
  }

  func testFullVoiceSatA3GradientMatchesFD() throws {
    let frames = 2048
    let a0: Float = 0.2
    let eps: Float = 1e-3

    LazyGraphContext.reset()
    let param = Signal.param(a0)
    let loss = buildVoiceLoss(fBase: Signal.constant(300.0), satA3: param)
    _ = try loss.backward(frames: frames)
    let autograd = param.grad?.data

    let fd = (try voiceLossValue(fBase: 300, satA3: a0 + eps, frames: frames)
      - (try voiceLossValue(fBase: 300, satA3: a0 - eps, frames: frames))) / (2 * eps)
    print("VOICE dL/dsatA3: autograd=\(autograd ?? .nan) fd=\(fd)")
    let relErr = abs((autograd ?? 0) - fd) / max(abs(fd), 1e-9)
    print("relErr=\(relErr)")
    XCTAssertLessThan(relErr, 0.05)
  }

  func testFullVoiceCutoffGradientAtRealFrameCount() throws {
    DGenConfig.maxFrameCount = 32768
    let frames = 26624
    let f0: Float = 300.0
    let eps: Float = 1.0

    LazyGraphContext.reset()
    let param = Signal.param(f0)
    let loss = buildVoiceLoss(fBase: param, satA3: Signal.constant(0.2))
    _ = try loss.backward(frames: frames)
    let autograd = param.grad?.data

    let fd = (try voiceLossValue(fBase: f0 + eps, satA3: 0.2, frames: frames)
      - (try voiceLossValue(fBase: f0 - eps, satA3: 0.2, frames: frames))) / (2 * eps)
    print("VOICE26k dL/dfBase: autograd=\(autograd ?? .nan) fd=\(fd)")
    let relErr = abs((autograd ?? 0) - fd) / max(abs(fd), 1e-9)
    print("relErr=\(relErr)")
    XCTAssertLessThan(relErr, 0.05)
  }

  // Step 5: MANY simultaneous gradient targets (the SynthID harness trains
  // all ~20 scalars at once; scratch steps 1-4 had a single target).
  private func buildVoiceLossMulti(params: [String: Signal]) -> Signal {
    let sr = Signal.constant(44100.0)
    let t = Signal.accum(Signal.constant(1.0 / 44100.0), reset: 0.0, min: 0.0, max: 1.0)
    let osc1 = polyblepOsc(
      frequency: Signal.constant(35.37), shape: params["shape"]!, pw: params["pw"]!)
    let osc2 = polyblepOsc(
      frequency: Signal.constant(35.37) - (params["vco2Detune"] ?? Signal.constant(0.65)),
      shape: params["shape"]!, pw: params["pw"]!)
    let mixed = (osc1 + params["vco2Level"]! * osc2) / (1.0 + params["vco2Level"]!)
    let y = params["satGain"]! * mixed + 0.12
    let shapedPre = y + params["satA2"]! * (y * y) + params["satA3"]! * (y * y * y)
    let preSat = shapedPre - (0.12 + params["satA2"]! * 0.0144 + params["satA3"]! * 0.001728)
    let cutoff = (params["fBase"]! + 900.0 * DGenLazy.exp(-t / 0.15)).clip(20.0, 20000.0)
    let g = DGenLazy.tan(Signal.constant(Float.pi) * cutoff / sr)
    let kDamp = 1.0 / params["res"]!
    let lp = svfLowpass(
      input: preSat, g: g, kDamp: kDamp, kSat: params["filtSat"]!)
    let attack = 1.0 - DGenLazy.exp(-t / 0.005)
    let decay = DGenLazy.exp(-t / 0.2)
    let release = 1.0 / (1.0 + DGenLazy.exp((t - 0.5) / 0.02))
    let driven = lp * attack * decay * release * params["drive"]!
    let shaped = driven / (1.0 + DGenLazy.abs(driven))
    return mse(shaped * 0.45, Signal.constant(0.0))
  }

  private let multiDefaults: [String: Float] = [
    "shape": 0.6, "pw": 0.55, "vco2Level": 0.6, "satGain": 1.8, "satA3": 0.2, "satA2": 0.15,
    "fBase": 300.0, "res": 1.2, "filtSat": 1.2, "drive": 2.5,
  ]

  private func multiLossValue(overrides: [String: Float], frames: Int) throws -> Float {
    LazyGraphContext.reset()
    var vals = multiDefaults
    for (k, v) in overrides { vals[k] = v }
    let sigs = vals.mapValues { Signal.constant($0) }
    return try buildVoiceLossMulti(params: sigs).realize(frames: frames).reduce(0, +)
  }

  func testFullVoiceManyTargetsGradientsWithTrainableDetune() throws {
    let frames = 2048
    LazyGraphContext.reset()
    var sigs = multiDefaults.mapValues { Signal.param($0) }
    sigs["vco2Detune"] = Signal.param(0.65)
    let loss = buildVoiceLossMulti(params: sigs)
    _ = try loss.backward(frames: frames)

    var failures: [String] = []
    for (name, eps) in [("fBase", Float(1.0)), ("satA3", Float(1e-3)),
                        ("res", Float(1e-2)), ("drive", Float(1e-2))] {
      let base = multiDefaults[name]!
      let fd = (try multiLossValue(overrides: [name: base + eps], frames: frames)
        - (try multiLossValue(overrides: [name: base - eps], frames: frames))) / (2 * eps)
      let auto = sigs[name]!.grad?.data ?? .nan
      let relErr = abs(auto - fd) / max(abs(fd), 1e-9)
      print("DETUNE-MULTI \(name): autograd=\(auto) fd=\(fd) relErr=\(relErr)")
      if relErr > 0.05 { failures.append(name) }
    }
    // Fixed 2026-08-15: the phasor's temporalGradStore/Scan/Read tape used to
    // split the scalar history recurrence across blocks (carry reads in one
    // block, carry writes stranded after the scan), truncating BPTT and
    // corrupting unrelated gradients ~10x too small. The fragmented layout is
    // now rebuilt by the detached scalar BPTT consolidation in BlockFormation.
    XCTAssertTrue(
      failures.isEmpty,
      "trainable phasor frequency corrupts other gradients: \(failures)")
  }

  func testFullVoiceManyTargetsGradients() throws {
    let frames = 2048
    LazyGraphContext.reset()
    let sigs = multiDefaults.mapValues { Signal.param($0) }
    let loss = buildVoiceLossMulti(params: sigs)
    _ = try loss.backward(frames: frames)

    let epsByName: [String: Float] = [
      "shape": 1e-3, "pw": 1e-3, "vco2Level": 1e-3, "satGain": 1e-3,
      "satA3": 1e-3, "satA2": 1e-3, "fBase": 1.0, "res": 1e-2, "filtSat": 1e-2, "drive": 1e-2,
    ]
    var failures: [String] = []
    for (name, eps) in epsByName.sorted(by: { $0.key < $1.key }) {
      let base = multiDefaults[name]!
      let fd = (try multiLossValue(overrides: [name: base + eps], frames: frames)
        - (try multiLossValue(overrides: [name: base - eps], frames: frames))) / (2 * eps)
      let auto = sigs[name]!.grad?.data ?? .nan
      let relErr = abs(auto - fd) / max(abs(fd), 1e-9)
      print("MULTI \(name): autograd=\(auto) fd=\(fd) relErr=\(relErr)")
      // pw's end-to-end float32 FD is ill-conditioned near the PolyBLEP
      // transition width (E0_FINDING.md froze a chain-rule instrument for
      // it); a ~10% central-difference spread there is expected.
      let bar: Float = name == "pw" ? 0.2 : 0.05
      if relErr > bar { failures.append(name) }
    }
    XCTAssertTrue(failures.isEmpty, "gradient mismatch for: \(failures)")
  }

  func testSVFGGradientMatchesFD() throws {
    let frames = 512
    let g0: Float = 0.03
    let eps: Float = 1e-3

    LazyGraphContext.reset()
    let param = Signal.param(g0)
    let loss = buildLoss(gValue: param)
    _ = try loss.backward(frames: frames)
    let autograd = param.grad?.data

    let lossPlus = try lossValue(g: g0 + eps, frames: frames)
    let lossMinus = try lossValue(g: g0 - eps, frames: frames)
    let fd = (lossPlus - lossMinus) / (2 * eps)

    print("SVF dL/dg: autograd=\(autograd ?? .nan) fd=\(fd)")
    let relErr = abs((autograd ?? 0) - fd) / max(abs(fd), 1e-9)
    print("relErr=\(relErr)")
    XCTAssertLessThan(relErr, 0.05, "temporal gradient through SVF is wrong/truncated")
  }
}
