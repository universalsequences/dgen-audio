// BatchTrainBench.swift
//
// Minimal repro harness for batched biquad *gradient* trajectories
// (docs/TENSOR_BIQUAD_GRADIENT_SPEC.md, spec test 6 + acceptance timing).
//
// Runs B independent elite-refinement-style Adam trajectories — same target,
// different initial params — either as ONE [B]-batched trajectory through
// `SignalTensor.biquad` backward, or as B separate single-lane trajectories,
// and compares per-step per-lane parameters and wall-clock cost.
//
// Like BatchBench, this file touches no production voice/trainer code. The
// voice here is deliberately minimal (sine -> time-varying-cutoff biquad ->
// gain) but keeps the exact control shape the subtractive-bass refinement
// trains: cutoff_i(t) = fBase_i + fAmt_i * exp(-t / fDecay_i).

import DGenLazy
import Foundation

enum BatchTrainBench {

  // Target patch (shared by every lane, like elites refining toward one target)
  struct Target {
    static let fBase: Float = 320
    static let fAmt: Float = 2400
    static let fDecay: Float = 0.035
    static let res: Float = 1.4
    static let gain: Float = 0.8
  }

  /// One lane's trainable parameters in log space (all-positive params).
  struct LaneParams {
    var zBase: Float
    var zAmt: Float
    var zDecay: Float
    var zRes: Float
    var zGain: Float

    static func perturbed(from t: Target.Type, rng: inout SplitMix64) -> LaneParams {
      func jitter(_ v: Float, _ spread: Float) -> Float {
        Foundation.log(v) + (rng.nextUniform() * 2 - 1) * spread
      }
      return LaneParams(
        zBase: jitter(t.fBase, 0.4),
        zAmt: jitter(t.fAmt, 0.4),
        zDecay: jitter(t.fDecay, 0.3),
        zRes: jitter(t.res, 0.25),
        zGain: jitter(t.gain, 0.2))
    }
  }

  struct SplitMix64 {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
      state &+= 0x9E37_79B9_7F4A_7C15
      var z = state
      z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
      z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
      return z ^ (z >> 31)
    }
    mutating func nextUniform() -> Float {
      Float(next() >> 40) / Float(1 << 24)
    }
  }

  // MARK: - Graph construction

  private static func sine(_ frequency: Float, sampleRate: Float) -> Signal {
    let phase = Signal.accum(
      Signal.constant(2 * Float.pi * frequency / sampleRate),
      reset: 0, min: 0, max: 1e9)
    return DGenLazy.sin(phase)
  }

  private static func timeSignal(sampleRate: Float, frames: Int) -> Signal {
    Signal.accum(
      Signal.constant(1 / sampleRate), reset: 0, min: 0,
      max: Float(frames + 1) / sampleRate + 1)
  }

  /// Builds the batched loss for one Adam step. Teacher is rendered in-graph
  /// from constant target params (no gradients flow into it).
  private static func buildLoss(
    zBase: Tensor, zAmt: Tensor, zDecay: Tensor, zRes: Tensor, zGain: Tensor,
    batch: Int, sampleRate: Float, frames: Int, oscFreq: Float,
    windowSize: Int, hop: Int
  ) -> Signal {
    let one = Signal.constant(1)
    let t = timeSignal(sampleRate: sampleRate, frames: frames)
    let input = Tensor([Float](repeating: 1, count: batch)) * sine(oscFreq, sampleRate: sampleRate)

    let cutoff =
      DGenLazy.exp(zBase * one)
      + DGenLazy.exp(zAmt * one) * DGenLazy.exp(-t / DGenLazy.exp(zDecay * one))
    let student =
      input.biquad(
        cutoff: cutoff,
        resonance: DGenLazy.exp(zRes * one),
        gain: Signal.constant(1),
        mode: Signal.constant(0)) * DGenLazy.exp(zGain * one)

    let teacherCutoff =
      Signal.constant(Target.fBase)
      + Signal.constant(Target.fAmt) * DGenLazy.exp(-t / Signal.constant(Target.fDecay))
    let teacher =
      (Tensor([Float](repeating: 1, count: batch)) * sine(oscFreq, sampleRate: sampleRate))
      .biquad(
        cutoff: teacherCutoff,
        resonance: Signal.constant(Target.res),
        gain: Signal.constant(1),
        mode: Signal.constant(0)) * Signal.constant(Target.gain)

    // The batched spectral loss returns the MEAN across lanes; rescale to a
    // sum so each lane's gradient is independent of the batch size (a [1]-lane
    // trajectory and one lane of a [B] trajectory then match exactly).
    return spectralLossFFT(
      student, teacher, windowSize: windowSize, lossMode: .l2, hop: hop, normalize: true)
      * Signal.constant(Float(batch))
  }

  /// Runs one Adam trajectory over `laneInits` (B lanes) for `steps` steps.
  /// Returns per-step per-lane parameter snapshots and per-step mean loss.
  private static func runTrajectory(
    laneInits: [LaneParams], steps: Int, lr: Float,
    sampleRate: Float, frames: Int, oscFreq: Float, windowSize: Int, hop: Int
  ) throws -> (paramTrace: [[[Float]]], lossTrace: [Float], secondsPerStep: Double) {
    LazyGraphContext.reset()
    let batch = laneInits.count
    let zBase = Tensor(laneInits.map { $0.zBase }, requiresGrad: true)
    let zAmt = Tensor(laneInits.map { $0.zAmt }, requiresGrad: true)
    let zDecay = Tensor(laneInits.map { $0.zDecay }, requiresGrad: true)
    let zRes = Tensor(laneInits.map { $0.zRes }, requiresGrad: true)
    let zGain = Tensor(laneInits.map { $0.zGain }, requiresGrad: true)
    let tensors = [zBase, zAmt, zDecay, zRes, zGain]
    let opt = Adam(params: tensors, lr: lr)

    var paramTrace: [[[Float]]] = []
    var lossTrace: [Float] = []
    let start = Date()
    for _ in 0..<steps {
      let loss = buildLoss(
        zBase: zBase, zAmt: zAmt, zDecay: zDecay, zRes: zRes, zGain: zGain,
        batch: batch, sampleRate: sampleRate, frames: frames, oscFreq: oscFreq,
        windowSize: windowSize, hop: hop)
      let lossValues = try loss.backward(frames: frames)
      lossTrace.append(lossValues.reduce(0, +))
      opt.step()
      opt.zeroGrad()
      paramTrace.append(tensors.map { $0.getData() ?? [] })
    }
    let elapsed = Date().timeIntervalSince(start)
    return (paramTrace, lossTrace, elapsed / Double(steps))
  }

  // MARK: - Entry point

  static func run(options: [String: String]) throws {
    let batch = Int(options["batch"] ?? "12") ?? 12
    let steps = Int(options["steps"] ?? "20") ?? 20
    let frames = Int(options["frames"] ?? "8192") ?? 8192
    let lr = Float(options["lr"] ?? "0.02") ?? 0.02
    let sampleRate: Float = 44_100
    let oscFreq: Float = 55
    let windowSize = Int(options["window"] ?? "1024") ?? 1024
    let hop = Int(options["hop"] ?? "256") ?? 256
    let mode = options["mode"] ?? "equivalence"

    DGenConfig.backend = .metal
    if let dump = ProcessInfo.processInfo.environment["TENSOR_BIQUAD_GRAD_DUMP"] {
      DGenConfig.kernelOutputPath = dump
    }
    DGenConfig.sampleRate = sampleRate
    DGenConfig.defaultFrameCount = frames
    DGenConfig.maxFrameCount = frames

    var rng = SplitMix64(seed: UInt64(options["seed"] ?? "6") ?? 6)
    let laneInits = (0..<batch).map { _ in LaneParams.perturbed(from: Target.self, rng: &rng) }

    print("=== BatchTrainBench: B=\(batch) steps=\(steps) frames=\(frames) mode=\(mode) ===")

    if options.keys.contains("probe-forward") {
      // Forward-only loss probe: batched loss vs per-lane [1]-batched losses,
      // no backward pass involved.
      func forwardLoss(_ lanes: [LaneParams]) throws -> Float {
        LazyGraphContext.reset()
        let loss = buildLoss(
          zBase: Tensor(lanes.map { $0.zBase }), zAmt: Tensor(lanes.map { $0.zAmt }),
          zDecay: Tensor(lanes.map { $0.zDecay }), zRes: Tensor(lanes.map { $0.zRes }),
          zGain: Tensor(lanes.map { $0.zGain }),
          batch: lanes.count, sampleRate: sampleRate, frames: frames, oscFreq: oscFreq,
          windowSize: windowSize, hop: hop)
        return try loss.realize(frames: frames).reduce(0, +)
      }
      let full = try forwardLoss(laneInits)
      var perLane = [Float]()
      for i in 0..<batch { perLane.append(try forwardLoss([laneInits[i]])) }
      print("probe-forward: batched=\(full) perLane=\(perLane) mean=\(perLane.reduce(0,+)/Float(batch))")
      return
    }

    if options.keys.contains("probe-grads") {
      // Single-backward gradient comparison: batched [B] vs per-lane [1].
      func grads(_ lanes: [LaneParams]) throws -> [[Float]] {
        LazyGraphContext.reset()
        let zB = Tensor(lanes.map { $0.zBase }, requiresGrad: true)
        let zA = Tensor(lanes.map { $0.zAmt }, requiresGrad: true)
        let zD = Tensor(lanes.map { $0.zDecay }, requiresGrad: true)
        let zR = Tensor(lanes.map { $0.zRes }, requiresGrad: true)
        let zG = Tensor(lanes.map { $0.zGain }, requiresGrad: true)
        let loss = buildLoss(
          zBase: zB, zAmt: zA, zDecay: zD, zRes: zR, zGain: zG,
          batch: lanes.count, sampleRate: sampleRate, frames: frames, oscFreq: oscFreq,
          windowSize: windowSize, hop: hop)
        _ = try loss.backward(frames: frames)
        return [zB, zA, zD, zR, zG].map { $0.grad?.getData() ?? [] }
      }
      let full = try grads(laneInits)
      DGenConfig.kernelOutputPath = nil  // keep the dump from the batched compile only
      let names = ["zBase", "zAmt", "zDecay", "zRes", "zGain"]
      var worst: Float = 0
      var worstDesc = ""
      for lane in 0..<batch {
        let single = try grads([laneInits[lane]])
        for p in 0..<names.count {
          let b = full[p][lane]
          let s = single[p][0]
          let rel = abs(b - s) / max(abs(s), 1e-6)
          if rel > worst {
            worst = rel
            worstDesc = "lane \(lane) \(names[p]): batched \(b) vs single \(s)"
          }
        }
      }
      print("probe-grads: worst rel diff \(worst)  (\(worstDesc))")
      for p in 0..<names.count {
        var singles = [Float]()
        for lane in 0..<batch { singles.append(try grads([laneInits[lane]])[p][0]) }
        print("  \(names[p]): batched \(full[p]) vs singles \(singles)")
      }
      return
    }

    print("--- batched [\(batch)]-lane trajectory ---")
    let batched = try runTrajectory(
      laneInits: laneInits, steps: steps, lr: lr,
      sampleRate: sampleRate, frames: frames, oscFreq: oscFreq,
      windowSize: windowSize, hop: hop)
    print(String(format: "batched: %.3f s/step, final mean loss %.6e",
                 batched.secondsPerStep, batched.lossTrace.last ?? -1))

    guard mode == "equivalence" else {
      print(String(format: "timing-only: batched %.3f s/step for %d lanes = %.4f s/lane-step",
                   batched.secondsPerStep, batch, batched.secondsPerStep / Double(batch)))
      return
    }

    print("--- \(batch) serial single-lane trajectories ---")
    var serialTraces: [(paramTrace: [[[Float]]], lossTrace: [Float])] = []
    var serialSecondsPerStep = 0.0
    for i in 0..<batch {
      let single = try runTrajectory(
        laneInits: [laneInits[i]], steps: steps, lr: lr,
        sampleRate: sampleRate, frames: frames, oscFreq: oscFreq,
        windowSize: windowSize, hop: hop)
      serialTraces.append((single.paramTrace, single.lossTrace))
      serialSecondsPerStep += single.secondsPerStep
    }

    // Per-step per-lane parameter comparison.
    var worstRelDiff: Float = 0
    var worstDesc = ""
    let paramNames = ["zBase", "zAmt", "zDecay", "zRes", "zGain"]
    for step in 0..<steps {
      for lane in 0..<batch {
        for p in 0..<paramNames.count {
          let b = batched.paramTrace[step][p][lane]
          let s = serialTraces[lane].paramTrace[step][p][0]
          let rel = abs(b - s) / max(abs(s), 1e-3)
          if rel > worstRelDiff {
            worstRelDiff = rel
            worstDesc = "step \(step) lane \(lane) \(paramNames[p]): batched \(b) vs serial \(s)"
          }
        }
      }
    }
    let meanSerialFinalLoss =
      serialTraces.map { $0.lossTrace.last ?? 0 }.reduce(0, +) / Float(batch)

    print("loss traces:")
    print("  batched mean: \(batched.lossTrace)")
    for lane in 0..<batch {
      print("  serial lane \(lane): \(serialTraces[lane].lossTrace)")
    }
    print("worst per-step per-lane param rel diff: \(worstRelDiff)  (\(worstDesc))")
    print(String(format: "final loss: batched mean %.6e vs mean of serial %.6e",
                 batched.lossTrace.last ?? -1, meanSerialFinalLoss))
    print(String(format: "timing: batched %.3f s/step vs serial sum %.3f s/step  (speedup %.1fx)",
                 batched.secondsPerStep, serialSecondsPerStep,
                 serialSecondsPerStep / batched.secondsPerStep))
    let pass = worstRelDiff < 0.05
    print("equivalencePass=\(pass)")
  }
}
