// BatchRefine.swift
//
// Lane-parallel batched gradient refinement for the subtractive-bass voice
// (docs/TENSOR_BIQUAD_PARALLEL_LANES_SPEC.md follow-up: the "wire batched
// refinement into the elite-polish path" item).
//
// Three population-search experiments over ONE [B]-batched Adam trajectory
// (B >= 32 per the parallel-lanes acceptance decision):
//
//   polish    - basin-search elites x jittered restarts packed into one batch;
//               best lane per elite reported by the independent CPU mrstft
//               score (same objective family as restart selection).
//   escape    - documented seed-6 stall (E1_POLICY_AUDIT_V2_FINDING.md):
//               lane 0 is the exact stuck init, lanes 1..B-1 are jittered
//               copies; tests whether population restarts escape the
//               compensated basin that swallows single trajectories.
//   lr-sweep  - all lanes share one init; each lane gets its own global LR
//               multiplier (lanes are independent through backward), re-deriving
//               a good LR post spectral-grad-scale fix in one batch.
//
// Conventions copied from BatchTrainBench (the working reference):
//   - LazyGraphContext.reset() once per trajectory, trainable Tensors created
//     AFTER the reset, loss graph rebuilt every step.
//   - Batched spectral loss returns the MEAN over lanes; the loss is scaled
//     by B in-graph so per-lane gradients are batch-size invariant.
//   - Gradient probes: [B] vs per-lane [1] (probe-grads) and [1] vs the
//     production scalar TrainableKickParams path (probe-scalar).
//
// Trainable parameters live in the SAME transformed coordinates as the
// production trainer (ParameterSpec reparam; exp/expm1 applied in-graph), with
// the production per-group LR ratios and post-step projection onto the
// transformed bounds.

import DGenLazy
import Foundation

enum BatchRefine {

  // The 12 subtractive-bass surface params, grouped exactly as
  // SynthIDTrainer.train() groups them (with the same LR scale factors
  // relative to the config LRs).
  struct ParamGroup {
    let names: [String]
    let lr: (SynthIDConfig) -> Float
  }

  static let groups: [ParamGroup] = [
    ParamGroup(names: ["sustain", "outGain"], lr: { $0.ampLR * 0.1 }),
    ParamGroup(
      names: ["attackTime", "decayTime", "releaseTime", "fDecay"], lr: { $0.decayLR / 3.0 }),
    ParamGroup(names: ["fBase", "fAmt", "res", "drive"], lr: { $0.toneLR }),
    ParamGroup(names: ["shape", "pw"], lr: { $0.toneLR * 0.1 }),
  ]

  static var paramNames: [String] { groups.flatMap(\.names) }

  static func groupLR(_ name: String, config: SynthIDConfig) -> Float {
    for group in groups where group.names.contains(name) {
      return group.lr(config)
    }
    fatalError("no LR group for \(name)")
  }

  // MARK: - Per-lane Adam (transformed space, per-lane LR, bounds projection)

  final class PerLaneAdam {
    let laneLR: [Float]  // global multiplier per lane
    let beta1: Float = 0.9
    let beta2: Float = 0.999
    let eps: Float = 1e-8
    private var m: [String: [Float]] = [:]
    private var v: [String: [Float]] = [:]
    private var t = 0
    /// Cosine scale applied on top of each param's group LR (mirrors the
    /// production cosineLRDecay schedule).
    var lrScale: Float = 1.0

    init(laneLR: [Float]) {
      self.laneLR = laneLR
    }

    func step(
      params: [(name: String, tensor: Tensor)], config: SynthIDConfig, gradClip: Float
    ) {
      t += 1
      let bc1 = 1.0 - Foundation.pow(beta1, Float(t))
      let bc2 = 1.0 - Foundation.pow(beta2, Float(t))
      for (name, tensor) in params {
        guard let gradData = tensor.grad?.getData(), let data = tensor.getData() else { continue }
        let spec = KickParamSpecs.byName[name]!
        let bounds = spec.transformedBounds
        let base = groupLR(name, config: config) * lrScale
        if m[name] == nil {
          m[name] = [Float](repeating: 0, count: data.count)
          v[name] = [Float](repeating: 0, count: data.count)
        }
        var newData = data
        for lane in 0..<data.count {
          var g = gradData[lane]
          if gradClip > 0 { g = Swift.max(-gradClip, Swift.min(gradClip, g)) }
          m[name]![lane] = beta1 * m[name]![lane] + (1 - beta1) * g
          v[name]![lane] = beta2 * v[name]![lane] + (1 - beta2) * g * g
          let mHat = m[name]![lane] / bc1
          let vHat = v[name]![lane] / bc2
          let updated = data[lane] - base * laneLR[lane] * mHat / (Foundation.sqrt(vHat) + eps)
          newData[lane] = Swift.min(bounds.max, Swift.max(bounds.min, updated))
        }
        tensor.updateDataLazily(newData)
      }
    }

    func zeroGrad(params: [(name: String, tensor: Tensor)]) {
      for (_, tensor) in params { tensor.grad = nil }
    }
  }

  // MARK: - Batched trainable voice (transformed [B] tensors)

  /// Transformed-space [B] parameter tensors. Create AFTER
  /// LazyGraphContext.reset() (CLAUDE.md tensor lifecycle rule).
  static func makeZ(lanes: [PatchValues]) -> [(name: String, tensor: Tensor)] {
    paramNames.map { name in
      let spec = KickParamSpecs.byName[name]!
      return (name, Tensor(lanes.map { spec.transform($0[name]) }, requiresGrad: true))
    }
  }

  static func naturalLaneValues(
    z: [(name: String, tensor: Tensor)], base: PatchValues, batch: Int
  ) -> [PatchValues] {
    var lanes = [PatchValues](repeating: base, count: batch)
    for (name, tensor) in z {
      let spec = KickParamSpecs.byName[name]!
      guard let data = tensor.getData() else { continue }
      for lane in 0..<batch {
        lanes[lane][name] = spec.inverse(data[lane])
      }
    }
    return lanes.map { $0.clamped() }
  }

  private static func polyblep(_ phase: SignalTensor, dt: Signal) -> SignalTensor {
    let leftX = phase / dt
    let left = leftX * 2.0 - leftX * leftX - 1.0
    let rightX = (phase - 1.0) / dt
    let right = rightX * rightX + rightX * 2.0 + 1.0
    return (phase < dt) * left + (phase > (Signal.constant(1.0) - dt)) * right
  }

  /// Batched student voice: SubtractiveBassVoice.build / BatchBench.buildAudio
  /// re-expressed with trainable transformed [B] tensors. Raw-reparam params
  /// (shape/pw/sustain) are the tensors themselves; log params go through
  /// exp() in-graph; fAmt (logOnePlus) through exp()-1.
  static func buildStudent(
    z zList: [(name: String, tensor: Tensor)], batch: Int, config: SynthIDConfig
  ) -> SignalTensor {
    let z = Dictionary(uniqueKeysWithValues: zList)
    let one = Signal.constant(1.0)
    let sr = Signal.constant(config.sampleRate)
    let t = Signal.accum(
      Signal.constant(1.0) / sr,
      reset: 0.0,
      min: 0.0,
      max: Float(config.frames + 1) / config.sampleRate + 1.0)

    // Raw-space trainables used directly (transform is identity).
    let shape = z["shape"]!
    let pw = z["pw"]!
    let sustain = z["sustain"]!
    // Log-space trainables mapped to natural units in-graph.
    let fBase = DGenLazy.exp(z["fBase"]! * one)
    let fAmt = DGenLazy.exp(z["fAmt"]! * one) - 1.0
    let fDecay = DGenLazy.exp(z["fDecay"]! * one)
    let res = DGenLazy.exp(z["res"]! * one)
    let attackTime = DGenLazy.exp(z["attackTime"]! * one)
    let decayTime = DGenLazy.exp(z["decayTime"]! * one)
    let releaseTime = DGenLazy.exp(z["releaseTime"]! * one)
    let drive = DGenLazy.exp(z["drive"]! * one)
    let outGain = DGenLazy.exp(z["outGain"]! * one)

    // f0 is fixed at 110 Hz in this topology (freezePitch), same as the
    // scalar voice and BatchBench.
    let freqTensor = Tensor([Float](repeating: 110.0, count: batch))
    let phase = Signal.statefulPhasor(freqTensor)
    let dt = (Signal.constant(110.0) / sr).clip(0.000001, 0.5)

    let saw = (phase * 2.0 - 1.0) - polyblep(phase, dt: dt)
    // pw's transformed bounds [0.03, 0.97] are inside the scalar voice's
    // clip(0.01, 0.99) and the optimizer projects onto them each step, so the
    // in-graph clip is an identity here and is omitted.
    let fallingPhase = mod(phase - pw, 1.0)
    let rawPulse = (phase < pw) * 2.0 - 1.0
    let pulse = rawPulse + polyblep(phase, dt: dt) - polyblep(fallingPhase, dt: dt)
    let oscillator = (Signal.constant(1.0) - shape) * saw + shape * pulse

    let cutoff = fBase + fAmt * DGenLazy.exp((Signal.constant(0.0) - t) / fDecay)
    let filtered = oscillator.biquad(
      cutoff: cutoff, resonance: res,
      gain: Signal.constant(1.0), mode: Signal.constant(0.0))

    let attack = Signal.constant(1.0) - DGenLazy.exp((Signal.constant(0.0) - t) / attackTime)
    let decay = sustain
      + (Signal.constant(1.0) - sustain) * DGenLazy.exp((Signal.constant(0.0) - t) / decayTime)
    let release = Signal.constant(1.0)
      / (Signal.constant(1.0) + DGenLazy.exp((t - Signal.constant(0.6)) / releaseTime))
    let driven = filtered * attack * decay * release * drive
    let shaped = driven / (Signal.constant(1.0) + DGenLazy.abs(driven))
    return shaped * outGain
  }

  /// Batched analogue of SynthIDLosses.multiResolutionSpectralLoss. The
  /// batched spectralLossFFT has no useSmoothLogMagnitude variant, so
  /// `smooth` approximates the production smooth loss with log-magnitude L2
  /// (kink removal is the property that matters for the schedule).
  /// Scaled by B so per-lane gradients are batch-size invariant.
  static func buildBatchedLoss(
    student: SignalTensor, target: SignalTensor, batch: Int,
    config: SynthIDConfig, smooth: Bool
  ) -> Signal {
    var total = Signal.constant(0.0)
    for (index, window) in config.spectralWindows.enumerated() {
      let weight = index < config.windowWeights.count ? config.windowWeights[index] : 1.0
      let hop = max(1, window / 4)
      let logMag = spectralLossFFT(
        student, target,
        windowSize: window,
        useHannWindow: config.useHannWindow,
        useLogMagnitude: true,
        lossMode: smooth ? .l2 : .l1,
        hop: hop,
        normalize: true)
      total = total + logMag * weight
      if !smooth && config.includeLinearMagnitude {
        let linear = spectralLossFFT(
          student, target,
          windowSize: window,
          useHannWindow: config.useHannWindow,
          useLogMagnitude: false,
          lossMode: .l1,
          hop: hop,
          normalize: true)
        total = total + linear * (weight * config.linearMagnitudeWeight)
      }
    }
    return total * Signal.constant(Float(batch))
  }

  // MARK: - One batched trajectory

  struct TrajectoryResult {
    var lanes: [PatchValues]
    var meanLossTrace: [Float]
    var secondsPerStep: Double
    var paramTrace: [[String: [Float]]]  // sampled every logEvery steps (natural units)
  }

  static func runTrajectory(
    laneInits: [PatchValues], targetSamples: [Float], config: SynthIDConfig,
    steps: Int, smoothSteps: Int, laneLR: [Float], logEvery: Int,
    label: String
  ) throws -> TrajectoryResult {
    let batch = laneInits.count
    precondition(laneLR.count == batch)
    LazyGraphContext.reset()
    config.applyRuntime()
    let z = makeZ(lanes: laneInits)
    let targetTensor = Tensor(targetSamples)
    let onesTensor = Tensor([Float](repeating: 1, count: batch))
    let opt = PerLaneAdam(laneLR: laneLR)

    var meanLossTrace: [Float] = []
    var paramTrace: [[String: [Float]]] = []
    let totalSteps = smoothSteps + steps
    let start = Date()
    for step in 0..<totalSteps {
      let smooth = step < smoothSteps
      let student = buildStudent(z: z, batch: batch, config: config)
      let target = onesTensor * targetTensor.toSignal(maxFrames: config.frames)
      let loss = buildBatchedLoss(
        student: student, target: target, batch: batch, config: config, smooth: smooth)
      let lossValues = try loss.backward(frames: config.frames)
      let meanLoss = lossValues.reduce(0, +) / Float(batch)
      meanLossTrace.append(meanLoss)
      guard meanLoss.isFinite else {
        print("[\(label)] stopping at step \(step): non-finite loss")
        break
      }
      if config.cosineLRDecay {
        // Anneal within each phase, matching the production trainer's shape.
        let phaseLen = smooth ? smoothSteps : steps
        let phaseStep = smooth ? step : step - smoothSteps
        let progress = Float(phaseStep) / Float(max(1, phaseLen - 1))
        opt.lrScale = 0.05 + 0.95 * 0.5 * (1.0 + Foundation.cos(Float.pi * progress))
      }
      opt.step(params: z, config: config, gradClip: config.gradClip)
      opt.zeroGrad(params: z)
      if logEvery > 0 && (step % logEvery == 0 || step == totalSteps - 1) {
        let elapsed = Date().timeIntervalSince(start)
        print(
          "[\(label)] step=\(step)\(smooth ? " (smooth)" : "")"
            + " meanLoss=\(String(format: "%.6f", meanLoss))"
            + " elapsed=\(String(format: "%.1f", elapsed))s")
        var snapshot: [String: [Float]] = [:]
        for (name, tensor) in z {
          let spec = KickParamSpecs.byName[name]!
          snapshot[name] = (tensor.getData() ?? []).map { spec.inverse($0) }
        }
        paramTrace.append(snapshot)
      }
    }
    let elapsed = Date().timeIntervalSince(start)
    let lanes = naturalLaneValues(z: z, base: laneInits[0], batch: batch)
    return TrajectoryResult(
      lanes: lanes,
      meanLossTrace: meanLossTrace,
      secondsPerStep: elapsed / Double(max(1, totalSteps)),
      paramTrace: paramTrace)
  }

  // MARK: - Forward render + CPU scoring (selection metric)

  /// Renders every lane with the verified forward batched voice
  /// (BatchBench.buildAudio) and scores each against the target with the
  /// independent CPU mrstft scorer.
  static func scoreLanes(
    lanes: [PatchValues], base: PatchValues, scorer: CPUSpectralScorer,
    config: SynthIDConfig
  ) throws -> [Float] {
    let batch = lanes.count
    LazyGraphContext.reset()
    config.applyRuntime()
    let params = BatchBench.makeParams(batchSize: batch)
    let audio = BatchBench.buildAudio(params: params, config: config)
    params.update(candidates: lanes.map { SubtractiveCandidate($0.clamped()) })
    let flat = try audio.realize(frames: config.frames)
    let deinterleaved = BatchBench.deinterleave(flat, frames: config.frames, batchSize: batch)
    var scores = [Float](repeating: 0, count: batch)
    scores.withUnsafeMutableBufferPointer { buf in
      DispatchQueue.concurrentPerform(iterations: batch) { i in
        buf[i] = scorer.score(deinterleaved[i])
      }
    }
    return scores
  }

  // MARK: - Jitter

  /// Gaussian jitter in transformed coordinates, sigma as a fraction of each
  /// param's transformed span (the same coordinate system basin-search v2
  /// resamples in). Clamped to the bounds with the standard 0.1% inset.
  static func jittered(
    _ values: PatchValues, sigmaFraction: Float, rng: inout SplitMix64
  ) -> PatchValues {
    var out = values
    for name in paramNames {
      let spec = KickParamSpecs.byName[name]!
      let bounds = spec.transformedBounds
      let span = bounds.max - bounds.min
      let inset = span * 0.001
      let (g, _) = BasinSearch.gaussianPair(&rng)
      let z = spec.transform(values[name]) + g * sigmaFraction * span
      out[name] = spec.inverse(Swift.min(bounds.max - inset, Swift.max(bounds.min + inset, z)))
    }
    return out
  }

  // MARK: - Gradient probes

  static func gradsFor(
    laneInits: [PatchValues], targetSamples: [Float], config: SynthIDConfig
  ) throws -> [String: [Float]] {
    LazyGraphContext.reset()
    config.applyRuntime()
    let z = makeZ(lanes: laneInits)
    let targetTensor = Tensor(targetSamples)
    let onesTensor = Tensor([Float](repeating: 1, count: laneInits.count))
    let student = buildStudent(z: z, batch: laneInits.count, config: config)
    let target = onesTensor * targetTensor.toSignal(maxFrames: config.frames)
    let loss = buildBatchedLoss(
      student: student, target: target, batch: laneInits.count, config: config, smooth: false)
    _ = try loss.backward(frames: config.frames)
    var grads: [String: [Float]] = [:]
    for (name, tensor) in z {
      grads[name] = tensor.grad?.getData() ?? []
    }
    return grads
  }

  static func probeGrads(
    laneInits: [PatchValues], targetSamples: [Float], config: SynthIDConfig
  ) throws {
    let batch = laneInits.count
    let full = try gradsFor(laneInits: laneInits, targetSamples: targetSamples, config: config)
    DGenConfig.kernelOutputPath = nil  // keep only the batched compile's dump
    var worst: Float = 0
    var worstDesc = ""
    var singles: [String: [Float]] = [:]
    for lane in 0..<batch {
      let single = try gradsFor(
        laneInits: [laneInits[lane]], targetSamples: targetSamples, config: config)
      for name in paramNames {
        let b = full[name]?[lane] ?? .nan
        let s = single[name]?[0] ?? .nan
        singles[name, default: []].append(s)
        let rel = abs(b - s) / max(abs(s), 1e-6)
        if rel > worst {
          worst = rel
          worstDesc = "lane \(lane) \(name): batched \(b) vs single \(s)"
        }
      }
    }
    print("probe-grads: worst rel diff \(worst)  (\(worstDesc))")
    for name in paramNames {
      print("  \(name): batched \(full[name] ?? []) vs singles \(singles[name] ?? [])")
    }
  }

  /// [1]-batched tensor-path gradients vs the production scalar
  /// TrainableKickParams path at the same parameter point. Validates the full
  /// batched topology's adjoints (polyblep/mod/comparison/biquad/VCA) against
  /// the path every published result was trained with.
  static func probeScalar(
    values: PatchValues, targetSamples: [Float], config: SynthIDConfig
  ) throws {
    let tensorGrads = try gradsFor(
      laneInits: [values], targetSamples: targetSamples, config: config)
    DGenConfig.kernelOutputPath = nil

    LazyGraphContext.reset()
    config.applyRuntime()
    let targetTensor = Tensor(targetSamples)
    let params = TrainableKickParams(
      initial: values, trainable: true, freezePitch: true)
    let synth = KickVoice.build(params: params, config: config)
    let target = targetTensor.toSignal(maxFrames: config.frames)
    let loss = SynthIDLosses.multiResolutionSpectralLoss(
      synth: synth, target: target, config: config)
    _ = try loss.backward(frames: config.frames)

    var worst: Float = 0
    var worstName = ""
    for name in paramNames {
      let t = tensorGrads[name]?[0] ?? .nan
      let s = params.transformedParams[name]?.grad?.data ?? .nan
      let rel = abs(t - s) / max(abs(s), 1e-6)
      if rel > worst {
        worst = rel
        worstName = name
      }
      print(
        "  \(name): tensor[1] \(String(format: "%12.5e", t))"
          + "  scalar \(String(format: "%12.5e", s))"
          + "  rel \(String(format: "%.3e", rel))")
    }
    print("probe-scalar: worst rel diff \(worst) (\(worstName))")
  }

  // MARK: - CLI entry point

  static func run(options: [String: String]) throws {
    var config = try loadConfig(url: options["config"].map { URL(fileURLWithPath: $0) })
    config.profile = "subtractive-bass"
    try config.applyCLI(options)
    config.profile = "subtractive-bass"
    config.enableNoiseFilter = true
    config.applyRuntime()
    DGenConfig.backend = .metal
    if let dump = ProcessInfo.processInfo.environment["BATCH_REFINE_KERNEL_DUMP"] {
      DGenConfig.kernelOutputPath = dump
    }

    guard let targetPath = options["target"] else {
      throw SynthIDError.message("batch-refine requires --target <wav>")
    }
    let mode = options["mode"] ?? "polish"
    let steps = try options["steps"].map { try parseInt($0, "--steps") } ?? 300
    let smoothSteps = try options["smooth-steps"].map { try parseInt($0, "--smooth-steps") } ?? 0
    let batchOpt = try options["batch"].map { try parseInt($0, "--batch") } ?? 64
    let jitterSigma = try options["jitter"].map { try parseFloat($0, "--jitter") } ?? 0.05
    let seed = try options["seed"].map { try parseInt($0, "--seed") } ?? 6
    let logEvery = config.logEvery > 0 ? max(1, config.logEvery) : 25

    let (rawTarget, _) = try AudioFile.load(url: URL(fileURLWithPath: targetPath))
    let targetSamples = fitOrPad(
      peakNormalized(rawTarget, peak: config.peakNormalizeTo), frames: config.frames)
    let scorer = try CPUSpectralScorer(target: targetSamples)
    var rng = SplitMix64(seed: UInt64(seed) &* 0x9E37_79B9_7F4A_7C15 &+ 0xBA7C4)

    func loadInit(_ key: String, fallback: String? = nil) throws -> PatchValues {
      guard let path = options[key] ?? fallback else {
        throw SynthIDError.message("batch-refine --mode \(mode) requires --\(key) <json>")
      }
      return try loadPatchValues(from: URL(fileURLWithPath: path))
    }

    let outDir = options["out"].map { URL(fileURLWithPath: $0) }
    if let outDir { try ensureDirectory(outDir) }

    // Production training loss for selected lanes (the gate metric's
    // numerator; ratio printed when --baseline <coldBaselineLoss> is given).
    let baseline = try options["baseline"].map { try parseFloat($0, "--baseline") }
    func productionLoss(_ values: PatchValues) throws -> Float {
      try SynthIDTrainer(config: config).evaluateLoss(
        values: values, targetSamples: rawTarget)
    }
    func gateSuffix(_ loss: Float) -> String {
      guard let baseline else { return "" }
      return String(format: " ratio=%.4f%%", 100 * loss / baseline)
    }

    print("=== batch-refine mode=\(mode) frames=\(config.frames) steps=\(steps)"
      + (smoothSteps > 0 ? " smoothSteps=\(smoothSteps)" : "") + " ===")

    switch mode {
    case "probe-grads":
      let base = try loadInit("init")
      var lanes = [base]
      while lanes.count < batchOpt {
        lanes.append(jittered(base, sigmaFraction: max(jitterSigma, 0.05), rng: &rng))
      }
      try probeGrads(laneInits: lanes, targetSamples: targetSamples, config: config)

    case "probe-scalar":
      let base = try loadInit("init")
      try probeScalar(values: base, targetSamples: targetSamples, config: config)

    case "polish":
      guard let elitesPath = options["elites"] else {
        throw SynthIDError.message("batch-refine --mode polish requires --elites <dir>")
      }
      let elitesDir = URL(fileURLWithPath: elitesPath)
      let eliteFiles = try FileManager.default.contentsOfDirectory(
        at: elitesDir, includingPropertiesForKeys: nil)
        .filter { $0.lastPathComponent.hasPrefix("elite-") && $0.pathExtension == "json" }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
      guard !eliteFiles.isEmpty else {
        throw SynthIDError.message("no elite-*.json in \(elitesDir.path)")
      }
      let elites = try eliteFiles.map { try loadPatchValues(from: $0) }
      let eliteNames = eliteFiles.map { $0.deletingPathExtension().lastPathComponent }

      // Rank elites by pre-refine CPU score so batch padding restarts the best.
      let preScores = try scoreLanes(
        lanes: elites, base: elites[0], scorer: scorer, config: config)
      let bestEliteIdx = preScores.enumerated().min { $0.element < $1.element }!.offset
      print("pre-refine elite scores:")
      for (i, s) in preScores.enumerated() {
        print("  \(eliteNames[i]): \(String(format: "%.5f", s))"
          + (i == bestEliteIdx ? "  <- best (padding source)" : ""))
      }

      // Pack elites x restarts, padding with extra restarts of the best elite
      // until B >= 32 (and >= --batch if given). Lane 0 of each elite's group
      // is the UNJITTERED elite (the serial-refinement baseline lane).
      let restarts = try options["restarts"].map { try parseInt($0, "--restarts") }
        ?? max(2, Int((Double(max(32, batchOpt)) / Double(elites.count)).rounded(.up)))
      var laneInits: [PatchValues] = []
      var laneOwner: [Int] = []  // elite index per lane (-1 = padding of best)
      for (i, elite) in elites.enumerated() {
        for r in 0..<restarts {
          laneInits.append(
            r == 0 ? elite : jittered(elite, sigmaFraction: jitterSigma, rng: &rng))
          laneOwner.append(i)
        }
      }
      while laneInits.count < max(32, batchOpt) {
        laneInits.append(jittered(elites[bestEliteIdx], sigmaFraction: jitterSigma, rng: &rng))
        laneOwner.append(bestEliteIdx)
      }
      let batch = laneInits.count
      print("packed \(elites.count) elites x \(restarts) restarts -> B=\(batch)"
        + " (jitter sigma=\(jitterSigma) of transformed span)")

      let result = try runTrajectory(
        laneInits: laneInits, targetSamples: targetSamples, config: config,
        steps: steps, smoothSteps: smoothSteps,
        laneLR: [Float](repeating: 1, count: batch), logEvery: logEvery,
        label: "polish")
      let finalScores = try scoreLanes(
        lanes: result.lanes, base: elites[0], scorer: scorer, config: config)

      print(String(format: "\n%.3f s/step for B=%d (%.4f s/lane-step)",
        result.secondsPerStep, batch, result.secondsPerStep / Double(batch)))
      print("\nper-elite results (CPU mrstft; lower is better):")
      var summary: [[String: Any]] = []
      for i in 0..<elites.count {
        let laneIdxs = (0..<batch).filter { laneOwner[$0] == i }
        let bestLane = laneIdxs.min { finalScores[$0] < finalScores[$1] }!
        let unjittered = laneIdxs.first!
        let bestProd = try productionLoss(result.lanes[bestLane])
        print("  \(eliteNames[i]): pre=\(String(format: "%.5f", preScores[i]))"
          + " unjittered=\(String(format: "%.5f", finalScores[unjittered]))"
          + " bestLane=\(String(format: "%.5f", finalScores[bestLane]))"
          + " (lane \(bestLane), \(laneIdxs.count) lanes)"
          + " prodLoss=\(String(format: "%.6f", bestProd))\(gateSuffix(bestProd))")
        if let outDir {
          try writeJSON(
            result.lanes[bestLane],
            to: outDir.appendingPathComponent("\(eliteNames[i])_best.json"))
        }
        summary.append([
          "elite": eliteNames[i],
          "preScore": preScores[i],
          "unjitteredFinalScore": finalScores[unjittered],
          "bestLaneScore": finalScores[bestLane],
          "bestLane": bestLane,
          "lanes": laneIdxs.count,
        ])
      }
      let globalBest = (0..<batch).min { finalScores[$0] < finalScores[$1] }!
      print("\nglobal best lane \(globalBest)"
        + " (\(eliteNames[laneOwner[globalBest]])):"
        + " \(String(format: "%.5f", finalScores[globalBest]))")
      if let outDir {
        try writeJSON(result.lanes[globalBest],
          to: outDir.appendingPathComponent("global_best.json"))
        let report: [String: Any] = [
          "mode": "polish", "batch": batch, "restartsPerElite": restarts,
          "steps": steps, "smoothSteps": smoothSteps, "jitterSigma": jitterSigma,
          "secondsPerStep": result.secondsPerStep,
          "meanLossTrace": result.meanLossTrace,
          "laneScores": finalScores, "laneOwner": laneOwner,
          "elites": summary,
        ]
        let data = try JSONSerialization.data(
          withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: outDir.appendingPathComponent("batch_refine_report.json"))
        print("wrote \(outDir.path)")
      }

    case "escape":
      let base = try loadInit("init")
      let batch = max(32, batchOpt)
      var laneInits: [PatchValues] = [base]  // lane 0: exact stuck init
      while laneInits.count < batch {
        laneInits.append(jittered(base, sigmaFraction: jitterSigma, rng: &rng))
      }
      print("escape: lane 0 = exact init, lanes 1..\(batch - 1) jittered"
        + " (sigma=\(jitterSigma))")
      let initScores = try scoreLanes(
        lanes: laneInits, base: base, scorer: scorer, config: config)
      let result = try runTrajectory(
        laneInits: laneInits, targetSamples: targetSamples, config: config,
        steps: steps, smoothSteps: smoothSteps,
        laneLR: [Float](repeating: 1, count: batch), logEvery: logEvery,
        label: "escape")
      let finalScores = try scoreLanes(
        lanes: result.lanes, base: base, scorer: scorer, config: config)

      print(String(format: "\n%.3f s/step for B=%d", result.secondsPerStep, batch))
      let order = (0..<batch).sorted { finalScores[$0] < finalScores[$1] }
      print("lane 0 (stall reproduction): init=\(String(format: "%.5f", initScores[0]))"
        + " final=\(String(format: "%.5f", finalScores[0]))")
      print("top 8 lanes by final CPU score:")
      for lane in order.prefix(8) {
        print("  lane \(lane): init=\(String(format: "%.5f", initScores[lane]))"
          + " final=\(String(format: "%.5f", finalScores[lane]))")
      }
      let better = (1..<batch).filter { finalScores[$0] < finalScores[0] }.count
      print("\(better)/\(batch - 1) jittered lanes beat the unjittered trajectory")
      let lane0Prod = try productionLoss(result.lanes[0])
      let bestProd = try productionLoss(result.lanes[order[0]])
      print("production loss: lane 0 \(String(format: "%.6f", lane0Prod))\(gateSuffix(lane0Prod))"
        + "  best lane \(order[0]) \(String(format: "%.6f", bestProd))\(gateSuffix(bestProd))")

      if let truePath = options["true-params"] {
        let truth = try loadPatchValues(from: URL(fileURLWithPath: truePath))
        let best = order[0]
        print("\nbest lane \(best) vs hidden truth (transformed-space abs err / span):")
        var recovered = 0
        for name in paramNames {
          let spec = KickParamSpecs.byName[name]!
          let bounds = spec.transformedBounds
          let span = bounds.max - bounds.min
          let err = abs(
            spec.transform(result.lanes[best][name]) - spec.transform(truth[name])) / span
          let ok = err <= spec.tolerance
          if ok { recovered += 1 }
          print("  \(name): lane=\(String(format: "%10.4f", result.lanes[best][name]))"
            + " truth=\(String(format: "%10.4f", truth[name]))"
            + " err=\(String(format: "%.4f", err)) \(ok ? "OK" : "MISS")")
        }
        print("recovered \(recovered)/\(paramNames.count) params within spec tolerance")
      }
      if let outDir {
        for lane in order.prefix(3) {
          try writeJSON(result.lanes[lane],
            to: outDir.appendingPathComponent(String(format: "lane-%02d.json", lane)))
        }
        let report: [String: Any] = [
          "mode": "escape", "batch": batch, "steps": steps,
          "smoothSteps": smoothSteps, "jitterSigma": jitterSigma,
          "secondsPerStep": result.secondsPerStep,
          "meanLossTrace": result.meanLossTrace,
          "initScores": initScores, "finalScores": finalScores,
        ]
        let data = try JSONSerialization.data(
          withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: outDir.appendingPathComponent("batch_refine_report.json"))
        print("wrote \(outDir.path)")
      }

    case "lr-sweep":
      let base = try loadInit("init")
      let batch = max(32, batchOpt)
      let lrMin = try options["lr-min"].map { try parseFloat($0, "--lr-min") } ?? 0.1
      let lrMax = try options["lr-max"].map { try parseFloat($0, "--lr-max") } ?? 10.0
      let laneLR = (0..<batch).map { lane -> Float in
        let f = Float(lane) / Float(max(1, batch - 1))
        return Foundation.exp(
          Foundation.log(lrMin) + f * (Foundation.log(lrMax) - Foundation.log(lrMin)))
      }
      let laneInits = [PatchValues](repeating: base, count: batch)
      print("lr-sweep: global LR multiplier \(lrMin)..\(lrMax) log-spaced over \(batch) lanes")
      print("(multiplies the production group LRs: amp \(config.ampLR * 0.1),"
        + " decay \(config.decayLR / 3.0), tone \(config.toneLR), osc \(config.toneLR * 0.1))")
      let result = try runTrajectory(
        laneInits: laneInits, targetSamples: targetSamples, config: config,
        steps: steps, smoothSteps: smoothSteps, laneLR: laneLR, logEvery: logEvery,
        label: "lr-sweep")
      let finalScores = try scoreLanes(
        lanes: result.lanes, base: base, scorer: scorer, config: config)
      print(String(format: "\n%.3f s/step for B=%d", result.secondsPerStep, batch))
      print("LR multiplier -> final CPU score:")
      for lane in 0..<batch {
        print("  x\(String(format: "%7.3f", laneLR[lane])):"
          + " \(String(format: "%.5f", finalScores[lane]))")
      }
      let best = (0..<batch).min { finalScores[$0] < finalScores[$1] }!
      print("best: x\(String(format: "%.3f", laneLR[best]))"
        + " (score \(String(format: "%.5f", finalScores[best])))")
      if let outDir {
        let report: [String: Any] = [
          "mode": "lr-sweep", "batch": batch, "steps": steps,
          "laneLR": laneLR, "finalScores": finalScores,
          "meanLossTrace": result.meanLossTrace,
          "secondsPerStep": result.secondsPerStep,
        ]
        let data = try JSONSerialization.data(
          withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: outDir.appendingPathComponent("batch_refine_report.json"))
        print("wrote \(outDir.path)")
      }

    default:
      throw SynthIDError.message(
        "unknown --mode \(mode); expected polish, escape, lr-sweep, probe-grads, or probe-scalar")
    }
  }
}
