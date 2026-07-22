// BatchBench.swift
//
// Additive-only experiment: prove that B independent candidate parameter sets
// of the `subtractive-bass` SynthID voice can be rendered + scored in ONE
// lazy graph (one compile, one dispatch cycle per evaluation), and measure
// per-candidate throughput at several batch sizes against the known serial
// baseline (~0.2-0.26 s/candidate).
//
// This file does not modify SubtractiveBassVoice, TrainableKickParams, the
// trainer, or any production loss path. It re-implements the voice's math
// once more using [B]-shaped SignalTensors so the existing scalar voice is
// left completely untouched.
//
// The tensor-biquad implementation gives every lane independent history and
// accepts [B]-shaped cutoff/resonance controls, so all twelve candidate
// parameters vary independently in the correctness and timing sweeps below.

import DGenLazy
import Foundation

// MARK: - Candidate representation

struct SubtractiveCandidate {
  var shape: Float
  var pw: Float
  var fBase: Float
  var fAmt: Float
  var fDecay: Float
  var res: Float
  var attackTime: Float
  var decayTime: Float
  var sustain: Float
  var releaseTime: Float
  var drive: Float
  var outGain: Float

  init(_ values: PatchValues) {
    shape = values.shape
    pw = values.pw
    fBase = values.fBase
    fAmt = values.fAmt
    fDecay = values.fDecay
    res = values.res
    attackTime = values.attackTime
    decayTime = values.decayTime
    sustain = values.sustain
    releaseTime = values.releaseTime
    drive = values.drive
    outGain = values.outGain
  }

  func patchValues(basedOn base: PatchValues) -> PatchValues {
    var v = base
    v.shape = shape
    v.pw = pw
    v.fBase = fBase
    v.fAmt = fAmt
    v.fDecay = fDecay
    v.res = res
    v.attackTime = attackTime
    v.decayTime = decayTime
    v.sustain = sustain
    v.releaseTime = releaseTime
    v.drive = drive
    v.outGain = outGain
    return v
  }
}

// MARK: - Batched parameter storage (created once, updated per-eval)

struct BatchedSubtractiveParams {
  let shape: Tensor
  let pw: Tensor
  let attackTime: Tensor
  let decayTime: Tensor
  let sustain: Tensor
  let releaseTime: Tensor
  let drive: Tensor
  let outGain: Tensor
  let fBase: Tensor
  let fAmt: Tensor
  let fDecay: Tensor
  let res: Tensor
  let batchSize: Int

  func update(candidates: [SubtractiveCandidate]) {
    precondition(candidates.count == batchSize)
    shape.updateDataLazily(candidates.map(\.shape))
    pw.updateDataLazily(candidates.map(\.pw))
    attackTime.updateDataLazily(candidates.map(\.attackTime))
    decayTime.updateDataLazily(candidates.map(\.decayTime))
    sustain.updateDataLazily(candidates.map(\.sustain))
    releaseTime.updateDataLazily(candidates.map(\.releaseTime))
    drive.updateDataLazily(candidates.map(\.drive))
    outGain.updateDataLazily(candidates.map(\.outGain))
    fBase.updateDataLazily(candidates.map(\.fBase))
    fAmt.updateDataLazily(candidates.map(\.fAmt))
    fDecay.updateDataLazily(candidates.map(\.fDecay))
    res.updateDataLazily(candidates.map(\.res))
  }
}

enum BatchBench {

  // MARK: PolyBLEP (SignalTensor port of SubtractiveBassVoice.polyblep*)

  private static func polyblep(_ phase: SignalTensor, dt: Signal) -> SignalTensor {
    let leftX = phase / dt
    let left = leftX * 2.0 - leftX * leftX - 1.0
    let rightX = (phase - 1.0) / dt
    let right = rightX * rightX + rightX * 2.0 + 1.0
    return (phase < dt) * left + (phase > (Signal.constant(1.0) - dt)) * right
  }

  // MARK: Batched voice construction
  //
  // Must be called AFTER LazyGraphContext.reset() (or clearComputationGraph(),
  // which refreshes `params`'s nodeIds itself) per CLAUDE.md's Tensor
  // creation-order rule. `params` is created once outside the timing loop and
  // reused across iterations; only ephemeral locals (freqTensor, t, dt, ...)
  // are recreated on every call, which is safe because they have no identity
  // that needs to survive a graph rebuild.
  static func makeParams(batchSize: Int) -> BatchedSubtractiveParams {
    let B = batchSize
    return BatchedSubtractiveParams(
      shape: Tensor([Float](repeating: 0.5, count: B)),
      pw: Tensor([Float](repeating: 0.5, count: B)),
      attackTime: Tensor([Float](repeating: 0.05, count: B)),
      decayTime: Tensor([Float](repeating: 0.2, count: B)),
      sustain: Tensor([Float](repeating: 0.5, count: B)),
      releaseTime: Tensor([Float](repeating: 0.08, count: B)),
      drive: Tensor([Float](repeating: 1.5, count: B)),
      outGain: Tensor([Float](repeating: 0.5, count: B)),
      fBase: Tensor([Float](repeating: 490, count: B)),
      fAmt: Tensor([Float](repeating: 108.55, count: B)),
      fDecay: Tensor([Float](repeating: 0.1, count: B)),
      res: Tensor([Float](repeating: 1.73, count: B)),
      batchSize: B)
  }

  static func buildAudio(
    params: BatchedSubtractiveParams, config: SynthIDConfig,
    frozen: PatchValues = PatchValues([:])
  ) -> SignalTensor {
    let B = params.batchSize
    let sr = Signal.constant(config.sampleRate)
    let t = Signal.accum(
      Signal.constant(1.0) / sr,
      reset: 0.0,
      min: 0.0,
      max: Float(config.frames + 1) / config.sampleRate + 1.0)

    // f0 frozen per-target (PatchValues.subF0, default 110 Hz) for every
    // candidate, exactly as the serial voice does. Routed through a [B]
    // frequency tensor -> Signal.statefulPhasor so the batched
    // stateful-phasor path is exercised (Signal.swift:222,
    // SignalTensor.swift:104), even though every lane holds the same value.
    let freqTensor = Tensor([Float](repeating: frozen.subF0, count: B))
    let phase = Signal.statefulPhasor(freqTensor)
    let dt = (Signal.constant(frozen.subF0) / sr).clip(0.000001, 0.5)

    let saw = (phase * 2.0 - 1.0) - polyblep(phase, dt: dt)
    let clippedWidth = params.pw.clip(0.01, 0.99)
    let fallingPhase = mod(phase - clippedWidth, 1.0)
    let rawPulse = (phase < clippedWidth) * 2.0 - 1.0
    let pulse = rawPulse + polyblep(phase, dt: dt) - polyblep(fallingPhase, dt: dt)
    let oscillator = (Signal.constant(1.0) - params.shape) * saw + params.shape * pulse

    // Per-candidate filter envelope, matching the scalar voice lane by lane.
    let cutoff = params.fBase + params.fAmt
      * DGenLazy.exp((Signal.constant(0.0) - t) / params.fDecay)
    let resonance = params.res * Signal.constant(1)
    let filtered = config.enableNoiseFilter
      ? oscillator.biquad(
        cutoff: cutoff, resonance: resonance,
        gain: Signal.constant(1.0), mode: Signal.constant(0.0))
      : oscillator

    let attack = Signal.constant(1.0)
      - DGenLazy.exp((Signal.constant(0.0) - t) / params.attackTime)
    let decay = params.sustain
      + (Signal.constant(1.0) - params.sustain)
      * DGenLazy.exp((Signal.constant(0.0) - t) / params.decayTime)
    let release = Signal.constant(1.0)
      / (Signal.constant(1.0)
        + DGenLazy.exp((t - Signal.constant(frozen.subNoteOff)) / params.releaseTime))
    let driven = filtered * attack * decay * release * params.drive
    let shaped = driven / (Signal.constant(1.0) + DGenLazy.abs(driven))
    return shaped * params.outGain
  }

  // MARK: Deinterleaving

  /// SignalTensor.realize returns a flat [frames*B] array, frame-major,
  /// batch-minor: [frame0_elem0 ... frame0_elemB-1, frame1_elem0, ...].
  static func deinterleave(_ flat: [Float], frames: Int, batchSize: Int) -> [[Float]] {
    var out = [[Float]](repeating: [Float](repeating: 0, count: frames), count: batchSize)
    for f in 0..<frames {
      let base = f * batchSize
      for b in 0..<batchSize {
        out[b][f] = flat[base + b]
      }
    }
    return out
  }

  // MARK: Cheap CPU-side proxy loss (readback fallback; see report)

  /// Mean squared error against a fixed reference. This is NOT the
  /// production MR-STFT loss (which has no public per-candidate GPU
  /// readback API before its batched-mean reduction, see
  /// Sources/DGenLazy/Functions.swift:911); it stands in for "whatever
  /// CPU-side scoring a population search would run on read-back audio",
  /// cheap enough (O(N), no FFT) not to dominate the batch-size sweep.
  static func proxyLoss(_ audio: [Float], target: [Float]) -> Float {
    let n = min(audio.count, target.count)
    guard n > 0 else { return 0 }
    var sum: Float = 0
    for i in 0..<n {
      let d = audio[i] - target[i]
      sum += d * d
    }
    return sum / Float(n)
  }

  // MARK: - Deterministic candidate generation

  static func randomCandidates(
    count: Int, seed: UInt64, filterBase: SubtractiveCandidate
  ) -> [SubtractiveCandidate] {
    var rng = SplitMix64(seed: seed)
    var candidates: [SubtractiveCandidate] = []
    candidates.reserveCapacity(count)
    for _ in 0..<count {
      var c = filterBase
      c.shape = rng.uniform(0.0, 1.0)
      c.pw = rng.uniform(0.05, 0.95)
      c.fBase = Foundation.exp(rng.uniform(Foundation.log(60), Foundation.log(2000)))
      c.fAmt = rng.uniform(0, 5000)
      c.fDecay = Foundation.exp(rng.uniform(Foundation.log(0.015), Foundation.log(0.6)))
      c.res = Foundation.exp(rng.uniform(Foundation.log(0.4), Foundation.log(4.0)))
      c.attackTime = Foundation.exp(rng.uniform(Foundation.log(0.005), Foundation.log(0.3)))
      c.decayTime = Foundation.exp(rng.uniform(Foundation.log(0.02), Foundation.log(1.0)))
      c.sustain = rng.uniform(0.1, 0.9)
      c.releaseTime = Foundation.exp(rng.uniform(Foundation.log(0.012), Foundation.log(0.5)))
      c.drive = Foundation.exp(rng.uniform(Foundation.log(0.3), Foundation.log(6.0)))
      c.outGain = rng.uniform(0.1, 1.5)
      candidates.append(c)
    }
    return candidates
  }

  // MARK: - Timing helper

  static func elapsedSeconds(_ start: DispatchTime, _ end: DispatchTime) -> Double {
    Double(end.uptimeNanoseconds - start.uptimeNanoseconds) / 1e9
  }

  // MARK: - Correctness gate (B = 8)

  struct CorrectnessResult {
    var maxAbsDiffs: [Float]
    var serialLosses: [Float]
    var batchedLosses: [Float]
  }

  static func runCorrectnessGate(
    config inputConfig: SynthIDConfig, seedDir: URL
  ) throws -> CorrectnessResult {
    var config = inputConfig
    config.enableNoiseFilter = true
    config.applyRuntime()
    let trueValues = try loadPatchValues(
      from: seedDir.appendingPathComponent("true_params.json"))
    let initialValues = try loadPatchValues(
      from: seedDir.appendingPathComponent("initial_params.json"))
    let recoveredValues = try loadPatchValues(
      from: seedDir.appendingPathComponent("recovered_params.json"))
    let (targetSamples, _) = try AudioFile.load(
      url: seedDir.appendingPathComponent("target.wav"))

    var candidates = [
      SubtractiveCandidate(trueValues),
      SubtractiveCandidate(initialValues),
      SubtractiveCandidate(recoveredValues),
    ]
    let perturbed = randomCandidates(
      count: 5, seed: 4242, filterBase: candidates[0])
    candidates.append(contentsOf: perturbed)
    precondition(candidates.count == 8)

    // Serial renders: one LazyGraphContext.reset() + realize() per candidate,
    // using the existing production SubtractiveBassVoice unmodified.
    var serialAudio: [[Float]] = []
    for candidate in candidates {
      LazyGraphContext.reset()
      config.applyRuntime()
      let values = candidate.patchValues(basedOn: trueValues)
      let params = TrainableKickParams(initial: values, trainable: false, freezePitch: true)
      let sig = SubtractiveBassVoice.build(
        params: params.subtractiveBassSignals, config: config)
      let audio = try sig.realize(frames: config.frames)
      serialAudio.append(audio)
    }

    // Batched render: ONE graph, ONE compile, ONE dispatch for all 8.
    LazyGraphContext.reset()
    config.applyRuntime()
    let batchedParams = makeParams(batchSize: 8)
    let audioTensor = buildAudio(params: batchedParams, config: config)
    batchedParams.update(candidates: candidates)
    let flat = try audioTensor.realize(frames: config.frames)
    let batchedAudio = deinterleave(flat, frames: config.frames, batchSize: 8)

    var maxAbsDiffs: [Float] = []
    var serialLosses: [Float] = []
    var batchedLosses: [Float] = []
    for b in 0..<8 {
      var maxDiff: Float = 0
      for f in 0..<config.frames {
        maxDiff = Swift.max(maxDiff, abs(serialAudio[b][f] - batchedAudio[b][f]))
      }
      maxAbsDiffs.append(maxDiff)
      serialLosses.append(proxyLoss(serialAudio[b], target: targetSamples))
      batchedLosses.append(proxyLoss(batchedAudio[b], target: targetSamples))
    }
    return CorrectnessResult(
      maxAbsDiffs: maxAbsDiffs, serialLosses: serialLosses, batchedLosses: batchedLosses)
  }

  // MARK: - Timing sweep

  struct TimingResult {
    var batchSize: Int
    var compileSeconds: Double
    var steadyStatePerEvalSeconds: Double
    var perCandidateMillis: Double
    var evalsPerHour: Double
    var iterSeconds: [Double]
  }

  static func runTimingSweep(
    config: SynthIDConfig, batchSizes: [Int], iterations: Int, seedDir: URL
  ) throws -> [TimingResult] {
    config.applyRuntime()
    let trueValues = try loadPatchValues(
      from: seedDir.appendingPathComponent("true_params.json"))
    let filterBase = SubtractiveCandidate(trueValues)

    var results: [TimingResult] = []
    for B in batchSizes {
      LazyGraphContext.reset()
      config.applyRuntime()
      let params = makeParams(batchSize: B)
      let audio = buildAudio(params: params, config: config)

      // Iteration 0: includes Metal kernel compilation (first-touch cost).
      let c0 = randomCandidates(count: B, seed: UInt64(9000 + B), filterBase: filterBase)
      params.update(candidates: c0)
      let compileStart = DispatchTime.now()
      _ = try audio.realize(frames: config.frames)
      let compileEnd = DispatchTime.now()
      let compileSeconds = elapsedSeconds(compileStart, compileEnd)

      // Steady-state iterations: KEEP the same compiled `audio` SignalTensor
      // (do not call clearComputationGraph()/rebuild between iterations).
      // We found (see report) that clearComputationGraph() + rebuild hits a
      // pre-existing DGenLazy caching bug for B>1: when fullCompilationCache
      // fingerprint-matches the prior build and skips recompilation, its
      // cached nodeToTensor map still points at the PRIOR graph's nodeId,
      // so SignalTensor.realize()'s tensorId lookup misses and silently
      // falls back to the scalar per-frame sum-output path (wrong shape:
      // [frames] instead of [frames*B]). Reusing the same `audio` object
      // avoids that mismatch (nodeId is stable) at the cost of `.sum`/
      // `.output` nodes accumulating on the graph every iteration; that
      // overhead is measured, not hidden - see the report's discussion of
      // whether this is dominating step latency at higher B.
      var iterSeconds: [Double] = []
      for i in 0..<iterations {
        let candidates = randomCandidates(
          count: B, seed: UInt64(10_000 + B * 1000 + i), filterBase: filterBase)
        let start = DispatchTime.now()
        params.update(candidates: candidates)
        let flat = try audio.realize(frames: config.frames)
        // Cheap CPU-side per-candidate scoring, included in the timed
        // region (readback + reduce is part of the deployment shape).
        let deinterleaved = deinterleave(flat, frames: config.frames, batchSize: B)
        for cand in deinterleaved {
          _ = proxyLoss(cand, target: cand)  // O(N) placeholder scoring cost
        }
        let end = DispatchTime.now()
        iterSeconds.append(elapsedSeconds(start, end))
      }

      // Use the first min(5, iterations) samples: the graph-node-accumulation
      // workaround above (no clearComputationGraph() between iterations,
      // to dodge the tensorId/fullCompilationCache mismatch bug) causes
      // step latency to drift upward after several iterations as
      // .sum/.output nodes pile up; the early iterations are the closest
      // proxy to true steady-state dispatch cost. Full per-iteration times
      // are still printed/reported for transparency.
      let steadyWindow = Array(iterSeconds.prefix(5))
      let steadyState = steadyWindow.reduce(0, +) / Double(max(1, steadyWindow.count))
      let perCandidateMs = (steadyState / Double(B)) * 1000.0
      let evalsPerHour = B > 0 ? 3600.0 / (steadyState / Double(B)) : 0
      results.append(
        TimingResult(
          batchSize: B, compileSeconds: compileSeconds,
          steadyStatePerEvalSeconds: steadyState, perCandidateMillis: perCandidateMs,
          evalsPerHour: evalsPerHour, iterSeconds: iterSeconds))
      let iterStr = iterSeconds.map { String(format: "%.4f", $0) }.joined(separator: ",")
      print(
        "B=\(B) compile=\(String(format: "%.4f", compileSeconds))s"
          + " steadyPerEval=\(String(format: "%.4f", steadyState))s"
          + " perCandidate=\(String(format: "%.2f", perCandidateMs))ms"
          + " evals/hr=\(String(format: "%.0f", evalsPerHour))"
          + " iters=[\(iterStr)]")
    }
    return results
  }

  // MARK: - CLI entry point

  // Minimal diagnostic probe used while isolating the correctness bug found
  // during Step 2 (batched lanes returning identical audio). Not part of the
  // benchmark's reported results; invoked only via --probe-only.
  static func debugProbe() throws {
    DGenConfig.sampleRate = 8000
    DGenConfig.defaultFrameCount = 64
    DGenConfig.maxFrameCount = 64
    LazyGraphContext.reset()
    let B = 4
    let pw = Tensor([Float]([0.1, 0.3, 0.6, 0.9]))
    let freqTensor = Tensor([Float](repeating: 110.0, count: B))
    let phase = Signal.statefulPhasor(freqTensor)
    let out = phase + pw
    let flat = try out.realize(frames: 64)
    print("probe1: count=\(flat.count) expected=\(64 * B)")
    print("probe1: frame0 = \(Array(flat.prefix(B)))")
    print("probe1: frame1 = \(Array(flat[B..<(2 * B)]))")

    // probe2: full buildAudio pipeline, with and without the biquad filter,
    // to isolate whether the filter or something upstream collapses lanes.
    var config = SynthIDConfig.default
    config.profile = "subtractive-bass"
    config.frames = 256
    config.sampleRate = 8000
    for (filterOn, B) in [(true, 1), (true, 2), (true, 4)] {
      config.enableNoiseFilter = filterOn
      LazyGraphContext.reset()
      config.applyRuntime()
      let params = makeParams(batchSize: B)
      let audio = buildAudio(params: params, config: config)
      let candidates = [
        SubtractiveCandidate.init(PatchValues.midpoint),
      ]
      _ = candidates
      var c0 = SubtractiveCandidate(PatchValues.midpoint)
      c0.fBase = 490; c0.fAmt = 108.55; c0.fDecay = 0.1; c0.res = 1.73
      var cs: [SubtractiveCandidate] = []
      for i in 0..<B {
        var c = c0
        c.shape = Float(i) * 0.25 + 0.1
        c.pw = Float(i) * 0.2 + 0.1
        cs.append(c)
      }
      params.update(candidates: cs)
      let flat2 = try audio.realize(frames: config.frames)
      print(
        "probe2 filterOn=\(filterOn) B=\(B): count=\(flat2.count) expected=\(config.frames * B)")
      if flat2.count >= 2 * B {
        print("  frame0=\(Array(flat2.prefix(B))) frame1=\(Array(flat2[B..<(2 * B)]))")
      }
      let mid = 100 * B
      if flat2.count >= mid + B {
        print("  frame100=\(Array(flat2[mid..<(mid + B)]))")
      }
      let mid2 = 200 * B
      if flat2.count >= mid2 + B {
        print("  frame200=\(Array(flat2[mid2..<(mid2 + B)]))")
      }
    }

    // probe3: isolate biquad in isolation - simple noise/sine through
    // SignalTensor.biquad([1]) vs Signal.biquad, with a Signal.param cutoff.
    LazyGraphContext.reset()
    config.applyRuntime()
    let cutoffParam = Signal.param(490.0)
    let resParam = Signal.param(1.73)
    let sinIn = DGenLazy.sin(
      Signal.accum(Signal.constant(2.0 * Float.pi * 220.0 / config.sampleRate),
        reset: 0.0, min: 0.0, max: 1e9))
    let filteredScalar = sinIn.biquad(
      cutoff: cutoffParam, resonance: resParam, gain: Signal.constant(1.0),
      mode: Signal.constant(0.0))
    let scalarOut = try filteredScalar.realize(frames: 64)
    print("probe3 scalar biquad: \(Array(scalarOut.suffix(8)))")

    LazyGraphContext.reset()
    config.applyRuntime()
    let cutoffParam2 = Signal.param(490.0)
    let resParam2 = Signal.param(1.73)
    let sinInT = Tensor([Float(1.0)]) * DGenLazy.sin(
      Signal.accum(Signal.constant(2.0 * Float.pi * 220.0 / config.sampleRate),
        reset: 0.0, min: 0.0, max: 1e9))
    let filteredTensor = sinInT.biquad(
      cutoff: cutoffParam2, resonance: resParam2, gain: Signal.constant(1.0),
      mode: Signal.constant(0.0))
    let tensorOut = try filteredTensor.realize(frames: 64)
    print("probe3 tensor[1] biquad: \(Array(tensorOut.suffix(8)))")

    // probe4: same tensor[1] biquad but at production sample rate (44100)
    // and with the actual buildAudio-shaped cutoff (fBase + fAmt*exp(-t/fDecay))
    // instead of a bare Signal.param, to see whether the zero collapse is
    // sample-rate-specific or cutoff-expression-specific.
    DGenConfig.sampleRate = 44100
    DGenConfig.defaultFrameCount = 2048
    DGenConfig.maxFrameCount = 2048
    LazyGraphContext.reset()
    let sr2 = Signal.constant(Float(44100.0))
    let t2 = Signal.accum(Signal.constant(1.0) / sr2, reset: 0.0, min: 0.0, max: 1.0)
    let fBase2 = Signal.param(490.0)
    let fAmt2 = Signal.param(108.55)
    let fDecay2 = Signal.param(0.1)
    let res2 = Signal.param(1.73)
    let cutoff2 = fBase2 + fAmt2 * DGenLazy.exp((Signal.constant(0.0) - t2) / fDecay2)
    let sinIn2 = Tensor([Float(1.0)]) * DGenLazy.sin(
      Signal.accum(Signal.constant(2.0 * Float.pi * 220.0 / 44100.0),
        reset: 0.0, min: 0.0, max: 1e9))
    let filtered2 = sinIn2.biquad(
      cutoff: cutoff2, resonance: res2, gain: Signal.constant(1.0), mode: Signal.constant(0.0))
    let out2 = try filtered2.realize(frames: 2048)
    print("probe4 44.1k tensor[1] biquad w/ real cutoff expr: \(Array(out2.suffix(8)))")
    print("  max abs = \(out2.map { abs($0) }.max() ?? -1)")

    // probe5: identical but via Signal (scalar) biquad for direct A/B.
    LazyGraphContext.reset()
    let sr3 = Signal.constant(Float(44100.0))
    let t3 = Signal.accum(Signal.constant(1.0) / sr3, reset: 0.0, min: 0.0, max: 1.0)
    let fBase3 = Signal.param(490.0)
    let fAmt3 = Signal.param(108.55)
    let fDecay3 = Signal.param(0.1)
    let res3 = Signal.param(1.73)
    let cutoff3 = fBase3 + fAmt3 * DGenLazy.exp((Signal.constant(0.0) - t3) / fDecay3)
    let sinIn3 = DGenLazy.sin(
      Signal.accum(Signal.constant(2.0 * Float.pi * 220.0 / 44100.0),
        reset: 0.0, min: 0.0, max: 1e9))
    let filtered3 = sinIn3.biquad(
      cutoff: cutoff3, resonance: res3, gain: Signal.constant(1.0), mode: Signal.constant(0.0))
    let out3 = try filtered3.realize(frames: 2048)
    print("probe5 44.1k SCALAR biquad w/ real cutoff expr: \(Array(out3.suffix(8)))")
    print("  max abs = \(out3.map { abs($0) }.max() ?? -1)")
  }

  static func run(options: [String: String]) throws {
    if options.keys.contains("probe-only") {
      try debugProbe()
      return
    }
    var config = try loadConfig(url: options["config"].map { URL(fileURLWithPath: $0) })
    config.profile = "subtractive-bass"
    try config.applyCLI(options)
    config.profile = "subtractive-bass"
    config.enableNoiseFilter = true
    config.applyRuntime()

    let seedDirPath = options["seed-dir"]
      ?? "output/e1_subtractive_fresh_seeds_6_7/seed-6"
    let seedDir = URL(fileURLWithPath: seedDirPath)
    let outPath = options["out"] ?? "output/batch_bench"
    let outDir = URL(fileURLWithPath: outPath)
    try ensureDirectory(outDir)

    let batchSizes = try options["batch-sizes"].map { try parseIntList($0, "--batch-sizes") }
      ?? [1, 8, 32, 128, 256]
    let iterations = try options["iters"].map { try parseInt($0, "--iters") } ?? 20

    print("=== Step 2: correctness gate (B=8) ===")
    let correctness = try runCorrectnessGate(config: config, seedDir: seedDir)
    let maxDiffOverall = correctness.maxAbsDiffs.max() ?? 0
    for (i, diff) in correctness.maxAbsDiffs.enumerated() {
      print(
        "  candidate \(i): maxAbsDiff=\(String(format: "%.3e", diff))"
          + " serialLoss=\(String(format: "%.6e", correctness.serialLosses[i]))"
          + " batchedLoss=\(String(format: "%.6e", correctness.batchedLosses[i]))")
    }
    print("  overall max abs diff = \(String(format: "%.3e", maxDiffOverall))")
    let correctnessPass = maxDiffOverall < 1e-4

    print("\n=== Step 3: timing sweep ===")
    let timing = try runTimingSweep(
      config: config, batchSizes: batchSizes, iterations: iterations, seedDir: seedDir)

    var report: [String: Any] = [:]
    report["filterEnabled"] = true
    report["correctnessPass"] = correctnessPass
    report["maxAbsDiffOverall"] = maxDiffOverall
    report["perCandidateMaxAbsDiff"] = correctness.maxAbsDiffs
    report["serialLosses"] = correctness.serialLosses
    report["batchedLosses"] = correctness.batchedLosses
    report["timing"] = timing.map { r -> [String: Any] in
      [
        "batchSize": r.batchSize,
        "compileSeconds": r.compileSeconds,
        "steadyStatePerEvalSeconds": r.steadyStatePerEvalSeconds,
        "perCandidateMillis": r.perCandidateMillis,
        "evalsPerHour": r.evalsPerHour,
      ]
    }
    let data = try JSONSerialization.data(
      withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
    try data.write(to: outDir.appendingPathComponent("batch_bench_report.json"))
    print("\nwrote=\(outDir.appendingPathComponent("batch_bench_report.json").path)")
    print("correctnessPass=\(correctnessPass)")
  }
}
