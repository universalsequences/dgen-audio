import DGenLazy
import Darwin
import Foundation

enum SynthIDCLI {
  static func run() throws {
    var args = Array(CommandLine.arguments.dropFirst())
    guard !args.isEmpty else {
      printUsage()
      return
    }

    let command = args.removeFirst()
    let parsed = try parseOptions(args)
    switch command {
    case "render":
      try render(options: parsed)
    case "train":
      try train(options: parsed)
    case "rung1":
      try rung1(options: parsed)
    case "probe":
      try probe(options: parsed)
    case "rung2":
      try rung2(options: parsed)
    case "rung3":
      try rung3(options: parsed)
    case "help", "--help", "-h":
      printUsage()
    default:
      throw SynthIDError.message("unknown command \(command)")
    }
  }

  private static func render(options: [String: String]) throws {
    guard let paramsPath = options["params"], let outPath = options["out"] else {
      throw SynthIDError.message("render requires --params <json> and --out <wav>")
    }
    var config = try loadConfig(url: options["config"].map { URL(fileURLWithPath: $0) })
    try config.applyCLI(options)
    config.applyRuntime()

    let params = try loadPatchValues(from: URL(fileURLWithPath: paramsPath))
    let out = URL(fileURLWithPath: outPath)
    try ensureDirectory(out.deletingLastPathComponent())
    try KickVoice.renderToWav(values: params, config: config, out: out)
    print("wrote=\(out.path)")
  }

  private static func train(options: [String: String]) throws {
    guard let targetPath = options["target"], let outPath = options["out"] else {
      throw SynthIDError.message("train requires --target <wav> and --out <dir>")
    }
    var config = try loadConfig(url: options["config"].map { URL(fileURLWithPath: $0) })
    try config.applyCLI(options)
    config.applyRuntime()

    let outDir = URL(fileURLWithPath: outPath)
    try ensureDirectory(outDir)
    let targetURL = URL(fileURLWithPath: targetPath)
    let (samples, sampleRate) = try AudioFile.load(url: targetURL)
    if abs(sampleRate - config.sampleRate) > 0.5 {
      print("warning: target sampleRate=\(sampleRate) config sampleRate=\(config.sampleRate); no resampling is applied")
    }
    let trueParams = try options["params"].map { try loadPatchValues(from: URL(fileURLWithPath: $0)) }
    try AudioFile.save(
      url: outDir.appendingPathComponent("target.wav"),
      samples: fitOrPad(peakNormalized(samples, peak: config.peakNormalizeTo), frames: config.frames),
      sampleRate: config.sampleRate)

    if let paramName = options["fdcheck"] {
      let result = try SynthIDTrainer(config: config).fdcheck(
        paramName: paramName,
        targetSamples: samples,
        initial: trueParams,
        outDir: outDir)
      print(
        "fdcheck param=\(result.paramName) baseLoss=\(String(format: "%.6e", result.baseLoss)) fd=\(String(format: "%.6e", result.finiteDifferenceGrad)) autograd=\(String(format: "%.6e", result.autogradGrad)) relErr=\(String(format: "%.6e", result.relativeError))"
      )
      return
    }

    let result = try SynthIDTrainer(config: config).train(
      targetSamples: samples,
      outDir: outDir,
      trueParams: trueParams)
    try renderLearnedAndReport(
      result: result,
      trueParams: trueParams,
      targetSamples: samples,
      config: config,
      outDir: outDir)
  }

  private static func rung1(options: [String: String]) throws {
    guard let outPath = options["out"] else {
      throw SynthIDError.message("rung1 requires --out <dir>")
    }
    var config = try loadConfig(url: options["config"].map { URL(fileURLWithPath: $0) })
    try config.applyCLI(options)
    config.rung = 1
    config.applyRuntime()

    let seeds = try options["seeds"].map { try parseIntList($0, "--seeds") }
      ?? [Int(config.seed)]
    let root = URL(fileURLWithPath: outPath)
    try ensureDirectory(root)

    var passed = 0
    for seed in seeds {
      let seedDir = seeds.count == 1 ? root : root.appendingPathComponent("seed-\(seed)")
      try ensureDirectory(seedDir)
      var seedConfig = config
      seedConfig.seed = UInt64(seed)
      let sampledParams = PatchValues.sample(seed: UInt64(seed))
      var trueParams = sampledParams
      try writeJSON(sampledParams, to: seedDir.appendingPathComponent("sampled_params.json"))

      let rawTarget = try KickVoice.render(
        values: sampledParams,
        config: seedConfig,
        parameterBacked: true)
      let normalizationScale = peakNormalizationScale(rawTarget, peak: seedConfig.peakNormalizeTo)
      let normalizedOutGain = trueParams.outGain * normalizationScale
      let outGainSpec = KickParamSpecs.byName["outGain"]!
      guard normalizedOutGain >= outGainSpec.min && normalizedOutGain <= outGainSpec.max else {
        throw SynthIDError.message(
          "rung1 seed \(seed) normalization would set outGain=\(normalizedOutGain), outside \(outGainSpec.min)...\(outGainSpec.max)")
      }
      trueParams.outGain = normalizedOutGain
      try writeJSON(trueParams, to: seedDir.appendingPathComponent("true_params.json"))

      let target = rawTarget.map { $0 * normalizationScale }
      try AudioFile.save(
        url: seedDir.appendingPathComponent("target.wav"),
        samples: target,
        sampleRate: seedConfig.sampleRate)

      var bestResult: TrainingRunResult?
      var bestRestartDir: URL?
      var allResults: [TrainingRunResult] = []
      for restart in 0..<max(1, seedConfig.restarts) {
        let restartDir = seedDir.appendingPathComponent("restart-\(restart)")
        let result = try SynthIDTrainer(config: seedConfig).train(
          targetSamples: target,
          outDir: restartDir,
          trueParams: trueParams,
          restartIndex: restart)
        allResults.append(result)
        if bestResult == nil || result.bestLoss < bestResult!.bestLoss {
          bestResult = result
          bestRestartDir = restartDir
        }
      }
      guard var result = bestResult else {
        throw SynthIDError.message("no rung1 result for seed \(seed)")
      }

      // Cross-restart recombination: restarts often solve different subspaces
      // (one nails pitch, another nails the click). Greedily stitch subspaces
      // across restarts, keeping any swap that lowers the audio loss, then
      // fine-tune the stitched candidate. Selection is by audio loss only.
      if allResults.count > 1 {
        let trainer = SynthIDTrainer(config: seedConfig)
        let subspaces: [[String]] = [
          ["clickFreq", "clickAmp", "clickDecay"],
          ["fStart", "fEnd", "pitchDecay"],
          ["noiseAmp", "noiseDecay"],
          ["bodyAmp", "drive", "outGain"],
        ]
        var stitched = result.recovered
        var stitchedLoss = try trainer.evaluateLoss(values: stitched, targetSamples: target)
        var improved = false
        for donor in allResults {
          for subspace in subspaces {
            var candidate = stitched
            for name in subspace { candidate[name] = donor.recovered[name] }
            let loss = try trainer.evaluateLoss(values: candidate, targetSamples: target)
            if loss < stitchedLoss {
              stitched = candidate
              stitchedLoss = loss
              improved = true
            }
          }
        }
        // clickFreq line search: the click is a milliseconds-long broadband
        // burst, so its loss landscape is rippled and gradient descent stalls
        // in side lobes (observed: every restart stuck at ~1550 Hz vs true
        // 2127 Hz). A coarse global search on this one axis, judged by the
        // same audio loss, then fine-tuned below, is restart-style mitigation.
        let cfSpec = KickParamSpecs.byName["clickFreq"]!
        let cfSteps = 24
        for i in 0...cfSteps {
          let t = Float(i) / Float(cfSteps)
          let cf = cfSpec.min * Foundation.exp(t * Foundation.log(cfSpec.max / cfSpec.min))
          var candidate = stitched
          candidate.clickFreq = cf
          let loss = try trainer.evaluateLoss(values: candidate, targetSamples: target)
          if loss < stitchedLoss {
            stitched = candidate
            stitchedLoss = loss
            improved = true
          }
        }
        if improved {
          var tuneConfig = seedConfig
          tuneConfig.epochs = max(200, seedConfig.epochs / 3)
          let stitchDir = seedDir.appendingPathComponent("restart-stitch")
          let tuned = try SynthIDTrainer(config: tuneConfig).train(
            targetSamples: target,
            outDir: stitchDir,
            trueParams: trueParams,
            initialOverride: stitched)
          print(
            "  stitch: base=\(String(format: "%.5f", result.bestLoss)) stitched=\(String(format: "%.5f", stitchedLoss)) tuned=\(String(format: "%.5f", tuned.bestLoss))"
          )
          if tuned.bestLoss < result.bestLoss {
            // Keep the original restart's init loss so the §7.1 ratio still
            // compares against a genuine cold start, not the stitched warm start.
            result = TrainingRunResult(
              recovered: tuned.recovered,
              initial: result.initial,
              pitchFit: result.pitchFit,
              initLoss: result.initLoss,
              bestLoss: tuned.bestLoss,
              bestEpoch: tuned.bestEpoch,
              losses: result.losses + tuned.losses)
            bestRestartDir = stitchDir
          }
        }
      }
      if let bestRestartDir {
        try copyIfPresent(
          from: bestRestartDir.appendingPathComponent("checkpoint.json"),
          to: seedDir.appendingPathComponent("checkpoint.json"))
        try copyIfPresent(
          from: bestRestartDir.appendingPathComponent("loss_curve.csv"),
          to: seedDir.appendingPathComponent("loss_curve.csv"))
        try copyIfPresent(
          from: bestRestartDir.appendingPathComponent("initial_params.json"),
          to: seedDir.appendingPathComponent("initial_params.json"))
        try copyIfPresent(
          from: bestRestartDir.appendingPathComponent("pitch_fit.json"),
          to: seedDir.appendingPathComponent("pitch_fit.json"))
        try copyIfPresent(
          from: bestRestartDir.appendingPathComponent("pitch_points.json"),
          to: seedDir.appendingPathComponent("pitch_points.json"))
      }
      try writeJSON(result.recovered, to: seedDir.appendingPathComponent("recovered_params.json"))
      try renderLearnedAndReport(
        result: result,
        trueParams: trueParams,
        targetSamples: target,
        config: seedConfig,
        outDir: seedDir)
      let report = ReportWriter.make(
        rung: 1,
        trueParams: trueParams,
        recovered: result.recovered,
        initLoss: result.initLoss,
        finalLoss: result.bestLoss,
        includeNoiseCutoff: seedConfig.enableNoiseFilter)
      if report.pass { passed += 1 }
      print("seed=\(seed) pass=\(report.pass) bestLoss=\(String(format: "%.6f", result.bestLoss))")
    }

    let requiredPasses = seeds.count >= 5 ? 4 : seeds.count
    if passed < requiredPasses && !options.keys.contains("allow-fail") {
      throw SynthIDError.message(
        "rung1 failed: \(passed)/\(seeds.count) seeds passed; required \(requiredPasses)")
    }
  }

  // Diagnostic: isolate where the loss floor comes from by pairing streaming
  // synth signals and tensor-backed target reads in every combination.
  private static func probe(options: [String: String]) throws {
    guard let targetPath = options["target"], let paramsPath = options["params"] else {
      throw SynthIDError.message("probe requires --target <wav> and --params <json>")
    }
    var config = try loadConfig(url: options["config"].map { URL(fileURLWithPath: $0) })
    try config.applyCLI(options)
    config.applyRuntime()

    let (raw, _) = try AudioFile.load(url: URL(fileURLWithPath: targetPath))
    let target = fitOrPad(
      peakNormalized(raw, peak: config.peakNormalizeTo), frames: config.frames)
    let values = try loadPatchValues(from: URL(fileURLWithPath: paramsPath))

    func lossValue(_ build: () -> (Signal, Signal)) throws -> Float {
      LazyGraphContext.reset()
      let (a, b) = build()
      let loss = SynthIDLosses.multiResolutionSpectralLoss(synth: a, target: b, config: config)
      return try loss.backward(frames: config.frames).reduce(0, +)
    }

    let tensorTensor = try lossValue {
      let t1 = Tensor(target)
      let t2 = Tensor(target)
      return (t1.toSignal(maxFrames: config.frames), t2.toSignal(maxFrames: config.frames))
    }
    print("loss(tensor, tensor)  = \(tensorTensor)")

    let synthSynth = try lossValue {
      let p1 = TrainableKickParams(initial: values, trainable: true, freezePitch: false)
      let p2 = TrainableKickParams(initial: values, trainable: false, freezePitch: false)
      return (
        KickVoice.build(params: p1.signals, config: config),
        KickVoice.build(params: p2.signals, config: config)
      )
    }
    print("loss(synth, synth)    = \(synthSynth)")

    let synthSynthBothTrainable = try lossValue {
      let p1 = TrainableKickParams(initial: values, trainable: true, freezePitch: false)
      let p2 = TrainableKickParams(initial: values, trainable: true, freezePitch: false)
      return (
        KickVoice.build(params: p1.signals, config: config),
        KickVoice.build(params: p2.signals, config: config)
      )
    }
    print("loss(synthT, synthT)  = \(synthSynthBothTrainable)")

    LazyGraphContext.reset()
    let dp1 = TrainableKickParams(initial: values, trainable: true, freezePitch: false)
    let dp2 = TrainableKickParams(initial: values, trainable: false, freezePitch: false)
    let diff =
      KickVoice.build(params: dp1.signals, config: config)
      - KickVoice.build(params: dp2.signals, config: config)
    let diffValues = try diff.realize(frames: config.frames)
    let maxAbs = diffValues.map { abs($0) }.max() ?? 0
    let firstBig = diffValues.firstIndex { abs($0) > 1e-5 } ?? -1
    print("synth twin diff: maxAbs=\(maxAbs) firstIdx>1e-5=\(firstBig)")
    if firstBig >= 0 {
      let lo = max(0, firstBig - 2)
      let hi = min(diffValues.count, firstBig + 6)
      print("  diff[\(lo)..<\(hi)] = \(Array(diffValues[lo..<hi]))")
    }

    let synthTensor = try lossValue {
      let p = TrainableKickParams(initial: values, trainable: true, freezePitch: false)
      let t = Tensor(target)
      return (
        KickVoice.build(params: p.signals, config: config),
        t.toSignal(maxFrames: config.frames)
      )
    }
    print("loss(synth, tensor)   = \(synthTensor)")

    let tensorSynth = try lossValue {
      let p = TrainableKickParams(initial: values, trainable: true, freezePitch: false)
      let t = Tensor(target)
      return (
        t.toSignal(maxFrames: config.frames),
        KickVoice.build(params: p.signals, config: config)
      )
    }
    print("loss(tensor, synth)   = \(tensorSynth)")
  }

  private static func rung2(options: [String: String]) throws {
    guard options["params"] != nil else {
      throw SynthIDError.message("rung2 requires --params <json> with external renderer truth")
    }
    var withRung = options
    withRung["rung"] = "2"
    try train(options: withRung)
  }

  private static func rung3(options: [String: String]) throws {
    var withRung = options
    withRung["rung"] = "3"
    try train(options: withRung)
  }

  private static func renderLearnedAndReport(
    result: TrainingRunResult,
    trueParams: PatchValues?,
    targetSamples: [Float],
    config: SynthIDConfig,
    outDir: URL
  ) throws {
    let learned = peakNormalized(
      try KickVoice.render(values: result.recovered, config: config, parameterBacked: true),
      peak: config.peakNormalizeTo)
    try AudioFile.save(
      url: outDir.appendingPathComponent("learned.wav"),
      samples: learned,
      sampleRate: config.sampleRate)

    let normalizedTarget = fitOrPad(
      peakNormalized(targetSamples, peak: config.peakNormalizeTo),
      frames: config.frames)
    let silence = [Float](repeating: 0, count: Int(config.sampleRate * 0.5))
    let ab = normalizedTarget + silence + fitOrPad(learned, frames: config.frames)
    try AudioFile.save(url: outDir.appendingPathComponent("ab.wav"), samples: ab, sampleRate: config.sampleRate)

    let report = ReportWriter.make(
      rung: config.rung,
      trueParams: trueParams,
      recovered: result.recovered,
      initLoss: result.initLoss,
      finalLoss: result.bestLoss,
      includeNoiseCutoff: config.enableNoiseFilter)
    try ReportWriter.write(report: report, to: outDir)
  }

  private static func parseOptions(_ args: [String]) throws -> [String: String] {
    var options: [String: String] = [:]
    var index = 0
    while index < args.count {
      let arg = args[index]
      guard arg.hasPrefix("--") else {
        throw SynthIDError.message("unexpected positional argument \(arg)")
      }
      let key = String(arg.dropFirst(2))
      let flags: Set<String> = [
        "freeze-pitch", "no-linear-mag", "no-noise-filter", "allow-fail", "no-lr-decay",
      ]
      if flags.contains(key) {
        options[key] = "true"
        index += 1
        continue
      }
      guard index + 1 < args.count else {
        throw SynthIDError.message("missing value for \(arg)")
      }
      options[key] = args[index + 1]
      index += 2
    }
    return options
  }

  private static func copyIfPresent(from source: URL, to destination: URL) throws {
    guard FileManager.default.fileExists(atPath: source.path) else { return }
    if FileManager.default.fileExists(atPath: destination.path) {
      try FileManager.default.removeItem(at: destination)
    }
    try FileManager.default.copyItem(at: source, to: destination)
  }

  private static func printUsage() {
    print(
      """
      SynthID

      swift run SynthID render --params <json> --out <wav> [--frames N]
      swift run SynthID train  --target <wav> --out <dir> [--rung 1|2|3] [--params <truth.json>]
      swift run SynthID train  --target <wav> --out <dir> --fdcheck <param> [--params <point.json>]
      swift run SynthID rung1  --seed <N> --out <dir> [--epochs N] [--restarts N]
      swift run SynthID rung2  --target <wav-from-numpy> --params <json> --out <dir>
      swift run SynthID rung3  --target <real-808-wav> --out <dir>

      Common flags: --frames N --windows a,b,c --no-linear-mag --linear-mag-weight W
                    --pitch-lr LR --amp-lr LR --decay-lr LR --tone-lr LR --noise-lr LR
                    --no-noise-filter --fd-eps EPS --backend metal|cpu
      """
    )
  }
}

do {
  try SynthIDCLI.run()
} catch {
  fputs("error: \(error)\n", stderr)
  exit(1)
}
