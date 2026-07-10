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
    _ = try renderLearnedAndReport(
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
      try writeJSON(sampledParams, to: seedDir.appendingPathComponent("sampled_params.json"))

      let rawTarget = try KickVoice.render(
        values: sampledParams,
        config: seedConfig,
        parameterBacked: true)
      let normalized = try normalizeTarget(
        rawTarget,
        params: sampledParams,
        config: seedConfig,
        context: "rung1 seed \(seed)")
      try writeJSON(normalized.params, to: seedDir.appendingPathComponent("true_params.json"))
      try AudioFile.save(
        url: seedDir.appendingPathComponent("target.wav"),
        samples: normalized.samples,
        sampleRate: seedConfig.sampleRate)

      let report = try recoverTarget(
        samples: normalized.samples,
        trueParams: normalized.params,
        config: seedConfig,
        outDir: seedDir,
        context: "rung1 seed \(seed)")
      if report.pass { passed += 1 }
      print(
        "seed=\(seed) pass=\(report.pass) lossRatio=\(String(format: "%.6f", report.lossRatio))")
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
    guard let outPath = options["out"] else {
      throw SynthIDError.message("rung2 requires --out <dir>")
    }
    var config = try loadConfig(url: options["config"].map { URL(fileURLWithPath: $0) })
    try config.applyCLI(options)
    config.rung = 2
    config.applyRuntime()

    let root = URL(fileURLWithPath: outPath)
    try ensureDirectory(root)
    let verifyOnly = options.keys.contains("verify-only")
    let manualTarget = options["target"].map { URL(fileURLWithPath: $0) }
    let manualParams = options["params"].map { URL(fileURLWithPath: $0) }
    guard (manualTarget == nil) == (manualParams == nil) else {
      throw SynthIDError.message("rung2 requires --target and --params together")
    }

    if let targetURL = manualTarget, let paramsURL = manualParams {
      let sampledParams = try loadPatchValues(from: paramsURL)
      let (referenceSamples, sampleRate) = try AudioFile.load(url: targetURL)
      try requireSampleRate(sampleRate, config: config, context: "rung2 external target")
      try writeJSON(sampledParams, to: root.appendingPathComponent("sampled_params.json"))
      try AudioFile.save(
        url: root.appendingPathComponent("reference_raw.wav"),
        samples: referenceSamples,
        sampleRate: config.sampleRate)
      try verifyReferenceRenderer(
        params: sampledParams,
        referenceSamples: referenceSamples,
        config: config,
        outDir: root,
        context: "rung2 external target")
      let normalized = try normalizeTarget(
        referenceSamples,
        params: sampledParams,
        config: config,
        context: "rung2 external target")
      try writeJSON(normalized.params, to: root.appendingPathComponent("true_params.json"))
      try AudioFile.save(
        url: root.appendingPathComponent("target.wav"),
        samples: normalized.samples,
        sampleRate: config.sampleRate)
      if verifyOnly { return }

      let lossFloor = try rendererLossFloor(
        samples: normalized.samples,
        trueParams: normalized.params,
        config: config,
        outDir: root,
        context: "rung2 external target")
      let report = try recoverTarget(
        samples: normalized.samples,
        trueParams: normalized.params,
        config: config,
        outDir: root,
        context: "rung2 external target",
        irreducibleLossFloor: lossFloor)
      print(
        "rung2 pass=\(report.pass) lossRatio=\(String(format: "%.6f", report.lossRatio))")
      if !report.pass && !options.keys.contains("allow-fail") {
        throw SynthIDError.message("rung2 external target failed recovery acceptance")
      }
      return
    }

    let seeds: [Int]
    if let rawSeeds = options["seeds"] {
      seeds = try parseIntList(rawSeeds, "--seeds")
    } else if options["seed"] != nil {
      seeds = [Int(config.seed)]
    } else {
      seeds = [1, 2, 3, 4, 5]
    }
    guard !seeds.isEmpty else {
      throw SynthIDError.message("rung2 requires at least one seed")
    }
    let rendererURL = options["renderer"].map { URL(fileURLWithPath: $0) }
    let python = options["python"] ?? "python3"

    var passed = 0
    for seed in seeds {
      let seedDir = seeds.count == 1 ? root : root.appendingPathComponent("seed-\(seed)")
      try ensureDirectory(seedDir)
      var seedConfig = config
      seedConfig.seed = UInt64(seed)
      let sampledParams = PatchValues.sample(seed: UInt64(seed))
      let sampledParamsURL = seedDir.appendingPathComponent("sampled_params.json")
      let referenceURL = seedDir.appendingPathComponent("reference_raw.wav")
      try writeJSON(sampledParams, to: sampledParamsURL)

      try ReferenceRenderer.render(
        paramsURL: sampledParamsURL,
        outputURL: referenceURL,
        config: seedConfig,
        scriptURL: rendererURL,
        python: python)
      let (referenceSamples, sampleRate) = try AudioFile.load(url: referenceURL)
      try requireSampleRate(
        sampleRate,
        config: seedConfig,
        context: "rung2 seed \(seed)")
      try verifyReferenceRenderer(
        params: sampledParams,
        referenceSamples: referenceSamples,
        config: seedConfig,
        outDir: seedDir,
        context: "rung2 seed \(seed)")

      let normalized = try normalizeTarget(
        referenceSamples,
        params: sampledParams,
        config: seedConfig,
        context: "rung2 seed \(seed)")
      try writeJSON(normalized.params, to: seedDir.appendingPathComponent("true_params.json"))
      try AudioFile.save(
        url: seedDir.appendingPathComponent("target.wav"),
        samples: normalized.samples,
        sampleRate: seedConfig.sampleRate)
      if verifyOnly { continue }

      let lossFloor = try rendererLossFloor(
        samples: normalized.samples,
        trueParams: normalized.params,
        config: seedConfig,
        outDir: seedDir,
        context: "rung2 seed \(seed)")
      let report = try recoverTarget(
        samples: normalized.samples,
        trueParams: normalized.params,
        config: seedConfig,
        outDir: seedDir,
        context: "rung2 seed \(seed)",
        irreducibleLossFloor: lossFloor)
      if report.pass { passed += 1 }
      print(
        "seed=\(seed) pass=\(report.pass) lossRatio=\(String(format: "%.6f", report.lossRatio))")
    }

    if verifyOnly {
      print("rung2 renderer equivalence passed for \(seeds.count)/\(seeds.count) seeds")
      return
    }
    let requiredPasses = seeds.count >= 5
      ? Int(Foundation.ceil(Double(seeds.count) * 0.6))
      : seeds.count
    if passed < requiredPasses && !options.keys.contains("allow-fail") {
      throw SynthIDError.message(
        "rung2 failed: \(passed)/\(seeds.count) seeds passed; required \(requiredPasses)")
    }
  }

  private static func recoverTarget(
    samples: [Float],
    trueParams: PatchValues?,
    config: SynthIDConfig,
    outDir: URL,
    context: String,
    irreducibleLossFloor: Float = 0
  ) throws -> SynthIDReport {
    var bestResult: TrainingRunResult?
    var bestRestartDir: URL?
    var allResults: [TrainingRunResult] = []
    for restart in 0..<max(1, config.restarts) {
      let restartDir = outDir.appendingPathComponent("restart-\(restart)")
      let result = try SynthIDTrainer(config: config).train(
        targetSamples: samples,
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
      throw SynthIDError.message("no recovery result for \(context)")
    }

    // Restarts often solve different subspaces. Greedily stitch them while
    // selecting exclusively by the same audio loss, then fine-tune the result.
    if allResults.count > 1 {
      let trainer = SynthIDTrainer(config: config)
      var noiseSubspace = ["noiseAmp", "noiseDecay"]
      if config.enableNoiseFilter { noiseSubspace.append("noiseCutoff") }
      let subspaces: [[String]] = [
        ["clickFreq", "clickAmp", "clickDecay"],
        ["fStart", "fEnd", "pitchDecay"],
        noiseSubspace,
        ["bodyAmp", "drive", "outGain"],
      ]
      var stitched = result.recovered
      var stitchedLoss = try trainer.evaluateLoss(values: stitched, targetSamples: samples)
      var improved = false
      for donor in allResults {
        for subspace in subspaces {
          var candidate = stitched
          for name in subspace { candidate[name] = donor.recovered[name] }
          let loss = try trainer.evaluateLoss(values: candidate, targetSamples: samples)
          if loss < stitchedLoss {
            stitched = candidate
            stitchedLoss = loss
            improved = true
          }
        }
      }

      // The millisecond click has a rippled frequency landscape. A coarse
      // audio-loss-only search provides the same honest basin selection used by
      // multiple random restarts, then gradient descent performs local tuning.
      let clickFrequencySpec = KickParamSpecs.byName["clickFreq"]!
      let clickFrequencySteps = 24
      for index in 0...clickFrequencySteps {
        let position = Float(index) / Float(clickFrequencySteps)
        let frequency = clickFrequencySpec.min
          * Foundation.exp(
            position * Foundation.log(clickFrequencySpec.max / clickFrequencySpec.min))
        var candidate = stitched
        candidate.clickFreq = frequency
        let loss = try trainer.evaluateLoss(values: candidate, targetSamples: samples)
        if loss < stitchedLoss {
          stitched = candidate
          stitchedLoss = loss
          improved = true
        }
      }

      if improved {
        var tuneConfig = config
        tuneConfig.epochs = max(200, config.epochs / 3)
        let stitchDir = outDir.appendingPathComponent("restart-stitch")
        let tuned = try SynthIDTrainer(config: tuneConfig).train(
          targetSamples: samples,
          outDir: stitchDir,
          trueParams: trueParams,
          initialOverride: stitched)
        print(
          "  stitch: base=\(String(format: "%.5f", result.bestLoss)) stitched=\(String(format: "%.5f", stitchedLoss)) tuned=\(String(format: "%.5f", tuned.bestLoss))"
        )
        if tuned.bestLoss < result.bestLoss {
          // Keep the cold-start loss for the acceptance ratio.
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
      for filename in [
        "checkpoint.json",
        "loss_curve.csv",
        "pitch_fit.json",
        "pitch_points.json",
      ] {
        try copyIfPresent(
          from: bestRestartDir.appendingPathComponent(filename),
          to: outDir.appendingPathComponent(filename))
      }
    }
    // A stitched run starts from donor-combined parameters, but acceptance is
    // measured from the selected cold start. Keep the root initialization and
    // its audio aligned with result.initLoss.
    try writeJSON(result.initial, to: outDir.appendingPathComponent("initial_params.json"))
    try writeJSON(result.recovered, to: outDir.appendingPathComponent("recovered_params.json"))
    return try renderLearnedAndReport(
      result: result,
      trueParams: trueParams,
      targetSamples: samples,
      config: config,
      outDir: outDir,
      irreducibleLossFloor: irreducibleLossFloor)
  }

  private static func rendererLossFloor(
    samples: [Float],
    trueParams: PatchValues,
    config: SynthIDConfig,
    outDir: URL,
    context: String
  ) throws -> Float {
    // This truth-derived diagnostic is intentionally excluded from training,
    // optimizer updates, restart selection, and parameter initialization. It is
    // used only to score the reducible portion of an external-renderer loss.
    let floor = try SynthIDTrainer(config: config).evaluateLoss(
      values: trueParams,
      targetSamples: samples)
    guard floor.isFinite && floor >= 0 else {
      throw SynthIDError.message("\(context) produced invalid renderer loss floor \(floor)")
    }
    try writeJSON(
      ["irreducibleLossFloor": floor],
      to: outDir.appendingPathComponent("renderer_loss_floor.json"))
    print("\(context) renderer lossFloor=\(String(format: "%.6e", floor))")
    return floor
  }

  private static func normalizeTarget(
    _ rawSamples: [Float],
    params: PatchValues,
    config: SynthIDConfig,
    context: String
  ) throws -> (samples: [Float], params: PatchValues) {
    let fitted = fitOrPad(rawSamples, frames: config.frames)
    let scale = peakNormalizationScale(fitted, peak: config.peakNormalizeTo)
    var normalizedParams = params
    let normalizedOutGain = params.outGain * scale
    guard let outGainSpec = KickParamSpecs.byName["outGain"],
      normalizedOutGain >= outGainSpec.min,
      normalizedOutGain <= outGainSpec.max
    else {
      let bounds = KickParamSpecs.byName["outGain"].map { "\($0.min)...\($0.max)" } ?? "unknown"
      throw SynthIDError.message(
        "\(context) normalization would set outGain=\(normalizedOutGain), outside \(bounds)")
    }
    normalizedParams.outGain = normalizedOutGain
    return (fitted.map { $0 * scale }, normalizedParams)
  }

  private static func verifyReferenceRenderer(
    params: PatchValues,
    referenceSamples: [Float],
    config: SynthIDConfig,
    outDir: URL,
    context: String
  ) throws {
    let equivalence = try ReferenceRenderer.verify(
      params: params,
      referenceSamples: referenceSamples,
      config: config)
    try writeJSON(equivalence, to: outDir.appendingPathComponent("renderer_equivalence.json"))
    print(
      "\(context) renderer maxAbs=\(String(format: "%.6e", equivalence.maxAbsoluteError)) threshold=\(String(format: "%.1e", equivalence.threshold)) pass=\(equivalence.pass)"
    )
    guard equivalence.pass else {
      let first = equivalence.firstFailingFrame.map(String.init) ?? "unknown"
      throw SynthIDError.message(
        "\(context) renderer equivalence failed: maxAbs=\(equivalence.maxAbsoluteError), first failing frame=\(first), threshold=\(equivalence.threshold)")
    }
  }

  private static func requireSampleRate(
    _ sampleRate: Float,
    config: SynthIDConfig,
    context: String
  ) throws {
    guard abs(sampleRate - config.sampleRate) <= 0.5 else {
      throw SynthIDError.message(
        "\(context) sampleRate=\(sampleRate), expected \(config.sampleRate); no resampling is applied")
    }
  }

  private static func rung3(options: [String: String]) throws {
    guard let targetPath = options["target"], let outPath = options["out"] else {
      throw SynthIDError.message("rung3 requires --target <real-808-wav> and --out <dir>")
    }
    var config = try loadConfig(url: options["config"].map { URL(fileURLWithPath: $0) })
    try config.applyCLI(options)
    config.rung = 3
    config.applyRuntime()

    let targetURL = URL(fileURLWithPath: targetPath)
    let outDir = URL(fileURLWithPath: outPath)
    try ensureDirectory(outDir)
    let (sourceSamples, sourceRate) = try AudioFile.load(url: targetURL)
    let onsetThresholdDB = try options["onset-threshold-db"]
      .map { try parseFloat($0, "--onset-threshold-db") } ?? -40
    let prepared = try Rung3TargetPreprocessor.prepare(
      samples: sourceSamples,
      sourceRate: sourceRate,
      sourcePath: targetURL.standardizedFileURL.path,
      config: config,
      onsetThresholdDB: onsetThresholdDB)
    try copyIfPresent(from: targetURL, to: outDir.appendingPathComponent("source.wav"))
    try writeJSON(prepared.report, to: outDir.appendingPathComponent("preprocessing.json"))
    let preparedTargetURL = outDir.appendingPathComponent("target.wav")
    try AudioFile.save(
      url: preparedTargetURL,
      samples: prepared.samples,
      sampleRate: config.sampleRate)
    print(
      "rung3 prepared sourceRate=\(Int(sourceRate.rounded())) targetRate=\(Int(config.sampleRate.rounded())) onset=\(String(format: "%.6f", prepared.report.onsetSeconds))s sourceFrames=\(sourceSamples.count) outputFrames=\(prepared.samples.count) cropped=\(prepared.report.cropped) padded=\(prepared.report.padded)"
    )
    if options.keys.contains("prepare-only") { return }
    if let paramName = options["fdcheck"] {
      let result = try SynthIDTrainer(config: config).fdcheck(
        paramName: paramName,
        targetSamples: prepared.samples,
        outDir: outDir)
      print(
        "fdcheck param=\(result.paramName) baseLoss=\(String(format: "%.6e", result.baseLoss)) fd=\(String(format: "%.6e", result.finiteDifferenceGrad)) autograd=\(String(format: "%.6e", result.autogradGrad)) relErr=\(String(format: "%.6e", result.relativeError))"
      )
      return
    }

    var report = try recoverTarget(
      samples: prepared.samples,
      trueParams: nil,
      config: config,
      outDir: outDir,
      context: "rung3 real target")
    let comparison = try Rung3Comparator.run(
      targetURL: preparedTargetURL,
      initialURL: outDir.appendingPathComponent("initial.wav"),
      learnedURL: outDir.appendingPathComponent("learned.wav"),
      outputImageURL: outDir.appendingPathComponent("compare.png"),
      outputJSONURL: outDir.appendingPathComponent("compare.json"),
      scriptURL: options["compare-script"].map { URL(fileURLWithPath: $0) },
      python: options["python"] ?? "python3")
    report.rung3Comparison = comparison
    report.pass = comparison.pass
    report.residualMismatch =
      "The learned patch is constrained to the fixed body, click, and filtered-noise voice. "
      + "Any remaining difference in `compare.png` or `ab.wav`—especially attack/beater "
      + "texture and the late decay—is treated as model mismatch rather than hidden lookup data."
    try ReportWriter.write(report: report, to: outDir)
    print(
      "rung3 pass=\(report.pass) improvement=\(String(format: "%.2f%%", comparison.improvement * 100))"
    )
    if !report.pass && !options.keys.contains("allow-fail") {
      throw SynthIDError.message(
        "rung3 failed: independent MR-STFT improvement \(comparison.improvement), required \(comparison.requiredImprovement)")
    }
  }

  private static func renderLearnedAndReport(
    result: TrainingRunResult,
    trueParams: PatchValues?,
    targetSamples: [Float],
    config: SynthIDConfig,
    outDir: URL,
    irreducibleLossFloor: Float = 0
  ) throws -> SynthIDReport {
    if config.rung == 3 {
      let initial = peakNormalized(
        try KickVoice.render(values: result.initial, config: config, parameterBacked: true),
        peak: config.peakNormalizeTo)
      try AudioFile.save(
        url: outDir.appendingPathComponent("initial.wav"),
        samples: initial,
        sampleRate: config.sampleRate)
    }
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
    let abFrames = config.rung == 3 ? Int(config.sampleRate.rounded()) : config.frames
    let ab = fitOrPad(normalizedTarget, frames: abFrames) + silence
      + fitOrPad(learned, frames: abFrames)
    try AudioFile.save(url: outDir.appendingPathComponent("ab.wav"), samples: ab, sampleRate: config.sampleRate)

    let report = ReportWriter.make(
      rung: config.rung,
      trueParams: trueParams,
      recovered: result.recovered,
      initLoss: result.initLoss,
      finalLoss: result.bestLoss,
      irreducibleLossFloor: irreducibleLossFloor,
      includeNoiseCutoff: config.enableNoiseFilter)
    try ReportWriter.write(report: report, to: outDir)
    return report
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
        "verify-only", "prepare-only",
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
      swift run SynthID rung2  --out <dir> [--seeds 1,2,3,4,5] [--verify-only]
      swift run SynthID rung2  --target <wav-from-numpy> --params <json> --out <dir> [--verify-only]
      swift run SynthID rung3  --target <real-808-wav> --out <dir> [--prepare-only]

      Common flags: --frames N --windows a,b,c --no-linear-mag --linear-mag-weight W
                    --pitch-lr LR --amp-lr LR --decay-lr LR --tone-lr LR --noise-lr LR
                    --no-noise-filter --fd-eps EPS --backend metal|cpu
      Rung 2 flags: --renderer <render_reference.py> --python <python3>
      Rung 3 flags: --onset-threshold-db DB --compare-script <compare.py> --python <python3>
                    --fdcheck <param>
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
