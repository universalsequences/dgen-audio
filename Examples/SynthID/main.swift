import DGenLazy
import DGenTrainProtocol
#if canImport(Darwin)
import Darwin
#else
import Glibc
#endif
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
    case "score":
      try score(options: parsed)
    case "rung1":
      try rung1(options: parsed)
    case "probe":
      try probe(options: parsed)
    case "loss-sweep":
      try lossSweep(options: parsed)
    case "rung2":
      try rung2(options: parsed)
    case "rung3":
      try rung3(options: parsed)
    case "batch-bench":
      try BatchBench.run(options: parsed)
    case "batch-train-bench":
      try BatchTrainBench.run(options: parsed)
    case "basin-search":
      try BasinSearch.run(options: parsed)
    case "batch-refine":
      try BatchRefine.run(options: parsed)
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
    let initialOverride = try options["initial-params"].map {
      try loadPatchValues(from: URL(fileURLWithPath: $0))
    }
    try AudioFile.save(
      url: outDir.appendingPathComponent("target.wav"),
      samples: fitOrPad(peakNormalized(samples, peak: config.peakNormalizeTo), frames: config.frames),
      sampleRate: config.sampleRate)

    if let paramName = options["fdcheck"] {
      if config.fdcheckDirectional == true {
        guard let trueParams else {
          throw SynthIDError.message("directional fdcheck requires --params <point.json>")
        }
        let result = try SynthIDTrainer(config: config).directionalFDCheck(
          paramName: paramName,
          targetSamples: samples,
          initial: trueParams,
          outDir: outDir)
        print(
          "directional-fdcheck param=\(result.paramName) fd=\(String(format: "%.6e", result.finiteDifferenceGrad)) directionalAutograd=\(String(format: "%.6e", result.directionalAutogradGrad)) fullVoiceAutograd=\(String(format: "%.6e", result.fullVoiceAutogradGrad)) relErr=\(String(format: "%.6e", result.relativeError)) chainRuleRelErr=\(String(format: "%.6e", result.chainRuleRelativeError))"
        )
        return
      }
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
      trueParams: trueParams,
      restartIndex: try options["restart-index"].map {
        try parseInt($0, "--restart-index")
      } ?? 0,
      initialOverride: initialOverride)
    _ = try renderLearnedAndReport(
      result: result,
      trueParams: trueParams,
      targetSamples: samples,
      config: config,
      outDir: outDir)
  }

  private static func score(options: [String: String]) throws {
    guard
      let targetPath = options["target"],
      let recoveredPath = options["params"],
      let truePath = options["true-params"],
      let initialPath = options["initial-params"],
      let outPath = options["out"]
    else {
      throw SynthIDError.message(
        "score requires --target, --params, --true-params, --initial-params, and --out")
    }
    var config = try loadConfig(url: options["config"].map { URL(fileURLWithPath: $0) })
    try config.applyCLI(options)
    config.rung = 1
    config.useSmoothTrainingLoss = false
    config.useSmoothBasinSearch = false
    config.applyRuntime()

    let (samples, _) = try AudioFile.load(url: URL(fileURLWithPath: targetPath))
    let recovered = try loadPatchValues(from: URL(fileURLWithPath: recoveredPath))
    let trueParams = try loadPatchValues(from: URL(fileURLWithPath: truePath))
    let initial = try loadPatchValues(from: URL(fileURLWithPath: initialPath))
    let trainer = SynthIDTrainer(config: config)
    let initLoss = try trainer.evaluateLoss(values: initial, targetSamples: samples)
    let finalLoss = try trainer.evaluateLoss(values: recovered, targetSamples: samples)
    let outDir = URL(fileURLWithPath: outPath)
    try ensureDirectory(outDir)
    let result = TrainingRunResult(
      recovered: recovered,
      initial: initial,
      pitchFit: PitchFit(fStart: 110, fEnd: 110, pitchDecay: -1, error: nil),
      initLoss: initLoss,
      bestLoss: finalLoss,
      bestEpoch: 0,
      losses: [])
    let report = try renderLearnedAndReport(
      result: result,
      trueParams: trueParams,
      targetSamples: samples,
      config: config,
      outDir: outDir)
    print(
      "score pass=\(report.pass) loss=\(String(format: "%.6f", finalLoss))"
        + " ratio=\(String(format: "%.6f", report.lossRatio))")
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

    let requiredPasses = config.profile == "subtractive-bass"
      ? seeds.count / 2 + 1
      : (seeds.count >= 5 ? 4 : seeds.count)
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
        KickVoice.build(params: p1, config: config),
        KickVoice.build(params: p2, config: config)
      )
    }
    print("loss(synth, synth)    = \(synthSynth)")

    let synthSynthBothTrainable = try lossValue {
      let p1 = TrainableKickParams(initial: values, trainable: true, freezePitch: false)
      let p2 = TrainableKickParams(initial: values, trainable: true, freezePitch: false)
      return (
        KickVoice.build(params: p1, config: config),
        KickVoice.build(params: p2, config: config)
      )
    }
    print("loss(synthT, synthT)  = \(synthSynthBothTrainable)")

    LazyGraphContext.reset()
    let dp1 = TrainableKickParams(initial: values, trainable: true, freezePitch: false)
    let dp2 = TrainableKickParams(initial: values, trainable: false, freezePitch: false)
    let diff =
      KickVoice.build(params: dp1, config: config)
      - KickVoice.build(params: dp2, config: config)
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
        KickVoice.build(params: p, config: config),
        t.toSignal(maxFrames: config.frames)
      )
    }
    print("loss(synth, tensor)   = \(synthTensor)")

    let tensorSynth = try lossValue {
      let p = TrainableKickParams(initial: values, trainable: true, freezePitch: false)
      let t = Tensor(target)
      return (
        t.toSignal(maxFrames: config.frames),
        KickVoice.build(params: p, config: config)
      )
    }
    print("loss(tensor, synth)   = \(tensorSynth)")
  }

  private static func lossSweep(options: [String: String]) throws {
    guard
      let targetPath = options["target"],
      let paramsPath = options["params"],
      let paramName = options["param"],
      let outPath = options["out"]
    else {
      throw SynthIDError.message(
        "loss-sweep requires --target, --params, --param, and --out")
    }
    var config = try loadConfig(url: options["config"].map { URL(fileURLWithPath: $0) })
    try config.applyCLI(options)
    config.applyRuntime()
    guard let spec = KickParamSpecs.byName[paramName] else {
      throw SynthIDError.message("unknown loss-sweep parameter \(paramName)")
    }
    let radius = try options["radius"].map { try parseFloat($0, "--radius") } ?? 5e-4
    let points = try options["points"].map { try parseInt($0, "--points") } ?? 201
    guard points >= 3 else { throw SynthIDError.message("loss-sweep --points must be >= 3") }
    let secondParamName = options["param2"]
    let secondSpec = try secondParamName.map { name in
      guard name != paramName else {
        throw SynthIDError.message("loss-sweep parameters must be distinct")
      }
      guard let spec = KickParamSpecs.byName[name] else {
        throw SynthIDError.message("unknown loss-sweep parameter \(name)")
      }
      return spec
    }
    let secondRadius = try options["radius2"].map { try parseFloat($0, "--radius2") }
      ?? radius
    let secondPoints = try options["points2"].map { try parseInt($0, "--points2") }
      ?? points
    guard secondPoints >= 3 else {
      throw SynthIDError.message("loss-sweep --points2 must be >= 3")
    }
    let thirdParamName = options["param3"]
    let thirdSpec = try thirdParamName.map { name in
      guard name != paramName, name != secondParamName else {
        throw SynthIDError.message("loss-sweep parameters must be distinct")
      }
      guard let spec = KickParamSpecs.byName[name] else {
        throw SynthIDError.message("unknown loss-sweep parameter \(name)")
      }
      return spec
    }
    guard thirdSpec == nil || secondSpec != nil else {
      throw SynthIDError.message("loss-sweep --param3 requires --param2")
    }
    let thirdRadius = try options["radius3"].map { try parseFloat($0, "--radius3") }
      ?? secondRadius
    let thirdPoints = try options["points3"].map { try parseInt($0, "--points3") }
      ?? secondPoints
    guard thirdPoints >= 3 else {
      throw SynthIDError.message("loss-sweep --points3 must be >= 3")
    }

    let (samples, _) = try AudioFile.load(url: URL(fileURLWithPath: targetPath))
    let center = try loadPatchValues(from: URL(fileURLWithPath: paramsPath))
    let centerZ = spec.transform(center[paramName])
    let trainer = SynthIDTrainer(config: config)
    let csvHeader: String
    if thirdSpec != nil {
      csvHeader = "delta,transformedValue,naturalValue,delta2,transformedValue2,naturalValue2,"
        + "delta3,transformedValue3,naturalValue3,loss\n"
    } else if secondSpec != nil {
      csvHeader = "delta,transformedValue,naturalValue,delta2,transformedValue2,naturalValue2,loss\n"
    } else {
      csvHeader = "delta,transformedValue,naturalValue,loss\n"
    }
    var csv = csvHeader
    for index in 0..<points {
      let fraction = Float(index) / Float(points - 1)
      let delta = -radius + 2.0 * radius * fraction
      var values = center
      values[paramName] = spec.inverse(centerZ + delta)
      if let secondSpec, let secondParamName {
        let secondCenterZ = secondSpec.transform(center[secondParamName])
        for secondIndex in 0..<secondPoints {
          let secondFraction = Float(secondIndex) / Float(secondPoints - 1)
          let secondDelta = -secondRadius + 2.0 * secondRadius * secondFraction
          values[secondParamName] = secondSpec.inverse(secondCenterZ + secondDelta)
          if let thirdSpec, let thirdParamName {
            let thirdCenterZ = thirdSpec.transform(center[thirdParamName])
            for thirdIndex in 0..<thirdPoints {
              let thirdFraction = Float(thirdIndex) / Float(thirdPoints - 1)
              let thirdDelta = -thirdRadius + 2.0 * thirdRadius * thirdFraction
              values[thirdParamName] = thirdSpec.inverse(thirdCenterZ + thirdDelta)
              let loss = try trainer.evaluateLoss(values: values, targetSamples: samples)
              csv += "\(delta),\(centerZ + delta),\(values[paramName]),\(secondDelta),"
                + "\(secondCenterZ + secondDelta),\(values[secondParamName]),\(thirdDelta),"
                + "\(thirdCenterZ + thirdDelta),\(values[thirdParamName]),\(loss)\n"
            }
          } else {
            let loss = try trainer.evaluateLoss(values: values, targetSamples: samples)
            csv += "\(delta),\(centerZ + delta),\(values[paramName]),\(secondDelta),"
              + "\(secondCenterZ + secondDelta),\(values[secondParamName]),\(loss)\n"
          }
        }
      } else {
        let loss = try trainer.evaluateLoss(values: values, targetSamples: samples)
        csv += "\(delta),\(centerZ + delta),\(values[paramName]),\(loss)\n"
      }
    }
    let out = URL(fileURLWithPath: outPath)
    try ensureDirectory(out.deletingLastPathComponent())
    try csv.write(to: out, atomically: true, encoding: .utf8)
    print(
      "wrote=\(out.path) points=\(points) radius=\(radius)"
        + (secondSpec == nil ? "" : " points2=\(secondPoints) radius2=\(secondRadius)")
        + (thirdSpec == nil ? "" : " points3=\(thirdPoints) radius3=\(thirdRadius)"))
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
    let rendererURL = options["renderer"].map { URL(fileURLWithPath: $0) }
    let comparatorURL = options["polyblep-comparator"].map { URL(fileURLWithPath: $0) }
    let python = options["python"] ?? "python3"
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
      if config.profile == "subtractive-bass" {
        try verifyPolyblepEquivalence(
          params: sampledParams,
          paramsURL: paramsURL,
          config: config,
          outDir: root,
          context: "rung2 external target",
          rendererURL: rendererURL,
          comparatorURL: comparatorURL,
          python: python)
      }
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
      if seedConfig.profile == "subtractive-bass" {
        try verifyPolyblepEquivalence(
          params: sampledParams,
          paramsURL: sampledParamsURL,
          config: seedConfig,
          outDir: seedDir,
          context: "rung2 seed \(seed)",
          rendererURL: rendererURL,
          comparatorURL: comparatorURL,
          python: python)
      }

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
    irreducibleLossFloor: Float = 0,
    python: String = "python3",
    scoreScriptURL: URL? = nil
  ) throws -> SynthIDReport {
    var bestResult: TrainingRunResult?
    var bestRestartDir: URL?
    var allResults: [TrainingRunResult] = []
    var restartDirs: [URL] = []
    for restart in 0..<max(1, config.restarts) {
      let restartDir = outDir.appendingPathComponent("restart-\(restart)")
      let result = try SynthIDTrainer(config: config).train(
        targetSamples: samples,
        outDir: restartDir,
        trueParams: trueParams,
        restartIndex: restart)
      allResults.append(result)
      restartDirs.append(restartDir)
      if bestResult == nil || result.bestLoss < bestResult!.bestLoss {
        bestResult = result
        bestRestartDir = restartDir
      }
    }
    guard var result = bestResult else {
      throw SynthIDError.message("no recovery result for \(context)")
    }

    // FIX 2: restart selection by the independent CPU metric (rung 3 only).
    // GPU training loss and the independent MR-STFT metric can disagree on
    // which restart's basin is best; rescore each restart's own learned
    // params with the independent metric and let that pick the winner
    // instead. Falls back to the GPU-loss winner already selected above (with
    // a logged warning) if the python subprocess path fails for any reason.
    if config.rung == 3 {
      let targetURL = outDir.appendingPathComponent("target.wav")
      do {
        var table: [(index: Int, gpuLoss: Float, cpuDistance: Float)] = []
        var cpuBest: (result: TrainingRunResult, dir: URL, index: Int, distance: Float)?
        for (index, candidate) in allResults.enumerated() {
          let paramsURL = restartDirs[index].appendingPathComponent("recovered_params.json")
          let distance = try Rung3IndependentScorer.score(
            targetURL: targetURL,
            paramsURL: paramsURL,
            frames: config.frames,
            sampleRate: config.sampleRate,
            profile: config.profile,
            scriptURL: scoreScriptURL,
            python: python)
          table.append((index, candidate.bestLoss, distance))
          if cpuBest == nil || distance < cpuBest!.distance {
            cpuBest = (candidate, restartDirs[index], index, distance)
          }
        }
        print("\(context) restart selection (independent CPU metric):")
        print("  restart  gpuLoss     cpuDistance")
        for entry in table {
          print(
            String(
              format: "  %7d  %9.5f  %11.6f", entry.index, entry.gpuLoss, entry.cpuDistance))
        }
        if let cpuBest {
          print(
            "  selected restart \(cpuBest.index) by independent CPU metric"
              + " (gpu-loss winner was \(String(format: "%.5f", bestResult!.bestLoss)))")
          result = cpuBest.result
          bestRestartDir = cpuBest.dir
        }
      } catch {
        print(
          "warning: \(context) independent CPU rescoring failed (\(error)); "
            + "falling back to GPU-loss restart selection")
      }
    }

    // Restarts often solve different subspaces. Greedily stitch them while
    // selecting exclusively by the same audio loss, then fine-tune the result.
    if allResults.count > 1 {
      let trainer = SynthIDTrainer(config: config)
      let isBass = config.profile == "hoodie-bass"
      let isSubtractive = config.profile == "subtractive-bass"
      var noiseSubspace = ["noiseAmp", "noiseDecay"]
      if config.enableNoiseFilter { noiseSubspace.append("noiseCutoff") }
      var ampSubspace = ["bodyAmp", "drive", "outGain"]
      if config.profile == "909" {
        ampSubspace.append("bodyHarmonic")
        ampSubspace.append(contentsOf: KickParamSpecs.tr909HarmonicCorrections.map(\.name))
      }
      let subspaces: [[String]] = isSubtractive
        ? [
          ["shape", "pw"],
          ["fBase", "fAmt", "fDecay", "res"],
          ["attackTime", "decayTime", "sustain", "releaseTime"],
          ["drive", "outGain"],
        ]
        : (isBass ? [
          ["f0"],
          ["attackTime", "decayTime", "sustain", "noteOff", "releaseTime"],
          ["brightnessDecay"]
            + KickParamSpecs.hoodieBassHarmonics.filter { $0.decay > 0 }.map(\.name),
          ["drive", "outGain"]
            + KickParamSpecs.hoodieBassHarmonics.filter { $0.decay == 0 }.map(\.name),
        ] : [
          ["clickFreq", "clickAmp", "clickDecay"],
          ["fStart", "fEnd", "pitchDecay"],
          noiseSubspace,
          ampSubspace,
        ])
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
      if !isBass && !isSubtractive {
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

    if config.profile == "subtractive-bass" {
      let trainer = SynthIDTrainer(config: config)
      let pwSpec = KickParamSpecs.byName["pw"]!
      let transitionWidth = 110.0 / config.sampleRate
      let searchStep = transitionWidth * 0.5

      func searchPulseWidth(
        around center: PatchValues,
        radius: Float,
        step: Float = searchStep
      ) throws -> (params: PatchValues, loss: Float, atEdge: Bool, points: Int) {
        let lower = max(pwSpec.min, center.pw - radius)
        let upper = min(pwSpec.max, center.pw + radius)
        let intervals = max(1, Int(Foundation.ceil((upper - lower) / step)))
        var best = center
        var bestLoss = try trainer.evaluateLoss(values: center, targetSamples: samples)
        var bestIndex = -1
        for index in 0...intervals {
          var candidate = center
          candidate.pw = lower + (upper - lower) * Float(index) / Float(intervals)
          let loss = try trainer.evaluateLoss(values: candidate, targetSamples: samples)
          if loss < bestLoss {
            best = candidate
            bestLoss = loss
            bestIndex = index
          }
        }
        let atEdge = bestIndex >= 0 && (bestIndex <= 1 || bestIndex >= intervals - 1)
        return (best, bestLoss, atEdge, intervals + 1)
      }

      let beforeSearchLoss = try trainer.evaluateLoss(
        values: result.recovered, targetSamples: samples)
      var basin = try searchPulseWidth(around: result.recovered, radius: 0.12)
      if basin.atEdge {
        let fullRadius = max(
          result.recovered.pw - pwSpec.min,
          pwSpec.max - result.recovered.pw)
        basin = try searchPulseWidth(around: result.recovered, radius: fullRadius)
      }
      print(
          "  pw basin: start=\(String(format: "%.6f", result.recovered.pw))"
            + " loss=\(String(format: "%.6f", beforeSearchLoss))"
            + " best=\(String(format: "%.6f", basin.params.pw))"
            + " loss=\(String(format: "%.6f", basin.loss)) points=\(basin.points)")
      try writeJSON(
        [
          "startPw": result.recovered.pw,
          "startLoss": beforeSearchLoss,
          "bestPw": basin.params.pw,
          "bestLoss": basin.loss,
          "step": searchStep,
        ],
        to: outDir.appendingPathComponent("pw_basin_search_initial.json"))

      var tuneConfig = config
      tuneConfig.epochs = max(300, config.epochs / 2)
      tuneConfig.restarts = 1
      tuneConfig.frozenParams = Array(Set(config.frozenParams).union(["pw"]))
      let tuneDir = outDir.appendingPathComponent("restart-pw-refine")
      let tuned = try SynthIDTrainer(config: tuneConfig).train(
        targetSamples: samples,
        outDir: tuneDir,
        trueParams: trueParams,
        initialOverride: basin.params)

      var smoothConfig = config
      smoothConfig.epochs = max(600, config.epochs)
      smoothConfig.restarts = 1
      smoothConfig.frozenParams = config.frozenParams
      smoothConfig.useSmoothTrainingLoss = true
      let smoothDir = outDir.appendingPathComponent("restart-smooth-refine")
      let smoothed = try SynthIDTrainer(config: smoothConfig).train(
        targetSamples: samples,
        outDir: smoothDir,
        trueParams: trueParams,
        initialOverride: tuned.recovered)

      let smoothProductionLoss = try trainer.evaluateLoss(
        values: smoothed.recovered, targetSamples: samples)
      var chosenParams = smoothed.recovered
      var chosenProductionLoss = smoothProductionLoss
      var chosenSmoothLoss = smoothed.bestLoss
      var chosenEpoch = smoothed.bestEpoch
      var chosenDir = smoothDir
      var refinementLosses = smoothed.losses

      if smoothProductionLoss / max(result.initLoss, 1e-12) > 0.02 {
        var rescueConfig = smoothConfig
        rescueConfig.useSmoothBasinSearch = true
        let rescueDir = outDir.appendingPathComponent("restart-smooth-basin-rescue")
        let rescued = try SynthIDTrainer(config: rescueConfig).train(
          targetSamples: samples,
          outDir: rescueDir,
          trueParams: trueParams,
          initialOverride: smoothed.recovered)
        let rescueProductionLoss = try trainer.evaluateLoss(
          values: rescued.recovered, targetSamples: samples)
        refinementLosses += rescued.losses
        if rescueProductionLoss < chosenProductionLoss {
          chosenParams = rescued.recovered
          chosenProductionLoss = rescueProductionLoss
          chosenSmoothLoss = rescued.bestLoss
          chosenEpoch = rescued.bestEpoch
          chosenDir = rescueDir
        }

        var settleConfig = smoothConfig
        settleConfig.useSmoothBasinSearch = false
        let settleDir = outDir.appendingPathComponent("restart-smooth-basin-settle")
        let settled = try SynthIDTrainer(config: settleConfig).train(
          targetSamples: samples,
          outDir: settleDir,
          trueParams: trueParams,
          initialOverride: rescued.recovered)
        let settleProductionLoss = try trainer.evaluateLoss(
          values: settled.recovered, targetSamples: samples)
        refinementLosses += settled.losses
        if settleProductionLoss < chosenProductionLoss {
          chosenParams = settled.recovered
          chosenProductionLoss = settleProductionLoss
          chosenSmoothLoss = settled.bestLoss
          chosenEpoch = settled.bestEpoch
          chosenDir = settleDir
        }
        print(
          "  smooth rescue: first=\(String(format: "%.6f", rescueProductionLoss))"
            + " settled=\(String(format: "%.6f", settleProductionLoss))"
            + " chosen=\(String(format: "%.6f", chosenProductionLoss))")
      }

      let fineStep = transitionWidth / 16.0
      let finalBasin = try searchPulseWidth(
        around: chosenParams,
        radius: transitionWidth * 2.0,
        step: fineStep)
      print(
          "  smooth refine: smooth=\(String(format: "%.6f", smoothed.bestLoss))"
            + " production=\(String(format: "%.6f", smoothProductionLoss))")
      print(
          "  pw final: step=\(String(format: "%.7f", fineStep))"
            + " pw=\(String(format: "%.6f", finalBasin.params.pw))"
            + " loss=\(String(format: "%.6f", finalBasin.loss))")
      try writeJSON(
        [
          "smoothObjectiveLoss": chosenSmoothLoss,
          "productionLossBeforeFinalPw": chosenProductionLoss,
          "bestPw": finalBasin.params.pw,
          "bestLoss": finalBasin.loss,
          "step": fineStep,
        ],
        to: outDir.appendingPathComponent("pw_basin_search_final.json"))
      try writeJSON(
        finalBasin.params, to: chosenDir.appendingPathComponent("recovered_params.json"))
      try updateCheckpointAfterRung3Refinement(
        outDir: chosenDir, params: finalBasin.params, loss: finalBasin.loss)

      result = TrainingRunResult(
        recovered: finalBasin.params,
        initial: result.initial,
        pitchFit: result.pitchFit,
        initLoss: result.initLoss,
        bestLoss: finalBasin.loss,
        bestEpoch: chosenEpoch,
        losses: result.losses + tuned.losses + refinementLosses)
      bestRestartDir = chosenDir
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
    // FIX 1: the emitted initial.wav/initial_params.json (and hence
    // compare.py's --initial, i.e. the reported improvement gate) must always
    // be the deterministic restart-0 reference initialization, regardless of
    // which restart actually won selection. Pinning the gate to a winning
    // restart's own cold start lets an aggressive bracket inflate the
    // relative-improvement metric by starting from a worse baseline. The
    // winner's own cold start is preserved separately for debugging.
    if config.rung == 3 {
      try writeJSON(result.initial, to: outDir.appendingPathComponent("winner_initial_params.json"))
      result.initial = SynthIDTrainer(config: config).restartInitial(
        pitchFit: result.pitchFit, restartIndex: 0)
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

  private static func verifyPolyblepEquivalence(
    params: PatchValues,
    paramsURL: URL,
    config: SynthIDConfig,
    outDir: URL,
    context: String,
    rendererURL: URL?,
    comparatorURL: URL?,
    python: String
  ) throws {
    let deploymentURL = outDir.appendingPathComponent("deployment_polyblep_oscillator.wav")
    let trainingURL = outDir.appendingPathComponent("training_polyblep_oscillator.wav")
    try ReferenceRenderer.render(
      paramsURL: paramsURL,
      outputURL: deploymentURL,
      config: config,
      scriptURL: rendererURL,
      python: python,
      oscillatorOnly: true)
    let (deploymentSamples, sampleRate) = try AudioFile.load(url: deploymentURL)
    try requireSampleRate(sampleRate, config: config, context: "\(context) deployment oscillator")

    let trainingSamples = try SubtractiveBassVoice.renderOscillator(values: params, config: config)
    try AudioFile.save(
      url: trainingURL, samples: trainingSamples, sampleRate: config.sampleRate)
    let timeReport = try ReferenceRenderer.verifyOscillator(
      trainingSamples: trainingSamples,
      referenceSamples: deploymentSamples,
      config: config)
    try writeJSON(timeReport, to: outDir.appendingPathComponent("oscillator_equivalence.json"))

    let spectralReport = try ReferenceRenderer.comparePolyblep(
      trainingURL: trainingURL,
      deploymentURL: deploymentURL,
      reportURL: outDir.appendingPathComponent("polyblep_equivalence.json"),
      scriptURL: comparatorURL,
      python: python)
    print(
      "\(context) PolyBLEP maxAbs=\(String(format: "%.6e", timeReport.maxAbsoluteError))"
        + " MRSTFT=\(String(format: "%.6e", spectralReport.distance))"
        + " threshold=\(String(format: "%.6e", spectralReport.threshold))"
        + " pass=\(timeReport.pass && spectralReport.pass)")
    guard timeReport.pass && spectralReport.pass else {
      throw SynthIDError.message(
        "\(context) PolyBLEP equivalence failed: maxAbs=\(timeReport.maxAbsoluteError),"
          + " MRSTFT=\(spectralReport.distance), threshold=\(spectralReport.threshold)")
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
    // Note: config.applyRuntime() is deferred until after the target-length
    // fit decision below (FIX 2), since it sets DGenConfig.defaultFrameCount
    // from config.frames and every downstream render/train call must see the
    // final (possibly shrunk) frame count.

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
    if prepared.report.fittedFrames != config.frames {
      print("fit length: \(config.frames) -> \(prepared.report.fittedFrames) frames")
      config.frames = prepared.report.fittedFrames
    }
    config.applyRuntime()
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
      context: "rung3 real target",
      python: options["python"] ?? "python3",
      scoreScriptURL: options["score-script"].map { URL(fileURLWithPath: $0) })
    if !options.keys.contains("no-refine") {
      let recoveredURL = outDir.appendingPathComponent("recovered_params.json")
      let preRefineURL = outDir.appendingPathComponent("pre_refine_params.json")
      try copyIfPresent(from: recoveredURL, to: preRefineURL)
      try Rung3Refiner.run(
        targetURL: preparedTargetURL,
        initialURL: outDir.appendingPathComponent("initial.wav"),
        paramsURL: preRefineURL,
        outputParamsURL: recoveredURL,
        outputJSONURL: outDir.appendingPathComponent("refinement.json"),
        profile: config.profile,
        scriptURL: options["refine-script"].map { URL(fileURLWithPath: $0) },
        python: options["python"] ?? "python3")
      let refined = try loadPatchValues(from: recoveredURL)
      // Canonicalize Python's JSON doubles through PatchValues/Float so the
      // recovered-params artifact and checkpoint carry byte-for-byte values.
      try writeJSON(refined, to: recoveredURL)
      let initial = try loadPatchValues(
        from: outDir.appendingPathComponent("initial_params.json"))
      let pitchFit = try JSONDecoder().decode(
        PitchFit.self,
        from: Data(contentsOf: outDir.appendingPathComponent("pitch_fit.json")))
      let refinedLoss = try SynthIDTrainer(config: config).evaluateLoss(
        values: refined,
        targetSamples: prepared.samples)
      let refinedResult = TrainingRunResult(
        recovered: refined,
        initial: initial,
        pitchFit: pitchFit,
        initLoss: report.initLoss,
        bestLoss: refinedLoss,
        bestEpoch: 0,
        losses: [])
      report = try renderLearnedAndReport(
        result: refinedResult,
        trueParams: nil,
        targetSamples: prepared.samples,
        config: config,
        outDir: outDir)
      try updateCheckpointAfterRung3Refinement(
        outDir: outDir,
        params: refined,
        loss: refinedLoss)
    }
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
    report.residualMismatch = config.profile == "hoodie-bass"
      ? "The learned patch is constrained to an integer-locked steady plus attack-decay Fourier basis and a smooth amplitude envelope. Remaining capture noise, filter motion outside that basis, and release-shape differences are model mismatch rather than hidden lookup data."
      : "The learned patch is constrained to the fixed body, click, and filtered-noise voice. Any remaining difference in `compare.png` or `ab.wav`—especially attack/beater texture and the late decay—is treated as model mismatch rather than hidden lookup data."
    try ReportWriter.write(report: report, to: outDir)
    print(
      "rung3 pass=\(report.pass) improvement=\(String(format: "%.2f%%", comparison.improvement * 100))"
    )
    if !report.pass && !options.keys.contains("allow-fail") {
      throw SynthIDError.message(
        "rung3 failed: independent MR-STFT improvement \(comparison.improvement), required \(comparison.requiredImprovement)")
    }
  }

  private static func updateCheckpointAfterRung3Refinement(
    outDir: URL,
    params: PatchValues,
    loss: Float
  ) throws {
    let url = outDir.appendingPathComponent("checkpoint.json")
    guard FileManager.default.fileExists(atPath: url.path) else { return }
    var checkpoint = try JSONDecoder().decode(
      SynthIDCheckpoint.self, from: Data(contentsOf: url))
    checkpoint.createdAtUTC = timestampUTC()
    checkpoint.loss = loss
    checkpoint.params = params
    checkpoint.transformedParams = Dictionary(
      uniqueKeysWithValues: KickParamSpecs.all.map { spec in
        (spec.name, spec.transform(params[spec.name]))
      })
    try writeJSON(checkpoint, to: url)
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
        "verify-only", "prepare-only", "fdcheck-log-l2", "fdcheck-time-mse",
        "fdcheck-directional", "smooth-training-loss", "smooth-basin-search",
        "probe-only",
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
      swift run SynthID train  --target <wav> --out <dir> [--initial-params <params.json> | --restart-index N]
      swift run SynthID train  --target <wav> --out <dir> --smooth-training-loss
      swift run SynthID score  --target <wav> --params <recovered.json> --true-params <truth.json> --initial-params <initial.json> --out <dir>
      swift run SynthID train  --target <wav> --out <dir> --fdcheck <param> [--params <point.json>]
      swift run SynthID loss-sweep --target <wav> --params <json> --param <name> --out <csv>
        [--param2 <name> --radius2 R --points2 N] [--param3 <name> --radius3 R --points3 N]
      swift run SynthID rung1  --seed <N> --out <dir> [--epochs N] [--restarts N]
      swift run SynthID rung2  --out <dir> [--seeds 1,2,3,4,5] [--verify-only]
      swift run SynthID rung2  --target <wav-from-numpy> --params <json> --out <dir> [--verify-only]
      swift run SynthID rung3  --target <real-808-wav> --out <dir> [--prepare-only]
      swift run SynthID batch-bench [--seed-dir <dir>] [--out <dir>] [--batch-sizes 1,8,32,128,256] [--iters 20]
      swift run SynthID basin-search --target <wav> --out <dir> --base-params <initial.json> [--count 8192] [--batch 256] [--seed N]
      swift run SynthID batch-refine --target <wav> --mode polish|escape|lr-sweep|probe-grads|probe-scalar [--elites <dir>] [--init <json>] [--batch 64] [--steps 300] [--jitter 0.05] [--out <dir>]

      Common flags: --frames N --windows a,b,c --no-linear-mag --linear-mag-weight W
                    --pitch-lr LR --amp-lr LR --decay-lr LR --tone-lr LR --noise-lr LR
                    --no-noise-filter --fd-eps EPS --fdcheck-log-l2 --fdcheck-time-mse
                    --freeze-params a,b,c
                    --fdcheck-directional --direction-eps EPS
                    --backend metal|cpu
                    --profile 808|909|hoodie-bass|subtractive-bass
      Rung 2 flags: --renderer <render_reference.py> --polyblep-comparator <compare_polyblep.py>
                    --python <python3>
      Rung 3 flags: --onset-threshold-db DB --compare-script <compare.py> --python <python3>
                    --refine-script <refine_rung3.py> --score-script <score_params.py>
                    --no-refine --fdcheck <param>
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
