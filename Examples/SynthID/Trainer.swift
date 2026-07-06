import DGenLazy
import Foundation

struct SynthIDCheckpoint: Codable {
  var createdAtUTC: String
  var config: SynthIDConfig
  var epoch: Int
  var loss: Float
  var params: PatchValues
  var transformedParams: [String: Float]
}

struct TrainingRunResult: Codable {
  var recovered: PatchValues
  var initial: PatchValues
  var pitchFit: PitchFit
  var initLoss: Float
  var bestLoss: Float
  var bestEpoch: Int
  var losses: [Float]
}

struct FDCheckResult: Codable {
  var paramName: String
  var epsilon: Float
  var baseNaturalValue: Float
  var baseTransformedValue: Float
  var lossMinus: Float
  var lossPlus: Float
  var finiteDifferenceGrad: Float
  var autogradGrad: Float
  var absoluteError: Float
  var relativeError: Float
}

final class SynthIDTrainer {
  let config: SynthIDConfig

  init(config: SynthIDConfig) {
    self.config = config
  }

  func train(
    targetSamples rawTargetSamples: [Float],
    outDir: URL?,
    trueParams: PatchValues?,
    restartIndex: Int = 0
  ) throws -> TrainingRunResult {
    let resolvedConfig = config
    resolvedConfig.applyRuntime()

    let targetSamples = fitOrPad(
      peakNormalized(rawTargetSamples, peak: resolvedConfig.peakNormalizeTo),
      frames: resolvedConfig.frames)
    let pitchPoints = PitchTrack.extract(samples: targetSamples, sampleRate: resolvedConfig.sampleRate)
    let pitchFit = PitchTrack.fit(samples: targetSamples, sampleRate: resolvedConfig.sampleRate)
    let initial = restartInitial(pitchFit: pitchFit, restartIndex: restartIndex)

    LazyGraphContext.reset()
    let targetTensor = Tensor(targetSamples)
    let params = TrainableKickParams(
      initial: initial,
      trainable: true,
      freezePitch: resolvedConfig.freezePitch)

    let pitchOpt = Adam(
      params: params.trainableStorage(names: ["fStart", "fEnd", "pitchDecay"]),
      lr: resolvedConfig.pitchLR)
    let ampOpt = Adam(
      params: params.trainableStorage(names: ["bodyAmp", "clickAmp", "noiseAmp", "outGain"]),
      lr: resolvedConfig.ampLR)
    let decayOpt = Adam(
      params: params.trainableStorage(names: ["ampDecay", "clickDecay", "noiseDecay"]),
      lr: resolvedConfig.decayLR)
    let toneOpt = Adam(
      params: params.trainableStorage(names: ["clickFreq", "noiseCutoff", "drive"]),
      lr: resolvedConfig.toneLR)
    let optimizers = [pitchOpt, ampOpt, decayOpt, toneOpt]

    func buildTarget() -> Signal {
      targetTensor.toSignal(maxFrames: resolvedConfig.frames)
    }

    func buildLoss() -> Signal {
      let synth = KickVoice.build(params: params.signals, config: resolvedConfig)
      return SynthIDLosses.multiResolutionSpectralLoss(
        synth: synth,
        target: buildTarget(),
        config: resolvedConfig)
    }

    var initLoss: Float = .infinity
    var bestLoss: Float = .infinity
    var bestEpoch = 0
    var bestParams = initial
    var losses: [Float] = []

    if let outDir {
      try ensureDirectory(outDir)
      try writeJSON(resolvedConfig, to: outDir.appendingPathComponent("config.json"))
      try writeJSON(initial, to: outDir.appendingPathComponent("initial_params.json"))
      try writeJSON(pitchFit, to: outDir.appendingPathComponent("pitch_fit.json"))
      try writeJSON(pitchPoints, to: outDir.appendingPathComponent("pitch_points.json"))
    }

    for epoch in 0..<resolvedConfig.epochs {
      let lossValues = try buildLoss().backward(frames: resolvedConfig.frames)
      let epochLoss = lossValues.reduce(0, +)
      if epoch == 0 { initLoss = epochLoss }
      losses.append(epochLoss)

      if epochLoss < bestLoss {
        bestLoss = epochLoss
        bestEpoch = epoch
        bestParams = params.naturalValues()
        if let outDir {
          try writeCheckpoint(
            outDir: outDir,
            name: "checkpoint_best.json",
            config: resolvedConfig,
            epoch: epoch,
            loss: epochLoss,
            params: params)
        }
      }

      if resolvedConfig.logEvery > 0
        && (epoch % resolvedConfig.logEvery == 0 || epoch == resolvedConfig.epochs - 1)
      {
        print("epoch=\(epoch) loss=\(String(format: "%.6f", epochLoss))")
        for spec in KickParamSpecs.all {
          let value = params.naturalValues()[spec.name]
          let grad = params.transformedParams[spec.name]?.grad?.data ?? .nan
          print(
            "  \(spec.name)=\(String(format: "%12.5f", value)) grad=\(String(format: "%12.4e", grad))"
          )
        }
      }

      if let outDir,
        resolvedConfig.checkpointEvery > 0,
        epoch > 0,
        epoch % resolvedConfig.checkpointEvery == 0
      {
        try writeCheckpoint(
          outDir: outDir,
          name: String(format: "checkpoint_epoch_%04d.json", epoch),
          config: resolvedConfig,
          epoch: epoch,
          loss: epochLoss,
          params: params)
      }

      guard epochLoss.isFinite else {
        print("stopping: non-finite loss")
        break
      }

      params.clipGradients(maxAbs: resolvedConfig.gradClip)
      for opt in optimizers { opt.step() }
      for opt in optimizers { opt.zeroGrad() }
    }

    params.apply(natural: bestParams)

    if let outDir {
      try writeCheckpoint(
        outDir: outDir,
        name: "checkpoint.json",
        config: resolvedConfig,
        epoch: bestEpoch,
        loss: bestLoss,
        params: params)
      try writeLossCurve(losses, to: outDir.appendingPathComponent("loss_curve.csv"))
      try writeJSON(bestParams, to: outDir.appendingPathComponent("recovered_params.json"))
    }

    return TrainingRunResult(
      recovered: bestParams,
      initial: initial,
      pitchFit: pitchFit,
      initLoss: initLoss,
      bestLoss: bestLoss,
      bestEpoch: bestEpoch,
      losses: losses)
  }

  func fdcheck(
    paramName: String,
    targetSamples rawTargetSamples: [Float],
    initial explicitInitial: PatchValues? = nil,
    outDir: URL? = nil
  ) throws -> FDCheckResult {
    let resolvedConfig = config
    resolvedConfig.applyRuntime()
    guard let spec = KickParamSpecs.byName[paramName] else {
      throw SynthIDError.message("unknown fdcheck parameter \(paramName)")
    }

    let targetSamples = fitOrPad(
      peakNormalized(rawTargetSamples, peak: resolvedConfig.peakNormalizeTo),
      frames: resolvedConfig.frames)
    let initial =
      explicitInitial
      ?? restartInitial(
        pitchFit: PitchTrack.fit(samples: targetSamples, sampleRate: resolvedConfig.sampleRate),
        restartIndex: 0)
    let baseZ = spec.transform(initial[paramName])
    let eps = resolvedConfig.fdEpsilon

    var minus = initial
    minus[paramName] = spec.inverse(baseZ - eps)
    minus = minus.clamped()

    var plus = initial
    plus[paramName] = spec.inverse(baseZ + eps)
    plus = plus.clamped()

    let lossMinus = try lossFor(values: minus, targetSamples: targetSamples, config: resolvedConfig)
    let lossPlus = try lossFor(values: plus, targetSamples: targetSamples, config: resolvedConfig)
    let fdGrad = (lossPlus - lossMinus) / (2.0 * eps)
    let autograd = try autogradGradient(
      paramName: paramName,
      values: initial,
      targetSamples: targetSamples,
      config: resolvedConfig)
    let absErr = abs(fdGrad - autograd)
    let relErr = absErr / max(abs(fdGrad), abs(autograd), 1e-12)
    let result = FDCheckResult(
      paramName: paramName,
      epsilon: eps,
      baseNaturalValue: initial[paramName],
      baseTransformedValue: baseZ,
      lossMinus: lossMinus,
      lossPlus: lossPlus,
      finiteDifferenceGrad: fdGrad,
      autogradGrad: autograd,
      absoluteError: absErr,
      relativeError: relErr)
    if let outDir {
      try ensureDirectory(outDir)
      try writeJSON(result, to: outDir.appendingPathComponent("fdcheck_\(paramName).json"))
    }
    return result
  }

  private func restartInitial(pitchFit: PitchFit, restartIndex: Int) -> PatchValues {
    var values = PatchValues.midpoint.withPitch(pitchFit)
    guard restartIndex > 0 else { return values }
    var rng = SplitMix64(seed: config.seed &+ UInt64(10_000 + restartIndex * 997))
    for spec in KickParamSpecs.all {
      if ["fStart", "fEnd", "pitchDecay"].contains(spec.name) { continue }
      let span = spec.max - spec.min
      let jitter = (rng.unit() - 0.5) * 0.30 * span
      values[spec.name] = Swift.min(Swift.max(values[spec.name] + jitter, spec.min), spec.max)
    }
    if values.noiseAmp <= 0 { values.noiseAmp = 0.02 }
    if values.clickAmp <= 0 { values.clickAmp = 0.05 }
    return values.clamped()
  }

  private func writeCheckpoint(
    outDir: URL,
    name: String,
    config: SynthIDConfig,
    epoch: Int,
    loss: Float,
    params: TrainableKickParams
  ) throws {
    let checkpoint = SynthIDCheckpoint(
      createdAtUTC: timestampUTC(),
      config: config,
      epoch: epoch,
      loss: loss,
      params: params.naturalValues(),
      transformedParams: params.transformedValues())
    try writeJSON(checkpoint, to: outDir.appendingPathComponent(name))
  }

  private func writeLossCurve(_ losses: [Float], to url: URL) throws {
    var text = "epoch,loss\n"
    for (epoch, loss) in losses.enumerated() {
      text += "\(epoch),\(loss)\n"
    }
    try text.write(to: url, atomically: true, encoding: .utf8)
  }

  private func lossFor(values: PatchValues, targetSamples: [Float], config: SynthIDConfig) throws
    -> Float
  {
    LazyGraphContext.reset()
    let targetTensor = Tensor(targetSamples)
    let params = TrainableKickParams(initial: values, trainable: true, freezePitch: false)
    let loss = SynthIDLosses.multiResolutionSpectralLoss(
      synth: KickVoice.build(params: params.signals, config: config),
      target: targetTensor.toSignal(maxFrames: config.frames),
      config: config)
    return try loss.backward(frames: config.frames).reduce(0, +)
  }

  private func autogradGradient(
    paramName: String,
    values: PatchValues,
    targetSamples: [Float],
    config: SynthIDConfig
  ) throws -> Float {
    LazyGraphContext.reset()
    let targetTensor = Tensor(targetSamples)
    let params = TrainableKickParams(initial: values, trainable: true, freezePitch: false)
    let loss = SynthIDLosses.multiResolutionSpectralLoss(
      synth: KickVoice.build(params: params.signals, config: config),
      target: targetTensor.toSignal(maxFrames: config.frames),
      config: config)
    _ = try loss.backward(frames: config.frames)
    guard let grad = params.transformedParams[paramName]?.grad?.data else {
      throw SynthIDError.message("fdcheck could not read autograd gradient for \(paramName)")
    }
    return grad
  }
}
