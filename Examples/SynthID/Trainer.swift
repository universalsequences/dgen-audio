import DGenLazy
import DGenTrainProtocol
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
  var baseLoss: Float
  var lossMinus: Float
  var lossPlus: Float
  var finiteDifferenceGrad: Float
  var autogradGrad: Float
  var absoluteError: Float
  var relativeError: Float
}

struct DirectionalFDCheckResult: Codable {
  var paramName: String
  var directionEpsilon: Float
  var epsilon: Float
  var baseLoss: Float
  var lossMinus: Float
  var lossPlus: Float
  var finiteDifferenceGrad: Float
  var directionalAutogradGrad: Float
  var fullVoiceAutogradGrad: Float
  var absoluteError: Float
  var relativeError: Float
  var chainRuleRelativeError: Float
}

struct CoordinateSearchStep: Codable {
  var pass: Int
  var parameter: String
  var naturalValue: Float
  var loss: Float
}

struct CoordinateSearchResult: Codable {
  var params: PatchValues
  var loss: Float
  var steps: [CoordinateSearchStep]
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
    restartIndex: Int = 0,
    initialOverride: PatchValues? = nil
  ) throws -> TrainingRunResult {
    let resolvedConfig = config
    resolvedConfig.applyRuntime()

    let targetSamples = fitOrPad(
      peakNormalized(rawTargetSamples, peak: resolvedConfig.peakNormalizeTo),
      frames: resolvedConfig.frames)
    let pitchSearchProfile: PitchSearchProfile =
      resolvedConfig.profile == "909" ? .tr909 : .tr808
    let pitchPoints = PitchTrack.extract(samples: targetSamples, sampleRate: resolvedConfig.sampleRate)
    let pitchFit: PitchFit
    if resolvedConfig.profile == "subtractive-bass" {
      pitchFit = PitchFit(fStart: 110, fEnd: 110, pitchDecay: -1, error: nil)
    } else if resolvedConfig.profile == "monologue-bass" {
      // Steady-bass median f0 over the sustained region; becomes the frozen
      // subF0 via withPitch. 8192-sample windows cover >5 periods at 35 Hz.
      let steady = PitchTrack.extract(
        samples: targetSamples,
        sampleRate: resolvedConfig.sampleRate,
        windowSize: 8192,
        hop: 1024,
        minHz: 25,
        maxHz: 160).filter { $0.time > 0.05 && $0.time < 0.5 }
      let sorted = steady.map(\.hz).sorted()
      let f0 = sorted.isEmpty ? 110.0 : sorted[sorted.count / 2]
      pitchFit = PitchFit(fStart: f0, fEnd: f0, pitchDecay: -1, error: nil)
    } else if resolvedConfig.profile == "hoodie-bass" {
      let steady = PitchTrack.extract(
        samples: targetSamples,
        sampleRate: resolvedConfig.sampleRate,
        windowSize: 8192,
        hop: 1024,
        minHz: 25,
        maxHz: 160).filter { $0.time > 0.2 && $0.time < 1.4 }
      let sorted = steady.map(\.hz).sorted()
      let f0 = sorted.isEmpty ? 32.7 : sorted[sorted.count / 2]
      pitchFit = PitchFit(fStart: f0, fEnd: f0, pitchDecay: -1, error: nil)
    } else {
      pitchFit = PitchTrack.fit(
        samples: targetSamples, sampleRate: resolvedConfig.sampleRate, profile: pitchSearchProfile)
    }
    var initial =
      initialOverride ?? restartInitial(pitchFit: pitchFit, restartIndex: restartIndex)

    if resolvedConfig.useSmoothBasinSearch {
      guard resolvedConfig.useSmoothTrainingLoss else {
        throw SynthIDError.message("--smooth-basin-search requires --smooth-training-loss")
      }
      let searched = try smoothCoordinateSearch(initial: initial, targetSamples: targetSamples)
      initial = searched.params
      if let outDir {
        try ensureDirectory(outDir)
        try writeJSON(searched, to: outDir.appendingPathComponent("smooth_basin_search.json"))
      }
      print("  smooth basin: loss=\(String(format: "%.6f", searched.loss))")
    }

    LazyGraphContext.reset()
    let targetTensor = Tensor(targetSamples)
    let params = TrainableKickParams(
      initial: initial,
      trainable: true,
      freezePitch: resolvedConfig.freezePitch,
      freezeBodyAsymmetry: resolvedConfig.rung != 3,
      frozenNames: Set(resolvedConfig.frozenParams))

    let isBass = resolvedConfig.profile == "hoodie-bass"
    let isMono = resolvedConfig.profile == "monologue-bass"
    let isSubtractive = resolvedConfig.profile == "subtractive-bass" || isMono
    let pitchOpt = Adam(
      params: params.trainableStorage(
        names: isSubtractive ? [] : (isBass ? ["f0"] : ["fStart", "fEnd", "pitchDecay"])),
      lr: resolvedConfig.pitchLR)
    let harmonicNames = resolvedConfig.profile == "909"
      ? KickParamSpecs.tr909HarmonicCorrections.map(\.name)
      : (isBass ? KickParamSpecs.hoodieBassHarmonics.map(\.name) : [])
    let ampOpt = Adam(
      params: params.trainableStorage(
        names: (isSubtractive ? ["sustain", "outGain"] + (isMono ? ["vco2Level"] : [])
          : (isBass ? ["sustain", "outGain"]
            : ["bodyAmp", "clickAmp", "outGain", "bodyAsymmetry", "bodyHarmonic"]))
          + harmonicNames),
      lr: isSubtractive ? resolvedConfig.ampLR * 0.1 : resolvedConfig.ampLR)
    let decayOpt = Adam(
      params: params.trainableStorage(names: isSubtractive
        ? ["attackTime", "decayTime", "releaseTime", "fDecay"]
        : (isBass
          ? ["attackTime", "decayTime", "noteOff", "releaseTime", "brightnessDecay"]
          : ["ampDecay", "clickDecay", "noiseDecay", "ampCurve"])),
      lr: isSubtractive ? resolvedConfig.decayLR / 3.0 : resolvedConfig.decayLR)
    let noiseOpt = Adam(
      params: params.trainableStorage(names: isSubtractive ? [] : ["noiseAmp"]),
      lr: resolvedConfig.noiseLR)
    var toneNames = isSubtractive
      ? ["fBase", "fAmt", "res", "drive"] + (isMono ? ["satGain", "filtSat"] : [])
      : (isBass ? ["drive"] : ["clickFreq", "drive"])
    if resolvedConfig.enableNoiseFilter && !isBass && !isSubtractive {
      toneNames.append("noiseCutoff")
    }
    let toneOpt = Adam(
      params: params.trainableStorage(names: toneNames),
      lr: resolvedConfig.toneLR)
    // Raw oscillator mix/width gradients are much larger than transformed
    // filter gradients. A dedicated slower group prevents pw/shape from
    // sprinting to a compensating bound before the filter basin settles.
    let oscillatorOpt = Adam(
      params: params.trainableStorage(
        names: isSubtractive
          ? ["shape", "pw"]
            + (isMono ? ["satBias", "satA2", "satA3", "satA5", "vco2Detune"] : [])
          : []),
      lr: resolvedConfig.toneLR * 0.1)
    let optimizers = [pitchOpt, ampOpt, decayOpt, noiseOpt, toneOpt, oscillatorOpt]
    let baseLRs = optimizers.map(\.lr)

    func buildTarget() -> Signal {
      targetTensor.toSignal(maxFrames: resolvedConfig.frames)
    }

    func buildLoss() -> Signal {
      let synth = KickVoice.build(params: params, config: resolvedConfig)
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
      if resolvedConfig.cosineLRDecay {
        // Adam's late-training oscillation floor scales with lr; anneal so the
        // final epochs settle into the minimum instead of orbiting it.
        let progress = Float(epoch) / Float(max(1, resolvedConfig.epochs - 1))
        let scale = 0.05 + 0.95 * 0.5 * (1.0 + Foundation.cos(Float.pi * progress))
        for (opt, base) in zip(optimizers, baseLRs) { opt.lr = base * scale }
      }
      for opt in optimizers { opt.step() }
      for opt in optimizers { opt.zeroGrad() }
    }

    params.apply(natural: bestParams)

    // Pitch refinement: the log-magnitude loss is far sharper in the pitch trio
    // than in any other parameter (sub-1% pitch error can carry ~90% of the
    // residual loss). Descend the 3-D pitch subspace alone from the best point;
    // the other groups stay fixed so their basins cannot be disturbed.
    if resolvedConfig.pitchRefineEpochs > 0 && !resolvedConfig.freezePitch {
      let refineBase = resolvedConfig.pitchLR * 0.3
      for epoch in 0..<resolvedConfig.pitchRefineEpochs {
        let lossValues = try buildLoss().backward(frames: resolvedConfig.frames)
        let epochLoss = lossValues.reduce(0, +)
        losses.append(epochLoss)
        if epochLoss < bestLoss {
          bestLoss = epochLoss
          bestEpoch = resolvedConfig.epochs + epoch
          bestParams = params.naturalValues()
        }
        guard epochLoss.isFinite else { break }
        params.clipGradients(maxAbs: resolvedConfig.gradClip)
        let progress = Float(epoch) / Float(max(1, resolvedConfig.pitchRefineEpochs - 1))
        pitchOpt.lr = refineBase * (0.02 + 0.98 * 0.5 * (1.0 + Foundation.cos(Float.pi * progress)))
        pitchOpt.step()
        for opt in optimizers { opt.zeroGrad() }
        if resolvedConfig.logEvery > 0 && epoch % resolvedConfig.logEvery == 0 {
          print("refine epoch=\(epoch) loss=\(String(format: "%.6f", epochLoss))")
        }
      }
      params.apply(natural: bestParams)
    }

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
        pitchFit: PitchTrack.fit(
          samples: targetSamples,
          sampleRate: resolvedConfig.sampleRate,
          profile: resolvedConfig.profile == "909" ? .tr909 : .tr808),
        restartIndex: 0)
    let baseZ = spec.transform(initial[paramName])
    let eps = resolvedConfig.fdEpsilon

    var minus = initial
    minus[paramName] = spec.inverse(baseZ - eps)
    minus = minus.clamped()

    var plus = initial
    plus[paramName] = spec.inverse(baseZ + eps)
    plus = plus.clamped()

    let smoothProbe = resolvedConfig.fdcheckLogMagnitudeL2 == true
    let timeProbe = resolvedConfig.fdcheckTimeMSE == true
    let baseLoss = try lossFor(
      values: initial, targetSamples: targetSamples, config: resolvedConfig,
      smoothProbe: smoothProbe, timeProbe: timeProbe)
    let lossMinus = try lossFor(
      values: minus, targetSamples: targetSamples, config: resolvedConfig,
      smoothProbe: smoothProbe, timeProbe: timeProbe)
    let lossPlus = try lossFor(
      values: plus, targetSamples: targetSamples, config: resolvedConfig,
      smoothProbe: smoothProbe, timeProbe: timeProbe)
    let fdGrad = (lossPlus - lossMinus) / (2.0 * eps)
    let autograd = try autogradGradient(
      paramName: paramName,
      values: initial,
      targetSamples: targetSamples,
      config: resolvedConfig,
      smoothProbe: smoothProbe,
      timeProbe: timeProbe)
    let absErr = abs(fdGrad - autograd)
    let relErr = absErr / max(abs(fdGrad), abs(autograd), 1e-12)
    let result = FDCheckResult(
      paramName: paramName,
      epsilon: eps,
      baseNaturalValue: initial[paramName],
      baseTransformedValue: baseZ,
      baseLoss: baseLoss,
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

  /// Isolate the spectral-loss input adjoint along the actual rendered voice's
  /// parameter tangent. The synth is used only to render x and
  /// v = d(x)/d(z); the checked graph is x + alpha*v -> spectral loss.
  func directionalFDCheck(
    paramName: String,
    targetSamples rawTargetSamples: [Float],
    initial: PatchValues,
    outDir: URL? = nil
  ) throws -> DirectionalFDCheckResult {
    let resolvedConfig = config
    resolvedConfig.applyRuntime()
    guard resolvedConfig.fdcheckLogMagnitudeL2 == true else {
      throw SynthIDError.message("directional fdcheck requires --fdcheck-log-l2")
    }
    guard let spec = KickParamSpecs.byName[paramName] else {
      throw SynthIDError.message("unknown fdcheck parameter \(paramName)")
    }

    let targetSamples = fitOrPad(
      peakNormalized(rawTargetSamples, peak: resolvedConfig.peakNormalizeTo),
      frames: resolvedConfig.frames)
    let baseZ = spec.transform(initial[paramName])
    let directionEps = resolvedConfig.directionEpsilon
    guard directionEps > 0 else {
      throw SynthIDError.message("--direction-eps must be positive")
    }

    var directionMinus = initial
    directionMinus[paramName] = spec.inverse(baseZ - directionEps)
    directionMinus = directionMinus.clamped()
    var directionPlus = initial
    directionPlus[paramName] = spec.inverse(baseZ + directionEps)
    directionPlus = directionPlus.clamped()

    let baseSignal = try KickVoice.render(values: initial, config: resolvedConfig)
    let minusSignal = try KickVoice.render(values: directionMinus, config: resolvedConfig)
    let plusSignal = try KickVoice.render(values: directionPlus, config: resolvedConfig)
    let tangent = zip(plusSignal, minusSignal).map { ($0 - $1) / (2 * directionEps) }

    func directionalLoss(alphaValue: Float, backward: Bool) throws -> (Float, Float?) {
      // Tensor-after-reset is essential: these three tensor node IDs must
      // belong to the graph that consumes them.
      LazyGraphContext.reset()
      let baseTensor = Tensor(baseSignal)
      let tangentTensor = Tensor(tangent)
      let targetTensor = Tensor(targetSamples)
      let alpha = Signal.param(alphaValue)
      let student = baseTensor.toSignal(maxFrames: resolvedConfig.frames)
        + tangentTensor.toSignal(maxFrames: resolvedConfig.frames) * alpha
      let target = targetTensor.toSignal(maxFrames: resolvedConfig.frames)
      let loss = SynthIDLosses.fdcheckLogMagnitudeL2Loss(
        synth: student, target: target, config: resolvedConfig)
      if backward {
        let value = try loss.backward(frames: resolvedConfig.frames).reduce(0, +)
        return (value, alpha.grad?.data)
      }
      return (try loss.realize(frames: resolvedConfig.frames).reduce(0, +), nil)
    }

    let eps = resolvedConfig.fdEpsilon
    let (baseLoss, directionalAutograd) = try directionalLoss(alphaValue: 0, backward: true)
    guard let directionalAutograd else {
      throw SynthIDError.message("directional fdcheck could not read alpha gradient")
    }
    let (lossMinus, _) = try directionalLoss(alphaValue: -eps, backward: false)
    let (lossPlus, _) = try directionalLoss(alphaValue: eps, backward: false)
    let fdGrad = (lossPlus - lossMinus) / (2 * eps)
    let fullVoiceAutograd = try autogradGradient(
      paramName: paramName,
      values: initial,
      targetSamples: targetSamples,
      config: resolvedConfig,
      smoothProbe: true)
    let absErr = abs(fdGrad - directionalAutograd)
    let relErr = absErr / max(abs(fdGrad), abs(directionalAutograd), 1e-12)
    let chainRuleRelErr = abs(directionalAutograd - fullVoiceAutograd)
      / max(abs(directionalAutograd), abs(fullVoiceAutograd), 1e-12)
    let result = DirectionalFDCheckResult(
      paramName: paramName,
      directionEpsilon: directionEps,
      epsilon: eps,
      baseLoss: baseLoss,
      lossMinus: lossMinus,
      lossPlus: lossPlus,
      finiteDifferenceGrad: fdGrad,
      directionalAutogradGrad: directionalAutograd,
      fullVoiceAutogradGrad: fullVoiceAutograd,
      absoluteError: absErr,
      relativeError: relErr,
      chainRuleRelativeError: chainRuleRelErr)

    if let outDir {
      try ensureDirectory(outDir)
      try writeJSON(
        result, to: outDir.appendingPathComponent("directional_fdcheck_\(paramName).json"))
      try writeFloat32(baseSignal, to: outDir.appendingPathComponent("base_signal.f32"))
      try writeFloat32(tangent, to: outDir.appendingPathComponent("\(paramName)_tangent.f32"))
      try writeFloat32(targetSamples, to: outDir.appendingPathComponent("target_signal.f32"))
    }
    return result
  }

  private func writeFloat32(_ values: [Float], to url: URL) throws {
    let data = values.withUnsafeBufferPointer { buffer in
      Data(bytes: buffer.baseAddress!, count: buffer.count * MemoryLayout<Float>.stride)
    }
    try data.write(to: url, options: .atomic)
  }

  // Internal (not private) so `recoverTarget` in main.swift can rebuild the
  // deterministic restart-0 reference initialization regardless of which
  // restart's own cold start actually won selection (rung 3 FIX 1).
  func restartInitial(pitchFit: PitchFit, restartIndex: Int) -> PatchValues {
    var values = PatchValues.midpoint.withPitch(pitchFit)
    if config.profile == "monologue-bass" {
      if let noteOff = config.subNoteOffOverride { values.subNoteOff = noteOff }
    }
    if config.profile == "subtractive-bass" || config.profile == "monologue-bass" {
      func set(_ name: String, _ fraction: Float) {
        guard let spec = KickParamSpecs.byName[name] else { return }
        let bounds = spec.transformedBounds
        values[name] = spec.inverse(bounds.min + fraction * (bounds.max - bounds.min))
      }
      let templates: [[String: Float]] = [
        [:],
        ["shape": 0.30, "pw": 0.35, "fBase": 0.35, "fAmt": 0.65,
         "fDecay": 0.35, "res": 0.35, "attackTime": 0.35, "decayTime": 0.35,
         "sustain": 0.65, "releaseTime": 0.35, "drive": 0.40, "outGain": 0.60],
        ["shape": 0.70, "pw": 0.65, "fBase": 0.80, "fAmt": 0.45,
         "fDecay": 0.65, "res": 0.65, "attackTime": 0.65, "decayTime": 0.65,
         "sustain": 0.35, "releaseTime": 0.65, "drive": 0.60, "outGain": 0.40],
        ["shape": 0.65, "pw": 0.25, "fBase": 0.60, "fAmt": 0.75,
         "fDecay": 0.25, "res": 0.70, "attackTime": 0.25, "decayTime": 0.70,
         "sustain": 0.55, "releaseTime": 0.30, "drive": 0.70, "outGain": 0.45],
      ]
      if restartIndex < templates.count {
        for (name, fraction) in templates[restartIndex] { set(name, fraction) }
      } else {
        var rng = SplitMix64(seed: config.seed &+ UInt64(10_000 + restartIndex * 997))
        for spec in KickParamSpecs.all {
          let bounds = spec.transformedBounds
          let fraction = rng.uniform(0.15, 0.85)
          values[spec.name] = spec.inverse(
            bounds.min + fraction * (bounds.max - bounds.min))
        }
      }
      return values.clamped()
    }
    if config.profile == "hoodie-bass" {
      let brightnessScales: [Float] = [1.0, 0.55, 1.65, 1.0]
      let attackScales: [Float] = [1.0, 0.6, 1.5, 1.0]
      if restartIndex < brightnessScales.count {
        values.brightnessDecay *= brightnessScales[restartIndex]
        values.attackTime *= attackScales[restartIndex]
        if restartIndex == 3 {
          for spec in KickParamSpecs.hoodieBassHarmonics where spec.cosine {
            values[spec.name] = 0.03 / Float(spec.harmonic)
          }
        }
      }
      return values.clamped()
    }
    // Window smearing makes the CPU contour fit see a slower, lower sweep, so
    // it underestimates |pitchDecay| and fStart together (worst on fast
    // sweeps). Bracket both coherently across the first three restarts, all
    // with midpoint amps so best-of-N loss selection compares pitch basins on
    // equal footing. fEnd stays anchored — the tail measurement is reliable.
    // r1 (mild) and r3 (strong) both correct the fast-sweep underestimate;
    // which lands depends on how fast the true sweep is, and best-of-N picks.
    //
    // The 909 profile's raw CPU pitch fit already tracks the measured sweep
    // well (see rung-3 909 diagnosis), so the 808 brackets are too aggressive
    // there and bias restart selection toward inflated cold starts. Use
    // gentler brackets centered on the raw fit, and since restarts 0 and 3
    // then coincide in pitch space, give restart 3 diversity through
    // non-pitch params instead (click amp/decay bracketed around the spec
    // midpoint).
    let pdScales: [Float]
    let fStartScales: [Float]
    if config.profile == "909" {
      pdScales = [1.00, 1.15, 0.85, 1.00]
      fStartScales = [1.00, 1.06, 0.94, 1.00]
    } else {
      pdScales = [1.0, 1.45, 0.75, 1.45]
      fStartScales = [1.0, 1.10, 0.92, 1.22]
    }
    if restartIndex < pdScales.count {
      values.pitchDecay *= pdScales[restartIndex]
      values.fStart *= fStartScales[restartIndex]
      if config.profile == "909" && restartIndex == 3 {
        if let clickAmpSpec = KickParamSpecs.byName["clickAmp"] {
          values.clickAmp = clickAmpSpec.midpoint * 0.5
        }
        if let clickDecaySpec = KickParamSpecs.byName["clickDecay"] {
          values.clickDecay = clickDecaySpec.midpoint * 1.5
        }
      }
    } else {
      var rng = SplitMix64(seed: config.seed &+ UInt64(10_000 + restartIndex * 997))
      for spec in KickParamSpecs.all {
        if ["fStart", "fEnd", "pitchDecay"].contains(spec.name) { continue }
        values[spec.name] = rng.uniform(spec.min, spec.max)
      }
      values.pitchDecay *= Foundation.exp(rng.uniform(-0.4, 0.4))
      values.fStart *= Foundation.exp(rng.uniform(-0.1, 0.1))
      if values.noiseAmp <= 0 { values.noiseAmp = 0.02 }
      if values.clickAmp <= 0 { values.clickAmp = 0.05 }
    }
    if config.rung == 3 && restartIndex == 4 && config.profile != "909" {
      // A dedicated, target-independent capture-floor hypothesis. PCM targets
      // can contain persistent broadband energy that the original -60/s noise
      // bound made structurally unreachable. The 909 recording's noise floor
      // is dead (measured), so this restart would be wasted there — fall
      // through to a plain midpoint restart instead.
      values.noiseAmp = 0.0001
      values.noiseDecay = -0.1
      values.noiseCutoff = 10_000
      values.bodyAsymmetry = 0
    }
    // Never initialize a param ON its trainable bound: projected Adam plus
    // compensation by other params forms a sticky local minimum there
    // (observed: fit pd = -15 exactly, recovery pinned at -15 all run).
    values.pitchDecay = Swift.min(Swift.max(values.pitchDecay, -76), -17)
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

  /// Evaluate the training loss for explicit parameter values against a target
  /// already normalized/padded the way `train` does it.
  func evaluateLoss(values: PatchValues, targetSamples rawTargetSamples: [Float]) throws -> Float {
    let resolvedConfig = config
    resolvedConfig.applyRuntime()
    let targetSamples = fitOrPad(
      peakNormalized(rawTargetSamples, peak: resolvedConfig.peakNormalizeTo),
      frames: resolvedConfig.frames)
    return try lossFor(values: values, targetSamples: targetSamples, config: resolvedConfig)
  }

  /// Target-independent coordinate basin search for the smooth E1 objective.
  /// The first pass covers each declared transformed bound; the second
  /// re-scans a local neighborhood. It is deliberately derivative-free so it
  /// can place sparse moving-edge controls in the right basin before Adam.
  func smoothCoordinateSearch(
    initial: PatchValues,
    targetSamples: [Float]
  ) throws -> CoordinateSearchResult {
    precondition(config.useSmoothTrainingLoss)
    let orderedNames = [
      "shape", "pw", "fBase", "fAmt", "fDecay", "res",
      "attackTime", "decayTime", "sustain", "releaseTime", "drive", "outGain",
    ]
    var best = initial.clamped()
    var bestLoss = try evaluateLoss(values: best, targetSamples: targetSamples)
    var trace: [CoordinateSearchStep] = []

    for pass in 0..<2 {
      for name in orderedNames {
        guard let spec = KickParamSpecs.byName[name] else { continue }
        let bounds = spec.transformedBounds
        let span = bounds.max - bounds.min
        let inset = span * 0.001
        let center = spec.transform(best[name])
        let radius = pass == 0 ? span : span * 0.06
        let lower = pass == 0
          ? bounds.min + inset
          : max(bounds.min + inset, center - radius)
        let upper = pass == 0
          ? bounds.max - inset
          : min(bounds.max - inset, center + radius)
        let points: Int
        if name == "pw" {
          points = pass == 0 ? 257 : 129
        } else if name == "shape" {
          points = pass == 0 ? 65 : 49
        } else {
          points = pass == 0 ? 33 : 25
        }

        var parameterBest = best
        var parameterLoss = bestLoss
        for index in 0..<points {
          let fraction = Float(index) / Float(max(1, points - 1))
          var candidate = best
          candidate[name] = spec.inverse(lower + (upper - lower) * fraction)
          let loss = try evaluateLoss(values: candidate, targetSamples: targetSamples)
          if loss < parameterLoss {
            parameterBest = candidate
            parameterLoss = loss
          }
        }
        best = parameterBest
        bestLoss = parameterLoss
        trace.append(
          CoordinateSearchStep(
            pass: pass,
            parameter: name,
            naturalValue: best[name],
            loss: bestLoss))
        print(
          "  basin pass=\(pass) \(name)=\(String(format: "%.6g", best[name]))"
            + " loss=\(String(format: "%.6f", bestLoss))")
      }
    }
    return CoordinateSearchResult(params: best, loss: bestLoss, steps: trace)
  }

  private func lossFor(
    values: PatchValues,
    targetSamples: [Float],
    config: SynthIDConfig,
    smoothProbe: Bool = false,
    timeProbe: Bool = false
  ) throws
    -> Float
  {
    LazyGraphContext.reset()
    let targetTensor = Tensor(targetSamples)
    let params = TrainableKickParams(initial: values, trainable: true, freezePitch: false)
    let synth = KickVoice.build(params: params, config: config)
    let target = targetTensor.toSignal(maxFrames: config.frames)
    let loss = timeProbe
      ? mse(synth, target)
      : (smoothProbe
        ? SynthIDLosses.fdcheckLogMagnitudeL2Loss(synth: synth, target: target, config: config)
        : SynthIDLosses.multiResolutionSpectralLoss(synth: synth, target: target, config: config))
    // Finite-difference/evaluation loss reads must be forward-only. Running
    // backward here adds gradient side effects to the supposedly independent
    // numerical measurement and can perturb memory scheduling/allocation.
    return try loss.realize(frames: config.frames).reduce(0, +)
  }

  private func autogradGradient(
    paramName: String,
    values: PatchValues,
    targetSamples: [Float],
    config: SynthIDConfig,
    smoothProbe: Bool = false,
    timeProbe: Bool = false
  ) throws -> Float {
    LazyGraphContext.reset()
    let targetTensor = Tensor(targetSamples)
    let params = TrainableKickParams(initial: values, trainable: true, freezePitch: false)
    let synth = KickVoice.build(params: params, config: config)
    let target = targetTensor.toSignal(maxFrames: config.frames)
    let loss = timeProbe
      ? mse(synth, target)
      : (smoothProbe
        ? SynthIDLosses.fdcheckLogMagnitudeL2Loss(synth: synth, target: target, config: config)
        : SynthIDLosses.multiResolutionSpectralLoss(synth: synth, target: target, config: config))
    _ = try loss.backward(frames: config.frames)
    guard let grad = params.transformedParams[paramName]?.grad?.data else {
      throw SynthIDError.message("fdcheck could not read autograd gradient for \(paramName)")
    }
    return grad
  }
}
