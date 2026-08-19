import DGen
import DGenLazy
import Foundation

struct ModalScoreCalibration: Codable, Equatable {
  var selfScore: Float
  var silenceScore: Float
  var whiteNoiseScore: Float
  var wrongSnareScore: Float
  var numericGate: Float
  var gateDerivation: String
}

struct RealSnareFitResult: Codable, Equatable {
  var modes: Int
  var includesNoise: Bool
  var initialScore: Float
  var bestScore: Float
  var bestStep: Int
  var wallClockSeconds: Double
  var modalEnergyFraction: Float
  var noiseEnergyFraction: Float
  var passesNumericGate: Bool
}

struct RealSnareSweepSummary: Codable, Equatable {
  var targetPath: String
  var wrongSnarePath: String
  var calibration: ModalScoreCalibration
  var fits: [RealSnareFitResult]
  var smallestPassingModes: Int?
  var winningModes: Int?
  var winningIncludesNoise: Bool?
  var noiseAblationPassed: Bool
  var listenGate: String
}

enum RealSnareFitError: Error, CustomStringConvertible {
  case invalidInput(String)

  var description: String {
    switch self {
    case .invalidInput(let message): return message
    }
  }
}

enum RealSnareFitter {
  /// Calibrates the independent score on percussion. Exact self-comparison is
  /// necessarily zero, so the automatic gate is one third below the nearest
  /// negative control: a passing fit must be at least 1.5x closer to the target
  /// than any of the controls.
  ///
  /// Silence is one of those controls deliberately. On a 0.75 s percussion
  /// window most frames are decayed tail, which compresses this metric hard:
  /// measured on the prepared corpus a *wrong snare* scores ~0.21 and pure
  /// silence only ~0.34, so a gate derived from the loud controls alone can
  /// sit above "render nothing". Including silence in the negative set makes
  /// the gate un-gameable by an empty render by construction, and records the
  /// metric's real dynamic range on this material.
  static func calibrate(
    target: [Float], wrongSnare: [Float], config: ModalDrumConfig
  ) throws -> ModalScoreCalibration {
    let white = deterministicWhiteNoise(
      count: target.count, rms: rms(target), seed: config.seed)
    let silence = [Float](repeating: 0, count: target.count)
    guard
      let selfScore = score(target, target, config),
      let silenceScore = score(silence, target, config),
      let whiteScore = score(white, target, config),
      let wrongScore = score(wrongSnare, target, config)
    else {
      throw RealSnareFitError.invalidInput("audio is too short for the MR-STFT windows")
    }
    let nearestNegative = min(silenceScore, min(whiteScore, wrongScore))
    return ModalScoreCalibration(
      selfScore: selfScore,
      silenceScore: silenceScore,
      whiteNoiseScore: whiteScore,
      wrongSnareScore: wrongScore,
      numericGate: selfScore + (nearestNegative - selfScore) / 1.5,
      gateDerivation: "self + (min(silence, whiteNoise, wrongSnare) - self) / 1.5")
  }

  static func runSweep(
    targetURL: URL,
    wrongSnareURL: URL,
    modeCounts: [Int],
    config input: ModalDrumConfig,
    runDirectory: URL,
    numericGateOverride: Float? = nil,
    kernelDumpPath: String? = nil
  ) throws -> RealSnareSweepSummary {
    var config = input
    config.frames = Int((0.75 * config.sampleRate).rounded())
    let target = try loadPreparedOneShot(targetURL, config: config)
    let wrong = try loadPreparedOneShot(wrongSnareURL, config: config)
    var calibration = try calibrate(target: target, wrongSnare: wrong, config: config)
    if let gate = numericGateOverride {
      calibration.numericGate = gate
      calibration.gateDerivation = "explicit --gate override"
    }

    try FileManager.default.createDirectory(at: runDirectory, withIntermediateDirectories: true)
    try writeJSON(calibration, to: runDirectory.appendingPathComponent("calibration.json"))
    try AudioFile.save(
      url: runDirectory.appendingPathComponent("target.wav"), samples: target,
      sampleRate: config.sampleRate)

    let modes = Array(Set(modeCounts)).filter { $0 > 0 }.sorted()
    guard !modes.isEmpty else {
      throw RealSnareFitError.invalidInput(
        "the K sweep must contain at least one positive mode count")
    }
    var fits: [RealSnareFitResult] = []
    for k in modes {
      for includesNoise in [false, true] {
        var fitConfig = config
        fitConfig.modes = k
        let name = String(format: "k%03d_%@", k, includesNoise ? "modal_noise" : "modal_only")
        let directory = runDirectory.appendingPathComponent(name, isDirectory: true)
        let dump = kernelDumpPath.map { "\($0)_\(name).metal" }
        let fit = try fit(
          target: target, config: fitConfig, includesNoise: includesNoise,
          numericGate: calibration.numericGate, runDirectory: directory,
          kernelDumpPath: dump)
        fits.append(fit)
      }
    }

    let passing = fits.filter(\.passesNumericGate)
    let winner = passing.min { lhs, rhs in
      lhs.modes == rhs.modes ? lhs.bestScore < rhs.bestScore : lhs.modes < rhs.modes
    }
    let noiseAblationPassed = modes.allSatisfy { k in
      guard
        let modal = fits.first(where: { $0.modes == k && !$0.includesNoise }),
        let both = fits.first(where: { $0.modes == k && $0.includesNoise })
      else { return false }
      return both.bestScore < modal.bestScore
    }
    let summary = RealSnareSweepSummary(
      targetPath: targetURL.path, wrongSnarePath: wrongSnareURL.path,
      calibration: calibration, fits: fits,
      smallestPassingModes: passing.map(\.modes).min(),
      winningModes: winner?.modes, winningIncludesNoise: winner?.includesNoise,
      noiseAblationPassed: noiseAblationPassed,
      listenGate: "PENDING: A/B full.wav against target.wav and audition modal.wav/noise.wav")
    try writeJSON(summary, to: runDirectory.appendingPathComponent("summary.json"))
    return summary
  }

  /// Per-mode gains warm-started from the target's spectral envelope, then
  /// scaled so the *summed* bank starts at the target's level.
  ///
  /// The envelope alone only fixes the shape. Without the scale step every
  /// mode near the spectral peak starts at the 0.9 cap, so the initial render
  /// overshoots (measured: peak 2.11 and 2.8x target RMS at K=32) and
  /// overshoots harder as K grows — which would bias the K sweep against large
  /// K for a reason that has nothing to do with model capacity. The closed
  /// form is exact for this synth: for y = sum_k g_k sin(2 pi f_k t) e^(-t/tau),
  /// mean square = (sum_k g_k^2 / 2) * (tau / 2T) * (1 - e^(-2T/tau)).
  static func spectralEnvelopeWarmStart(
    samples: [Float], frequencies: [Float], sampleRate: Float,
    decaySeconds: Float = 0.15, durationSeconds: Float = 0.75
  ) -> [Float] {
    guard !samples.isEmpty, !frequencies.isEmpty else { return [] }
    let analysisCount = min(samples.count, 8_192)
    var magnitudes = [Float](repeating: 0, count: frequencies.count)
    for (frequencyIndex, frequency) in frequencies.enumerated() {
      let omega = 2 * Double.pi * Double(frequency) / Double(sampleRate)
      var re = 0.0
      var im = 0.0
      for i in 0..<analysisCount {
        let window = 0.5 - 0.5 * cos(2 * Double.pi * Double(i) / Double(max(1, analysisCount)))
        let value = Double(samples[i]) * window
        re += value * cos(omega * Double(i))
        im -= value * sin(omega * Double(i))
      }
      magnitudes[frequencyIndex] = Float(sqrt(re * re + im * im))
    }
    let peak = max(magnitudes.max() ?? 0, 1e-12)
    let shape = magnitudes.map { max(0.005, $0 / peak) }
    let tau = max(1e-4, decaySeconds)
    let duration = max(1e-4, durationSeconds)
    let envelopeMeanSquare =
      (tau / (2 * duration)) * (1 - Foundation.exp(-2 * duration / tau))
    let unitMeanSquare = shape.reduce(0) { $0 + $1 * $1 } / 2 * envelopeMeanSquare
    let scale = rms(samples) / max(Foundation.sqrt(unitMeanSquare), 1e-12)
    return shape.map { min(0.9, max(1e-4, scale * $0)) }
  }

  private static func fit(
    target: [Float],
    config: ModalDrumConfig,
    includesNoise: Bool,
    numericGate: Float,
    runDirectory: URL,
    kernelDumpPath: String?
  ) throws -> RealSnareFitResult {
    let started = Date()
    for path in [
      runDirectory, runDirectory.appendingPathComponent("checkpoints"),
      runDirectory.appendingPathComponent("previews"),
    ] {
      try FileManager.default.createDirectory(at: path, withIntermediateDirectories: true)
    }
    config.applyRuntime(kernelOutputPath: kernelDumpPath)
    let frequenciesData = modalFrequencyGrid(count: config.modes)
    var initial = flatInitialPatch(modes: config.modes)
    initial.gains = spectralEnvelopeWarmStart(
      samples: target, frequencies: frequenciesData, sampleRate: config.sampleRate,
      decaySeconds: initial.decaySeconds.first ?? 0.15,
      durationSeconds: Float(config.frames) / config.sampleRate)
    if !includesNoise { initial.noiseGain = 0.00001 }

    LazyGraphContext.reset()
    let targetTensor = Tensor(target)
    let frequencies = Tensor(frequenciesData)
    let highModeMask = Tensor(frequenciesData.map { $0 > 6_000 ? 1 : 0 })
    let params = ModalDrumParameters(patch: initial, trainable: true)
    let optimized: [any LazyValue] =
      includesNoise
      ? params.all : [params.rawGains, params.rawModeTaus]
    let optimizer = Adam(params: optimized, lr: config.learningRate)

    try writeJSON(config, to: runDirectory.appendingPathComponent("resolved_config.json"))
    try writeJSON(initial, to: runDirectory.appendingPathComponent("initial_params.json"))
    let initialPreview = try ModalDrumSynth.render(
      params: params, frequencies: frequencies, includeModal: true,
      includeNoise: includesNoise
    ).realize(frames: config.frames)
    guard let initialScore = score(initialPreview, target, config) else {
      throw RealSnareFitError.invalidInput("audio is too short for the MR-STFT windows")
    }
    try AudioFile.save(
      url: runDirectory.appendingPathComponent("previews/initial.wav"), samples: initialPreview,
      sampleRate: config.sampleRate)
    var bestScore = initialScore
    var bestStep = -1
    var bestPatch = initial
    try ModalCheckpointStore.write(
      ModalDrumCheckpoint(
        step: bestStep, loss: bestScore,
        createdAtUTC: ISO8601DateFormatter().string(from: Date()), patch: bestPatch),
      to: runDirectory.appendingPathComponent("checkpoints/model_best.json"))
    var csv = "step,training_loss,selection_score\n"

    for step in 0..<max(1, config.steps) {
      LazyGraphContext.current.clearComputationGraph()
      let prediction = ModalDrumSynth.render(
        params: params, frequencies: frequencies, includeNoise: includesNoise)
      var loss = ModalDrumLosses.trainingLoss(
        prediction: prediction, target: targetTensor.toSignal(maxFrames: config.frames),
        config: config)
      let highModePenalty = (params.gains * highModeMask).sum() / Float(max(1, config.modes))
      let highModeLoss = (Tensor([0]) + highModePenalty * config.highModeL1Weight).peek(
        Signal.constant(0))
      loss = loss + highModeLoss
      let values = try loss.backward(frames: config.frames)
      let trainingLoss = values.reduce(0, +) / Float(max(1, values.count))
      let progress = Float(step) / Float(max(1, config.steps - 1))
      optimizer.lr =
        config.learningRate
        * max(0.01, 0.5 * (1 + Foundation.cos(Float.pi * progress)))
      optimizer.step()
      optimizer.zeroGrad()

      var scoreText = ""
      let artifactStep =
        step == 0 || step == config.steps - 1
        || (step + 1) % max(1, config.renderEvery) == 0
      if artifactStep {
        LazyGraphContext.current.clearComputationGraph()
        let preview = try ModalDrumSynth.render(
          params: params, frequencies: frequencies, includeModal: true,
          includeNoise: includesNoise
        ).realize(frames: config.frames)
        if let selection = score(preview, target, config) {
          scoreText = "\(selection)"
          if selection < bestScore {
            bestScore = selection
            bestStep = step
            bestPatch = params.naturalPatch()
            try ModalCheckpointStore.write(
              ModalDrumCheckpoint(
                step: step, loss: selection,
                createdAtUTC: ISO8601DateFormatter().string(from: Date()), patch: bestPatch),
              to: runDirectory.appendingPathComponent("checkpoints/model_best.json"))
          }
        }
        try AudioFile.save(
          url: runDirectory.appendingPathComponent(String(format: "previews/step_%06d.wav", step)),
          samples: preview, sampleRate: config.sampleRate)
      }
      if (step + 1) % max(1, config.checkpointEvery) == 0 || step == config.steps - 1 {
        try ModalCheckpointStore.write(
          ModalDrumCheckpoint(
            step: step, loss: trainingLoss,
            createdAtUTC: ISO8601DateFormatter().string(from: Date()), patch: params.naturalPatch()),
          to: runDirectory.appendingPathComponent(
            String(format: "checkpoints/step_%06d.json", step)))
      }
      csv += "\(step),\(trainingLoss),\(scoreText)\n"
    }

    let full = try render(
      patch: bestPatch, config: config, includeModal: true, includeNoise: includesNoise)
    let modal = try render(
      patch: bestPatch, config: config, includeModal: true, includeNoise: false)
    let noise =
      includesNoise
      ? try render(patch: bestPatch, config: config, includeModal: false, includeNoise: true)
      : [Float](repeating: 0, count: config.frames)
    try AudioFile.save(
      url: runDirectory.appendingPathComponent("full.wav"), samples: full,
      sampleRate: config.sampleRate)
    try AudioFile.save(
      url: runDirectory.appendingPathComponent("modal.wav"), samples: modal,
      sampleRate: config.sampleRate)
    try AudioFile.save(
      url: runDirectory.appendingPathComponent("noise.wav"), samples: noise,
      sampleRate: config.sampleRate)
    try csv.write(
      to: runDirectory.appendingPathComponent("loss.csv"), atomically: true, encoding: .utf8)
    try writeJSON(bestPatch, to: runDirectory.appendingPathComponent("best_params.json"))

    let modalEnergy = energy(modal)
    let noiseEnergy = energy(noise)
    let branchEnergy = max(modalEnergy + noiseEnergy, 1e-20)
    let result = RealSnareFitResult(
      modes: config.modes, includesNoise: includesNoise,
      initialScore: initialScore, bestScore: bestScore, bestStep: bestStep,
      wallClockSeconds: Date().timeIntervalSince(started),
      modalEnergyFraction: modalEnergy / branchEnergy,
      noiseEnergyFraction: noiseEnergy / branchEnergy,
      passesNumericGate: bestScore <= numericGate)
    try writeJSON(result, to: runDirectory.appendingPathComponent("summary.json"))
    return result
  }

  private static func render(
    patch: ModalPatch, config: ModalDrumConfig, includeModal: Bool, includeNoise: Bool
  ) throws -> [Float] {
    config.applyRuntime()
    LazyGraphContext.reset()
    let frequencies = Tensor(modalFrequencyGrid(count: patch.gains.count))
    let params = ModalDrumParameters(patch: patch, trainable: false)
    return try ModalDrumSynth.render(
      params: params, frequencies: frequencies, includeModal: includeModal,
      includeNoise: includeNoise
    ).realize(frames: config.frames)
  }

  private static func loadPreparedOneShot(_ url: URL, config: ModalDrumConfig) throws -> [Float] {
    let loaded = try AudioFile.load(url: url, mono: true)
    guard abs(loaded.sampleRate - config.sampleRate) < 0.5 else {
      throw RealSnareFitError.invalidInput(
        "\(url.path) is \(loaded.sampleRate) Hz; prepare M1 inputs at \(config.sampleRate) Hz")
    }
    guard loaded.samples.count == config.frames else {
      throw RealSnareFitError.invalidInput(
        "\(url.path) has \(loaded.samples.count) frames; expected fixed 0.75 s (\(config.frames) frames)"
      )
    }
    return loaded.samples
  }

  private static func score(
    _ prediction: [Float], _ target: [Float], _ config: ModalDrumConfig
  ) -> Float? {
    ModalBestCheckpointScorer.multiScaleSpectralScore(
      prediction: prediction, target: target, windowSizes: config.spectralWindows,
      hopDivisor: config.spectralHopDivisor, logEpsilon: config.logMagnitudeEpsilon)
  }

  private static func deterministicWhiteNoise(count: Int, rms targetRMS: Float, seed: UInt64)
    -> [Float]
  {
    var state = seed == 0 ? 0x9e37_79b9_7f4a_7c15 : seed
    var result = [Float](repeating: 0, count: count)
    for i in result.indices {
      state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
      let unit = Float((state >> 40) & 0xFFFFFF) / Float(0xFFFFFF)
      result[i] = unit * 2 - 1
    }
    let scale = targetRMS / max(rms(result), 1e-20)
    return result.map { $0 * scale }
  }

  private static func rms(_ samples: [Float]) -> Float {
    guard !samples.isEmpty else { return 0 }
    return sqrt(energy(samples) / Float(samples.count))
  }

  private static func energy(_ samples: [Float]) -> Float {
    samples.reduce(0) { $0 + $1 * $1 }
  }

  private static func writeJSON<T: Encodable>(_ value: T, to url: URL) throws {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    try encoder.encode(value).write(to: url, options: .atomic)
  }
}
