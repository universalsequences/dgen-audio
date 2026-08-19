import DGen
import DGenLazy
import Foundation

struct ModalRecoverySummary: Codable {
  var initialLoss: Float
  var bestLoss: Float
  var lossReduction: Float
  var gainCosine: Float
  var maxAudibleTauRelativeError: Float
  var deterministicTarget: Bool
  var bestStep: Int
  var passedRecoveryGate: Bool
}

enum ModalDrumTrainer {
  static func train(config: ModalDrumConfig, runDirectory: URL, kernelDumpPath: String? = nil)
    throws
    -> ModalRecoverySummary
  {
    try createRunDirectories(runDirectory)
    config.applyRuntime(kernelOutputPath: kernelDumpPath)

    let truth = knownModalPatch(modes: config.modes)
    let target = try render(patch: truth, config: config)
    let targetAgain = try render(patch: truth, config: config)
    let deterministic = target == targetAgain
    // render() configures standalone rendering; restore this run's requested
    // kernel dump before compiling the trainable graph.
    config.applyRuntime(kernelOutputPath: kernelDumpPath)
    try AudioFile.save(
      url: runDirectory.appendingPathComponent("target.wav"), samples: target,
      sampleRate: config.sampleRate)

    LazyGraphContext.reset()
    let targetTensor = Tensor(target)
    let frequencies = Tensor(modalFrequencyGrid(count: config.modes))
    let highModeMask = Tensor(modalFrequencyGrid(count: config.modes).map { $0 > 6_000 ? 1 : 0 })
    let params = ModalDrumParameters(patch: flatInitialPatch(modes: config.modes), trainable: true)
    let optimizer = Adam(params: params.all, lr: config.learningRate)

    try writeJSON(config, to: runDirectory.appendingPathComponent("resolved_config.json"))
    try writeJSON(truth, to: runDirectory.appendingPathComponent("true_params.json"))

    var csv = "step,loss,selection_score\n"
    var initialLoss: Float = .nan
    var bestLoss = Float.greatestFiniteMagnitude
    var bestSelection = Float.greatestFiniteMagnitude
    var bestStep = 0
    var bestPatch = params.naturalPatch()

    for step in 0..<max(1, config.steps) {
      if step > 0 { LazyGraphContext.current.clearComputationGraph() }
      let prediction = ModalDrumSynth.render(params: params, frequencies: frequencies)
      var loss = ModalDrumLosses.trainingLoss(
        prediction: prediction, target: targetTensor.toSignal(maxFrames: config.frames),
        config: config)
      // Canonicalize the shell/wire split: lightly budget modal gains above 6 kHz.
      let highModePenalty = (params.gains * highModeMask).sum() * (1 / Float(max(1, config.modes)))
      let highModeLoss = (Tensor([0.0]) + highModePenalty * 0.001).peek(Signal.constant(0))
      loss = loss + highModeLoss

      let values = try loss.backward(frames: config.frames)
      let stepLoss = values.reduce(0, +) / Float(max(1, values.count))
      // Capture metrics before zeroGrad(), per the lazy gradient lifecycle.
      if step == 0 { initialLoss = stepLoss }
      if stepLoss.isFinite, stepLoss < bestLoss {
        bestLoss = stepLoss
      }
      let progress = Float(step) / Float(max(1, config.steps - 1))
      optimizer.lr =
        config.learningRate * max(0.01, 0.5 * (1 + Foundation.cos(Float.pi * progress)))
      optimizer.step()
      optimizer.zeroGrad()

      var selectionText = ""
      let artifactStep =
        step == 0 || step == config.steps - 1
        || (step + 1) % max(1, config.renderEvery) == 0
      if artifactStep {
        LazyGraphContext.current.clearComputationGraph()
        let preview = try ModalDrumSynth.render(params: params, frequencies: frequencies)
          .realize(frames: config.frames)
        let score = ModalBestCheckpointScorer.multiScaleSpectralScore(
          prediction: preview, target: target, windowSizes: config.spectralWindows,
          hopDivisor: config.spectralHopDivisor, logEpsilon: config.logMagnitudeEpsilon)
        if let score {
          selectionText = "\(score)"
          if score < bestSelection {
            bestSelection = score
            bestStep = step
            bestPatch = params.naturalPatch()
            try ModalCheckpointStore.write(
              ModalDrumCheckpoint(
                step: step, loss: score,
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
            step: step, loss: stepLoss,
            createdAtUTC: ISO8601DateFormatter().string(from: Date()),
            patch: params.naturalPatch()),
          to: runDirectory.appendingPathComponent(
            String(format: "checkpoints/step_%06d.json", step)))
      }
      csv += "\(step),\(stepLoss),\(selectionText)\n"
      if step == 0 || step == config.steps - 1 || step % 10 == 0 {
        print("step=\(step) loss=\(String(format: "%.6g", stepLoss))")
      }
    }

    try csv.write(
      to: runDirectory.appendingPathComponent("loss.csv"), atomically: true, encoding: .utf8)
    try writeJSON(bestPatch, to: runDirectory.appendingPathComponent("recovered_params.json"))
    let gainCosine = cosine(bestPatch.gains, truth.gains)
    let tauErrors = zip(zip(bestPatch.decaySeconds, truth.decaySeconds), truth.gains)
      .filter { $0.1 > 0.01 }
      .map { abs($0.0.0 - $0.0.1) / $0.0.1 }
    let maxTauError = tauErrors.max() ?? 0
    let reduction = initialLoss / max(bestLoss, 1e-20)
    let summary = ModalRecoverySummary(
      initialLoss: initialLoss, bestLoss: bestLoss, lossReduction: reduction,
      gainCosine: gainCosine, maxAudibleTauRelativeError: maxTauError,
      deterministicTarget: deterministic, bestStep: bestStep,
      passedRecoveryGate: reduction >= 100 && gainCosine >= 0.99 && maxTauError <= 0.10
        && deterministic)
    try writeJSON(summary, to: runDirectory.appendingPathComponent("summary.json"))
    return summary
  }

  static func render(patch: ModalPatch, config: ModalDrumConfig) throws -> [Float] {
    config.applyRuntime()
    LazyGraphContext.reset()
    let frequencies = Tensor(modalFrequencyGrid(count: patch.gains.count))
    let params = ModalDrumParameters(patch: patch, trainable: false)
    return try ModalDrumSynth.render(params: params, frequencies: frequencies)
      .realize(frames: config.frames)
  }

  private static func createRunDirectories(_ root: URL) throws {
    for path in [
      root, root.appendingPathComponent("checkpoints"), root.appendingPathComponent("previews"),
    ] {
      try FileManager.default.createDirectory(at: path, withIntermediateDirectories: true)
    }
  }

  private static func writeJSON<T: Encodable>(_ value: T, to url: URL) throws {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    try encoder.encode(value).write(to: url, options: .atomic)
  }

  private static func cosine(_ a: [Float], _ b: [Float]) -> Float {
    guard a.count == b.count, !a.isEmpty else { return 0 }
    let dot = zip(a, b).reduce(Float(0)) { $0 + $1.0 * $1.1 }
    let aa = a.reduce(Float(0)) { $0 + $1 * $1 }
    let bb = b.reduce(Float(0)) { $0 + $1 * $1 }
    return dot / max(Foundation.sqrt(aa * bb), 1e-20)
  }
}
