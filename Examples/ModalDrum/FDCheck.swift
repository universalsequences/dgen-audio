import DGenLazy
import Foundation

struct ModalFDGroupResult: Codable {
  var group: String
  var cosine: Float
  var finiteDifference: [Float]
  var autograd: [Float]
}

struct ModalFDCheckReport: Codable {
  var epsilon: Float
  var frames: Int
  var groups: [ModalFDGroupResult]
  var passed: Bool
}

enum ModalFDChecker {
  static func run(
    config input: ModalDrumConfig,
    epsilon: Float = 0.003,
    outputURL: URL? = nil
  ) throws -> ModalFDCheckReport {
    var config = input
    config.modes = 4
    config.frames = min(max(512, input.frames), 2_048)
    config.spectralWindows = [64, 128, 256]
    config.applyRuntime()
    let targetPatch = knownModalPatch(modes: config.modes)
    let target = try renderPatch(targetPatch, config: config)
    let initial = flatInitialPatch(modes: config.modes)

    let baseParams = makeParams(initial, config: config)
    let baseRaw = Dictionary(
      uniqueKeysWithValues: baseParams.rawGroups().map {
        ($0.name, $0.tensor.getData() ?? [])
      })
    var results: [ModalFDGroupResult] = []

    for group in [
      "g", "log_tau", "fir", "noise_gain", "noise_log_tau",
      "noise_tail_fir", "noise_tail_gain", "noise_tail_log_tau",
    ] {
      guard let values = baseRaw[group] else { continue }
      var fd = [Float](repeating: 0, count: values.count)
      for index in values.indices {
        var minus = baseRaw
        var plus = baseRaw
        minus[group]![index] -= epsilon
        plus[group]![index] += epsilon
        let lm = try evaluate(raw: minus, initial: initial, target: target, config: config)
        let lp = try evaluate(raw: plus, initial: initial, target: target, config: config)
        fd[index] = (lp - lm) / (2 * epsilon)
      }
      let ad = try autograd(
        group: group, raw: baseRaw, initial: initial, target: target, config: config)
      let cosine = vectorCosine(fd, ad)
      results.append(
        ModalFDGroupResult(
          group: group, cosine: cosine, finiteDifference: fd, autograd: ad))
    }

    let report = ModalFDCheckReport(
      epsilon: epsilon, frames: config.frames, groups: results,
      passed: results.count == 8 && results.allSatisfy { $0.cosine > 0.999 })
    if let outputURL {
      let encoder = JSONEncoder()
      encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
      try encoder.encode(report).write(to: outputURL, options: .atomic)
    }
    return report
  }

  private static func makeParams(_ patch: ModalPatch, config: ModalDrumConfig)
    -> ModalDrumParameters
  {
    LazyGraphContext.reset()
    return ModalDrumParameters(patch: patch, trainable: true)
  }

  private static func renderPatch(_ patch: ModalPatch, config: ModalDrumConfig) throws -> [Float] {
    LazyGraphContext.reset()
    let frequencies = Tensor(modalFrequencyGrid(count: config.modes))
    let params = ModalDrumParameters(patch: patch, trainable: false)
    return try ModalDrumSynth.render(params: params, frequencies: frequencies)
      .realize(frames: config.frames)
  }

  private static func apply(_ raw: [String: [Float]], to params: ModalDrumParameters) {
    for (name, tensor) in params.rawGroups() {
      if let values = raw[name] { tensor.updateDataLazily(values) }
    }
  }

  private static func evaluate(
    raw: [String: [Float]], initial: ModalPatch, target: [Float], config: ModalDrumConfig
  ) throws -> Float {
    LazyGraphContext.reset()
    let targetTensor = Tensor(target)
    let frequencies = Tensor(modalFrequencyGrid(count: config.modes))
    let params = ModalDrumParameters(patch: initial, trainable: true)
    apply(raw, to: params)
    let prediction = ModalDrumSynth.render(params: params, frequencies: frequencies)
    let loss = ModalDrumLosses.fdLoss(
      prediction: prediction, target: targetTensor.toSignal(maxFrames: config.frames))
    return try loss.realize(frames: config.frames).reduce(0, +)
  }

  private static func autograd(
    group: String, raw: [String: [Float]], initial: ModalPatch, target: [Float],
    config: ModalDrumConfig
  ) throws -> [Float] {
    LazyGraphContext.reset()
    let targetTensor = Tensor(target)
    let frequencies = Tensor(modalFrequencyGrid(count: config.modes))
    let params = ModalDrumParameters(patch: initial, trainable: true)
    apply(raw, to: params)
    let prediction = ModalDrumSynth.render(params: params, frequencies: frequencies)
    let loss = ModalDrumLosses.fdLoss(
      prediction: prediction, target: targetTensor.toSignal(maxFrames: config.frames))
    _ = try loss.backward(frames: config.frames)
    guard let tensor = params.rawGroups().first(where: { $0.name == group })?.tensor,
      let gradient = tensor.grad?.getData()
    else { return [] }
    return gradient
  }

  private static func vectorCosine(_ a: [Float], _ b: [Float]) -> Float {
    guard a.count == b.count, !a.isEmpty else { return 0 }
    let dot = zip(a, b).reduce(Float(0)) { $0 + $1.0 * $1.1 }
    let aa = a.reduce(Float(0)) { $0 + $1 * $1 }
    let bb = b.reduce(Float(0)) { $0 + $1 * $1 }
    return dot / max(Foundation.sqrt(aa * bb), 1e-20)
  }
}
