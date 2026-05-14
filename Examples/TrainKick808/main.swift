import DGen
import DGenLazy
import Foundation

struct Config {
  var targetPath =
    "/Users/alecresende/code/learning/anthropic/eseq/crates/sequencer/samples/manufacturers/EMU/Kicky Fdz.wav"
  var frames = 4096
  var epochs = 40
  var windowSize = 512
  var outputDir = "/tmp/dgen-train-kick808"
  var learningRateScale: Float = 1.0
  var waveformWeight: Float = 35.0
  var transientWeight: Float = 90.0
  var slopeWeight: Float = 0.0
  var envSize = 96
  var freqScalarLRScale: Float = 1.0
  var ampLRScale: Float = 1.0
  var decayLRScale: Float = 1.0
  var envLRScale: Float = 1.0
  var freqCurveLRScale: Float = 1.0
  var residualLRScale: Float = 1.0
  var residualFrames = 1536
  var residualPoints: Int?
  var checkpointIn: String?
  var checkpointOut: String?
}

struct TrainingCheckpoint: Codable {
  var params: [String: Float]
  var tensors: [String: [Float]]
}

func loadCheckpoint(
  path: String,
  params: [Signal],
  names: [String],
  tensors: [DGenLazy.Tensor],
  tensorNames: [String]
) throws {
  let data = try Data(contentsOf: URL(fileURLWithPath: path))
  let checkpoint = try JSONDecoder().decode(TrainingCheckpoint.self, from: data)
  var loadedParams = 0
  var loadedTensors = 0

  for (name, param) in zip(names, params) {
    if let value = checkpoint.params[name] {
      param.updateDataLazily(value)
      loadedParams += 1
    }
  }

  for (name, tensor) in zip(tensorNames, tensors) {
    guard let values = checkpoint.tensors[name] else { continue }
    let currentCount = tensor.getData()?.count ?? 0
    if values.count == currentCount {
      tensor.updateDataLazily(values)
      loadedTensors += 1
    } else {
      print("checkpoint skipped tensor=\(name) count=\(values.count) expected=\(currentCount)")
    }
  }

  print("checkpoint loaded=\(path) params=\(loadedParams) tensors=\(loadedTensors)")
}

func saveCheckpoint(
  path: String,
  params: [Signal],
  names: [String],
  tensors: [DGenLazy.Tensor],
  tensorNames: [String]
) throws {
  var scalarValues: [String: Float] = [:]
  var tensorValues: [String: [Float]] = [:]

  for (name, param) in zip(names, params) {
    scalarValues[name] = param.data ?? 0.0
  }
  for (name, tensor) in zip(tensorNames, tensors) {
    tensorValues[name] = tensor.getData() ?? []
  }

  let checkpoint = TrainingCheckpoint(params: scalarValues, tensors: tensorValues)
  let encoder = JSONEncoder()
  encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
  let data = try encoder.encode(checkpoint)
  let url = URL(fileURLWithPath: path)
  try FileManager.default.createDirectory(
    at: url.deletingLastPathComponent(),
    withIntermediateDirectories: true)
  try data.write(to: url)
  print("wrote=\(url.path)")
}

func parseArgs() -> Config {
  var config = Config()
  var i = 1
  while i < CommandLine.arguments.count {
    let arg = CommandLine.arguments[i]
    func next() -> String? {
      guard i + 1 < CommandLine.arguments.count else { return nil }
      i += 1
      return CommandLine.arguments[i]
    }

    switch arg {
    case "--target":
      if let value = next() { config.targetPath = value }
    case "--frames":
      if let value = next(), let n = Int(value) { config.frames = n }
    case "--epochs":
      if let value = next(), let n = Int(value) { config.epochs = n }
    case "--window":
      if let value = next(), let n = Int(value) { config.windowSize = n }
    case "--out":
      if let value = next() { config.outputDir = value }
    case "--lr-scale":
      if let value = next(), let n = Float(value) { config.learningRateScale = n }
    case "--waveform-weight":
      if let value = next(), let n = Float(value) { config.waveformWeight = n }
    case "--transient-weight":
      if let value = next(), let n = Float(value) { config.transientWeight = n }
    case "--slope-weight":
      if let value = next(), let n = Float(value) { config.slopeWeight = n }
    case "--env-size":
      if let value = next(), let n = Int(value) { config.envSize = n }
    case "--freq-scalar-lr-scale":
      if let value = next(), let n = Float(value) { config.freqScalarLRScale = n }
    case "--amp-lr-scale":
      if let value = next(), let n = Float(value) { config.ampLRScale = n }
    case "--decay-lr-scale":
      if let value = next(), let n = Float(value) { config.decayLRScale = n }
    case "--env-lr-scale":
      if let value = next(), let n = Float(value) { config.envLRScale = n }
    case "--freq-curve-lr-scale":
      if let value = next(), let n = Float(value) { config.freqCurveLRScale = n }
    case "--residual-lr-scale":
      if let value = next(), let n = Float(value) { config.residualLRScale = n }
    case "--residual-frames":
      if let value = next(), let n = Int(value) { config.residualFrames = n }
    case "--residual-points":
      if let value = next(), let n = Int(value) { config.residualPoints = n }
    case "--checkpoint-in":
      if let value = next() { config.checkpointIn = value }
    case "--checkpoint-out":
      if let value = next() { config.checkpointOut = value }
    case "--help", "-h":
      print(
        """
        Usage: swift run TrainKick808 [options]

          --target <wav>     Target kick WAV
          --frames <n>       Training frame count (default: 4096)
          --epochs <n>       Training epochs (default: 40)
          --window <n>       Spectral loss FFT window (default: 512)
          --out <dir>        Output directory
          --lr-scale <x>     Multiplier for optimizer learning rates
          --waveform-weight <x>
          --transient-weight <x>
          --slope-weight <x>
          --env-size <n>     Learnable amplitude envelope points
          --freq-scalar-lr-scale <x>
          --amp-lr-scale <x>
          --decay-lr-scale <x>
          --env-lr-scale <x>
          --freq-curve-lr-scale <x>
          --residual-lr-scale <x>
          --residual-frames <n>
          --residual-points <n>
          --checkpoint-in <json>
          --checkpoint-out <json>
        """)
      exit(0)
    default:
      if !arg.hasPrefix("-") {
        config.targetPath = arg
      } else {
        fputs("Unknown argument: \(arg)\n", stderr)
      }
    }
    i += 1
  }
  return config
}

func transientWindow(_ samples: [Float], frames: Int) -> [Float] {
  let searchCount = min(samples.count, max(frames * 2, frames))
  let threshold = (samples.prefix(searchCount).map { abs($0) }.max() ?? 0) * 0.08
  let onset = samples.prefix(searchCount).firstIndex { abs($0) >= threshold } ?? 0
  let paddedStart = max(0, onset - 16)
  var out = Array(samples.dropFirst(paddedStart).prefix(frames))
  if out.count < frames {
    out += [Float](repeating: 0, count: frames - out.count)
  }
  let peak = out.map { abs($0) }.max() ?? 1
  if peak > 0 {
    out = out.map { $0 / peak * 0.9 }
  }
  return out
}

func findOnset(_ samples: [Float], searchFrames: Int) -> Int {
  let searchCount = min(samples.count, searchFrames)
  let peak = samples.prefix(searchCount).map { abs($0) }.max() ?? 0
  let threshold = peak * 0.08
  return samples.prefix(searchCount).firstIndex { abs($0) >= threshold } ?? 0
}

func estimateFundamental(
  samples: [Float],
  sampleRate: Float,
  startFrame: Int,
  frameCount: Int,
  minHz: Float = 25,
  maxHz: Float = 220
) -> Float? {
  guard startFrame < samples.count else { return nil }
  let endFrame = min(samples.count, startFrame + frameCount)
  let window = Array(samples[startFrame..<endFrame])
  guard window.count >= Int(sampleRate / minHz) else { return nil }

  let mean = window.reduce(0, +) / Float(window.count)
  let centered = window.map { $0 - mean }
  let energy = centered.reduce(Float(0)) { $0 + $1 * $1 }
  guard energy > 1e-8 else { return nil }

  let minLag = max(1, Int(sampleRate / maxHz))
  let maxLag = min(centered.count - 1, Int(sampleRate / minHz))
  guard minLag < maxLag else { return nil }

  var bestLag = minLag
  var bestScore: Float = -.infinity
  for lag in minLag...maxLag {
    var sum: Float = 0
    var normA: Float = 0
    var normB: Float = 0
    let limit = centered.count - lag
    for i in 0..<limit {
      let a = centered[i]
      let b = centered[i + lag]
      sum += a * b
      normA += a * a
      normB += b * b
    }
    let score = sum / max(sqrt(normA * normB), 1e-8)
    if score > bestScore {
      bestScore = score
      bestLag = lag
    }
  }

  guard bestScore > 0.15 else { return nil }
  return sampleRate / Float(bestLag)
}

func pitchTrack(samples: [Float], sampleRate: Float, onset: Int) -> [(ms: Float, hz: Float)] {
  let offsetsMs: [Float] = [6, 14, 24, 38, 56, 76]
  return offsetsMs.compactMap { ms in
    let start = onset + Int(sampleRate * ms / 1000.0)
    guard
      let hz = estimateFundamental(
        samples: samples,
        sampleRate: sampleRate,
        startFrame: start,
        frameCount: Int(sampleRate * 0.045),
        minHz: 25,
        maxHz: 220
      )
    else { return nil }
    return (ms: ms, hz: hz)
  }
}

func peakEnvelope(samples: [Float], size: Int) -> [Float] {
  let samplesPerPoint = max(1, samples.count / size)
  return (0..<size).map { i in
    let start = i * samplesPerPoint
    let end = min(samples.count, start + samplesPerPoint)
    guard start < end else { return Float(0.0) }
    return samples[start..<end].map { abs($0) }.max() ?? 0.0
  }
}

func zeroCrossingFrequencyCurve(samples: [Float], sampleRate: Float, size: Int, fallbackEndHz: Float) -> [Float] {
  var crossings: [Int] = []
  var previous = samples.first ?? 0
  for i in 1..<samples.count {
    let value = samples[i]
    if (previous < 0 && value >= 0) || (previous > 0 && value <= 0) {
      crossings.append(i)
    }
    if value != 0 {
      previous = value
    }
  }

  var points: [(frame: Float, hz: Float)] = []
  for i in 0..<(crossings.count - 2) {
    let a = crossings[i]
    let b = crossings[i + 2]
    guard b > a else { continue }
    let center = Float(a + b) * 0.5
    let hz = sampleRate / Float(b - a)
    if hz >= 25 && hz <= 900 {
      points.append((center, hz))
    }
  }

  guard !points.isEmpty else {
    return (0..<size).map { i -> Float in
      let x = Float(i) / Float(max(1, size - 1))
      let frames = x * Float(samples.count)
      return fallbackEndHz + (390.0 - fallbackEndHz) * exp(-frames / 650.0)
    }
  }

  var curve: [Float] = []
  for i in 0..<size {
    let frame = Float(i) / Float(max(1, size - 1)) * Float(samples.count - 1)
    if frame <= points[0].frame {
      curve.append(points[0].hz)
      continue
    }
    if frame >= points.last!.frame {
      let tailFrame = frame - points.last!.frame
      curve.append(fallbackEndHz + (points.last!.hz - fallbackEndHz) * exp(-tailFrame / 1800.0))
      continue
    }
    var j = 0
    while j + 1 < points.count && points[j + 1].frame < frame {
      j += 1
    }
    let a = points[j]
    let b = points[j + 1]
    let mix = (frame - a.frame) / max(1.0, b.frame - a.frame)
    curve.append(a.hz * (1 - mix) + b.hz * mix)
  }
  return curve.map { max(25.0, min(900.0, $0)) }
}

@discardableResult
func timed<T>(_ block: () throws -> T) rethrows -> (T, Double) {
  let start = DispatchTime.now().uptimeNanoseconds
  let value = try block()
  let end = DispatchTime.now().uptimeNanoseconds
  return (value, Double(end - start) / 1_000_000.0)
}

func run() throws {
  var config = parseArgs()
  if config.windowSize > config.frames {
    config.windowSize = config.frames
  }

  let targetURL = URL(fileURLWithPath: config.targetPath)
  let (rawSamples, sampleRate) = try AudioFile.load(url: targetURL, mono: true)
  let onset = findOnset(rawSamples, searchFrames: max(config.frames * 2, config.frames))
  let targetSamples = transientWindow(rawSamples, frames: config.frames)
  let targetEnv = peakEnvelope(samples: targetSamples, size: config.envSize)
  let analyzedPitches = pitchTrack(samples: rawSamples, sampleRate: sampleRate, onset: onset)
  let endFreqGuess = estimateFundamental(
    samples: rawSamples,
    sampleRate: sampleRate,
    startFrame: onset + Int(sampleRate * 0.045),
    frameCount: Int(sampleRate * 0.080),
    minHz: 25,
    maxHz: 140
  ) ?? analyzedPitches.suffix(2).map(\.hz).min() ?? 52.0
  let earlyMax = analyzedPitches.filter { $0.ms <= 38 }.map(\.hz).max()
  let startFreqGuess = min(max(earlyMax ?? endFreqGuess * 3.0, endFreqGuess * 2.6), 180.0)

  try FileManager.default.createDirectory(
    atPath: config.outputDir, withIntermediateDirectories: true)

  DGenConfig.sampleRate = sampleRate
  let previousMaxFrames = DGenConfig.maxFrameCount
  DGenConfig.maxFrameCount = config.frames
  defer { DGenConfig.maxFrameCount = previousMaxFrames }
  LazyGraphContext.reset()

  let firstTarget = targetSamples.first ?? 0.0
  let initialPhase = asin(max(-0.95, min(0.95, firstTarget / 0.9)))
  let targetTensor = Tensor(targetSamples)
  let bodyEnvTensor = Tensor.param([config.envSize], data: targetEnv.map { max($0, 0.03) })
  bodyEnvTensor.minBound = 0.0
  bodyEnvTensor.maxBound = 1.5
  let subEnvTensor = Tensor.param([config.envSize], data: targetEnv.map { max($0 * 0.25, 0.01) })
  subEnvTensor.minBound = 0.0
  subEnvTensor.maxBound = 1.2
  let harmEnvTensor = Tensor.param([config.envSize], data: targetEnv.map { max($0 * 0.18, 0.01) })
  harmEnvTensor.minBound = 0.0
  harmEnvTensor.maxBound = 1.2
  let clickFrontPoints = max(4, config.envSize / 8)
  let clickEnvTensor = Tensor.param([config.envSize], data: targetEnv.enumerated().map { i, v in
    i < clickFrontPoints ? max(v * 0.5, 0.02) : 0.0
  })
  clickEnvTensor.minBound = 0.0
  clickEnvTensor.maxBound = 1.5
  let freqCurveInit = zeroCrossingFrequencyCurve(
    samples: targetSamples,
    sampleRate: sampleRate,
    size: config.envSize,
    fallbackEndHz: endFreqGuess)
  let freqCurveTensor = Tensor.param([config.envSize], data: freqCurveInit)
  freqCurveTensor.minBound = 25.0
  freqCurveTensor.maxBound = 900.0
  let residualFrames = max(1, min(config.residualFrames, config.frames))
  let residualPointCount = max(2, config.residualPoints ?? residualFrames)
  let residualTensor = Tensor.param(
    [residualPointCount], data: [Float](repeating: 0.0, count: residualPointCount))
  residualTensor.minBound = -1.5
  residualTensor.maxBound = 1.5

  let startFreq = Signal.param(startFreqGuess, min: 25.0, max: 220.0)
  let endFreq = Signal.param(endFreqGuess, min: 25.0, max: 120.0)
  let freqLogDecay = Signal.param(log(0.9965), min: -0.03, max: -1e-6)
  let phase = Signal.param(initialPhase, min: -Float.pi, max: Float.pi)

  let bodyAmp = Signal.param(0.85, min: 0.0, max: 2.0)
  let bodyLogDecay = Signal.param(log(0.99985), min: -0.01, max: -1e-6)
  let subAmp = Signal.param(0.2, min: 0.0, max: 1.5)
  let subLogDecay = Signal.param(log(0.9999), min: -0.01, max: -1e-6)

  let harmAmp = Signal.param(0.12, min: 0.0, max: 1.0)
  let harmLogDecay = Signal.param(log(0.996), min: -0.05, max: -1e-6)
  let clickAmp = Signal.param(0.25, min: 0.0, max: 1.8)
  let clickFreq = Signal.param(380.0, min: 80.0, max: 6000.0)
  let clickLogDecay = Signal.param(log(0.997), min: -0.2, max: -1e-6)
  let impulseAmp = Signal.param(0.12, min: -1.5, max: 1.5)
  let ringAmp = Signal.param(0.18, min: -1.5, max: 1.5)
  let ringFreq = Signal.param(max(startFreqGuess * 2.0, 110.0), min: 40.0, max: 1800.0)
  let drive = Signal.param(1.2, min: 0.3, max: 8.0)

  let params: [Signal] = [
    startFreq, endFreq, freqLogDecay, phase,
    bodyAmp, bodyLogDecay,
    subAmp, subLogDecay,
    harmAmp, harmLogDecay,
    clickAmp, clickFreq, clickLogDecay, impulseAmp, ringAmp, ringFreq,
    drive,
  ]
  let names = [
    "startFreq", "endFreq", "freqLogDecay", "phase",
    "bodyAmp", "bodyLogDecay",
    "subAmp", "subLogDecay",
    "harmAmp", "harmLogDecay",
    "clickAmp", "clickFreq", "clickLogDecay", "impulseAmp", "ringAmp", "ringFreq",
    "drive",
  ]
  let tensorParams = [
    bodyEnvTensor, subEnvTensor, harmEnvTensor, clickEnvTensor, freqCurveTensor, residualTensor,
  ]
  let tensorNames = ["bodyEnv", "subEnv", "harmEnv", "clickEnv", "freqCurve", "residual"]

  if let checkpointIn = config.checkpointIn {
    try loadCheckpoint(
      path: checkpointIn,
      params: params,
      names: names,
      tensors: tensorParams,
      tensorNames: tensorNames)
  }

  let freqOpt = Adam(
    params: [startFreq, endFreq, clickFreq, ringFreq],
    lr: 0.75 * config.learningRateScale * config.freqScalarLRScale)
  let ampOpt = Adam(
    params: [phase, bodyAmp, subAmp, harmAmp, clickAmp, impulseAmp, ringAmp, drive],
    lr: 0.015 * config.learningRateScale * config.ampLRScale)
  let decayOpt = Adam(
    params: [freqLogDecay, bodyLogDecay, subLogDecay, harmLogDecay, clickLogDecay],
    lr: 0.00002 * config.learningRateScale * config.decayLRScale)
  let envOpt = Adam(
    params: [bodyEnvTensor, subEnvTensor, harmEnvTensor, clickEnvTensor],
    lr: 0.002 * config.learningRateScale * config.envLRScale)
  let freqCurveOpt = Adam(
    params: [freqCurveTensor],
    lr: 0.7 * config.learningRateScale * config.freqCurveLRScale)
  let residualOpt = Adam(
    params: [residualTensor],
    lr: 0.08 * config.learningRateScale * config.residualLRScale)

  func buildSynth() -> Signal {
    let t = Signal.accum(
      Signal.constant(1.0),
      reset: 0.0,
      min: 0.0,
      max: Float(config.frames + 1)
    )
    let playhead = Signal.accum(
      Signal.constant(Float(config.envSize - 1) / Float(config.frames)),
      reset: 0.0,
      min: 0.0,
      max: Float(config.envSize - 1)
    )

    let freq = freqCurveTensor.peek(playhead)

    let bodyEnv = exp(bodyLogDecay * t)
    let bodyPhase = Signal.phasor(freq) * Float.pi * 2.0 + phase
    let body = sin(bodyPhase) * bodyEnv * bodyEnvTensor.peek(playhead) * bodyAmp

    let subEnv = exp(subLogDecay * t)
    let sub = sin(Signal.phasor(freq * 0.5) * Float.pi * 2.0 + phase * 0.5)
      * subEnv * subEnvTensor.peek(playhead) * subAmp

    let harmEnv = exp(harmLogDecay * t)
    let harmonic = sin(Signal.phasor(freq * 2.0) * Float.pi * 2.0 + phase * 2.0)
      * harmEnv * harmEnvTensor.peek(playhead) * harmAmp

    let clickEnv = exp(clickLogDecay * t)
    let click = sin(Signal.phasor(clickFreq) * Float.pi * 2.0)
      * clickEnv * clickEnvTensor.peek(playhead) * clickAmp
    let impulse = Signal.click()
    let ring = impulse.biquad(cutoff: ringFreq, resonance: Signal.constant(1.8), gain: Signal.constant(1.0), mode: Signal.constant(2.0)) * ringAmp
    let residualIndex = (t * (Float(residualPointCount - 1) / Float(max(1, residualFrames - 1))))
      .clip(0.0, Double(residualPointCount - 1))
    let residualGate = t < Double(residualFrames - 1)
    let residual = gswitch(residualGate, residualTensor.peek(residualIndex), 0.0)

    return tanh((body + sub + harmonic + click + impulse * impulseAmp + ring + residual) * drive)
  }

  func buildTarget() -> Signal {
    targetTensor.toSignal(maxFrames: config.frames)
  }

  let targetOut = URL(fileURLWithPath: config.outputDir).appendingPathComponent("target.wav")
  try AudioFile.save(url: targetOut, samples: targetSamples, sampleRate: sampleRate)

  print("target=\(targetURL.path)")
  print(
    "frames=\(config.frames) sampleRate=\(sampleRate) durationMs=\(String(format: "%.1f", Float(config.frames) / sampleRate * 1000)) window=\(config.windowSize) epochs=\(config.epochs) waveformWeight=\(config.waveformWeight) transientWeight=\(config.transientWeight) slopeWeight=\(config.slopeWeight) envSize=\(config.envSize) residualFrames=\(residualFrames) residualPoints=\(residualPointCount)"
  )
  print(
    "lrScales freqScalar=\(config.freqScalarLRScale) amp=\(config.ampLRScale) decay=\(config.decayLRScale) env=\(config.envLRScale) freqCurve=\(config.freqCurveLRScale) residual=\(config.residualLRScale)"
  )
  print(
    "analysis onsetFrame=\(onset) startFreqGuess=\(String(format: "%.2f", startFreqGuess))Hz endFreqGuess=\(String(format: "%.2f", endFreqGuess))Hz"
  )
  if !analyzedPitches.isEmpty {
    let formatted = analyzedPitches
      .map { "\(String(format: "%.0f", $0.ms))ms:\(String(format: "%.1f", $0.hz))Hz" }
      .joined(separator: " ")
    print("pitchTrack \(formatted)")
  }

  let (_, warmupMs) = try timed {
    let synth = buildSynth()
    let target = buildTarget()
    let loss = combinedLoss(synth: synth, target: target)
    _ = try loss.backward(frames: config.frames)
    freqOpt.zeroGrad()
    ampOpt.zeroGrad()
    decayOpt.zeroGrad()
    envOpt.zeroGrad()
    freqCurveOpt.zeroGrad()
    residualOpt.zeroGrad()
  }
  print("warmupMs=\(String(format: "%.2f", warmupMs))")

  func combinedLoss(synth: Signal, target: Signal) -> Signal {
    let spectral = spectralLossFFT(
      synth, target,
      windowSize: config.windowSize,
      useHannWindow: true,
      hop: max(1, config.windowSize / 4),
      normalize: true)
    let perFrameMSE = mse(synth, target)
    let frameScale = 1.0 / Float(config.frames)
    let t = Signal.accum(
      Signal.constant(1.0),
      reset: 0.0,
      min: 0.0,
      max: Float(config.frames + 1)
    )
    let transientWeight = exp(Signal.constant(log(0.996)) * t)
    let synthHistory = Signal.history()
    let targetHistory = Signal.history()
    let synthDelta = synthHistory.write(synth) - synthHistory.read
    let targetDelta = targetHistory.write(target) - targetHistory.read
    let slopeMSE = mse(synthDelta, targetDelta)
    let slopeWindow = exp(Signal.constant(log(0.992)) * t)
    return spectral
      + perFrameMSE * (config.waveformWeight * frameScale)
      + perFrameMSE * transientWeight * (config.transientWeight * frameScale)
      + slopeMSE * slopeWindow * (config.slopeWeight * frameScale)
  }

  var firstLoss: Float = 0
  var lastLoss: Float = 0
  var bestLoss: Float = .infinity
  var bestEpoch = 0
  var bestValues: [Float] = params.map { $0.data ?? 0.0 }
  var bestTensorValues: [[Float]] = tensorParams.map { $0.getData() ?? [] }
  var epochTimes: [Double] = []

  for epoch in 0..<config.epochs {
    let ((lossValues), epochMs) = try timed {
      let loss = combinedLoss(synth: buildSynth(), target: buildTarget())
      return try loss.backward(frames: config.frames)
    }
    epochTimes.append(epochMs)
    let epochLoss = lossValues.reduce(0, +)
    if epoch == 0 { firstLoss = epochLoss }
    lastLoss = epochLoss
    if epochLoss < bestLoss {
      bestLoss = epochLoss
      bestEpoch = epoch
      bestValues = params.map { $0.data ?? 0.0 }
      bestTensorValues = tensorParams.map { $0.getData() ?? [] }
    }

    if epoch % 5 == 0 || epoch == config.epochs - 1 {
      print(
        "epoch=\(epoch) loss=\(String(format: "%.6f", epochLoss)) stepMs=\(String(format: "%.2f", epochMs))"
      )
      for (name, param) in zip(names, params) {
        let value = param.data ?? .nan
        let grad = param.grad?.data ?? .nan
        print(
          "  \(name)=\(String(format: "%12.6f", value)) grad=\(String(format: "%12.4e", grad))"
        )
      }
      for (name, tensor) in zip(tensorNames, tensorParams) {
        let data = tensor.getData() ?? []
        let grad = tensor.grad?.getData() ?? []
        let gMax = grad.map { abs($0) }.max() ?? 0
        print(
          "  \(name): max=\(String(format: "%8.4f", data.max() ?? 0)) gradMax=\(String(format: "%10.4e", gMax))"
        )
      }
    }

    if epochLoss.isNaN || epochLoss.isInfinite {
      print("stopping: non-finite loss")
      break
    }

    freqOpt.step()
    ampOpt.step()
    decayOpt.step()
    envOpt.step()
    freqCurveOpt.step()
    residualOpt.step()
    freqOpt.zeroGrad()
    ampOpt.zeroGrad()
    decayOpt.zeroGrad()
    envOpt.zeroGrad()
    freqCurveOpt.zeroGrad()
    residualOpt.zeroGrad()
  }

  for (param, value) in zip(params, bestValues) {
    param.updateDataLazily(value)
  }
  for (tensor, values) in zip(tensorParams, bestTensorValues) {
    tensor.updateDataLazily(values)
  }
  let checkpointOut = config.checkpointOut
    ?? URL(fileURLWithPath: config.outputDir).appendingPathComponent("checkpoint.json").path
  try saveCheckpoint(
    path: checkpointOut,
    params: params,
    names: names,
    tensors: tensorParams,
    tensorNames: tensorNames)

  let learned = try buildSynth().realize(frames: config.frames)
  let learnedOut = URL(fileURLWithPath: config.outputDir).appendingPathComponent("learned.wav")
  try AudioFile.save(url: learnedOut, samples: learned, sampleRate: sampleRate)

  let avgMs = epochTimes.reduce(0, +) / Double(max(1, epochTimes.count))
  print("firstLoss=\(firstLoss) finalLoss=\(lastLoss)")
  print("bestLoss=\(bestLoss) bestEpoch=\(bestEpoch)")
  print("bestReduction=\(String(format: "%.2fx", firstLoss / max(bestLoss, 1e-12)))")
  print("avgStepMs=\(String(format: "%.2f", avgMs))")
  print("wrote=\(targetOut.path)")
  print("wrote=\(learnedOut.path)")
}

do {
  try run()
} catch {
  fputs("TrainKick808 error: \(error)\n", stderr)
  exit(1)
}
