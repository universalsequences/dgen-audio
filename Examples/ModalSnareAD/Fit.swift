import DGen
import DGenLazy
import Foundation

struct FitConfig {
  var ids: [String] = VillainData.snareIDs()
  var frames = 48_000
  var steps = 300
  var lr: Float = 0.05
  var windows = [128, 256, 512, 1024, 2048]
  var metricWindows = [64, 128, 256, 512, 1024, 2048]
  var hopDivisor = 4
  var linearWeight: Float = 1
  var logWeight: Float = 1
  var lossMode: SpectralLossMode = .l1
  var unfreezeHeadAt: Int? = nil  // step at which head residuals join the optimizer
  var headL2: Float = 1e-3
  var includeCoupling = true
  var includeNoise = true
  var evalEvery = 50
  var targetKind = "out"
  var frozenGroups: Set<String> = []
  var freezeOutputUntil: Int = 0  // output-stage (gain/drive) params join the optimizer at this step (Int.max: never)  // "out" | "out_clean" | "head"
  var out: URL = Paths.e1.appendingPathComponent("fits")
  var initFrom: URL? = nil
}

enum Scores {
  static func score(_ p: [Float], _ t: [Float], windows: [Int]) -> Float {
    ModalBestCheckpointScorer.multiScaleSpectralScore(
      prediction: p, target: t, windowSizes: windows, hopDivisor: 4, logEpsilon: 1e-3) ?? .nan
  }
}

func loadTarget(_ id: String, kind: String, frames: Int) throws -> [Float] {
  let url: URL
  switch kind {
  case "out": url = Paths.out(id)
  default: url = Paths.stem(id, kind)
  }
  var s = try WAV.read(url)
  if s.count < frames { s += [Float](repeating: 0, count: frames - s.count) }
  return Array(s.prefix(frames))
}

enum SnareFitter {
  /// CPU calibration of the selection metric on the target set: self, a
  /// wrong snare (every other target; min / median), silence, RMS-matched
  /// white noise.
  static func calibrate(targets: [[Float]], windows: [Int]) -> [[String: Any]] {
    var rows: [[String: Any]] = []
    var rng = SystemRandomNumberGenerator()
    for (i, t) in targets.enumerated() {
      let selfS = Scores.score(t, t, windows: windows)
      let others = targets.indices.filter { $0 != i }.map { Scores.score(targets[$0], t, windows: windows) }
      let silence = Scores.score([Float](repeating: 0, count: t.count), t, windows: windows)
      let rms = (t.reduce(0) { $0 + $1 * $1 } / Float(t.count)).squareRoot()
      let noise = (0..<t.count).map { _ in Float.random(in: -1...1, using: &rng) * rms * 1.7320508 }
      let noiseS = Scores.score(noise, t, windows: windows)
      let sorted = others.sorted()
      rows.append([
        "self": selfS, "wrong_min": sorted.first ?? .nan, "wrong_median": sorted[sorted.count / 2],
        "wrong_nearest_index": others.firstIndex(of: sorted.first ?? 0).map { $0 >= i ? $0 + 1 : $0 } ?? -1,
        "silence": silence, "noise_rms": noiseS,
      ])
    }
    return rows
  }

  static func run(config cfg: FitConfig) throws -> [String: Any] {
    configureRuntime(frames: cfg.frames)
    LazyGraphContext.preserveCompilationCaches = ProcessInfo.processInfo.environment["MSAD_NO_PRESERVE"] == nil
    try FileManager.default.createDirectory(at: cfg.out, withIntermediateDirectories: true)
    let renders = try VillainData.loadRenders(ids: cfg.ids)
    let B = renders.count
    let targets = try cfg.ids.map { try loadTarget($0, kind: cfg.targetKind, frames: cfg.frames) }
    let bank = try NoiseBank.load()
    let constants = SnareConstants(renders: renders)

    print("calibrating metric on \(B) targets ...")
    let calib = calibrate(targets: targets, windows: cfg.metricWindows)

    var initial: [String: [Float]] = [:]
    if let u = cfg.initFrom,
      let j = try? JSONSerialization.jsonObject(with: Data(contentsOf: u)) as? [String: Any],
      let raw = j["raw"] as? [String: [NSNumber]]
    {
      for (k, v) in raw { initial[k] = v.map { $0.floatValue } }
    }

    LazyGraphContext.reset()
    let targetTable = Tensor(interleave(targets, frames: cfg.frames)).reshape([cfg.frames, B])
    let noiseTable = Tensor(bank.timeMajor(frames: cfg.frames)).reshape([cfg.frames, bank.bands])
    let params = SnareParams(constants: constants, bands: bank.bands, initial: initial)
    var trainable = (cfg.includeNoise ? params.noiseGroups : []) + (cfg.freezeOutputUntil == 0 ? params.outputGroups : [])
    if !cfg.includeCoupling { trainable.removeAll { $0.0 == "noise_couple" } }
    trainable.removeAll { cfg.frozenGroups.contains($0.0) }
    var optimizer = Adam(params: trainable.map { $0.1 }, lr: cfg.lr)
    var headOn = false

    func options() -> RenderOptions {
      RenderOptions(
        includeHead: true, includeNoise: cfg.includeNoise, includeCoupling: cfg.includeCoupling,
        trainHead: headOn)
    }

    func evaluate() throws -> ([[Float]], [Float]) {
      LazyGraphContext.current.clearComputationGraph()
      let r = SnareRenderer.render(
        params: params, constants: constants, noiseTable: noiseTable, frames: cfg.frames,
        sampleRate: VillainData.sampleRate, options: options())
      let flat = try r.output.realize(frames: cfg.frames)
      let per = deinterleave(flat, batch: B)
      let scores = zip(per, targets).map { Scores.score($0, $1, windows: cfg.metricWindows) }
      return (per, scores)
    }

    var csv = "step,loss,mean_score,median_score,sec\n"
    let skipInit = ProcessInfo.processInfo.environment["MSAD_SKIP_INIT_EVAL"] != nil
    let (_, initScores) = skipInit ? ([], [Float](repeating: 9, count: B)) : try evaluate()
    var best = initScores
    var bestMean = initScores.reduce(0, +) / Float(B)
    var bestSnapshot = params.snapshot()
    var bestRender: [[Float]] = []
    print(String(format: "init mean score %.4f", initScores.reduce(0, +) / Float(B)))
    let t0 = Date()
    var stepTimes: [Double] = []
    for step in 0..<cfg.steps {
      if cfg.freezeOutputUntil > 0, step == cfg.freezeOutputUntil {
        trainable += params.outputGroups
        optimizer = Adam(params: trainable.map { $0.1 }, lr: optimizer.lr)
        print("step \(step): unfreezing output stage")
      }
      if let u = cfg.unfreezeHeadAt, step == u, !headOn {
        headOn = true
        trainable += params.headGroups
        let lr = optimizer.lr
        optimizer = Adam(params: trainable.map { $0.1 }, lr: lr)
        print("step \(step): unfreezing head residuals")
      }
      let ts = Date()
      if step > 0 || !skipInit { LazyGraphContext.current.clearComputationGraph() }
      let r = SnareRenderer.render(
        params: params, constants: constants, noiseTable: noiseTable, frames: cfg.frames,
        sampleRate: VillainData.sampleRate, options: options())
      let playhead = floor(Clock.seconds(frames: cfg.frames, sampleRate: VillainData.sampleRate) * VillainData.sampleRate + 0.5)
      let target = targetTable.sampleRow(playhead)
      var loss = Signal.constant(0)
      let wScale = 1 / Float(cfg.windows.count)
      for w in cfg.windows where w <= cfg.frames {
        let hop = max(1, w / cfg.hopDivisor)
        if cfg.linearWeight > 0 {
          loss = loss + spectralLossFFT(r.output, target, windowSize: w, lossMode: cfg.lossMode, hop: hop, normalize: true) * (cfg.linearWeight * wScale)
        }
        if cfg.logWeight > 0 {
          loss = loss + spectralLossFFT(r.output, target, windowSize: w, useLogMagnitude: true, lossMode: cfg.lossMode, hop: hop, normalize: true) * (cfg.logWeight * wScale)
        }
      }
      if headOn && cfg.headL2 > 0 {
        let reg = ((p_sq(params.rawHF) + p_sq(params.rawHA) + p_sq(params.rawHD)) * (cfg.headL2 / Float(B * params.K)))
        loss = loss + (Tensor([0.0]) + reg).peek(Signal.constant(0)) * (1 / Float(cfg.frames))
      }
      if ProcessInfo.processInfo.environment["MSAD_DEBUG_LOSS"] != nil {
        // Forward-only value of the same loss, then rebuild for backward.
        let fv = try loss.realize(frames: cfg.frames).reduce(0, +)
        print("debug forward loss", fv)
        LazyGraphContext.current.clearComputationGraph()
        let r2 = SnareRenderer.render(
          params: params, constants: constants, noiseTable: noiseTable, frames: cfg.frames,
          sampleRate: VillainData.sampleRate, options: options())
        let ph2 = floor(Clock.seconds(frames: cfg.frames, sampleRate: VillainData.sampleRate) * VillainData.sampleRate + 0.5)
        let tg2 = targetTable.sampleRow(ph2)
        loss = Signal.constant(0)
        for w in cfg.windows where w <= cfg.frames {
          let hop = max(1, w / cfg.hopDivisor)
          loss = loss + spectralLossFFT(r2.output, tg2, windowSize: w, lossMode: cfg.lossMode, hop: hop, normalize: true) * (cfg.linearWeight * wScale)
          loss = loss + spectralLossFFT(r2.output, tg2, windowSize: w, useLogMagnitude: true, lossMode: cfg.lossMode, hop: hop, normalize: true) * (cfg.logWeight * wScale)
        }
      }
      let values = try loss.backward(frames: cfg.frames)
      let stepLoss = values.reduce(0, +)
      if let d = ProcessInfo.processInfo.environment["MSAD_DUMP_AT"].flatMap(Int.init), step == d {
        try writeJSON(["raw": params.snapshot()], to: cfg.out.appendingPathComponent("params_step\(d).json"))
      }
      if let ls = ProcessInfo.processInfo.environment["MSAD_LINESEARCH"].flatMap(Int.init), step == ls {
        let snap = params.snapshot()
        var grads: [String: [Float]] = [:]
        for (n, t) in trainable { grads[n] = t.grad?.getData() ?? [] }
        let gnorm = grads.values.flatMap { $0 }.reduce(Float(0)) { $0 + $1 * $1 }.squareRoot()
        for (n, g) in grads.sorted(by: { $0.key < $1.key }) {
          print("grad \(n) norm", g.reduce(Float(0)) { $0 + $1 * $1 }.squareRoot())
        }
        var trials: [(String, Float, Set<String>)] = [0, 1e-4, 1e-3, 1e-2, 1e-1].map { ("all", $0, Set(grads.keys)) }
        for n in grads.keys.sorted() { trials.append((n, 1e-2, [n])) }
        for (label, alpha, which) in trials {
          var v = snap
          let gn = which.compactMap { grads[$0] }.flatMap { $0 }.reduce(Float(0)) { $0 + $1 * $1 }.squareRoot()
          for (n, g) in grads where !g.isEmpty && which.contains(n) { v[n] = zip(snap[n]!, g).map { $0 - alpha * $1 / max(gn, 1e-12) } }
          print("linesearch group \(label) predicted dL \(-alpha * gn)", terminator: " ")
          params.load(v)
          LazyGraphContext.current.clearComputationGraph()
          let rr = SnareRenderer.render(params: params, constants: constants, noiseTable: noiseTable, frames: cfg.frames, sampleRate: VillainData.sampleRate, options: options())
          let ph = floor(Clock.seconds(frames: cfg.frames, sampleRate: VillainData.sampleRate) * VillainData.sampleRate + 0.5)
          let tg = targetTable.sampleRow(ph)
          var l = Signal.constant(0)
          for w in cfg.windows where w <= cfg.frames {
            let hop = max(1, w / cfg.hopDivisor)
            l = l + spectralLossFFT(rr.output, tg, windowSize: w, lossMode: cfg.lossMode, hop: hop, normalize: true) * (cfg.linearWeight * wScale)
            l = l + spectralLossFFT(rr.output, tg, windowSize: w, useLogMagnitude: true, lossMode: cfg.lossMode, hop: hop, normalize: true) * (cfg.logWeight * wScale)
          }
          print("linesearch alpha=\(alpha) loss=\(try l.realize(frames: cfg.frames).reduce(0, +))")
        }
        params.load(snap)
      }
      let progress = Float(step) / Float(max(1, cfg.steps - 1))
      optimizer.lr = cfg.lr * max(0.02, 0.5 * (1 + Foundation.cos(Float.pi * progress)))
      optimizer.step()
      optimizer.zeroGrad()
      let dt = Date().timeIntervalSince(ts)
      stepTimes.append(dt)

      var scoreText = ""
      if (step + 1) % cfg.evalEvery == 0 || step == cfg.steps - 1 {
        if ProcessInfo.processInfo.environment["MSAD_DEBUG_EVAL"] != nil {
          let (_, a) = try evaluate()
          print("debug in-loop first eval", a.reduce(0, +) / Float(B))
        }
        let (per, scores) = try evaluate()
        let mean = scores.reduce(0, +) / Float(B)
        let med = scores.sorted()[B / 2]
        scoreText = String(format: "%.5f,%.5f", mean, med)
        for b in 0..<B where scores[b] < best[b] { best[b] = scores[b] }
        if mean < bestMean {
          bestMean = mean
          bestSnapshot = params.snapshot()
          bestRender = per
        }
        print(String(format: "step %d loss %.5f mean %.4f median %.4f (%.2f s/step)", step, stepLoss, mean, med, dt))
        let worst = (0..<B).sorted { scores[$0] - initScores[$0] > scores[$1] - initScores[$1] }.prefix(4)
        print("  worst vs init: " + worst.map { String(format: "%@ %.3f->%.3f", cfg.ids[$0], initScores[$0], scores[$0]) }.joined(separator: ", "))
      } else if step % 10 == 0 || ProcessInfo.processInfo.environment["MSAD_VERBOSE"] != nil {
        print(String(format: "step %d loss %.5f (%.2f s/step)", step, stepLoss, dt))
      }
      csv += "\(step),\(stepLoss),\(scoreText.isEmpty ? "," : scoreText),\(dt)\n"
    }
    let wall = Date().timeIntervalSince(t0)

    // Final: render best (last-eval) params, write WAVs + JSON.
    if ProcessInfo.processInfo.environment["MSAD_DEBUG_EVAL"] != nil {
      let (_, a) = try evaluate(); let (_, b2) = try evaluate()
      print("debug eval pre-load", a.reduce(0, +) / Float(B), b2.reduce(0, +) / Float(B))
    }
    params.load(bestSnapshot)
    let (finalRender, finalScores) = try evaluate()
    if ProcessInfo.processInfo.environment["MSAD_DEBUG_EVAL"] != nil {
      let (_, a) = try evaluate()
      print("debug eval post-load", finalScores.reduce(0, +) / Float(B), a.reduce(0, +) / Float(B))
    }
    // Component renders for listening (head-only and noise-only through the stage).
    var headOnly: [[Float]] = [], noiseOnly: [[Float]] = []
    do {
      LazyGraphContext.current.clearComputationGraph()
      var o = options(); o.includeNoise = false
      let r = SnareRenderer.render(params: params, constants: constants, noiseTable: noiseTable, frames: cfg.frames, sampleRate: VillainData.sampleRate, options: o)
      headOnly = deinterleave(try r.output.realize(frames: cfg.frames), batch: B)
      LazyGraphContext.current.clearComputationGraph()
      var o2 = options(); o2.includeHead = false
      let r2 = SnareRenderer.render(params: params, constants: constants, noiseTable: noiseTable, frames: cfg.frames, sampleRate: VillainData.sampleRate, options: o2)
      noiseOnly = deinterleave(try r2.output.realize(frames: cfg.frames), batch: B)
    }
    let headOnlyScores = zip(headOnly, targets).map { Scores.score($0, $1, windows: cfg.metricWindows) }
    var rows: [[String: Any]] = []
    for b in 0..<B {
      let id = cfg.ids[b]
      try WAV.write(cfg.out.appendingPathComponent("\(id).fit.wav"), finalRender[b])
      try WAV.write(cfg.out.appendingPathComponent("\(id).target.wav"), targets[b])
      try WAV.write(cfg.out.appendingPathComponent("\(id).fit_headpart.wav"), headOnly[b])
      try WAV.write(cfg.out.appendingPathComponent("\(id).fit_noisepart.wav"), noiseOnly[b])
      var row = calib[b]
      row["id"] = id
      row["init"] = initScores[b]
      row["fit"] = finalScores[b]
      row["head_only_stage"] = headOnlyScores[b]
      let wm = (row["wrong_min"] as? Float) ?? .nan
      row["fit_over_wrong_min"] = finalScores[b] / wm
      rows.append(row)
    }
    var raw: [String: [Float]] = bestSnapshot
    raw = raw.filter { !$0.value.isEmpty }
    let natural = naturalParams(params: params, constants: constants, ids: cfg.ids)
    let summary: [String: Any] = [
      "config": [
        "B": B, "frames": cfg.frames, "steps": cfg.steps, "lr": cfg.lr, "windows": cfg.windows,
        "metric_windows": cfg.metricWindows, "loss_mode": cfg.lossMode == .l1 ? "l1" : "l2",
        "unfreeze_head_at": cfg.unfreezeHeadAt ?? -1, "coupling": cfg.includeCoupling,
        "noise": cfg.includeNoise, "target": cfg.targetKind,
      ],
      "wall_s": wall, "s_per_step_median": stepTimes.sorted()[stepTimes.count / 2],
      "per_snare": rows,
      "mean_fit": finalScores.reduce(0, +) / Float(B),
      "mean_init": initScores.reduce(0, +) / Float(B),
      "mean_head_only_stage": headOnlyScores.reduce(0, +) / Float(B),
      "mean_wrong_min": rows.map { ($0["wrong_min"] as? Float) ?? 0 }.reduce(0, +) / Float(B),
      "mean_silence": rows.map { ($0["silence"] as? Float) ?? 0 }.reduce(0, +) / Float(B),
    ]
    try writeJSON(summary, to: cfg.out.appendingPathComponent("summary.json"))
    try writeJSON(["raw": raw, "natural": natural], to: cfg.out.appendingPathComponent("params.json"))
    try csv.write(to: cfg.out.appendingPathComponent("loss.csv"), atomically: true, encoding: .utf8)
    return summary
  }

  static func p_sq(_ t: DGenLazy.Tensor) -> DGenLazy.Tensor { (t * t).sum() }

  static func naturalParams(params p: SnareParams, constants c: SnareConstants, ids: [String]) -> [[String: Any]] {
    let NB = p.NB
    let A = p.rawA.getData() ?? [], D1 = p.rawD1.getData() ?? [], L = p.rawL.getData() ?? []
    let D2 = p.rawD2.getData() ?? [], C = p.rawC.getData() ?? []
    let R = p.rawRise.getData() ?? [], O = p.rawOnset.getData() ?? []
    let DV = p.rawDrive.getData() ?? [], DA = p.rawDAmt.getData() ?? [], G = p.rawGain.getData() ?? []
    let HF = p.rawHF.getData() ?? [], HA = p.rawHA.getData() ?? [], HD = p.rawHD.getData() ?? []
    return ids.indices.map { b in
      let sl = (b * NB)..<((b + 1) * NB)
      let hk = (b * p.K)..<((b + 1) * p.K)
      return [
        "id": ids[b],
        "noise_amp": sl.map { logMapCPU(A[$0], Ranges.noiseAmp.0, Ranges.noiseAmp.1) },
        "noise_d1_s": sl.map { logMapCPU(D1[$0], Ranges.noiseD1.0, Ranges.noiseD1.1) },
        "noise_tail_frac": sl.map { sigmoidF(L[$0]) },
        "noise_d2_s": sl.map { logMapCPU(D2[$0], Ranges.noiseD2.0, Ranges.noiseD2.1) },
        "noise_couple": sl.map { logMapCPU(C[$0], Ranges.noiseCouple.0, Ranges.noiseCouple.1) },
        "noise_rise_s": logMapCPU(R[b], Ranges.noiseRise.0, Ranges.noiseRise.1),
        "noise_onset_s": sigmoidF(O[b]) * Ranges.noiseOnsetMax,
        "drive": logMapCPU(DV[b], Ranges.drive.0, Ranges.drive.1),
        "drive_amt": sigmoidF(DA[b]),
        "out_gain": logMapCPU(G[b], Ranges.outGain.0, Ranges.outGain.1),
        "head_freq_mul": hk.map { Foundation.exp(Foundation.tanh(HF[$0]) * Ranges.headFreqSpan) },
        "head_amp_mul": hk.map { Foundation.exp(Foundation.tanh(HA[$0]) * Ranges.headAmpSpan) },
        "head_dec_mul": hk.map { Foundation.exp(Foundation.tanh(HD[$0])) },
      ] as [String: Any]
    }
  }
}
