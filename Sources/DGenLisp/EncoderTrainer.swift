// EncoderTrainer.swift — `train-dataset --stage encode`: an audio → latent-code encoder for a
// trained latent instrument (docs/LATENT_ENCODER_SPEC.md, phase 1).
//
//   dgenlisp train-dataset --stage encode --patch voice.lisp --encoder encoder.lisp \
//       --dataset real[,synth] --codes RUN/export/codes.json[,SYNTH/renders/labels.json] \
//       --init-from RUN --job-dir ENC [--batch 32] [--val 10] [--u-weight 1] [--prior 0]
//       [--augment] [--clip X] [--weight-decay X] [--train-only w:enc_]
//
// The encoder patch reads its per-row input with `(row-data @name feats @shape [bands frames] …)`
// and marks its [B, D] output with `(latent-out Z)`. The trainer evaluates it under the same
// dataset context and binds its output into the voice's `latent-table`, so the voice and its
// decoder are used unchanged (and frozen: they come from --init-from).
//
// Loss (capture mode: the voice stops at `param-row`, no render):
//   mean ‖enc(f(x)) − z‖²  +  u-weight · mean ‖U(enc(f(x))) − U(z)‖²  +  prior · mean nearest-code d²
// Features are host log-mel (EncoderFeatures.swift), normalised with per-band statistics of
// the training rows (export/feat_stats.json). With several --dataset directories each
// minibatch draws evenly from each; the --val rows are held out from the first.
//
// At the end the held-out rows are rendered through the voice twice -- from the encoder's
// code and from the learned code (the ceiling) -- and scored with the CPU MR/band metrics.

import DGen
import DGenLazy
import DGenTrainProtocol
import Foundation

struct EncoderState {
  var extractor: EncoderFeatureExtractor? = nil
  var config: EncoderFeatureConfig { extractor!.config }
  /// Per-row feature width.
  var W: Int { config.width }
  var stats = EncoderFeatureStats(mean: [], std: [])
  /// Normalised features per row (un-augmented).
  var feats: [[Float]] = []
  var codes: [[Float]] = []
  /// U(z) per row (empty when there is no param-row or --u-weight is 0).
  var uTargets: [[Float]] = []
  var trainIdx: [Int] = []
  var valIdx: [Int] = []
  var D = 0, P = 0
  var tableName = ""
  /// Leaves the encoder patch created ("w:…").
  var leafNames: [String] = []
  var learnables: [LearnableTensorInfo] = []
  var queues: [[Int]] = []
  var groupTrain: [[Int]] = []
  // persistent data tensors
  var featBatch: Tensor? = nil
  var zBatch: Tensor? = nil
  var uBatch: Tensor? = nil
  var featVal: Tensor? = nil
  var cloudT: Tensor? = nil
  var cloudSq: Tensor? = nil
}

extension DatasetTrainer {
  var B: Int { min(o.batch, enc.trainIdx.count) }
  var K: Int { enc.valIdx.count }

  // MARK: data

  /// Mean squared error of a code against row r's label.
  func codeMSE(_ z: [Float], row r: Int) -> Double {
    zip(z, enc.codes[r]).reduce(0.0) { $0 + Double(($1.0 - $1.1) * ($1.0 - $1.1)) } / Double(enc.D)
  }

  /// Raw (un-normalised) features, from --features or the host log-mel of the targets.
  func rawFeatures(_ cfg: EncoderFeatureConfig) throws -> [[Float]] {
    if let path = o.features {
      let j = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: path)))
      guard let d = j as? [String: Any], let n = d["names"] as? [String],
        let rows = d["data"] as? [[NSNumber]]
      else { throw TrainProtocolError("features \(path): expected {names, shape, data}") }
      var byName: [String: [Float]] = [:]
      for (k, r) in zip(n, rows) { byName[k] = r.map(\.floatValue) }
      return try names.map {
        guard let f = byName[$0], f.count == cfg.width else {
          throw TrainProtocolError("features \(path): no \(cfg.width)-value row for \($0)")
        }
        return f
      }
    }
    let ex = enc.extractor!
    var out = [[Float]](repeating: [], count: N)
    out.withUnsafeMutableBufferPointer { buf in
      DispatchQueue.concurrentPerform(iterations: N) { r in buf[r] = ex.logMel(targets[r]) }
    }
    return out
  }

  func prepareEncoder() throws {
    guard let table = tables.first else { throw TrainProtocolError("encode: the voice has no latent-table") }
    enc.tableName = table.name
    enc.D = table.dim
    for info in learnables where store.preset["w:" + info.name] == nil {
      throw TrainProtocolError("encode: decoder weight \(info.name) is not in --init-from (train the voice first)")
    }
    var cfg = EncoderFeatureConfig(sampleRate: sampleRate)
    cfg.window = o.featWindow; cfg.hop = o.featHop; cfg.bands = o.featBands; cfg.seconds = o.featSeconds
    guard cfg.frames > 0, cfg.cropSamples <= o.frames else {
      throw TrainProtocolError("encode: --feat-seconds \(cfg.seconds) must cover one window and fit in --frames \(o.frames)")
    }
    enc.extractor = EncoderFeatureExtractor(cfg)

    // labels: codes files map by name onto each dataset group
    enc.codes = [[Float]](repeating: [], count: N)
    for (g, path) in zip(groups, o.codes) {
      let c = try DatasetCodes.load(path)
      guard c.dim == enc.D else {
        throw TrainProtocolError("codes \(path) have dim \(c.dim); latent-table \(table.name) has \(enc.D)")
      }
      var byName: [String: [Float]] = [:]
      for (n, z) in zip(c.names, c.codes) { byName[n] = z }
      for r in g {
        guard let z = byName[names[r]], z.count == enc.D else {
          throw TrainProtocolError("codes \(path) have no code for \(names[r])")
        }
        enc.codes[r] = z
      }
    }

    // split: --val rows held out from the first dataset
    var first = Array(groups[0])
    rng.shuffle(&first)
    let k = min(o.val, max(0, first.count - 1))
    enc.valIdx = Array(first.prefix(k)).sorted()
    let val = Set(enc.valIdx)
    enc.groupTrain = groups.map { $0.filter { !val.contains($0) } }.filter { !$0.isEmpty }
    enc.trainIdx = enc.groupTrain.flatMap { $0 }
    enc.queues = enc.groupTrain.map { _ in [] }
    guard !enc.trainIdx.isEmpty else { throw TrainProtocolError("encode: no training rows left after --val") }

    // features + normalisation from the training rows
    let raw = try rawFeatures(cfg)
    enc.stats = EncoderFeatureStats.fit(enc.trainIdx.map { raw[$0] }, bands: cfg.bands, frames: cfg.frames)
    enc.feats = raw.map { enc.stats.apply($0, bands: cfg.bands, frames: cfg.frames) }

    // U(z) targets, decoded once from the labels (decoder-only, chunked)
    if let pr = paramRows.first, let L = layouts[pr.layout], o.uWeight > 0 {
      enc.P = L.width
      enc.uTargets = []
      let chunk = 256
      var start = 0
      while start < N {
        let n = min(chunk, N - start)
        LazyGraphContext.current.clearComputationGraph()
        let z = Tensor(enc.codes[start..<(start + n)].flatMap { $0 }).reshape([n, enc.D])
        let (_, ctx) = try evaluate(mode: .capture, rows: n, latent: { _, _ in z })
        guard let u = ctx.captured[pr.name] else { throw TrainProtocolError("param-row was not reached") }
        let flat = try realizeStatic(u, count: n * L.width)
        for r in 0..<n { enc.uTargets.append(Array(flat[(r * L.width)..<((r + 1) * L.width)])) }
        start += n
      }
    }

    // persistent data tensors, created before the first step
    LazyGraphContext.current.clearComputationGraph()
    enc.featBatch = Tensor([Float](repeating: 0, count: B * enc.W)).reshape([B, enc.W])
    enc.zBatch = Tensor([Float](repeating: 0, count: B * enc.D)).reshape([B, enc.D])
    if enc.P > 0 { enc.uBatch = Tensor([Float](repeating: 0, count: B * enc.P)).reshape([B, enc.P]) }
    if K > 0 {
      enc.featVal = Tensor(enc.valIdx.flatMap { enc.feats[$0] }).reshape([K, enc.W])
    }
    if o.prior > 0 {
      // the code cloud: every training row's code, [D, M] and their squared norms [1, M]
      let cloud = enc.trainIdx.map { enc.codes[$0] }
      let M = cloud.count
      var ct = [Float](repeating: 0, count: enc.D * M)
      for (j, c) in cloud.enumerated() { for d in 0..<enc.D { ct[d * M + j] = c[d] } }
      enc.cloudT = Tensor(ct).reshape([enc.D, M])
      enc.cloudSq = Tensor(cloud.map { $0.reduce(0) { $0 + $1 * $1 } }).reshape([1, M])
    }

    let ex = run.appendingPathComponent("export")
    try enc.stats.write(ex.appendingPathComponent("feat_stats.json"), config: cfg)
    try writeTensorJSON([cfg.bins, cfg.bands], enc.extractor!.melFilters,
                        to: ex.appendingPathComponent("mel_filters.json"))
    log("[encode] \(enc.trainIdx.count) train rows in \(enc.groupTrain.count) source(s), \(K) val; features [\(cfg.bands), \(cfg.frames)] = \(enc.W); D \(enc.D), P \(enc.P); batch \(B)")
  }

  // MARK: encoder evaluation

  /// Evaluates the encoder patch on `feats` [rows, W]; returns its `latent-out` [rows, D].
  func encoderForward(rows: Int, feats: Tensor) throws -> Tensor {
    var created: [String] = []
    let ctx = DatasetContext(
      rows: rows, frames: o.frames, sampleRate: sampleRate, rowMode: .decode,
      learnable: { [self] name, shape, fileData, initKind in
        let info = LearnableTensorInfo(name: name, shape: shape, file: nil, initKind: initKind)
        created.append("w:" + name)
        return store.leaf("w:" + name, shape) { initData(info, fileData: fileData) }
      },
      latentRows: { name, _ in
        throw TrainProtocolError("encoder patch: latent-table \(name) is not allowed (the encoder makes the code)")
      })
    let cfg = enc.config
    ctx.rowData = { [self] name, shape in
      guard shape == [cfg.bands, cfg.frames] || shape == [enc.W] else {
        throw TrainProtocolError(
          "row-data \(name): @shape \(shape) does not match the features [\(cfg.bands) \(cfg.frames)] ([bands frames], time last)")
      }
      return feats
    }
    let ev = LispEvaluator(sourceDirectory: encoderDir!)
    ev.dataset = ctx
    do {
      try ev.evaluate(nodes: encoderNodes)
    } catch let e as LispError {
      throw TrainProtocolError("encoder evaluation failed: \(e.message)")
    }
    guard ctx.latentOut.count == 1, let z = ctx.latentOut.first else {
      throw TrainProtocolError("encoder patch must call (latent-out Z) exactly once (got \(ctx.latentOut.count))")
    }
    guard z.shape == [rows, enc.D] else {
      throw TrainProtocolError("encoder latent-out is \(z.shape), expected [\(rows), \(enc.D)]")
    }
    if enc.leafNames.isEmpty {
      let voice = Set(learnables.map { "w:" + $0.name })
      if let clash = created.first(where: voice.contains) {
        throw TrainProtocolError("encoder tensor \(clash.dropFirst(2)) has the same name as a decoder weight")
      }
      enc.leafNames = created
      enc.learnables = ev.learnableTensors
    }
    return z
  }

  // MARK: minibatches

  func nextBatch() -> [Int] {
    var rows: [Int] = []
    for i in 0..<B {
      let g = i % enc.groupTrain.count
      if enc.queues[g].isEmpty {
        var q = enc.groupTrain[g]
        rng.shuffle(&q)
        enc.queues[g] = q
      }
      rows.append(enc.queues[g].removeLast())
    }
    return rows
  }

  /// Features for a batch: cached, or recomputed from augmented audio.
  func batchFeatures(_ rows: [Int]) -> [Float] {
    guard o.augment, let ex = enc.extractor else {
      return rows.flatMap { enc.feats[$0] }
    }
    let seeds = rows.map { _ in rng.next() }
    let cfg = ex.config, stats = enc.stats
    var out = [[Float]](repeating: [], count: rows.count)
    out.withUnsafeMutableBufferPointer { buf in
      DispatchQueue.concurrentPerform(iterations: rows.count) { i in
        var g = SplitMix64(seed: seeds[i])
        let (x, tilt) = EncoderAugment.apply(targets[rows[i]], sampleRate: sampleRate, rng: &g)
        var f = ex.logMel(x)
        EncoderAugment.tilt(&f, dB: tilt, bands: cfg.bands, frames: cfg.frames)
        buf[i] = stats.apply(f, bands: cfg.bands, frames: cfg.frames)
      }
    }
    return out.flatMap { $0 }
  }

  // MARK: stage

  func encode() throws -> [String: Any] {
    try prepareEncoder()
    let pr = paramRows.first
    let trainable: () -> [String] = { [self] in
      if o.trainOnly.isEmpty { return enc.leafNames }
      return store.order.filter { n in o.trainOnly.contains { n.hasPrefix($0) } }
    }
    try loop({ [self] step in
      if o.evalEvery > 0, step % o.evalEvery == 0, K > 0 {
        log(String(format: "[encode] step %d val code MSE %.5f", step, try valCodeErrors().reduce(0, +) / Double(K)))
      }
      LazyGraphContext.current.clearComputationGraph()
      let rows = nextBatch()
      enc.featBatch!.updateDataLazily(batchFeatures(rows))
      enc.zBatch!.updateDataLazily(rows.flatMap { enc.codes[$0] })
      let zHat = try encoderForward(rows: B, feats: enc.featBatch!)
      let d = zHat - enc.zBatch!
      var loss = (d * d).sum() * (1 / Float(B * enc.D))
      if let pr, let uBatch = enc.uBatch {
        uBatch.updateDataLazily(rows.flatMap { enc.uTargets[$0] })
        let (_, ctx) = try evaluate(mode: .capture, rows: B, latent: { _, _ in zHat })
        guard let u = ctx.captured[pr.name] else { throw TrainProtocolError("param-row was not reached") }
        let du = u - uBatch
        loss = loss + (du * du).sum() * (o.uWeight / Float(B * enc.P))
      }
      if o.prior > 0, let cloudT = enc.cloudT, let cloudSq = enc.cloudSq {
        // squared distance to the nearest training code: ‖z‖² + ‖c‖² − 2 z·c, min over codes
        let zz = (zHat * zHat).sum(axis: 1).reshape([B, 1])
        let d2 = zz + cloudSq - zHat.matmul(cloudT) * 2
        let nearest = (d2 * -1).max(axis: 1) * -1
        loss = loss + nearest.sum() * (o.prior / Float(B))
      }
      let v = try staticToSignal(loss, frames: 1).backward(frames: 1).reduce(0, +)
      return (0, v)
    }, trainable: trainable)
    return try finishEncoder()
  }

  /// The encoder's codes for the held-out rows.
  func valCodes() throws -> [[Float]] {
    LazyGraphContext.current.clearComputationGraph()
    let z = try encoderForward(rows: K, feats: enc.featVal!)
    let flat = try realizeStatic(z, count: K * enc.D)
    return (0..<K).map { Array(flat[($0 * enc.D)..<(($0 + 1) * enc.D)]) }
  }

  func valCodeErrors() throws -> [Double] {
    zip(try valCodes(), enc.valIdx).map { codeMSE($0, row: $1) }
  }

  /// Encoder codes for every row (in batches through the persistent batch tensor).
  func encodeAll() throws -> [[Float]] {
    var out: [[Float]] = []
    var start = 0
    while start < N {
      let n = min(B, N - start)
      var f = (start..<(start + n)).flatMap { enc.feats[$0] }
      f += [Float](repeating: 0, count: (B - n) * enc.W)
      LazyGraphContext.current.clearComputationGraph()
      enc.featBatch!.updateDataLazily(f)
      let z = try encoderForward(rows: B, feats: enc.featBatch!)
      let flat = try realizeStatic(z, count: B * enc.D)
      for r in 0..<n { out.append(Array(flat[(r * enc.D)..<((r + 1) * enc.D)])) }
      start += n
    }
    return out
  }

  /// Renders `codes` (one per row of `rows`) through the frozen voice; [row][frame].
  func renderCodes(_ codes: [[Float]], rows: [Int]) throws -> [[Float]] {
    let n = rows.count
    LazyGraphContext.current.clearComputationGraph()
    let z = Tensor(codes.flatMap { $0 }).reshape([n, enc.D])
    let (ev, _) = try evaluate(mode: .decode, rows: n, latent: { _, _ in z })
    let flat = try output(ev, rows: n).realize(frames: o.frames)
    return (0..<n).map { r in (0..<o.frames).map { flat[$0 * n + r] } }
  }

  func finishEncoder() throws -> [String: Any] {
    try exportLearnables(enc.learnables)
    let all = try encodeAll()
    try DatasetCodes.write(run.appendingPathComponent("export/encoded.json"), names: names,
                           codes: all, table: enc.tableName, extra: ["val": enc.valIdx.map { names[$0] }])
    var summary: [String: Any] = ["stage": o.stage, "steps": o.steps, "val": K, "train": enc.trainIdx.count]
    let trainCodeErr = enc.trainIdx.map { codeMSE(all[$0], row: $0) }
    summary["train_code_mse_median"] = median(trainCodeErr)
    guard K > 0 else {
      try writeSummary(summary)
      return summary
    }
    let zHat = enc.valIdx.map { all[$0] }
    let codeErr = zip(zHat, enc.valIdx).map { codeMSE($0, row: $1) }
    summary["val_code_mse_median"] = median(codeErr)
    guard !o.noRender else {
      try writeSummary(summary)
      return summary
    }
    let encRenders = try renderCodes(zHat, rows: enc.valIdx)
    let ceilRenders = try renderCodes(enc.valIdx.map { enc.codes[$0] }, rows: enc.valIdx)
    let trainPeak = enc.trainIdx.map { targets[$0].map(abs).max() ?? 0 }.max() ?? 0
    var perRow: [String: Any] = [:]
    var encMR: [Double] = [], ceilMR: [Double] = [], encBand: [Double] = [], ceilBand: [Double] = []
    var peaks: [Double] = [], ceilPeaks: [Double] = []
    let windows = [64, 128, 256, 512, 1024, 2048, 4096]
    var sc = [((Double, Double), (Double, Double))](repeating: ((0, 0), (0, 0)), count: K)
    DispatchQueue.concurrentPerform(iterations: K) { i in
      let t = targets[enc.valIdx[i]]
      let a = encRenders[i].allSatisfy(\.isFinite) ? MetricLoss.scores(encRenders[i], t, windows: windows) : (.infinity, .infinity)
      let b = ceilRenders[i].allSatisfy(\.isFinite) ? MetricLoss.scores(ceilRenders[i], t, windows: windows) : (.infinity, .infinity)
      sc[i] = ((a.0, a.1), (b.0, b.1))
    }
    let dir = run.appendingPathComponent("renders/val")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    for (i, r) in enc.valIdx.enumerated() {
      let n = names[r]
      let pk = Double(encRenders[i].map(abs).max() ?? 0)
      try FloatWav.write(dir.appendingPathComponent("\(n)-enc.wav"), encRenders[i], sampleRate: Int(sampleRate))
      try FloatWav.write(dir.appendingPathComponent("\(n)-code.wav"), ceilRenders[i], sampleRate: Int(sampleRate))
      perRow[n] = [
        "code_mse": codeErr[i], "mr": sc[i].0.1, "band": sc[i].0.0, "ceiling_mr": sc[i].1.1,
        "ceiling_band": sc[i].1.0, "peak": pk, "peak_ratio": pk / Double(max(trainPeak, 1e-9)),
      ]
      encBand.append(sc[i].0.0); encMR.append(sc[i].0.1)
      ceilBand.append(sc[i].1.0); ceilMR.append(sc[i].1.1)
      peaks.append(pk / Double(max(trainPeak, 1e-9)))
      ceilPeaks.append(Double(ceilRenders[i].map(abs).max() ?? 0) / Double(max(trainPeak, 1e-9)))
    }
    let mrRatio = median(encMR) / max(median(ceilMR), 1e-12)
    let peakMax = peaks.max() ?? 0
    summary["val_mr_median"] = median(encMR)
    summary["val_band_median"] = median(encBand)
    summary["ceiling_mr_median"] = median(ceilMR)
    summary["ceiling_band_median"] = median(ceilBand)
    summary["mr_ratio"] = mrRatio
    summary["peak_ratio_max"] = peakMax
    // the learned codes' own renders, for context: a voice that is not level-matched to the
    // (peak-normalised) targets exceeds 1 here too
    summary["ceiling_peak_ratio_max"] = ceilPeaks.max() ?? 0
    summary["train_peak"] = Double(trainPeak)
    // §4 acceptance: median MR within 15 % of the learned-code ceiling; no val render above
    // 1.2× the loudest training hit.
    summary["accept"] = mrRatio <= 1.15 && peakMax <= 1.2
    summary["scores"] = perRow
    try writeSummary(summary)
    log(String(format: "[encode] done: val code MSE median %.4f; MR median %.4f vs ceiling %.4f (×%.3f); band %.3f vs %.3f; peak ≤ %.2f× training (ceiling %.2f×); accept %@",
               median(codeErr), median(encMR), median(ceilMR), mrRatio, median(encBand), median(ceilBand),
               peakMax, ceilPeaks.max() ?? 0, (summary["accept"] as! Bool) ? "yes" : "no"))
    return summary
  }

  func median(_ v: [Double]) -> Double {
    guard !v.isEmpty else { return .nan }
    let s = v.sorted()
    return s.count % 2 == 1 ? s[s.count / 2] : (s[s.count / 2 - 1] + s[s.count / 2]) / 2
  }

  func writeSummary(_ summary: [String: Any]) throws {
    try JSONSerialization.data(withJSONObject: summary, options: [.prettyPrinted, .sortedKeys])
      .write(to: run.appendingPathComponent("summary-\(o.stage).json"))
  }
}
