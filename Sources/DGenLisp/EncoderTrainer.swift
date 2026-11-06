// EncoderTrainer.swift — `train-dataset --stage encode`: an audio → latent-code encoder for a
// trained latent instrument (docs/LATENT_ENCODER_SPEC.md, phase 1).
//
//   dgenlisp train-dataset --stage encode --patch voice.lisp --encoder encoder.lisp \
//       --dataset real[,synth] --codes RUN/export/codes.json[,SYNTH/renders/labels.json] \
//       --init-from RUN --job-dir ENC [--batch 32] [--val 10] [--u-weight 1] [--prior 0]
//       [--augment] [--clip X] [--weight-decay X] [--train-only w:enc_]
//
// The encoder patch reads its per-row input with `(row-data @name feats @shape [bands frames] …)`
// and, with --aux-features FILE ({names, features: [[Float]]}, e.g. analyze-kicks measurements),
// a per-row vector `(row-data @name aux @shape [A])`, z-scored with the training rows' statistics.
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
// --audio-loss --low-weight W --rows FILE: + W · the low-end anchor (lowEndLoss): for every mode
// whose row frequency is below lowHz, its pitch curve hz(t) = f·(1 + g·e^(−t/td) [+ g2·e^(−t/t2)
// on mode 0]) in semitones at bodyTimes, its level in dB/6 above a floor 40 dB under mode 0,
// and its decay in (semitone-scaled log ratio)/4, each vs the row's analysis values.
//
// At the end the held-out rows are rendered through the voice twice -- from the encoder's
// code and from the learned code (the ceiling) -- and scored with the CPU MR/band metrics.
//
// --amortized (the encoder DEFINES the codes; no --codes, --rows required):
//   mean ‖U(enc(f(x))) − u_x‖²  +  path · mean ‖U(mix(z_a, z_b)) − mix(U(z_a), U(z_b))‖²
//   +  code-l2 · mean ‖enc(f(x))‖²
// with the encoder AND the decoder training (every `w:` leaf). The held-out rows are kept out of
// both, so their renders measure how the space generalises to a kick it never saw. The export
// holds the decoder weights and export/codes.json (the encoder's code for every row).

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
  /// Rows with a code label (unlabelled rows are allowed under --audio-loss: graded on sound only).
  var labeled: [Bool] = []
  /// U(z) per row (empty when there is no param-row or --u-weight is 0).
  var uTargets: [[Float]] = []
  /// Per-column weights of the U loss (0 on --freeze segments), [1, P].
  var uColumns: Tensor? = nil
  /// --amortized --path: in-batch selector of in-between codes, [M, B].
  var pathSel: DatasetTrainer.PathSelect? = nil
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
  /// 1 for a labelled row of the batch, 0 for an unlabelled one, [B, 1].
  var labelBatch: Tensor? = nil
  var uBatch: Tensor? = nil
  var featVal: Tensor? = nil
  /// --aux-features: normalised per-row vectors, and their batch / held-out tensors.
  var aux: [[Float]] = []
  var auxBatch: Tensor? = nil
  var auxVal: Tensor? = nil
  /// --low-weight: per-row targets (lowTargets layout: per mode the curve's ln hz at bodyTimes,
  /// ln(amp + floor), ln dec; then the floor) and low-mode mask, their batches, times and ones [1, T].
  var lowTargets: [[Float]] = []
  var lowMask: [[Float]] = []
  var lowBatch: Tensor? = nil
  var lowMaskBatch: Tensor? = nil
  var lowTimes: Tensor? = nil
  var lowOnes: Tensor? = nil
  var cloudT: Tensor? = nil
  var cloudSq: Tensor? = nil
  // --audio-loss: persistent latch leaves / their gradients for B rows, and dL/dy [frames, B]
  var latchLeaves: [Tensor] = []
  var latchGrads: [Tensor] = []
  var gBatch: Tensor? = nil
}

extension DatasetTrainer {
  var B: Int { min(o.batch, enc.trainIdx.count) }
  var K: Int { enc.valIdx.count }
  var hasCodes: Bool { !o.codes.isEmpty }

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
    for info in learnables where !o.amortized && store.preset["w:" + info.name] == nil {
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
    enc.labeled = [Bool](repeating: true, count: N)
    // graded on sound, a row needs no code (codes only anchor --prior and --code-weight)
    let unlabeledOK = o.audioLoss
    for (g, path) in zip(groups, o.codes) {  // none with --amortized
      let c = try DatasetCodes.load(path)
      guard c.dim == enc.D else {
        throw TrainProtocolError("codes \(path) have dim \(c.dim); latent-table \(table.name) has \(enc.D)")
      }
      var byName: [String: [Float]] = [:]
      for (n, z) in zip(c.names, c.codes) { byName[n] = z }
      for r in g {
        guard let z = byName[names[r]], z.count == enc.D else {
          if unlabeledOK {
            enc.codes[r] = [Float](repeating: 0, count: enc.D)
            enc.labeled[r] = false
            continue
          }
          throw TrainProtocolError("codes \(path) have no code for \(names[r])")
        }
        enc.codes[r] = z
      }
    }

    // split: --val-names, else --val rows held out from the first dataset
    if o.valNames.isEmpty {
      var first = Array(groups[0])
      rng.shuffle(&first)
      let k = min(o.val, max(0, first.count - 1))
      enc.valIdx = Array(first.prefix(k)).sorted()
    } else {
      let row = Dictionary(names.enumerated().map { ($1, $0) }, uniquingKeysWith: { a, _ in a })
      enc.valIdx = try o.valNames.map {
        guard let r = row[$0] else { throw TrainProtocolError("--val-names: no target named \($0)") }
        return r
      }.sorted()
    }
    let val = Set(enc.valIdx)
    enc.groupTrain = groups.map { $0.filter { !val.contains($0) } }.filter { !$0.isEmpty }
    enc.trainIdx = enc.groupTrain.flatMap { $0 }
    enc.queues = enc.groupTrain.map { _ in [] }
    guard !enc.trainIdx.isEmpty else { throw TrainProtocolError("encode: no training rows left after --val") }
    if let r = enc.valIdx.first(where: { !enc.labeled[$0] }) {
      throw TrainProtocolError("held-out row \(names[r]) has no code (the held-out rows need one)")
    }

    // features + normalisation from the training rows
    let raw = try rawFeatures(cfg)
    if let path = o.featStats {
      let (stats, c) = try EncoderFeatureStats.load(URL(fileURLWithPath: path))
      guard stats.mean.count == cfg.bands, c.window == cfg.window, c.hop == cfg.hop, c.fMin == cfg.fMin else {
        throw TrainProtocolError("--feat-stats \(path): made for \(stats.mean.count) bands, window \(c.window), hop \(c.hop); features here are \(cfg.bands), \(cfg.window), \(cfg.hop)")
      }
      enc.stats = stats
    } else {
      enc.stats = EncoderFeatureStats.fit(enc.trainIdx.map { raw[$0] }, bands: cfg.bands, frames: cfg.frames)
    }
    enc.feats = raw.map { enc.stats.apply($0, bands: cfg.bands, frames: cfg.frames) }
    if let path = o.auxFeatures {
      guard !o.augment else {
        throw TrainProtocolError("--aux-features describe the un-augmented audio; drop --augment")
      }
      guard let d = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: path))) as? [String: Any],
        let an = d["names"] as? [String], let av = d["features"] as? [[NSNumber]], an.count == av.count
      else { throw TrainProtocolError("--aux-features \(path): expected {names, features: [[Float]]}") }
      let by = Dictionary(zip(an, av.map { $0.map(\.floatValue) }), uniquingKeysWith: { a, _ in a })
      let A = av.first?.count ?? 0
      let rawAux = try names.map { n -> [Float] in
        guard let v = by[n], v.count == A, A > 0 else { throw TrainProtocolError("--aux-features \(path): no \(A)-vector for \(n)") }
        return v
      }
      // z-score each dimension with the training rows
      var mean = [Float](repeating: 0, count: A), sd = mean
      for r in enc.trainIdx { for j in 0..<A { mean[j] += rawAux[r][j] } }
      mean = mean.map { $0 / Float(enc.trainIdx.count) }
      for r in enc.trainIdx { for j in 0..<A { sd[j] += (rawAux[r][j] - mean[j]) * (rawAux[r][j] - mean[j]) } }
      sd = sd.map { Swift.max(($0 / Float(enc.trainIdx.count)).squareRoot(), 1e-6) }
      enc.aux = rawAux.map { v in (0..<A).map { (v[$0] - mean[$0]) / sd[$0] } }
      try JSONSerialization.data(withJSONObject: ["mean": mean.map(Double.init), "std": sd.map(Double.init)])
        .write(to: run.appendingPathComponent("export/aux_stats.json"))
    }

    if o.amortized {
      // U targets: the refined rows themselves
      guard let pr = paramRows.first, let L = layouts[pr.layout], let u = rowsU[pr.name] else {
        throw TrainProtocolError("--amortized needs a param-row and --rows")
      }
      enc.P = L.width
      enc.uTargets = (0..<N).map { Array(u[($0 * L.width)..<(($0 + 1) * L.width)]) }
    } else if let pr = paramRows.first, let L = layouts[pr.layout], o.uWeight > 0 {
      // U(z) targets, decoded once from the labels (decoder-only, chunked)
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

    if o.lowWeight > 0 {
      guard o.audioLoss, let pr = paramRows.first, let L = layouts[pr.layout], let u = rowsU[pr.name] else {
        throw TrainProtocolError("--low-weight needs --audio-loss, a param-row and --rows")
      }
      let seg = try Dictionary(uniqueKeysWithValues: Self.lowSegments.map { n -> (String, RowSegment) in
        guard let s = L.segment(n) else { throw TrainProtocolError("--low-weight: layout has no segment \(n)") }
        return (n, s)
      })
      let M = seg["f"]!.width
      for r in 0..<N {
        func v(_ n: String, _ i: Int) -> Float { seg[n]!.fromU(u[r * L.width + seg[n]!.offset + i]) }
        let floor = v("amp", 0) * 0.01
        var t: [Float] = []
        for i in 0..<M {
          let (g2, t2): (Float, Float) = i == 0 ? (v("g2", 0), v("t2", 0)) : (0, 1)
          t += Self.bodyTimes.map { x in
            Foundation.log(v("f", i) * (1 + v("g", i) * Foundation.exp(-x / v("td", i)) + g2 * Foundation.exp(-x / t2)))
          }
          t += [Foundation.log(v("amp", i) + floor), Foundation.log(v("dec", i))]
        }
        enc.lowTargets.append(t + [floor])
        enc.lowMask.append((0..<M).map { v("f", $0) < Self.lowHz ? 1 : 0 })
      }
    }

    // persistent data tensors, created before the first step
    LazyGraphContext.current.clearComputationGraph()
    enc.featBatch = Tensor([Float](repeating: 0, count: B * enc.W)).reshape([B, enc.W])
    enc.zBatch = Tensor([Float](repeating: 0, count: B * enc.D)).reshape([B, enc.D])
    enc.labelBatch = Tensor([Float](repeating: 1, count: B)).reshape([B, 1])
    if enc.P > 0 { enc.uBatch = Tensor([Float](repeating: 0, count: B * enc.P)).reshape([B, enc.P]) }
    if o.amortized, let pr = paramRows.first, let L = layouts[pr.layout] {
      enc.uColumns = Tensor(columnWeights(L)).reshape([1, L.width])
      if o.path > 0 { enc.pathSel = PathSelect(M: o.pathPairs, N: B) }
    }
    if K > 0 {
      enc.featVal = Tensor(enc.valIdx.flatMap { enc.feats[$0] }).reshape([K, enc.W])
    }
    if let A = enc.aux.first?.count {
      enc.auxBatch = Tensor([Float](repeating: 0, count: B * A)).reshape([B, A])
      if K > 0 { enc.auxVal = Tensor(enc.valIdx.flatMap { enc.aux[$0] }).reshape([K, A]) }
    }
    if o.lowWeight > 0, let t = enc.lowTargets.first, let m = enc.lowMask.first {
      let T = Self.bodyTimes.count
      enc.lowBatch = Tensor([Float](repeating: 0, count: B * t.count)).reshape([B, t.count])
      enc.lowMaskBatch = Tensor([Float](repeating: 0, count: B * m.count)).reshape([B, m.count])
      enc.lowTimes = Tensor(Self.bodyTimes).reshape([1, T])
      enc.lowOnes = Tensor([Float](repeating: 1, count: T)).reshape([1, T])
    }
    if o.prior > 0 {
      // the code cloud: every training row's code, [D, M] and their squared norms [1, M]
      let cloud = enc.trainIdx.filter { enc.labeled[$0] }.map { enc.codes[$0] }
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
    log("[encode]\(o.amortized ? " amortized:" : "") \(enc.trainIdx.count) train rows in \(enc.groupTrain.count) source(s), \(K) val; features [\(cfg.bands), \(cfg.frames)] = \(enc.W); D \(enc.D), P \(enc.P); batch \(B)")
  }

  // MARK: encoder evaluation

  /// Evaluates the encoder patch on `feats` [rows, W] (and `aux` [rows, A] under --aux-features);
  /// returns its `latent-out` [rows, D].
  func encoderForward(rows: Int, feats: Tensor, aux: Tensor?) throws -> Tensor {
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
      if name == "aux" {
        guard let aux, shape == [aux.shape[1]] else {
          throw TrainProtocolError("row-data aux: @shape \(shape) needs --aux-features with that width")
        }
        return aux
      }
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
    let cfg = ex.config, stats = enc.stats, noise = o.augNoise
    var out = [[Float]](repeating: [], count: rows.count)
    out.withUnsafeMutableBufferPointer { buf in
      DispatchQueue.concurrentPerform(iterations: rows.count) { i in
        var g = SplitMix64(seed: seeds[i])
        let (x, tilt) = EncoderAugment.apply(targets[rows[i]], sampleRate: sampleRate, noise: noise, rng: &g)
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
      if o.trainOnly.isEmpty { return o.amortized ? store.order.filter { $0.hasPrefix("w:") } : enc.leafNames }
      return store.order.filter { n in o.trainOnly.contains { n.hasPrefix($0) } }
    }
    try loop({ [self] step in
      if o.evalEvery > 0, step % o.evalEvery == 0, K > 0, o.amortized {
        log(String(format: "[encode] step %d val U MSE %.5f", step, try valUError()))
      }
      if o.evalEvery > 0, step % o.evalEvery == 0, K > 0, hasCodes {
        log(String(format: "[encode] step %d val code MSE %.5f", step, try valCodeErrors().reduce(0, +) / Double(K)))
      }
      if o.evalEvery > 0, step % o.evalEvery == 0, K > 0, o.audioLoss {
        let mr = try valMR()
        log(String(format: "[encode] step %d val MR median %.4f mean %.4f", step, median(mr), mr.reduce(0, +) / Double(K)))
      }
      LazyGraphContext.current.clearComputationGraph()
      let rows = nextBatch()
      enc.featBatch!.updateDataLazily(batchFeatures(rows))
      enc.auxBatch?.updateDataLazily(rows.flatMap { enc.aux[$0] })
      enc.zBatch!.updateDataLazily(rows.flatMap { enc.codes[$0] })
      if o.audioLoss { return try encoderAudioStep(rows: rows) }
      let zHat = try encoderForward(rows: B, feats: enc.featBatch!, aux: enc.auxBatch)
      if o.amortized { return (0, try amortizedLoss(rows: rows, zHat: zHat)) }
      let d = zHat - enc.zBatch!
      var loss = (d * d).sum() * (1 / Float(B * enc.D))
      if let pr, let uBatch = enc.uBatch {
        uBatch.updateDataLazily(rows.flatMap { enc.uTargets[$0] })
        let (_, ctx) = try evaluate(mode: .capture, rows: B, latent: { _, _ in zHat })
        guard let u = ctx.captured[pr.name] else { throw TrainProtocolError("param-row was not reached") }
        let du = u - uBatch
        loss = loss + (du * du).sum() * (o.uWeight / Float(B * enc.P))
      }
      if let p = priorLoss(zHat) { loss = loss + p }
      let v = try staticToSignal(loss, frames: 1).backward(frames: 1).reduce(0, +)
      return (0, v)
    }, trainable: trainable)
    return try finishEncoder()
  }

  /// --prior: squared distance from each code to the nearest training code (‖z‖² + ‖c‖² − 2 z·c,
  /// min over codes), mean over the batch, times --prior.
  func priorLoss(_ z: Tensor) -> Tensor? {
    guard o.prior > 0, let cloudT = enc.cloudT, let cloudSq = enc.cloudSq else { return nil }
    let B = z.shape[0]
    let d2 = (z * z).sum(axis: 1).reshape([B, 1]) + cloudSq - z.matmul(cloudT) * 2
    return ((d2 * -1).max(axis: 1) * -1).sum() * (o.prior / Float(B))
  }

  /// --audio-loss step: the latch-boundary step of `latchStep`, for a minibatch whose codes come
  /// from the encoder (decoder frozen):
  ///   1. static pass: encoder → decoder → realise every latched value of the B rows;
  ///   2. voice pass: render from leaves holding them, CPU metric loss against the batch's own
  ///      targets, backward of Σ y·g → dL/dleaf;
  ///   3. static pass again: backward of Σ latched·dL/dleaf + code-weight · code MSE.
  func encoderAudioStep(rows: [Int]) throws -> (Float, Float) {
    let B = rows.count
    var zHat: Tensor? = nil
    let encoded: (String, Int) throws -> Tensor = { [self] _, _ in
      let z = try encoderForward(rows: B, feats: enc.featBatch!, aux: enc.auxBatch)
      zHat = z
      return z
    }
    // 1
    let (_, c1) = try evaluate(mode: .decode, latch: .capture, rows: B, latent: encoded)
    let values = try realizeAll(c1.latched)
    if enc.latchLeaves.isEmpty {
      enc.latchLeaves = zip(c1.latched, values).map { Tensor.param($0.shape, data: $1) }
      enc.latchGrads = c1.latched.map { Tensor([Float](repeating: 0, count: $0.shape.reduce(1, *))).reshape($0.shape) }
      enc.gBatch = Tensor([Float](repeating: 0, count: o.frames * B)).reshape([o.frames, B])
    }
    guard enc.latchLeaves.count == values.count else { throw TrainProtocolError("latch count changed between steps") }
    // 2: the latched leaves stand in for everything upstream (the code is not used)
    let zeros: (String, Int) throws -> Tensor = { _, d in Tensor([Float](repeating: 0, count: B * d)).reshape([B, d]) }
    LazyGraphContext.current.clearComputationGraph()
    for (leaf, v) in zip(enc.latchLeaves, values) { leaf.updateDataLazily(v); leaf.grad = nil }
    let (ev, _) = try evaluate(mode: .decode, latch: .leaves(enc.latchLeaves), rows: B, latent: zeros)
    let flat = try output(ev, rows: B).realize(frames: o.frames)
    let (la, g) = metricGradient(flat, rows: rows)
    LazyGraphContext.current.clearComputationGraph()
    for leaf in enc.latchLeaves { leaf.grad = nil }
    enc.gBatch!.updateDataLazily(g)
    let (ev2, _) = try evaluate(mode: .decode, latch: .leaves(enc.latchLeaves), rows: B, latent: zeros)
    let playhead = floor(seconds() * sampleRate + 0.5)
    _ = try (try output(ev2, rows: B) * enc.gBatch!.sampleRow(playhead)).sum().backward(frames: o.frames)
    let grads = enc.latchLeaves.map { l in l.grad?.getData() ?? [Float](repeating: 0, count: l.shape.reduce(1, *)) }
    // 3
    LazyGraphContext.current.clearComputationGraph()
    for (t, v) in zip(enc.latchGrads, grads) { t.updateDataLazily(v) }
    for l in enc.latchLeaves { l.grad = nil }
    let (_, c3) = try evaluate(mode: .decode, latch: .capture, rows: B, latent: encoded)
    guard c3.latched.count == enc.latchGrads.count, let z = zHat else {
      throw TrainProtocolError("latch count changed in pass 3")
    }
    var reg = Tensor([0.0]).sum()
    for (x, gt) in zip(c3.latched, enc.latchGrads) { reg = reg + (x * gt).sum() }
    if o.codeWeight > 0 {
      // only labelled rows have a code to match
      let n = rows.filter { enc.labeled[$0] }.count
      if n > 0 {
        enc.labelBatch!.updateDataLazily(rows.map { enc.labeled[$0] ? 1 : 0 })
        let d = (z - enc.zBatch!) * enc.labelBatch!
        reg = reg + (d * d).sum() * (o.codeWeight / Float(n * enc.D))
      }
    }
    if let p = priorLoss(z) { reg = reg + p }
    if let p = try lowEndLoss(c3.captured, rows: rows) { reg = reg + p }
    let ls = try staticToSignal(reg, frames: 1).backward(frames: 1).reduce(0, +)
    return (la, ls)
  }

  /// Row segments the low-end anchor reads (per-mode f, g, td, amp, dec; mode 0's g2, t2).
  static let lowSegments = ["f", "g", "td", "amp", "dec", "g2", "t2"]
  /// A mode is low end when its row frequency is below this (Hz).
  static let lowHz: Float = 250
  /// Times (s) the pitch curves are compared at: the drop and the body.
  static let bodyTimes: [Float] = [0.002, 0.005, 0.01, 0.02, 0.035, 0.06, 0.1, 0.15, 0.22, 0.3]

  /// --low-weight: W · mean over the batch's low modes of (curve st² averaged over bodyTimes
  /// + (level dB / 6)² + (decay st-scaled log ratio / 4)²).
  func lowEndLoss(_ captured: [String: Tensor], rows: [Int]) throws -> Tensor? {
    guard o.lowWeight > 0, let pr = paramRows.first, let L = layouts[pr.layout], let target = enc.lowBatch,
          let mask = enc.lowMaskBatch, let times = enc.lowTimes, let ones = enc.lowOnes else { return nil }
    guard let u = captured[pr.name] else { throw TrainProtocolError("param-row was not reached") }
    let B = rows.count, T = Self.bodyTimes.count, M = enc.lowMask[0].count, stride = T + 2
    let count = rows.reduce(Float(0)) { $0 + enc.lowMask[$1].reduce(0, +) }
    guard count > 0 else { return nil }
    target.updateDataLazily(rows.flatMap { enc.lowTargets[$0] })
    mask.updateDataLazily(rows.flatMap { enc.lowMask[$0] })
    func cols(_ t: Tensor, _ a: Int, _ w: Int = 1) -> Tensor { t.shrink([(0, B), (a, a + w)]) }
    // segment `name`'s slot i: its natural value, or its ln for a log segment, [B, 1]
    func value(_ name: String, _ i: Int) -> Tensor {
      let seg = L.segment(name)!
      let s = cols(u, seg.offset + i).sigmoid()
      guard seg.squash == .log else { return s * (seg.hi - seg.lo) + seg.lo }
      let a = Foundation.log(seg.lo)
      return s * (Foundation.log(seg.hi) - a) + a
    }
    let decay = { (lnT: Tensor) in ((lnT * -1).exp().matmul(times) * -1).exp() }  // e^(−t/T), [B, T]
    let floor = cols(target, M * stride)
    let st = 12 / Float(Foundation.log(2.0)), db = 20 / Float(Foundation.log(10.0))
    var total = Tensor([0.0]).sum()
    for i in 0..<M {
      var drop = value("g", i).matmul(ones) * decay(value("td", i))
      if i == 0 { drop = drop + value("g2", 0).matmul(ones) * decay(value("t2", 0)) }
      let dc = value("f", i).matmul(ones) + (drop + 1).log() - cols(target, i * stride, T)
      let dl = (value("amp", i).exp() + floor).log() - cols(target, i * stride + T)
      let dd = value("dec", i) - cols(target, i * stride + T + 1)
      let e = (dc * dc).sum(axis: 1).reshape([B, 1]) * (st * st / Float(T))
        + dl * dl * (db * db / 36) + dd * dd * (st * st / 16)
      total = total + (e * cols(mask, i)).sum()
    }
    return total * (o.lowWeight / count)
  }

  /// --amortized step loss (backward run): U error to the rows, path term, code L2.
  func amortizedLoss(rows: [Int], zHat: Tensor) throws -> Float {
    guard let pr = paramRows.first, let uBatch = enc.uBatch, let cols = enc.uColumns else {
      throw TrainProtocolError("--amortized needs a param-row")
    }
    uBatch.updateDataLazily(rows.flatMap { enc.uTargets[$0] })
    let S = enc.pathSel.map { $0.updated(self) }
    let M = S?.shape[0] ?? 0, B = self.B
    let (_, ctx) = try evaluate(mode: .capture, rows: B + M, latent: { _, _ in
      guard let S else { return zHat }
      return zHat.pad([(0, M), (0, 0)]) + S.matmul(zHat).pad([(B, 0), (0, 0)])
    })
    guard let uAll = ctx.captured[pr.name] else { throw TrainProtocolError("param-row was not reached") }
    let u = S == nil ? uAll : uAll.shrink([(0, B), (0, enc.P)])
    let du = u - uBatch
    let active = Float(max(1, columnWeights(layouts[pr.layout]!).filter { $0 > 0 }.count))
    var loss = (du * du * cols).sum() * (1 / (Float(B) * active))
    if let S {
      let mid = uAll.shrink([(B, B + M), (0, enc.P)])
      let d = mid - S.matmul(u)
      loss = loss + (d * d).sum() * (o.path / Float(M * enc.P))
    }
    if o.codeL2 > 0 { loss = loss + (zHat * zHat).sum() * (o.codeL2 / Float(B * enc.D)) }
    return try staticToSignal(loss, frames: 1).backward(frames: 1).reduce(0, +)
  }

  /// --amortized: mean squared U error of the held-out rows (decoded from their encoder codes).
  func valUError() throws -> Double {
    guard let pr = paramRows.first else { return .nan }
    LazyGraphContext.current.clearComputationGraph()
    let z = try encoderForward(rows: K, feats: enc.featVal!, aux: enc.auxVal)
    let (_, ctx) = try evaluate(mode: .capture, rows: K, latent: { _, _ in z })
    guard let u = ctx.captured[pr.name] else { throw TrainProtocolError("param-row was not reached") }
    let flat = try realizeStatic(u, count: K * enc.P)
    let target = enc.valIdx.flatMap { enc.uTargets[$0] }
    return zip(flat, target).reduce(0.0) { $0 + Double(($1.0 - $1.1) * ($1.0 - $1.1)) } / Double(flat.count)
  }

  /// The encoder's codes for the held-out rows.
  func valCodes() throws -> [[Float]] {
    LazyGraphContext.current.clearComputationGraph()
    let z = try encoderForward(rows: K, feats: enc.featVal!, aux: enc.auxVal)
    let flat = try realizeStatic(z, count: K * enc.D)
    return (0..<K).map { Array(flat[($0 * enc.D)..<(($0 + 1) * enc.D)]) }
  }

  /// Held-out MR of the encoder's codes, rendered through the voice.
  func valMR() throws -> [Double] {
    let renders = try renderCodes(try valCodes(), rows: enc.valIdx)
    var mr = [Double](repeating: .infinity, count: K)
    let windows = [64, 128, 256, 512, 1024, 2048, 4096]
    DispatchQueue.concurrentPerform(iterations: K) { i in
      guard renders[i].allSatisfy(\.isFinite) else { return }
      mr[i] = MetricLoss.scores(renders[i], targets[enc.valIdx[i]], windows: windows).1
    }
    return mr
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
      if let A = enc.aux.first?.count {
        enc.auxBatch!.updateDataLazily((start..<(start + n)).flatMap { enc.aux[$0] } + [Float](repeating: 0, count: (B - n) * A))
      }
      let z = try encoderForward(rows: B, feats: enc.featBatch!, aux: enc.auxBatch)
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
    if o.amortized {
      // the trained decoder, and the codes it is used with, where an instrument builder expects them
      try exportLearnables(learnables)
      try DatasetCodes.write(run.appendingPathComponent("export/codes.json"), names: names,
                             codes: all, table: enc.tableName, extra: ["val": enc.valIdx.map { names[$0] }])
      return try finishAmortized(all)
    }
    var summary: [String: Any] = ["stage": o.stage, "steps": o.steps, "val": K, "train": enc.trainIdx.count]
    let trainCodeErr = enc.trainIdx.filter { enc.labeled[$0] }.map { codeMSE(all[$0], row: $0) }
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

  /// --amortized report: MR / band of the encoder-code renders, held-out rows vs a sample of
  /// training rows (the gap is how far the space generalises).
  func finishAmortized(_ all: [[Float]]) throws -> [String: Any] {
    var summary: [String: Any] = ["stage": o.stage, "amortized": true, "steps": o.steps, "val": K,
                                  "train": enc.trainIdx.count]
    if K > 0 { summary["val_u_mse"] = try valUError() }
    guard !o.noRender else {
      try writeSummary(summary)
      return summary
    }
    var probe = enc.trainIdx
    rng.shuffle(&probe)
    let trainRows = Array(probe.prefix(min(30, probe.count))).sorted()
    let windows = [64, 128, 256, 512, 1024, 2048, 4096]
    let dir = run.appendingPathComponent("renders/val")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    var perRow: [String: Any] = [:]
    for (key, rows) in [("val", enc.valIdx), ("train", trainRows)] where !rows.isEmpty {
      let renders = try renderCodes(rows.map { all[$0] }, rows: rows)
      var sc = [(Double, Double)](repeating: (0, 0), count: rows.count)
      DispatchQueue.concurrentPerform(iterations: rows.count) { i in
        sc[i] = renders[i].allSatisfy(\.isFinite)
          ? MetricLoss.scores(renders[i], targets[rows[i]], windows: windows) : (.infinity, .infinity)
      }
      for (i, r) in rows.enumerated() where key == "val" {
        try FloatWav.write(dir.appendingPathComponent("\(names[r])-enc.wav"), renders[i], sampleRate: Int(sampleRate))
        perRow[names[r]] = ["mr": sc[i].1, "band": sc[i].0, "code": all[r].map(Double.init)]
      }
      summary["\(key)_mr_median"] = median(sc.map(\.1))
      summary["\(key)_band_median"] = median(sc.map(\.0))
    }
    summary["scores"] = perRow
    try writeSummary(summary)
    log(String(format: "[encode] amortized done: held-out MR median %.4f (band %.3f); train sample MR %.4f (band %.3f)",
               summary["val_mr_median"] as? Double ?? .nan, summary["val_band_median"] as? Double ?? .nan,
               summary["train_mr_median"] as? Double ?? .nan, summary["train_band_median"] as? Double ?? .nan))
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
