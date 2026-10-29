// TrainDataset.swift — `dgenlisp train-dataset`: train a row-batched patch (DatasetForms.swift)
// against a folder of target WAVs.
//
//   dgenlisp train-dataset --patch voice.lisp --dataset DIR --job-dir RUN --stage STAGE
//       [--rows rows.json] [--init-from RUN] [--resume] [--steps N] [--lr X] [--frames F] …
//
// Stages (SPEC §4):
//   refine    `param-row` is a free learnable [N,P] table initialised from --rows (natural
//             values per layout segment); audio loss (+ optional anchor to the initial rows).
//   pretrain  decoder only, parameter space: codes + learnable tensors regress U(codes) onto
//             --rows. No audio.
//   audio     decoder + codes against the targets, two-pass: (1) U = decoder(codes) realised;
//             (2) the voice renders from a leaf U, audio loss backprops to dL/dU; (3) the
//             decoder graph backprops Σ U·dL/dU (+ anchor) to codes and weights.
//
// Everything the run produces goes under --job-dir: ckpt/ (resumable: leaves + Adam moments +
// step), export/ (the patch's @file tensors, codes.json, rows.json, decoder_layout.json),
// renders/, targets/, loss.csv. Progress goes to stderr; stdout carries one JSON summary line.

import DGen
import DGenLazy
import DGenTrainProtocol
import Foundation

struct TrainDatasetOptions {
  var patch = ""
  var dataset = ""
  var jobDir = ""
  var stage = "refine"
  var rows: String?
  var initFrom: String?
  var resume = false
  var steps = 200
  var lr: Float = 1e-2
  var frames = 32768
  var anchor: Float = 0
  var codeL2: Float = 1e-3
  var windows = [64, 128, 256, 512, 1024, 2048, 4096]
  var linWeight: Float = 1
  var logWeight: Float = 1
  var checkpointEvery = 50
  var reportEvery = 10
  var evalEvery = 0
  var seed: UInt64 = 1
  var initFiles = false
  /// Comma-separated segment names whose columns do not move (refine: the leaf columns;
  /// audio: their gradient is zeroed before the decoder pass).
  var freeze: Set<String> = []
  var pitchHz: Double = 261.6255653
  var noRender = false

  static func parse(_ args: [String]) throws -> TrainDatasetOptions {
    var o = TrainDatasetOptions()
    var i = 0
    func value() throws -> String {
      i += 1
      guard i < args.count else { throw TrainProtocolError("missing value for \(args[i - 1])") }
      return args[i]
    }
    while i < args.count {
      let a = args[i]
      switch a {
      case "--patch": o.patch = try value()
      case "--dataset": o.dataset = try value()
      case "--job-dir": o.jobDir = try value()
      case "--stage": o.stage = try value()
      case "--rows": o.rows = try value()
      case "--init-from": o.initFrom = try value()
      case "--resume": o.resume = true
      case "--steps": o.steps = Int(try value()) ?? o.steps
      case "--lr": o.lr = Float(try value()) ?? o.lr
      case "--frames": o.frames = Int(try value()) ?? o.frames
      case "--anchor": o.anchor = Float(try value()) ?? o.anchor
      case "--code-l2": o.codeL2 = Float(try value()) ?? o.codeL2
      case "--windows": o.windows = try value().split(separator: ",").compactMap { Int($0) }
      case "--lin-weight": o.linWeight = Float(try value()) ?? o.linWeight
      case "--log-weight": o.logWeight = Float(try value()) ?? o.logWeight
      case "--checkpoint-every": o.checkpointEvery = Int(try value()) ?? o.checkpointEvery
      case "--report-every": o.reportEvery = Int(try value()) ?? o.reportEvery
      case "--eval-every": o.evalEvery = Int(try value()) ?? o.evalEvery
      case "--seed": o.seed = UInt64(try value()) ?? o.seed
      case "--init-files": o.initFiles = true
      case "--freeze": o.freeze = Set(try value().split(separator: ",").map(String.init))
      case "--pitch-hz": o.pitchHz = Double(try value()) ?? o.pitchHz
      case "--no-render": o.noRender = true
      default: throw TrainProtocolError("unknown train-dataset option \(a)")
      }
      i += 1
    }
    guard !o.patch.isEmpty, !o.dataset.isEmpty, !o.jobDir.isEmpty else {
      throw TrainProtocolError("train-dataset requires --patch, --dataset and --job-dir")
    }
    guard ["refine", "pretrain", "audio"].contains(o.stage) else {
      throw TrainProtocolError("--stage must be refine, pretrain or audio")
    }
    return o
  }
}

// MARK: - Targets

struct DatasetTarget {
  let name: String
  /// Prepared: mono, onset-aligned, peak 0.9, cropped/padded to the frame count.
  let samples: [Float]
}

enum DatasetTargets {
  /// Onset = first sample above −40 dB re peak, minus a 2 ms preroll (clamped at 0).
  static let onsetDB: Float = -40
  static let prerollSeconds: Float = 0.002
  static let peak: Float = 0.9

  static func prepare(_ x: [Float], sampleRate: Float, frames: Int) -> [Float] {
    let pk = x.map(abs).max() ?? 0
    guard pk > 0 else { return [Float](repeating: 0, count: frames) }
    let thr = pk * Foundation.pow(10, onsetDB / 20)
    let first = x.firstIndex { abs($0) > thr } ?? 0
    let start = max(0, first - Int((prerollSeconds * sampleRate).rounded()))
    var y = [Float](repeating: 0, count: frames)
    let scale = peak / pk
    for i in 0..<min(frames, x.count - start) { y[i] = x[start + i] * scale }
    return y
  }

  static func load(dir: String, frames: Int) throws -> (targets: [DatasetTarget], sampleRate: Float) {
    let url = URL(fileURLWithPath: dir)
    let files = try FileManager.default.contentsOfDirectory(at: url, includingPropertiesForKeys: nil)
      .filter { $0.pathExtension.lowercased() == "wav" }
      .sorted { $0.lastPathComponent < $1.lastPathComponent }
    guard !files.isEmpty else { throw TrainProtocolError("no .wav files in \(dir)") }
    var rate: Float? = nil
    var out: [DatasetTarget] = []
    for f in files {
      let (samples, sr) = try AudioFile.load(url: f, mono: true)
      if let rate, rate != sr {
        throw TrainProtocolError("mixed sample rates in dataset (\(rate) vs \(sr) in \(f.lastPathComponent))")
      }
      rate = sr
      out.append(DatasetTarget(
        name: f.deletingPathExtension().lastPathComponent,
        samples: prepare(samples, sampleRate: sr, frames: frames)))
    }
    return (out, rate!)
  }
}

// MARK: - Rows file

/// `{"layout": NAME, "rows": {"K51": {"seg": [natural values…], …}, …}}`
struct DatasetRows {
  let layout: String
  let rows: [String: [String: [Float]]]

  static func load(_ path: String) throws -> DatasetRows {
    let j = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: path)))
    guard let d = j as? [String: Any], let layout = d["layout"] as? String,
      let rows = d["rows"] as? [String: [String: Any]]
    else { throw TrainProtocolError("rows file \(path): expected {layout, rows}") }
    var parsed: [String: [String: [Float]]] = [:]
    for (name, segs) in rows {
      var r: [String: [Float]] = [:]
      for (seg, v) in segs {
        if let a = v as? [NSNumber] { r[seg] = a.map { $0.floatValue } }
        else if let x = v as? NSNumber { r[seg] = [x.floatValue] }
      }
      parsed[name] = r
    }
    return DatasetRows(layout: layout, rows: parsed)
  }

  /// [N*P] u-space matrix in target order; a missing row or segment is an error.
  func uMatrix(_ L: RowLayout, names: [String]) throws -> [Float] {
    var u = [Float](repeating: 0, count: names.count * L.width)
    for (r, n) in names.enumerated() {
      guard let row = rows[n] else { throw TrainProtocolError("rows file has no row for \(n)") }
      for s in L.segments {
        guard let v = row[s.name], v.count == s.width else {
          throw TrainProtocolError("rows \(n).\(s.name): expected \(s.width) values")
        }
        for k in 0..<s.width { u[r * L.width + s.offset + k] = s.toU(v[k]) }
      }
    }
    return u
  }

  static func write(_ path: URL, layout L: RowLayout, names: [String], u: [Float]) throws {
    var rows: [String: Any] = [:]
    for (r, n) in names.enumerated() {
      var segs: [String: Any] = [:]
      for s in L.segments {
        segs[s.name] = (0..<s.width).map { Double(s.fromU(u[r * L.width + s.offset + $0])) }
      }
      rows[n] = segs
    }
    let d: [String: Any] = ["layout": L.name, "rows": rows]
    try JSONSerialization.data(withJSONObject: d, options: [.prettyPrinted, .sortedKeys]).write(to: path)
  }
}

// MARK: - Leaves and optimiser state

/// Trainer-owned learnable tensors, by name ("w:<tensor>", "z:<table>", "row:<param-row>").
final class LeafStore {
  private(set) var order: [String] = []
  private(set) var leaves: [String: Tensor] = [:]
  /// Initial data for leaves not yet created (checkpoint / --init-from).
  var preset: [String: [Float]] = [:]

  func leaf(_ name: String, _ shape: [Int], _ make: () -> [Float]) -> Tensor {
    if let t = leaves[name] { return t }
    let n = shape.reduce(1, *)
    var data = preset[name] ?? make()
    if data.count != n {
      FileHandle.standardError.write(Data("[train-dataset] \(name): preset has \(data.count) values, expected \(n); reinitialised\n".utf8))
      data = make()
    }
    let t = Tensor.param(shape, data: data)
    leaves[name] = t
    order.append(name)
    return t
  }

  func data(_ name: String) -> [Float] { leaves[name]?.getData() ?? preset[name] ?? [] }
}

struct AdamMoments {
  var m: [String: [Float]] = [:]
  var v: [String: [Float]] = [:]
  var t = 0

  mutating func step(_ store: LeafStore, names: [String], grads: [String: [Float]], lr: Float,
                     clip: Float = 0) {
    t += 1
    let b1: Float = 0.9, b2: Float = 0.999, eps: Float = 1e-8
    let c1 = 1 - Foundation.pow(b1, Float(t)), c2 = 1 - Foundation.pow(b2, Float(t))
    for name in names {
      guard let g0 = grads[name], let x = store.leaves[name]?.getData() else { continue }
      var g = g0
      if clip > 0 {
        let norm = g.reduce(Float(0)) { $0 + $1 * $1 }.squareRoot()
        if norm > clip { g = g.map { $0 * clip / norm } }
      }
      var mm = m[name] ?? [Float](repeating: 0, count: g.count)
      var vv = v[name] ?? [Float](repeating: 0, count: g.count)
      var y = x
      for i in 0..<g.count {
        let gi = g[i].isFinite ? g[i] : 0
        mm[i] = b1 * mm[i] + (1 - b1) * gi
        vv[i] = b2 * vv[i] + (1 - b2) * gi * gi
        y[i] = x[i] - lr * (mm[i] / c1) / ((vv[i] / c2).squareRoot() + eps)
      }
      m[name] = mm
      v[name] = vv
      store.leaves[name]!.updateDataLazily(y)
    }
  }
}

// MARK: - Checkpoints

enum DatasetCheckpoint {
  static func write(_ url: URL, stage: String, step: Int, store: LeafStore, adam: AdamMoments,
                    extra: [String: Any] = [:]) throws {
    var leaves: [String: Any] = [:]
    for name in store.order {
      let t = store.leaves[name]!
      leaves[name] = ["shape": t.shape, "data": (t.getData() ?? []).map(Double.init)]
    }
    var d: [String: Any] = [
      "version": 1, "stage": stage, "step": step, "leaves": leaves,
      "adam": ["t": adam.t, "m": adam.m.mapValues { $0.map(Double.init) },
               "v": adam.v.mapValues { $0.map(Double.init) }],
    ]
    for (k, v) in extra { d[k] = v }
    let tmp = url.appendingPathExtension("tmp")
    try JSONSerialization.data(withJSONObject: d, options: [.sortedKeys]).write(to: tmp)
    _ = try? FileManager.default.removeItem(at: url)
    try FileManager.default.moveItem(at: tmp, to: url)
  }

  struct Loaded {
    var stage: String
    var step: Int
    var leaves: [String: [Float]]
    var adam: AdamMoments
  }

  static func load(_ url: URL) throws -> Loaded {
    let j = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
    var leaves: [String: [Float]] = [:]
    for (k, v) in (j["leaves"] as? [String: [String: Any]] ?? [:]) {
      leaves[k] = (v["data"] as? [NSNumber] ?? []).map { $0.floatValue }
    }
    var adam = AdamMoments()
    if let a = j["adam"] as? [String: Any] {
      adam.t = (a["t"] as? NSNumber)?.intValue ?? 0
      for (k, v) in (a["m"] as? [String: [NSNumber]] ?? [:]) { adam.m[k] = v.map { $0.floatValue } }
      for (k, v) in (a["v"] as? [String: [NSNumber]] ?? [:]) { adam.v[k] = v.map { $0.floatValue } }
    }
    return Loaded(stage: j["stage"] as? String ?? "", step: (j["step"] as? NSNumber)?.intValue ?? 0,
                  leaves: leaves, adam: adam)
  }

  /// Newest ckpt/step-*.json in a run directory.
  static func latest(in run: URL) -> URL? {
    let dir = run.appendingPathComponent("ckpt")
    let files = (try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? []
    return files.filter { $0.lastPathComponent.hasPrefix("step-") && $0.pathExtension == "json" }
      .sorted { $0.lastPathComponent < $1.lastPathComponent }.last
  }
}

// MARK: - Trainer

final class DatasetTrainer {
  let o: TrainDatasetOptions
  let run: URL
  let names: [String]
  let targets: [[Float]]
  let sampleRate: Float
  let nodes: [ASTNode]
  let patchDir: URL
  let store = LeafStore()
  var adam = AdamMoments()
  var startStep = 0
  var rng: SplitMix64

  // discovered on the plan evaluation
  var layouts: [String: RowLayout] = [:]
  var learnables: [LearnableTensorInfo] = []
  var tables: [LatentTableInfo] = []
  var paramRows: [ParamRowInfo] = []
  var rowsU: [String: [Float]] = [:]  // param-row name → [N*P] u-space targets / init

  var N: Int { names.count }

  init(_ o: TrainDatasetOptions) throws {
    self.o = o
    run = URL(fileURLWithPath: o.jobDir)
    rng = SplitMix64(seed: o.seed)
    for sub in ["ckpt", "export", "renders", "targets"] {
      try FileManager.default.createDirectory(
        at: run.appendingPathComponent(sub), withIntermediateDirectories: true)
    }
    let (t, sr) = try DatasetTargets.load(dir: o.dataset, frames: o.frames)
    names = t.map(\.name)
    targets = t.map(\.samples)
    sampleRate = sr
    let patchURL = URL(fileURLWithPath: o.patch)
    patchDir = patchURL.deletingLastPathComponent()
    let source = try String(contentsOf: patchURL, encoding: .utf8)
    // Training lowering: modulation stripped, inlets driven by the one-hit excitation.
    let stripped = ExcitationLowering.stripModulation(nodes: try parseSource(source))
    let lowered = try lowerModulation(in: stripped)
    let driven = ExcitationLowering.drive(nodes: lowered, pitchHz: o.pitchHz, gateFrames: o.frames)
    if !driven.undriven.isEmpty {
      throw TrainProtocolError("inlets outside the excitation convention: \(driven.undriven.joined(separator: ", "))")
    }
    nodes = driven.nodes
    for (n, x) in zip(names, targets) {
      try MiniWav.write(url: run.appendingPathComponent("targets/\(n).wav"), samples: x,
                        sampleRate: Int(sampleRate))
    }
  }

  func log(_ s: String) { FileHandle.standardError.write(Data((s + "\n").utf8)) }

  func configure() {
    DGenConfig.backend = .metal
    DGenConfig.sampleRate = sampleRate
    DGenConfig.maxFrameCount = o.frames
    DGenConfig.defaultFrameCount = o.frames
    DGenSpectralConfig.logMagnitudeEpsilon = 1e-3
    DGenGradientConfig.detachPhasorFrequency = false
    LazyGraphContext.preserveCompilationCaches = true
    LazyGraphContext.collectExecutionTiming = timing
  }

  let timing = ProcessInfo.processInfo.environment["DGENLISP_TRAIN_TIMING"] == "1"

  func logTiming(_ what: String) {
    guard timing else { return }
    let t = LazyGraphContext.current.lastExecutionTiming
    log(String(format: "[timing] %@ compile %.0f ms runtime %.0f ms exec %.0f ms (cache full %@ runtime %@)",
               what, t.compilationMS, t.runtimeCreationMS, t.executionMS,
               t.fullCompilationCacheHit ? "hit" : "miss", t.runtimeCacheHit ? "hit" : "miss"))
  }

  // MARK: initialisation

  func gaussian(_ n: Int, _ sd: Float) -> [Float] { (0..<n).map { _ in rng.gaussian() * sd } }

  func initData(_ info: LearnableTensorInfo, fileData: [Float]?) -> [Float] {
    let n = info.shape.reduce(1, *)
    if let fileData, o.initFiles || info.initKind == "file" { return fileData }
    switch info.initKind {
    case "zeros": return [Float](repeating: 0, count: n)
    case "small": return gaussian(n, 0.01)
    default:
      let fanIn = info.shape.count > 1 ? info.shape[0] : 1
      let fanOut = info.shape.last ?? 1
      let a = (6 / Float(fanIn + fanOut)).squareRoot()
      return (0..<n).map { _ in (rng.uniform() * 2 - 1) * a }
    }
  }

  /// Codes initialised from the principal components of the row targets (whitened), or
  /// small Gaussians when there are none.
  func initCodes(_ dim: Int) -> [Float] {
    guard let u = rowsU.values.first, !u.isEmpty else { return gaussian(N * dim, 0.5) }
    let P = u.count / N
    var mean = [Float](repeating: 0, count: P)
    for r in 0..<N { for c in 0..<P { mean[c] += u[r * P + c] / Float(N) } }
    var X = [Float](repeating: 0, count: N * P)
    for r in 0..<N { for c in 0..<P { X[r * P + c] = u[r * P + c] - mean[c] } }
    // Gram matrix [N,N] and its top eigenvectors by power iteration with deflation.
    var G = [Float](repeating: 0, count: N * N)
    for a in 0..<N { for b in a..<N {
      var s: Float = 0
      for c in 0..<P { s += X[a * P + c] * X[b * P + c] }
      G[a * N + b] = s; G[b * N + a] = s
    } }
    var codes = [Float](repeating: 0, count: N * dim)
    for d in 0..<dim {
      var v = gaussian(N, 1)
      var lambda: Float = 0
      for _ in 0..<200 {
        var w = [Float](repeating: 0, count: N)
        for a in 0..<N { for b in 0..<N { w[a] += G[a * N + b] * v[b] } }
        let norm = w.reduce(Float(0)) { $0 + $1 * $1 }.squareRoot()
        guard norm > 0 else { break }
        lambda = norm
        v = w.map { $0 / norm }
      }
      for a in 0..<N { for b in 0..<N { G[a * N + b] -= lambda * v[a] * v[b] } }
      // whitened: unit std per dim
      let mu = v.reduce(0, +) / Float(N)
      let sd = (v.map { ($0 - mu) * ($0 - mu) }.reduce(0, +) / Float(N)).squareRoot()
      for a in 0..<N { codes[a * dim + d] = sd > 0 ? (v[a] - mu) / sd : 0 }
    }
    return codes
  }

  // MARK: evaluation

  /// Evaluates the training patch with a dataset context; returns the evaluator.
  @discardableResult
  func evaluate(mode: DatasetContext.RowMode, latch: DatasetContext.LatchMode = .identity) throws
    -> (LispEvaluator, DatasetContext)
  {
    let ctx = DatasetContext(
      rows: N, frames: o.frames, sampleRate: sampleRate, rowMode: mode,
      learnable: { [self] name, shape, fileData, initKind in
        let info = LearnableTensorInfo(name: name, shape: shape, file: nil, initKind: initKind)
        return store.leaf("w:" + name, shape) { initData(info, fileData: fileData) }
      },
      latentRows: { [self] name, dim in
        store.leaf("z:" + name, [N, dim]) { initCodes(dim) }
      })
    ctx.latchMode = latch
    let ev = LispEvaluator(sourceDirectory: patchDir)
    ev.dataset = ctx
    do {
      try ev.evaluate(nodes: nodes)
    } catch is DatasetCaptureDone {
    } catch let e as LispError {
      throw TrainProtocolError("patch evaluation failed: \(e.message)")
    }
    return (ev, ctx)
  }

  func output(_ ev: LispEvaluator) throws -> SignalTensor {
    guard let out = (ev.tensorOutputs.first { $0.channel == 0 } ?? ev.tensorOutputs.first)?.signal
    else { throw TrainProtocolError("patch has no row-out output; end the voice with (out (row-out y) 1)") }
    guard out.shape == [N] else { throw TrainProtocolError("row-out gave \(out.shape), expected [\(N)]") }
    return out
  }

  lazy var targetTable: Tensor = {
    var data = [Float](repeating: 0, count: o.frames * N)
    for (r, x) in targets.enumerated() { for f in 0..<o.frames { data[f * N + r] = x[f] } }
    return Tensor(data).reshape([o.frames, N])
  }()

  func seconds() -> Signal {
    let n = Float(o.frames)
    return Signal.phasor(Tensor([sampleRate / (2 * n)])).sum() * (2 * n / sampleRate)
  }

  func audioLoss(_ y: SignalTensor) -> Signal {
    let playhead = floor(seconds() * sampleRate + 0.5)
    let t = targetTable.sampleRow(playhead)
    var loss = Signal.constant(0)
    let ws = o.windows.filter { $0 <= o.frames }
    let scale = 1 / Float(max(ws.count, 1))
    for w in ws {
      let hop = max(1, w / 4)
      if o.linWeight > 0 {
        loss = loss + spectralLossFFT(y, t, windowSize: w, lossMode: .l1, hop: hop, normalize: true)
          * (o.linWeight * scale)
      }
      if o.logWeight > 0 {
        loss = loss + spectralLossFFT(y, t, windowSize: w, useLogMagnitude: true, lossMode: .l1,
                                      hop: hop, normalize: true) * (o.logWeight * scale)
      }
    }
    return loss
  }

  func staticToSignal(_ t: Tensor, frames: Int) -> Signal {
    (Tensor([0.0]) + t).peek(Signal.constant(0)) * (1 / Float(frames))
  }

  /// Mean squared u error to the row targets (frozen columns excluded).
  func anchorLoss(_ u: Tensor, row: String) -> Tensor? {
    guard let target = rowsU[row], let L = layoutOf(row) else { return nil }
    let w = columnWeights(L)
    let d = u - Tensor(target).reshape([N, L.width])
    return (d * d * Tensor(w + [Float](repeating: 0, count: 0)).reshape([1, L.width])).sum()
      * (1 / Float(N * max(1, w.filter { $0 > 0 }.count)))
  }

  func layoutOf(_ row: String) -> RowLayout? {
    paramRows.first { $0.name == row }.flatMap { layouts[$0.layout] }
  }

  func columnWeights(_ L: RowLayout) -> [Float] {
    var w = [Float](repeating: 1, count: L.width)
    for s in L.segments where o.freeze.contains(s.name) {
      for k in s.offset..<(s.offset + s.width) { w[k] = 0 }
    }
    return w
  }

  func grads(_ names: [String]) -> [String: [Float]] {
    var g: [String: [Float]] = [:]
    for n in names { if let d = store.leaves[n]?.grad?.getData() { g[n] = d } }
    return g
  }

  func zeroGrads() { for t in store.leaves.values { t.grad = nil } }

  // MARK: plan

  func plan() throws {
    configure()
    LazyGraphContext.reset()
    if let initFrom = o.initFrom {
      let url = URL(fileURLWithPath: initFrom)
      let ck = url.pathExtension == "json" ? url : DatasetCheckpoint.latest(in: url)
      guard let ck else { throw TrainProtocolError("--init-from \(initFrom): no checkpoint") }
      store.preset = try DatasetCheckpoint.load(ck).leaves
      log("[train-dataset] init from \(ck.path) (\(store.preset.count) leaves)")
    }
    if o.resume, let ck = DatasetCheckpoint.latest(in: run) {
      let l = try DatasetCheckpoint.load(ck)
      guard l.stage == o.stage else {
        throw TrainProtocolError("--resume: checkpoint stage \(l.stage) != \(o.stage)")
      }
      store.preset.merge(l.leaves) { _, new in new }
      adam = l.adam
      startStep = l.step
      log("[train-dataset] resume \(ck.lastPathComponent) at step \(startStep)")
    }
    // Discovery pass (decode mode, no leaves created for codes yet — creation is lazy and
    // initCodes needs the rows, which need the layout: read layouts with a compile evaluation).
    let probe = LispEvaluator(sourceDirectory: patchDir)
    probe.dataset = nil
    do { try probe.evaluate(nodes: Self.declarationsOnly(nodes)) } catch {
      throw TrainProtocolError("reading row layouts failed: \(error)")
    }
    layouts = probe.rowLayouts
    if let rowsPath = o.rows {
      let rows = try DatasetRows.load(rowsPath)
      guard let L = layouts[rows.layout] else {
        throw TrainProtocolError("rows layout \(rows.layout) is not declared in the patch")
      }
      rowsU["row"] = try rows.uMatrix(L, names: names)
    }
    let mode: DatasetContext.RowMode = .decode
    let (ev, _) = try evaluate(mode: mode)
    learnables = ev.learnableTensors
    tables = ev.latentTables
    paramRows = ev.paramRows
    // rows keyed "row" above → the real param-row name
    if let u = rowsU.removeValue(forKey: "row") {
      guard let pr = paramRows.first else { throw TrainProtocolError("--rows given but the patch has no param-row") }
      rowsU[pr.name] = u
    }
    // Persistent data tensors must exist before the first step: one created mid-step registers
    // in a different order on later steps, shifting memory offsets and missing the kernel cache.
    _ = targetTable
    log("[train-dataset] \(N) targets @ \(Int(sampleRate)) Hz, \(o.frames) frames; layouts \(layouts.keys.sorted()); learnable \(learnables.map { "\($0.name)\($0.shape)" }); tables \(tables.map { "\($0.name)[\($0.dim)]" }); rows \(paramRows.map(\.name))")
    LazyGraphContext.current.clearComputationGraph()
  }

  /// Only `row-layout` forms and plain numeric defs: layouts can be read without building the
  /// voice.
  static func declarationsOnly(_ nodes: [ASTNode]) -> [ASTNode] {
    nodes.filter {
      if case .list(let e) = $0, case .atom("row-layout")? = e.first { return true }
      return false
    }
  }

  // MARK: stages

  func train() throws -> [String: Any] {
    try plan()
    switch o.stage {
    case "refine": return try refine()
    case "pretrain": return try pretrain()
    default: return try audio()
    }
  }

  func checkpoint(_ step: Int) throws {
    let url = run.appendingPathComponent(String(format: "ckpt/step-%06d.json", step))
    try DatasetCheckpoint.write(url, stage: o.stage, step: step, store: store, adam: adam,
                                extra: ["names": names, "patch": o.patch])
  }

  func lrAt(_ step: Int) -> Float {
    let p = Float(step) / Float(max(1, o.steps - 1))
    return o.lr * max(0.05, 0.5 * (1 + Foundation.cos(Float.pi * p)))
  }

  var csv = "step,loss,static,sec\n"

  func loop(_ body: (Int) throws -> (Float, Float), trainable: () -> [String]) throws {
    if startStep >= o.steps { log("[train-dataset] already at step \(startStep)"); return }
    let t0 = Date()
    for step in startStep..<o.steps {
      let ts = Date()
      zeroGrads()
      let (la, ls) = try body(step)
      guard la.isFinite, ls.isFinite else {
        throw TrainProtocolError("loss diverged at step \(step) (audio \(la), static \(ls))")
      }
      let names = trainable()
      adam.step(store, names: names, grads: grads(names), lr: lrAt(step))
      let dt = Date().timeIntervalSince(ts)
      csv += "\(step),\(la),\(ls),\(dt)\n"
      if step % o.reportEvery == 0 || step == o.steps - 1 {
        log(String(format: "[%@] step %d loss %.5f static %.5f (%.2f s/step, %.0f s)", o.stage,
                   step, la, ls, dt, Date().timeIntervalSince(t0)))
      }
      if (step + 1) % o.checkpointEvery == 0 || step == o.steps - 1 { try checkpoint(step + 1) }
    }
    try? csv.write(to: run.appendingPathComponent("loss-\(o.stage).csv"), atomically: true, encoding: .utf8)
  }

  func refine() throws -> [String: Any] {
    guard let pr = paramRows.first, let L = layouts[pr.layout] else {
      throw TrainProtocolError("refine needs a param-row")
    }
    guard let u0 = rowsU[pr.name] else { throw TrainProtocolError("refine needs --rows") }
    let leafName = "row:" + pr.name
    let leaf = store.leaf(leafName, [N, L.width]) { u0 }
    let w = columnWeights(L)
    try loop({ step in
      let r = try latchStep(step, rowMode: .leaf(leaf), row: pr.name, anchorOn: leaf)
      // frozen columns: zero their gradient
      if w.contains(0), let g = leaf.grad?.getData() {
        leaf.grad = Tensor(zip(g.indices, g).map { i, x in w[i % L.width] > 0 ? x : 0 }).reshape([N, L.width])
      }
      return r
    }, trainable: { [leafName] })
    try export()
    return try finish()
  }

  func pretrain() throws -> [String: Any] {
    guard let pr = paramRows.first, rowsU[pr.name] != nil else {
      throw TrainProtocolError("pretrain needs a param-row and --rows")
    }
    let staticFrames = 64
    try loop({ step in
      LazyGraphContext.current.clearComputationGraph()
      let (_, ctx) = try evaluate(mode: .capture)
      guard let u = ctx.captured[pr.name] else { throw TrainProtocolError("param-row was not reached") }
      var reg = anchorLoss(u, row: pr.name)!
      for t in tables where o.codeL2 > 0 {
        let z = store.leaves["z:" + t.name]!
        reg = reg + (z * z).sum() * (o.codeL2 / Float(N * t.dim))
      }
      let v = try staticToSignal(reg, frames: staticFrames).backward(frames: staticFrames).reduce(0, +)
      return (0, v)
    }, trainable: { store.order.filter { !$0.hasPrefix("row:") } })
    try export()
    return try finish()
  }

  func audio() throws -> [String: Any] {
    guard let pr = paramRows.first else { throw TrainProtocolError("audio needs a param-row") }
    try loop({ step in
      try latchStep(step, rowMode: .decode, row: pr.name, anchorOn: nil)
    }, trainable: { store.order.filter { !$0.hasPrefix("row:") } })
    try export()
    return try finish()
  }

  // MARK: latch-boundary step

  /// Persistent voice-side leaves (one per static latch) and their gradient tensors.
  var latchLeaves: [Tensor] = []
  var latchGrads: [Tensor] = []

  /// One gradient of (audio loss + static terms) w.r.t. the static side (row leaf, or decoder
  /// weights and codes), cut at the latches:
  ///   1. static pass: realise every latched value;
  ///   2. voice pass: render from leaves holding those values, audio loss backward → dL/dleaf
  ///      (already summed over frames);
  ///   3. static pass: backward of Σ latched·dL/dleaf (+ anchor, code prior).
  func latchStep(_ step: Int, rowMode: DatasetContext.RowMode, row: String, anchorOn: Tensor?)
    throws -> (Float, Float)
  {
    // 1
    LazyGraphContext.current.clearComputationGraph()
    if ProcessInfo.processInfo.environment["DGENLISP_LIVE_TENSORS"] != nil {
      let live = LazyGraphContext.current.liveTensorSummary
      log("[live] \(live.count) tensors (\(live.filter(\.data).count) with data), signals: \(LazyGraphContext.current.liveSignalCount)")
    }
    let (_, c1) = try evaluate(mode: rowMode, latch: .capture)
    let values = try realizeAll(c1.latched)
    if latchLeaves.count != values.count {
      guard latchLeaves.isEmpty else { throw TrainProtocolError("latch count changed between steps") }
      latchLeaves = zip(c1.latched, values).map { Tensor.param($0.shape, data: $1) }
      latchGrads = c1.latched.map { Tensor([Float](repeating: 0, count: $0.shape.reduce(1, *))).reshape($0.shape) }
      log("[train-dataset] \(values.count) latch leaves, \(values.reduce(0) { $0 + $1.count }) values")
    }
    // 2
    LazyGraphContext.current.clearComputationGraph()
    for (leaf, v) in zip(latchLeaves, values) { leaf.updateDataLazily(v); leaf.grad = nil }
    let (ev, _) = try evaluate(mode: rowMode, latch: .leaves(latchLeaves))
    if let dump = ProcessInfo.processInfo.environment["DGENLISP_KERNEL_DUMP"] {
      DGenConfig.kernelOutputPath = "\(dump)/step\(step)"
    }
    let te = Date()
    let la = try audioLoss(try output(ev)).backward(frames: o.frames).reduce(0, +)
    DGenConfig.kernelOutputPath = nil
    logTiming(String(format: "voice backward %.0f ms wall", Date().timeIntervalSince(te) * 1000))
    let grads = latchLeaves.map { l in l.grad?.getData() ?? [Float](repeating: 0, count: l.shape.reduce(1, *)) }
    // 3
    LazyGraphContext.current.clearComputationGraph()
    for (g, v) in zip(latchGrads, grads) { g.updateDataLazily(v) }
    for l in latchLeaves { l.grad = nil }
    let (_, c3) = try evaluate(mode: rowMode, latch: .capture)
    guard c3.latched.count == latchGrads.count else { throw TrainProtocolError("latch count changed in pass 3") }
    var reg = Tensor([0.0]).sum()
    for (x, g) in zip(c3.latched, latchGrads) { reg = reg + (x * g).sum() }
    if o.anchor > 0 {
      let u = anchorOn ?? c3.captured[row]
      if let u, let a = anchorLoss(u, row: row) { reg = reg + a * o.anchor }
    }
    for t in tables where o.codeL2 > 0 && anchorOn == nil {
      let z = store.leaves["z:" + t.name]!
      reg = reg + (z * z).sum() * (o.codeL2 / Float(N * t.dim))
    }
    let staticFrames = 64
    let ls = try staticToSignal(reg, frames: staticFrames).backward(frames: staticFrames).reduce(0, +)
    return (la, ls)
  }

  /// Realises several static tensors in one compile (flattened, padded and summed into one).
  func realizeAll(_ ts: [Tensor]) throws -> [[Float]] {
    let counts = ts.map { $0.shape.reduce(1, *) }
    let total = counts.reduce(0, +)
    guard total > 0 else { return ts.map { _ in [] } }
    var packed: Tensor? = nil
    var off = 0
    for (t, n) in zip(ts, counts) {
      let p = t.reshape([n]).pad([(off, total - off - n)])
      packed = packed.map { $0 + p } ?? p
      off += n
    }
    let flat = try realizeStatic(packed!, count: total)
    var out: [[Float]] = []
    off = 0
    for n in counts { out.append(Array(flat[off..<(off + n)])); off += n }
    return out
  }

  /// Realises a static tensor's values (one frame).
  func realizeStatic(_ t: Tensor, count: Int) throws -> [Float] {
    let flat = t.reshape([count])
    let st = SignalTensor.lift(flat)
    let v = try st.realize(frames: 1)
    return Array(v.prefix(count))
  }

  // MARK: export, renders

  func currentRows() throws -> (RowLayout, String, [Float])? {
    guard let pr = paramRows.first, let L = layouts[pr.layout] else { return nil }
    if let leaf = store.leaves["row:" + pr.name] { return (L, pr.name, leaf.getData() ?? []) }
    LazyGraphContext.current.clearComputationGraph()
    let (_, ctx) = try evaluate(mode: .capture)
    guard let u = ctx.captured[pr.name] else { return nil }
    return (L, pr.name, try realizeStatic(u, count: N * L.width))
  }

  func export() throws {
    let ex = run.appendingPathComponent("export")
    func writeTensor(_ shape: [Int], _ data: [Float], to url: URL) throws {
      let d: [String: Any] = ["shape": shape, "data": data.map(Double.init)]
      try JSONSerialization.data(withJSONObject: d).write(to: url)
    }
    for info in learnables {
      guard let t = store.leaves["w:" + info.name] else { continue }
      let file = info.file.map { URL(fileURLWithPath: $0).lastPathComponent } ?? "\(info.name).json"
      try writeTensor(info.shape, t.getData() ?? [], to: ex.appendingPathComponent(file))
    }
    for t in tables {
      guard let z = store.leaves["z:" + t.name]?.getData() else { continue }
      let codes = (0..<N).map { r in (0..<t.dim).map { Double(z[r * t.dim + $0]) } }
      let d: [String: Any] = ["names": names, "codes": codes, "dim": t.dim]
      try JSONSerialization.data(withJSONObject: d, options: [.prettyPrinted]).write(
        to: ex.appendingPathComponent("\(t.name).json"))
    }
    for (name, L) in layouts {
      let segs = L.segments.map {
        ["name": $0.name, "offset": $0.offset, "width": $0.width, "squash": $0.squash.rawValue,
         "lo": Double($0.lo), "hi": Double($0.hi)] as [String: Any]
      }
      let d: [String: Any] = ["layout": name, "P": L.width, "segments": segs]
      try JSONSerialization.data(withJSONObject: d, options: [.prettyPrinted]).write(
        to: ex.appendingPathComponent("layout-\(name).json"))
    }
    if let (L, _, u) = try currentRows() {
      try DatasetRows.write(ex.appendingPathComponent("rows.json"), layout: L, names: names, u: u)
    }
  }

  /// Forward render of every row from the current state; per-row MR-STFT (the CPU scorer).
  func finish() throws -> [String: Any] {
    guard !o.noRender, let (L, _, u) = try currentRows() else { return ["stage": o.stage] }
    LazyGraphContext.current.clearComputationGraph()
    let leaf = Tensor(u).reshape([N, L.width])
    let (ev, _) = try evaluate(mode: .leaf(leaf))
    let flat = try output(ev).realize(frames: o.frames)
    var scores: [String: Double] = [:]
    for (r, n) in names.enumerated() {
      var y = [Float](repeating: 0, count: o.frames)
      for f in 0..<o.frames { y[f] = flat[f * N + r] }
      try MiniWav.write(url: run.appendingPathComponent("renders/\(n).wav"), samples: y,
                        sampleRate: Int(sampleRate))
      let scorer = try TrainSpectralScorer(target: targets[r], windows: o.windows.filter { $0 <= o.frames },
                                           epsilon: 1e-3)
      scores[n] = Double(y.allSatisfy(\.isFinite) ? scorer.score(y) : .infinity)
    }
    let sorted = scores.values.sorted()
    let summary: [String: Any] = [
      "stage": o.stage, "steps": o.steps, "mr_median": sorted[sorted.count / 2],
      "mr_mean": sorted.reduce(0, +) / Double(sorted.count), "scores": scores,
    ]
    try JSONSerialization.data(withJSONObject: summary, options: [.prettyPrinted, .sortedKeys])
      .write(to: run.appendingPathComponent("summary-\(o.stage).json"))
    log(String(format: "[train-dataset] %@ done: MR median %.4f mean %.4f", o.stage,
               summary["mr_median"] as! Double, summary["mr_mean"] as! Double))
    return summary
  }
}

enum TrainDatasetCommand {
  static func run(arguments: [String]) -> Never {
    do {
      let o = try TrainDatasetOptions.parse(arguments)
      let trainer = try DatasetTrainer(o)
      let summary = try trainer.train()
      let line = try JSONSerialization.data(withJSONObject: summary.filter { $0.key != "scores" })
      print(String(data: line, encoding: .utf8)!)
      exit(0)
    } catch {
      fputs("train-dataset error: \(error)\n", stderr)
      exit(1)
    }
  }
}

/// Deterministic initialisation RNG (SplitMix64).
struct SplitMix64 {
  var state: UInt64
  init(seed: UInt64) { state = seed }
  mutating func next() -> UInt64 {
    state &+= 0x9E37_79B9_7F4A_7C15
    var z = state
    z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
    z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
    return z ^ (z >> 31)
  }
  mutating func uniform() -> Float { Float(next() >> 40) / Float(1 << 24) }
  mutating func gaussian() -> Float {
    let u1 = max(uniform(), 1e-7), u2 = uniform()
    return (-2 * Foundation.log(u1)).squareRoot() * Foundation.cos(2 * Float.pi * u2)
  }
}
