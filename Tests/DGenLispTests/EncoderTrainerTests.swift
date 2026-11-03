import DGen
import DGenLazy
import DGenTrainProtocol
import Foundation
import XCTest

@testable import DGenLisp

/// The latent-encoder path (docs/LATENT_ENCODER_SPEC.md phase 1): `row-data` / `latent-out` /
/// `windows1d` forms, host features, and `train-dataset --stage encode` end to end on a tiny
/// synthetic instrument whose codes are recoverable from its audio.
final class EncoderTrainerTests: XCTestCase {
  private var tempDir: URL!
  private let sr = 8000

  override func setUpWithError() throws {
    try super.setUpWithError()
    LazyGraphContext.reset()
    tempDir = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
      .appendingPathComponent("dgenlisp-encoder-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
  }

  override func tearDownWithError() throws {
    // the trainer configures globals for its run; put back what other suites assume
    DGenConfig.maxFrameCount = 4096
    LazyGraphContext.preserveCompilationCaches = false
    LazyGraph.runtimeCacheLimit = 8
    DGenSpectralConfig.logMagnitudeEpsilon = 1e-8
    LazyGraphContext.reset()
    if let tempDir { try? FileManager.default.removeItem(at: tempDir) }
    try super.tearDownWithError()
  }

  /// u = z @ w + b; f (pitch) and d (decay) carry the code; a is normalised away by the targets'
  /// peak normalisation.
  private let voice = """
    (def trigger (click))
    (row-layout v (f 1 log 60 1200) (d 1 log 0.02 0.4) (a 1 lin 0 2))
    (def z (event-hold (latent-table @name codes @dim 2 (tensor @shape [1 2] @data [0 0])) trigger))
    (def w (tensor @shape [2 3] @file "w.json" @learnable true))
    (def b (tensor @shape [1 3] @file "b.json" @learnable true))
    (def u (param-row (+ (matmul z w) b) @layout v))
    (def t (hit-time trigger))
    (def f (latch (seg u v f) trigger))
    (def d (latch (seg u v d) trigger))
    (def a (latch (seg u v a) trigger))
    (def y (sum-axis (* a (exp (/ (* -1 t) d)) (sin (* twopi f t))) @axis 1))
    (out (row-out y) 1)
    """

  /// feats [B, 24·28] → conv (k 5, s 2) over time → mean over time → MLP → [B, 2]
  private let encoder = """
    (def x (reshape (row-data @name feats @shape [24 28] (zeros [24 28])) @shape [-1 24 28]))
    (def win (windows1d x @size 5 @stride 2))
    (def cols (reshape (transpose win @axes [0 2 1 3]) @shape [-1 120]))
    (def h1 (relu (matmul cols (tensor @shape [120 16] @learnable true @name enc_c1))))
    (def g (mean-axis (reshape h1 @shape [-1 12 16]) @axis 1))
    (def h2 (tanh (+ (matmul g (tensor @shape [16 16] @learnable true @name enc_w1))
                     (tensor @shape [1 16] @learnable true @name enc_b1 @init zeros))))
    (latent-out (+ (matmul h2 (tensor @shape [16 2] @learnable true @name enc_w2))
                   (tensor @shape [1 2] @learnable true @name enc_b2 @init zeros)))
    """

  private let w: [Float] = [1.0, -0.4, 0.3, 0.2, 1.0, -0.2]
  private let b: [Float] = [0.0, 0.0, 0.5]

  func testRowDataAndLatentOutInCompileMode() throws {
    let ev = LispEvaluator(sourceDirectory: tempDir)
    try ev.evaluate(nodes: parseSource("""
      (def x (row-data @name feats @shape [2 3] (tensor @shape [2 3] @data [1 2 3 4 5 6])))
      (def z (latent-out (shrink x @ranges [0:1,0:2])))
      """))
    XCTAssertEqual(ev.rowDataInfo, [RowDataInfo(name: "feats", shape: [2, 3])])
    XCTAssertEqual(ev.latentOutShapes, [[1, 2]])
    guard case .tensor(let x)? = ev.definitions["x"] else { return XCTFail("row-data is a tensor") }
    XCTAssertEqual(x.shape, [1, 6])
  }

  func testRowDataUnderDatasetIsTheTrainerTensor() throws {
    let feed = Tensor([Float](repeating: 2, count: 12)).reshape([2, 6])
    let ctx = DatasetContext(
      rows: 2, frames: 16, sampleRate: 1000, rowMode: .decode,
      learnable: { _, shape, _, _ in Tensor.param(shape, data: [Float](repeating: 0, count: shape.reduce(1, *))) },
      latentRows: { _, _ in throw TrainProtocolError("none") })
    ctx.rowData = { name, shape in
      XCTAssertEqual(name, "feats")
      XCTAssertEqual(shape, [2, 3])
      return feed
    }
    let ev = LispEvaluator(sourceDirectory: tempDir)
    ev.dataset = ctx
    try ev.evaluate(nodes: parseSource("""
      (def x (row-data @name feats @shape [2 3] (undefined-symbol-not-evaluated)))
      (latent-out (reshape x @shape [-1 6]))
      """))
    XCTAssertEqual(ctx.latentOut.count, 1)
    XCTAssertEqual(ctx.latentOut.first?.shape, [2, 6])
  }

  func testWindows1DShapeAndReshapeInference() throws {
    let ev = LispEvaluator(sourceDirectory: tempDir)
    try ev.evaluate(nodes: parseSource("""
      (def x (tensor @shape [2 3 11] @data [\((0..<66).map(String.init).joined(separator: " "))]))
      (def w (windows1d x @size 3 @stride 2))
      (def r (reshape w @shape [-1 3]))
      """))
    guard case .tensor(let w)? = ev.definitions["w"], case .tensor(let r)? = ev.definitions["r"] else {
      return XCTFail("expected tensors")
    }
    XCTAssertEqual(w.shape, [2, 3, 5, 3])
    XCTAssertEqual(r.shape, [30, 3])
    let v = try r.realize()
    XCTAssertEqual(Array(v.prefix(6)), [0, 1, 2, 2, 3, 4])
    XCTAssertThrowsError(try LispEvaluator(sourceDirectory: tempDir).evaluate(nodes: parseSource(
      "(reshape (zeros [2 3]) @shape [-1 4])")))
  }

  func testHostLogMelShapesAndMelBank() {
    var c = EncoderFeatureConfig(sampleRate: 48000)
    XCTAssertEqual(c.frames, 90)  // the spec's recipe: 0.5 s, 1024 / 256
    c.bands = 64
    let ex = EncoderFeatureExtractor(c)
    XCTAssertEqual(ex.melFilters.count, 513 * 64)
    for band in 0..<64 {
      XCTAssertGreaterThan((0..<513).map { ex.melFilters[$0 * 64 + band] }.max()!, 0, "band \(band) is empty")
    }
    // a 1 kHz tone: the loudest band sits near 1 kHz
    let x = (0..<48000).map { Float(Foundation.sin(2 * Double.pi * 1000 * Double($0) / 48000)) }
    let f = ex.logMel(x)
    XCTAssertEqual(f.count, 64 * 90)
    var means = [Float](repeating: 0, count: 64)
    for band in 0..<64 { for t in 0..<90 { means[band] += f[band * 90 + t] / 90 } }
    let peak = means.indices.max { means[$0] < means[$1] }!
    func mel(_ hz: Float) -> Float { 2595 * Foundation.log10(1 + hz / 700) }
    let lo = mel(30), hi = mel(24000)
    let m: Float = lo + (hi - lo) * Float(peak + 1) / 65
    let centre: Float = 700 * (Foundation.pow(10, m / 2595) - 1)
    XCTAssertEqual(centre, 1000, accuracy: 120)
  }

  /// The example encoder (spec architecture) evaluates to [B, D] under a dataset context and
  /// its forward matches a CPU reference of the first conv layer's shape bookkeeping.
  func testExampleEncoderShapes() throws {
    let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
      .appendingPathComponent("../../Examples/LatentEncoder/encoder.lisp").standardizedFileURL
    let nodes = try parseSource(try String(contentsOf: url, encoding: .utf8))
    var created: [String: [Int]] = [:]
    let ctx = DatasetContext(
      rows: 2, frames: 16, sampleRate: 48000, rowMode: .decode,
      learnable: { name, shape, _, _ in
        created[name] = shape
        return Tensor.param(shape, data: (0..<shape.reduce(1, *)).map { Float(Foundation.sin(Double($0))) * 0.05 })
      },
      latentRows: { _, _ in throw TrainProtocolError("none") })
    ctx.rowData = { _, shape in
      Tensor((0..<(2 * shape.reduce(1, *))).map { Float(Foundation.cos(Double($0) * 0.01)) })
        .reshape([2, shape.reduce(1, *)])
    }
    let ev = LispEvaluator(sourceDirectory: url.deletingLastPathComponent())
    ev.dataset = ctx
    try ev.evaluate(nodes: nodes)
    XCTAssertEqual(ctx.latentOut.first?.shape, [2, 8])
    XCTAssertEqual(created["enc_c1"], [320, 96])
    XCTAssertEqual(created["enc_c2"], [480, 128])
    let z = try XCTUnwrap(ctx.latentOut.first).realize()
    XCTAssertEqual(z.count, 16)
    XCTAssertTrue(z.allSatisfy(\.isFinite))
  }

  // MARK: end to end

  private func write(_ s: String, _ name: String) throws -> URL {
    let url = tempDir.appendingPathComponent(name)
    try s.write(to: url, atomically: true, encoding: .utf8)
    return url
  }

  private func json(_ o: Any, _ name: String) throws -> URL {
    let url = tempDir.appendingPathComponent(name)
    try JSONSerialization.data(withJSONObject: o).write(to: url)
    return url
  }

  func testEncodeStageRecoversHeldOutCodes() throws {
    _ = try json(["shape": [2, 3], "data": w.map(Double.init)], "w.json")
    _ = try json(["shape": [1, 3], "data": b.map(Double.init)], "b.json")
    let voiceURL = try write(voice, "voice.lisp")
    let encURL = try write(encoder, "encoder.lisp")
    // the "trained voice": a checkpoint holding the decoder weights
    let voiceRun = tempDir.appendingPathComponent("voice-run")
    try FileManager.default.createDirectory(
      at: voiceRun.appendingPathComponent("ckpt"), withIntermediateDirectories: true)
    let ck: [String: Any] = [
      "version": 1, "stage": "audio", "step": 1,
      "leaves": ["w:w": ["shape": [2, 3], "data": w.map(Double.init)],
                 "w:b": ["shape": [1, 3], "data": b.map(Double.init)]],
    ]
    try JSONSerialization.data(withJSONObject: ck).write(
      to: voiceRun.appendingPathComponent("ckpt/step-000001.json"))

    // 1. synthetic set: render a grid of codes (render stage writes labels.json)
    var z: [[Double]] = []
    for i in 0..<6 { for j in 0..<6 { z.append([-1.5 + 0.6 * Double(i), -1.5 + 0.6 * Double(j)]) } }
    let zURL = try json(["names": z.indices.map { String(format: "h%02d", $0) }, "z": z, "sample_rate": sr], "z.json")
    let synth = tempDir.appendingPathComponent("synth")
    let ro = try TrainDatasetOptions.parse([
      "--stage", "render", "--patch", voiceURL.path, "--job-dir", synth.path, "--z-rows", zURL.path,
      "--init-from", voiceRun.path, "--frames", "8192",
    ])
    _ = try DatasetTrainer(ro).train()
    let labelsURL = synth.appendingPathComponent("renders/labels.json")
    let labels = try JSONSerialization.jsonObject(with: Data(contentsOf: labelsURL)) as! [String: Any]
    XCTAssertEqual((labels["codes"] as! [[NSNumber]]).count, 36)
    XCTAssertEqual(labels["table"] as? String, "codes")

    // 2. encoder, supervised on the labels, 6 rows held out
    let job = tempDir.appendingPathComponent("enc")
    let eo = try TrainDatasetOptions.parse([
      "--stage", "encode", "--patch", voiceURL.path, "--encoder", encURL.path,
      "--dataset", synth.appendingPathComponent("renders").path, "--codes", labelsURL.path,
      "--init-from", voiceRun.path, "--job-dir", job.path, "--frames", "8192",
      "--feat-window", "512", "--feat-hop", "128", "--feat-bands", "24", "--feat-seconds", "0.5",
      "--steps", "400", "--lr", "3e-3", "--batch", "15", "--val", "6", "--clip", "5",
      "--report-every", "100", "--checkpoint-every", "1000",
    ])
    let summary = try DatasetTrainer(eo).train()
    let valMSE = try XCTUnwrap(summary["val_code_mse_median"] as? Double)
    let trainMSE = try XCTUnwrap(summary["train_code_mse_median"] as? Double)
    // labels span [-1.5, 1.5]²: variance ≈ 1.05 per dim
    XCTAssertLessThan(trainMSE, 0.05)
    XCTAssertLessThan(valMSE, 0.15)
    let mr = try XCTUnwrap(summary["val_mr_median"] as? Double)
    let ceiling = try XCTUnwrap(summary["ceiling_mr_median"] as? Double)
    XCTAssertTrue(mr.isFinite && ceiling.isFinite)
    for f in ["feat_stats.json", "mel_filters.json", "encoded.json", "enc_c1.json", "enc_w2.json"] {
      XCTAssertTrue(FileManager.default.fileExists(atPath: job.appendingPathComponent("export/\(f)").path), f)
    }
    XCTAssertFalse(FileManager.default.fileExists(atPath: job.appendingPathComponent("export/w.json").path),
                   "the frozen decoder is not re-exported by the encode stage")
  }
}
