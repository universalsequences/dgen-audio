import Foundation

/// Paths into the eseq research workspace (overridable with ESEQ_MODAL_AD_ROOT).
enum Paths {
  static var research: URL {
    if let root = ProcessInfo.processInfo.environment["ESEQ_MODAL_AD_ROOT"] {
      return URL(fileURLWithPath: root, isDirectory: true)
    }
    return URL(
      fileURLWithPath:
        "/Users/alecresende/code/learning/anthropic/eseq/.local/research/modal-autograd",
      isDirectory: true)
  }
  static var modesJSON: URL { research.appendingPathComponent("data/villain_modes.json") }
  static var manifest: URL { research.appendingPathComponent("data/villain/manifest.json") }
  static var villain: URL { research.appendingPathComponent("data/villain") }
  static var e1: URL { research.appendingPathComponent("e1") }
  static func stem(_ id: String, _ kind: String) -> URL {
    villain.appendingPathComponent("stems/\(id).\(kind).wav")
  }
  static func out(_ id: String) -> URL { villain.appendingPathComponent("\(id).wav") }
}

/// One head mode slot as VILLAIN's dsp computes it (tuning-1 Hz, seconds).
struct VillainSlot {
  var f, amp, dec0, g0, tau, rise0, t0, ph, mh, md: Double
}

/// Block-rate values of one render needed for the head and the chain.
struct VillainRender {
  var id: String
  var vel: Double
  var slots: [VillainSlot]
  var chain: [String: Double]
  var global: [String: Double]

  func c(_ key: String) -> Double { chain[key] ?? 0 }
}

enum VillainData {
  static let sampleRate: Float = 48_000
  static let c4: Double = 261.6255653
  static var tuning: Double { 2 * min(max(c4 / 440, 0.25), 4) }

  static func loadRenders(ids: [String]) throws -> [VillainRender] {
    let modes = try JSONSerialization.jsonObject(with: Data(contentsOf: Paths.modesJSON))
      as! [String: Any]
    let renders = modes["renders"] as! [String: Any]
    let manifest = try JSONSerialization.jsonObject(with: Data(contentsOf: Paths.manifest))
      as! [String: Any]
    var vel: [String: Double] = [:]
    for item in manifest["items"] as! [[String: Any]] {
      vel[item["id"] as! String] = (item["vel"] as? NSNumber)?.doubleValue ?? 1
    }
    return try ids.map { id in
      guard let r = renders[id] as? [String: Any] else {
        throw NSError(domain: "ModalSnareAD", code: 1, userInfo: [NSLocalizedDescriptionKey: "no render \(id)"])
      }
      let slots = (r["slots"] as! [[String: Any]]).map { s -> VillainSlot in
        func v(_ k: String) -> Double { (s[k] as? NSNumber)?.doubleValue ?? 0 }
        return VillainSlot(
          f: v("f"), amp: v("amp"), dec0: v("dec0"), g0: v("g0"), tau: v("tau"),
          rise0: v("rise0"), t0: v("t0"), ph: v("ph"), mh: v("mh"), md: v("md"))
      }
      func dict(_ k: String) -> [String: Double] {
        var out: [String: Double] = [:]
        for (key, value) in (r[k] as? [String: Any]) ?? [:] {
          if let n = value as? NSNumber { out[key] = n.doubleValue }
        }
        return out
      }
      return VillainRender(
        id: id, vel: vel[id] ?? 1, slots: slots, chain: dict("chain"), global: dict("global"))
    }
  }

  static func allIDs() -> [String] {
    guard let data = try? Data(contentsOf: Paths.manifest),
      let m = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
      let items = m["items"] as? [[String: Any]]
    else { return [] }
    return items.compactMap { $0["id"] as? String }
  }

  static func snareIDs(velocity: String = "v100") -> [String] {
    (0..<32).map { String(format: "snare_%02d_%@", $0, velocity) }
  }
}

enum WAV {
  /// Read mono float32 or int16 PCM WAV (first channel).
  static func read(_ url: URL) throws -> [Float] {
    let (samples, _) = try readWithRate(url)
    return samples
  }

  static func readWithRate(_ url: URL) throws -> ([Float], Float) {
    let d = try Data(contentsOf: url)
    func u32(_ o: Int) -> UInt32 { d.subdata(in: o..<o + 4).withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) } }
    func u16(_ o: Int) -> UInt16 { d.subdata(in: o..<o + 2).withUnsafeBytes { $0.loadUnaligned(as: UInt16.self) } }
    var off = 12
    var fmt = 3, channels = 1, bits = 32
    var rate: Float = 48_000
    while off + 8 <= d.count {
      let id = String(data: d.subdata(in: off..<off + 4), encoding: .ascii) ?? ""
      let size = Int(u32(off + 4))
      if id == "fmt " {
        fmt = Int(u16(off + 8)); channels = Int(u16(off + 10)); rate = Float(u32(off + 12))
        bits = Int(u16(off + 22))
        if fmt == 0xFFFE { fmt = Int(u16(off + 32)) }
      } else if id == "data" {
        let body = d.subdata(in: off + 8..<min(d.count, off + 8 + size))
        let frameBytes = channels * bits / 8
        let n = body.count / frameBytes
        var out = [Float](repeating: 0, count: n)
        body.withUnsafeBytes { raw in
          for i in 0..<n {
            if fmt == 3 && bits == 32 {
              out[i] = raw.loadUnaligned(fromByteOffset: i * frameBytes, as: Float.self)
            } else if bits == 16 {
              out[i] = Float(raw.loadUnaligned(fromByteOffset: i * frameBytes, as: Int16.self)) / 32768
            }
          }
        }
        return (out, rate)
      }
      off += 8 + size + (size & 1)
    }
    throw NSError(domain: "WAV", code: 2, userInfo: [NSLocalizedDescriptionKey: "no data chunk \(url.path)"])
  }

  static func write(_ url: URL, _ samples: [Float], sampleRate: Float = VillainData.sampleRate) throws {
    var d = Data()
    func put32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) } }
    func put16(_ v: UInt16) { withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) } }
    d.append(contentsOf: Array("RIFF".utf8)); put32(UInt32(36 + samples.count * 4))
    d.append(contentsOf: Array("WAVE".utf8)); d.append(contentsOf: Array("fmt ".utf8))
    put32(16); put16(3); put16(1); put32(UInt32(sampleRate)); put32(UInt32(sampleRate) * 4)
    put16(4); put16(32)
    d.append(contentsOf: Array("data".utf8)); put32(UInt32(samples.count * 4))
    samples.withUnsafeBufferPointer { d.append(Data(buffer: $0)) }
    try FileManager.default.createDirectory(
      at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    try d.write(to: url, options: .atomic)
  }
}

func writeJSON(_ value: Any, to url: URL) throws {
  try FileManager.default.createDirectory(
    at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
  let data = try JSONSerialization.data(withJSONObject: value, options: [.prettyPrinted, .sortedKeys])
  try data.write(to: url, options: .atomic)
}

/// Time-major interleave of B equal-length (padded/truncated) signals: [t*B + b].
func interleave(_ signals: [[Float]], frames: Int) -> [Float] {
  let B = signals.count
  var out = [Float](repeating: 0, count: frames * B)
  for (b, s) in signals.enumerated() {
    for t in 0..<min(frames, s.count) { out[t * B + b] = s[t] }
  }
  return out
}

func deinterleave(_ flat: [Float], batch B: Int) -> [[Float]] {
  let frames = flat.count / max(1, B)
  return (0..<B).map { b in (0..<frames).map { flat[$0 * B + b] } }
}
