import DGenLazy
import Foundation

// Latent Kick decoder contract: Examples/ModalKickAD/scripts/kick_layout.py, written to
// decoder_layout.json by scripts/gen_dsp.py. The voice it drives is scripts/kick_np.py
// (mode="trainer" here: the frame-parallel subset).

enum KickPaths {
  /// The data directory the scripts in scripts/ write ($MODAL_AD_ROOT, see README.md).
  static var root: URL {
    guard let root = ProcessInfo.processInfo.environment["MODAL_AD_ROOT"], !root.isEmpty else {
      FileHandle.standardError.write(
        "MODAL_AD_ROOT is not set: point it at the data directory built by Examples/ModalKickAD/scripts (see Examples/ModalKickAD/README.md)\n"
          .data(using: .utf8)!)
      exit(2)
    }
    return URL(fileURLWithPath: (root as NSString).expandingTildeInPath, isDirectory: true)
  }
  static var kick: URL { root.appendingPathComponent("kick") }
  static var data: URL { kick.appendingPathComponent("data") }
  static var layout: URL { kick.appendingPathComponent("Latent Kick/decoder_layout.json") }
  static var teacher: URL { data.appendingPathComponent("teacher.json") }
  static var tables: URL { data.appendingPathComponent("tables") }
}

struct KickSegment {
  var name: String
  var offset: Int
  var width: Int
  var squash: String
  var lo: Float
  var hi: Float
}

struct KickLayout {
  var P: Int, K: Int, K2: Int, J: Int, NB: Int
  var hidden: [Int]
  var segments: [KickSegment]
  var byName: [String: KickSegment]

  static func load(_ url: URL = KickPaths.layout) throws -> KickLayout {
    let j = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
    let segs = (j["segments"] as! [[String: Any]]).map { s in
      KickSegment(
        name: s["name"] as! String, offset: (s["offset"] as! NSNumber).intValue,
        width: (s["width"] as! NSNumber).intValue, squash: s["squash"] as! String,
        lo: (s["lo"] as! NSNumber).floatValue, hi: (s["hi"] as! NSNumber).floatValue)
    }
    var by: [String: KickSegment] = [:]
    for s in segs { by[s.name] = s }
    func i(_ k: String) -> Int { (j[k] as! NSNumber).intValue }
    return KickLayout(
      P: i("P"), K: i("K"), K2: i("K2"), J: i("J"), NB: i("B"),
      hidden: (j["hidden"] as! [NSNumber]).map { $0.intValue }, segments: segs, byName: by)
  }

  /// u [B, P] -> natural values per segment, each [B, width]
  func squash(_ u: Tensor, batch B: Int) -> [String: Tensor] {
    var out: [String: Tensor] = [:]
    for s in segments {
      let x = u.shrink([(0, B), (s.offset, s.offset + s.width)])
      switch s.squash {
      case "log":
        let a = Foundation.log(s.lo), span = Foundation.log(s.hi) - a
        out[s.name] = (x.sigmoid() * span + a).exp()
      case "lin":
        out[s.name] = x.sigmoid() * (s.hi - s.lo) + s.lo
      default:  // "pi"
        out[s.name] = tanh(x) * Float.pi
      }
    }
    return out
  }
}

/// Teacher rows ($MODAL_AD_ROOT/kick/data/teacher.json, scripts/targets.py): decoder-space u per render id.
enum KickTeacher {
  static func load(_ url: URL = KickPaths.teacher) throws -> [String: [Float]] {
    let j = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
    var out: [String: [Float]] = [:]
    for (id, row) in j["rows"] as! [String: Any] {
      out[id] = ((row as! [String: Any])["u"] as! [NSNumber]).map { $0.floatValue }
    }
    return out
  }
}

/// Fixed noise signals (scripts/export_tables.py): what the instrument computes on a
/// fresh voice from the seed-1 xorshift white.
struct KickTables {
  var frames: Int
  var bands: Int
  var bandsTimeMajor: [Float]  // [frames, bands]
  var hum: [Float]  // [frames]

  static func load(dir: URL = KickPaths.tables, bands: Int) throws -> KickTables {
    func f32(_ name: String) throws -> [Float] {
      let d = try Data(contentsOf: dir.appendingPathComponent(name))
      return d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }
    let b = try f32("bands.f32"), h = try f32("hum.f32")
    precondition(b.count == h.count * bands, "bands.f32 / hum.f32 size mismatch")
    return KickTables(frames: h.count, bands: bands, bandsTimeMajor: b, hum: h)
  }
}

/// One render's performance context: velocity, pitch and the transform knobs (VILLAIN names).
struct KickHit {
  var vel: Double = 1
  var pitch: Double = 261.6255653
  var knobs: [String: Double] = [:]

  static let defaults: [String: (Double, Double, Double)] = [
    "tune": (1, 0.5, 2), "drop_time": (1, 0.25, 4), "tilt": (0, -2, 2), "spread": (1, 0.5, 1.6),
    "beat": (1, 0, 4), "decay": (1, 0.25, 2.5), "bend": (1, 0, 2.5), "attack": (1, 0.25, 3),
    "body": (1, 0, 2), "knock": (1, 0, 3), "tick": (1, 0, 3), "shape": (1, 0, 3), "growl": (1, 0, 3),
    "noise": (1, 0, 4), "hold": (1, 0.25, 4), "release": (1, 0.25, 4), "length": (1, 0.5, 4),
    "drive": (1, 1, 5), "clip_amt": (1, 0, 1), "ceiling": (1, 0.25, 2), "level": (1, 0, 2),
  ]

  func k(_ name: String) -> Double {
    let (d, lo, hi) = Self.defaults[name]!
    return min(max(knobs[name] ?? d, lo), hi)
  }
  var velClamped: Double { min(max(vel, 1e-4), 1) }
  var tuning: Double { 2 * min(max(pitch / 440, 0.125), 4) * k("tune") }
}
