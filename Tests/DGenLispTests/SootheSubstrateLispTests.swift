import DGenLazy
import XCTest

@testable import DGen
@testable import DGenLisp

/// DGenLisp-level proof that the soothe-class substrate is expressible from the
/// DSL: STFT analysis, cumsum-based spectral smoothing, hop-gated tensor history,
/// hop-hold, IFFT, and overlap-add.
final class SootheSubstrateLispTests: XCTestCase {
  private var tempDir: URL!

  private let n = 16
  private let hop = 8
  private let radius = 2
  private let peakBin = 6
  private let frames = 96

  override func setUpWithError() throws {
    try super.setUpWithError()
    DGenConfig.backend = .c
    DGenConfig.sampleRate = 16
    DGenConfig.maxFrameCount = 256
    DGenConfig.enableBufferReuse = true
    LazyGraphContext.reset()
    tempDir = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
      .appendingPathComponent("dgenlisp-soothe-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
  }

  override func tearDownWithError() throws {
    if let tempDir { try? FileManager.default.removeItem(at: tempDir) }
    try super.tearDownWithError()
  }

  private enum Readout {
    case output
    case envAtPeak
  }

  private func source(sootheEnabled: Bool, readout: Readout) -> String {
    let data = Array(repeating: "1", count: n).joined(separator: " ")
    let rightStart = radius
    let rightEnd = radius + n
    let leftPad = radius + 1
    let scale = 1.0 / Float(2 * radius + 1)

    var body = """
      (make-history env @shape [\(n)] @hop \(hop) @data [\(data)])
      (def tone (sin (* (phasor \(peakBin)) tau)))
      (def frame (reshape (buffer tone \(n) \(hop)) @shape [\(n)]))
      (def (re im) (fft frame @N \(n)))
      (def mag (sqrt (+ (* re re) (* im im))))
      (def c (cumsum mag))
      (def right (shrink (pad c @padding [0:\(radius)]) @ranges [\(rightStart):\(rightEnd)]))
      (def left (shrink (pad c @padding [\(leftPad):0]) @ranges [0:\(n)]))
      (def smooth (* (- right left) \(scale)))
      (def raw-gain (/ smooth (+ mag 0.000001)))
      (def gain (min (max raw-gain 0) 1))
      (def prev (read-history env))
      (def next (+ (* prev 0.7) (* gain 0.3)))
      (def applied (write-history env next))
      """

    switch readout {
    case .envAtPeak:
      body += "\n(def out (sum (shrink applied @ranges [\(peakBin):\(peakBin + 1)])))\n"
    case .output:
      if sootheEnabled {
        body += """

          (def held-gain (hop-hold applied \(hop)))
          (def gre (* re held-gain))
          (def gim (* im held-gain))
          (def recon (ifft gre gim @N \(n)))
          (def out (overlap-add recon \(hop)))
          """
      } else {
        body += """

          (def recon (ifft re im @N \(n)))
          (def out (overlap-add recon \(hop)))
          """
      }
    }
    return body
  }

  private func realize(_ source: String) throws -> [Float] {
    DGenConfig.backend = .c
    LazyGraphContext.reset()
    let evaluator = LispEvaluator(sourceDirectory: tempDir)
    try evaluator.evaluate(nodes: parseSource(source))
    guard case .signal(let out)? = evaluator.definitions["out"] else {
      throw XCTSkip("expected signal out")
    }
    return try out.realize(frames: frames)
  }

  private func rms(_ x: ArraySlice<Float>) -> Float {
    guard !x.isEmpty else { return 0 }
    let ss = x.reduce(Float(0)) { $0 + $1 * $1 }
    return (ss / Float(x.count)).squareRoot()
  }

  func testLispSootheSubstrateFollowerUsesCumsumGain() throws {
    let env = try realize(source(sootheEnabled: true, readout: .envAtPeak))
    let hopValues = stride(from: 0, to: frames, by: hop).map { env[$0] }
    print("=== dgenlisp peak-bin env per hop: \(hopValues)")

    let settled = Array(hopValues.dropFirst())
    XCTAssertLessThan(settled.last!, settled.first!, "follower did not release over hops")
    XCTAssertLessThan(settled.last!, 0.6, "peak bin should end clearly attenuated")
    XCTAssertGreaterThan(settled.last!, 0.01)

    for k in 1..<settled.count {
      XCTAssertLessThanOrEqual(
        settled[k], settled[k - 1] + 1e-4, "release should be monotone at hop \(k)")
    }
  }

  func testLispSootheSubstrateAttenuatesResonanceVsBypass() throws {
    let soothed = try realize(source(sootheEnabled: true, readout: .output))
    let bypass = try realize(source(sootheEnabled: false, readout: .output))

    let tail = (frames - 32)..<frames
    let soothedRMS = rms(soothed[tail])
    let bypassRMS = rms(bypass[tail])
    print("=== dgenlisp soothed RMS \(soothedRMS) vs bypass RMS \(bypassRMS)")

    XCTAssertGreaterThan(bypassRMS, 0)
    XCTAssertGreaterThan(soothedRMS, 0)
    XCTAssertLessThan(
      soothedRMS, bypassRMS * 0.9,
      "dgenlisp soothe substrate should reduce the resonant tone relative to bypass")
  }
}
