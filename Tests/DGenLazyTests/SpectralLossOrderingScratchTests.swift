import DGen
import XCTest

@testable import DGenLazy

/// Pure-audio check: does the GPU spectral loss rank two candidate renders
/// against the target the same way as an offline log-magnitude replica?
final class SpectralLossOrderingScratchTests: XCTestCase {
  override func setUp() {
    super.setUp()
    DGenConfig.maxFrameCount = 32768
    LazyGraphContext.reset()
  }

  private func load(_ path: String) throws -> [Float] {
    let (samples, _) = try AudioFile.load(url: URL(fileURLWithPath: path))
    return Array(samples.prefix(26624))
  }

  private func gpuLoss(_ a: [Float], _ b: [Float], smooth: Bool) throws -> Float {
    LazyGraphContext.reset()
    DGenSpectralConfig.logMagnitudeEpsilon = 1e-3
    let sa = Tensor(a).toSignal(maxFrames: 26624)
    let sb = Tensor(b).toSignal(maxFrames: 26624)
    var total = Signal.constant(0.0)
    for w in [256, 512, 1024, 2048] {
      total = total + spectralLossFFT(
        sa, sb, windowSize: w, useHannWindow: true,
        useLogMagnitude: true, useSmoothLogMagnitude: smooth,
        lossMode: smooth ? .l2 : .l1, hop: max(1, w / 4), normalize: true)
    }
    return try total.realize(frames: 26624).reduce(0, +)
  }

  func testOrderingAgainstOfflineReplica() throws {
    let root = "/Users/alecresende/code/swift/dgen/output/monologue_bass"
    let scratch = "/private/tmp/claude-501/-Users-alecresende-code-swift-dgen/e7b2e321-319c-4904-a7d7-537ce36ef6a1/scratchpad"
    let paths = [
      "\(root)/mono_rung2_target.wav",
      "\(scratch)/e10_start.wav",
      "\(scratch)/p3_03.wav",
    ]
    guard paths.allSatisfy({ FileManager.default.fileExists(atPath: $0) }) else {
      throw XCTSkip("scratch audio fixtures are not available on this machine")
    }
    let target = try load(paths[0])
    let start = try load(paths[1])
    let polished = try load(paths[2])
    for smooth in [false, true] {
      let ls = try gpuLoss(start, target, smooth: smooth)
      let lp = try gpuLoss(polished, target, smooth: smooth)
      print("GPU \(smooth ? "smoothL2" : "prodL1"): start=\(ls) polished=\(lp) "
        + (lp < ls ? "POLISHED WINS" : "start wins"))
    }
  }
}
