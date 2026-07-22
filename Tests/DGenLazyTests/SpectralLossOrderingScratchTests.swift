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
    let target = try load("\(root)/mono_rung2_target.wav")
    let start = try load("\(scratch)/e10_start.wav")
    let polished = try load("\(scratch)/p3_03.wav")
    for smooth in [false, true] {
      let ls = try gpuLoss(start, target, smooth: smooth)
      let lp = try gpuLoss(polished, target, smooth: smooth)
      print("GPU \(smooth ? "smoothL2" : "prodL1"): start=\(ls) polished=\(lp) "
        + (lp < ls ? "POLISHED WINS" : "start wins"))
    }
  }
}
