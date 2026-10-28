import XCTest

@testable import DGen
@testable import DGenLazy

/// A static block with several shape regions (e.g. a 4-D expand*mul feeding a sum over an inner
/// axis, and the backward of that) was emitted as plain element loops in ONE staticThreads(1)
/// kernel: a single GPU thread walked every element serially. With ~2M elements that kernel took
/// 420 ms of a 650 ms step (research E6 latent-path). Static Metal regions are now emitted as
/// parallel ranges, and splitStaticParallelRanges (which only accepted .sequential static items)
/// gives each region its own N-thread kernel. Results must be unchanged and no static kernel may
/// keep a large serial element loop.
final class StaticRegionSplitTests: XCTestCase {
  override func tearDown() {
    unsetenv("DGEN_NO_STATIC_REGION_SPLIT")
    DGenConfig.maxFrameCount = DGenConfig.defaultFrameCount
    super.tearDown()
  }

  private func run(split: Bool, dumpTo dir: String? = nil) throws -> (Float, [Float], [Float]) {
    if split { unsetenv("DGEN_NO_STATIC_REGION_SPLIT") } else { setenv("DGEN_NO_STATIC_REGION_SPLIT", "1", 1) }
    DGenConfig.backend = .metal
    DGenConfig.sampleRate = 48_000
    DGenConfig.maxFrameCount = 64
    LazyGraphContext.reset()
    let R = 6, T = 5, K = 8, NB = 7
    let a = Tensor.param([R, K], data: (0..<(R * K)).map { Float(($0 * 37) % 11) / 11 - 0.3 })
    let c = Tensor.param([1, NB], data: (0..<NB).map { Float($0) / 3 - 1 })
    let tt = Tensor((0..<T).map { Float($0) * 0.25 }).reshape([1, T, 1]).expand([R, T, K])
    let e = exp(tt * a.reshape([R, 1, K]).expand([R, T, K]) * -1.0)  // [R,T,K]
    let x = a.reshape([R, 1, K, 1]).expand([R, T, K, NB]) - c.reshape([1, 1, 1, NB]).expand([R, T, K, NB])
    let w = exp(x * x * -0.5) * (e * e).reshape([R, T, K, 1]).expand([R, T, K, NB])
    let h = w.sum(axis: 2)  // [R,T,NB]
    let loss = log(h + 0.1).sum()
    let v = try (Tensor([0.0]) + loss).peek(Signal.constant(0)).backward(frames: 64)
    if let dir { LazyGraphContext.current.dumpKernelSources(to: dir) }
    return (v[0], a.grad?.getData() ?? [], c.grad?.getData() ?? [])
  }

  func testSplitStaticRegionsMatchSerialAndParallelize() throws {
    let dir = NSTemporaryDirectory() + "static-region-split-\(getpid())"
    try? FileManager.default.removeItem(atPath: dir)
    let (l0, ga0, gc0) = try run(split: false)
    let (l1, ga1, gc1) = try run(split: true, dumpTo: dir)
    XCTAssertEqual(ga0.count, 48)
    XCTAssertEqual(gc0.count, 7)
    XCTAssertEqual(l1, l0, accuracy: 1e-4 * max(1, abs(l0)))
    for (p, q) in zip(ga1 + gc1, ga0 + gc0) { XCTAssertEqual(p, q, accuracy: 1e-4 * max(1, abs(q))) }
    // No static kernel keeps a serial loop over the 1680-element 4-D region.
    let files = try FileManager.default.contentsOfDirectory(atPath: dir)
    XCTAssertFalse(files.isEmpty)
    for f in files {
      let src = try String(contentsOfFile: dir + "/" + f, encoding: .utf8)
      XCTAssertFalse(src.contains("< 1680;"), "\(f) still loops serially over the 4-D region")
    }
  }
}
