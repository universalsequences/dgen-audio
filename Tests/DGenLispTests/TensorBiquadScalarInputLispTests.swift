import Foundation
import XCTest

@testable import DGen
@testable import DGenLazy
@testable import DGenLisp

/// `(biquad <scalar input> <tensor cutoffs> …)` runs one filter per cutoff over
/// the shared input and returns a tensor. The scalar input used to stay a
/// scalar-shaped view, and the filter's per-element input history then trapped
/// the compiler (Emit+State historyWrite force-unwrapping `nodeToTensor`).
final class TensorBiquadScalarInputLispTests: XCTestCase {
  private var tempDir: URL!
  private let totalFrames = 512

  override func setUpWithError() throws {
    try super.setUpWithError()
    DGenConfig.backend = .c
    DGenConfig.sampleRate = 48000
    LazyGraphContext.reset()
    tempDir = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
      .appendingPathComponent("dgenlisp-biquad-scalar-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
  }

  override func tearDownWithError() throws {
    if let tempDir { try? FileManager.default.removeItem(at: tempDir) }
    try super.tearDownWithError()
  }

  private func source(input: String, element: Int) -> String {
    """
    (def x (cos (* (phasor 220 0) 6.283185)))
    (def freqs (tensor @shape [4] @data [200 800 2000 5000]))
    (def b (biquad \(input) freqs 0.707 1 0))
    (def outsig (sample b \(Float(element) / 4.0) 0))
    """
  }

  private func render(_ source: String, blockSize: Int = 128) throws -> [Float] {
    DGenConfig.maxFrameCount = blockSize
    LazyGraphContext.reset()
    let lazy = LazyGraphContext.current
    let evaluator = LispEvaluator(sourceDirectory: tempDir)
    try evaluator.evaluate(nodes: parseSource(source))
    guard case .signal(let outSig)? = evaluator.definitions["outsig"] else {
      XCTFail("expected a signal named 'outsig'")
      return []
    }
    lazy.addOutput(outSig, channel: 0)
    let result = try CompilationPipeline.compile(
      graph: lazy.graph, backend: .c, options: .init(frameCount: blockSize, debug: false))
    let runtime = try CLazyRuntime(
      kernels: result.kernels, cellAllocations: result.cellAllocations,
      memorySize: result.totalMemorySlots, frameCount: blockSize,
      defaultHostSampleRate: DGenConfig.sampleRate)
    runtime.zeroAllBuffers()
    if let mem = runtime.memoryPointer() { DGen.injectTensorData(result: result, memory: mem) }
    guard let outPtr = runtime.outputsPointer() else {
      XCTFail("no outputs pointer")
      return []
    }
    var out = [Float]()
    for _ in 0..<(totalFrames / blockSize) {
      runtime.runNoCopy(frameCount: blockSize)
      for i in 0..<blockSize { out.append(outPtr[i]) }
    }
    return out
  }

  /// Element k of a tensor biquad is exactly the scalar biquad at cutoff k.
  func testTensorElementMatchesScalarBiquad() throws {
    for (element, freq) in [(0, 200), (3, 5000)] {
      for input in ["x", "(+ x (* freqs 0))"] {
        let tensor = try render(source(input: input, element: element))
        let scalar = try render(
          """
          (def x (cos (* (phasor 220 0) 6.283185)))
          (def outsig (biquad x \(freq) 0.707 1 0))
          """)
        let maxDiff = zip(tensor, scalar).map { abs($0 - $1) }.max() ?? 0
        XCTAssertLessThan(maxDiff, 1e-4, "input \(input) element \(element) vs scalar \(freq) Hz")
      }
    }
  }

  /// Each element filters the shared scalar input at its own cutoff, exactly as
  /// if the input had been broadcast to the filter shape by hand.
  func testScalarInputMatchesExplicitBroadcast() throws {
    for element in [0, 3] {
      let lifted = try render(source(input: "x", element: element))
      let explicit = try render(source(input: "(+ x (* freqs 0))", element: element))
      XCTAssertEqual(lifted.count, totalFrames)
      XCTAssertTrue(lifted.allSatisfy { $0.isFinite })
      XCTAssertGreaterThan(lifted.map { abs($0) }.max() ?? 0, 1e-3)
      for (i, (a, b)) in zip(lifted, explicit).enumerated() {
        XCTAssertEqual(a, b, accuracy: 1e-5, "element \(element) frame \(i)")
      }
    }
    // 200 Hz LP passes the 220 Hz input far less than 5 kHz LP: the elements
    // really are separate filters, not one filter copied.
    let low = try render(source(input: "x", element: 0))
    let high = try render(source(input: "x", element: 3))
    let diff = zip(low, high).map { abs($0 - $1) }.max() ?? 0
    XCTAssertGreaterThan(diff, 1e-2)
  }
}
