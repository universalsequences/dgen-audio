import XCTest

@testable import DGen
@testable import DGenLazy
@testable import DGenLisp

/// Zero-filled buffers carry no manifest init data, and whole-number peeks
/// read memory once instead of interpolating.
final class TensorStorageTests: XCTestCase {
  override func setUp() {
    super.setUp()
    DGenConfig.backend = .c
    DGenConfig.sampleRate = 48_000
    DGenConfig.maxFrameCount = 64
    LazyGraphContext.reset()
  }

  private func manifest(_ source: String) throws -> (PatchManifest, String) {
    let evaluator = LispEvaluator()
    try evaluator.evaluate(nodes: parseSource(source))
    let graph = LazyGraphContext.current
    for output in evaluator.outputs {
      graph.addOutput(output.signal, channel: output.channel)
    }
    let compilation = try graph.compileOnly(frameCount: 8, voiceCount: 1)
    let source = compilation.kernels.map(\.source).joined(separator: "\n")
    let manifest = generateManifest(
      compilerResult: CompilerResult(
        dylibPath: "", cSourcePath: "", compilationResult: compilation, cSource: ""),
      evaluator: evaluator,
      options: CompilerOptions(
        outputDir: ".", name: "patch", sampleRate: 48_000, maxFrames: 64, voiceCount: 1,
        debug: false))
    return (manifest, source)
  }

  func testZeroFilledBufferHasNoInitDataButKeepsMetadata() throws {
    let (m, _) = try manifest("""
      (def ring (tensor @shape [48000 2] @name ring))
      (def table (tensor @shape [4] @data [0 0.5 0 -0.5] @name table))
      (out (seq (poke ring 3 1 0.25) (+ (peek ring 3 1) (peek table 1))) 1)
      """)
    // Only the nonzero table needs initializing; memory starts zeroed.
    XCTAssertEqual(m.tensorInitData.count, 1)
    XCTAssertEqual(m.tensorInitData[0].data, [0, 0.5, 0, -0.5])
    // The zero buffer is still located for hosts, at its own cells.
    let ring = try XCTUnwrap(m.tensors.first { $0.name == "ring" })
    XCTAssertEqual(ring.shape, [48000, 2])
    XCTAssertNotEqual(ring.cellOffset, m.tensorInitData[0].offset)
  }

  func testIntegralPeekReadsOnce() throws {
    let fractional = try manifest("""
      (def t (tensor @shape [8 2] @data [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15]))
      (out (peek t (* (phasor 100) 7) 1) 1)
      """).1
    LazyGraphContext.reset()
    let integral = try manifest("""
      (def t (tensor @shape [8 2] @data [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15]))
      (out (peek t (floor (* (phasor 100) 7)) 1) 1)
      """).1
    LazyGraphContext.reset()
    let baseline = try manifest("""
      (def t (tensor @shape [8 2] @data [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15]))
      (out (floor (* (phasor 100) 7)) 1)
      """).1
    func reads(_ c: String) -> Int {
      c.components(separatedBy: "memory[").count - 1
    }
    // Channel 1 of 2 is whole, so only the index can interpolate: two reads
    // for a fractional index, one for a floored one (per emitted copy).
    let f = reads(fractional) - reads(baseline)
    let i = reads(integral) - reads(baseline)
    XCTAssertGreaterThan(i, 0)
    XCTAssertEqual(f, 2 * i)
  }

  func testIntegralityAnalysis() {
    LazyGraphContext.reset()
    let g = LazyGraphContext.current.graph
    let x = g.n(.input(0), [])
    let fl = g.n(.floor, [x])
    let three = g.n(.constant(3), [])
    let half = g.n(.constant(0.5), [])
    let sum = g.n(.add, [fl, three])
    let cond = g.n(.gt, [x, half])
    XCTAssertTrue(isProvablyIntegral(sum, in: g))
    XCTAssertTrue(isProvablyIntegral(g.n(.gswitch, [x, sum, three]), in: g))
    XCTAssertTrue(isProvablyIntegral(g.n(.mul, [cond, sum]), in: g))
    XCTAssertFalse(isProvablyIntegral(x, in: g))
    XCTAssertFalse(isProvablyIntegral(g.n(.add, [fl, half]), in: g))
    XCTAssertFalse(isProvablyIntegral(g.n(.div, [fl, three]), in: g))
    XCTAssertFalse(isProvablyIntegral(g.n(.gswitch, [x, sum, half]), in: g))
  }
}

/// Values are unchanged: integral reads equal the interpolating ones.
final class IntegralPeekValueTests: XCTestCase {
  private func render(_ source: String) throws -> [Float] {
    DGenConfig.backend = .c
    DGenConfig.sampleRate = 48000
    DGenConfig.maxFrameCount = 64
    LazyGraphContext.reset()
    let lazy = LazyGraphContext.current
    let evaluator = LispEvaluator()
    try evaluator.evaluate(nodes: parseSource(source))
    guard case .signal(let output)? = evaluator.definitions["result"] else {
      throw LispError.invalidArgument("test requires result signal")
    }
    lazy.addOutput(output, channel: 0)
    let compiled = try CompilationPipeline.compile(
      graph: lazy.graph, backend: .c, options: .init(frameCount: 64, debug: false))
    let runtime = try CLazyRuntime(
      kernels: compiled.kernels, cellAllocations: compiled.cellAllocations,
      memorySize: compiled.totalMemorySlots, frameCount: 64, defaultHostSampleRate: 48000)
    runtime.zeroAllBuffers()
    DGen.injectTensorData(result: compiled, memory: try XCTUnwrap(runtime.memoryPointer()))
    var result: [Float] = []
    for _ in 0..<4 {
      runtime.runNoCopy(frameCount: 64)
      let out = try XCTUnwrap(runtime.outputsPointer())
      result.append(contentsOf: UnsafeBufferPointer(start: out, count: 64))
    }
    return result
  }

  func testIntegralAndFractionalReadsAgree() throws {
    let table = (0..<32).map { String(Float($0) * 0.25 - 3) }.joined(separator: " ")
    // A ring written every frame, read back at whole-number offsets on both
    // channels, including negative indices that wrap.
    let body = """
      (def t (tensor @shape [16 2] @data [\(table)]))
      (def buf (tensor @shape [64 2]))
      (make-history h)
      (def w (read-history h))
      (write-history h (+ w 1))
      (def i (seq (poke buf w 0 (* w 0.5)) (poke buf w 1 (- w)) w))
      """
    let integral = try render(body + """
      (def result (+ (peek t (floor (- (* i 3) 7)) 1) (peek buf (floor (- i 5)) 0) (peek buf (floor i) 1)))
      """)
    let plain = try render(body + """
      (def result (+ (peek t (- (* i 3) 7) 1) (peek buf (- i 5) 0) (peek buf i 1)))
      """)
    XCTAssertEqual(integral, plain)
    XCTAssertNotEqual(Set(integral).count, 1)
  }

  func testFractionalPeekStillInterpolates() throws {
    let v = try render("""
      (def t (tensor @shape [4] @data [0 1 2 3]))
      (def result (peek t 1.5))
      """)
    XCTAssertEqual(v[0], 1.5, accuracy: 1e-6)
  }
}
