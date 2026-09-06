import Foundation
import XCTest

@testable import DGen
@testable import DGenLazy
@testable import DGenLisp

final class MutableBufferTests: XCTestCase {
  private func render(_ source: String, blockSize: Int) throws -> [Float] {
    let previousBackend = DGenConfig.backend
    let previousRate = DGenConfig.sampleRate
    let previousFrames = DGenConfig.maxFrameCount
    defer {
      DGenConfig.backend = previousBackend
      DGenConfig.sampleRate = previousRate
      DGenConfig.maxFrameCount = previousFrames
    }
    DGenConfig.backend = .c
    DGenConfig.sampleRate = 48000
    DGenConfig.maxFrameCount = blockSize
    LazyGraphContext.reset()
    let lazy = LazyGraphContext.current
    let evaluator = LispEvaluator()
    try evaluator.evaluate(nodes: parseSource(source))
    guard case .signal(let output)? = evaluator.definitions["result"] else {
      throw LispError.invalidArgument("test requires result signal")
    }
    lazy.addOutput(output, channel: 0)
    let compiled = try CompilationPipeline.compile(
      graph: lazy.graph, backend: .c, options: .init(frameCount: blockSize, debug: false))
    let runtime = try CLazyRuntime(
      kernels: compiled.kernels, cellAllocations: compiled.cellAllocations,
      memorySize: compiled.totalMemorySlots, frameCount: blockSize,
      defaultHostSampleRate: 48000)
    runtime.zeroAllBuffers()
    DGen.injectTensorData(result: compiled, memory: try XCTUnwrap(runtime.memoryPointer()))
    var result: [Float] = []
    for _ in 0..<(1024 / blockSize) {
      runtime.runNoCopy(frameCount: blockSize)
      let output = try XCTUnwrap(runtime.outputsPointer())
      result.append(contentsOf: UnsafeBufferPointer(start: output, count: blockSize))
    }
    return result
  }

  private func check(_ source: String, expected: (Int) -> Float,
                     file: StaticString = #filePath, line: UInt = #line) throws {
    for block in [1, 64, 512] {
      let values = try render(source, blockSize: block)
      for (i, value) in values.enumerated() {
        XCTAssertEqual(value, expected(i), accuracy: 1e-6,
                       "block=\(block) frame=\(i)", file: file, line: line)
      }
    }
  }

  func testWriteThenReadPersistentIncrement() throws {
    try check("""
      (def b (tensor @shape [8]))
      (def result (seq (poke b 0 (+ (peek b 0) 1)) (peek b 0)))
      """, expected: { Float($0 + 1) })
  }

  func testReadThenWriteReturnsSnapshot() throws {
    try check("""
      (def b (tensor @shape [8]))
      (def old (peek b 0))
      (def result (seq old (poke b 0 (+ old 1)) old))
      """, expected: { Float($0) })
  }

  func testRepeatedWritesAliasesAndNestedSequence() throws {
    try check("""
      (def b (tensor @shape [8]))
      (def alias b)
      (def result
        (seq (poke b 0 3)
             (seq (poke alias 0 (+ (peek b 0) 2)) (peek alias 0))))
      """, expected: { _ in 5 })
  }

  func testPureMiddleOperandDoesNotDropEarlierEffects() throws {
    let source = """
      (def b (tensor @shape [2]))
      (def result (seq (poke b 0 3) 42 (poke b 0 7) (peek b 0)))
      """
    LazyGraphContext.reset()
    try LispEvaluator().evaluate(nodes: parseSource(source))
    let graph = LazyGraphContext.current.graph
    let writes = graph.nodes.values.filter {
      if case .memoryWrite = $0.op { return true }
      return false
    }.sorted { $0.id < $1.id }
    XCTAssertEqual(writes.count, 2)
    XCTAssertTrue(writes[1].temporalDependencies.contains(writes[0].id))
    try check(source, expected: { _ in 7 })
  }

  func testWrappedFractionalIndexAndChannelClamp() throws {
    try check("""
      (def b (tensor @shape [4 2]))
      (def result
        (seq (poke b -1.5 9 8)
             (poke b 7 -4 4)
             (+ (peek b 2.5 1) (peek b 3 0))))
      """, expected: { _ in 8 })
  }

  func testRingBufferWithHistoryWriteHead() throws {
    try check("""
      (def b (tensor @shape [8]))
      (make-history cursor)
      (def i (read-history cursor))
      (write-history cursor (+ i 1))
      (def result (seq (poke b i (+ i 1)) (peek b (- i 3))))
      """, expected: { $0 < 3 ? 0 : Float($0 - 2) })
  }

  func testTwoBuffersAndFractionalReadAcrossWrap() throws {
    try check("""
      (def a (tensor @shape [4] @data [1 2 3 4]))
      (def b (tensor @shape [4]))
      (def result
        (seq (poke a 0 8)
             (poke b 0 (peek a 3.5))
             (poke b 1 10)
             (peek b 0.5)))
      """, expected: { _ in 8 })
  }

  func testMutableTensorMathRejectsInsteadOfReadingStaleStorage() throws {
    XCTAssertThrowsError(try render("""
      (def b (tensor @shape [4]))
      (def copy (+ b b))
      (def result (seq (poke b 0 1) (peek copy 0)))
      """, blockSize: 64)) { error in
      XCTAssertTrue(String(describing: error).contains("mutable buffers support scalar"), "\(error)")
    }
  }

  func testWriteOnlyIsPerFrameAndReturnsValue() throws {
    try check("""
      (def b (tensor @shape [2]))
      (def result (poke b 0 7))
      """, expected: { _ in 7 })
  }

  func testInvalidArgumentsRejectWithoutTrapping() throws {
    let invalid = [
      "(poke 1 0 2)", "(poke (tensor @shape [2]) 0)",
      "(poke (tensor @shape [2 2 2]) 0 1)",
      "(poke (+ (tensor @shape [2]) 1) 0 1)",
      "(poke (reshape (tensor @shape [4]) @shape [2 2]) 0 1)",
      "(seq)", "(seq 1)", "(seq 1 (tensor @shape [2]))"
    ]
    for source in invalid {
      LazyGraphContext.reset()
      XCTAssertThrowsError(try LispEvaluator().evaluate(nodes: parseSource(source)), source)
    }
  }
}
