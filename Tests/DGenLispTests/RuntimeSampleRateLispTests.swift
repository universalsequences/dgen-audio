import DGenLazy
import XCTest

@testable import DGenLisp

final class RuntimeSampleRateLispTests: XCTestCase {
  override func setUp() {
    super.setUp()
    DGenConfig.backend = .c
    DGenConfig.sampleRate = 44_100
    DGenConfig.maxFrameCount = 64
    LazyGraphContext.reset()
  }

  func testSamplerateIsRuntimeSignalInArithmeticAndMacros() throws {
    let evaluator = LispEvaluator()
    try evaluator.evaluate(nodes: parseSource("""
      (defmacro half-samplerate ()
        (/ samplerate 2.0))
      (out (half-samplerate) 1 @name audio)
      """))

    let graph = LazyGraphContext.current
    for output in evaluator.outputs {
      graph.addOutput(output.signal, channel: output.channel)
    }
    let compilation = try graph.compileOnly(frameCount: 64, voiceCount: 1)
    let source = compilation.kernels.map(\.source).joined(separator: "\n")

    XCTAssertTrue(source.contains("float hostSampleRate"))
    XCTAssertTrue(source.contains("hostSampleRate"))
    XCTAssertFalse(source.contains("44100.000000"))
  }
}
