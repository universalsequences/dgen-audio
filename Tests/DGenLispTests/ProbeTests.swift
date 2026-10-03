import Foundation
import XCTest

@testable import DGen
@testable import DGenLazy
@testable import DGenLisp

final class ProbeTests: XCTestCase {
  private var savedConfig: (Backend, Float, Int)!

  override func setUpWithError() throws {
    savedConfig = (DGenConfig.backend, DGenConfig.sampleRate, DGenConfig.maxFrameCount)
    DGenConfig.backend = .c
    DGenConfig.sampleRate = 48000
    DGenConfig.maxFrameCount = 128
    LazyGraphContext.reset()
  }

  override func tearDownWithError() throws {
    (DGenConfig.backend, DGenConfig.sampleRate, DGenConfig.maxFrameCount) = savedConfig
    LazyGraphContext.reset()
  }

  private func evaluate(_ source: String, probes: Bool = true) throws -> LispEvaluator {
    LazyGraphContext.reset()
    let evaluator = LispEvaluator()
    evaluator.probesEnabled = probes
    try evaluator.evaluate(nodes: parseSource(source))
    return evaluator
  }

  private func compile(_ evaluator: LispEvaluator) throws -> CompilationResult {
    for output in evaluator.outputs {
      LazyGraphContext.current.addOutput(output.signal, channel: output.channel)
    }
    return try LazyGraphContext.current.compileOnly(frameCount: 128, voiceCount: 1)
  }

  private func manifest(_ evaluator: LispEvaluator) throws -> PatchManifest {
    let compilation = try compile(evaluator)
    return generateManifest(
      compilerResult: CompilerResult(
        dylibPath: "", cSourcePath: "", compilationResult: compilation, cSource: ""),
      evaluator: evaluator,
      options: CompilerOptions(
        outputDir: ".", name: "patch", sampleRate: 48000, maxFrames: 128, voiceCount: 1,
        debug: false))
  }

  /// Renders `frames` samples into `channelCount` output buffers.
  private func render(_ evaluator: LispEvaluator, channelCount: Int, frames: Int = 128) throws
    -> [[Float]]
  {
    let result = try compile(evaluator)
    let kernel = CCompiledKernel(
      source: result.kernels.map { $0.source }.joined(separator: "\n\n"),
      cellAllocations: result.cellAllocations, memorySize: result.totalMemorySlots,
      defaultHostSampleRate: 48000)
    try kernel.compileAndLoad()
    defer { kernel.cleanup() }
    let count = max(1024, result.totalMemorySlots)
    let memory = UnsafeMutablePointer<Float>.allocate(capacity: count)
    memory.initialize(repeating: 0, count: count)
    let buffers = (0..<channelCount).map { _ -> UnsafeMutablePointer<Float> in
      let buffer = UnsafeMutablePointer<Float>.allocate(capacity: frames)
      buffer.initialize(repeating: -999, count: frames)
      return buffer
    }
    defer {
      memory.deallocate()
      buffers.forEach { $0.deallocate() }
    }
    DGen.injectTensorData(result: result, memory: memory)
    let process = try XCTUnwrap(kernel.getProcessFunction())
    let outputs: [UnsafeMutablePointer<Float>?] = buffers
    var context = DGenProcessContextV1(sampleRate: 48000)
    outputs.withUnsafeBufferPointer { op in
      withUnsafePointer(to: &context) { cp in
        process(nil, op.baseAddress, UInt32(frames), memory, UnsafeRawPointer(cp), nil)
      }
    }
    return buffers.map { Array(UnsafeBufferPointer(start: $0, count: frames)) }
  }

  func testProbeIsIdentityAndRecordsDefaults() throws {
    let evaluator = try evaluate("(out (* (probe (phasor 100)) 0.5) 1 @name audio)")
    XCTAssertEqual(evaluator.probes.count, 1)
    let probe = try XCTUnwrap(evaluator.probes.first)
    XCTAssertEqual(probe.id, "probe-0")
    XCTAssertEqual(probe.occurrence, 0)
    XCTAssertEqual(probe.view, "number")
    XCTAssertNil(probe.name)
    XCTAssertEqual(probe.channel, 1)
    XCTAssertEqual(evaluator.outputs.filter { !$0.probe }.map(\.channel), [0])
  }

  func testProbeChannelsFollowOutsDeclaredLaterInSource() throws {
    let source = """
      (def env (probe (phasor 0.5) @id "env" @view scope @name envelope))
      (probe (* env 2))
      (out (* env (phasor 220)) 1 @name audio)
      (out (> env 0.0001) 2 @name amp @amp true)
      (out (phasor 0.25) 4 @name macro-a @modulator 1)
      """
    let evaluator = try evaluate(source)
    let manifest = try manifest(evaluator)

    XCTAssertEqual(manifest.ampOutput?.channel, 1)
    XCTAssertEqual(manifest.probes.map(\.id), ["env", "probe-1"])
    XCTAssertEqual(manifest.probes.map(\.channel), [4, 5])
    XCTAssertEqual(manifest.probes.map(\.view), ["scope", "number"])
    XCTAssertEqual(manifest.probes.map(\.name), ["envelope", nil])
    // Probe channels are also listed in outputs[], so hosts that size their
    // buffers by the highest output channel allocate them.
    XCTAssertEqual(manifest.outputs.map(\.channel), [0, 1, 3, 4, 5])

    let json = String(decoding: try JSONEncoder().encode(manifest), as: UTF8.self)
    XCTAssertTrue(json.contains("\"probes\":["))
    XCTAssertTrue(json.contains("\"name\":null"))
  }

  func testDanglingTopLevelProbeSurvivesAndCompiles() throws {
    let evaluator = try evaluate(
      """
      (out (phasor 100) 1)
      (probe (+ (phasor 1000) 3) @id "dangling")
      """)
    let result = try compile(evaluator)
    let source = result.kernels.map { $0.source }.joined(separator: "\n")
    XCTAssertTrue(source.contains("out[1]"), "probe channel was eliminated:\n\(source)")
  }

  func testDuplicateIdsCountOccurrencesInEvaluationOrder() throws {
    let evaluator = try evaluate(
      """
      (defmacro tap (x) (probe x @id "tap"))
      (def a (tap (phasor 1)))
      (def b (probe (phasor 2) @id "other"))
      (def c (tap (phasor 3)))
      (out (+ a b c) 1)
      """)
    XCTAssertEqual(evaluator.probes.map(\.id), ["tap", "other", "tap"])
    XCTAssertEqual(evaluator.probes.map(\.occurrence), [0, 0, 1])
    XCTAssertEqual(evaluator.probes.map(\.channel), [1, 2, 3])
  }

  func testProbeRejectsTensorSignals() throws {
    XCTAssertThrowsError(
      try evaluate(
        """
        (def t (ones [4]))
        (probe t)
        (out (phasor 1) 1)
        """)
    ) { error in
      XCTAssertTrue("\(error)".contains("probe requires a scalar signal"), "\(error)")
    }
  }

  func testNoProbesLowersToIdentity() throws {
    let evaluator = try evaluate(
      """
      (out (* (probe (phasor 100) @id "p") 0.5) 1)
      (probe (phasor 3))
      """, probes: false)
    XCTAssertTrue(evaluator.probes.isEmpty)
    XCTAssertEqual(evaluator.outputs.map(\.channel), [0])
    XCTAssertTrue(try manifest(evaluator).probes.isEmpty)
  }

  func testProbeChannelReceivesSignalValues() throws {
    let evaluator = try evaluate(
      """
      (def x (phasor 1000))
      (out (* (probe x @id "ph") 0.5) 1)
      (probe (+ x 3) @id "shifted")
      """)
    XCTAssertEqual(evaluator.probes.map(\.channel), [1, 2])
    let channels = try render(evaluator, channelCount: 3)
    for i in 0..<128 {
      XCTAssertEqual(channels[1][i] * 0.5, channels[0][i], accuracy: 1e-6, "frame \(i)")
      XCTAssertEqual(channels[2][i], channels[1][i] + 3, accuracy: 1e-5, "frame \(i)")
    }
    // The phasor actually moved, so the probe isn't a constant.
    XCTAssertGreaterThan(channels[1].max()! - channels[1].min()!, 0.5)
  }
}
