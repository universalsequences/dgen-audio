import Foundation
import XCTest

@testable import DGen
@testable import DGenLazy
@testable import DGenLisp

/// Plain `latch` regions are scheduled at event rate by EventRatePromotionPass.
/// Every test renders the same program with the pass enabled and disabled and
/// requires the same audio, across irregular process() partitions.
final class EventRatePromotionTests: XCTestCase {
  private var savedConfig: (Backend, Float, Int)!

  override func setUpWithError() throws {
    savedConfig = (DGenConfig.backend, DGenConfig.sampleRate, DGenConfig.maxFrameCount)
    DGenConfig.backend = .c
    DGenConfig.sampleRate = 48000
    DGenConfig.maxFrameCount = 128
  }

  override func tearDownWithError() throws {
    (DGenConfig.backend, DGenConfig.sampleRate, DGenConfig.maxFrameCount) = savedConfig
    unsetenv("DGEN_DISABLE_EVENT_PROMOTION")
    LazyGraphContext.reset()
  }

  private struct Compiled {
    let result: CompilationResult
    let params: [String: Int]
  }

  private func compile(_ source: String, promote: Bool) throws -> Compiled {
    if promote {
      unsetenv("DGEN_DISABLE_EVENT_PROMOTION")
    } else {
      setenv("DGEN_DISABLE_EVENT_PROMOTION", "1", 1)
    }
    defer { unsetenv("DGEN_DISABLE_EVENT_PROMOTION") }
    LazyGraphContext.reset()
    let evaluator = LispEvaluator()
    try evaluator.evaluate(nodes: parseSource(source))
    for output in evaluator.outputs {
      LazyGraphContext.current.addOutput(output.signal, channel: output.channel)
    }
    let result = try CompilationPipeline.compile(
      graph: LazyGraphContext.current.graph, backend: .c,
      options: .init(frameCount: 128, debug: false))
    var params: [String: Int] = [:]
    for param in evaluator.params {
      guard let cell = param.cellId else { continue }
      params[param.name] = result.cellAllocations.cellMappings[cell] ?? cell
    }
    return Compiled(result: result, params: params)
  }

  /// Renders `total` frames in `parts`-sized calls. `beforeCall(call, memory)`
  /// runs before each process() call, where a host writes parameters.
  private func render(
    _ compiled: Compiled, parts: [Int], total: Int = 1024,
    beforeCall: (Int, UnsafeMutablePointer<Float>) -> Void = { _, _ in }
  ) throws -> [Float] {
    let result = compiled.result
    let kernel = CCompiledKernel(
      source: result.kernels.map { $0.source }.joined(separator: "\n\n"),
      cellAllocations: result.cellAllocations, memorySize: result.totalMemorySlots,
      defaultHostSampleRate: 48000)
    try kernel.compileAndLoad()
    defer { kernel.cleanup() }
    let count = max(1024, result.totalMemorySlots)
    let memory = UnsafeMutablePointer<Float>.allocate(capacity: count)
    memory.initialize(repeating: 0, count: count)
    let output = UnsafeMutablePointer<Float>.allocate(capacity: 132)
    output.initialize(repeating: 0, count: 132)
    defer { memory.deallocate(); output.deallocate() }
    DGen.injectTensorData(result: result, memory: memory)
    let process = try XCTUnwrap(kernel.getProcessFunction())
    let outputs: [UnsafeMutablePointer<Float>?] = [output]
    var context = DGenProcessContextV1(sampleRate: 48000)
    var audio: [Float] = []
    var call = 0
    while audio.count < total {
      let frames = min(parts[call % parts.count], total - audio.count)
      beforeCall(call, memory)
      outputs.withUnsafeBufferPointer { op in
        withUnsafePointer(to: &context) { cp in
          process(nil, op.baseAddress, UInt32(frames), memory, UnsafeRawPointer(cp), nil)
        }
      }
      audio.append(contentsOf: UnsafeBufferPointer(start: output, count: frames))
      call += 1
    }
    return audio
  }

  private func eventLatchCount(_ compiled: Compiled) -> Int {
    compiled.result.uopBlocks.reduce(0) { total, block in
      total + block.ops.filter { if case .eventLatch = $0.op { return true } else { return false } }.count
    }
  }

  private func assertSameAudio(
    _ actual: [Float], _ expected: [Float], _ label: String,
    file: StaticString = #filePath, line: UInt = #line
  ) {
    XCTAssertEqual(actual.count, expected.count, label, file: file, line: line)
    for (frame, pair) in zip(actual, expected).enumerated() {
      let tolerance = max(0.00002, 0.00002 * abs(pair.1))
      if abs(pair.0 - pair.1) > tolerance {
        XCTFail("\(label): frame \(frame) \(pair.0) != \(pair.1)", file: file, line: line)
        return
      }
    }
  }

  /// A drum-hit shape: per-hit latched controls, a table row picked by a
  /// latched selector, and per-sample decay/oscillator math on top.
  private let hit = """
    (param tune @default 1.3)
    (param rowknob @default 2)
    (param shape @default 0.7)
    (def position (accum 1 0 0 4096))
    (def trig (max (eq (% position 97) 3) (eq (% position 211) 40) (eq (% position 211) 41)))
    (make-history age_h)
    (def age (gswitch trig 0 (+ 1 (read-history age_h))))
    (write-history age_h age)
    (def rows (tensor @shape [4 3] @data [0.1 2 30  0.2 3 40  0.3 5 50  0.4 7 60]))
    (def row (latch (floor rowknob) trig))
    (def vel (latch (+ 0.5 (* 0.001 (% position 300))) trig))
    (def t (latch tune trig))
    (def decay (* (peek rows row 0) (exp (* 0.5 vel)) t))
    (def freq (* (peek rows row 2) t (sqrt vel)))
    (def bend (* (peek rows row 1) (pow vel 1.5)))
    (def env (exp (/ (* -1 age) (* 48000 decay))))
    (def body (sin (* twopi (/ age 48000) (+ freq (* bend (exp (* -0.01 age)))))))
    """

  func testPromotedHitMatchesPerSampleLatchesAcrossPartitions() throws {
    let source = hit + "\n(out (* env body shape) 1)"
    let plain = try compile(source, promote: false)
    let promoted = try compile(source, promote: true)
    XCTAssertEqual(eventLatchCount(plain), 0)
    XCTAssertGreaterThan(eventLatchCount(promoted), 0, "The hit's coefficient math must be promoted")
    let expected = try render(plain, parts: [128])
    for parts in [[128], [1], [37], [3, 17, 64, 7], [127, 128, 1, 4]] {
      assertSameAudio(try render(promoted, parts: parts), expected, "parts \(parts)")
    }
  }

  func testLiveParameterChangesReachPromotedRegionsAtTheNextCall() throws {
    // `shape` and `rowknob` feed promoted math without being latched (shape)
    // or between triggers (rowknob): both must behave exactly as per sample.
    let source = hit + "\n(out (* env body (+ shape (* shape vel))) 1)"
    let plain = try compile(source, promote: false)
    let promoted = try compile(source, promote: true)
    XCTAssertGreaterThan(eventLatchCount(promoted), 0)
    for parts in [[128], [37], [3, 17, 64, 7]] {
      let host: (Compiled) -> (Int, UnsafeMutablePointer<Float>) -> Void = { compiled in
        { call, memory in
          memory[compiled.params["shape"]!] = 0.3 + 0.1 * Float(call % 7)
          memory[compiled.params["rowknob"]!] = Float(call % 4)
          memory[compiled.params["tune"]!] = 1 + 0.05 * Float(call % 5)
        }
      }
      assertSameAudio(
        try render(promoted, parts: parts, beforeCall: host(promoted)),
        try render(plain, parts: parts, beforeCall: host(plain)), "parts \(parts)")
    }
  }

  func testRegionsOnDifferentTriggersStayIndependent() throws {
    // Two hit slots on alternating triggers, as in a crossfading drum voice.
    let source = """
      (param tune @default 1)
      (def position (accum 1 0 0 4096))
      (def a (eq (% position 150) 7))
      (def b (eq (% position 150) 80))
      (def fa (* 110 (latch tune a) (exp (* 0.1 (latch (% position 13) a)))))
      (def fb (* 220 (latch tune b) (exp (* 0.1 (latch (% position 11) b)))))
      (def mixed (+ (sin (* twopi (/ position 48000) fa)) (sin (* twopi (/ position 48000) fb))))
      (out mixed 1)
      """
    let plain = try compile(source, promote: false)
    let promoted = try compile(source, promote: true)
    XCTAssertGreaterThan(eventLatchCount(promoted), 0)
    let host: (Compiled) -> (Int, UnsafeMutablePointer<Float>) -> Void = { compiled in
      { call, memory in memory[compiled.params["tune"]!] = 1 + 0.25 * Float(call % 3) }
    }
    for parts in [[128], [5, 64, 13]] {
      assertSameAudio(
        try render(promoted, parts: parts, beforeCall: host(promoted)),
        try render(plain, parts: parts, beforeCall: host(plain)), "parts \(parts)")
    }
  }

  func testEventLatchMatchesLatchOfEventHeldCoefficient() throws {
    let program = """
      (def position (accum 1 0 0 1024))
      (def event (max (eq (% position 16) 0) (eq (% position 29) 5) (eq (% position 29) 6)))
      (def held (event-hold (+ 0.1 (* position 0.0001)) event))
      (out (* (BACK (exp (* -3 held)) event) (sin (* 0.01 position))) 1)
      """
    let reference = try compile(program.replacingOccurrences(of: "BACK", with: "latch"), promote: false)
    let latched = try compile(program.replacingOccurrences(of: "BACK", with: "event-latch"), promote: false)
    XCTAssertGreaterThan(eventLatchCount(latched), 0)
    let expected = try render(reference, parts: [128])
    for parts in [[128], [1], [3, 17, 64, 7]] {
      assertSameAudio(try render(latched, parts: parts), expected, "parts \(parts)")
    }
  }
}
