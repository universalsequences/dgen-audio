import Foundation
import XCTest

@testable import DGen
@testable import DGenLazy
@testable import DGenLisp

final class EventHoldTests: XCTestCase {
  private var savedConfig: (Backend, Float, Int)!

  override func setUpWithError() throws {
    savedConfig = (DGenConfig.backend, DGenConfig.sampleRate, DGenConfig.maxFrameCount)
    DGenConfig.backend = .c
    DGenConfig.sampleRate = 48000
    DGenConfig.maxFrameCount = 128
  }

  override func tearDownWithError() throws {
    (DGenConfig.backend, DGenConfig.sampleRate, DGenConfig.maxFrameCount) = savedConfig
    LazyGraphContext.reset()
  }

  private func compile(_ source: String, reuse: Bool = true) throws -> CompilationResult {
    LazyGraphContext.reset()
    let evaluator = LispEvaluator()
    try evaluator.evaluate(nodes: parseSource(source))
    for output in evaluator.outputs {
      LazyGraphContext.current.addOutput(output.signal, channel: output.channel)
    }
    return try CompilationPipeline.compile(graph: LazyGraphContext.current.graph,
      backend: .c, options: .init(frameCount: 128, debug: false, enableBufferReuse: reuse))
  }

  private func render(_ source: String, parts: [Int]) throws -> [Float] {
    let result = try compile(source)
    let kernel = CCompiledKernel(source: result.kernels.map { $0.source }.joined(separator: "\n\n"),
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
    var part = 0
    while audio.count < 512 {
      let frames = min(parts[part % parts.count], 512 - audio.count)
      outputs.withUnsafeBufferPointer { op in
        withUnsafePointer(to: &context) { cp in
          process(nil, op.baseAddress, UInt32(frames), memory, UnsafeRawPointer(cp), nil)
        }
      }
      audio.append(contentsOf: UnsafeBufferPointer(start: output, count: frames))
      part += 1
    }
    return audio
  }

  func testEventCoefficientsMatchEagerLatchesAcrossPartitions() throws {
    // Multiple events can share a nominal 16-sample interval. A compressed
    // per-hop scratch slot would overwrite the earlier event's coefficients.
    for width in [1, 3, 8] {
      let values = (1...width).map { String($0) }.joined(separator: " ")
      let weights = width == 1 ? "1" : "(tensor @shape [\(width)] @data [\(values)])"
      let program = """
        (def position (accum 1 0 0 1024))
        (def event (max (eq (% position 16) 0) (eq (% position 29) 5)
          (eq (% position 29) 6)))
        (def first (HOLD (+ 0.1 (* position 0.0001)) event))
        (def second (HOLD (+ 0.2 (* position 0.0002)) event))
        (def coefficient (latch (exp (* -1 \(weights) (+ first second))) event))
        (make-history h)
        (def audio (+ 0.1 (* 0.7 (read-history h))))
        (write-history h audio)
        (out (* audio \(width == 1 ? "coefficient" : "(sum coefficient)")) 1)
        """
      let baseline = try render(program.replacingOccurrences(of: "HOLD", with: "latch"), parts: [128])
      for parts in [[128], [1], [3, 17, 64, 7]] {
        let actual = try render(program.replacingOccurrences(of: "HOLD", with: "event-hold"), parts: parts)
        XCTAssertEqual(actual.count, baseline.count)
        for (frame, pair) in zip(actual, baseline).enumerated() {
          XCTAssertEqual(pair.0, pair.1, accuracy: 0.00001,
            "width \(width), parts \(parts), frame \(frame)")
        }
      }
    }
  }

  func testTensorEventsZeroFillAndHoldAtAdjacentEvents() throws {
    let source = """
      (def position (accum 1 0 0 1024))
      (def event (max (eq position 5) (eq position 6) (eq position 127) (eq position 128)))
      (def values (event-hold (* (+ position 1) (tensor @shape [3] @data [1 2 3])) event))
      (out (sum (* values 2)) 1)
      """
    for parts in [[128], [1], [7, 31, 3]] {
      let actual = try render(source, parts: parts)
      for (frame, value) in actual.enumerated() {
        let expected: Float = [5, 6, 127, 128].contains(frame) ? Float((frame + 1) * 12) : 0
        XCTAssertEqual(value, expected, accuracy: 0.0001, "frame \(frame), parts \(parts)")
      }
    }
  }

  func testEventCoefficientsPreserveCoupledScalarFeedback() throws {
    let source = """
      (defmacro pole (input coefficient)
        (make-history previous)
        (def value (mix input (read-history previous) coefficient))
        (write-history previous value)
        value)
      (def position (accum 1 0 0 1024))
      (def event (eq (% position 16) 0))
      (def control (HOLD (+ 0.4 (* 0.01 (sin position))) event))
      (def coefficient (latch (exp (- control)) event))
      (def force (pole (pole (sin position) coefficient) coefficient))
      (make-history a)
      (make-history b)
      (def period (latch (+ 8 (* control 2)) event))
      (def arrived_a (delay (read-history a) period))
      (def arrived_b (delay (read-history b) (+ period 3)))
      (def damping (latch (exp (* -0.02 control)) event))
      (def next_a (+ force (* damping arrived_b)))
      (def next_b (- (* damping arrived_a) force))
      (write-history a next_a)
      (write-history b next_b)
      (out (+ arrived_a arrived_b force) 1)
      """
    let baseline = try render(source.replacingOccurrences(of: "HOLD", with: "latch"), parts: [1])
    for parts in [[128], [1], [7, 31, 3]] {
      let audio = try render(source.replacingOccurrences(of: "HOLD", with: "event-hold"), parts: parts)
      for (frame, pair) in zip(audio, baseline).enumerated() {
        XCTAssertEqual(pair.0, pair.1, accuracy: 0.00002, "frame \(frame), parts \(parts)")
      }
    }
  }

  func testIndependentClocksRequireExplicitAudioBoundary() throws {
    let declarations = """
      (def position (accum 1 0 0 1024))
      (def a_tick (eq (% position 3) 0))
      (def b_tick (eq (% position 5) 1))
      (def a (event-hold position a_tick))
      (def b (event-hold position b_tick))
      """
    XCTAssertThrowsError(try compile(declarations + "(out (+ a b) 1)")) { error in
      XCTAssertTrue(String(describing: error).contains("Independent event/hop clocks"))
    }
    let audio = try render(declarations + "(out (+ (latch a a_tick) (latch b b_tick)) 1)", parts: [7, 19])
    var a: Float = 0
    var b: Float = 0
    for (frame, value) in audio.enumerated() {
      if frame % 3 == 0 { a = Float(frame) }
      if frame % 5 == 1 { b = Float(frame) }
      XCTAssertEqual(value, a + b)
    }
  }

  func testFrameLocalTensorScratchPreservesCoupledFeedbackAndRateChanges() throws {
    let source = """
      (def position (accum 1 0 0 1024))
      (def tick (eq (% position 16) 0))
      (def held (event-hold (+ 0.5 (* position 0.0005)) tick))
      (def coefficients (latch (* held (tensor @shape [4] @data [1 0.9 0.8 0.7])) tick))
      (make-history drive)
      (make-tensor-history modes @shape [4])
      (def force (+ 0.1 (* 0.01 (read-history drive))))
      (def next (+ (* force (tensor @shape [4] @data [1 2 3 4]))
        (* coefficients (read-tensor-history modes))))
      (write-tensor-history modes next)
      (def total (sum next))
      (write-history drive total)
      (out total 1)
      """
    for parts in [[128], [1], [7, 31, 3]] {
      let audio = try render(source, parts: parts)
      var state = [Float](repeating: 0, count: 4)
      var drive: Float = 0
      for (frame, sample) in audio.enumerated() {
        let held = 0.5 + Float((frame / 16) * 16) * 0.0005
        let force = 0.1 + 0.01 * drive
        for lane in 0..<4 {
          state[lane] = force * Float(lane + 1)
            + held * (1 - Float(lane) * 0.1) * state[lane]
        }
        drive = state.reduce(0, +)
        XCTAssertEqual(sample, drive, accuracy: 0.00001, "frame \(frame), parts \(parts)")
      }
    }
  }
  func testUnsupportedBackendsAndGradientsFailBeforeCodeGeneration() throws {
    let result = try compile("(out (event-hold (in 1) (in 2)) 1)")
    XCTAssertThrowsError(try CompilationPipeline.compile(graph: result.graph, backend: .metal)) {
      XCTAssertTrue(String(describing: $0).contains("event-hold currently requires the C backend"))
    }
    let source = try XCTUnwrap(result.graph.nodes.values.first { if case .input(0) = $0.op { return true }; return false })
    _ = result.graph.computeGradients(loss: source.id, targets: [source.id])
    XCTAssertThrowsError(try CompilationPipeline.compile(graph: result.graph, backend: .c)) {
      XCTAssertTrue(String(describing: $0).contains("event-hold does not yet support automatic differentiation"))
    }
  }

}
