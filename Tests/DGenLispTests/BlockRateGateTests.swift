import Foundation
import XCTest

@testable import DGen
@testable import DGenLazy
@testable import DGenLisp

/// Block-rate coefficient math and delay lines inside block-gated regions.
final class BlockRateGateTests: XCTestCase {
  private var savedConfig: (Backend, Float, Int)!
  private let toggles = ["DGEN_DISABLE_EVENT_PROMOTION"]

  override func setUpWithError() throws {
    savedConfig = (DGenConfig.backend, DGenConfig.sampleRate, DGenConfig.maxFrameCount)
    DGenConfig.backend = .c
    DGenConfig.sampleRate = 48000
    DGenConfig.maxFrameCount = 128
    for key in toggles { unsetenv(key) }
  }

  override func tearDownWithError() throws {
    for key in toggles { unsetenv(key) }
    (DGenConfig.backend, DGenConfig.sampleRate, DGenConfig.maxFrameCount) = savedConfig
    LazyGraphContext.reset()
  }

  private struct Program {
    let result: CompilationResult
    let paramCells: [String: Int]
    let paramDefaults: [Int: Float]
  }

  private func compile(_ source: String) throws -> Program {
    LazyGraphContext.reset()
    let evaluator = LispEvaluator()
    try evaluator.evaluate(nodes: lowerModulation(in: parseSource(source)))
    for output in evaluator.outputs {
      LazyGraphContext.current.addOutput(output.signal, channel: output.channel)
    }
    let result = try CompilationPipeline.compile(graph: LazyGraphContext.current.graph,
      backend: .c, options: .init(frameCount: 128, debug: false))
    var cells: [String: Int] = [:]
    var defaults: [Int: Float] = [:]
    for param in evaluator.params {
      if let logical = param.cellId {
        let physical = result.cellAllocations.cellMappings[logical] ?? logical
        cells[param.name] = physical
        defaults[physical] = param.defaultValue
      }
    }
    return Program(result: result, paramCells: cells, paramDefaults: defaults)
  }

  /// Renders 512 frames in calls of `parts` frames; `beforeCall(call, frame, memory)`
  /// may move parameters between calls.
  private func render(
    _ program: Program, parts: [Int],
    beforeCall: (Int, Int, UnsafeMutablePointer<Float>, [String: Int]) -> Void = { _, _, _, _ in }
  ) throws -> [Float] {
    let result = program.result
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
    for (cell, value) in program.paramDefaults { memory[cell] = value }
    let process = try XCTUnwrap(kernel.getProcessFunction())
    let outputs: [UnsafeMutablePointer<Float>?] = [output]
    var context = DGenProcessContextV1(sampleRate: 48000)
    var audio: [Float] = []
    var call = 0
    while audio.count < 512 {
      let frames = min(parts[call % parts.count], 512 - audio.count)
      beforeCall(call, audio.count, memory, program.paramCells)
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

  private func assertClose(
    _ actual: [Float], _ expected: [Float], accuracy: Float = 0.00002, _ label: String,
    file: StaticString = #filePath, line: UInt = #line
  ) {
    XCTAssertEqual(actual.count, expected.count, label, file: file, line: line)
    for (frame, pair) in zip(actual, expected).enumerated()
    where abs(pair.0 - pair.1) > accuracy {
      XCTFail("\(label): frame \(frame) got \(pair.0), expected \(pair.1)", file: file, line: line)
      return
    }
  }

  /// Latched block-rate coefficient math (promoted to event rate) may feed a
  /// gated body. When the gate reopens, the coefficients are current even
  /// though the latch's trigger fired while the gate was closed.
  func testPromotedCoefficientsFeedGatedBodyAndRefreshOnReopen() throws {
    let source = """
      (param enabled @default 1 @min 0 @max 1)
      (param pitch @default 0.3 @min 0 @max 1)
      (def position (accum 1 0 0 1024))
      (def tick (eq (% position 16) 0))
      (def held (latch (+ pitch (* position 0.001)) tick))
      (def g (tan (* 0.4 (exp (* -1 held)))))
      (def coefficient (/ g (+ 1 (* g (+ g 0.7)))))
      (make-history h)
      (def audio (+ 0.1 (* 0.9 (read-history h))))
      (write-history h audio)
      (out (block-gate enabled (* audio (+ coefficient (sin held)))) 1)
      """
    func toggle(call: Int, frame: Int, memory: UnsafeMutablePointer<Float>, cells: [String: Int]) {
      // Closed across several ticks, reopened mid-interval, pitch moved while closed.
      memory[cells["enabled"]!] = frame >= 100 && frame < 230 ? 0 : 1
      memory[cells["pitch"]!] = frame >= 150 ? 0.6 : 0.3
    }
    setenv("DGEN_DISABLE_EVENT_PROMOTION", "1", 1)
    let reference = try compile(source)
    unsetenv("DGEN_DISABLE_EVENT_PROMOTION")
    let promoted = try compile(source)
    XCTAssertTrue(promoted.result.uopBlocks.contains { block in
      if case .hopBased = block.temporality { return block.executionDemand != .always }
      return false
    }, "the coefficients should run at event rate inside the gate")
    for parts in [[1], [10, 20, 30], [128]] {
      let baseline = try render(reference, parts: parts, beforeCall: toggle)
      let actual = try render(promoted, parts: parts, beforeCall: toggle)
      assertClose(actual, baseline, "parts \(parts)")
    }
  }

  /// A delay line inside a gated body freezes while the gate is closed:
  /// its write head and contents resume exactly where they stopped.
  func testDelayInsideGatedBodyFreezesAndResumes() throws {
    let source = """
      (param enabled @default 1 @min 0 @max 1)
      (def position (accum 1 0 0 100000))
      (make-history fb)
      (def line (delay (+ (eq position 0) (* 0.5 (read-history fb))) 5 @max-delay 64))
      (write-history fb line)
      (out (block-gate enabled line) 1)
      """
    let program = try compile(source)
    for parts in [[1], [7, 13], [128]] {
      // The parameter is set per call, so record which frames ran gated.
      var enabled = [Bool](repeating: true, count: 512)
      let output = try render(program, parts: parts) { call, frame, memory, cells in
        let on = !(frame >= 12 && frame < 40)
        memory[cells["enabled"]!] = on ? 1 : 0
        let frames = parts[call % parts.count]
        for i in frame..<min(512, frame + frames) { enabled[i] = on }
      }
      // The impulse leaves the line after 5 active frames; the feedback
      // history adds one more per echo, which halves. Gated frames output 0
      // and do not advance the line.
      var step = 0
      for frame in 0..<512 {
        guard enabled[frame] else {
          if output[frame] != 0 { XCTFail("parts \(parts), gated frame \(frame): \(output[frame])"); break }
          continue
        }
        let expected: Float = step % 6 == 5 ? pow(0.5, Float(step / 6)) : 0
        if abs(output[frame] - expected) > 0.00001 {
          XCTFail("parts \(parts), frame \(frame): got \(output[frame]), expected \(expected)")
          break
        }
        step += 1
      }
    }
  }


  /// A `poke`/`peek` ring buffer inside a gated body freezes like `delay`.
  func testPokeRingInsideGatedBodyFreezesAndResumes() throws {
    let source = """
      (param enabled @default 1 @min 0 @max 1)
      (def line (tensor @shape [64]))
      (def position (accum 1 0 0 100000))
      (def write_at (accum 1 0 0 64))
      (make-history fb)
      (def x (+ (eq position 0) (* 0.5 (read-history fb))))
      (def echo (seq (poke line write_at x) (peek line (% (+ write_at 59) 64))))
      (write-history fb echo)
      (out (block-gate enabled echo) 1)
      """
    let program = try compile(source)
    for parts in [[1], [7, 13], [128]] {
      var enabled = [Bool](repeating: true, count: 512)
      let output = try render(program, parts: parts) { call, frame, memory, cells in
        let on = !(frame >= 12 && frame < 40)
        memory[cells["enabled"]!] = on ? 1 : 0
        let frames = parts[call % parts.count]
        for i in frame..<min(512, frame + frames) { enabled[i] = on }
      }
      // Reading 5 slots behind the write head plus the feedback history:
      // echoes every 6 active frames, halving.
      var step = 0
      for frame in 0..<512 {
        guard enabled[frame] else {
          if output[frame] != 0 { XCTFail("parts \(parts), gated frame \(frame): \(output[frame])"); break }
          continue
        }
        let expected: Float = step % 6 == 5 ? pow(0.5, Float(step / 6)) : 0
        if abs(output[frame] - expected) > 0.00001 {
          XCTFail("parts \(parts), frame \(frame): got \(output[frame]), expected \(expected)")
          break
        }
        step += 1
      }
    }
  }
}
