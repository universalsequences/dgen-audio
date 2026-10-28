import Foundation
import XCTest

@testable import DGen
@testable import DGenLazy
@testable import DGenLisp

/// Behavioral tests for exclusive scalar execution regions and inactive modulation.
final class ExecutionGateTests: XCTestCase {
  private var tempDir: URL!
  private let toggles = ["DGEN_NO_DCE", "DGEN_NO_STATIC_HOIST", "DGEN_NO_COALESCE"]
  private var savedConfig: (backend: Backend, sampleRate: Float, maxFrameCount: Int)!

  override func setUpWithError() throws {
    try super.setUpWithError()
    savedConfig = (DGenConfig.backend, DGenConfig.sampleRate, DGenConfig.maxFrameCount)
    DGenConfig.backend = .c
    DGenConfig.sampleRate = 48000
    DGenConfig.maxFrameCount = 64
    for key in toggles { unsetenv(key) }
    tempDir = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
      .appendingPathComponent("dgenlisp-perfpass-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
  }

  override func tearDownWithError() throws {
    for key in toggles { unsetenv(key) }
    // Other suites read these globals without setting them; leave no trace.
    DGenConfig.backend = savedConfig.backend
    DGenConfig.sampleRate = savedConfig.sampleRate
    DGenConfig.maxFrameCount = savedConfig.maxFrameCount
    LazyGraphContext.reset()
    if let tempDir { try? FileManager.default.removeItem(at: tempDir) }
    try super.tearDownWithError()
  }

  private struct Compiled {
    let result: CompilationResult
    let source: String
    let paramCells: [String: Int]
    let paramDefaults: [Int: Float]
    let nodeCount: Int
  }

  private func compile(_ source: String, blockSize: Int = 64, voiceCount: Int = 1) throws -> Compiled {
    DGenConfig.maxFrameCount = blockSize
    LazyGraphContext.reset()
    let lazy = LazyGraphContext.current
    let evaluator = LispEvaluator(sourceDirectory: tempDir)
    try evaluator.evaluate(nodes: lowerModulation(in: parseSource(source)))
    for output in evaluator.outputs {
      lazy.addOutput(output.signal, channel: output.channel)
    }
    let result = try CompilationPipeline.compile(
      graph: lazy.graph, backend: .c, options: .init(frameCount: blockSize, debug: false, voiceCount: voiceCount))
    var cells: [String: Int] = [:]
    var defaults: [Int: Float] = [:]
    for param in evaluator.params {
      if let logical = param.cellId {
        let physical = result.cellAllocations.cellMappings[logical] ?? logical
        cells[param.name] = physical
        defaults[physical] = param.defaultValue
      }
    }
    return Compiled(
      result: result,
      source: result.kernels.map { $0.source }.joined(separator: "\n\n"),
      paramCells: cells,
      paramDefaults: defaults,
      nodeCount: lazy.graph.nodes.count)
  }

  /// Renders `blocks` blocks of `blockSize` frames, calling `beforeBlock` with
  /// the memory pointer before each so tests can move parameters mid-stream.
  private func render(
    _ compiled: Compiled, blockSize: Int = 64, blocks: Int = 4, input: Float = 0, inputChannels: [Float]? = nil,
    beforeBlock: (Int, UnsafeMutablePointer<Float>) -> Void = { _, _ in }
  ) throws -> [Float] {
    let kernel = CCompiledKernel(
      source: compiled.source, cellAllocations: compiled.result.cellAllocations,
      memorySize: compiled.result.totalMemorySlots, defaultHostSampleRate: 48000)
    try kernel.compileAndLoad()
    defer { kernel.cleanup() }
    let mem = UnsafeMutablePointer<Float>.allocate(capacity: max(1024, compiled.result.totalMemorySlots))
    mem.initialize(repeating: 0, count: max(1024, compiled.result.totalMemorySlots))
    let out = UnsafeMutablePointer<Float>.allocate(capacity: blockSize + 4)
    out.initialize(repeating: 0, count: blockSize + 4)
    let inputs = (inputChannels ?? [input]).map { value -> UnsafeMutablePointer<Float> in
      let buffer = UnsafeMutablePointer<Float>.allocate(capacity: blockSize + 4)
      buffer.initialize(repeating: value, count: blockSize + 4)
      return buffer
    }
    defer { mem.deallocate(); out.deallocate(); inputs.forEach { $0.deallocate() } }
    let process = try XCTUnwrap(kernel.getProcessFunction())
    let inputPointers: [UnsafePointer<Float>?] = inputs.map { UnsafePointer($0) }
    let outputPointers: [UnsafeMutablePointer<Float>?] = [out]
    var context = DGenProcessContextV1(sampleRate: 48000)
    DGen.injectTensorData(result: compiled.result, memory: mem)
    for (cell, value) in compiled.paramDefaults { mem[cell] = value }
    var samples: [Float] = []
    for b in 0..<blocks {
      beforeBlock(b, mem)
      inputPointers.withUnsafeBufferPointer { ip in
        outputPointers.withUnsafeBufferPointer { op in
          withUnsafePointer(to: &context) { contextPointer in
            process(ip.baseAddress, op.baseAddress, UInt32(blockSize), mem,
              UnsafeRawPointer(contextPointer), nil)
          }
        }
      }
      for i in 0..<blockSize { samples.append(out[i]) }
    }
    return samples
  }

  // MARK: - sample-gate

  /// A decaying voice whose gate condition reads the voice's own envelope: the
  /// body runs on live frames only, so a counter inside it counts live frames.
  func testSampleGateConditionReadsTheBodysOwnState() throws {
    let program = try compile("""
      (def trig (eq (accum 1 0 0 40) 0))
      (make-history h_env)
      (def alive (or trig (gt (read-history h_env) 0)))
      (def (e c)
        (sample-gate alive
          (do
            (make-history h_count)
            (def e0 (gswitch trig 1 (* (read-history h_env) 0.5)))
            (def e1 (gswitch (lt e0 0.001) 0 e0))
            (write-history h_env e1)
            (def c (+ (read-history h_count) 1))
            (write-history h_count c)
            (tuple e1 c))))
      (out (+ c (* 0 e)) 1)
      """)
    for size in [1, 16, 64] {
      let output = try render(program, blockSize: size, blocks: 400 / size + 1)
      var env: Float = 0, count: Float = 0
      for (frame, value) in output.enumerated() {
        let trig = frame % 40 == 0
        let alive = trig || env > 0
        var expected: Float = 0
        if alive {
          var e = trig ? 1 : env * 0.5
          if e < 0.001 { e = 0 }
          env = e
          count += 1
          expected = count
        }
        XCTAssertEqual(value, expected, "frame \(frame), block \(size)")
      }
    }
  }

  /// State written outside a sample-gate keeps running while the gate is
  /// closed, even when only the gated body reads it.
  func testSampleGateLeavesOutsideStateRunning() throws {
    let program = try compile("""
      (make-history h_frames)
      (def frames (read-history h_frames))
      (write-history h_frames (+ frames 1))
      (def open (eq (% frames 8) 0))
      (out (sample-gate open frames) 1)
      """)
    let output = try render(program, blockSize: 16, blocks: 6)
    for (frame, value) in output.enumerated() {
      XCTAssertEqual(value, frame % 8 == 0 ? Float(frame) : 0, "frame \(frame)")
    }
  }

  /// A nested gate's state belongs to the nested gate: it advances only when
  /// both gates are open.
  func testNestedSampleGateOwnsItsState() throws {
    let program = try compile("""
      (make-history h_frames)
      (def frames (read-history h_frames))
      (write-history h_frames (+ frames 1))
      (def outer (lt (% frames 10) 6))
      (def inner (eq (% frames 2) 0))
      (out (sample-gate outer
             (do
               (make-history h_outer)
               (write-history h_outer (+ (read-history h_outer) 1))
               (+ (* 100 (read-history h_outer))
                  (sample-gate inner
                    (do
                      (make-history h_inner)
                      (def n (+ (read-history h_inner) 1))
                      (write-history h_inner n)
                      n)))))
           1)
      """)
    let output = try render(program, blockSize: 8, blocks: 5)
    var outerCount: Float = 0, innerCount: Float = 0
    for (frame, value) in output.enumerated() {
      var expected: Float = 0
      if frame % 10 < 6 {
        let before = outerCount
        outerCount += 1
        var inner: Float = 0
        if frame % 2 == 0 { innerCount += 1; inner = innerCount }
        expected = 100 * before + inner
      }
      XCTAssertEqual(value, expected, "frame \(frame)")
    }
  }

  func testScalarLoopCompilesOneFrameLoop() throws {
    let program = try compile("""
      (scalar-loop)
      (def x (in 1))
      (make-history h)
      (def y (+ (* (read-history h) 0.5) x))
      (write-history h y)
      (out (+ (* x 3) (sin y)) 1)
      """)
    let loops = program.source.components(separatedBy: "for (int i = 0; i < frameCount;").count - 1
    XCTAssertEqual(loops, 1)
  }

  /// A ring index already wrapped into the buffer is read and written without
  /// another modulo.
  func testInRangeRingIndexSkipsTheWrap() throws {
    let program = try compile("""
      (def buf (tensor @shape [64 1]))
      (make-history h)
      (def next (+ (clip (read-history h) 0 63) 63))
      (def wp (gswitch (gte next 64) (- next 64) next))
      (write-history h wp)
      (def ahead (+ wp 9))
      (def r (gswitch (gte ahead 64) (- ahead 64) ahead))
      (def w (poke buf wp 0 wp))
      (out (seq w (peek buf r 0)) 1)
      """)
    XCTAssertFalse(program.source.contains("/ 64.0f"), "a wrapped index needs no modulo")
    // The write pointer steps back one slot a frame; reading 9 slots ahead
    // returns the pointer value written 9 frames earlier.
    let output = try render(program, blockSize: 32, blocks: 4)
    for (frame, value) in output.enumerated() where frame >= 10 {
      let wpAt: (Int) -> Float = { f in Float(((63 - f) % 64 + 64) % 64) }
      XCTAssertEqual(value, wpAt(frame - 9), "frame \(frame)")
    }
  }

  /// Two delay lines of one length delay independently (clamped read path).
  func testDelayLinesOfOneLengthDelayIndependently() throws {
    let program = try compile("""
      (def ramp (accum 1 0 0 100000))
      (out (+ (delay ramp 3 @max-delay 64) (* 1000 (delay (* ramp 2) 5 @max-delay 64))) 1)
      """)
    let output = try render(program, blockSize: 16, blocks: 12)
    for (frame, value) in output.enumerated() {
      let a: Float = frame >= 3 ? Float(frame - 3) : 0
      let b: Float = frame >= 5 ? Float(2 * (frame - 5)) : 0
      XCTAssertEqual(value, a + 1000 * b, accuracy: 0.01, "frame \(frame)")
    }
  }

  func testModulationRegionPreservesDelayRingWrap() throws {
    for size in [8, 64] {
      let program = try compile("""
        (def signal (in 1 @name signal @modulator 1))
        (param gain @default 0.5 @min 0 @max 1 @mod true @mod-mode additive)
        (out (delay (* signal (mod gain)) 18.4) 1)
        """, blockSize: size)
      // Read across the ring boundary at startup and again after its 88000
      // samples wrap. The ring counter follows the buffer and is not audio.
      let output = try render(program, blockSize: size,
        blocks: 89000 / size + 1, input: 1) { _, _ in }
      for (frame, value) in output.enumerated() {
        let expected: Float = frame < 18 ? 0 : (frame == 18 ? 0.3 : 0.5)
        XCTAssertEqual(value, expected, accuracy: 0.003, "frame \(frame), block \(size)")
      }
    }
  }

  func testModulationSplitKeepsScalarPoleOutsideTensorLoop() throws {
    for size in [1, 8, 64] {
      let program = try compile("""
        (def mod1 (in 1 @name mod1 @modulator 1))
        (param pole @default 0.9 @min 0 @max 1 @mod true @mod-mode additive)
        (def weights (tensor @shape [8] @data [1 2 3 4 5 6 7 8]))
        (def tick (eq (accum 1 0 0 16) 0))
        (def coefficients (* weights (hop-hold (mod pole) 16)))
        (make-history h)
        (def value (+ (* (read-history h) (mod pole)) 0.1))
        (write-history h value)
        (out (+ value (sum (* value (latch coefficients tick)))) 1)
        """, blockSize: size)
      let output = try render(program, blockSize: size, blocks: 8)
      var value: Float = 0
      for (frame, actual) in output.enumerated() {
        value = value * 0.9 + 0.1
        XCTAssertEqual(actual, value * (1 + 36 * 0.9), accuracy: 0.0001,
          "scalar pole must advance once at frame \(frame), block \(size)")
      }
    }
  }

  func testGateReadsLeafTableAndFreezesOnlyItsScalarHistory() throws {
    for size in [1, 8, 64] {
      let program = try compile("""
        (param enabled @default 1 @min 0 @max 1)
        (def bank (tensor @shape [2 2] @data [1 3 5 7]))
        (make-history counter)
        (def count (+ (read-history counter) 1))
        (write-history counter count)
        (out (block-gate enabled (* count (peek bank (in 1) (in 2)))) 1)
        """, blockSize: size)
      let output = try render(program, blockSize: size, blocks: 4,
        inputChannels: [0.5, 0.5]) { block, memory in
        memory[program.paramCells["enabled"]!] = block == 1 || block == 2 ? 0 : 1
      }
      // Bilinear interpolation is 4; the table survives disabled blocks while
      // the counter resumes at precisely its prior value.
      XCTAssertEqual(Array(output.prefix(size)), (1...size).map { Float($0) * 4 })
      XCTAssertEqual(Array(output[size..<(3*size)]), Array(repeating: 0, count: 2*size))
      XCTAssertEqual(Array(output.suffix(size)), ((size+1)...(2*size)).map { Float($0) * 4 })
    }
  }

  func testGateFreezesAndResumesHistory() throws {
    for size in [1, 8, 64] {
      let program = try compile("""
        (param enabled @default 1 @min 0 @max 1)
        (make-history counter)
        (def count (+ (read-history counter) 1))
        (write-history counter count)
        (out (block-gate enabled count) 1)
        """, blockSize: size)
      let output = try render(program, blockSize: size, blocks: 4) { b, mem in
        mem[program.paramCells["enabled"]!] = b == 1 || b == 2 ? 0 : 1
      }
      XCTAssertEqual(Array(output.prefix(size)), (1...size).map(Float.init))
      XCTAssertEqual(Array(output[size..<(3*size)]), Array(repeating: 0, count: 2*size))
      XCTAssertEqual(Array(output.suffix(size)), ((size+1)...(2*size)).map(Float.init))
      XCTAssertTrue(program.source.contains("if ("))
    }
  }

  func testGateInsideFeedbackPreservesOneSampleDelay() throws {
    for size in [1, 8, 64] {
      let program = try compile("""
        (param enabled @default 1)
        (make-history feedback)
        (def value (block-gate enabled (+ (read-history feedback) 1)))
        (write-history feedback value)
        (out value 1)
        """, blockSize: size)
      XCTAssertEqual(try render(program, blockSize: size, blocks: 4),
        (1...(4*size)).map(Float.init))
      let switched = try render(program, blockSize: size, blocks: 4) { block, memory in
        memory[program.paramCells["enabled"]!] = block == 1 || block == 2 ? 0 : 1
      }
      XCTAssertEqual(switched, (1...size).map(Float.init)
        + Array(repeating: 0, count: 2*size)
        + ((size+1)...(2*size)).map(Float.init))
    }
  }

  func testSharedHistoryRemainsLiveForUngatedConsumer() throws {
    let program = try compile("""
      (param enabled @default 0 @min 0 @max 1)
      (make-history counter)
      (def count (+ (read-history counter) 1))
      (write-history counter count)
      (out (+ (block-gate enabled count) count) 1)
      """)
    XCTAssertEqual(try render(program), (1...256).map(Float.init))
  }

  func testTwoGatesShareProducerWithUnionOfDemand() throws {
    let program = try compile("""
      (param a @default 0 @min 0 @max 1)
      (param b @default 0 @min 0 @max 1)
      (make-history counter)
      (def count (+ (read-history counter) 1))
      (write-history counter count)
      (out (+ (block-gate a count) (block-gate b count)) 1)
      """)
    let output = try render(program) { block, mem in
      mem[program.paramCells["a"]!] = block == 1 ? 1 : 0
      mem[program.paramCells["b"]!] = block == 2 ? 1 : 0
    }
    XCTAssertEqual(Array(output[64..<192]), (1...128).map(Float.init))
    XCTAssertEqual(Array(output.prefix(64)), Array(repeating: 0, count: 64))
    XCTAssertEqual(Array(output.suffix(64)), Array(repeating: 0, count: 64))
  }

  func testAudioConditionRunsWholeBlockButMasksEachSample() throws {
    let program = try compile("""
      (make-history phase)
      (def position (% (+ (read-history phase) 1) 8))
      (write-history phase position)
      (make-history counter)
      (def count (+ (read-history counter) 1))
      (write-history counter count)
      (out (block-gate (lt position 4) count) 1)
      """, blockSize: 8)
    let output = try render(program, blockSize: 8, blocks: 2)
    XCTAssertEqual(output, (1...16).map { $0 % 8 < 4 ? Float($0) : 0 })
  }

  func testGateRejectsTensorBodyAndMetalBackend() throws {
    XCTAssertThrowsError(try compile("(out (block-gate 0 (sum (tensor 1 2 3))) 1)"))
    XCTAssertThrowsError(try compile("""
      (param enabled @default 1)
      (def bank (tensor @shape [2] @data [1 3]))
      (out (block-gate enabled (peek (+ bank bank) 0)) 1)
      """))
    _ = try compile("(param enabled @default 1) (out (block-gate enabled (phasor 100)) 1)")
    XCTAssertThrowsError(try CompilationPipeline.compile(
      graph: LazyGraphContext.current.graph, backend: .metal, options: .init(frameCount: 64)))
  }
  func testNestedGateFreezesUntilBothConditionsAreEnabled() throws {
    let program = try compile("""
      (param a @default 0) (param b @default 0)
      (make-history h)
      (def count (+ (read-history h) 1)) (write-history h count)
      (out (block-gate a (block-gate b count)) 1)
      """)
    let output = try render(program) { block, mem in
      mem[program.paramCells["a"]!] = block >= 1 ? 1 : 0
      mem[program.paramCells["b"]!] = block >= 2 ? 1 : 0
    }
    XCTAssertEqual(Array(output.prefix(128)), Array(repeating: 0, count: 128))
    XCTAssertEqual(Array(output.suffix(128)), (1...128).map(Float.init))
  }

  func testModulationAssignmentChangesPreserveModesAndBase() throws {
    for (mode, expected) in [("additive", Float(112)), ("multiplicative", 1300), ("semitone", 200)] {
      for voices in [1, 12] {
        let program = try compile("""
          (def mod1 (in 1 @name mod1 @modulator 1))
          (def mod2 (in 2 @name mod2 @modulator 2))
          (def mod3 (in 3 @name mod3 @modulator 3))
          (def mod4 (in 4 @name mod4 @modulator 4))
          (param level @default 100 @min 1 @max 2000 @mod true @mod-mode \(mode))
          (out (mod level) 1)
          """, voiceCount: voices)
        let output = try render(program, inputChannels: [0.5, -0.25, 0.75, -1]) { b, mem in
          mem[program.paramCells["__mod__level__active"]!] = b % 2 == 1 ? 1 : 0
          // Four independently valued sources sum to 12 semitones/units.
          for (slot, depth) in [Float(2), 4, 8, -6].enumerated() {
            mem[program.paramCells["__mod__level__depth__slot\(slot + 1)"]!] = depth
          }
        }
        for b in 0..<4 {
          for sample in output[(b*64)..<((b+1)*64)] {
            XCTAssertEqual(sample, b % 2 == 1 ? expected : 100, accuracy: 0.0001)
          }
        }
      }
    }
  }

  func testRecompilingGatedGraphRestoresDependencies() throws {
    let first = try compile("""
      (param enabled @default 1)
      (make-history h) (def x (+ (read-history h) 1)) (write-history h x)
      (out (block-gate enabled x) 1)
      """)
    let graph = LazyGraphContext.current.graph
    let dependencies = graph.nodes.mapValues(\.temporalDependencies)
    let second = try CompilationPipeline.compile(graph: graph, backend: .c, options: .init(frameCount: 64))
    XCTAssertEqual(dependencies, graph.nodes.mapValues(\.temporalDependencies))
    XCTAssertEqual(first.source, second.kernels.map(\.source).joined(separator: "\n\n"))
  }

  func testSharedPredicateDependenciesAreScheduledBeforeGatedBody() throws {
    let program = try compile("""
      (param enabled @default 1)
      (def shared (/ (in 1) 2))
      (make-history h) (def count (+ (read-history h) 1)) (write-history h count)
      (out (block-gate enabled (+ shared (block-gate (gt shared 0) count))) 1)
      """)
    XCTAssertEqual(try render(program, input: 2), (2...257).map(Float.init))
  }

  func testModulationLoweringIsScopedToCompilation() throws {
    _ = try compile("""
      (def mod1 (in 1 @modulator 1))
      (param level @default 100 @min 1 @max 2000 @mod true @mod-mode additive)
      (out (mod level) 1)
      """)
    let graph = LazyGraphContext.current.graph
    let originalIds = Set(graph.nodes.keys)
    let originalGates = graph.executionGates
    XCTAssertTrue(graph.nodes.values.contains { if case .modulatedParam = $0.op { return true }; return false })
    _ = try CompilationPipeline.compile(graph: graph, backend: .c, options: .init(frameCount: 64))
    XCTAssertEqual(Set(graph.nodes.keys), originalIds)
    XCTAssertEqual(graph.executionGates, originalGates)
  }

}
