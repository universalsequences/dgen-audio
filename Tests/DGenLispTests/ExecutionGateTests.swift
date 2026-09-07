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
