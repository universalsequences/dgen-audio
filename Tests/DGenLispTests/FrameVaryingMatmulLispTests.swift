import Foundation
import XCTest

@testable import DGen
@testable import DGenLazy
@testable import DGenLisp

/// `matmul` and broadcast views (transpose / reshape / expand) over a
/// FRAME-VARYING tensor. The axis reduction that fuses the broadcast multiply
/// read a frame-aware operand through `tensor.strides`, which are the base
/// strides, so the view's extra dims were dropped: `x @ W` read `x[i][0]` for
/// every k and gave [11 22 33] for [1 100] @ [[1 2 3] [10 20 30]]. Frame-
/// invariant operands (hoisted out of the frame loop) were unaffected.
final class FrameVaryingMatmulLispTests: XCTestCase {
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

  /// Renders `frames` samples of output channel 1 in 128-frame blocks.
  private func render(_ source: String, frames: Int = 256) throws -> [Float] {
    LazyGraphContext.reset()
    let evaluator = LispEvaluator()
    try evaluator.evaluate(nodes: parseSource(source))
    for output in evaluator.outputs {
      LazyGraphContext.current.addOutput(output.signal, channel: output.channel)
    }
    let result = try CompilationPipeline.compile(
      graph: LazyGraphContext.current.graph, backend: .c,
      options: .init(frameCount: 128, debug: false))
    let kernel = CCompiledKernel(
      source: result.kernels.map { $0.source }.joined(separator: "\n\n"),
      cellAllocations: result.cellAllocations, memorySize: result.totalMemorySlots,
      defaultHostSampleRate: 48000)
    try kernel.compileAndLoad()
    defer { kernel.cleanup() }
    let count = max(1024, result.totalMemorySlots)
    let memory = UnsafeMutablePointer<Float>.allocate(capacity: count)
    memory.initialize(repeating: 0, count: count)
    let output = UnsafeMutablePointer<Float>.allocate(capacity: 128)
    output.initialize(repeating: 0, count: 128)
    defer { memory.deallocate(); output.deallocate() }
    DGen.injectTensorData(result: result, memory: memory)
    let process = try XCTUnwrap(kernel.getProcessFunction())
    let outputs: [UnsafeMutablePointer<Float>?] = [output]
    var context = DGenProcessContextV1(sampleRate: 48000)
    var audio: [Float] = []
    while audio.count < frames {
      let n = min(128, frames - audio.count)
      outputs.withUnsafeBufferPointer { op in
        withUnsafePointer(to: &context) { cp in
          process(nil, op.baseAddress, UInt32(n), memory, UnsafeRawPointer(cp), nil)
        }
      }
      audio.append(contentsOf: UnsafeBufferPointer(start: output, count: n))
    }
    return audio
  }

  /// x(t) = [a(t) b(t)] with a = t mod 64 + 1, b = 100 - a: both vary per frame.
  /// y = <FORM> must equal x @ W; the output weighs its three columns by
  /// [1 0.5 0.25] so a dropped or misplaced element shows up in one number.
  private func program(_ form: String) -> String {
    """
    (def t (accum 1 0 0 64))
    (def a (+ t 1))
    (def b (- 100 a))
    (def e0 (tensor @shape [1 2] @data [1 0]))
    (def e1 (tensor @shape [1 2] @data [0 1]))
    (def x (+ (* a e0) (* b e1)))
    (def w (tensor @shape [2 3] @data [1 2 3 10 20 30]))
    (def y \(form))
    (out (sum (* y (tensor @shape [1 3] @data [1 0.5 0.25]))) 1)
    """
  }

  /// Reference: (a*[1 2 3] + b*[10 20 30]) . [1 0.5 0.25] = 2.75 * (a + 10 b).
  private func expected(_ a: Float) -> Float { 2.75 * (a + 10 * (100 - a)) }

  private func assertMatmul(_ form: String, file: StaticString = #filePath, line: UInt = #line)
    throws
  {
    let audio = try render(program(form))
    for (i, v) in audio.enumerated() {
      let a = Float(i % 64 + 1)
      XCTAssertEqual(v, expected(a), accuracy: 1e-2, "\(form) frame \(i)", file: file, line: line)
      if abs(v - expected(a)) > 1e-2 { return }
    }
  }

  func testMatmulOfFrameVaryingLeftOperand() throws {
    try assertMatmul("(matmul x w)")
  }

  func testMatmulOfFrameVaryingRightOperand() throws {
    // (W^T @ x^T)^T: the frame-varying operand is the transposed right side.
    try assertMatmul("(transpose (matmul (transpose w) (transpose x)))")
  }

  func testTransposeBroadcastSumAxisOfFrameVaryingTensor() throws {
    try assertMatmul(
      "(reshape (sum-axis (* (transpose x) w) @axis 0) @shape [1 3])")
  }

  func testReshapeBroadcastSumAxisOfFrameVaryingTensor() throws {
    try assertMatmul(
      "(reshape (sum-axis (* (reshape x @shape [2 1]) w) @axis 0) @shape [1 3])")
  }

  func testExpandBroadcastSumAxisOfFrameVaryingTensor() throws {
    try assertMatmul(
      "(reshape (sum-axis (* (expand (reshape x @shape [2 1]) @shape [2 3]) w) @axis 0) @shape [1 3])"
    )
  }

  func testTwoLayerMatmulOfFrameVaryingTensor() throws {
    // x @ I @ W: the hidden layer is itself a frame-varying matmul result.
    try assertMatmul(
      "(matmul (matmul x (tensor @shape [2 2] @data [1 0 0 1])) w)")
  }

  func testMatmulOfHopHeldTensor() throws {
    // Hop-gated block: the tensor tape is indexed by i / hop, and the output
    // is written on hop frames. Check those frames.
    let source = program("(matmul x w)").replacingOccurrences(
      of: "(def a (+ t 1))", with: "(def a (hop-hold (+ t 1) 16))")
    let audio = try render(source)
    for i in stride(from: 0, to: audio.count, by: 16) {
      XCTAssertEqual(audio[i], expected(Float(i % 64 + 1)), accuracy: 1e-2, "frame \(i)")
    }
  }
}
