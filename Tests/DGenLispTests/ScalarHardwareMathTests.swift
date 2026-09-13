import Foundation
import XCTest

@testable import DGen
@testable import DGenLazy
@testable import DGenLisp

final class ScalarHardwareMathTests: XCTestCase {
  func testHardwareMathInFeedbackMatchesSignedReferenceAcrossPartitions() throws {
    let saved = (DGenConfig.backend, DGenConfig.sampleRate, DGenConfig.maxFrameCount)
    defer {
      (DGenConfig.backend, DGenConfig.sampleRate, DGenConfig.maxFrameCount) = saved
      LazyGraphContext.reset()
    }
    DGenConfig.backend = .c
    DGenConfig.sampleRate = 48000
    DGenConfig.maxFrameCount = 128
    LazyGraphContext.reset()
    let evaluator = LispEvaluator()
    try evaluator.evaluate(nodes: parseSource("""
      (def x (- (/ (accum 1 0 0 32) 4) 4))
      (make-history h)
      (def y (+ (floor x) (* 3 (ceil x)) (* 7 (round x))
        (* 11 (min x 0.75)) (* 13 (max x -0.75)) (* 17 (abs x))
        (* 19 (sqrt (abs x))) (* 23 (pow (abs x) 0.5))
        (* 0.001 (read-history h))))
      (write-history h y)
      (out y 1)
      """))
    for output in evaluator.outputs {
      LazyGraphContext.current.addOutput(output.signal, channel: output.channel)
    }
    let result = try CompilationPipeline.compile(graph: LazyGraphContext.current.graph,
      backend: .c, options: .init(frameCount: 128, debug: false))
    let source = result.kernels.map { $0.source }.joined(separator: "\n")
    let kernel = CCompiledKernel(source: source, cellAllocations: result.cellAllocations,
      memorySize: result.totalMemorySlots, defaultHostSampleRate: 48000)
    try kernel.compileAndLoad()
    defer { kernel.cleanup() }
    let process = try XCTUnwrap(kernel.getProcessFunction())
    for parts in [[128], [1], [7, 31, 3]] {
      let count = max(1024, result.totalMemorySlots)
      let memory = UnsafeMutablePointer<Float>.allocate(capacity: count)
      memory.initialize(repeating: 0, count: count)
      let output = UnsafeMutablePointer<Float>.allocate(capacity: 132)
      output.initialize(repeating: 0, count: 132)
      defer { memory.deallocate(); output.deallocate() }
      DGen.injectTensorData(result: result, memory: memory)
      var context = DGenProcessContextV1(sampleRate: 48000)
      let outputs: [UnsafeMutablePointer<Float>?] = [output]
      var offset = 0
      var part = 0
      var previous: Float = 0
      while offset < 512 {
        let frames = min(parts[part % parts.count], 512-offset)
        outputs.withUnsafeBufferPointer { op in
          withUnsafePointer(to: &context) { cp in
            process(nil, op.baseAddress, UInt32(frames), memory, UnsafeRawPointer(cp), nil)
          }
        }
        for frame in 0..<frames {
          let x = Float((offset+frame) % 32)/4-4
          let rounded = x.rounded(.down) + 3*x.rounded(.up) + 7*x.rounded(.toNearestOrAwayFromZero)
          let bounded = 11*min(x, 0.75) + 13*max(x, -0.75) + 17*abs(x)
          let expected = rounded + bounded + 42*sqrt(abs(x)) + 0.001*previous
          XCTAssertEqual(output[frame], expected, accuracy: 0.00005,
            "parts \(parts), frame \(offset+frame)")
          previous = expected
        }
        offset += frames
        part += 1
      }
    }
  }
}
