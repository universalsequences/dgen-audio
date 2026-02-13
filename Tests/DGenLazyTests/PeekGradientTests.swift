import XCTest
import Foundation

@testable import DGenLazy

final class PeekGradientTests: XCTestCase {
  override func setUp() {
    super.setUp()
    DGenConfig.sampleRate = 2000.0
    DGenConfig.debug = false
    DGenConfig.kernelOutputPath = nil
    LazyGraphContext.reset()
  }

  func testPeekChannelBackpropagatesThroughIntermediateTensor() throws {
    let controlFrames = 8
    let numChannels = 4
    let frameCount = 64

    let teacherW = Tensor([[0.9, -0.6, 0.4, 0.2]])
    let teacherB = Tensor([[0.1, 0.2, -0.1, 0.05]])

    let studentW = Tensor.param([1, numChannels], data: [0.3, -0.2, 0.1, 0.0])
    let studentB = Tensor.param([1, numChannels], data: [0.0, 0.0, 0.0, 0.0])
    let optimizer = SGD(params: [studentW, studentB], lr: 0.01)

    func buildLoss() -> Signal {
      let t = (0..<controlFrames).map { [Float($0) / Float(controlFrames - 1)] }
      let time = Tensor(t)

      // student is an intermediate tensor expression, not a direct tensorRef.
      let student = time.matmul(studentW) + studentB
      let teacher = time.matmul(teacherW) + teacherB

      let index = Signal.phasor(DGenConfig.sampleRate / Float(frameCount)) * Float(controlFrames - 1)

      var studentMix = Signal.constant(0.0)
      var teacherMix = Signal.constant(0.0)
      for ch in 0..<numChannels {
        let chSel = Signal.constant(Float(ch))
        let weight = Float(ch + 1)
        studentMix = studentMix + student.peek(index, channel: chSel) * weight
        teacherMix = teacherMix + teacher.peek(index, channel: chSel) * weight
      }

      return mse(studentMix, teacherMix)
    }

    let lossValues = try buildLoss().backward(frames: frameCount)
    let initialLoss = lossValues.reduce(0, +) / Float(frameCount)
    XCTAssertGreaterThan(initialLoss, 0.0)

    let wGradNorm2 = (studentW.grad?.getData() ?? []).map { $0 * $0 }.reduce(0, +)
    let bGradNorm2 = (studentB.grad?.getData() ?? []).map { $0 * $0 }.reduce(0, +)
    XCTAssertGreaterThan(wGradNorm2, 0.0, "Expected non-zero gradient through peek(channel:)")
    XCTAssertGreaterThan(bGradNorm2, 0.0, "Expected non-zero gradient through peek(channel:)")

    let wBefore = studentW.getData() ?? []
    let bBefore = studentB.getData() ?? []
    optimizer.step()
    optimizer.zeroGrad()
    let wAfter = studentW.getData() ?? []
    let bAfter = studentB.getData() ?? []

    let wDelta = zip(wBefore, wAfter).reduce(Float(0)) { $0 + abs($1.0 - $1.1) }
    let bDelta = zip(bBefore, bAfter).reduce(Float(0)) { $0 + abs($1.0 - $1.1) }
    XCTAssertGreaterThan(wDelta, 0.0, "Expected optimizer step to move studentW")
    XCTAssertGreaterThan(bDelta, 0.0, "Expected optimizer step to move studentB")
  }

  func testPeekGradReduceStaticKernelParallelizesTensorDimension() throws {
    let prevSampleRate = DGenConfig.sampleRate
    let prevMaxFrameCount = DGenConfig.maxFrameCount
    let prevKernelOutputPath = DGenConfig.kernelOutputPath
    defer {
      DGenConfig.sampleRate = prevSampleRate
      DGenConfig.maxFrameCount = prevMaxFrameCount
      DGenConfig.kernelOutputPath = prevKernelOutputPath
      LazyGraphContext.reset()
    }

    DGenConfig.sampleRate = 2000.0
    DGenConfig.maxFrameCount = 512
    DGenConfig.debug = false

    let kernelPath =
      FileManager.default.temporaryDirectory
      .appendingPathComponent("peek_reduce_parallel_\(UUID().uuidString).metal").path
    DGenConfig.kernelOutputPath = kernelPath

    let controlFrames = 9
    let numChannels = 5
    let frameCount = 64
    let totalSize = controlFrames * numChannels

    let control = Tensor.param(
      [controlFrames, numChannels],
      data: [Float](repeating: 0.25, count: controlFrames * numChannels)
    )
    let playhead = Signal.phasor(DGenConfig.sampleRate / Float(frameCount)) * Float(controlFrames - 1)
    let picked = control.peek(playhead, channel: Signal.constant(0.0))
    let loss = mse(picked, Signal.constant(0.1))

    _ = try loss.backward(frames: frameCount)

    let source = try String(contentsOfFile: kernelPath, encoding: .utf8)
    let lines = source.components(separatedBy: "\n")

    var sawThreadParallelTotalSize = false
    var sawSerializedTensorLoop = false

    if source.contains("if (id < \(totalSize)) { uint _pr") {
      sawThreadParallelTotalSize = true
    }

    for i in lines.indices {
      let line = lines[i]
      if line.contains("for (uint _pr"), line.contains("< \(totalSize);") {
        let start = max(0, i - 8)
        let context = lines[start...i].joined(separator: "\n")
        if context.contains("id < (uint)(1)") {
          sawSerializedTensorLoop = true
        }
      }
    }

    XCTAssertTrue(
      sawThreadParallelTotalSize,
      "Expected static peekGradReduce-style kernel to parallelize tensor dimension with id < \(totalSize)"
    )
    XCTAssertFalse(
      sawSerializedTensorLoop,
      "Found serialized static reduction loop for \(totalSize) elements (id < 1 + for _pr < \(totalSize))"
    )
  }
}
