// Analyze - Static kernel analysis without execution
//
// Provides .analyze() on Tensor, Signal, and SignalTensor that compiles
// the graph and analyzes the Metal IR to report work, span, memory,
// and kernel count — without creating a runtime or running any GPU code.

import DGen
import Foundation

// MARK: - LazyGraph Compilation for Analysis

extension LazyGraph {
  /// Compile the graph for analysis only (no runtime creation).
  /// Returns the CompilationResult and the Metal ScheduleItems.
  func compileForAnalysis(frameCount: Int) throws -> (CompilationResult, [ScheduleItem]) {
    let result = try CompilationPipeline.compile(
      graph: graph,
      backend: .metal,
      options: .init(frameCount: frameCount, debug: false)
    )

    let renderer = MetalRenderer()
    var scheduleItems: [ScheduleItem] = []
    renderer.prepareSchedule(&scheduleItems, result.uopBlocks, result.context, frameCount)

    return (result, scheduleItems)
  }
}

// MARK: - Tensor Analyze

extension Tensor {
  /// Analyze the computation graph without executing it.
  ///
  /// Returns kernel count, work (FLOPs), span (sequential depth), and memory usage.
  ///
  /// ```swift
  /// let a = Tensor([1, 2, 3, 4])
  /// let b = Tensor([5, 6, 7, 8])
  /// let c = (a + b).sum()
  /// let analysis = try c.analyze()
  /// print("Kernels: \(analysis.kernelCount), Work: \(analysis.work)")
  /// ```
  ///
  /// - Parameters:
  ///   - backward: If true, include backward pass (gradient computation)
  ///   - frameCount: Number of frames (default 1 for static tensors)
  /// - Returns: KernelAnalysis with per-kernel breakdown
  public func analyze(backward: Bool = false, frameCount: Int = 1) throws -> KernelAnalysis {
    let includesBackward = backward

    if backward {
      // Set up gradients (same as runBackward)
      graph.markDirty()
      _ = graph.setupGradients(loss: nodeId, frameCount: frameCount)

      if graph.graph.gradientSideEffects.isEmpty {
        _ = graph.node(.output(0), [nodeId])
      } else {
        let chainedValue = graph.graph.chainGradientSideEffects(after: nodeId)
        _ = graph.node(.output(0), [chainedValue])
        graph.graph.gradientSideEffects = []
      }
    } else {
      // Forward only — same as realize()
      if isScalar {
        _ = graph.node(.output(0), [nodeId])
      } else {
        let sumNode = graph.node(.sum, [nodeId])
        _ = graph.node(.output(0), [sumNode])
      }

      if !isScalar {
        graph.graph.materializeNodes.insert(nodeId)
      }
    }

    let (result, scheduleItems) = try graph.compileForAnalysis(frameCount: frameCount)

    let analysis = analyzeScheduleItems(
      scheduleItems,
      frameCount: frameCount,
      totalMemorySlots: result.totalMemorySlots,
      includesBackward: includesBackward
    )

    graph.clearComputationGraph()
    return analysis
  }
}

// MARK: - Signal Analyze

extension Signal {
  /// Analyze the computation graph without executing it.
  ///
  /// ```swift
  /// let freq = Signal.param(440.0)
  /// let phase = Signal.phasor(freq)
  /// let wave = sin(phase * 2 * .pi)
  /// let analysis = try wave.analyze(frames: 64)
  /// print("Kernels: \(analysis.kernelCount), Work: \(analysis.work)")
  /// ```
  ///
  /// - Parameters:
  ///   - backward: If true, include backward pass
  ///   - frames: Number of frames to analyze
  /// - Returns: KernelAnalysis with per-kernel breakdown
  public func analyze(backward: Bool = false, frames: Int = DGenConfig.defaultFrameCount) throws -> KernelAnalysis {
    let includesBackward = backward

    if backward {
      graph.markDirty()
      _ = graph.setupGradients(loss: nodeId, frameCount: frames)

      if graph.graph.gradientSideEffects.isEmpty {
        _ = graph.node(.output(0), [nodeId])
      } else {
        let chainedValue = graph.graph.chainGradientSideEffects(after: nodeId)
        _ = graph.node(.output(0), [chainedValue])
        graph.graph.gradientSideEffects = []
      }
    } else {
      _ = graph.node(.output(0), [nodeId])
    }

    let (result, scheduleItems) = try graph.compileForAnalysis(frameCount: frames)

    let analysis = analyzeScheduleItems(
      scheduleItems,
      frameCount: frames,
      totalMemorySlots: result.totalMemorySlots,
      includesBackward: includesBackward
    )

    graph.clearComputationGraph()
    return analysis
  }
}

// MARK: - SignalTensor Analyze

extension SignalTensor {
  /// Analyze the computation graph without executing it.
  ///
  /// - Parameters:
  ///   - backward: If true, include backward pass
  ///   - frames: Number of frames to analyze
  /// - Returns: KernelAnalysis with per-kernel breakdown
  public func analyze(backward: Bool = false, frames: Int = DGenConfig.defaultFrameCount) throws -> KernelAnalysis {
    let includesBackward = backward

    if backward {
      graph.markDirty()
      _ = graph.setupGradients(loss: nodeId, frameCount: frames)

      if graph.graph.gradientSideEffects.isEmpty {
        _ = graph.node(.output(0), [nodeId])
      } else {
        let chainedValue = graph.graph.chainGradientSideEffects(after: nodeId)
        _ = graph.node(.output(0), [chainedValue])
        graph.graph.gradientSideEffects = []
      }
    } else {
      let sumNode = graph.node(.sum, [nodeId])
      _ = graph.node(.output(0), [sumNode])
      graph.graph.materializeNodes.insert(nodeId)
    }

    let (result, scheduleItems) = try graph.compileForAnalysis(frameCount: frames)

    let analysis = analyzeScheduleItems(
      scheduleItems,
      frameCount: frames,
      totalMemorySlots: result.totalMemorySlots,
      includesBackward: includesBackward
    )

    graph.clearComputationGraph()
    return analysis
  }
}
