// RunInvariantCache.swift — render a graph's run-invariant frontier once, read it back every step.
//
// A trainer rebuilds and re-renders the same program every step. The part of it that does not
// depend on anything that changes between steps (noise through fixed filters; see
// `Graph.runInvariantAnalysis`) renders the same frames each time, yet its sequential kernels
// (feedback filters, the random-number recurrences) cost as much as the rest of a render. The
// cache renders each frontier node once in a probe graph and then, in every build, replaces it
// with a per-frame read of those frames, so the invariant subgraph is no longer compiled.

import DGen
import Foundation

public final class RunInvariantCache {
  /// Per frontier key: its per-frame shape (nil = scalar) and [frames × elements] values.
  private var tables: [UInt64: (shape: [Int]?, data: [Float])] = [:]
  /// What the tables were rendered at; a change of either invalidates them.
  private var frames = 0
  private var sampleRate: Float = 0

  /// Frontier nodes substituted by the last `build` (for logging).
  public private(set) var lastSubstituted = 0
  /// Distinct frontier values rendered by probes so far.
  public var cachedCount: Int { tables.count }

  public init() {}

  /// Runs `build` (which builds the program into the current graph) and substitutes every cached
  /// frontier node with a per-frame table read indexed by `frameIndex()` (an integer frame number
  /// per frame). The first time a frontier is seen, it is rendered by a probe: the graph is
  /// cleared, rebuilt, pruned to that node's invariant subgraph and realized over `frames`; then
  /// the graph is rebuilt once more for the caller. `build`'s result is always from the final
  /// build, so the caller's handles stay valid: only nodes inside the frontier's invariant
  /// subgraphs are removed, and only when nothing else reads them.
  public func build<T>(frames: Int, frameIndex: () -> Signal, _ build: () throws -> T) throws -> T {
    let sr = LazyGraphContext.current.graph.sampleRate
    if frames != self.frames || sr != sampleRate {
      tables.removeAll()
      self.frames = frames
      sampleRate = sr
    }
    var result = try build()
    var analysis = LazyGraphContext.current.graph.runInvariantAnalysis()
    let missing = analysis.frontier.filter { tables[$0.key] == nil }
    if !missing.isEmpty {
      for f in missing where tables[f.key] == nil {
        tables[f.key] = (f.shape, try probe(key: f.key, frames: frames, build))
      }
      LazyGraphContext.current.clearComputationGraph()
      result = try build()
      analysis = LazyGraphContext.current.graph.runInvariantAnalysis()
    }
    try substitute(analysis, frames: frames, frameIndex: frameIndex)
    return result
  }

  /// Renders the frontier node with `key` alone: rebuild, prune to its invariant subgraph, realize.
  private func probe<T>(key: UInt64, frames: Int, _ build: () throws -> T) throws -> [Float] {
    let lazy = LazyGraphContext.current
    lazy.clearComputationGraph()
    _ = try build()
    let analysis = lazy.graph.runInvariantAnalysis()
    guard let f = analysis.frontier.first(where: { $0.key == key }) else {
      throw DGenLazyError.runtimeError("run-invariant probe: frontier node vanished on rebuild")
    }
    try lazy.graph.pruneToInvariantClosure(of: f.node, analysis: analysis)
    let values: [Float]
    if let shape = f.shape {
      values = try SignalTensor(nodeId: f.node, graph: lazy, shape: shape).realize(frames: frames)
    } else {
      values = try Signal(nodeId: f.node, graph: lazy, graphId: lazy.id).realize(frames: frames)
    }
    let size = f.shape?.reduce(1, *) ?? 1
    guard values.count == frames * size else {
      throw DGenLazyError.runtimeError(
        "run-invariant probe: got \(values.count) values, expected \(frames) × \(size)")
    }
    lazy.clearComputationGraph()
    return values
  }

  private func substitute(
    _ analysis: Graph.RunInvariantAnalysis, frames: Int, frameIndex: () -> Signal
  ) throws {
    lastSubstituted = 0
    guard !analysis.frontier.isEmpty else { return }
    let lazy = LazyGraphContext.current
    let index = frameIndex()
    var replacements: [NodeID: NodeID] = [:]
    for f in analysis.frontier {
      guard let table = tables[f.key], table.shape == f.shape,
        table.data.count == frames * (f.shape?.reduce(1, *) ?? 1)
      else {
        throw DGenLazyError.runtimeError(
          "run-invariant cache: no table of shape \(f.shape ?? []) × \(frames) frames for a frontier node")
      }
      let t = Tensor(table.data)
      if let shape = f.shape {
        replacements[f.node] = t.reshape([frames] + shape).sampleRow(index).nodeId
      } else {
        replacements[f.node] = t.reshape([frames, 1]).sampleRow(index).sum().nodeId
      }
    }
    lazy.graph.replaceUses(replacements)
    lazy.graph.removeDeadInvariant(analysis)
    lastSubstituted = replacements.count
  }
}
