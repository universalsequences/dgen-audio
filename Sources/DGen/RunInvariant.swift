// RunInvariant.swift — finding the part of a graph that renders the same audio on every run.
//
// A training render starts from scratch (every buffer zeroed, so every noise generator and
// history restarts identically), so a subgraph built only from constants, literal tensors, noise
// and histories of such values produces identical frames on every run, e.g. a hiss layer (white
// noise through fixed band-pass filters). `runInvariantAnalysis` finds those nodes and the
// frontier where they feed the rest of the graph; DGenLazy's `RunInvariantCache` renders each
// frontier node once and substitutes a per-frame table read for it, so its sequential kernels
// stop running every step.

import Foundation

extension Graph {
  /// A run-invariant node consumed by a run-varying one, and worth caching (it depends on noise or
  /// a history). `key` identifies it structurally across rebuilds of the same program.
  public struct RunInvariantFrontier {
    public let node: NodeID
    public let key: UInt64
    /// nil for a scalar signal, else the tensor shape it has per frame.
    public let shape: [Int]?
  }

  /// The result of `runInvariantAnalysis`, kept for the rewrite that follows it.
  public struct RunInvariantAnalysis {
    public let invariant: Set<NodeID>
    public let frontier: [RunInvariantFrontier]
    /// Nodes writing each state cell (history writes included).
    let writers: [CellID: [NodeID]]
  }

  /// Ops whose result depends only on their inputs: no state cell, no external data.
  private static func isPureOp(_ op: LazyOp) -> Bool {
    switch op {
    case .modulatedParam, .historyRead: return false  // a parameter; a history is resolved separately
    case .sum, .seq, .expand, .expandAxis: return true
    default: return DeadCodeEliminationPass.isPure(op) || op.isMemorylessView || isAxisReduceOp(op)
    }
  }

  /// Ops that generate frames by themselves: what makes a frontier worth caching.
  private static func isGenerator(_ op: LazyOp) -> Bool {
    switch op {
    case .noise, .tensorNoise, .historyRead: return true
    default: return false
    }
  }

  /// Every node writing each state cell.
  private func writersByCell() -> [CellID: [NodeID]] {
    var writers: [CellID: [NodeID]] = [:]
    for (id, node) in nodes {
      for cell in stateAccess(node).writes { writers[cell, default: []].append(id) }
    }
    return writers
  }

  /// `roots` and everything they read, a history read reading every writer of its cell.
  private func closure(of roots: [NodeID], writers: [CellID: [NodeID]]) -> Set<NodeID> {
    var seen = Set<NodeID>()
    var stack = roots
    while let id = stack.popLast() {
      guard seen.insert(id).inserted, let node = nodes[id] else { continue }
      stack.append(contentsOf: node.inputs)
      stack.append(contentsOf: node.temporalDependencies)
      if case .historyRead(let cell) = node.op { stack.append(contentsOf: writers[cell, default: []]) }
    }
    return seen
  }

  /// The run-invariant nodes and their cacheable frontier.
  ///
  /// Invariant: a pure op over invariant inputs, a literal tensor (`constantTensors`), `noise` /
  /// `tensorNoise` (seeded the same on every run), or a history read whose every writer is
  /// invariant (resolved to a fixed point: start optimistic, retract until stable). Node ids are
  /// taken as a topological order (inputs are created first); a temporal dependency on a later
  /// node only makes its consumer conservatively varying.
  public func runInvariantAnalysis() -> RunInvariantAnalysis {
    let writers = writersByCell()
    var invariant = Set<NodeID>()
    var historyOK = Set<CellID>(writers.keys)
    let order = nodes.keys.sorted()
    while true {
      invariant.removeAll(keepingCapacity: true)
      for id in order {
        guard let node = nodes[id] else { continue }
        let ok: Bool
        switch node.op {
        case .noise, .tensorNoise: ok = true
        case .historyRead(let cell): ok = historyOK.contains(cell)
        case .tensorRef(let tid): ok = constantTensors.contains(tid)
        case .historyWrite: ok = node.inputs.allSatisfy(invariant.contains)
        default:
          ok = Self.isPureOp(node.op) && node.inputs.allSatisfy(invariant.contains)
            && node.temporalDependencies.allSatisfy(invariant.contains)
        }
        if ok { invariant.insert(id) }
      }
      let stillOK = historyOK.filter { writers[$0, default: []].allSatisfy(invariant.contains) }
      if stillOK == historyOK { break }
      historyOK = stillOK
    }

    // invariant nodes that depend (transitively) on a generator
    var generated = Set<NodeID>()
    for id in order where invariant.contains(id) {
      guard let node = nodes[id] else { continue }
      if Self.isGenerator(node.op) || node.inputs.contains(where: generated.contains) {
        generated.insert(id)
      }
    }
    // frontier: generated invariant nodes with a varying consumer
    var frontierIds = Set<NodeID>()
    for (id, node) in nodes where !invariant.contains(id) {
      for d in node.allDependencies where generated.contains(d) { frontierIds.insert(d) }
    }
    var memo: [NodeID: UInt64] = [:]
    let frontier = frontierIds.sorted().compactMap { id -> RunInvariantFrontier? in
      guard let node = nodes[id] else { return nil }
      var shape: [Int]? = nil
      if case .tensor(let s) = node.shape { shape = s }
      var visiting = Set<NodeID>()
      return RunInvariantFrontier(
        node: id, key: structuralKey(id, writers: writers, memo: &memo, visiting: &visiting), shape: shape)
    }
    return RunInvariantAnalysis(invariant: invariant, frontier: frontier, writers: writers)
  }

  /// A node's op with its cell / tensor ids stripped (they change between rebuilds).
  private static func structuralTag(_ op: LazyOp) -> String {
    switch op {
    case .noise: return "noise"
    case .tensorNoise(_, _, let size): return "tensorNoise \(size)"
    case .historyRead: return "historyRead"
    case .historyWrite: return "historyWrite"
    case .tensorRef: return "tensorRef"
    default: return "\(op)"  // pure ops: no ids
    }
  }

  /// A hash of a node's invariant subgraph that ignores node and cell ids. A history read hashes
  /// the values written to its cell; a feedback cycle hashes as a marker, so a node first reached
  /// inside a cycle hashes relative to the walk, which is deterministic for a given program.
  /// Hasher is seeded per process: keys are stable within one process, all the in-memory cache
  /// needs.
  private func structuralKey(
    _ id: NodeID, writers: [CellID: [NodeID]], memo: inout [NodeID: UInt64], visiting: inout Set<NodeID>
  ) -> UInt64 {
    if let k = memo[id] { return k }
    guard let node = nodes[id] else { return 0 }
    if !visiting.insert(id).inserted { return 0x9e37_79b9_7f4a_7c15 }
    defer { visiting.remove(id) }
    var h = Hasher()
    h.combine(Self.structuralTag(node.op))
    h.combine("\(String(describing: node.shape))")
    switch node.op {
    case .tensorRef(let tid):
      if let t = tensors[tid] { h.combine(t.shape); h.combine(t.data ?? []) }
    case .historyRead(let cell):
      for w in writers[cell, default: []] {
        h.combine(structuralKey(w, writers: writers, memo: &memo, visiting: &visiting))
      }
      if let tid = cellToTensor[cell], let t = tensors[tid] { h.combine(t.data ?? []) }
    default: break
    }
    for d in node.inputs { h.combine(structuralKey(d, writers: writers, memo: &memo, visiting: &visiting)) }
    let k = UInt64(bitPattern: Int64(h.finalize()))
    memo[id] = k
    return k
  }

  /// Keeps only `root` and the invariant subgraph it needs (including the history writes that feed
  /// its reads), dropping everything else, outputs included: what a probe render of `root` needs.
  public func pruneToInvariantClosure(of root: NodeID, analysis: RunInvariantAnalysis) throws {
    let keep = closure(of: [root], writers: analysis.writers)
    guard keep.isSubset(of: analysis.invariant) else {
      throw DGenError.compilationFailed("run-invariant probe closure left the invariant subgraph")
    }
    removeNodes(Set(nodes.keys).subtracting(keep))
  }

  /// Points every use of a key node at its value instead (inputs and temporal dependencies).
  public func replaceUses(_ replacements: [NodeID: NodeID]) {
    guard !replacements.isEmpty else { return }
    let sources = Set(replacements.values)
    for (id, node) in nodes where !sources.contains(id) {
      guard node.inputs.contains(where: { replacements[$0] != nil })
        || node.temporalDependencies.contains(where: { replacements[$0] != nil })
      else { continue }
      var updated = Node(id: id, op: node.op, inputs: node.inputs.map { replacements[$0] ?? $0 })
      updated.temporalDependencies = node.temporalDependencies.map { replacements[$0] ?? $0 }
      updated.shape = node.shape
      nodes[id] = updated
    }
  }

  /// After `replaceUses` cut the frontier off, deletes the nodes of the frontier's subgraphs that
  /// nothing else reads any more, so their kernels never compile. Only nodes inside those
  /// subgraphs are candidates; anything still read by a varying node, by a graph-side map (as for
  /// dead-code elimination) or by another live node stays.
  public func removeDeadInvariant(_ analysis: RunInvariantAnalysis) {
    let candidates = closure(of: analysis.frontier.map(\.node), writers: analysis.writers)
    var roots = nodes.keys.filter { !candidates.contains($0) }
    roots.append(contentsOf: DeadCodeEliminationPass.sideMapRoots(self))
    roots.append(contentsOf: executionGates.values)
    roots.append(contentsOf: eventHoldClocks.values)
    roots.append(contentsOf: schedulingKeys.values)
    let live = closure(of: roots, writers: analysis.writers)
    removeNodes(candidates.subtracting(live))
  }

  private func removeNodes(_ dead: Set<NodeID>) {
    guard !dead.isEmpty else { return }
    for id in dead {
      nodes.removeValue(forKey: id)
      nodeToTensor.removeValue(forKey: id)
      nodeHopRate.removeValue(forKey: id)
      eventHoldClocks.removeValue(forKey: id)
      schedulingKeys.removeValue(forKey: id)
      nodePositionDep.removeValue(forKey: id)
      executionGates.removeValue(forKey: id)
      sampleGates.removeValue(forKey: id)
      materializeNodes.remove(id)
      boundedIndexNodes.remove(id)
      eventClockNodes.remove(id)
      sparseEventClocks.remove(id)
    }
    gradientSideEffects.removeAll { dead.contains($0) }
  }
}
