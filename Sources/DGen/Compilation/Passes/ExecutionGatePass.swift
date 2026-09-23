import Foundation

/// A disjunction of conjunctions of positive block-gate conditions.
/// An empty conjunction means unconditional demand; no terms means no demand.
public struct ExecutionDemand: Equatable {
  public var terms: [Set<NodeID>]
  public static let always = ExecutionDemand(terms: [[]])

  mutating func include(_ term: Set<NodeID>) -> Bool {
    if terms.contains(where: { $0.isSubset(of: term) }) { return false }
    terms.removeAll { term.isSubset(of: $0) }
    terms.append(term)
    return true
  }
}

/// Computes demand through value and history dependencies, before block formation.
/// Shared producers execute whenever ANY consumer needs them. A gate may never
/// hide arbitrary side effects or gate only one half of a feedback recurrence.
enum ExecutionGatePass {
  struct Plan {
    var demands: [NodeID: ExecutionDemand] = [:]
    var originalDependencies: [NodeID: [NodeID]] = [:]

    func restoreDependencies(graph: Graph) {
      for (id, deps) in originalDependencies { graph.nodes[id]?.temporalDependencies = deps }
    }

    func split(blocks: [Block]) throws -> [Block] {
      guard !demands.isEmpty else { return blocks }
      return try blocks.enumerated().flatMap { index, block -> [Block] in
        var result: [Block] = []
        for id in block.nodes {
          let demand = demands[id] ?? .always
          if result.last?.executionDemand == demand {
            result[result.count - 1].nodes.append(id)
          } else {
            var part = block
            part.nodes = [id]
            part.executionDemand = demand
            result.append(part)
          }
        }
        if block.frameOrder == .sequential && result.count > 1 {
          // Splitting a recurrence into separate frame loops changes a
          // one-sample history edge into a block delay. Keep these fragments
          // together and put their conditionals INSIDE the sample loop.
          let internalConditions = Set(result.flatMap { $0.executionDemand.terms.flatMap { $0 } })
            .intersection(block.nodes)
          if !internalConditions.isEmpty {
            throw DGenError.compilationFailed("block-gate predicate must be available before its feedback region")
          }
          for i in result.indices { result[i].sequentialFrameGroup = index }
        }
        return result
      }
    }
  }

  static func prepare(graph: Graph, backend: Backend) throws -> Plan {
    guard backend == .c, !graph.executionGates.isEmpty else { return Plan() }
    var writers: [CellID: [NodeID]] = [:]
    for (id, node) in graph.nodes {
      switch node.op {
      case .historyWrite(let cell), .historyReadWrite(let cell):
        writers[cell, default: []].append(id)
      default: break
      }
    }
    // A scalar lookup may be skipped while its externally owned table remains
    // available. Stop execution demand at a leaf table reference; do not extend
    // this to computed tensors, tensor histories, or hop-rate producers.
    func isLeafTableRead(_ node: Node) -> Bool {
      guard case .peek = node.op, let tableID = node.inputs.first,
        let table = graph.nodes[tableID], case .tensorRef = table.op,
        table.allDependencies.isEmpty else { return false }
      return true
    }
    func dependencies(_ id: NodeID) -> [NodeID] {
      guard let node = graph.nodes[id] else { return [] }
      if isLeafTableRead(node) {
        return Array(node.inputs.dropFirst()) + node.temporalDependencies
      }
      switch node.op {
      case .historyRead(let cell), .historyReadWrite(let cell):
        return node.allDependencies + (writers[cell] ?? [])
      default: return node.allDependencies
      }
    }
    var candidates = Set<NodeID>()
    var stack = graph.executionGates.keys.compactMap { graph.nodes[$0]?.inputs.dropFirst().first }
    while let id = stack.popLast() {
      guard candidates.insert(id).inserted else { continue }
      stack.append(contentsOf: dependencies(id))
    }
    for id in candidates {
      guard let node = graph.nodes[id] else { continue }
      guard (ScalarBlockCoalescingPass.isPlainScalarOp(node.op) || isLeafTableRead(node)),
        graph.nodeToTensor[id] == nil, graph.nodeHopRate[id] == nil else {
        throw DGenError.compilationFailed("block-gate supports scalar DSP only; unsupported node \(id): \(node.op)")
      }
      if case .tensor? = node.shape {
        throw DGenError.compilationFailed("block-gate does not support tensor state")
      }
    }
    // Parameters and pure functions of parameters stay unconditional and can
    // still be hoisted. This also keeps every gate condition available.
    var invariant = Set<NodeID>()
    var changed = true
    while changed {
      changed = false
      for (id, node) in graph.nodes where !invariant.contains(id) {
        if StaticHoistPass.isHoistableOp(node.op), node.temporalDependencies.isEmpty,
          node.inputs.allSatisfy({ invariant.contains($0) }) {
          invariant.insert(id); changed = true
        }
      }
    }
    var demands: [NodeID: ExecutionDemand] = [:]
    var work: [(NodeID, Set<NodeID>)] = []
    for (id, node) in graph.nodes {
      if !candidates.contains(id) || invariant.contains(id) || node.op.isOutput {
        work.append((id, []))
      }
    }
    // Gate predicates must be available before choosing a region. Their entire
    // dependency cone is unconditional, including other muxes: conditioning a
    // shared producer on a downstream predicate would introduce a scheduling
    // cycle. Conservatively keep predicate preparation outside every gate.
    var predicateDependencies = Set<NodeID>()
    var pendingPredicates = Array(graph.executionGates.values)
    while let id = pendingPredicates.popLast() {
      guard predicateDependencies.insert(id).inserted else { continue }
      pendingPredicates.append(contentsOf: dependencies(id))
    }
    work.append(contentsOf: predicateDependencies.map { ($0, []) })
    work.append(contentsOf: graph.materializeNodes.map { ($0, []) })
    work.append(contentsOf: graph.gradientSideEffects.map { ($0, []) })
    // Propagation only preserves a conjunction or adds a predicate. Exhaust
    // shorter conjunctions first so a broad demand reaches a shared producer
    // before narrower paths through its feedback cycle can multiply.
    var workBySize = [work]
    work.removeAll()
    var size = 0
    func enqueue(_ id: NodeID, _ term: Set<NodeID>) {
      while workBySize.count <= term.count { workBySize.append([]) }
      workBySize[term.count].append((id, term))
    }
    while size < workBySize.count {
      guard let (id, term) = workBySize[size].popLast() else {
        size += 1
        continue
      }
      guard let node = graph.nodes[id] else { continue }
      var demand = demands[id] ?? ExecutionDemand(terms: [])
      guard demand.include(term) else { continue }
      demands[id] = demand
      if let condition = graph.executionGates[id], node.inputs.count == 3 {
        enqueue(node.inputs[0], term)
        enqueue(node.inputs[1], term.union([condition]))
        enqueue(node.inputs[2], term)
        for dependency in node.temporalDependencies { enqueue(dependency, term) }
      } else {
        for dependency in dependencies(id) { enqueue(dependency, term) }
      }
    }
    // Canonical ordering is needed for region equality, not during fixed-point
    // propagation. Sort each accepted predicate set once after convergence.
    for id in demands.keys {
      let sorted = demands[id]!.terms.map { $0.sorted() }.sorted {
        $0.lexicographicallyPrecedes($1)
      }
      demands[id]!.terms = sorted.map { Set($0) }
    }
    var plan = Plan(demands: demands)
    for (id, demand) in demands where demand != .always {
      let conditions = Set(demand.terms.flatMap { $0 }).sorted()
      guard let node = graph.nodes[id] else { continue }
      plan.originalDependencies[id] = node.temporalDependencies
      for condition in conditions where !node.allDependencies.contains(condition) {
        graph.nodes[id]?.temporalDependencies.append(condition)
      }
    }
    return plan
  }
}
