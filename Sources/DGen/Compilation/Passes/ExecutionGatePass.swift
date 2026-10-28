import Foundation

/// A disjunction of conjunctions of positive block-gate conditions.
/// An empty conjunction means unconditional demand; no terms means no demand.
public struct ExecutionDemand: Hashable {
  public var terms: [Set<NodeID>]
  public static let always = ExecutionDemand(terms: [[]])

  mutating func include(_ term: Set<NodeID>) -> Bool {
    if terms.contains(where: { $0.isSubset(of: term) }) { return false }
    terms.removeAll { term.isSubset(of: $0) }
    terms.append(term)
    return true
  }

  /// Every condition any term tests.
  var conditions: Set<NodeID> { Set(terms.flatMap { $0 }) }

  /// Whether any term tests one of `conditions`.
  func mentions(_ conditions: Set<NodeID>) -> Bool {
    !terms.allSatisfy { $0.isDisjoint(with: conditions) }
  }

  /// This demand with `conditions` no longer tested.
  func removing(_ conditions: Set<NodeID>) -> ExecutionDemand {
    guard mentions(conditions) else { return self }
    var result = ExecutionDemand(terms: [])
    for term in terms { _ = result.include(term.subtracting(conditions)) }
    return result
  }
}

/// Computes demand through value and history dependencies, before block formation.
/// Shared producers execute whenever ANY consumer needs them. A gate may never
/// hide arbitrary side effects or gate only one half of a feedback recurrence.
enum ExecutionGatePass {
  /// A stateless node whose demand has more alternatives than this runs
  /// under the conditions they share (see `prepare`).
  static let maxAlternatives = 2

  struct Plan {
    var demands: [NodeID: ExecutionDemand] = [:]
    var originalDependencies: [NodeID: [NodeID]] = [:]
    /// Conditions tested per frame inside the sample loop.
    var sampleConditions: Set<NodeID> = []

    func restoreDependencies(graph: Graph) {
      for (id, deps) in originalDependencies { graph.nodes[id]?.temporalDependencies = deps }
    }

    /// Drops per-frame conditions from `demand`. A frame-parallel block has no
    /// state for a per-frame gate to hold, so it computes every frame.
    private func withoutSampleConditions(_ demand: ExecutionDemand) -> ExecutionDemand {
      demand.removing(sampleConditions)
    }

    /// Splits off the part of `block` that computes per-call gate conditions
    /// used inside it: their dependency cone, plus every access to a state cell
    /// the cone touches (a recurrence moves whole). Nil when that would take a
    /// gated node along, or when the block has no such condition.
    private func peelConditionCone(_ block: Block, graph: Graph) -> (Block, Block)? {
      let members = Set(block.nodes)
      let conditions = Set(block.nodes.flatMap { (demands[$0] ?? .always).conditions })
        .intersection(members).subtracting(sampleConditions)
      guard !conditions.isEmpty else { return nil }
      var accessors: [CellID: [NodeID]] = [:]
      func cells(_ node: Node) -> [CellID] {
        let access = graph.stateAccess(node)
        return access.reads + access.writes
      }
      for id in block.nodes {
        guard let node = graph.nodes[id] else { continue }
        for cell in cells(node) { accessors[cell, default: []].append(id) }
      }
      var cone = Set<NodeID>()
      var visitedCells = Set<CellID>()
      var stack = Array(conditions)
      while let id = stack.popLast() {
        guard members.contains(id), cone.insert(id).inserted, let node = graph.nodes[id] else { continue }
        stack.append(contentsOf: node.allDependencies)
        for cell in cells(node) where visitedCells.insert(cell).inserted {
          stack.append(contentsOf: accessors[cell] ?? [])
        }
      }
      guard cone.allSatisfy({ (demands[$0] ?? .always) == .always }) else { return nil }
      // A value-edge from outside into the cone would run after its consumer.
      for id in cone {
        guard let node = graph.nodes[id] else { continue }
        if node.allDependencies.contains(where: { members.contains($0) && !cone.contains($0) }) { return nil }
      }
      var head = block, tail = block
      head.nodes = block.nodes.filter(cone.contains)
      tail.nodes = block.nodes.filter { !cone.contains($0) }
      return (head, tail)
    }

    func split(blocks: [Block], graph: Graph) throws -> [Block] {
      guard !demands.isEmpty else { return blocks }
      // A recurrence computing a per-call condition that landed in the same
      // sequential block as the region it gates runs as its own loop first.
      var prepared: [Block] = []
      for block in blocks {
        if block.frameOrder == .sequential, let (head, tail) = peelConditionCone(block, graph: graph) {
          prepared.append(head)
          prepared.append(tail)
        } else {
          prepared.append(block)
        }
      }
      return try prepared.enumerated().flatMap { index, block -> [Block] in
        var result: [Block] = []
        let sequential = block.frameOrder == .sequential
        let gatedHere = sequential && block.nodes.contains { demands[$0].map { $0 != .always } ?? false }
        let nodes = gatedHere
          ? DemandScheduling.order(block.nodes, demands: demands, graph: graph) : block.nodes
        for id in nodes {
          var demand = demands[id] ?? .always
          if !sequential { demand = withoutSampleConditions(demand) }
          if result.last?.executionDemand == demand {
            result[result.count - 1].nodes.append(id)
          } else {
            var part = block
            part.nodes = [id]
            part.executionDemand = demand
            result.append(part)
          }
        }
        let perFrame = sequential && result.contains { $0.executionDemand.mentions(sampleConditions) }
        if sequential && (result.count > 1 || perFrame) {
          // Splitting a recurrence into separate frame loops changes a
          // one-sample history edge into a block delay. Keep these fragments
          // together and put their conditionals INSIDE the sample loop. A
          // per-frame condition is tested there each frame; a per-call one
          // must be ready before the loop starts.
          let internalConditions = Set(result.flatMap { $0.executionDemand.conditions })
            .intersection(block.nodes).subtracting(sampleConditions)
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
    let sampleConditions = graph.sampleGateConditions
    let ownedCells = graph.sampleGateCells
    for (gate, condition) in graph.executionGates
    where graph.sampleGates[gate] == nil && sampleConditions.contains(condition) {
      throw DGenError.compilationFailed(
        "node \(condition) is the condition of both a block-gate and a sample-gate")
    }
    var writers: [CellID: [NodeID]] = [:]
    var memoryWriters: [CellID: [NodeID]] = [:]
    for (id, node) in graph.nodes {
      switch node.op {
      case .historyWrite(let cell), .historyReadWrite(let cell):
        writers[cell, default: []].append(id)
      case .memoryWrite(let cell):
        memoryWriters[cell, default: []].append(id)
      default: break
      }
    }
    // State a per-frame gate owns: its writers run with the gate's body.
    func ownedWriters(_ gate: NodeID) -> [NodeID] {
      (graph.sampleGates[gate] ?? []).sorted().flatMap { (writers[$0] ?? []) + (memoryWriters[$0] ?? []) }
    }
    // A scalar ring buffer (`delay`, or `poke`/`peek` on a stored tensor)
    // freezes with its gate like a history: every read of a ring depends on
    // all of its writes, so a ring is either wholly gated or wholly live.
    func isScalarRingAccess(_ node: Node) -> Bool {
      if case .tensor? = node.shape { return false }
      guard graph.nodeToTensor[node.id] == nil else { return false }
      switch node.op {
      case .memoryRead, .memoryWrite: return true
      default: return graph.mutableTensorReadCell(node) != nil
      }
    }
    // Event-rate coefficient math (EventRatePromotionPass) may run inside a
    // gate: its clock fires on the first frame of every call where the region
    // is gated, so a region skipped while the gate was closed is current again
    // on the call that reopens it. The clock is gated with its region.
    func isPromotedEventNode(_ id: NodeID, _ node: Node) -> Bool {
      if case .blockStart = node.op { return true }
      if case .eventLatch = node.op {
        return node.inputs.count == 2 && graph.sparseEventClocks.contains(node.inputs[1])
      }
      guard let rate = graph.nodeHopRate[id] else { return false }
      return graph.sparseEventClocks.contains(rate.1)
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
    // Reads demand their writers, so state advances whenever it is observed.
    // Per-frame gates own their state lexically instead: state a sample-gate
    // owns is demanded only through its gate, and state created outside every
    // sample-gate is not slowed down by one. Seen from inside a sample-gate
    // body, such state (and a self-contained stateful op such as a phasor)
    // runs under the demand that remains without the per-frame conditions.
    func stateCell(_ node: Node) -> CellID? {
      switch node.op {
      case .historyRead(let cell), .historyReadWrite(let cell), .memoryRead(let cell): return cell
      default: return graph.mutableTensorReadCell(node)
      }
    }
    func isSelfContainedState(_ node: Node) -> Bool {
      switch node.op {
      case .historyRead, .historyWrite, .historyReadWrite: return false
      default: return !node.op.persistentStateCellIds.isEmpty
      }
    }
    // Nodes that must run under `base` because state outside the gate needs them.
    func unownedState(_ node: Node) -> [NodeID] {
      if isSelfContainedState(node) {
        return node.op.persistentStateCellIds.allSatisfy(ownedCells.contains) ? [] : [node.id]
      }
      guard let cell = stateCell(node), !ownedCells.contains(cell) else { return [] }
      var result: [NodeID] = []
      if case .historyReadWrite = node.op { result.append(node.id) }
      switch node.op {
      case .historyRead, .historyReadWrite: result += writers[cell] ?? []
      default: result += memoryWriters[cell] ?? []
      }
      return result
    }
    func dependencies(_ id: NodeID, sampleScoped: Bool = false) -> [NodeID] {
      guard let node = graph.nodes[id] else { return [] }
      let followWriters: Bool
      if let cell = stateCell(node) {
        followWriters = !sampleScoped && !ownedCells.contains(cell)
      } else {
        followWriters = false
      }
      if isLeafTableRead(node) {
        let ringWriters = followWriters
          ? (graph.mutableTensorReadCell(node).flatMap { memoryWriters[$0] } ?? []) : []
        return Array(node.inputs.dropFirst()) + node.temporalDependencies + ringWriters
      }
      guard followWriters else { return node.allDependencies }
      switch node.op {
      case .historyRead(let cell), .historyReadWrite(let cell):
        return node.allDependencies + (writers[cell] ?? [])
      case .memoryRead(let cell):
        return node.allDependencies + (memoryWriters[cell] ?? [])
      default: return node.allDependencies
      }
    }
    var candidates = Set<NodeID>()
    var blockStack: [NodeID] = []
    var sampleStack: [NodeID] = []
    for (gate, _) in graph.executionGates {
      guard let body = graph.nodes[gate]?.inputs.dropFirst().first else { continue }
      if graph.sampleGates[gate] != nil {
        sampleStack.append(body)
        sampleStack.append(contentsOf: ownedWriters(gate))
      } else {
        blockStack.append(body)
      }
    }
    var visited = Set<NodeID>()
    while let id = blockStack.popLast() {
      guard visited.insert(id).inserted else { continue }
      candidates.insert(id)
      blockStack.append(contentsOf: dependencies(id))
    }
    visited.removeAll()
    while let id = sampleStack.popLast() {
      guard visited.insert(id).inserted else { continue }
      candidates.insert(id)
      sampleStack.append(contentsOf: dependencies(id, sampleScoped: true))
    }
    for id in candidates {
      guard let node = graph.nodes[id] else { continue }
      if isScalarRingAccess(node) || isPromotedEventNode(id, node) { continue }
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
    // Block-gate predicates must be available before choosing a region. Their
    // entire dependency cone is unconditional, including other muxes:
    // conditioning a shared producer on a downstream predicate would introduce
    // a scheduling cycle. Conservatively keep predicate preparation outside
    // every gate.
    var predicateDependencies = Set<NodeID>()
    var pendingPredicates = Array(Set(graph.executionGates.values).subtracting(sampleConditions))
    while let id = pendingPredicates.popLast() {
      guard predicateDependencies.insert(id).inserted else { continue }
      pendingPredicates.append(contentsOf: dependencies(id))
    }
    work.append(contentsOf: predicateDependencies.map { ($0, []) })
    // A sample-gate predicate is evaluated every frame, ahead of its body in
    // the same sample loop, so its cone only needs ordinary demand: it may
    // read the body's state, and it does not reach into other gates' bodies.
    work.append(contentsOf: sampleConditions.map { ($0, []) })
    work.append(contentsOf: graph.materializeNodes.map { ($0, []) })
    work.append(contentsOf: graph.gradientSideEffects.map { ($0, []) })
    // Propagation only preserves a conjunction or adds a predicate. Exhaust
    // shorter conjunctions first so a broad demand reaches a shared producer
    // before narrower paths through its feedback cycle can multiply.
    var workBySize: [[(NodeID, Set<NodeID>)]] = []
    var size = 0
    func enqueue(_ id: NodeID, _ term: Set<NodeID>) {
      while workBySize.count <= term.count { workBySize.append([]) }
      workBySize[term.count].append((id, term))
      size = min(size, term.count)
    }
    func propagate() {
      size = 0
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
          let gated = term.union([condition])
          enqueue(node.inputs[0], term)
          enqueue(node.inputs[1], gated)
          enqueue(node.inputs[2], term)
          for dependency in node.temporalDependencies { enqueue(dependency, term) }
          for writer in ownedWriters(id) { enqueue(writer, gated) }
        } else {
          let sampleScoped = !term.isDisjoint(with: sampleConditions)
          for dependency in dependencies(id, sampleScoped: sampleScoped) { enqueue(dependency, term) }
          if sampleScoped {
            let base = term.subtracting(sampleConditions)
            for state in unownedState(node) { enqueue(state, base) }
          }
        }
      }
    }
    for (id, term) in work { enqueue(id, term) }
    propagate()
    // Writers of state no ungated reader demands (e.g. a ring written outside
    // a sample-gate and read only inside it) keep running unconditionally.
    let undemanded = graph.nodes.keys.filter { demands[$0] == nil }
    if !undemanded.isEmpty {
      for id in undemanded { enqueue(id, []) }
      propagate()
    }
    // A node a per-frame condition is computed from runs before that test, so
    // it cannot wait for it: drop the condition from the demand of its whole
    // value cone (a shared producer the body also uses runs ungated by it).
    for condition in sampleConditions {
      var cone = Set<NodeID>()
      var stack = [condition]
      while let id = stack.popLast() {
        guard cone.insert(id).inserted, let node = graph.nodes[id] else { continue }
        stack.append(contentsOf: node.allDependencies)
      }
      for id in cone {
        if let demand = demands[id] { demands[id] = demand.removing([condition]) }
      }
    }
    // A stateless node wanted by many alternative regions (e.g. arithmetic
    // shared by every voice) costs less to compute than to test for: run it
    // whenever the conditions every alternative shares hold. Its consumers
    // keep their own gates, and it holds no state, so a value computed from
    // inputs that did not run this frame is unused.
    for (id, demand) in demands where demand.terms.count > maxAlternatives {
      guard let node = graph.nodes[id], ScalarBlockCoalescingPass.isPlainScalarOp(node.op),
        node.op.persistentStateCellIds.isEmpty, stateCell(node) == nil else { continue }
      switch node.op {
      case .historyWrite, .historyReadWrite, .memoryWrite, .output, .seq: continue
      default:
        let common = demand.terms.dropFirst().reduce(demand.terms[0]) { $0.intersection($1) }
        demands[id] = ExecutionDemand(terms: [common])
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
    var plan = Plan(demands: demands, sampleConditions: sampleConditions)
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
