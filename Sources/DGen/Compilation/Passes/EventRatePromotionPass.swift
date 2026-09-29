import Foundation

/// Schedules math on latched coefficients at event rate, automatically.
///
/// `latch(v, trigger)` only changes on trigger frames, whatever `v` is. Pure scalar math over
/// such latches and other frame-invariant values is therefore piecewise
/// constant: it can only change on a trigger frame, or on the first frame of a
/// process call, when parameters may have been written. Written plainly, every
/// one of those expressions still runs per sample. Instruments that latch
/// per-hit controls (a drum hit's tuning, velocity curve, selected table row)
/// pay for hundreds of coefficient expressions every frame.
///
/// This pass rewrites each such region onto the `event-hold` machinery, with
/// the event clock firing on `trigger`, or on the first frame of a process call
/// when a frame-invariant input of the region changed (or on the first call):
///
///   - region nodes read each root latch on the event clock;
///   - region nodes then run only on event frames (TemporalityPass);
///   - values the region hands to frame-rate code return via `eventLatch`,
///     which keeps the consumer loop SIMD (one event test per lane group).
///
/// Results are identical to the per-sample form: region values are recomputed
/// on every frame where any input can have changed.
///
/// The rewrite is scoped to one C compilation and restored afterwards.
enum EventRatePromotionPass {
  static var isEnabled: Bool {
    ProcessInfo.processInfo.environment["DGEN_DISABLE_EVENT_PROMOTION"] == nil
  }

  struct Changes {
    var originals: [NodeID: Node] = [:]
    var addedNodes: Set<NodeID> = []
    var clockKeys: [NodeID] = []
    var hopRateKeys: [NodeID] = []
    var sparseClocks: [NodeID] = []

    func restore(graph: Graph) {
      for id in addedNodes {
        graph.nodes.removeValue(forKey: id)
        graph.schedulingKeys.removeValue(forKey: id)
      }
      for (id, node) in originals { graph.nodes[id] = node }
      for key in clockKeys {
        if let clock = graph.eventHoldClocks.removeValue(forKey: key) {
          graph.eventClockNodes.remove(clock)
        }
      }
      for key in hopRateKeys { graph.nodeHopRate.removeValue(forKey: key) }
      for clock in sparseClocks { graph.sparseEventClocks.remove(clock) }
    }
  }

  struct Report {
    var triggers = 0
    var roots = 0
    var promotedNodes = 0
    var frontier = 0
  }

  static func run(graph: Graph, debug: Bool = false) -> Changes {
    var changes = Changes()
    guard !graph.hasComputedGradients else { return changes }
    let order = topologicalOrder(graph)
    let consumers = consumerMap(graph)
    let invariant = frameInvariantNodes(graph, order: order)

    // Root latches, grouped by trigger node.
    var rootTrigger: [NodeID: NodeID] = [:]
    var triggers: [NodeID] = []
    for id in order {
      // Whatever it samples, a latch only changes on its trigger's frames.
      guard let node = graph.nodes[id], case .latch = node.op, node.inputs.count == 2,
        isScalar(node), graph.nodeHopRate[id] == nil
      else { continue }
      let trigger = node.inputs[1]
      guard graph.nodeHopRate[trigger] == nil, !graph.eventClockNodes.contains(trigger) else {
        continue
      }
      if !triggers.contains(trigger) { triggers.append(trigger) }
      rootTrigger[id] = trigger
    }
    guard !rootTrigger.isEmpty else { return changes }

    // Grow each trigger's region through pure scalar math whose inputs are all
    // frame-invariant or already in the same region.
    var region: [NodeID: NodeID] = rootTrigger
    for id in order where region[id] == nil {
      guard let node = graph.nodes[id], isPromotable(node, graph: graph),
        node.temporalDependencies.isEmpty
      else { continue }
      var trigger: NodeID?
      var eligible = true
      for input in node.inputs {
        if let t = region[input] {
          if trigger == nil { trigger = t } else if trigger != t { eligible = false; break }
        } else if !invariant.contains(input) {
          eligible = false
          break
        }
      }
      if eligible, let trigger { region[id] = trigger }
    }

    let usesOutput = Set(graph.nodes.values.flatMap { $0.temporalDependencies })
    var report = Report()
    var blockStart: NodeID?
    var changedFlags: [NodeID: NodeID] = [:]
    func make(_ op: LazyOp, _ inputs: [NodeID]) -> NodeID {
      let id = graph.n(op, inputs)
      changes.addedNodes.insert(id)
      return id
    }
    func setInputs(_ id: NodeID, _ transform: ([NodeID]) -> [NodeID]) {
      guard let node = graph.nodes[id] else { return }
      if changes.originals[id] == nil, !changes.addedNodes.contains(id) {
        changes.originals[id] = node
      }
      var rewired = Node(id: id, op: node.op, inputs: transform(node.inputs))
      rewired.temporalDependencies = node.temporalDependencies
      rewired.shape = node.shape
      graph.nodes[id] = rewired
    }

    for trigger in triggers {
      let members = order.filter { region[$0] == trigger }
      let roots = members.filter { rootTrigger[$0] != nil }
      let rootSet = Set(roots)
      var inRegion = Set(members.filter { rootTrigger[$0] == nil })
      func isFrontier(_ id: NodeID) -> Bool {
        (consumers[id] ?? []).contains { !inRegion.contains($0) } || usesOutput.contains(id)
      }
      // Latching a value back costs about `latchCost` per lane group. A cheap
      // value whose region inputs are already latched back (or are roots) is
      // cheaper to recompute per frame, so demote it; repeat until stable.
      var demoted = true
      while demoted {
        demoted = false
        for id in members.reversed() where inRegion.contains(id) {
          guard !(consumers[id] ?? []).contains(where: { inRegion.contains($0) }),
            !usesOutput.contains(id), let node = graph.nodes[id]
          else { continue }
          let newlyExposed = Set(node.inputs).filter {
            inRegion.contains($0) && !(consumers[$0] ?? []).contains { c in c != id && !inRegion.contains(c) }
          }
          if frameCost(node) + latchCost * newlyExposed.count < latchCost {
            inRegion.remove(id)
            demoted = true
          }
        }
      }
      let interior = members.filter { inRegion.contains($0) }
      guard !interior.isEmpty else { continue }
      let memberSet = inRegion.union(rootSet)
      // Interior values read by frame-rate code (or by anything outside the
      // region) must be latched back to frame rate.
      let frontier = interior.filter(isFrontier)
      let benefit = interior.reduce(0) { $0 + frameCost(graph.nodes[$1]!) }
      guard benefit > latchCost * frontier.count else { continue }

      let start = blockStart ?? make(.blockStart, [])
      blockStart = start
      let zero = make(.constant(0), [])
      // Frame-invariant inputs change only between process calls. Re-run the
      // region at a call's first frame only when one of them did (or on the
      // first call), not on every call: `changed` is hoisted to once per call.
      var live: [NodeID] = []
      for id in interior {
        for input in graph.nodes[id]!.inputs where invariant.contains(input) {
          // Constants and immutable tables never change; only scalar values can.
          guard let node = graph.nodes[input], isScalar(node) else { continue }
          switch node.op {
          case .constant, .tensorRef: continue
          default: break
          }
          if !live.contains(input) { live.append(input) }
        }
      }
      if live.isEmpty { live.append(zero) }
      var dirty: NodeID?
      for input in live {
        let flag: NodeID
        if let existing = changedFlags[input] {
          flag = existing
        } else {
          flag = make(.changed(graph.alloc(), graph.alloc()), [input])
          changedFlags[input] = flag
        }
        dirty = dirty.map { make(.max, [$0, flag]) } ?? flag
      }
      let clockTrigger = make(.max, [make(.gt, [trigger, zero]), make(.mul, [start, dirty!])])

      let clocksBefore = Set(graph.eventHoldClocks.keys)
      let hopBefore = Set(graph.nodeHopRate.keys)
      let nodesBefore = Set(graph.nodes.keys)

      // Region nodes run only on event frames, where each root's frame-rate
      // value is current, so they read the root through a clock-tagged
      // pass-through instead of a second, frame-serial `eventHold` latch.
      let clock = graph.eventClock(for: clockTrigger)
      graph.sparseEventClocks.insert(clock)
      changes.sparseClocks.append(clock)
      var held: [NodeID: NodeID] = [:]
      for root in roots where (consumers[root] ?? []).contains(where: { memberSet.contains($0) }) {
        let read = make(.add, [root, zero])
        graph.nodeHopRate[read] = (1, clock)
        // Event frames are known only once the clock is computed. `eventHold`
        // reaches its clock through its predicate input; this read must be
        // ordered after the clock explicitly.
        graph.nodes[read]!.temporalDependencies.append(clock)
        held[root] = read
      }
      for id in interior {
        setInputs(id) { $0.map { held[$0] ?? $0 } }
      }
      // One latch per value here; `localizeEventLatches` gives each consuming
      // block its own copy once blocks exist.
      var keys: [NodeID: NodeID] = held.reduce(into: [:]) { $0[$1.value] = $1.key }
      for id in frontier {
        let back = graph.eventLatch(id, when: clockTrigger)
        keys[back] = id
        for consumer in consumers[id] ?? [] where !memberSet.contains(consumer) {
          setInputs(consumer) { $0.map { $0 == id ? back : $0 } }
        }
      }

      // New nodes sort where the node they stand in for would; the clock
      // plumbing right after the trigger.
      let added = Set(graph.nodes.keys).subtracting(nodesBefore)
      for id in added { graph.schedulingKeys[id] = keys[id] ?? trigger }
      changes.addedNodes.formUnion(added)
      changes.clockKeys += Array(Set(graph.eventHoldClocks.keys).subtracting(clocksBefore))
      changes.hopRateKeys += Array(Set(graph.nodeHopRate.keys).subtracting(hopBefore))
      report.triggers += 1
      report.roots += held.count
      report.promotedNodes += interior.count
      report.frontier += frontier.count
    }
    if debug, report.triggers > 0 {
      print(
        "[event-promotion] \(report.triggers) triggers, \(report.roots) roots, "
          + "\(report.promotedNodes) nodes at event rate, \(report.frontier) latched back")
    }
    return changes
  }

  // MARK: - Block-local latches

  /// Gives every frame-rate block that reads an `eventLatch` its own copy.
  ///
  /// Inside the block that computes it, a held value is a register broadcast.
  /// Read from another block it must travel through a frame tape: a store and
  /// a load per lane group, and hundreds of held values spill tapes out of L1.
  /// A copy is cheap (a broadcast in its consumer's loop) and needs its own
  /// cell: a later loop cannot share the first loop's cell, which by then holds
  /// the block's last event value for every frame. Originals left without
  /// readers are dropped. Returns the added nodes (frame-rate) for temporality.
  static func localizeEventLatches(blocks: inout [Block], graph: Graph, changes: inout Changes)
    -> [NodeID]
  {
    var home: [NodeID: Int] = [:]
    for (index, block) in blocks.enumerated() {
      for id in block.nodes { home[id] = index }
    }
    func isEventLatch(_ id: NodeID) -> Bool {
      if case .eventLatch = graph.nodes[id]?.op { return true }
      return false
    }
    var added: [NodeID] = []
    for index in blocks.indices where blocks[index].temporality == .frameBased {
      var copies: [NodeID: NodeID] = [:]
      var nodes: [NodeID] = []
      for id in blocks[index].nodes {
        guard let node = graph.nodes[id] else { nodes.append(id); continue }
        var inputs = node.inputs
        for (slot, input) in inputs.enumerated() where isEventLatch(input) && home[input] != index {
          if copies[input] == nil {
            let original = graph.nodes[input]!
            let cell = graph.alloc()
            graph.persistentCells.insert(cell)
            let copy = graph.n(.eventLatch(cell), original.inputs, shape: original.shape)
            graph.nodes[copy]!.temporalDependencies = original.temporalDependencies
            copies[input] = copy
            nodes.append(copy)
            added.append(copy)
            changes.addedNodes.insert(copy)
          }
          inputs[slot] = copies[input]!
        }
        if inputs != node.inputs {
          if changes.originals[id] == nil, !changes.addedNodes.contains(id) {
            changes.originals[id] = node
          }
          var rewired = Node(id: id, op: node.op, inputs: inputs)
          rewired.temporalDependencies = node.temporalDependencies
          rewired.shape = node.shape
          graph.nodes[id] = rewired
        }
        nodes.append(id)
      }
      blocks[index].nodes = nodes
    }
    guard !added.isEmpty else { return added }
    var read = Set<NodeID>()
    for node in graph.nodes.values {
      read.formUnion(node.inputs)
      read.formUnion(node.temporalDependencies)
    }
    for index in blocks.indices {
      blocks[index].nodes.removeAll { isEventLatch($0) && !read.contains($0) }
    }
    return added
  }

  // MARK: - Scheduling

  /// Moves each event-rate node up to just after its last dependency.
  ///
  /// Topological order interleaves an event region with the frame-rate code
  /// around it, which splits both into many small loops (and small parallel
  /// fragments are then coalesced into scalar loops). Anchoring event nodes to
  /// their latest dependency keeps each region contiguous without reordering
  /// any frame-rate node. Feedback clusters stay contiguous: a node anchored
  /// inside one moves past its end. Returns `sorted` unchanged if the result
  /// would not be topological.
  static func clusterEventNodes(
    sorted: [NodeID], graph: Graph, hopBasedNodes: [NodeID: (Int, NodeID)],
    feedbackClusters: [[NodeID]]
  ) -> [NodeID] {
    func isEvent(_ id: NodeID) -> Bool {
      hopBasedNodes[id].map { graph.eventClockNodes.contains($0.1) } ?? false
    }
    guard sorted.contains(where: isEvent) else { return sorted }
    var position: [NodeID: Int] = [:]
    for (index, id) in sorted.enumerated() { position[id] = index }
    var clusterEnd: [NodeID: Int] = [:]
    for cluster in feedbackClusters {
      let end = cluster.compactMap { position[$0] }.max() ?? -1
      for id in cluster { clusterEnd[id] = end }
    }
    var anchor: [NodeID: Int] = [:]
    var buckets: [Int: [NodeID]] = [:]
    for id in sorted where isEvent(id) {
      var at = -1
      for dep in graph.nodes[id]?.allDependencies ?? [] {
        if let a = anchor[dep] {
          at = max(at, a)
        } else if let p = position[dep] {
          at = max(at, clusterEnd[dep] ?? p)
        }
      }
      anchor[id] = at
      buckets[at, default: []].append(id)
    }
    var result = buckets[-1] ?? []
    result.reserveCapacity(sorted.count)
    for (index, id) in sorted.enumerated() where anchor[id] == nil {
      result.append(id)
      if let bucket = buckets[index] { result.append(contentsOf: bucket) }
    }
    var placed = Set<NodeID>()
    let members = Set(sorted)
    let debug = ProcessInfo.processInfo.environment["DGEN_DEBUG_EVENT_SCHEDULE"] != nil
    for id in result {
      for dep in graph.nodes[id]?.allDependencies ?? [] where members.contains(dep) {
        if !placed.contains(dep) {
          if debug { print("[event-schedule] fallback: \(id) before its input \(dep)") }
          return sorted
        }
      }
      placed.insert(id)
    }
    if debug { print("[event-schedule] moved \(anchor.count) event nodes into \(buckets.count) runs") }
    return result.count == sorted.count ? result : sorted
  }

  // MARK: - Classification

  private static func isScalar(_ node: Node) -> Bool {
    if case .tensor = node.shape { return false }
    return true
  }

  /// Pure scalar ops that are safe and cheap to evaluate only on event frames.
  private static func isPromotable(_ node: Node, graph: Graph) -> Bool {
    guard isScalar(node), graph.nodeHopRate[node.id] == nil else { return false }
    switch node.op {
    case .add, .sub, .mul, .div, .mod, .min, .max, .abs, .sign, .floor, .ceil, .round,
      .sin, .cos, .tan, .atan, .tanh, .exp, .log, .log10, .sqrt, .pow, .atan2,
      .gt, .gte, .lt, .lte, .eq, .and, .or, .xor, .gswitch, .selector, .mix, .neg:
      return true
    case .peek:
      // Scalar lookups in immutable stored tables only.
      guard let source = node.inputs.first, let stored = graph.nodes[source],
        case .tensorRef(let tensorId) = stored.op, let tensor = graph.tensors[tensorId],
        tensor.data != nil, tensor.transforms.isEmpty,
        !graph.mutableTensorCells.contains(tensor.cellId)
      else { return false }
      return true
    default:
      return false
    }
  }

  /// Approximate NEON cycles per four-frame lane group for an `eventLatch`
  /// read (event test, held-cell load, broadcast, tape store).
  static let latchCost = 4

  /// Approximate NEON cycles per four-frame lane group to compute `node` at
  /// frame rate: what promoting it saves on frames without events.
  private static func frameCost(_ node: Node) -> Int {
    switch node.op {
    case .pow, .atan2: return 30
    case .exp, .log, .log10, .sin, .cos, .tan, .tanh, .atan: return 16
    case .peek, .mod: return 8
    case .sqrt: return 4
    case .div: return 3
    case .mix: return 2
    case .selector: return node.inputs.count
    default: return 1
    }
  }

  /// Nodes whose value cannot change within one process call.
  private static func frameInvariantNodes(_ graph: Graph, order: [NodeID]) -> Set<NodeID> {
    var invariant = Set<NodeID>()
    for id in order {
      guard let node = graph.nodes[id] else { continue }
      if TemporalityPass.isIntrinsicallyFrameBased(node.op) || graph.isMutableTensorAccess(node)
        || graph.nodeHopRate[id] != nil || graph.eventClockNodes.contains(id)
      {
        continue
      }
      switch node.op {
      case .blockStart, .eventLatch: continue
      default: break
      }
      let deps = node.inputs + node.temporalDependencies
      if deps.allSatisfy({ invariant.contains($0) }) { invariant.insert(id) }
    }
    return invariant
  }

  private static func consumerMap(_ graph: Graph) -> [NodeID: [NodeID]] {
    var map: [NodeID: [NodeID]] = [:]
    for node in graph.nodes.values {
      for input in Set(node.inputs) { map[input, default: []].append(node.id) }
    }
    return map
  }

  /// Inputs before consumers; nodes on cycles are left out (never promoted).
  private static func topologicalOrder(_ graph: Graph) -> [NodeID] {
    var order: [NodeID] = []
    var state: [NodeID: Int] = [:]  // 1 visiting, 2 done
    for start in graph.nodes.keys.sorted() {
      guard state[start] == nil else { continue }
      var stack: [(NodeID, Int)] = [(start, 0)]
      state[start] = 1
      while let (id, next) = stack.last {
        let deps = (graph.nodes[id]?.inputs ?? []) + (graph.nodes[id]?.temporalDependencies ?? [])
        if next < deps.count {
          stack[stack.count - 1].1 = next + 1
          let dep = deps[next]
          if state[dep] == nil, graph.nodes[dep] != nil {
            state[dep] = 1
            stack.append((dep, 0))
          }
        } else {
          stack.removeLast()
          state[id] = 2
          order.append(id)
        }
      }
    }
    return order
  }
}
