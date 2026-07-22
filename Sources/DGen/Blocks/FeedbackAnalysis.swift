/// Feedback detection, grouping, scalar classification, and topological sort.
import Foundation

// Find all nodes that participate in feedback loops (not just minimal cycles)
public func findFeedbackLoops(_ g: Graph) -> [[NodeID]] {
  // Build maps of history cells to their read/write nodes
  // historyRead/historyWrite now handle both scalar and tensor cases.
  // A cell can have MULTIPLE reads (e.g. the same tensor history read twice);
  // every one of them closes the write -> read feedback edge, so track them
  // all. Iterate node IDs sorted so the resulting clusters are deterministic
  // (dictionary order used to pick an arbitrary "winning" read per cell,
  // which made compilation of duplicate-read graphs nondeterministic).
  var cellReads: [Int: [NodeID]] = [:]
  var cellWrites: [Int: [NodeID]] = [:]

  for nodeId in g.nodes.keys.sorted() {
    guard let node = g.nodes[nodeId] else { continue }
    switch node.op {
    case .historyRead(let cell):
      cellReads[cell, default: []].append(nodeId)
    case .historyWrite(let cell):
      cellWrites[cell, default: []].append(nodeId)
    default:
      break
    }
  }

  // Build consumer map for efficient traversal
  var consumers = [NodeID: [NodeID]]()
  for (nodeId, node) in g.nodes {
    for dep in node.allDependencies {
      consumers[dep, default: []].append(nodeId)
    }
  }

  // Helper to find all nodes reachable forward from a set of nodes
  func reachForward(from nodes: Set<NodeID>) -> Set<NodeID> {
    var reached = nodes
    var queue = Array(nodes)

    while let node = queue.popLast() {
      // Add all regular consumers
      for consumer in consumers[node] ?? [] {
        if reached.insert(consumer).inserted {
          queue.append(consumer)
        }
      }

      // Handle implicit historyWrite -> historyRead connection
      if let n = g.nodes[node] {
        if case .historyWrite(let cellId) = n.op {
          for readNode in cellReads[cellId] ?? [] {
            if reached.insert(readNode).inserted {
              queue.append(readNode)
            }
          }
        }
      }
    }

    return reached
  }

  // Helper to find all nodes reachable backward from a set of nodes
  func reachBackward(from nodes: Set<NodeID>) -> Set<NodeID> {
    var reached = nodes
    var queue = Array(nodes)

    while let node = queue.popLast() {
      // Add all regular dependencies
      if let n = g.nodes[node] {
        for dep in n.allDependencies {
          if reached.insert(dep).inserted {
            queue.append(dep)
          }
        }
      }

      // Handle implicit historyRead -> historyWrite connection
      if let n = g.nodes[node] {
        if case .historyRead(let cellId) = n.op {
          for writeNode in cellWrites[cellId] ?? [] {
            if reached.insert(writeNode).inserted {
              queue.append(writeNode)
            }
          }
        }
      }
    }

    return reached
  }

  // Find feedback clusters - a cluster exists when any history write depends on any history read
  var clusters: [Set<NodeID>] = []
  var processedWrites = Set<NodeID>()
  let allWriteNodes = Set(cellWrites.values.flatMap { $0 })

  for cellId in cellWrites.keys.sorted() {
    for writeNode in cellWrites[cellId]!.sorted() {
    if processedWrites.contains(writeNode) { continue }

    // Find all nodes this write depends on
    let writeDeps = reachBackward(from: [writeNode])

    // Find all history reads (for this cell) that this write depends on
    let dependentReads = writeDeps.intersection(Set(cellReads[cellId] ?? []))

    if !dependentReads.isEmpty {
      // This write creates feedback - find all writes reachable from the reads it depends on
      var allReadsInCluster = dependentReads
      var allWritesInCluster = Set([writeNode])

      // OPTIMIZATION: Precompute which nodes can reach writeNode (done once)
      let canReachWriteNode = reachBackward(from: [writeNode])

      // Keep expanding until we find all connected reads and writes
      var changed = true
      while changed {
        changed = false

        // From all reads in cluster, find what writes they can reach
        let forwardFromReads = reachForward(from: allReadsInCluster)
        let newWrites = forwardFromReads.intersection(allWriteNodes)

        // OPTIMIZATION: Instead of reachForward per write, just check precomputed set
        for newWrite in newWrites {
          if canReachWriteNode.contains(newWrite) {
            if allWritesInCluster.insert(newWrite).inserted {
              changed = true
            }
          }
        }
      }

      // Mark all writes in this cluster as processed
      processedWrites.formUnion(allWritesInCluster)

      // Find ALL nodes that participate in the feedback computation
      // These are all nodes that are on any path from any read to any write in the cluster
      var clusterNodes = Set<NodeID>()

      // Add all reads and writes in the cluster
      clusterNodes.formUnion(allReadsInCluster)
      clusterNodes.formUnion(allWritesInCluster)

      // CRITICAL: Ensure read/write pairs for the same cell are both included
      // If a read is in the cluster, its corresponding write must also be included
      for readNode in allReadsInCluster {
        if let node = g.nodes[readNode] {
          if case .historyRead(let cell) = node.op {
            for writeNode in cellWrites[cell] ?? [] {
              clusterNodes.insert(writeNode)
              allWritesInCluster.insert(writeNode)
            }
          }
        }
      }

      // If a write is in the cluster, ALL reads of that cell must also be
      // included — a read left out of the cluster can be scheduled in a
      // different block, where it observes block-stale state instead of the
      // previous frame's value.
      for writeNode in allWritesInCluster {
        if let node = g.nodes[writeNode] {
          if case .historyWrite(let cell) = node.op {
            for readNode in cellReads[cell] ?? [] {
              clusterNodes.insert(readNode)
              allReadsInCluster.insert(readNode)
            }
          }
        }
      }

      // For each read in the cluster, find all nodes that can reach any write in the cluster
      // OPTIMIZATION: Instead of checking reachability per-node (O(N²)), compute:
      // 1. All nodes reachable forward from reads (done once)
      // 2. All nodes that can reach writes backward (done once)
      // 3. Intersection gives nodes on paths from reads to writes
      let forwardFromReadsAll = reachForward(from: allReadsInCluster)
      let backwardToWrites = reachBackward(from: allWritesInCluster)

      // Nodes on feedback paths = reachable from reads AND can reach writes
      let nodesOnFeedbackPaths = forwardFromReadsAll.intersection(backwardToWrites)
      clusterNodes.formUnion(nodesOnFeedbackPaths)

      // Same-cell + path closure. Two invariants:
      // (1) Every read and write of any history cell touched by the cluster
      //     must execute in the same sequential frame loop. Reads can enter
      //     the cluster via path analysis alone (e.g. biquad's x-history read
      //     feeding the y recursion) while the cell's write dangles outside;
      //     a write scheduled in a different kernel runs for ALL frames before
      //     the cluster's loop, so every frame reads the final frame's state.
      // (2) Any node on a path between cluster state ops (e.g. a consumer of
      //     a pass-through historyWrite output that feeds the recursion) must
      //     join the cluster, otherwise its group both depends on and is a
      //     dependency of the cluster group — a scheduling cycle.
      var closureChanged = true
      while closureChanged {
        closureChanged = false

        for nodeId in Array(clusterNodes) {
          guard let node = g.nodes[nodeId] else { continue }
          let cell: Int?
          switch node.op {
          case .historyRead(let c), .historyWrite(let c):
            cell = c
          default:
            cell = nil
          }
          guard let cell else { continue }
          for other in (cellReads[cell] ?? []) + (cellWrites[cell] ?? []) {
            if clusterNodes.insert(other).inserted { closureChanged = true }
          }
        }

        var stateNodes = Set<NodeID>()
        var stateWrites = Set<NodeID>()
        for nodeId in clusterNodes {
          guard let node = g.nodes[nodeId] else { continue }
          switch node.op {
          case .historyRead:
            stateNodes.insert(nodeId)
          case .historyWrite:
            stateNodes.insert(nodeId)
            stateWrites.insert(nodeId)
          default:
            break
          }
        }
        let onStatePaths = reachForward(from: stateNodes)
          .intersection(reachBackward(from: stateWrites))
        let beforeCount = clusterNodes.count
        clusterNodes.formUnion(onStatePaths)
        if clusterNodes.count != beforeCount { closureChanged = true }
      }

      // NOTE: For tensor feedback loops like Conv2D, nodes downstream of historyRead
      // but not on the path to historyWrite (like sumAxis -> output) also need
      // sequential execution. However, including ALL forward-reachable nodes is too
      // aggressive and breaks other tests. Tensor feedback loop support requires
      // nested loop execution (sequential frames, parallel tensor elements).
      // TODO: Implement proper nested loop execution for tensor feedback clusters.

      clusters.append(clusterNodes)
    }
    }
  }

  // Convert sets to arrays for compatibility
  return clusters.map { Array($0).sorted() }
}

// Determines feedback groups - groups of scalar nodes that must execute together
// Returns a mapping from nodeId to group index (-1 means not in a group)
public func determineFeedbackGroups(_ g: Graph, feedbackClusters: [[NodeID]]) -> [NodeID: Int] {
  var nodeToGroupId: [NodeID: Int] = [:]
  var nextGroupId = 0

  // Use findFeedbackLoops to identify all nodes in feedback loops
  let feedbackLoops = feedbackClusters

  // Assign group IDs to nodes in feedback loops
  for loop in feedbackLoops {
    let groupId = nextGroupId
    nextGroupId += 1

    for nodeId in loop {
      nodeToGroupId[nodeId] = groupId
    }
  }

  // Also handle other memory operations that aren't in feedback loops
  var reads = [Int: [NodeID]]()
  var writes = [Int: [NodeID]]()

  g.nodes.values.forEach {
    switch $0.op {
    case .historyRead(let c):
      reads[c, default: []].append($0.id)
    case .historyWrite(let c):
      writes[c, default: []].append($0.id)
    case .memoryRead(let c):
      reads[c, default: []].append($0.id)
    case .memoryWrite(let c):
      writes[c, default: []].append($0.id)
    case .accum(let c):
      writes[c, default: []].append($0.id)
    case .latch(let c):
      writes[c, default: []].append($0.id)
    case .phasor(let c):
      writes[c, default: []].append($0.id)
    default: break
    }
  }

  // Assign -1 to nodes not in any group
  for nodeId in g.nodes.keys {
    if nodeToGroupId[nodeId] == nil {
      nodeToGroupId[nodeId] = -1
    }
  }

  return nodeToGroupId
}

public func findSequentialNodes(_ g: Graph, feedbackClusters: [[NodeID]], backend: Backend = .metal) -> Set<
  NodeID
> {
  var scalar: Set<NodeID> = []

  // Track nodes that are SIMD-safe due to using atomics (should never be scalar)
  var simdSafe: Set<NodeID> = []
  g.nodes.values.forEach {
    switch $0.op {
    case .memoryAccumulate(_):
      // memoryAccumulate uses atomics - safe for SIMD execution
      simdSafe.insert($0.id)
    case .tensorAccumulate(_):
      // tensorAccumulate uses atomics internally - safe for SIMD execution
      simdSafe.insert($0.id)
    case .chunkPartialsReduceToCell(_, _, _, _, _):
      // Deterministic chunk reduction maps one thread per output element.
      simdSafe.insert($0.id)
    default: break
    }
  }

  // First, mark inherently scalar operations (stateful ops with frame-to-frame dependencies)
  g.nodes.values.forEach {
    switch $0.op {
    case .accum(_):
      scalar.insert($0.id)  // Accum operations need to be scalar (stateful)
    case .latch(_):
      scalar.insert($0.id)  // Latch operations need to be scalar (stateful)
    case .phasor(_):
      scalar.insert($0.id)  // Phasor operations need to be scalar (stateful)
    case .click(_):
      scalar.insert($0.id)  // Click reads/writes cell — needs sequential frame execution
    case .noise(_):
      scalar.insert($0.id)  // Noise PRNG reads/writes state cell — needs sequential frame execution
    case .temporalGradStore, .temporalGradRead:
      // These ops use explicit frame-indexed tape addressing. Metal maps that
      // safely to one thread per frame; the C backend's frame-axis SIMD upgrade
      // cannot represent their integer offsets lane-wise, so keep it scalar.
      if backend == .c { scalar.insert($0.id) }
    case .tensorNoise(_, _, _):
      scalar.insert($0.id)  // Tensor noise shares a single PRNG state across N per-frame iterations — must stay scalar so the xorshift advances sequentially rather than being SIMD-vectorized 4-at-once
    case .hopTensorNoise(_, _, _):
      scalar.insert($0.id)  // Same reasoning as tensorNoise — the shared xorshift state must step sequentially, and the hop gate is emitted as scalar `if`
    case .spectrumDelay(_, _, _, _, _):
      scalar.insert($0.id)  // Ring-buffer write + row advance must happen sequentially on hop boundaries; the op emits its own scalar `if` gate
    case .spectrumDelayMod(_, _, _, _, _):
      scalar.insert($0.id)  // Same reasoning — plus the fractional interpolation reads two ring rows per element
    case .historyRead, .historyWrite:
      // History is state across frames whether its cell is scalar or tensor.
      // Emitting a non-feedback read/write chain in a parallel per-frame kernel
      // races on the persistent state (biquad's x[n-1]/x[n-2] chain is the
      // canonical example). Feedback detection catches closed recurrences, but
      // not every stateful chain, so all history operations stay frame-serial;
      // tensor blocks still iterate/index their elements inside each frame.
      scalar.insert($0.id)
    case .memoryRead(let cellId), .memoryWrite(let cellId):
      // Vector-width gradient carry cells are the reverse-time analogue of
      // history: state carried across (reverse) frames. A parallel per-frame
      // kernel would race on them exactly like scalar history cells.
      if g.tensorGradCarryCells.contains(cellId) {
        scalar.insert($0.id)
      }
    default: break
    }
  }

  // Identify tensor source nodes for tracking
  var tensorNodes: Set<NodeID> = []
  g.nodes.values.forEach {
    switch $0.op {
    case .tensorRef(_):
      tensorNodes.insert($0.id)
    case .historyRead(let cellId):
      if g.cellToTensor[cellId] != nil {
        tensorNodes.insert($0.id)
      }
    default: break
    }
  }

  // Propagate tensor status through the graph (needed for both backends)
  var changed = true
  while changed {
    changed = false
    g.nodes.values.forEach { node in
      for inputId in node.inputs {
        if tensorNodes.contains(inputId) && !tensorNodes.contains(node.id) {
          tensorNodes.insert(node.id)
          changed = true
        }
      }
    }
  }

  // For C backend: Mark tensor ops as scalar (they need sequential element loops)
  // For Metal: Tensor ops can run SIMD across frames with internal parallelRange loops
  if case .c = backend {
    // C backend: conservative approach - tensor ops are scalar
    g.nodes.values.forEach {
      switch $0.op {
      case .historyRead(let cellId):
        if g.cellToTensor[cellId] != nil {
          scalar.insert($0.id)
        }
      case .historyWrite(let cellId):
        if g.cellToTensor[cellId] != nil {
          scalar.insert($0.id)
        }
      case .conv2d(_), .sum, .maxAxis, .meanAxis, .peek:
        scalar.insert($0.id)
      default: break
      }
    }

    // Mark nodes with tensor inputs as scalar for C
    g.nodes.values.forEach {
      if simdSafe.contains($0.id) { return }
      for inputId in $0.inputs {
        if tensorNodes.contains(inputId) {
          scalar.insert($0.id)
        }
      }
    }

    // Propagate scalar status through tensor chains for C
    changed = true
    while changed {
      changed = false
      g.nodes.values.forEach { node in
        if simdSafe.contains(node.id) { return }
        for inputId in node.inputs {
          if tensorNodes.contains(inputId) && !scalar.contains(node.id) {
            scalar.insert(node.id)
            changed = true
          }
        }
      }
    }
  }

  // Metal: historyReadWrite must be scalar — the SIMD delay1 segmented
  // dispatch is broken. Sequential frame-by-frame processing is correct
  // and outputs cross to SIMD blocks via the outbound tape mechanism.
  if case .metal = backend {
    g.nodes.values.forEach {
      if case .historyReadWrite(_) = $0.op {
        scalar.insert($0.id)
      }
    }
  }

  // Hop-gated tensor history feedback must run sequentially across frames: the
  // state at hop N depends on hop N-1. Element-parallelizing it across frames
  // (ThreadCountScale → perFrameScaled) would split the read+update into a
  // parallel kernel and the write-back into a separate sequential kernel, so
  // every hop would read stale state. Marking read/write scalar keeps the whole
  // feedback path in one sequential frame loop (matching the C tensor-history
  // path). Gated on the explicit hop tag so per-sample history (membrane / FDTD)
  // keeps its existing Metal SIMD-across-frames behavior.
  g.nodes.values.forEach {
    switch $0.op {
    case .historyRead(let cellId), .historyWrite(let cellId):
      if g.nodeHopRate[$0.id] != nil, g.cellToTensor[cellId] != nil {
        scalar.insert($0.id)
      }
    default:
      break
    }
  }

  // Use feedback loop detection to mark all nodes in feedback loops as scalar
  // This is the core reason for scalar execution - frame-to-frame state dependencies
  for loop in feedbackClusters {
    for nodeId in loop {
      scalar.insert(nodeId)
    }
  }

  // Vector-width BPTT (tensor-history biquad): the reverse-time recurrence —
  // carry reads, the grad arithmetic between them, and the carry writes — must
  // land in one sequential block so it can be wrapped in a single reverse
  // frame loop. Backward tensor grad nodes are otherwise classified parallel,
  // fragmenting the chain across blocks. Mark every backward-partition
  // ancestor of a tensor carry write as scalar.
  if !g.tensorGradCarryCells.isEmpty, let lastFwd = g.lastForwardNodeId {
    scalar.formUnion(tensorBPTTRecurrenceClosure(g: g, lastForwardId: lastFwd))
  }

  // Remove SIMD-safe nodes from scalar set - these use atomics and are safe for parallel execution
  scalar.subtract(simdSafe)

  return scalar
}

// Feedback group-aware topological sort
public func topologicalSort(
  _ g: Graph, feedbackClusters: [[NodeID]], scalarNodeSet: Set<NodeID>, debug: Bool = false
) -> [NodeID] {
  let nodeToGroupId = determineFeedbackGroups(g, feedbackClusters: feedbackClusters)

  // Handle seq operators - if any input to seq is scalar, make all inputs scalar
  let finalScalarSet = scalarNodeSet

  // Group nodes by feedback group (much simpler grouping)
  var groupToNodes: [Int: [NodeID]] = [:]
  var nodeToFinalGroupId: [NodeID: Int] = [:]
  var nextGroupId = 1000

  for nodeId in g.nodes.keys.sorted() {
    let groupId = nodeToGroupId[nodeId] ?? -1
    if groupId != -1 {
      // Node is in a real feedback group
      groupToNodes[groupId, default: []].append(nodeId)
      nodeToFinalGroupId[nodeId] = groupId
    } else {
      // Node gets its own group
      groupToNodes[nextGroupId] = [nodeId]
      nodeToFinalGroupId[nodeId] = nextGroupId
      nextGroupId += 1
    }
  }

  if debug {
    print("🏗️ Feedback groups:")
    for (groupId, nodes) in groupToNodes.sorted(by: { $0.key < $1.key })
    where groupId < 1000 {
      print("  FeedbackGroup \(groupId): \(nodes)")
    }
  }

  // Build group-level dependencies
  var groupDeps: [Int: Set<Int>] = [:]
  var groupIndegree: [Int: Int] = [:]

  // Initialize
  for groupId in groupToNodes.keys {
    groupDeps[groupId] = []
    groupIndegree[groupId] = 0
  }

  // Calculate dependencies between groups
  for (groupId, nodes) in groupToNodes {
    var depGroups = Set<Int>()
    for nodeId in nodes {
      if let node = g.nodes[nodeId] {
        for dep in node.allDependencies {
          if let depGroupId = nodeToFinalGroupId[dep] {
            if depGroupId != groupId {
              depGroups.insert(depGroupId)
            }
          } else {
          }
        }
      }
    }
    groupDeps[groupId] = depGroups
    groupIndegree[groupId] = depGroups.count
  }

  // Simple topological sort on groups (Kahn's algorithm)
  var queue = groupIndegree.filter { $0.value == 0 }.map { $0.key }.sorted()
  var sortedGroups: [Int] = []

  while let groupId = queue.first {
    queue.removeFirst()
    sortedGroups.append(groupId)

    // Update consumers
    for (otherGroupId, deps) in groupDeps {
      if deps.contains(groupId) {
        groupIndegree[otherGroupId]! -= 1
        if groupIndegree[otherGroupId] == 0 {
          // Insert in sorted order for determinism
          let insertIndex = queue.firstIndex { $0 > otherGroupId } ?? queue.count
          queue.insert(otherGroupId, at: insertIndex)
        }
      }
    }
  }

  // Handle cycles (simple fallback: add remaining in sorted order)
  let remaining = Set(groupToNodes.keys).subtracting(sortedGroups)
  if !remaining.isEmpty {
    if debug {
      print(
        "⚠️ Cycle detected, adding remaining groups in sorted order: \(remaining.sorted())"
      )
    }
    sortedGroups.append(contentsOf: remaining.sorted())
  }

  // Expand groups back to nodes
  var result: [NodeID] = []
  for groupId in sortedGroups {
    let nodes = groupToNodes[groupId]!

    if nodes.count == 1 {
      result.append(nodes[0])
    } else {
      // Simple topological sort within group
      result.append(contentsOf: topologicalSortWithinGroup(nodes: nodes, g: g))
    }
  }

  if debug {
    let nodeDescriptions = result.map { nodeId in
      let op = g.nodes[nodeId]?.op ?? .add
      let isScalar = finalScalarSet.contains(nodeId)
      let groupId = nodeToFinalGroupId[nodeId] ?? -1
      let color = isScalar ? ANSI.red : ""
      let reset = isScalar ? ANSI.reset : ""
      let groupMarker = groupId < 1000 ? "~\(groupId)" : ""
      return "\(color)(\(nodeId),\(op)\(groupMarker))\(reset)"
    }
    print("📋 Topo sort: [\(nodeDescriptions.joined(separator: " | "))]")
  }

  return result
}

/// Topological sort within a feedback group that separates forward and gradient nodes.
/// If lastForwardNodeId is set, forward nodes (id <= lastForwardNodeId) are processed
/// before gradient nodes (id > lastForwardNodeId) to prevent gradient additions from
/// affecting forward node ordering.
private func topologicalSortWithinGroup(nodes: [NodeID], g: Graph) -> [NodeID] {
  let lastForwardId = g.lastForwardNodeId

  // Separate forward and gradient nodes if applicable
  let forwardNodes: [NodeID]
  let gradientNodes: [NodeID]
  if let lastFwd = lastForwardId {
    forwardNodes = nodes.filter { $0 <= lastFwd }.sorted()
    gradientNodes = nodes.filter { $0 > lastFwd }.sorted()
  } else {
    forwardNodes = nodes.sorted()
    gradientNodes = []
  }

  // Topological sort a subset of nodes using Kahn's algorithm
  func topoSort(_ subsetNodes: [NodeID]) -> [NodeID] {
    guard !subsetNodes.isEmpty else { return [] }
    let subsetSet = Set(subsetNodes)
    var indegree: [NodeID: Int] = [:]

    // Calculate in-degrees (only counting deps within the subset)
    for nodeId in subsetNodes {
      var count = 0
      if let node = g.nodes[nodeId] {
        for dep in node.allDependencies where subsetSet.contains(dep) {
          count += 1
        }
      }
      indegree[nodeId] = count
    }

    // Kahn's algorithm
    var queue = indegree.filter { $0.value == 0 }.map { $0.key }.sorted()
    var result: [NodeID] = []

    while let nodeId = queue.first {
      queue.removeFirst()
      result.append(nodeId)

      // Update consumers within subset
      for otherNodeId in subsetNodes {
        if let otherNode = g.nodes[otherNodeId],
          otherNode.allDependencies.contains(nodeId)
        {
          indegree[otherNodeId]! -= 1
          if indegree[otherNodeId] == 0 {
            let insertIndex = queue.firstIndex { $0 > otherNodeId } ?? queue.count
            queue.insert(otherNodeId, at: insertIndex)
          }
        }
      }
    }

    // Add any remaining nodes (cycles)
    let remaining = subsetSet.subtracting(result)
    result.append(contentsOf: remaining.sorted())

    return result
  }

  // Sort forward nodes first, then gradient nodes
  var result = topoSort(forwardNodes)
  result.append(contentsOf: topoSort(gradientNodes))

  return result
}
