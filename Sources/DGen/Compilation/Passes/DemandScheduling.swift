import Foundation

/// Orders the nodes of one gated sequential block so each gated region runs
/// as few conditional fragments as its dependencies allow.
///
/// The block keeps every value and ordering edge, a history write stays after
/// the reads of its cell, and accesses to one state cell keep their order
/// (reads between two writes may move among themselves). Within that, three
/// steps shape the order:
///
/// 1. A depth-first walk from ungated nodes and from sinks (state writes) in
///    their old order. Reaching a gated region's connected piece queues the
///    rest of that piece right after the current root, so a body is emitted
///    together instead of interleaved with the next one.
/// 2. Each node takes the priority of the earliest piece that needs it: a
///    region's ungated inputs run just before it, not in its middle.
/// 3. A list schedule by (priority, ungated first, walk position) that sticks
///    with the region it last scheduled while that region has ready nodes.
enum DemandScheduling {
  static func order(_ nodes: [NodeID], demands: [NodeID: ExecutionDemand], graph: Graph) -> [NodeID] {
    let count = nodes.count
    let preds = predecessors(nodes, graph: graph)
    // Each distinct demand as a small id; 0 is unconditional.
    var demandIds: [ExecutionDemand: Int] = [.always: 0]
    let demandOf = nodes.map { id -> Int in
      let demand = demands[id] ?? .always
      if let existing = demandIds[demand] { return existing }
      demandIds[demand] = demandIds.count
      return demandIds.count - 1
    }
    let always = 0

    // Connected pieces of each gated region (one gate used at both ends of a
    // chain gives two pieces).
    var piece = [Int](repeating: -1, count: count)
    var pieces: [[Int]] = []
    var adjacent = [[Int]](repeating: [], count: count)
    for (index, list) in preds.enumerated() where demandOf[index] != always {
      for p in list where demandOf[p] == demandOf[index] {
        adjacent[index].append(p)
        adjacent[p].append(index)
      }
    }
    for start in 0..<count where demandOf[start] != always && piece[start] < 0 {
      var members: [Int] = []
      var stack = [start]
      piece[start] = pieces.count
      while let index = stack.popLast() {
        members.append(index)
        for other in adjacent[index] where piece[other] < 0 {
          piece[other] = pieces.count
          stack.append(other)
        }
      }
      pieces.append(members.sorted())
    }

    // 1. Depth-first walk.
    var isPred = [Bool](repeating: false, count: count)
    for list in preds { for p in list { isPred[p] = true } }
    var done = [Bool](repeating: false, count: count)
    var onStack = [Bool](repeating: false, count: count)
    var walk: [Int] = []
    walk.reserveCapacity(count)
    var expanded = Set<Int>()
    var deferred: [Int] = []
    func note(_ index: Int) {
      if piece[index] >= 0, expanded.insert(piece[index]).inserted {
        deferred.append(contentsOf: pieces[piece[index]])
      }
    }
    func walkFrom(_ root: Int) {
      guard !done[root] else { return }
      var stack: [(index: Int, next: Int)] = [(root, 0)]
      onStack[root] = true
      note(root)
      while let (index, next) = stack.last {
        if next < preds[index].count {
          stack[stack.count - 1].next += 1
          let p = preds[index][next]
          if !done[p] && !onStack[p] {
            onStack[p] = true
            note(p)
            stack.append((p, 0))
          }
        } else {
          stack.removeLast()
          onStack[index] = false
          done[index] = true
          walk.append(index)
        }
      }
    }
    func visit(_ root: Int) {
      walkFrom(root)
      while !deferred.isEmpty {
        let batch = deferred
        deferred.removeAll()
        batch.forEach(walkFrom)
      }
    }
    for index in 0..<count where demandOf[index] == always || !isPred[index] { visit(index) }
    for index in 0..<count { visit(index) }
    guard walk.count == count else { return nodes }  // a cycle: keep the old order

    // 2. Priorities.
    var walkPosition = [Int](repeating: 0, count: count)
    for (position, index) in walk.enumerated() { walkPosition[index] = position }
    var priority = walkPosition
    let keyed = pieces.map { members in (key: members.map { walkPosition[$0] }.min()!, members: members) }
    for (key, members) in keyed.sorted(by: { $0.key < $1.key }) {
      var stack = members
      while let index = stack.popLast() {
        guard priority[index] > key else { continue }
        priority[index] = key
        stack.append(contentsOf: preds[index])
      }
    }

    // 3. Sticky list schedule.
    var succs = [[Int]](repeating: [], count: count)
    var pending = [Int](repeating: 0, count: count)
    for (index, list) in preds.enumerated() {
      pending[index] = list.count
      for p in list { succs[p].append(index) }
    }
    let less: (Int, Int) -> Bool = { a, b in
      (priority[a], demandOf[a] == always ? 0 : 1, walkPosition[a])
        < (priority[b], demandOf[b] == always ? 0 : 1, walkPosition[b])
    }
    var ready = IndexHeap(less: less)
    var readyByDemand: [Int: IndexHeap] = [:]
    var taken = [Bool](repeating: false, count: count)
    func makeReady(_ index: Int) {
      ready.push(index)
      if demandOf[index] != always {
        readyByDemand[demandOf[index], default: IndexHeap(less: less)].push(index)
      }
    }
    for index in 0..<count where pending[index] == 0 { makeReady(index) }
    var scheduled: [NodeID] = []
    scheduled.reserveCapacity(count)
    var current = always
    while scheduled.count < count {
      var pick: Int? = nil
      if current != always { pick = readyByDemand[current]?.popFirst { !taken[$0] } }
      if pick == nil { pick = ready.popFirst { !taken[$0] } }
      guard let index = pick else { return nodes }
      taken[index] = true
      current = demandOf[index]
      scheduled.append(nodes[index])
      for s in succs[index] {
        pending[s] -= 1
        if pending[s] == 0 { makeReady(s) }
      }
    }
    return scheduled
  }

  /// In-block predecessors of each node (by position): value and ordering
  /// edges, plus state-cell order.
  private static func predecessors(_ nodes: [NodeID], graph: Graph) -> [[Int]] {
    let position = Dictionary(uniqueKeysWithValues: nodes.enumerated().map { ($1, $0) })
    var preds = [[Int]](repeating: [], count: nodes.count)
    var lastWrite: [CellID: Int] = [:]
    var readsSince: [CellID: [Int]] = [:]
    for (index, id) in nodes.enumerated() {
      guard let node = graph.nodes[id] else { continue }
      for dep in node.allDependencies { if let p = position[dep] { preds[index].append(p) } }
      let (readCells, writeCells) = graph.stateAccess(node)
      for cell in readCells {
        if let w = lastWrite[cell] { preds[index].append(w) }
        readsSince[cell, default: []].append(index)
      }
      for cell in writeCells {
        if let w = lastWrite[cell] { preds[index].append(w) }
        preds[index].append(contentsOf: readsSince[cell] ?? [])
        readsSince[cell] = []
        lastWrite[cell] = index
      }
    }
    return preds.enumerated().map { index, list in Array(Set(list).subtracting([index])).sorted() }
  }
}

/// A binary min-heap of node positions under a caller-supplied order.
private struct IndexHeap {
  let less: (Int, Int) -> Bool
  private var items: [Int] = []

  init(less: @escaping (Int, Int) -> Bool) { self.less = less }

  mutating func push(_ x: Int) {
    items.append(x)
    var i = items.count - 1
    while i > 0, less(items[i], items[(i - 1) / 2]) {
      items.swapAt(i, (i - 1) / 2)
      i = (i - 1) / 2
    }
  }

  /// Removes entries until one satisfies `isLive`, and returns it.
  mutating func popFirst(where isLive: (Int) -> Bool) -> Int? {
    while !items.isEmpty {
      let top = items[0]
      let last = items.removeLast()
      if !items.isEmpty {
        items[0] = last
        var i = 0
        while true {
          let l = 2 * i + 1, r = l + 1
          var m = i
          if l < items.count, less(items[l], items[m]) { m = l }
          if r < items.count, less(items[r], items[m]) { m = r }
          if m == i { break }
          items.swapAt(i, m)
          i = m
        }
      }
      if isLive(top) { return top }
    }
    return nil
  }
}
