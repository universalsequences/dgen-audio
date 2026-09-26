import Foundation

/// Folds constant expressions at graph level before scheduling and block partitioning.
extension GraphPrepPasses {
  static func foldConstants(_ graph: Graph, options: CompilationPipeline.Options) {
    // Track known constant values as nodes are folded in-place.
    var constantValues: [NodeID: Float] = [:]

    // Initialize with existing constants.
    for nodeId in graph.nodes.keys.sorted() {
      guard let node = graph.nodes[nodeId] else { continue }
      if case .constant(let value) = node.op {
        constantValues[nodeId] = value
      }
    }

    // Build consumer map: input -> [consumers].
    var consumers: [NodeID: [NodeID]] = [:]
    for nodeId in graph.nodes.keys.sorted() {
      guard let node = graph.nodes[nodeId] else { continue }
      for input in node.inputs {
        consumers[input, default: []].append(nodeId)
      }
    }

    // Initialize worklist with foldable nodes that have all-constant inputs.
    var worklist: [NodeID] = []
    var queued = Set<NodeID>()
    for nodeId in graph.nodes.keys.sorted() {
      guard let node = graph.nodes[nodeId] else { continue }
      if canFoldOp(node.op) && !node.inputs.isEmpty
        && node.inputs.allSatisfy({ constantValues[$0] != nil })
      {
        worklist.append(nodeId)
        queued.insert(nodeId)
      }
    }

    var foldedCount = 0

    while !worklist.isEmpty {
      let nodeId = worklist.removeFirst()
      queued.remove(nodeId)
      guard let node = graph.nodes[nodeId] else { continue }

      let inputValues = node.inputs.compactMap { constantValues[$0] }
      guard inputValues.count == node.inputs.count else { continue }

      guard let result = evaluateConstantOp(node.op, inputValues), result.isFinite else {
        continue
      }

      // Replace node with constant (preserves NodeID, no rewiring needed).
      constantValues[nodeId] = result
      graph.nodes[nodeId] = Node(id: nodeId, op: .constant(result), inputs: [])
      foldedCount += 1

      // Add newly-eligible consumers to worklist.
      for consumer in (consumers[nodeId] ?? []).sorted() {
        if let consumerNode = graph.nodes[consumer],
          canFoldOp(consumerNode.op),
          consumerNode.inputs.allSatisfy({ constantValues[$0] != nil }),
          !queued.contains(consumer)
        {
          worklist.append(consumer)
          queued.insert(consumer)
        }
      }
    }

    if options.debug && foldedCount > 0 {
      print("Constant folding: folded \(foldedCount) nodes")
    }
  }

  /// `DGEN_NO_ALGEBRAIC_FOLD=1` disables `foldAlgebraicIdentities` for A/B runs.
  static var algebraicFoldingEnabled: Bool {
    ProcessInfo.processInfo.environment["DGEN_NO_ALGEBRAIC_FOLD"] != "1"
  }

  /// Node replacements made by `foldAlgebraicIdentities`, undone after one compile.
  struct AlgebraicFoldChanges {
    var originals: [NodeID: Node] = [:]

    func restore(graph: Graph) {
      for (id, node) in originals { graph.nodes[id] = node }
    }
  }

  /// Folds scalar nodes whose result is decided by SOME constant operands:
  /// `selector`/`gswitch`/`mix` with a constant index, `x * 0`, and the
  /// identities `x + 0`, `x - 0`, `x * 1`, `x / 1`.
  ///
  /// Must run before feedback analysis. A literal-algorithm macro such as
  /// `(* fb (selector 2 1 0 1 ...))` or `(* b1 gain (selector 2 0 0 1 ...))` is
  /// always zero, but the unfolded graph still carries an edge from the
  /// feedback history into every downstream lookup, so `findFeedbackLoops`
  /// pulls feedback-free work into the frame-serial loop. C emission and clang
  /// fold these later, after the loop layout is already decided.
  ///
  /// `x * 0 -> 0` relies on DGen's finite-only fast-math contract. The graph is
  /// shared with the lazy front end, so every rewrite is recorded for restore.
  static func foldAlgebraicIdentities(_ graph: Graph) -> AlgebraicFoldChanges {
    var changes = AlgebraicFoldChanges()
    var pinned = Set<NodeID>(graph.executionGates.keys)
    pinned.formUnion(graph.executionGates.values)
    pinned.formUnion(graph.nodeToTensor.keys)
    pinned.formUnion(graph.nodeHopRate.keys)
    pinned.formUnion(graph.nodeHopRate.values.map { $0.1 })
    pinned.formUnion(graph.eventHoldClocks.keys)
    pinned.formUnion(graph.eventHoldClocks.values)
    pinned.formUnion(graph.eventClockNodes)
    pinned.formUnion(graph.nodePositionDep.keys)
    pinned.formUnion(graph.nodePositionDep.values)
    pinned.formUnion(graph.tensorGradCells.keys)
    pinned.formUnion(graph.frameAwareCellClocks.values)
    pinned.formUnion(graph.materializeNodes)
    pinned.formUnion(graph.gradientSideEffects)
    pinned.formUnion(graph.simdOptimizedConv2Ds)
    pinned.formUnion(graph.conv2dMaskCells.keys)
    if let last = graph.lastForwardNodeId { pinned.insert(last) }

    func isScalar(_ id: NodeID) -> Bool {
      guard let node = graph.nodes[id], graph.nodeToTensor[id] == nil else { return false }
      if case .tensor? = node.shape { return false }
      return true
    }
    func constant(_ id: NodeID) -> Float? {
      if case .constant(let value)? = graph.nodes[id]?.op { return value }
      return nil
    }
    func record(_ id: NodeID) {
      if changes.originals[id] == nil, let node = graph.nodes[id] { changes.originals[id] = node }
    }

    enum Fold { case value(Float), alias(NodeID) }
    func fold(_ node: Node) -> Fold? {
      let ins = node.inputs
      let values = ins.map(constant)
      if canFoldOp(node.op), !ins.isEmpty, values.allSatisfy({ $0 != nil }),
        let result = evaluateConstantOp(node.op, values.map { $0! }), result.isFinite
      {
        return .value(result)
      }
      switch node.op {
      case .add where ins.count == 2:
        if values[1] == 0 { return .alias(ins[0]) }
        if values[0] == 0 { return .alias(ins[1]) }
      case .sub where ins.count == 2:
        if values[1] == 0 { return .alias(ins[0]) }
      case .mul where ins.count == 2:
        if values[0] == 0 || values[1] == 0 { return .value(0) }
        if values[1] == 1 { return .alias(ins[0]) }
        if values[0] == 1 { return .alias(ins[1]) }
      case .div where ins.count == 2:
        if values[1] == 1 { return .alias(ins[0]) }
      case .gswitch where ins.count == 3:
        if let cond = values[0] { return .alias(cond > 0 ? ins[1] : ins[2]) }
      case .mix where ins.count == 3:
        if values[2] == 0 { return .alias(ins[0]) }
        if values[2] == 1 { return .alias(ins[1]) }
      case .selector where ins.count >= 2:
        guard let mode = values[0] else { return nil }
        if let index = selectorInputIndex(mode: mode, optionCount: ins.count - 1) {
          return .alias(ins[index])
        }
        return .value(0)
      default:
        break
      }
      return nil
    }

    // `replacement` maps a folded-away node to the node its consumers use now.
    var replacement: [NodeID: NodeID] = [:]
    func resolve(_ id: NodeID) -> NodeID {
      var current = id
      while let next = replacement[current] { current = next }
      return current
    }

    var changed = true
    while changed {
      changed = false
      for id in graph.nodes.keys.sorted() {
        guard var node = graph.nodes[id] else { continue }
        let inputs = node.inputs.map(resolve)
        let temporal = node.temporalDependencies.map(resolve)
        if inputs != node.inputs || temporal != node.temporalDependencies {
          record(id)
          var rewired = Node(id: id, op: node.op, inputs: inputs)
          rewired.temporalDependencies = temporal
          rewired.shape = node.shape
          graph.nodes[id] = rewired
          node = rewired
          changed = true
        }
        guard replacement[id] == nil, !pinned.contains(id), node.temporalDependencies.isEmpty,
          isScalar(id), let result = fold(node)
        else { continue }
        switch result {
        case .value(let value):
          record(id)
          var folded = Node(id: id, op: .constant(value), inputs: [])
          folded.shape = .scalar
          graph.nodes[id] = folded
          changed = true
        case .alias(let target):
          guard target != id, isScalar(target) else { continue }
          replacement[id] = target
          changed = true
        }
      }
    }
    return changes
  }

  private static func canFoldOp(_ op: LazyOp) -> Bool {
    switch op {
    // Arithmetic
    case .add, .sub, .mul, .div, .pow, .mod, .min, .max:
      return true
    // Comparisons
    case .gt, .gte, .lt, .lte, .eq:
      return true
    // Logical
    case .and, .or, .xor:
      return true
    // Unary math
    case .abs, .sign, .sin, .cos, .tan, .atan, .tanh, .exp, .log, .log10, .sqrt,
      .floor, .ceil, .round, .atan2:
      return true
    // Control flow (key for biquad)
    case .gswitch, .mix, .selector:
      return true
    default:
      return false
    }
  }

  /// Input index (1-based, past the mode) a constant-mode selector picks, or nil when it
  /// yields 0. Mirrors the emitted `mode <= i` comparison chain (CRenderer, IRBuilder.selector)
  /// so fractional modes round up and NaN/out-of-range modes produce 0 without trapping.
  private static func selectorInputIndex(mode: Float, optionCount: Int) -> Int? {
    guard optionCount >= 1, mode > 0 else { return nil }
    return (1...optionCount).first { mode <= Float($0) }
  }

  private static func evaluateConstantOp(_ op: LazyOp, _ inputs: [Float]) -> Float? {
    switch op {
    // Unary
    case .abs: return inputs.count == 1 ? Swift.abs(inputs[0]) : nil
    case .sign: return inputs.count == 1 ? (inputs[0] > 0 ? 1 : (inputs[0] < 0 ? -1 : 0)) : nil
    case .sin: return inputs.count == 1 ? sin(inputs[0]) : nil
    case .cos: return inputs.count == 1 ? cos(inputs[0]) : nil
    case .tan: return inputs.count == 1 ? tan(inputs[0]) : nil
    case .atan: return inputs.count == 1 ? atan(inputs[0]) : nil
    case .tanh: return inputs.count == 1 ? tanh(inputs[0]) : nil
    case .exp: return inputs.count == 1 ? exp(inputs[0]) : nil
    case .log: return inputs.count == 1 && inputs[0] > 0 ? log(inputs[0]) : nil
    case .log10: return inputs.count == 1 && inputs[0] > 0 ? log10(inputs[0]) : nil
    case .sqrt: return inputs.count == 1 && inputs[0] >= 0 ? sqrt(inputs[0]) : nil
    case .floor: return inputs.count == 1 ? floor(inputs[0]) : nil
    case .ceil: return inputs.count == 1 ? ceil(inputs[0]) : nil
    case .round: return inputs.count == 1 ? round(inputs[0]) : nil

    // Binary
    case .add: return inputs.count == 2 ? inputs[0] + inputs[1] : nil
    case .sub: return inputs.count == 2 ? inputs[0] - inputs[1] : nil
    case .mul: return inputs.count == 2 ? inputs[0] * inputs[1] : nil
    case .div: return inputs.count == 2 && inputs[1] != 0 ? inputs[0] / inputs[1] : nil
    case .pow: return inputs.count == 2 ? pow(inputs[0], inputs[1]) : nil
    case .mod:
      return inputs.count == 2 && inputs[1] != 0
        ? inputs[0].truncatingRemainder(dividingBy: inputs[1]) : nil
    case .min: return inputs.count == 2 ? Swift.min(inputs[0], inputs[1]) : nil
    case .max: return inputs.count == 2 ? Swift.max(inputs[0], inputs[1]) : nil
    case .atan2: return inputs.count == 2 ? atan2(inputs[0], inputs[1]) : nil

    // Comparisons (return 1.0 for true, 0.0 for false)
    case .gt: return inputs.count == 2 ? (inputs[0] > inputs[1] ? 1 : 0) : nil
    case .gte: return inputs.count == 2 ? (inputs[0] >= inputs[1] ? 1 : 0) : nil
    case .lt: return inputs.count == 2 ? (inputs[0] < inputs[1] ? 1 : 0) : nil
    case .lte: return inputs.count == 2 ? (inputs[0] <= inputs[1] ? 1 : 0) : nil
    case .eq: return inputs.count == 2 ? (inputs[0] == inputs[1] ? 1 : 0) : nil

    // Logical
    case .and: return inputs.count == 2 ? ((inputs[0] != 0 && inputs[1] != 0) ? 1 : 0) : nil
    case .or: return inputs.count == 2 ? ((inputs[0] != 0 || inputs[1] != 0) ? 1 : 0) : nil
    case .xor: return inputs.count == 2 ? (((inputs[0] != 0) != (inputs[1] != 0)) ? 1 : 0) : nil

    // Ternary (key for biquad mode selection)
    case .gswitch:
      // gswitch(cond, ifTrue, ifFalse): returns ifTrue if cond > 0
      return inputs.count == 3 ? (inputs[0] > 0 ? inputs[1] : inputs[2]) : nil
    case .mix:
      // mix(a, b, t) = a * (1-t) + b * t
      return inputs.count == 3 ? inputs[0] * (1 - inputs[2]) + inputs[1] * inputs[2] : nil

    // N-ary (key for biquad mode selection)
    case .selector:
      // selector(mode, options...): 1-indexed, mode<=0 returns 0
      guard inputs.count >= 2 else { return nil }
      if let index = selectorInputIndex(mode: inputs[0], optionCount: inputs.count - 1) {
        return inputs[index]
      }
      return 0.0

    default:
      return nil
    }
  }
}
