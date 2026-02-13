import Foundation

/// Folds constant expressions at graph level before scheduling and block partitioning.
extension GraphPrepPasses {
  static func foldConstants(_ graph: Graph, options: CompilationPipeline.Options) {
    // Track known constant values as nodes are folded in-place.
    var constantValues: [NodeID: Float] = [:]

    // Initialize with existing constants.
    for (nodeId, node) in graph.nodes {
      if case .constant(let value) = node.op {
        constantValues[nodeId] = value
      }
    }

    // Build consumer map: input -> [consumers].
    var consumers: [NodeID: [NodeID]] = [:]
    for (nodeId, node) in graph.nodes {
      for input in node.inputs {
        consumers[input, default: []].append(nodeId)
      }
    }

    // Initialize worklist with foldable nodes that have all-constant inputs.
    var worklist = Set<NodeID>()
    for (nodeId, node) in graph.nodes {
      if canFoldOp(node.op) && !node.inputs.isEmpty
        && node.inputs.allSatisfy({ constantValues[$0] != nil })
      {
        worklist.insert(nodeId)
      }
    }

    var foldedCount = 0

    while let nodeId = worklist.popFirst() {
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
      for consumer in consumers[nodeId] ?? [] {
        if let consumerNode = graph.nodes[consumer],
          canFoldOp(consumerNode.op),
          consumerNode.inputs.allSatisfy({ constantValues[$0] != nil })
        {
          worklist.insert(consumer)
        }
      }
    }

    if options.debug && foldedCount > 0 {
      print("Constant folding: folded \(foldedCount) nodes")
    }
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
    case .abs, .sign, .sin, .cos, .tan, .tanh, .exp, .log, .log10, .sqrt,
      .floor, .ceil, .round, .atan2:
      return true
    // Control flow (key for biquad)
    case .gswitch, .mix, .selector:
      return true
    default:
      return false
    }
  }

  private static func evaluateConstantOp(_ op: LazyOp, _ inputs: [Float]) -> Float? {
    switch op {
    // Unary
    case .abs: return inputs.count == 1 ? Swift.abs(inputs[0]) : nil
    case .sign: return inputs.count == 1 ? (inputs[0] > 0 ? 1 : (inputs[0] < 0 ? -1 : 0)) : nil
    case .sin: return inputs.count == 1 ? sin(inputs[0]) : nil
    case .cos: return inputs.count == 1 ? cos(inputs[0]) : nil
    case .tan: return inputs.count == 1 ? tan(inputs[0]) : nil
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
      let mode = Int(inputs[0])
      if mode <= 0 { return 0.0 }
      if mode <= inputs.count - 1 {
        return inputs[mode]
      }
      return 0.0  // Out of range

    default:
      return nil
    }
  }
}
