import Foundation

/// Exposes the opaque modulation operator's arithmetic to C execution gating.
/// The graph-level operator remains intact for differentiation and other
/// backends; these nodes exist only during one C compilation.
enum ModulationGateLoweringPass {
  struct Changes {
    var originals: [NodeID: Node] = [:]
    var added: [NodeID] = []

    func restore(graph: Graph) {
      for id in added { graph.nodes.removeValue(forKey: id) }
      for (id, node) in originals {
        graph.nodes[id] = node
        graph.executionGates.removeValue(forKey: id)
      }
    }
  }

  static func run(graph: Graph) -> Changes {
    var changes = Changes()
    var params: [CellID: NodeID] = [:]
    var inputs: [Int: NodeID] = [:]
    for id in graph.nodes.keys.sorted() {
      switch graph.nodes[id]!.op {
      case .param(let cell): if params[cell] == nil { params[cell] = id }
      case .input(let channel): if inputs[channel] == nil { inputs[channel] = id }
      default: break
      }
    }
    func make(_ op: LazyOp, _ operands: [NodeID] = []) -> NodeID {
      let id = graph.n(op, operands)
      changes.added.append(id)
      return id
    }
    func param(_ cell: CellID) -> NodeID {
      if let id = params[cell] { return id }
      let id = make(.param(cell)); params[cell] = id; return id
    }
    func input(_ channel: Int) -> NodeID {
      if let id = inputs[channel] { return id }
      let id = make(.input(channel)); inputs[channel] = id; return id
    }
    for id in graph.nodes.keys.sorted() {
      guard let node = graph.nodes[id],
        case .modulatedParam(let mode, let minimum, let maximum, let baseCell, let activeCell, let lanes) = node.op
      else { continue }
      let base = param(baseCell)
      let active = param(activeCell)
      var modulation = make(.constant(0))
      for lane in lanes {
        let product = make(.mul, [input(lane.modulatorChannel), param(lane.depthCellId)])
        modulation = make(.add, [modulation, product])
      }
      let value: NodeID
      switch mode {
      case .additive: value = make(.add, [base, modulation])
      case .multiplicative:
        value = make(.mul, [base, make(.add, [make(.constant(1)), modulation])])
      case .semitone:
        let semitones = make(.div, [modulation, make(.constant(12))])
        value = make(.mul, [base, make(.exp, [make(.mul, [make(.constant(logf(2))), semitones])])])
      }
      let bounded = make(.min, [make(.max, [value, make(.constant(minimum))]), make(.constant(maximum))])
      changes.originals[id] = node
      var mux = Node(id: id, op: .gswitch, inputs: [active, bounded, base])
      mux.temporalDependencies = node.temporalDependencies
      mux.shape = node.shape
      graph.nodes[id] = mux
      graph.executionGates[id] = active
    }
    return changes
  }
}
