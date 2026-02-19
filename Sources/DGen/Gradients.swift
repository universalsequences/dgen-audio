import Foundation

// MARK: - Gradient Computation (tinygrad-style)
//
// Instead of emitting backward IR directly, we build new graph nodes.
// This means:
// - One emission path (forward handles everything)
// - Gradients are just more LazyOps in the graph
// - Block scheduling doesn't need to know about "forward" vs "backward"
// - Parallelization falls out naturally from tensor ops
//
// For temporal operations (historyRead/historyWrite), we use gradient carry cells
// to pass gradients backwards through time, similar to BPTT.

extension Graph {

  /// Get or create a gradient carry cell for a history cell
  public func getGradCarryCell(for historyCellId: CellID) -> CellID {
    if let existing = gradCarryCells[historyCellId] {
      return existing
    }
    let carryCell = alloc()
    gradCarryCells[historyCellId] = carryCell
    return carryCell
  }

  /// Compute gradients by building new nodes in the graph.
  /// Returns a mapping from original forward nodes to their gradient nodes.
  ///
  /// - Parameters:
  ///   - loss: The loss node (root of backward pass)
  ///   - targets: Set of nodes we want gradients for (e.g., parameters)
  /// - Returns: Dictionary mapping forward NodeID -> gradient NodeID
  public func computeGradients(loss: NodeID, targets: Set<NodeID>) -> [NodeID: NodeID] {
    var grads: [NodeID: NodeID] = [:]
    gradientSideEffects = []
    accumulatedGradProxyCellByNode.removeAll()

    // Seed: gradient of loss w.r.t. itself = 1.0
    grads[loss] = n(.constant(1.0), [])

    // Walk in reverse topological order
    let reverseOrder = reverseTopologicalOrder(from: loss, targets: targets)
    for nodeId in reverseOrder {
      guard let upstreamGrad = grads[nodeId],
        let node = nodes[nodeId]
      else { continue }

      // Apply backward rule -> get gradient NodeIDs for inputs
      // Also handles side effects like storing to gradient carry cells
      // For ops that store per-frame data in cells (like spectralLossFFT), the backward
      // pass needs to run AFTER the forward op. We sequence gradOutput with nodeId
      // to create this ordering dependency.
      let sequencedGrad: NodeID
      if case .spectralLossFFT = node.op {
        sequencedGrad = n(.seq, [nodeId, upstreamGrad])
      } else {
        sequencedGrad = upstreamGrad
      }
      let inputGrads = node.op.backward(graph: self, node: node, gradOutput: sequencedGrad)

      // Accumulate gradients (as graph nodes, not values!)
      for (inputId, grad) in zip(node.inputs, inputGrads) {
        guard let grad = grad else { continue }

        if let existing = grads[inputId] {
          // Multiple paths contribute to this gradient.
          // For accumulated grad proxies (same backing grad cell), chain with seq
          // so we keep one read of the shared accumulated cell instead of adding
          // two tensor reads (which double-counts and can explode kernel work).
          if let existingCell = accumulatedGradProxyCellByNode[existing],
            let newCell = accumulatedGradProxyCellByNode[grad],
            existingCell == newCell
          {
            let merged = n(.seq, [existing, grad])
            accumulatedGradProxyCellByNode[merged] = existingCell
            grads[inputId] = merged
          } else {
            grads[inputId] = n(.add, [existing, grad])
          }
        } else {
          grads[inputId] = grad
        }
      }
    }

    // BPTT: Create carry cell writes for historyRead nodes.
    // historyRead has no graph inputs, so it's pruned from reverseTopologicalOrder
    // and its backward never runs. But the gradient w.r.t. historyRead's output
    // (accumulated in grads[]) is the temporal carry that should be written to the
    // carry cell for the previous frame to read.
    for (nodeId, node) in nodes {
      if case .historyRead(let cellId) = node.op, let gradNodeId = grads[nodeId] {
        let carryCell = getGradCarryCell(for: cellId)
        let zero = n(.seq, [gradNodeId, n(.constant(0.0), [])])
        let writeNode = n(.memoryWrite(carryCell), [zero, gradNodeId])
        addGradientSideEffect(writeNode)
      }
    }

    return grads
  }

  /// Add a side-effect node that should execute during gradient computation
  public func addGradientSideEffect(_ nodeId: NodeID) {
    gradientSideEffects.append(nodeId)
  }

  /// Compute reverse topological order from loss, only including nodes on paths to targets.
  private func reverseTopologicalOrder(from root: NodeID, targets: Set<NodeID>) -> [NodeID] {
    // First, find which nodes are on paths to targets
    var onTargetPath: [NodeID: Bool] = [:]

    for nodeId in topologicalOrder(from: root) {
      guard let node = nodes[nodeId] else { continue }
      let isTarget = targets.contains(nodeId)
      let hasTargetDescendant = node.inputs.contains { onTargetPath[$0] == true }
      onTargetPath[nodeId] = isTarget || hasTargetDescendant
    }

    // Now collect nodes in topological order, filtering to target paths
    let topo = topologicalOrder(from: root).filter { onTargetPath[$0] == true }

    // Return reversed
    return topo.reversed()
  }

  /// Standard topological sort from a root node.
  private func topologicalOrder(from root: NodeID) -> [NodeID] {
    var visited = Set<NodeID>()
    var order: [NodeID] = []

    func visit(_ nodeId: NodeID) {
      if visited.contains(nodeId) { return }
      visited.insert(nodeId)

      if let node = nodes[nodeId] {
        for input in node.inputs {
          visit(input)
        }
      }
      order.append(nodeId)
    }

    visit(root)
    return order
  }
}

// MARK: - LazyOp Backward Rules

extension LazyOp {

  /// Returns gradient NodeIDs for each input (nil = no gradient / non-differentiable).
  /// This builds NEW nodes in the graph rather than emitting IR directly.
  func backward(graph g: Graph, node: Node, gradOutput: NodeID) -> [NodeID?] {
    switch self {

    // MARK: Binary Arithmetic

    case .add:
      // d(x+y)/dx = 1, d(x+y)/dy = 1
      // When broadcasting, need to sum along broadcast dimensions
      let lhs = node.inputs[0]
      let rhs = node.inputs[1]
      var gradLhs = gradOutput
      var gradRhs = gradOutput

      if let lhsNode = g.nodes[lhs], let rhsNode = g.nodes[rhs] {
        gradLhs = reduceBroadcastGradient(g, grad: gradOutput, toShape: lhsNode.shape)
        gradRhs = reduceBroadcastGradient(g, grad: gradOutput, toShape: rhsNode.shape)
      }

      return [gradLhs, gradRhs]

    case .sub:
      // d(x-y)/dx = 1, d(x-y)/dy = -1
      // When broadcasting, need to sum along broadcast dimensions
      let lhs = node.inputs[0]
      let rhs = node.inputs[1]
      let negGrad = g.n(.neg, [gradOutput])
      var gradLhs = gradOutput
      var gradRhs = negGrad

      if let lhsNode = g.nodes[lhs], let rhsNode = g.nodes[rhs] {
        gradLhs = reduceBroadcastGradient(g, grad: gradOutput, toShape: lhsNode.shape)
        gradRhs = reduceBroadcastGradient(g, grad: negGrad, toShape: rhsNode.shape)
      }

      return [gradLhs, gradRhs]

    case .mul:
      // d(x*y)/dx = y * grad, d(x*y)/dy = x * grad
      // When broadcasting, need to sum along broadcast dimensions
      let lhs = node.inputs[0]
      let rhs = node.inputs[1]
      var gradLhs = g.n(.mul, [rhs, gradOutput])
      var gradRhs = g.n(.mul, [lhs, gradOutput])

      // Reduce gradients to match input shapes (handles broadcasting)
      if let lhsNode = g.nodes[lhs], let rhsNode = g.nodes[rhs] {
        gradLhs = reduceBroadcastGradient(g, grad: gradLhs, toShape: lhsNode.shape)
        gradRhs = reduceBroadcastGradient(g, grad: gradRhs, toShape: rhsNode.shape)
      }

      return [gradLhs, gradRhs]

    case .div:
      // d(x/y)/dx = grad / y
      // d(x/y)/dy = -grad * x / y^2 = -grad * (x/y) / y
      // When broadcasting, need to sum along broadcast dimensions
      let lhs = node.inputs[0]
      let rhs = node.inputs[1]
      var gradLhs = g.n(.div, [gradOutput, rhs])
      // For rhs: -grad * lhs / (rhs * rhs)
      let rhsSquared = g.n(.mul, [rhs, rhs])
      let lhsOverRhsSq = g.n(.div, [lhs, rhsSquared])
      let negGrad = g.n(.neg, [gradOutput])
      var gradRhs = g.n(.mul, [negGrad, lhsOverRhsSq])

      // Reduce gradients to match input shapes (handles broadcasting)
      if let lhsNode = g.nodes[lhs], let rhsNode = g.nodes[rhs] {
        gradLhs = reduceBroadcastGradient(g, grad: gradLhs, toShape: lhsNode.shape)
        gradRhs = reduceBroadcastGradient(g, grad: gradRhs, toShape: rhsNode.shape)
      }

      return [gradLhs, gradRhs]

    case .mod:
      // d(fmod(a,b))/da ~ 1, d/db ~ 0 (for DSP wrapping)
      let zero = g.n(.constant(0.0), [])
      return [gradOutput, zero]

    case .pow:
      // d(x^y)/dx = y * x^(y-1) * grad
      // d(x^y)/dy = x^y * ln(x) * grad
      let base = node.inputs[0]
      let exp = node.inputs[1]
      let one = g.n(.constant(1.0), [])
      let expMinusOne = g.n(.sub, [exp, one])
      let powExpM1 = g.n(.pow, [base, expMinusOne])
      let gradBase = g.n(.mul, [g.n(.mul, [exp, powExpM1]), gradOutput])

      let powXY = g.n(.pow, [base, exp])
      let lnBase = g.n(.log, [base])
      let gradExp = g.n(.mul, [g.n(.mul, [powXY, lnBase]), gradOutput])
      return [gradBase, gradExp]

    case .atan2:
      // d(atan2(y,x))/dy = x / (x^2 + y^2) * grad
      // d(atan2(y,x))/dx = -y / (x^2 + y^2) * grad
      let y = node.inputs[0]
      let x = node.inputs[1]
      let xSq = g.n(.mul, [x, x])
      let ySq = g.n(.mul, [y, y])
      let denom = g.n(.add, [xSq, ySq])
      let gradY = g.n(.mul, [g.n(.div, [x, denom]), gradOutput])
      let negY = g.n(.neg, [y])
      let gradX = g.n(.mul, [g.n(.div, [negY, denom]), gradOutput])
      return [gradY, gradX]

    case .min:
      // d(min(x,y))/dx = (x <= y) ? grad : 0
      // d(min(x,y))/dy = (y < x) ? grad : 0
      let x = node.inputs[0]
      let y = node.inputs[1]
      let zero = g.n(.constant(0.0), [])
      let xLeY = g.n(.lte, [x, y])
      let yLtX = g.n(.lt, [y, x])
      let gradX = g.n(.gswitch, [xLeY, gradOutput, zero])
      let gradY = g.n(.gswitch, [yLtX, gradOutput, zero])
      return [gradX, gradY]

    case .max:
      // d(max(x,y))/dx = (x >= y) ? grad : 0
      // d(max(x,y))/dy = (y > x) ? grad : 0
      let x = node.inputs[0]
      let y = node.inputs[1]
      let zero = g.n(.constant(0.0), [])
      let xGeY = g.n(.gte, [x, y])
      let yGtX = g.n(.gt, [y, x])
      let gradX = g.n(.gswitch, [xGeY, gradOutput, zero])
      let gradY = g.n(.gswitch, [yGtX, gradOutput, zero])
      return [gradX, gradY]

    case .mse:
      // loss = (a - b)^2
      // d/da = 2*(a-b) * grad
      // d/db = -2*(a-b) * grad
      let a = node.inputs[0]
      let b = node.inputs[1]
      let diff = g.n(.sub, [a, b])
      let two = g.n(.constant(2.0), [])
      let twoDiff = g.n(.mul, [two, diff])
      let gradA = g.n(.mul, [twoDiff, gradOutput])
      let gradB = g.n(.neg, [g.n(.mul, [twoDiff, gradOutput])])
      return [gradA, gradB]

    // MARK: Unary Math

    case .neg:
      // d(-x)/dx = -grad
      return [g.n(.neg, [gradOutput])]

    case .abs:
      // d(|x|)/dx = sign(x) * grad
      let x = node.inputs[0]
      let signX = g.n(.sign, [x])
      return [g.n(.mul, [signX, gradOutput])]

    case .sign:
      // sign is not differentiable (zero gradient)
      return [g.n(.constant(0.0), [])]

    case .sin:
      // d(sin(x))/dx = cos(x) * grad
      let x = node.inputs[0]
      let cosX = g.n(.cos, [x])
      return [g.n(.mul, [cosX, gradOutput])]

    case .cos:
      // d(cos(x))/dx = -sin(x) * grad
      let x = node.inputs[0]
      let sinX = g.n(.sin, [x])
      let negSinX = g.n(.neg, [sinX])
      let grad = g.n(.mul, [negSinX, gradOutput])
      return [grad]

    case .tan:
      // d(tan(x))/dx = 1/cos^2(x) * grad = sec^2(x) * grad
      let x = node.inputs[0]
      let cosX = g.n(.cos, [x])
      let cosXSq = g.n(.mul, [cosX, cosX])
      let one = g.n(.constant(1.0), [])
      let secSq = g.n(.div, [one, cosXSq])
      return [g.n(.mul, [secSq, gradOutput])]

    case .tanh:
      // d(tanh(x))/dx = (1 - tanh^2(x)) * grad
      let x = node.inputs[0]
      let tanhX = g.n(.tanh, [x])
      let tanhSq = g.n(.mul, [tanhX, tanhX])
      let one = g.n(.constant(1.0), [])
      let oneMinusTanhSq = g.n(.sub, [one, tanhSq])
      return [g.n(.mul, [oneMinusTanhSq, gradOutput])]

    case .exp:
      // d(exp(x))/dx = exp(x) * grad
      let x = node.inputs[0]
      let expX = g.n(.exp, [x])
      return [g.n(.mul, [expX, gradOutput])]

    case .log:
      // d(ln(x))/dx = (1/x) * grad
      let x = node.inputs[0]
      return [g.n(.div, [gradOutput, x])]

    case .log10:
      // d(log10(x))/dx = 1/(x * ln(10)) * grad
      let x = node.inputs[0]
      let ln10 = g.n(.constant(2.302585092994046), [])
      let xTimesLn10 = g.n(.mul, [x, ln10])
      return [g.n(.div, [gradOutput, xTimesLn10])]

    case .sqrt:
      // d(sqrt(x))/dx = 1/(2*sqrt(x)) * grad
      let x = node.inputs[0]
      let sqrtX = g.n(.sqrt, [x])
      let two = g.n(.constant(2.0), [])
      let twoSqrtX = g.n(.mul, [two, sqrtX])
      return [g.n(.div, [gradOutput, twoSqrtX])]

    case .floor, .ceil, .round:
      // Not differentiable, gradient = 0
      return [g.n(.constant(0.0), [])]

    // MARK: Comparisons (non-differentiable)

    case .gt, .gte, .lt, .lte, .eq:
      let zero = g.n(.constant(0.0), [])
      return [zero, zero]

    // MARK: Control Flow

    case .gswitch:
      // gswitch(cond, x, y) = cond ? x : y
      // d/dcond = 0 (non-differentiable)
      // d/dx = cond ? grad : 0
      // d/dy = cond ? 0 : grad
      let cond = node.inputs[0]
      let zero = g.n(.constant(0.0), [])
      let gradX = g.n(.gswitch, [cond, gradOutput, zero])
      let gradY = g.n(.gswitch, [cond, zero, gradOutput])
      return [zero, gradX, gradY]

    case .selector:
      // selector(mode, options...) -> gradient flows to selected option only
      guard node.inputs.count >= 2 else { return [] }
      let mode = node.inputs[0]
      let zero = g.n(.constant(0.0), [])

      var grads: [NodeID?] = [zero]  // mode has zero gradient
      for i in 1..<node.inputs.count {
        let optionIndex = g.n(.constant(Float(i - 1)), [])
        let isSelected = g.n(.eq, [mode, optionIndex])
        let gradOption = g.n(.gswitch, [isSelected, gradOutput, zero])
        grads.append(gradOption)
      }
      return grads

    case .mix:
      // mix(x, y, t) = x * (1-t) + y * t
      // d/dx = (1-t) * grad
      // d/dy = t * grad
      // d/dt = (y - x) * grad
      let x = node.inputs[0]
      let y = node.inputs[1]
      let t = node.inputs[2]
      let one = g.n(.constant(1.0), [])
      let oneMinusT = g.n(.sub, [one, t])
      let gradX = g.n(.mul, [oneMinusT, gradOutput])
      let gradY = g.n(.mul, [t, gradOutput])
      let yMinusX = g.n(.sub, [y, x])
      let gradT = g.n(.mul, [yMinusX, gradOutput])
      return [gradX, gradY, gradT]

    // MARK: Tensor Reductions

    case .sum:
      // d(sum(x))/dx[i] = grad for all i (broadcast)
      // Note: We return the scalar gradient directly instead of using expand.
      // Downstream ops (like peekRow) that need the gradient for tensor elements
      // can use this scalar directly since the gradient is uniform across all elements.
      // This avoids the tensor allocation timing issue where computeGradients runs
      // before allocateTensorOutputs.

      // TODO - the comment above seems sketchy, investigate if this is correct
      if let nodeInput = g.nodes[node.inputs[0]],
        case .tensor(let shape) = nodeInput.shape
      {
        let expand = g.n(.expand(shape), [gradOutput])
        return [expand]
      }
      return [gradOutput]
    case .sumAxis(let axis):
      // Gradient broadcasts back along the reduced axis
      guard let inputNode = g.nodes[node.inputs[0]],
        case .tensor(let inputShape) = inputNode.shape
      else {
        return [gradOutput]
      }
      // Expand gradient back to input shape along the reduced axis
      return [g.n(.expandAxis(inputShape, axis), [gradOutput])]

    case .maxAxis(let axis):
      // Gradient flows only to max element(s), split evenly on ties:
      // mask = (input == max_expanded), normMask = mask / sum(mask, axis), grad = normMask * grad_expanded
      guard let inputNode = g.nodes[node.inputs[0]],
        case .tensor(let inputShape) = inputNode.shape
      else {
        return [gradOutput]
      }
      let input = node.inputs[0]
      let maxExpanded = g.n(.expandAxis(inputShape, axis), [node.id])
      let mask = g.n(.eq, [input, maxExpanded])
      let tieCount = try! g.sum(mask, axis: axis)
      let tieExpanded = g.n(.expandAxis(inputShape, axis), [tieCount])
      let normMask = g.n(.div, [mask, tieExpanded])
      let gradExpanded = g.n(.expandAxis(inputShape, axis), [gradOutput])
      return [g.n(.mul, [normMask, gradExpanded])]

    case .meanAxis(let axis):
      // Gradient is uniformly distributed: grad_input = grad_output / axis_size, broadcast back
      guard let inputNode = g.nodes[node.inputs[0]],
        case .tensor(let inputShape) = inputNode.shape
      else {
        return [gradOutput]
      }
      let scale = g.n(.constant(1.0 / Float(inputShape[axis])))
      let scaled = g.n(.mul, [gradOutput, scale])
      return [g.n(.expandAxis(inputShape, axis), [scaled])]

    // MARK: Tensor Shape Operations

    case .reshape(_):
      // Gradient needs to be reshaped back to original shape
      guard let inputNode = g.nodes[node.inputs[0]],
        case .tensor(let origShape) = inputNode.shape
      else {
        return [gradOutput]
      }
      return [g.n(.reshape(origShape), [gradOutput])]

    case .asStrided(_, _):
      // asStrided gradient: reshape back to original shape
      // (The strided view is used for pool/im2col, gradient flows back to original layout)
      guard let inputNode = g.nodes[node.inputs[0]],
        case .tensor(let origShape) = inputNode.shape
      else {
        return [gradOutput]
      }
      return [g.n(.reshape(origShape), [gradOutput])]

    case .transpose(let perm):
      // Inverse permutation for gradient
      let inversePerm = invertPermutation(perm)
      return [g.n(.transpose(inversePerm), [gradOutput])]

    case .shrink(let ranges):
      // Gradient needs to be padded back
      guard let inputNode = g.nodes[node.inputs[0]],
        case .tensor(let origShape) = inputNode.shape
      else {
        return [gradOutput]
      }
      // Convert shrink ranges to pad amounts
      let padAmounts = ranges.enumerated().map { (i, range) -> (Int, Int) in
        if let r = range {
          return (r.0, origShape[i] - r.1)
        } else {
          return (0, 0)
        }
      }
      return [g.n(.pad(padAmounts), [gradOutput])]

    case .pad(let amounts):
      // Gradient needs to be shrunk
      guard let inputNode = g.nodes[node.inputs[0]],
        case .tensor(let origShape) = inputNode.shape
      else {
        return [gradOutput]
      }
      // Convert pad amounts to shrink ranges
      let shrinkRanges: [(Int, Int)?] = amounts.enumerated().map { (i, amt) in
        return (amt.0, origShape[i] + amt.0)
      }
      return [g.n(.shrink(shrinkRanges), [gradOutput])]

    case .expand, .expandAxis:
      // expand backward is sum along expanded dimensions
      // For now, return the gradient (TODO: implement proper reduction)
      return [gradOutput]

    case .expandView(let targetShape):
      // expandView backward: sum along dimensions that were expanded (size-1 -> larger)
      // For each dim where input was 1 and output is > 1, we need to sum the gradients
      guard let inputNode = g.nodes[node.inputs[0]],
        case .tensor(let inputShape) = inputNode.shape
      else {
        return [gradOutput]
      }

      // Find which axes were expanded (input size 1 -> target size > 1)
      var result = gradOutput
      for (axis, (inSize, outSize)) in zip(inputShape, targetShape).enumerated().reversed() {
        if inSize == 1 && outSize > 1 {
          // This dimension was broadcast - need to sum gradients along it
          result = try! g.sum(result, axis: axis)
        }
      }
      // sumAxis removes dimensions, but we need to keep them as size-1
      // Reshape back to original input shape
      result = g.n(.reshape(inputShape), [result])
      return [result]

    case .repeatView(_):
      // repeatView backward: sum over the repeated tiles to get gradient for original
      // For now, return the gradient (TODO: implement proper reduction for repeated dims)
      return [gradOutput]

    // MARK: Stateful Operations

    case .phasor(_):
      // d(phase)/d(freq) = frameIndex / sampleRate
      let sampleRate = g.n(.constant(g.sampleRate), [])
      return [g.n(.gradPhasor(node.id), [gradOutput, sampleRate])]

    case .deterministicPhasor:
      // d(phase)/d(freq) = frameIndex / sampleRate
      // Similar to phasor but stateless
      let sampleRate = g.n(.constant(g.sampleRate), [])
      return [g.n(.gradDeterministicPhasor, [gradOutput, sampleRate])]

    case .accum(_):
      // Accumulator gradient is complex due to temporal dependencies
      // For now, pass gradient to increment input, zero to others
      let zero = g.n(.constant(0.0), [])
      // inputs: [increment, reset, min, max]
      return [gradOutput, zero, zero, zero]

    case .latch(_):
      // Gradient flows through value when condition was true
      // inputs: [value, condition]
      let cond = node.inputs[1]
      let zero = g.n(.constant(0.0), [])
      let gradValue = g.n(.gswitch, [g.n(.gt, [cond, zero]), gradOutput, zero])
      return [gradValue, zero]

    case .historyRead:
      // historyRead has no graph inputs, so no gradients to return.
      // The carry cell write for BPTT is handled in computeGradients() after
      // the main loop, using the accumulated gradient from grads[historyRead].
      return []

    case .historyWrite(let cellId):
      // historyWrite is pass-through: it stores the input AND outputs it.
      // The gradient for the input is gradOutput (from downstream loss) PLUS
      // the temporal carry gradient (from future timestep's historyRead.backward).
      let carryCell = g.getGradCarryCell(for: cellId)
      // Use seq to order the carry read AFTER gradOutput, ensuring the memoryRead
      // node is a backward node (nodeId > lastForwardNodeId) and stays in the
      // BPTT block alongside other backward ops.
      let zero = g.n(.seq, [gradOutput, g.n(.constant(0.0), [])])
      // Read gradient from carry cell (from future timestep)
      let carryGrad = g.n(.memoryRead(carryCell), [zero])
      // Pass through downstream gradient + temporal carry
      return [g.n(.add, [gradOutput, carryGrad])]

    case .historyReadWrite(let cellId):
      // Combined read/write - stores gradOutput to carry, reads carry as input gradient
      let carryCell = g.getGradCarryCell(for: cellId)
      let zero = g.n(.constant(0.0), [])
      // Read current carry (from future)
      let carryGrad = g.n(.memoryRead(carryCell), [zero])
      // Store gradOutput for previous timestep
      _ = g.n(.memoryWrite(carryCell), [zero, gradOutput])
      return [carryGrad]

    // MARK: I/O and Constants

    case .constant(_):
      return []  // No inputs

    case .input(_):
      return []  // Leaf node

    case .output(_):
      return [gradOutput]  // Pass through

    case .param(_):
      return []  // Leaf node (but we want gradients FOR these)

    case .tensorRef(_):
      // TensorRef is a tensor parameter
      // Gradients for tensor parameters are handled via gradCarryCells
      // which are populated by ops like peekRow that read from this tensor
      return []  // Leaf node

    case .seq:
      // Detect bufferView pattern: seq(memoryWrite(cellId), ..., tensorRef(tensorId))
      // where tensor.cellId == memoryWrite's cellId
      if node.inputs.count >= 2,
        let firstNode = g.nodes[node.inputs[0]],
        let lastNode = g.nodes[node.inputs[node.inputs.count - 1]],
        case .memoryWrite(let cellId) = firstNode.op,
        case .tensorRef(let tensorId) = lastNode.op,
        let tensor = g.tensors[tensorId],
        tensor.cellId == cellId,
        case .tensor(let shape) = lastNode.shape
      {
        // bufferView backward: convert tensor gradient → scalar gradient
        let windowSize = shape.reduce(1, *)

        // Allocate frame-indexed gradient cell
        let gradCell = g.allocFrameAware(tensorSize: windowSize, frameCount: g.maxFrameCount)

        // Phase 1: Store gradient tensor elements to frame-indexed cell
        let storeOp = g.n(
          .bufferViewGradStore(gradCell: gradCell, windowSize: windowSize),
          [gradOutput])
        g.addGradientSideEffect(storeOp)

        // Phase 2: Read scalar gradient per frame (cross-frame sum)
        let readOp = g.n(
          .bufferViewGradRead(gradCell: gradCell, windowSize: windowSize),
          [storeOp])

        // Scalar gradient goes to memoryWrite (input 0), zero for others
        let zero = g.n(.constant(0.0), [])
        return node.inputs.enumerated().map { (i, _) in
          i == 0 ? readOp : zero
        }
      }

      // Default: gradient flows only to the last input
      let zero = g.n(.constant(0.0), [])
      return node.inputs.enumerated().map { (i, _) in
        i == node.inputs.count - 1 ? gradOutput : zero
      }

    // MARK: Memory Operations

    case .memoryRead(_):
      // Gradient for offset is zero
      return [g.n(.constant(0.0), [])]

    case .memoryWrite(_):
      // inputs: [offset, value]
      let zero = g.n(.constant(0.0), [])
      return [zero, gradOutput]

    case .memoryAccumulate(_):
      // inputs: [offset, value] - same as memoryWrite
      let zero = g.n(.constant(0.0), [])
      return [zero, gradOutput]

    case .memoryCellSum(_, _):
      // memoryCellSum is a reduction op - no input nodes to propagate gradients to
      // The gradient flows to the memory cell contents which are handled separately
      return []

    case .tensorAccumulate(_):
      // tensorAccumulate is a side-effect op for gradient accumulation
      // No gradients to propagate through it
      return [nil]

    // MARK: Noise (non-differentiable)

    case .noise(_):
      return [g.n(.constant(0.0), [])]

    case .click(_):
      return [g.n(.constant(0.0), [])]

    // MARK: Complex operations - need special handling

    case .conv1d(_), .conv2d(_):
      // Convolution gradients are complex - would need dedicated nodes
      // For now, return nil (not yet supported in graph form)
      return node.inputs.map { _ in nil }

    case .spectralLossFFT(
      let windowSize, _, let windowCell,
      let fft1Cell, let fft2Cell, let mag1Cell, let mag2Cell, _):
      // FFT-based spectral loss backward pass using O(N log N) IFFT
      // Reads the forward pass's per-frame FFT/magnitude cells directly,
      // avoiding the O(N²) DFT recomputation of GradInline.
      let sig1 = node.inputs[0]
      let sig2 = node.inputs[1]
      // inputs[2..5] are precomputed tensor refs: hann, twRe, twIm, bitRev
      let twReNode = node.inputs[3]
      let twImNode = node.inputs[4]
      let bitRevNode = node.inputs[5]
      let hopCounter = node.inputs.count > 6 ? node.inputs[6] : nil
      let fftSize = windowSize * 2

      // Allocate per-frame gradient spectrum cells (complex: real + imag)
      let gradSpec1Cell = g.alloc(vectorWidth: fftSize * g.maxFrameCount)
      let gradSpec2Cell = g.alloc(vectorWidth: fftSize * g.maxFrameCount)

      // Allocate per-frame time-domain gradient cells
      let gradTime1Cell = g.alloc(vectorWidth: g.maxFrameCount * windowSize)
      let gradTime2Cell = g.alloc(vectorWidth: g.maxFrameCount * windowSize)

      // Step 1: Compute gradient w.r.t. complex spectrum from forward's stored data
      // sig1/sig2 are ordering-only inputs — they ensure GradSpec runs after the
      // forward FFT (which also depends on sig1/sig2) in the topological sort
      var gradSpecInputs: [NodeID] = [gradOutput, sig1, sig2]
      if let hopCounter { gradSpecInputs.append(hopCounter) }
      let gradSpec = g.n(
        .spectralLossFFTGradSpec(
          windowSize: windowSize,
          fft1Cell: fft1Cell,
          fft2Cell: fft2Cell,
          mag1Cell: mag1Cell,
          mag2Cell: mag2Cell,
          gradSpec1Cell: gradSpec1Cell,
          gradSpec2Cell: gradSpec2Cell
        ), gradSpecInputs)

      g.addGradientSideEffect(gradSpec)

      // Step 2: IFFT gradient spectrum → time-domain gradients
      // Pass precomputed twiddle/bitrev tensors for table-lookup IFFT
      var gradIFFTInputs: [NodeID] = [gradSpec, twReNode, twImNode, bitRevNode]
      if let hopCounter { gradIFFTInputs.append(hopCounter) }
      let gradIFFT = g.n(
        .spectralLossFFTGradIFFT(
          windowSize: windowSize,
          gradSpec1Cell: gradSpec1Cell,
          gradSpec2Cell: gradSpec2Cell,
          gradTime1Cell: gradTime1Cell,
          gradTime2Cell: gradTime2Cell,
          windowCell: windowCell
        ), gradIFFTInputs)

      g.addGradientSideEffect(gradIFFT)

      // Step 3: Read the gradient for the current frame's sample
      var gradReadInputs: [NodeID] = [gradIFFT]
      if let hopCounter { gradReadInputs.append(hopCounter) }
      let gradPassResult = g.n(
        .spectralLossFFTGradRead(
          windowSize: windowSize,
          gradTime1Cell: gradTime1Cell,
          gradTime2Cell: gradTime2Cell
        ), gradReadInputs)

      let grad2Node = g.n(
        .spectralLossFFTGradRead2(
          windowSize: windowSize,
          gradTime2Cell: gradTime2Cell
        ), gradReadInputs)

      // Gradients: sig1, sig2 get time-domain gradients; tensor refs (hann, twRe, twIm, bitRev) and
      // hop counter are non-differentiable
      var result: [NodeID?] = [gradPassResult, grad2Node]
      for _ in 2..<node.inputs.count { result.append(nil) }
      return result

    case .spectralLossFFTGradSpec(_, _, _, _, _, _, _):
      // Gradient ops don't need their own gradients
      return node.inputs.map { _ in nil }

    case .spectralLossFFTGradIFFT(_, _, _, _, _, _):
      // Gradient ops don't need their own gradients
      return node.inputs.map { _ in nil }

    case .spectralLossFFTGradInline(_, _, _, _, _):
      // Gradient ops don't need their own gradients
      return node.inputs.map { _ in nil }

    case .spectralLossFFTGradRead(_, _, _):
      // Gradient read ops don't need their own gradients
      return node.inputs.map { _ in nil }

    case .spectralLossFFTGradRead2(_, _):
      // Gradient read ops don't need their own gradients
      return node.inputs.map { _ in nil }

    case .selectRow:
      // selectRow(tensor2D, rowIndex) -> 1D tensor [numCols]
      // Gradient uses frame-indexed storage for determinism (no atomics in write phase)
      guard node.inputs.count == 2 else {
        return [nil, nil]
      }

      let tensorInput = node.inputs[0]
      guard let inputNode = g.nodes[tensorInput],
        case .tensor(let shape) = inputNode.shape,
        shape.count == 2
      else {
        let zero = g.n(.constant(0.0), [])
        return [nil, zero]
      }

      let numRows = shape[0]
      let numCols = shape[1]
      let totalSize = numRows * numCols

      let gradCell = getOrCreateGradCell(g, tensorInput: tensorInput, totalSize: totalSize)

      // Allocate frame-indexed storage for deterministic gradient accumulation
      let gradWriteCell = g.alloc(vectorWidth: g.maxFrameCount * numCols)
      let rowIdxCell = g.alloc(vectorWidth: g.maxFrameCount)

      // Phase 1: Write gradients to frame-indexed storage (no atomics)
      let rowIndex = node.inputs[1]
      let writeOp = g.n(
        .selectRowGradWrite(
          gradWriteCell: gradWriteCell,
          rowIdxCell: rowIdxCell,
          numRows: numRows,
          numCols: numCols
        ), [gradOutput, rowIndex])
      g.addGradientSideEffect(writeOp)

      // Phase 2: Reduce across frames (sums to gradCell)
      let reduceOp = g.n(
        .selectRowGradReduce(
          gradWriteCell: gradWriteCell,
          rowIdxCell: rowIdxCell,
          gradCell: gradCell,
          numRows: numRows,
          numCols: numCols,
          maxFrameCount: g.maxFrameCount
        ), [writeOp])
      g.addGradientSideEffect(reduceOp)

      let sequencedGrad = createSequencedGradTensor(
        g, gradCell: gradCell, shape: shape, afterOp: reduceOp)
      let zero = g.n(.constant(0.0), [])
      return [sequencedGrad, zero]

    case .selectRowGradWrite(_, _, _, _), .selectRowGradReduce(_, _, _, _, _, _):
      // Gradient ops don't need their own gradients
      return node.inputs.map { _ in nil }

    case .peekRowInline(_, let numRows, let numCols):
      // peekRowInline(tensor2D, rowIndex) -> 1D tensor [numCols]
      // Gradient scatters to both floor and ceil rows with interpolation weights
      guard node.inputs.count == 2 else {
        return [nil, nil]
      }

      let tensorInput = node.inputs[0]
      guard let inputNode = g.nodes[tensorInput],
        case .tensor(let shape) = inputNode.shape,
        shape.count == 2
      else {
        let zero = g.n(.constant(0.0), [])
        return [nil, zero]
      }

      let totalSize = numRows * numCols

      let gradCell = getOrCreateGradCell(g, tensorInput: tensorInput, totalSize: totalSize)

      // Allocate frame-indexed storage
      let floorGradCell = g.alloc(vectorWidth: g.maxFrameCount * numCols)
      let ceilGradCell = g.alloc(vectorWidth: g.maxFrameCount * numCols)
      let rowIdxCell = g.alloc(vectorWidth: g.maxFrameCount * 2)  // floor and ceil indices
      let fracCell = g.alloc(vectorWidth: g.maxFrameCount)

      // Phase 1: Write weighted gradients to frame-indexed storage
      let rowIndex = node.inputs[1]
      let writeOp = g.n(
        .peekRowGradWrite(
          floorGradCell: floorGradCell,
          ceilGradCell: ceilGradCell,
          rowIdxCell: rowIdxCell,
          fracCell: fracCell,
          numRows: numRows,
          numCols: numCols,
          maxFrameCount: g.maxFrameCount
        ), [gradOutput, rowIndex])
      g.addGradientSideEffect(writeOp)

      // Phase 2: Reduce across frames
      let reduceOp = g.n(
        .peekRowGradReduce(
          floorGradCell: floorGradCell,
          ceilGradCell: ceilGradCell,
          rowIdxCell: rowIdxCell,
          fracCell: fracCell,
          gradCell: gradCell,
          numRows: numRows,
          numCols: numCols,
          maxFrameCount: g.maxFrameCount
        ), [writeOp])
      g.addGradientSideEffect(reduceOp)

      let sequencedGrad = createSequencedGradTensor(
        g, gradCell: gradCell, shape: shape, afterOp: reduceOp)
      let zero = g.n(.constant(0.0), [])
      return [sequencedGrad, zero]

    case .sampleInline(_, let numRows, let remainingShape):
      // sampleInline(tensorND, index) -> tensor with remainingShape
      // Gradient scatters to both floor and ceil rows with interpolation weights
      guard node.inputs.count == 2 else {
        return [nil, nil]
      }

      let tensorInput = node.inputs[0]
      guard let inputNode = g.nodes[tensorInput],
        case .tensor(let shape) = inputNode.shape,
        shape.count >= 2
      else {
        let zero = g.n(.constant(0.0), [])
        return [nil, zero]
      }

      let remainingSize = remainingShape.reduce(1, *)
      let totalSize = numRows * remainingSize

      let gradCell = getOrCreateGradCell(g, tensorInput: tensorInput, totalSize: totalSize)

      // Allocate frame-indexed storage
      let floorGradCell = g.alloc(vectorWidth: g.maxFrameCount * remainingSize)
      let ceilGradCell = g.alloc(vectorWidth: g.maxFrameCount * remainingSize)
      let rowIdxCell = g.alloc(vectorWidth: g.maxFrameCount * 2)
      let fracCell = g.alloc(vectorWidth: g.maxFrameCount)

      // Phase 1: Write weighted gradients to frame-indexed storage
      let rowIndex = node.inputs[1]
      let writeOp = g.n(
        .sampleGradWrite(
          floorGradCell: floorGradCell,
          ceilGradCell: ceilGradCell,
          rowIdxCell: rowIdxCell,
          fracCell: fracCell,
          numRows: numRows,
          remainingShape: remainingShape,
          maxFrameCount: g.maxFrameCount
        ), [gradOutput, rowIndex])
      g.addGradientSideEffect(writeOp)

      // Phase 2: Reduce across frames
      let reduceOp = g.n(
        .sampleGradReduce(
          floorGradCell: floorGradCell,
          ceilGradCell: ceilGradCell,
          rowIdxCell: rowIdxCell,
          fracCell: fracCell,
          gradCell: gradCell,
          numRows: numRows,
          remainingShape: remainingShape,
          maxFrameCount: g.maxFrameCount
        ), [writeOp])
      g.addGradientSideEffect(reduceOp)

      let sequencedGrad = createSequencedGradTensor(
        g, gradCell: gradCell, shape: shape, afterOp: reduceOp)
      let zero = g.n(.constant(0.0), [])
      return [sequencedGrad, zero]

    case .sampleGradWrite(_, _, _, _, _, _, _), .sampleGradReduce(_, _, _, _, _, _, _, _),
      .peekRowGradWrite(_, _, _, _, _, _, _), .peekRowGradReduce(_, _, _, _, _, _, _, _),
      .peekGradWrite(_, _, _, _, _, _, _), .peekGradReduce(_, _, _, _, _, _, _),
      .overlapAddGradStore(_), .overlapAddGradGather(_, _, _, _),
      .bufferViewGradStore(_, _), .bufferViewGradRead(_, _):
      // Gradient ops don't need their own gradients
      return node.inputs.map { _ in nil }

    case .overlapAdd(let windowSize, let hopSize, _, _, _):
      // overlapAdd backward: two-phase gradient (store per-frame grad, then gather)
      guard let tensorInput = node.inputs.first,
        let inputNode = g.nodes[tensorInput],
        case .tensor(let shape) = inputNode.shape
      else { return [nil] }

      let totalSize = shape.reduce(1, *)

      // Phase 1: Store output gradient per frame
      let gradStoreCell = g.alloc(vectorWidth: g.maxFrameCount)
      let storeOp = g.n(
        .overlapAddGradStore(gradStoreCell: gradStoreCell),
        [gradOutput])
      g.addGradientSideEffect(storeOp)

      // Phase 2: Gather into gradient tensor (frame-aware)
      let gradInputCell = g.allocFrameAware(tensorSize: totalSize, frameCount: g.maxFrameCount)

      let gatherOp = g.n(
        .overlapAddGradGather(
          windowSize: windowSize, hopSize: hopSize,
          gradStoreCell: gradStoreCell, gradInputCell: gradInputCell),
        [storeOp])
      g.addGradientSideEffect(gatherOp)

      // Return gradient tensor sequenced after gather
      let sequencedGrad = createSequencedGradTensor(
        g, gradCell: gradInputCell, shape: shape, afterOp: gatherOp)
      return [sequencedGrad]

    case .peek:
      // peek(tensor, index, channel) -> interpolated scalar read
      guard node.inputs.count == 3 else {
        return [nil, nil, nil]
      }

      let tensorInput = node.inputs[0]
      guard let inputNode = g.nodes[tensorInput],
        case .tensor(let originalShape) = inputNode.shape
      else {
        let zero = g.n(.constant(0.0), [])
        return [nil, zero, zero]
      }
      let shape = originalShape.count == 1 ? [originalShape[0], 1] : originalShape

      let channelSize = shape[0]
      let numChannels = shape[1]
      let totalSize = channelSize * numChannels

      // Get or create gradient cell for this tensor
      let gradCell = getOrCreateGradCell(g, tensorInput: tensorInput, totalSize: totalSize)
      let zero = g.n(.constant(0.0), [])

      if false {  //DGenGradientConfig.useDeterministicPeekGradients {
        // Deterministic two-phase write+reduce:
        // 1) write per-frame grad + interpolation metadata
        // 2) reduce over frames to build tensor gradient
        let gradWriteCell = g.alloc(vectorWidth: g.maxFrameCount)
        let floorPosCell = g.alloc(vectorWidth: g.maxFrameCount)
        let nextPosCell = g.alloc(vectorWidth: g.maxFrameCount)
        let fracCell = g.alloc(vectorWidth: g.maxFrameCount)

        let writeOp = g.n(
          .peekGradWrite(
            gradWriteCell: gradWriteCell,
            floorPosCell: floorPosCell,
            nextPosCell: nextPosCell,
            fracCell: fracCell,
            channelSize: channelSize,
            numChannels: numChannels,
            maxFrameCount: g.maxFrameCount
          ),
          [gradOutput, node.inputs[1], node.inputs[2]]
        )
        g.addGradientSideEffect(writeOp)

        let reduceOp = g.n(
          .peekGradReduce(
            gradWriteCell: gradWriteCell,
            floorPosCell: floorPosCell,
            nextPosCell: nextPosCell,
            fracCell: fracCell,
            gradCell: gradCell,
            totalSize: totalSize,
            maxFrameCount: g.maxFrameCount
          ),
          [writeOp]
        )
        g.addGradientSideEffect(reduceOp)

        let sequencedGrad = createSequencedGradTensor(
          g,
          gradCell: gradCell,
          shape: originalShape,
          afterOp: reduceOp
        )
        return [sequencedGrad, zero, zero]
      }

      // Legacy fast scatter path:
      // dL/d(tensor[pos1]) += gradOut * (1-frac)
      // dL/d(tensor[pos2]) += gradOut * frac
      let index = node.inputs[1]
      let channel = node.inputs[2]
      let one = g.n(.constant(1.0), [])
      let channelSizeFloat = g.n(.constant(Float(channelSize)), [])

      let wrappedIndex = g.n(.mod, [index, channelSizeFloat])
      let isNegative = g.n(.lt, [wrappedIndex, zero])
      let positiveIndex = g.n(
        .gswitch, [isNegative, g.n(.add, [wrappedIndex, channelSizeFloat]), wrappedIndex])

      let numChannelsMinusOne = g.n(.constant(Float(numChannels - 1)), [])
      let clampedChannel = g.n(
        .floor, [g.n(.max, [zero, g.n(.min, [channel, numChannelsMinusOne])])])
      let channelOffset = g.n(.mul, [channelSizeFloat, clampedChannel])

      let finalReadPos = g.n(.add, [channelOffset, positiveIndex])
      let flooredPos = g.n(.floor, [finalReadPos])
      let frac = g.n(.sub, [finalReadPos, flooredPos])

      let nextPos = g.n(.add, [flooredPos, one])
      let nextChannelOffset = g.n(.add, [channelOffset, channelSizeFloat])
      let nextPosWrapped = g.n(
        .gswitch, [g.n(.gte, [nextPos, nextChannelOffset]), channelOffset, nextPos])

      let oneMinusFrac = g.n(.sub, [one, frac])
      let grad1 = g.n(.mul, [gradOutput, oneMinusFrac])
      let grad2 = g.n(.mul, [gradOutput, frac])

      let scatter1 = g.n(.memoryAccumulate(gradCell), [flooredPos, grad1])
      let scatter2 = g.n(.memoryAccumulate(gradCell), [nextPosWrapped, grad2])
      g.addGradientSideEffect(scatter1)
      g.addGradientSideEffect(scatter2)

      if DGenGradientConfig.dropPeekTensorInputGradient {
        // Legacy behavior (fast but wrong for upstream learning): accumulate into grad cell
        // for direct tensor params, but do not return a tensor-input gradient node.
        return [nil, zero, zero]
      }

      // Keep the fast scatter accumulation path, but also return a proper tensor gradient
      // for upstream backpropagation. Without this, peek() acts as a gradient sink and
      // decoder weights upstream of tensor-producing ops (e.g., matmul -> sigmoid -> peek)
      // receive no gradients.
      let scatterDone = g.n(.seq, [scatter1, scatter2])
      let sequencedGrad = createSequencedGradTensor(
        g,
        gradCell: gradCell,
        shape: originalShape,
        afterOp: scatterDone
      )
      g.accumulatedGradProxyCellByNode[sequencedGrad] = gradCell

      return [sequencedGrad, zero, zero]

    // MARK: Logical ops (non-differentiable)

    case .and, .or, .xor:
      let zero = g.n(.constant(0.0), [])
      return [zero, zero]

    // MARK: Non-differentiable compute ops

    case .gradPhasor(_), .gradDeterministicPhasor, .gemm(_, _, _, _, _), .sumMulAxis0,
      .gemmSmall(_, _, _, _, _),
      .gemmChunkPartials(_, _, _, _, _, _, _), .chunkPartialsReduceToCell(_, _, _, _, _):
      return node.inputs.map { _ in nil }
    }
  }

  /// Helper to invert a permutation
  private func invertPermutation(_ perm: [Int]) -> [Int] {
    var inverse = Array(repeating: 0, count: perm.count)
    for (i, p) in perm.enumerated() {
      inverse[p] = i
    }
    return inverse
  }

  /// Get or create a gradient cell for a tensor input.
  /// For direct tensorRefs, uses gradCarryCells to share gradient storage.
  /// For intermediate results, allocates a fresh cell.
  private func getOrCreateGradCell(
    _ g: Graph,
    tensorInput: NodeID,
    totalSize: Int
  ) -> CellID {
    if let tensorId = g.nodeToTensor[tensorInput],
      let tensor = g.tensors[tensorId]
    {
      if let existing = g.gradCarryCells[tensor.cellId] {
        return existing
      }
      let gradCell = g.alloc(vectorWidth: totalSize)
      g.gradCarryCells[tensor.cellId] = gradCell
      return gradCell
    }
    return g.alloc(vectorWidth: totalSize)
  }

  /// Create a tensor backed by gradCell and return a tensorRef sequenced after the given op.
  /// This ensures the gradient is computed before reading.
  private func createSequencedGradTensor(
    _ g: Graph,
    gradCell: CellID,
    shape: [Int],
    afterOp: NodeID
  ) -> NodeID {
    let gradTensorId = g.nextTensorId
    g.nextTensorId += 1
    g.tensors[gradTensorId] = Tensor(id: gradTensorId, shape: shape, cellId: gradCell)
    g.cellToTensor[gradCell] = gradTensorId
    let gradTensorRef = g.n(.tensorRef(gradTensorId), [], shape: .tensor(shape))
    g.nodeToTensor[gradTensorRef] = gradTensorId
    let sequencedGrad = g.n(.seq, afterOp, gradTensorRef)
    g.nodeToTensor[sequencedGrad] = gradTensorId
    return sequencedGrad
  }

  /// Reduce gradient along broadcast dimensions to match target shape.
  /// When A[M,1,K] * B[1,N,K] -> C[M,N,K], the gradient for A needs to be summed
  /// along axis 1 (where A had size 1 but C has size N).
  private func reduceBroadcastGradient(_ g: Graph, grad: NodeID, toShape targetShape: ValueShape?)
    -> NodeID
  {
    guard let gradNode = g.nodes[grad],
      case .tensor(let gradShape) = gradNode.shape,
      case .tensor(let targetShapeArray) = targetShape
    else {
      // Scalar or unknown shape - no reduction needed
      return grad
    }

    // If shapes already match, no reduction needed
    if gradShape == targetShapeArray {
      return grad
    }

    // Handle case where target has fewer dimensions (sum over leading dims)
    let gradRank = gradShape.count
    let targetRank = targetShapeArray.count

    var result = grad

    // If gradient has more dimensions, we need to sum over leading dimensions first
    if gradRank > targetRank {
      // Sum over the extra leading dimensions
      for _ in 0..<(gradRank - targetRank) {
        result = g.n(.sumAxis(0), [result])
      }
      // Update shape tracking for the result
      if let newNode = g.nodes[result], case .tensor(let newShape) = newNode.shape {
        // Now newShape should have same rank as targetShapeArray
        // Continue to check for broadcast dimensions
        return reduceSameSizeGradient(
          g, grad: result, gradShape: newShape, targetShape: targetShapeArray)
      }
    }

    return reduceSameSizeGradient(
      g, grad: result, gradShape: gradShape, targetShape: targetShapeArray)
  }

  /// Helper to reduce gradient when shapes have same rank but different sizes (broadcast case)
  private func reduceSameSizeGradient(
    _ g: Graph, grad: NodeID, gradShape: [Int], targetShape: [Int]
  ) -> NodeID {
    // Right-align shapes for comparison (NumPy-style broadcasting)
    let gradRank = gradShape.count
    let targetRank = targetShape.count

    if gradRank != targetRank {
      // Shapes should have same rank at this point
      return grad
    }

    var result = grad

    // Find axes where target has size 1 but grad has size > 1
    // We need to sum along these axes, going from highest to lowest
    // to preserve axis indices
    var axesToReduce: [Int] = []
    for i in 0..<gradRank {
      if targetShape[i] == 1 && gradShape[i] > 1 {
        axesToReduce.append(i)
      }
    }

    // Sum along axes in reverse order (highest first) to preserve indices
    for axis in axesToReduce.reversed() {
      result = g.n(.sumAxis(axis), [result])
    }

    // After summing, the reduced dimensions become size 1
    // We may need to reshape to ensure the shape matches exactly
    if !axesToReduce.isEmpty {
      // The result should now have shape matching targetShape
      // sumAxis removes the axis, so we might need to reshape
      // Actually, sumAxis removes the dimension, so we need to add it back with size 1
      // Let's reshape to target shape
      result = g.n(.reshape(targetShape), [result])
    }

    return result
  }
}

// MARK: - New LazyOps for Gradient Computation

// These need to be added to the LazyOp enum in Operators.swift:
//
// case neg                           // Unary negation: -x
// case expand(Shape)                 // Broadcast scalar to tensor shape
// case expandAxis(Shape, Int)        // Broadcast along a specific axis
// case gradPhasor(NodeID)            // Special gradient for phasor (needs frame index)
// case gradDeterministicPhasor       // Special gradient for deterministic phasor
