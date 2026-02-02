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
      let inputGrads = node.op.backward(graph: self, node: node, gradOutput: upstreamGrad)

      // Accumulate gradients (as graph nodes, not values!)
      for (inputId, grad) in zip(node.inputs, inputGrads) {
        guard let grad = grad else { continue }

        if let existing = grads[inputId] {
          // Multiple paths contribute to this gradient -> add them
          grads[inputId] = n(.add, [existing, grad])
        } else {
          grads[inputId] = grad
        }
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

    // MARK: Tensor Shape Operations

    case .reshape(_):
      // Gradient needs to be reshaped back to original shape
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

    // MARK: Stateful Operations

    case .phasor(_):
      // d(phase)/d(freq) = frameIndex / sampleRate
      // Note: This requires frame index, which is tricky in graph form
      // For now, we use a special gradPhasor op that handles this
      let freq = node.inputs[0]
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

    case .historyRead(let cellId):
      // historyRead exposes previous state. The gradient w.r.t. this read's output
      // must be passed to the previous timestep via the gradient carry cell.
      // This is like storeGradMemory in the legacy backward.
      let carryCell = g.getGradCarryCell(for: cellId)
      let zero = g.n(.constant(0.0), [])
      // Store gradOutput to carry cell for previous timestep
      let writeNode = g.n(.memoryWrite(carryCell), [zero, gradOutput])
      // Register as side effect so it gets scheduled
      g.addGradientSideEffect(writeNode)
      return []  // No graph inputs

    case .historyWrite(let cellId):
      // historyWrite stores current input into the cell.
      // The gradient for the input comes from future reads via the carry cell.
      // This is like loadGradMemory in the legacy backward.
      let carryCell = g.getGradCarryCell(for: cellId)
      let zero = g.n(.constant(0.0), [])
      // Read gradient from carry cell (from future timestep)
      let carryGrad = g.n(.memoryRead(carryCell), [zero])
      return [carryGrad]

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
      // Gradient flows only to the last input
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
      let windowSize, let useHann, let windowCell,
      _, _, _, _, _):
      // FFT-based spectral loss backward pass
      // Recompute DFT inline to avoid race conditions from shared FFT cells
      let sig1 = node.inputs[0]
      let sig2 = node.inputs[1]

      // Allocate per-frame time-domain gradient cells to avoid race conditions
      // Layout: frame0: [0..windowSize), frame1: [windowSize..2*windowSize), etc.
      let gradTime1Cell = g.alloc(vectorWidth: g.maxFrameCount * windowSize)
      let gradTime2Cell = g.alloc(vectorWidth: g.maxFrameCount * windowSize)

      // Single gradient pass that recomputes DFT inline and scatters to time domain
      // Uses frame-indexed storage to avoid race conditions
      let gradPass = g.n(
        .spectralLossFFTGradInline(
          windowSize: windowSize,
          useHann: useHann,
          windowCell: windowCell,
          gradTime1Cell: gradTime1Cell,
          gradTime2Cell: gradTime2Cell
        ), [gradOutput, sig1, sig2])

      g.addGradientSideEffect(gradPass)

      // Read the gradient for the current frame's sample from frame-indexed storage
      // The gradient at position (frame * windowSize + windowSize - 1) is the gradient
      // for the current frame's input (the newest sample in the window)
      let gradPassResult = g.n(
        .spectralLossFFTGradRead(
          windowSize: windowSize,
          gradTime1Cell: gradTime1Cell,
          gradTime2Cell: gradTime2Cell
        ), [gradPass])

      // Split the result into two separate gradient values
      // The gradPassResult returns grad1, we need another node for grad2
      let grad2Node = g.n(
        .spectralLossFFTGradRead2(
          windowSize: windowSize,
          gradTime2Cell: gradTime2Cell
        ), [gradPass])

      return [gradPassResult, grad2Node]

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

    case .peekRowGradWrite(_, _, _, _, _, _, _), .peekRowGradReduce(_, _, _, _, _, _, _, _):
      // Gradient ops don't need their own gradients
      return node.inputs.map { _ in nil }

    case .fft(_, _, _, _, _, _), .ifft(_, _, _, _, _, _):
      // FFT gradients need special handling
      return node.inputs.map { _ in nil }

    case .peek:
      // peek(tensor, index, channel) -> interpolated scalar read
      // Gradient scatters back to two tensor positions
      guard node.inputs.count == 3 else {
        return [nil, nil, nil]
      }

      let tensorInput = node.inputs[0]
      guard let inputNode = g.nodes[tensorInput],
        case .tensor(let shape) = inputNode.shape,
        shape.count >= 2,
        let tensorId = g.nodeToTensor[tensorInput],
        let tensor = g.tensors[tensorId]
      else {
        let zero = g.n(.constant(0.0), [])
        return [nil, zero, zero]
      }

      let channelSize = shape[0]
      let numChannels = shape[1]
      let totalSize = channelSize * numChannels
      let tensorCellId = tensor.cellId

      // Get or create gradient cell for this tensor
      let gradCell: CellID
      if let existing = g.gradCarryCells[tensorCellId] {
        gradCell = existing
      } else {
        gradCell = g.alloc(vectorWidth: totalSize)
        g.gradCarryCells[tensorCellId] = gradCell
      }

      // Recompute interpolation positions (same logic as forward)
      let index = node.inputs[1]
      let channel = node.inputs[2]

      let one = g.n(.constant(1.0), [])
      let zero = g.n(.constant(0.0), [])
      let channelSizeFloat = g.n(.constant(Float(channelSize)), [])

      // Wrap index within channel using modulo
      let wrappedIndex = g.n(.mod, [index, channelSizeFloat])
      let isNegative = g.n(.lt, [wrappedIndex, zero])
      let positiveIndex = g.n(
        .gswitch, [isNegative, g.n(.add, [wrappedIndex, channelSizeFloat]), wrappedIndex])

      // Clamp channel to valid range [0, numChannels-1]
      let numChannelsMinusOne = g.n(.constant(Float(numChannels - 1)), [])
      let clampedChannel = g.n(
        .floor, [g.n(.max, [zero, g.n(.min, [channel, numChannelsMinusOne])])])
      let channelOffset = g.n(.mul, [channelSizeFloat, clampedChannel])

      // Calculate final read position
      let finalReadPos = g.n(.add, [channelOffset, positiveIndex])
      let flooredPos = g.n(.floor, [finalReadPos])
      let frac = g.n(.sub, [finalReadPos, flooredPos])

      // Next position with wrapping at channel boundary
      let nextPos = g.n(.add, [flooredPos, one])
      let nextChannelOffset = g.n(.add, [channelOffset, channelSizeFloat])
      let nextPosWrapped = g.n(
        .gswitch, [g.n(.gte, [nextPos, nextChannelOffset]), channelOffset, nextPos])

      // Scatter gradients: dL/d(tensor[pos1]) += gradOut * (1-frac)
      //                    dL/d(tensor[pos2]) += gradOut * frac
      let oneMinusFrac = g.n(.sub, [one, frac])
      let grad1 = g.n(.mul, [gradOutput, oneMinusFrac])
      let grad2 = g.n(.mul, [gradOutput, frac])

      let scatter1 = g.n(.memoryAccumulate(gradCell), [flooredPos, grad1])
      let scatter2 = g.n(.memoryAccumulate(gradCell), [nextPosWrapped, grad2])

      g.addGradientSideEffect(scatter1)
      g.addGradientSideEffect(scatter2)

      // Tensor gradient handled via side effects, index/channel gradients are zero
      return [nil, zero, zero]

    // MARK: Logical ops (non-differentiable)

    case .and, .or, .xor:
      let zero = g.n(.constant(0.0), [])
      return [zero, zero]

    // MARK: Gradient-specific ops (should not appear in forward graph)

    case .gradPhasor(_), .gradDeterministicPhasor:
      // These are gradient ops, shouldn't need their own gradients
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
