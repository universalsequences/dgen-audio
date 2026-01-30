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
        for nodeId in reverseTopologicalOrder(from: loss, targets: targets) {
            guard let upstreamGrad = grads[nodeId],
                  let node = nodes[nodeId] else { continue }

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
            // Gradient passes through unchanged to both inputs
            return [gradOutput, gradOutput]

        case .sub:
            // d(x-y)/dx = 1, d(x-y)/dy = -1
            let negGrad = g.n(.neg, [gradOutput])
            return [gradOutput, negGrad]

        case .mul:
            // d(x*y)/dx = y * grad, d(x*y)/dy = x * grad
            let lhs = node.inputs[0]
            let rhs = node.inputs[1]
            let gradLhs = g.n(.mul, [rhs, gradOutput])
            let gradRhs = g.n(.mul, [lhs, gradOutput])
            return [gradLhs, gradRhs]

        case .div:
            // d(x/y)/dx = grad / y
            // d(x/y)/dy = -grad * x / y^2 = -grad * (x/y) / y
            let lhs = node.inputs[0]
            let rhs = node.inputs[1]
            let gradLhs = g.n(.div, [gradOutput, rhs])
            // For rhs: -grad * lhs / (rhs * rhs)
            let rhsSquared = g.n(.mul, [rhs, rhs])
            let lhsOverRhsSq = g.n(.div, [lhs, rhsSquared])
            let negGrad = g.n(.neg, [gradOutput])
            let gradRhs = g.n(.mul, [negGrad, lhsOverRhsSq])
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
            return [g.n(.mul, [negSinX, gradOutput])]

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
            guard let inputNode = g.nodes[node.inputs[0]],
                  case .tensor(let shape) = inputNode.shape else {
                // Scalar: just pass through
                return [gradOutput]
            }
            // Expand scalar gradient to tensor shape
            return [g.n(.expand(shape), [gradOutput])]

        case .sumAxis(let axis):
            // Gradient broadcasts back along the reduced axis
            guard let inputNode = g.nodes[node.inputs[0]],
                  case .tensor(let inputShape) = inputNode.shape else {
                return [gradOutput]
            }
            // Expand gradient back to input shape along the reduced axis
            return [g.n(.expandAxis(inputShape, axis), [gradOutput])]

        // MARK: Tensor Shape Operations

        case .reshape(_):
            // Gradient needs to be reshaped back to original shape
            guard let inputNode = g.nodes[node.inputs[0]],
                  case .tensor(let origShape) = inputNode.shape else {
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
                  case .tensor(let origShape) = inputNode.shape else {
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
                  case .tensor(let origShape) = inputNode.shape else {
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
            return []  // Reference, no gradient

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

        case .spectralLossPass1(_, _), .spectralLossPass2(_, _):
            // Spectral loss has complex multi-pass gradient
            // Not yet supported in pure graph form
            return node.inputs.map { _ in nil }

        case .fft(_, _, _, _, _, _), .ifft(_, _, _, _, _, _):
            // FFT gradients need special handling
            return node.inputs.map { _ in nil }

        case .peek, .peekRow:
            // Interpolated reads - complex gradient
            return node.inputs.map { _ in nil }

        case .parallelMap2DTestPass1(_, _), .parallelMap2DTestPass2(_, _):
            // Test ops, no gradient
            return node.inputs.map { _ in g.n(.constant(0.0), []) }

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
}

// MARK: - New LazyOps for Gradient Computation

// These need to be added to the LazyOp enum in Operators.swift:
//
// case neg                           // Unary negation: -x
// case expand(Shape)                 // Broadcast scalar to tensor shape
// case expandAxis(Shape, Int)        // Broadcast along a specific axis
// case gradPhasor(NodeID)            // Special gradient for phasor (needs frame index)
// case gradDeterministicPhasor       // Special gradient for deterministic phasor
