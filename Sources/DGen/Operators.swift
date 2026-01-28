public typealias NodeID = Int
public typealias VarID = Int
public typealias ConstantID = Int
public typealias CellID = Int
public typealias GradID = Int
public typealias ChannelNumber = Int

// MARK: - Tensor Emit Helpers

/// Emit a binary op for scalars or tensors.
func emitBinaryOp(
    b: IRBuilder,
    g: Graph,
    node: Node,
    inputs: [Lazy],
    op: (Expr, Expr) -> Expr
) throws {
    let a = try b.readInput(node, inputs, at: 0)
    let c = try b.readInput(node, inputs, at: 1)
    let result = op(a, c)

    try b.writeOutput(node, result)
}

/// Emit a unary op for scalars or tensors.
func emitUnaryOp(
    b: IRBuilder,
    g: Graph,
    node: Node,
    inputs: [Lazy],
    op: (Expr) -> Expr
) throws {
    let a = try b.readInput(node, inputs, at: 0)
    let result = op(a)
    try b.writeOutput(node, result)
}

/// Emit a ternary op for scalars or tensors.
func emitTernaryOp(
    b: IRBuilder,
    g: Graph,
    node: Node,
    inputs: [Lazy],
    op: (Expr, Expr, Expr) -> Expr
) throws {
    let a = try b.readInput(node, inputs, at: 0)
    let c = try b.readInput(node, inputs, at: 1)
    let d = try b.readInput(node, inputs, at: 2)

    let result = op(a, c, d)

    try b.writeOutput(node, result)
}

public struct BackwardsEmitResult {
    let ops: [UOp]
    let dependencies: [NodeID]
}

/// Emit backward pass for binary operations, handling all tensor/scalar combinations.
/// - Parameters:
///   - gradLhs: Closure computing dL/dLhs given (gradOutput, lhsValue, rhsValue)
///   - gradRhs: Closure computing dL/dRhs given (gradOutput, lhsValue, rhsValue)
func emitBinaryOpBackward(
    b: IRBuilder,
    g: Graph,
    ctx: IRContext,
    node: Node,
    gradOutput: Lazy,
    gradLhs: @escaping (Expr, Expr, Expr) -> Expr,
    gradRhs: @escaping (Expr, Expr, Expr) -> Expr
) throws {
    let lhsNodeId = node.inputs[0]
    let rhsNodeId = node.inputs[1]
    let lhsNode = g.nodes[lhsNodeId]
    let rhsNode = g.nodes[rhsNodeId]

    let lhsIsTensor: Bool
    let rhsIsTensor: Bool
    if case .tensor = lhsNode?.shape { lhsIsTensor = true } else { lhsIsTensor = false }
    if case .tensor = rhsNode?.shape { rhsIsTensor = true } else { rhsIsTensor = false }

    if lhsIsTensor && rhsIsTensor {
        // Tensor op Tensor (element-wise)
        guard case .tensor(let lhsShape) = lhsNode?.shape,
              case .tensor(let rhsShape) = rhsNode?.shape,
              case .tensor(let outShape) = node.shape else {
            fatalError("Expected tensor shapes")
        }

        let lhsSize = lhsShape.reduce(1, *)
        let rhsSize = rhsShape.reduce(1, *)
        let outSize = outShape.reduce(1, *)

        guard let lhsTensorId = g.nodeToTensor[lhsNodeId],
              let lhsTensor = g.tensors[lhsTensorId],
              let rhsTensorId = g.nodeToTensor[rhsNodeId],
              let rhsTensor = g.tensors[rhsTensorId] else {
            fatalError("Could not find tensors for backward")
        }

        // Allocate tensor gradients
        let _ = ctx.useTensorGradient(src: lhsNodeId, size: lhsSize)
        let _ = ctx.useTensorGradient(src: rhsNodeId, size: rhsSize)
        let _ = ctx.useTensorGradient(src: node.id, size: outSize)

        if lhsShape == rhsShape {
            // Same shape - simple element-wise
            b.parallelRange(outSize) { idx in
                let gradOut = b.loadTensorGrad(node.id, index: idx)
                let lhsVal = b.readTensorElement(cellId: lhsTensor.cellId, at: idx)
                let rhsVal = b.readTensorElement(cellId: rhsTensor.cellId, at: idx)

                let gLhs = gradLhs(gradOut, lhsVal, rhsVal)
                let gRhs = gradRhs(gradOut, lhsVal, rhsVal)

                b.tensorGrad(lhsNodeId, index: idx, value: gLhs.lazy)
                b.tensorGrad(rhsNodeId, index: idx, value: gRhs.lazy)
            }
        } else {
            // Different shapes - broadcast handling
            let lhsStrides = Tensor.computeRowMajorStrides(lhsShape)
            let rhsStrides = Tensor.computeRowMajorStrides(rhsShape)
            let outStrides = Tensor.computeRowMajorStrides(outShape)

            b.parallelRange(outSize) { outIdx in
                let gradOut = b.loadTensorGrad(node.id, index: outIdx)

                // Compute broadcast indices
                var lhsIdx = b.int(0)
                var rhsIdx = b.int(0)

                for dim in 0..<outShape.count {
                    let coord = b.mod(b.div(outIdx, b.int(outStrides[dim])), b.int(outShape[dim]))

                    if dim < lhsShape.count && lhsShape[dim] > 1 {
                        lhsIdx = b.add(lhsIdx, b.mul(coord, b.int(lhsStrides[dim])))
                    }
                    if dim < rhsShape.count && rhsShape[dim] > 1 {
                        rhsIdx = b.add(rhsIdx, b.mul(coord, b.int(rhsStrides[dim])))
                    }
                }

                let lhsIdxFloat = b.cast(lhsIdx, to: .float)
                let rhsIdxFloat = b.cast(rhsIdx, to: .float)
                let lhsVal = b.readTensorElement(cellId: lhsTensor.cellId, at: lhsIdxFloat)
                let rhsVal = b.readTensorElement(cellId: rhsTensor.cellId, at: rhsIdxFloat)

                let gLhs = gradLhs(gradOut, lhsVal, rhsVal)
                let gRhs = gradRhs(gradOut, lhsVal, rhsVal)

                b.tensorGrad(lhsNodeId, index: lhsIdxFloat, value: gLhs.lazy)
                b.tensorGrad(rhsNodeId, index: rhsIdxFloat, value: gRhs.lazy)
            }
        }

    } else if lhsIsTensor && !rhsIsTensor {
        // Tensor op Scalar
        guard case .tensor(let lhsShape) = lhsNode?.shape else { fatalError("Expected tensor shape") }
        let lhsSize = lhsShape.reduce(1, *)

        guard let lhsTensorId = g.nodeToTensor[lhsNodeId],
              let lhsTensor = g.tensors[lhsTensorId] else {
            fatalError("Could not find tensor for backward")
        }

        let _ = ctx.useTensorGradient(src: lhsNodeId, size: lhsSize)
        let _ = ctx.useTensorGradient(src: node.id, size: lhsSize)

        let rhsVal = b.tapeValue(rhsNodeId)

        // dL/dTensor[i] using the gradLhs closure
        b.parallelRange(lhsSize) { idx in
            let gradOut = b.loadTensorGrad(node.id, index: idx)
            let lhsVal = b.readTensorElement(cellId: lhsTensor.cellId, at: idx)
            let gLhs = gradLhs(gradOut, lhsVal, rhsVal)
            b.tensorGrad(lhsNodeId, index: idx, value: gLhs.lazy)
        }

        // dL/dScalar = sum over tensor elements
        let scalarGradAcc = b.float(0.0)
        b.loop(lhsSize) { idx in
            let gradOut = b.loadTensorGrad(node.id, index: idx)
            let lhsVal = b.readTensorElement(cellId: lhsTensor.cellId, at: idx)
            let gRhs = gradRhs(gradOut, lhsVal, rhsVal)
            scalarGradAcc.accumulate(gRhs)
        }
        b.grad(rhsNodeId, value: scalarGradAcc.value.lazy)

    } else if !lhsIsTensor && rhsIsTensor {
        // Scalar op Tensor
        guard case .tensor(let rhsShape) = rhsNode?.shape else { fatalError("Expected tensor shape") }
        let rhsSize = rhsShape.reduce(1, *)

        guard let rhsTensorId = g.nodeToTensor[rhsNodeId],
              let rhsTensor = g.tensors[rhsTensorId] else {
            fatalError("Could not find tensor for backward")
        }

        let _ = ctx.useTensorGradient(src: rhsNodeId, size: rhsSize)
        let _ = ctx.useTensorGradient(src: node.id, size: rhsSize)

        let lhsVal = b.tapeValue(lhsNodeId)

        // dL/dTensor[i] using the gradRhs closure
        b.parallelRange(rhsSize) { idx in
            let gradOut = b.loadTensorGrad(node.id, index: idx)
            let rhsVal = b.readTensorElement(cellId: rhsTensor.cellId, at: idx)
            let gRhs = gradRhs(gradOut, lhsVal, rhsVal)
            b.tensorGrad(rhsNodeId, index: idx, value: gRhs.lazy)
        }

        // dL/dScalar = sum over tensor elements
        let scalarGradAcc = b.float(0.0)
        b.loop(rhsSize) { idx in
            let gradOut = b.loadTensorGrad(node.id, index: idx)
            let rhsVal = b.readTensorElement(cellId: rhsTensor.cellId, at: idx)
            let gLhs = gradLhs(gradOut, lhsVal, rhsVal)
            scalarGradAcc.accumulate(gLhs)
        }
        b.grad(lhsNodeId, value: scalarGradAcc.value.lazy)

    } else {
        // Scalar op Scalar (original)
        let lhsVal = b.tapeValue(lhsNodeId)
        let rhsVal = b.tapeValue(rhsNodeId)
        let gradOutExpr = b.value(gradOutput)

        let gLhs = gradLhs(gradOutExpr, lhsVal, rhsVal)
        let gRhs = gradRhs(gradOutExpr, lhsVal, rhsVal)

        b.grad(lhsNodeId, value: gLhs.lazy)
        b.grad(rhsNodeId, value: gRhs.lazy)
    }
}

// frontend
public enum LazyOp {
    case add, sub, div, mul, abs, sign, sin, cos, tan, tanh, exp, log, log10, sqrt, atan2, gt, gte,
        lte,
        lt, eq,
        gswitch, mix, pow, floor, ceil, round, mod, min, max, and, or, xor
    case mse  // mean squared error per-sample: (a-b)^2
    case spectralLossPass1(Int, CellID)  // Pass 1: compute loss & store DFT contributions
    case spectralLossPass2(Int, CellID)  // Pass 2: reduce contributions to gradients (no-op in forward)
    case selector  // selector(mode, options[])
    case memoryRead(CellID)
    case memoryWrite(CellID)
    case historyWrite(CellID)
    case historyReadWrite(CellID)
    case param(CellID)
    case latch(CellID)
    case click(CellID)
    case historyRead(CellID)
    case phasor(CellID)
    case accum(CellID)
    case noise(CellID)
    case constant(Float)
    case output(Int)
    case input(Int)
    case tensorRef(TensorID)
    case seq  // Sequential execution - returns value of last input

    // Tensor operations (historyRead/historyWrite handle tensors automatically based on cell size)
    case conv1d(Int)    // 1D convolution, Int is kernel size
    case conv2d(Shape)  // 2D convolution, Shape is kernel shape [kH, kW]
    case sum  // Reduce tensor to scalar by summing all elements
    case sumAxis(Int)  // Reduce along a specific axis
    case reshape(Shape)  // Reshape tensor (metadata only, no data movement)
    case transpose([Int])  // Transpose/permute axes (metadata only)
    case shrink([(Int, Int)?])  // Shrink/slice tensor (metadata only, no data movement)
    case pad([(Int, Int)])      // Pad tensor with zeros (virtual view, conditional reads)
    case peek  // Read from 2D tensor at (index, channel) with interpolation - lazy version

    public func emit(ctx: IRContext, g: Graph, nodeId: NodeID) throws -> [UOp] {
        guard let node = g.nodes[nodeId] else { return [] }

        // collect operands
        let inputs: [Lazy] = node.inputs.compactMap { ctx.values[$0] }
        var ops: [UOp] = []
        let b = IRBuilder(ctx: ctx, nodeId: nodeId)

        switch self {
        case .constant(let value):
            _ = ctx.useConstant(src: nodeId, value: value)
            return []
        case .tensorRef(_):
            // Register a placeholder value so that downstream ops can find this input
            // The actual tensor data is accessed via nodeToTensor lookup
            ctx.values[nodeId] = .empty
            return []
        case .add:
            guard inputs.count == 2 else {
                throw DGenError.insufficientInputs(
                    operator: "add", expected: 2, actual: inputs.count)
            }
            try emitBinaryOp(b: b, g: g, node: node, inputs: inputs) { $0 + $1 }
        case .sub:
            guard inputs.count == 2 else {
                throw DGenError.insufficientInputs(
                    operator: "sub", expected: 2, actual: inputs.count)
            }
            try emitBinaryOp(b: b, g: g, node: node, inputs: inputs) { $0 - $1 }
        case .mul:
            guard inputs.count == 2 else {
                print("mul failing \(node.id)")
                throw DGenError.insufficientInputs(
                    operator: "mul", expected: 2, actual: inputs.count)
            }
            try emitBinaryOp(b: b, g: g, node: node, inputs: inputs) { $0 * $1 }
        case .div:
            guard inputs.count == 2 else {
                throw DGenError.insufficientInputs(
                    operator: "div", expected: 2, actual: inputs.count)
            }
            try emitBinaryOp(b: b, g: g, node: node, inputs: inputs) { $0 / $1 }
        case .mod:
            guard inputs.count == 2 else {
                throw DGenError.insufficientInputs(
                    operator: "mod", expected: 2, actual: inputs.count)
            }
            try emitBinaryOp(b: b, g: g, node: node, inputs: inputs) { $0 % $1 }
        case .min:
            guard inputs.count == 2 else {
                throw DGenError.insufficientInputs(
                    operator: "min", expected: 2, actual: inputs.count)
            }
            try emitBinaryOp(b: b, g: g, node: node, inputs: inputs) { b.min($0, $1) }
        case .max:
            guard inputs.count == 2 else {
                throw DGenError.insufficientInputs(
                    operator: "max", expected: 2, actual: inputs.count)
            }
            try emitBinaryOp(b: b, g: g, node: node, inputs: inputs) { b.max($0, $1) }
        case .and:
            guard inputs.count == 2 else {
                throw DGenError.insufficientInputs(
                    operator: "and", expected: 2, actual: inputs.count)
            }
            try emitBinaryOp(b: b, g: g, node: node, inputs: inputs) { b.and($0, $1) }
        case .or:
            guard inputs.count == 2 else {
                throw DGenError.insufficientInputs(
                    operator: "or", expected: 2, actual: inputs.count)
            }
            try emitBinaryOp(b: b, g: g, node: node, inputs: inputs) { b.or($0, $1) }
        case .xor:
            guard inputs.count == 2 else {
                throw DGenError.insufficientInputs(
                    operator: "xor", expected: 2, actual: inputs.count)
            }
            try emitBinaryOp(b: b, g: g, node: node, inputs: inputs) { b.xor($0, $1) }
        case .abs:
            guard inputs.count == 1 else {
                throw DGenError.insufficientInputs(
                    operator: "abs", expected: 1, actual: inputs.count)
            }
            try emitUnaryOp(b: b, g: g, node: node, inputs: inputs) { b.abs($0) }
        case .sign:
            guard inputs.count == 1 else {
                throw DGenError.insufficientInputs(
                    operator: "sign", expected: 1, actual: inputs.count)
            }
            try emitUnaryOp(b: b, g: g, node: node, inputs: inputs) { b.sign($0) }
        case .sin:
            guard inputs.count == 1 else {
                throw DGenError.insufficientInputs(
                    operator: "sin", expected: 1, actual: inputs.count)
            }
            try emitUnaryOp(b: b, g: g, node: node, inputs: inputs) { b.sin($0) }
        case .cos:
            guard inputs.count == 1 else {
                throw DGenError.insufficientInputs(
                    operator: "cos", expected: 1, actual: inputs.count)
            }
            try emitUnaryOp(b: b, g: g, node: node, inputs: inputs) { b.cos($0) }
        case .tan:
            guard inputs.count == 1 else {
                throw DGenError.insufficientInputs(
                    operator: "tan", expected: 1, actual: inputs.count)
            }
            try emitUnaryOp(b: b, g: g, node: node, inputs: inputs) { b.tan($0) }
        case .tanh:
            guard inputs.count == 1 else {
                throw DGenError.insufficientInputs(
                    operator: "tanh", expected: 1, actual: inputs.count)
            }
            try emitUnaryOp(b: b, g: g, node: node, inputs: inputs) { b.tanh($0) }
        case .exp:
            guard inputs.count == 1 else {
                throw DGenError.insufficientInputs(
                    operator: "exp", expected: 1, actual: inputs.count)
            }
            try emitUnaryOp(b: b, g: g, node: node, inputs: inputs) { b.exp($0) }
        case .log:
            guard inputs.count == 1 else {
                throw DGenError.insufficientInputs(
                    operator: "log", expected: 1, actual: inputs.count)
            }
            try emitUnaryOp(b: b, g: g, node: node, inputs: inputs) { b.log($0) }
        case .log10:
            guard inputs.count == 1 else {
                throw DGenError.insufficientInputs(
                    operator: "log10", expected: 1, actual: inputs.count)
            }
            try emitUnaryOp(b: b, g: g, node: node, inputs: inputs) { b.log10($0) }
        case .sqrt:
            guard inputs.count == 1 else {
                throw DGenError.insufficientInputs(
                    operator: "sqrt", expected: 1, actual: inputs.count)
            }
            try emitUnaryOp(b: b, g: g, node: node, inputs: inputs) { b.sqrt($0) }
        case .pow:
            guard inputs.count == 2 else {
                throw DGenError.insufficientInputs(
                    operator: "pow", expected: 2, actual: inputs.count)
            }
            try emitBinaryOp(b: b, g: g, node: node, inputs: inputs) { b.pow($0, $1) }
        case .floor:
            guard inputs.count == 1 else {
                throw DGenError.insufficientInputs(
                    operator: "floor", expected: 1, actual: inputs.count)
            }
            try emitUnaryOp(b: b, g: g, node: node, inputs: inputs) { b.floor($0) }
        case .round:
            guard inputs.count == 1 else {
                throw DGenError.insufficientInputs(
                    operator: "round", expected: 1, actual: inputs.count)
            }
            try emitUnaryOp(b: b, g: g, node: node, inputs: inputs) { b.round($0) }
        case .ceil:
            guard inputs.count == 1 else {
                throw DGenError.insufficientInputs(
                    operator: "ceil", expected: 1, actual: inputs.count)
            }
            try emitUnaryOp(b: b, g: g, node: node, inputs: inputs) { b.ceil($0) }
        case .memoryRead(let cellId):
            guard inputs.count == 1 else {
                throw DGenError.insufficientInputs(
                    operator: "memoryRead", expected: 1, actual: inputs.count)
            }
            b.use(val: b.memoryRead(cellId, b.value(inputs[0])))
        case .memoryWrite(let cellId):
            guard inputs.count == 2 else {
                throw DGenError.insufficientInputs(
                    operator: "memoryWrite", expected: 2, actual: inputs.count)
            }
            b.use(val: b.memoryWrite(cellId, b.value(inputs[0]), b.value(inputs[1])))
        case .atan2:
            guard inputs.count == 2 else {
                throw DGenError.insufficientInputs(
                    operator: "atan2", expected: 2, actual: inputs.count)
            }
            b.use(val: b.atan2(b.value(inputs[0]), b.value(inputs[1])))
        case .mse:
            guard inputs.count == 2 else {
                throw DGenError.insufficientInputs(
                    operator: "mse", expected: 2, actual: inputs.count)
            }
            let (a, b2) = b.values(inputs, count: 2)
            b.use(val: u_mse(a, b2)(b))

        case let .spectralLossPass1(windowSize, scratchCell):
            guard inputs.count == 2 else {
                throw DGenError.insufficientInputs(
                    operator: "spectralLossPass1", expected: 2, actual: inputs.count)
            }
            let (sig1, sig2) = b.values(inputs, count: 2)
            // Forward: compute spectral loss normally (Pass1 does the actual work)
            b.use(val: u_spectralLoss(sig1: sig1, sig2: sig2, windowSize: windowSize)(b))

        case .spectralLossPass2(_, _):
            guard inputs.count == 1 else {
                throw DGenError.insufficientInputs(
                    operator: "spectralLossPass2", expected: 1, actual: inputs.count)
            }
            // Forward: no-op, just forward the value from Pass1
            let pass1Result = b.value(inputs[0])
            b.use(val: pass1Result)

        case .gt:
            guard inputs.count == 2 else {
                throw DGenError.insufficientInputs(
                    operator: "gt", expected: 2, actual: inputs.count)
            }
            try emitBinaryOp(b: b, g: g, node: node, inputs: inputs) { $0 > $1 }
        case .gte:
            guard inputs.count == 2 else {
                throw DGenError.insufficientInputs(
                    operator: "gte", expected: 2, actual: inputs.count)
            }
            try emitBinaryOp(b: b, g: g, node: node, inputs: inputs) { $0 >= $1 }
        case .lte:
            guard inputs.count == 2 else {
                throw DGenError.insufficientInputs(
                    operator: "lte", expected: 2, actual: inputs.count)
            }
            try emitBinaryOp(b: b, g: g, node: node, inputs: inputs) { $0 <= $1 }
        case .lt:
            guard inputs.count == 2 else {
                throw DGenError.insufficientInputs(
                    operator: "lt", expected: 2, actual: inputs.count)
            }
            try emitBinaryOp(b: b, g: g, node: node, inputs: inputs) { $0 < $1 }
        case .eq:
            guard inputs.count == 2 else {
                throw DGenError.insufficientInputs(
                    operator: "eq", expected: 2, actual: inputs.count)
            }
            try emitBinaryOp(b: b, g: g, node: node, inputs: inputs) { $0 == $1 }
        case .gswitch:
            guard inputs.count == 3 else {
                throw DGenError.insufficientInputs(
                    operator: "gswitch", expected: 3, actual: inputs.count)
            }
            try emitTernaryOp(b: b, g: g, node: node, inputs: inputs) { b.gswitch($0, $1, $2) }
        case .selector:
            guard inputs.count >= 2 else {
                throw DGenError.insufficientInputs(
                    operator: "selector", expected: 2, actual: inputs.count)
            }
            let mode = inputs[0]
            let options = Array(inputs.dropFirst())
            b.use(val: b.selector(b.value(mode), options.map { b.value($0) }))
        case .historyWrite(let cellId):
            // Unified history write - handles both scalar and tensor based on cellToTensor mapping
            if let tensorId = g.cellToTensor[cellId], let tensor = g.tensors[tensorId] {
                // Tensor write: copy from input tensor to history cell
                guard node.inputs.count >= 1 else {
                    throw DGenError.insufficientInputs(
                        operator: "historyWrite", expected: 1, actual: node.inputs.count)
                }
                let inputTensorId = g.nodeToTensor[node.inputs[0]]!
                let inputCellId = g.tensors[inputTensorId]!.cellId
                let size = tensor.size

                guard let index = ctx.tensorIndices[nodeId] else {
                    throw DGenError.insufficientInputs(
                        operator: "historyWrite", expected: 1, actual: node.inputs.count)
                }
                let idx = b.value(index)
                // tload for cached register, but ALWAYS write - history persists across frames
                let value = b.tload(inputCellId, idx)
                _ = b.memoryWrite(cellId, b.cast(idx, to: .int), value)
            } else {
                // Scalar write
                guard inputs.count == 1 else {
                    throw DGenError.insufficientInputs(
                        operator: "history write", expected: 1, actual: inputs.count)
                }
                b.use(val: b.store(cellId, b.value(inputs[0])))
            }
        case .param(let cellId):
            b.use(val: b.load(cellId))
        case .historyReadWrite(let cellId):
            // for simd its beyond just this -- we need to ensure that we shift the results 1
            guard inputs.count == 1 else {
                throw DGenError.insufficientInputs(
                    operator: "history write", expected: 1, actual: inputs.count)
            }
            b.use(val: u_historyWrite(cellId: cellId, b.value(inputs[0]))(b))
        case .historyRead(let cellId):
            // Unified history read - handles both scalar and tensor based on cellToTensor mapping
            if let tensorId = g.cellToTensor[cellId], let tensor = g.tensors[tensorId] {
                // Tensor read: copy from history cell to output tensor
                let outputTensorId = g.nodeToTensor[node.id]!
                let outputCellId = g.tensors[outputTensorId]!.cellId
                let size = tensor.size

                guard let index = ctx.tensorIndices[nodeId] else {
                    throw DGenError.insufficientInputs(
                        operator: "historyWrite", expected: 1, actual: node.inputs.count)
                }
                let idx = b.value(index)
                let value = b.tload(cellId, idx)
                _ = b.tstore(outputCellId, idx, value)
                // Register placeholder for downstream ops
                ctx.values[nodeId] = .empty
            } else {
                // Scalar read
                b.use(val: b.load(cellId))
            }
        case .latch(let cellId):
            guard inputs.count == 2 else {
                throw DGenError.insufficientInputs(
                    operator: "latch", expected: 2, actual: inputs.count)
            }

            // Check if we're in a tensor context
            if let tensorIndex = ctx.tensorIndices[nodeId] {
                // Tensor latch: read inputs from tensors, use indexed state
                // Mark as requiring scalar execution - latch state updates sample-by-sample
                b.markRequiresScalar()

                let value = try b.readInput(node, inputs, at: 0)
                let cond = try b.readInput(node, inputs, at: 1)
                let idx = b.value(tensorIndex)

                let zero = b.constant(0.0)

                // Load current latched value from indexed position
                let latched = b.memoryRead(cellId, b.cast(idx, to: .int))

                // If cond > 0, store new value; else keep latched
                // Use gswitch for SIMD compatibility
                let newLatched = b.gswitch(cond > zero, value, latched)

                // Store the result
                _ = b.memoryWrite(cellId, b.cast(idx, to: .int), newLatched)

                // Output the latched value (returns old value, like scalar latch)
                try b.writeOutput(node, latched)
            } else {
                // Scalar latch: use original implementation
                let value = b.value(inputs[0])
                let cond = b.value(inputs[1])
                b.use(val: u_latch(cellId, value: value, cond: cond)(b))
            }
        case .mix:
            guard inputs.count == 3 else {
                throw DGenError.insufficientInputs(
                    operator: "mix", expected: 3, actual: inputs.count)
            }
            try emitTernaryOp(b: b, g: g, node: node, inputs: inputs) {
                let val = u_mix($0, $1, lerp: $2)(b)
                b.use(val: val)
                return val
            }
        case .accum(let cellId):
            guard inputs.count == 4 else {
                throw DGenError.insufficientInputs(
                    operator: "accum", expected: 4, actual: inputs.count)
            }

            // Check if we're in a tensor context
            if let tensorIndex = ctx.tensorIndices[nodeId] {
                // Tensor accum: read inputs from tensors, use indexed state
                // Mark as requiring scalar execution - accum state accumulates sample-by-sample
                b.markRequiresScalar()

                let incr = try b.readInput(node, inputs, at: 0)
                let reset = try b.readInput(node, inputs, at: 1)
                let min = try b.readInput(node, inputs, at: 2)
                let max = try b.readInput(node, inputs, at: 3)
                let idx = b.value(tensorIndex)

                let zero = b.constant(0.0)
                let span = max - min

                // Load current state from indexed position
                let acc = b.memoryRead(cellId, b.cast(idx, to: .int))

                let nextCand = acc + incr
                let next = b.gswitch(reset > zero, min, nextCand)

                // Modulo wrap to [min, max)
                let rel = next - min
                let k = b.floor(rel / span)
                let wBase = next - (k * span)

                // Correct if >= max, using gswitch for SIMD compatibility
                let corrected = b.gswitch(wBase >= max, wBase - span, wBase)

                // Reset override using gswitch
                let finalValue = b.gswitch(reset > zero, min, corrected)

                // Store final value
                _ = b.memoryWrite(cellId, b.cast(idx, to: .int), finalValue)

                // Output the previous value (like scalar accum)
                try b.writeOutput(node, acc)
            } else {
                // Scalar accum: use original implementation
                let (incr, reset, min, max) = b.values(inputs, count: 4)
                b.use(val: u_accum(cellId, incr: incr, reset: reset, min: min, max: max)(b))
            }
        case .click(let cellId):
            guard inputs.count == 0 else {
                throw DGenError.insufficientInputs(
                    operator: "click", expected: 0, actual: inputs.count)
            }
            b.use(val: u_click(cellId)(b))
        case .noise(let cellId):
            guard inputs.count == 0 else {
                throw DGenError.insufficientInputs(
                    operator: "noise", expected: 0, actual: inputs.count)
            }
            b.use(val: u_noise(cellId)(b))
        case .phasor(let cellId):
            guard inputs.count == 2 else {
                throw DGenError.insufficientInputs(
                    operator: "phasor", expected: 2, actual: inputs.count)
            }

            // Check if we're in a tensor context
            if let tensorIndex = ctx.tensorIndices[nodeId] {
                // Tensor phasor: read freq from tensor, use indexed state
                // Mark as requiring scalar execution - phasor state accumulates sample-by-sample
                b.markRequiresScalar()

                let freq = try b.readInput(node, inputs, at: 0)
                let reset = try b.readInput(node, inputs, at: 1)
                let idx = b.value(tensorIndex)

                // Phasor accumulator logic with indexed state
                // Uses gswitch instead of if statements for SIMD compatibility
                let sampleRate = b.constant(44100.0)
                let incr = freq / sampleRate
                let zero = b.constant(0.0)
                let one = b.constant(1.0)

                // Load current state from indexed position
                let acc = b.memoryRead(cellId, b.cast(idx, to: .int))

                let nextCand = acc + incr
                let next = b.gswitch(reset > zero, zero, nextCand)

                // Modulo wrap to [0, 1)
                let k = b.floor(next)
                let wBase = next - k

                // Correct if >= 1, using gswitch for SIMD compatibility
                let corrected = b.gswitch(wBase >= one, wBase - one, wBase)

                // Reset override using gswitch
                let finalValue = b.gswitch(reset > zero, zero, corrected)

                // Store final value
                _ = b.memoryWrite(cellId, b.cast(idx, to: .int), finalValue)

                // Output the previous value (like scalar phasor)
                try b.writeOutput(node, acc)
            } else {
                // Scalar phasor: use original implementation
                let (freq, reset) = b.values(inputs, count: 2)
                b.use(val: u_phasor(cellId, freq: freq, reset: reset)(b))
            }
        case .output(let outputNumber):
            guard inputs.count == 1 else {
                throw DGenError.insufficientInputs(
                    operator: "output", expected: 1, actual: inputs.count)
            }
            b.use(val: b.output(outputNumber, b.value(inputs[0])))
        case .input(let inputNumber):
            b.use(val: b.input(inputNumber))
        case .seq:
            // Sequential execution - assumes all input nodes have been emitted in topological order
            // This operator simply returns the value of the last input
            guard node.inputs.count >= 2 else {
                throw DGenError.insufficientInputs(
                    operator: "seq", expected: 2, actual: node.inputs.count)
            }

            // Use the value of the last input node (all inputs should already be emitted)
            if let lastInputId = node.inputs.last,
                let lastValue = ctx.values[lastInputId]
            {
                b.use(val: b.value(lastValue))
            } else {
                throw DGenError.insufficientInputs(
                    operator: "seq", expected: node.inputs.count,
                    actual: node.inputs.compactMap { ctx.values[$0] }.count)
            }

        // MARK: - Tensor Operations

        case .conv1d(let kernelSize):
            guard node.inputs.count >= 2,
                case .tensor(let outShape) = node.shape,
                case .tensor(let inShape) = g.nodes[node.inputs[0]]?.shape, inShape.count == 1,
                let outCell = g.nodeToTensor[node.id].flatMap({ g.tensors[$0] })?.cellId,
                let inTensor = g.nodeToTensor[node.inputs[0]].flatMap({ g.tensors[$0] }),
                let kTensor = g.nodeToTensor[node.inputs[1]].flatMap({ g.tensors[$0] })
            else {
                throw DGenError.tensorError(op: "conv1d", reason: "requires 1D input/output tensors")
            }

            let inLen = inShape[0]
            let pad = kernelSize / 2

            b.parallelRange(outShape.reduce(1, *)) { flatIdx in
                let outX = b.cast(flatIdx, to: .int)
                let acc = b.float(0.0)

                b.loop(kernelSize) { kx in
                    let inX = outX + b.cast(kx, to: .float) - b.constant(Float(pad))
                    let inBounds = (inX >= b.constant(0)) * (inX < b.constant(Float(inLen)))

                    let rawIdx = b.tensorMemoryIndex(inTensor, indices: [b.cast(inX, to: .int)])
                    let safeIdx = b.gswitch(inBounds, rawIdx, b.constant(0))
                    let inVal = b.gswitch(inBounds, b.memoryRead(inTensor.cellId, safeIdx), b.constant(0))

                    let kMemIdx = b.tensorMemoryIndex(kTensor, indices: [b.cast(kx, to: .int)])
                    let kVal = b.memoryRead(kTensor.cellId, kMemIdx)

                    acc.accumulate(inVal * kVal)
                }
                _ = b.memoryWrite(outCell, b.cast(flatIdx, to: .int), acc.value)
            }

        case .conv2d(let kernelShape):
            guard node.inputs.count >= 2,
                case .tensor(let outShape) = node.shape,
                case .tensor(let inShape) = g.nodes[node.inputs[0]]?.shape, inShape.count == 2,
                let outCell = g.nodeToTensor[node.id].flatMap({ g.tensors[$0] })?.cellId,
                let inTensor = g.nodeToTensor[node.inputs[0]].flatMap({ g.tensors[$0] }),
                let kTensor = g.nodeToTensor[node.inputs[1]].flatMap({ g.tensors[$0] })
            else {
                throw DGenError.tensorError(
                    op: "conv2d", reason: "requires 2D input/output tensors")
            }

            let (inH, inW) = (inShape[0], inShape[1])
            let (kH, kW) = (kernelShape[0], kernelShape[1])
            let (padH, padW) = (kH / 2, kW / 2)

            b.parallelRange(outShape.reduce(1, *)) { flatIdx in
                let outY = b.cast(flatIdx, to: .int) / b.constant(Float(inW))
                let outX = b.cast(flatIdx, to: .int) % b.constant(Float(inW))
                let acc = b.float(0.0)

                b.loop(kH) { ky in
                    b.loop(kW) { kx in
                        let inY = outY + b.cast(ky, to: .float) - b.constant(Float(padH))
                        let inX = outX + b.cast(kx, to: .float) - b.constant(Float(padW))

                        let inBounds =
                            (inY >= b.constant(0)) * (inY < b.constant(Float(inH)))
                            * (inX >= b.constant(0)) * (inX < b.constant(Float(inW)))

                        let rawIdx = b.tensorMemoryIndex(inTensor, indices: [b.cast(inY, to: .int), b.cast(inX, to: .int)])
                        let safeIdx = b.gswitch(inBounds, rawIdx, b.constant(0))
                        let inVal = b.gswitch(inBounds, b.memoryRead(inTensor.cellId, safeIdx), b.constant(0))

                        let kMemIdx = b.tensorMemoryIndex(kTensor, indices: [b.cast(ky, to: .int), b.cast(kx, to: .int)])
                        let kVal = b.memoryRead(kTensor.cellId, kMemIdx)

                        acc.accumulate(inVal * kVal)
                    }
                }
                _ = b.memoryWrite(outCell, b.cast(flatIdx, to: .int), acc.value)
            }

        case .sum:
            guard case .tensor(let shape) = g.nodes[node.inputs[0]]?.shape,
                let inTensor = g.nodeToTensor[node.inputs[0]].flatMap({ g.tensors[$0] })
            else {
                if let s = inputs.first { b.use(val: b.value(s)) }
                break
            }
            let acc = b.float(0.0)
            b.loop(shape.reduce(1, *)) { i in
                let val = b.tensorRead(inTensor, flatIdx: i, shape: shape)
                acc.accumulate(val)
            }
            b.use(val: acc.value)

        case .sumAxis(let axis):
            guard case .tensor(let inShape) = g.nodes[node.inputs[0]]?.shape,
                case .tensor = node.shape,
                let inTensor = g.nodeToTensor[node.inputs[0]].flatMap({ g.tensors[$0] }),
                let outCell = g.nodeToTensor[node.id].flatMap({ g.tensors[$0] })?.cellId,
                let loopIdx = b.ctx.tensorIndices[node.id],
                axis >= 0 && axis < inShape.count
            else {
                throw DGenError.tensorError(op: "sumAxis", reason: "invalid input")
            }

            let outIdx = b.value(loopIdx)
            let acc = b.float(0.0)

            b.loop(inShape[axis]) { reduceIdx in
                let rIdx = b.cast(reduceIdx, to: .float)
                let oIdx = b.cast(outIdx, to: .float)

                let indices: [Expr]
                switch inShape.count {
                case 1:
                    indices = [rIdx]
                case 2:
                    indices = axis == 0 ? [rIdx, oIdx] : [oIdx, rIdx]
                case 3:
                    let innerDim = axis == 2 ? inShape[1] : inShape[2]
                    let outer = b.floor(b.cast(outIdx, to: .int) / b.constant(Float(innerDim)))
                    let inner = b.cast(outIdx, to: .int) - outer * b.constant(Float(innerDim))
                    switch axis {
                    case 0: indices = [rIdx, outer, inner]
                    case 1: indices = [outer, rIdx, inner]
                    default: indices = [outer, inner, rIdx]
                    }
                default:
                    // Fallback for higher dimensions - uses tensorRead
                    let val = b.tensorRead(inTensor, flatIdx: reduceIdx, shape: inShape)
                    acc.accumulate(val)
                    return
                }

                let val = b.tensorRead(inTensor, indices: indices)
                acc.accumulate(val)
                _ = b.memoryWrite(outCell, b.cast(outIdx, to: .int), acc.value)
            }

        case .reshape(let newShape):
            // Reshape is metadata-only - the data stays in place
            // Just register that this node produces a tensor view
            // The actual shape change is handled by the tensor metadata
            ctx.values[nodeId] = .empty
            // Emit marker UOp for debugging and to signal SIMD should be disabled
            ops.append(UOp(op: .reshape(newShape), value: .empty))

        case .transpose(let axes):
            // Transpose is metadata-only for contiguous layouts
            // For non-trivial transposes, we may need to copy data
            // For now, just register as a view - emit will use strides
            ctx.values[nodeId] = .empty
            // Emit marker UOp for debugging and to signal SIMD should be disabled
            ops.append(UOp(op: .transpose(axes), value: .empty))

        case .shrink(let ranges):
            // Shrink is metadata-only - uses offset + strides to access slice
            ctx.values[nodeId] = .empty
            // Emit marker UOp for debugging and to signal SIMD should be disabled
            ops.append(UOp(op: .shrink(ranges), value: .empty))

        case .pad(let padding):
            // Pad is a virtual view - reads return 0 for padded regions
            ctx.values[nodeId] = .empty
            // Emit marker UOp for debugging and to signal SIMD should be disabled
            ops.append(UOp(op: .pad(padding), value: .empty))

        case .peek:
            // Lazy peek: read from 2D tensor at (index, channel) with linear interpolation
            // Inputs: [tensor, index, channel]
            guard node.inputs.count == 3 else {
                throw DGenError.insufficientInputs(operator: "peek", expected: 3, actual: node.inputs.count)
            }

            let tensorInput = node.inputs[0]

            // Get tensor shape from the input node
            guard let inputNode = g.nodes[tensorInput],
                  case .tensor(let shape) = inputNode.shape,
                  shape.count >= 2 else {
                throw DGenError.tensorError(op: "peek", reason: "requires 2D tensor input")
            }

            // Try to get concrete tensor, or use shape info to compute access
            let channelSize = shape[0]
            let numChannels = shape[1]

            // Read index and channel inputs
            let index = try b.readInput(node, inputs, at: 1)
            let channel = try b.readInput(node, inputs, at: 2)

            let one = b.constant(1.0)
            let zero = b.constant(0.0)
            let channelSizeFloat = b.constant(Float(channelSize))

            // Wrap index within channel using modulo
            let wrappedIndex = b.mod(index, channelSizeFloat)
            let isNegative = wrappedIndex < zero
            let positiveIndex = b.gswitch(isNegative, wrappedIndex + channelSizeFloat, wrappedIndex)

            // Clamp channel to valid range [0, numChannels-1]
            let clampedChannel = b.floor(b.max(zero, b.min(channel, b.constant(Float(numChannels - 1)))))
            let channelOffset = channelSizeFloat * clampedChannel

            // Calculate final read position
            let finalReadPos = channelOffset + positiveIndex

            // Linear interpolation for fractional indices
            let flooredPos = b.floor(finalReadPos)
            let frac = finalReadPos - flooredPos

            // Get tensor cellId - either from concrete tensor or from input tensor
            let cellId: CellID
            if let tensorId = g.nodeToTensor[tensorInput],
               let tensor = g.tensors[tensorId] {
                cellId = tensor.cellId
            } else {
                throw DGenError.tensorError(op: "peek", reason: "frame-based tensor peek requires tensor context - not yet implemented")
            }

            // Read two samples for interpolation
            let sample1 = b.memoryRead(cellId, b.cast(flooredPos, to: .int))
            let nextPos = flooredPos + one

            // Wrap nextPos if it crosses channel boundary
            let nextChannelOffset = channelOffset + channelSizeFloat
            let nextPosWrapped = b.gswitch(nextPos >= nextChannelOffset, channelOffset, nextPos)

            let sample2 = b.memoryRead(cellId, b.cast(nextPosWrapped, to: .int))

            // Linear interpolation: (1-frac)*sample1 + frac*sample2
            let interpolated = b.mix(sample1, sample2, frac)
            b.use(val: interpolated)
        }
        ops.append(contentsOf: b.ops)
        return ops
    }

    func emitBackward(ctx: IRContext, g: Graph, nodeId: NodeID) throws
        -> [UOp]
    {
        guard let node = g.nodes[nodeId] else {
            return []
        }

        var ops: [UOp] = []
        let b = IRBuilder(ctx: ctx, nodeId: nodeId)

        // we'll assume that gradient seeds are set to 1 before running an epoch, and everything else is 0
        var gradOutput = ctx.useConstant(src: nil, value: 1.0)
        if let gradCellId = ctx.gradients[nodeId] {
            gradOutput = b.loadGrad(gradCellId).lazy
        } else {
            // Allocate a gradient ID for this node (not a seed).
            // Only explicit loss nodes should be seeds.
            let gradCellId = ctx.useGradient(src: nodeId, seed: false)
            gradOutput = b.loadGrad(gradCellId).lazy
        }

        // Collect operands
        let inputs: [Lazy] = node.inputs.compactMap { ctx.values[$0] }

        switch self {
        case .constant(_):
            // Constants have no gradients to propagate
            // should we return 0 so that it can actually propagate?
            // constant is a leaf so it'd be at the very end anyway so probably fine
            return []
        case .tensorRef(_):
            return []
        case .and, .or, .xor:
            return []
        case .click:
            // TODO - implement backprop for click
            break
        case .noise:
            // Noise is non-differentiable (random values have zero gradient)
            break
        case .add:
            // d(x+y)/dx = 1, d(x+y)/dy = 1
            guard node.inputs.count == 2 else { fatalError("add requires 2 inputs") }
            try emitBinaryOpBackward(
                b: b, g: g, ctx: ctx, node: node, gradOutput: gradOutput,
                gradLhs: { gradOut, _, _ in gradOut },
                gradRhs: { gradOut, _, _ in gradOut }
            )
        case .sub:
            // z = x - y  =>  dz/dx = 1, dz/dy = -1
            guard node.inputs.count == 2 else { fatalError("sub requires 2 inputs") }
            try emitBinaryOpBackward(
                b: b, g: g, ctx: ctx, node: node, gradOutput: gradOutput,
                gradLhs: { gradOut, _, _ in gradOut },
                gradRhs: { gradOut, _, _ in b.constant(0.0) - gradOut }
            )
        case .mod:
            // z = fmod(a, b) ≈ a - b*trunc(a/b). Treat trunc grad as 0 almost everywhere.
            // -> dz/da ≈ 1, dz/db ≈ 0 (stable choice for DSP wrapping)
            guard node.inputs.count == 2 else { fatalError("mod requires 2 inputs") }
            b.grad(node.inputs[0], value: gradOutput)
            b.grad(node.inputs[1], value: ctx.useConstant(src: nil, value: 0.0))
        case .param(_):
            // the canonical gradient for the param already lives in gradients, in the row for that param’s gradId.
            // There’s nothing left to push. It’s a leaf.
            break
        case .min:
            // d(min(x,y))/dx = (x <= y) ? 1 : 0, d(min(x,y))/dy = (y < x) ? 1 : 0
            guard inputs.count == 2 else { fatalError("min requires 2 inputs") }
            let x = b.tapeValue(node.inputs[0])
            let y = b.tapeValue(node.inputs[1])
            let xIsMin = x <= y
            let yIsMin = y < x
            let gradX = b.gswitch(xIsMin, b.value(gradOutput), b.constant(0.0))
            let gradY = b.gswitch(yIsMin, b.value(gradOutput), b.constant(0.0))
            b.grad(node.inputs[0], value: gradX.lazy)
            b.grad(node.inputs[1], value: gradY.lazy)
        case .max:
            // d(max(x,y))/dx = (x >= y) ? 1 : 0, d(max(x,y))/dy = (y > x) ? 1 : 0
            guard inputs.count == 2 else { fatalError("max requires 2 inputs") }
            let x = b.tapeValue(node.inputs[0])
            let y = b.tapeValue(node.inputs[1])
            let xIsMax = x >= y
            let yIsMax = y > x
            let gradX = b.gswitch(xIsMax, b.value(gradOutput), b.constant(0.0))
            let gradY = b.gswitch(yIsMax, b.value(gradOutput), b.constant(0.0))
            b.grad(node.inputs[0], value: gradX.lazy)
            b.grad(node.inputs[1], value: gradY.lazy)
        case .mul:
            // d(x*y)/dx = y, d(x*y)/dy = x
            guard inputs.count == 2 else { fatalError("mul \(node.id) requires 2 inputs") }
            try emitBinaryOpBackward(
                b: b, g: g, ctx: ctx, node: node, gradOutput: gradOutput,
                gradLhs: { gradOut, lhsVal, rhsVal in gradOut * rhsVal },
                gradRhs: { gradOut, lhsVal, rhsVal in gradOut * lhsVal }
            )
        case .div:
            // z = x / y  =>  dz/dx = 1/y, dz/dy = -x / y^2
            guard inputs.count == 2 else { fatalError("div \(node.id) requires 2 inputs") }
            try emitBinaryOpBackward(
                b: b, g: g, ctx: ctx, node: node, gradOutput: gradOutput,
                gradLhs: { gradOut, _, rhsVal in gradOut / rhsVal },
                gradRhs: { gradOut, lhsVal, rhsVal in (b.constant(0.0) - gradOut) * (lhsVal / (rhsVal * rhsVal)) }
            )
        case .abs:
            // d(abs(x))/dx = sign(x), but zero at x=0
            guard inputs.count == 1 else { fatalError("abs requires 1 input") }
            let input = b.tapeValue(node.inputs[0])
            let grad = b.value(gradOutput) * b.sign(input)
            b.grad(node.inputs[0], value: grad.lazy)
        case .sign:
            // d(sign(x))/dx = 0 everywhere except at x=0 where it's undefined
            guard inputs.count == 1 else { fatalError("sign requires 1 input") }
            let zero = ctx.useConstant(src: nil, value: 0.0)
            b.grad(node.inputs[0], value: zero)
        case .sin:
            // d(sin(x))/dx = cos(x)
            guard inputs.count == 1 else { fatalError("sin requires 1 input") }
            let input = b.tapeValue(node.inputs[0])
            let grad = b.value(gradOutput) * b.cos(input)
            b.grad(node.inputs[0], value: grad.lazy)
        case .cos:
            // d(cos(x))/dx = -sin(x)
            guard inputs.count == 1 else { fatalError("cos requires 1 input") }
            let input = b.tapeValue(node.inputs[0])
            let grad = b.value(gradOutput) * (b.constant(0.0) - b.sin(input))
            b.grad(node.inputs[0], value: grad.lazy)
        case .tan:
            // d(tan(x))/dx = sec²(x) = 1/cos²(x)
            guard inputs.count == 1 else { fatalError("tan requires 1 input") }
            let input = b.tapeValue(node.inputs[0])
            let cosInput = b.cos(input)
            let sec2 = b.constant(1.0) / (cosInput * cosInput)
            let grad = b.value(gradOutput) * sec2
            b.grad(node.inputs[0], value: grad.lazy)
        case .tanh:
            // d(tanh(x))/dx = 1 - tanh(x)^2
            guard inputs.count == 1 else { fatalError("tanh requires 1 input") }
            let input = b.tapeValue(node.inputs[0])
            let t = b.tanh(input)
            let grad = b.value(gradOutput) * (b.constant(1.0) - t * t)
            b.grad(node.inputs[0], value: grad.lazy)
        case .exp:
            // d(exp(x))/dx = exp(x)
            guard inputs.count == 1 else { fatalError("exp requires 1 input") }
            let input = b.tapeValue(node.inputs[0])
            let grad = b.value(gradOutput) * b.exp(input)
            b.grad(node.inputs[0], value: grad.lazy)
        case .log:
            // d(log(x))/dx = 1/x
            guard inputs.count == 1 else { fatalError("log requires 1 input") }
            let input = b.tapeValue(node.inputs[0])
            let grad = b.value(gradOutput) / input
            b.grad(node.inputs[0], value: grad.lazy)
        case .log10:
            // d(log10(x))/dx = 1 / (x * ln(10))
            guard inputs.count == 1 else { fatalError("log10 requires 1 input") }
            let input = b.tapeValue(node.inputs[0])
            let ln10 = b.constant(2.302585092994046)  // natural log of 10
            let grad = b.value(gradOutput) / (input * ln10)
            b.grad(node.inputs[0], value: grad.lazy)
        case .sqrt:
            // d(sqrt(x))/dx = 1/(2*sqrt(x))
            guard inputs.count == 1 else { fatalError("sqrt requires 1 input") }
            let input = b.tapeValue(node.inputs[0])
            let grad = b.value(gradOutput) / (b.constant(2.0) * b.sqrt(input))
            b.grad(node.inputs[0], value: grad.lazy)
        case .pow:
            // d(x^y)/dx = y * x^(y-1), d(x^y)/dy = x^y * ln(x)
            guard inputs.count == 2 else { fatalError("pow requires 2 inputs") }
            let base = b.tapeValue(node.inputs[0])
            let exponent = b.tapeValue(node.inputs[1])
            let result = b.pow(base, exponent)

            // Gradient w.r.t. base: y * x^(y-1)
            let baseGrad = b.value(gradOutput) * exponent * b.pow(base, exponent - b.constant(1.0))
            b.grad(node.inputs[0], value: baseGrad.lazy)

            // Gradient w.r.t. exponent: x^y * ln(x)
            let expGrad = b.value(gradOutput) * result * b.log(base)
            b.grad(node.inputs[1], value: expGrad.lazy)
        case .atan2:
            // d(atan2(y,x))/dy = x/(x²+y²), d(atan2(y,x))/dx = -y/(x²+y²)
            guard inputs.count == 2 else { fatalError("atan2 requires 2 inputs") }
            let y = b.tapeValue(node.inputs[0])
            let x = b.tapeValue(node.inputs[1])
            let denom = x * x + y * y
            let gradY = b.value(gradOutput) * (x / denom)
            let gradX = b.value(gradOutput) * (b.constant(0.0) - y / denom)
            b.grad(node.inputs[0], value: gradY.lazy)
            b.grad(node.inputs[1], value: gradX.lazy)
        case .mse:
            // loss = (a - b)^2; grads: d/da = 2*(a-b)*go, d/db = -2*(a-b)*go
            guard inputs.count == 2 else { fatalError("mse requires 2 inputs") }
            let a = b.tapeValue(node.inputs[0])
            let c = b.tapeValue(node.inputs[1])
            let diff = a - c
            let two = b.constant(2.0)
            let gradA = b.value(gradOutput) * two * diff
            let gradB = b.value(gradOutput) * (b.constant(0.0) - two * diff)
            b.grad(node.inputs[0], value: gradA.lazy)
            b.grad(node.inputs[1], value: gradB.lazy)

        case let .spectralLossPass1(windowSize, scratchCell):
            guard inputs.count == 2 else { fatalError("spectralLossPass1 requires 2 inputs") }
            let sig1 = b.tapeValue(node.inputs[0])
            let sig2 = b.tapeValue(node.inputs[1])
            let gradId1 = b.ctx.useGradient(src: node.inputs[0])
            let gradId2 = b.ctx.useGradient(src: node.inputs[1])

            // Pass B: Reduce from memory to gradients (READ)
            // This runs SECOND in backward (after Pass2), so memory has been written
            let (grad1, grad2) = u_spectralLossBackwardPass2(
                windowSize, scratchCell, sig1, sig2, b.value(gradOutput), gradId1, gradId2
            )(b)

            // Propagate gradients to original signals
            b.grad(node.inputs[0], value: grad1.lazy)
            b.grad(node.inputs[1], value: grad2.lazy)

        case let .spectralLossPass2(windowSize, scratchCell):
            guard inputs.count == 1 else { fatalError("spectralLossPass2 requires 1 input") }

            // Get the original signal inputs from Pass1's node
            let pass1Node = g.nodes[node.inputs[0]]!
            guard pass1Node.inputs.count == 2 else { fatalError("Expected Pass1 to have 2 inputs") }

            let sig1 = b.tapeValue(pass1Node.inputs[0])
            let sig2 = b.tapeValue(pass1Node.inputs[1])

            // Pass A: Accumulate per-window gradient contributions to memory (WRITE)
            // This runs FIRST in backward (before Pass1), writing data that Pass1 will read
            // Don't propagate gradients - Pass1 will handle that
            u_spectralLossBackwardPass1(
                windowSize, scratchCell, sig1, sig2, b.value(gradOutput)
            )(b)
        case .floor:
            // d(floor(x))/dx = 0 (floor is not differentiable, but we set to 0)
            guard inputs.count == 1 else { fatalError("floor requires 1 input") }
            let zeroGrad = ctx.useConstant(src: nil, value: 0.0)
            b.grad(node.inputs[0], value: zeroGrad)
        case .ceil:
            // d(ceil(x))/dx = 0 (floor is not differentiable, but we set to 0)
            guard inputs.count == 1 else { fatalError("floor requires 1 input") }
            let zeroGrad = ctx.useConstant(src: nil, value: 0.0)
            b.grad(node.inputs[0], value: zeroGrad)
        case .round:
            // d(ceil(x))/dx = 0 (floor is not differentiable, but we set to 0)
            guard inputs.count == 1 else { fatalError("floor requires 1 input") }
            let zeroGrad = ctx.useConstant(src: nil, value: 0.0)
            b.grad(node.inputs[0], value: zeroGrad)
        case .memoryRead(_):
            // For memoryRead, gradient flows through to the values written to memory
            // This is complex and depends on the memory write operations
            guard inputs.count == 1 else { fatalError("memoryRead requires 1 input") }
            // For now, treat as zero gradient for the offset
            let zeroGrad = ctx.useConstant(src: nil, value: 0.0)
            b.grad(node.inputs[0], value: zeroGrad)
        case .memoryWrite(_):
            // For memoryWrite, gradient flows through to both offset and value inputs
            guard inputs.count == 2 else { fatalError("memoryWrite requires 2 inputs") }
            // Gradient for offset is typically zero (address computation)
            let zeroGrad = ctx.useConstant(src: nil, value: 0.0)
            b.grad(node.inputs[0], value: zeroGrad)
            // Gradient for value flows through
            b.grad(node.inputs[1], value: gradOutput)
        case .gt, .gte, .lte, .lt, .eq:
            // Comparisons have zero gradient (non-differentiable)
            guard node.inputs.count == 2 else { fatalError("comparison requires 2 inputs") }
            let zero = ctx.useConstant(src: nil, value: 0.0)
            b.grad(node.inputs[0], value: zero)
            b.grad(node.inputs[1], value: zero)

        case .gswitch:
            // gswitch(cond, x, y) = cond ? x : y
            guard inputs.count == 3 else { fatalError("gswitch requires 3 inputs") }
            let cond = b.tapeValue(node.inputs[0])
            let gradX = b.gswitch(cond, b.value(gradOutput), b.constant(0.0))
            let gradY = b.gswitch(cond, b.constant(0.0), b.value(gradOutput))
            b.grad(node.inputs[0], value: ctx.useConstant(src: nil, value: 0.0))
            b.grad(node.inputs[1], value: gradX.lazy)
            b.grad(node.inputs[2], value: gradY.lazy)

        case .selector:
            // selector(mode, options[]) -> gradient flows only to the selected option
            guard inputs.count >= 2 else { fatalError("selector requires at least 2 inputs") }

            // Gradient for mode is always zero (index is non-differentiable)
            b.grad(node.inputs[0], value: ctx.useConstant(src: nil, value: 0.0))

            // For each option, gradient is non-zero only if it was selected
            let mode = b.tapeValue(node.inputs[0])
            for i in 1..<node.inputs.count {
                let optionIndex = b.constant(Float(i - 1))
                let isSelected = mode == optionIndex
                let gradOption = b.gswitch(isSelected, b.value(gradOutput), b.constant(0.0))
                b.grad(node.inputs[i], value: gradOption.lazy)
            }

        case .mix:
            // mix(x, y, t) = x * (1-t) + y * t
            guard inputs.count == 3 else { fatalError("mix requires 3 inputs") }
            let x = b.tapeValue(node.inputs[0])
            let y = b.tapeValue(node.inputs[1])
            let t = b.tapeValue(node.inputs[2])

            // d/dx = (1-t)
            let gradX = b.value(gradOutput) * (b.constant(1.0) - t)
            // d/dy = t
            let gradY = b.value(gradOutput) * t
            // d/dt = y - x
            let gradT = b.value(gradOutput) * (y - x)

            b.grad(node.inputs[0], value: gradX.lazy)
            b.grad(node.inputs[1], value: gradY.lazy)
            b.grad(node.inputs[2], value: gradT.lazy)

        case .historyReadWrite(let cellId):
            // Combined read (returns previous state) and write (stores current input)
            // Backward:
            //  - Carry to previous timestep must be the gradient w.r.t. y[i] (the read output),
            //    not the input gradient. Do not re-add existing carry here; that causes runaway.
            //  - The input only receives gradient from the future via the carry.
            guard inputs.count == 1 else { fatalError("historyReadWrite requires 1 input") }
            let carry = b.loadGradMemory(cellId)
            // Set next carry for i-1 to the upstream grad at this read
            _ = b.storeGradMemory(cellId, b.value(gradOutput))
            // Gradient to the written input is only the carry from future
            b.grad(node.inputs[0], value: carry.lazy)

        case .historyWrite(let cellId):
            // Write stores current input into the cell; its forward output is the previous value.
            // Backward: input only gets gradient from future reads (the carry). There is no
            // local gradient path from this op's output to its input.
            guard inputs.count == 1 else { fatalError("history write requires 1 input") }
            let carry = b.loadGradMemory(cellId)
            b.grad(node.inputs[0], value: carry.lazy)
        case .historyRead(let cellId):
            // Read exposes previous state; the carry for the previous timestep must equal
            // the gradient w.r.t. this read's output at the current time.
            // Do not add the existing carry here — that was already used to form gradOutput.
            _ = b.storeGradMemory(cellId, b.value(gradOutput))
        case .latch(_):
            // Gradient flows through value input only when condition was true
            guard inputs.count == 2 else { fatalError("latch requires 2 inputs") }
            let cond = b.tapeValue(node.inputs[1])
            let gradValue = b.gswitch(cond > b.constant(0), b.value(gradOutput), b.constant(0.0))
            b.grad(node.inputs[0], value: gradValue.lazy)
            b.grad(node.inputs[1], value: ctx.useConstant(src: nil, value: 0.0))

        case .accum(let cellId):
            let reset = b.tapeValue(node.inputs[1])

            // Gradient for increment (blocked by reset)
            let gradIncr = b.gswitch(reset > b.constant(0), b.constant(0.0), b.value(gradOutput))

            // handle temporal gradient flow:
            // The gradient also flows to the previous accumulated value
            let gradFromFuture = b.loadGradMemory(cellId)

            // The total gradient flowing backward through time
            let gradToPrev = b.value(gradOutput) + gradFromFuture

            // Store for previous timestep's accum
            _ = b.storeGradMemory(cellId, gradToPrev)

            b.grad(node.inputs[0], value: gradIncr.lazy)
            b.grad(node.inputs[1], value: ctx.useConstant(src: nil, value: 0.0))
            b.grad(node.inputs[2], value: ctx.useConstant(src: nil, value: 0.0))
            b.grad(node.inputs[3], value: ctx.useConstant(src: nil, value: 0.0))

        case .phasor(_):
            // NOTE ON GRADIENT SCALE VS frameCount
            // ------------------------------------------------------------
            // The phasor’s backward pass produces a gradient proportional
            // to the current frame index (time). Even though Training.swift
            // averages gradients across frames, the mean of i/sampleRate over
            // i = 0..(frameCount-1) grows with the time horizon (~O(frameCount)).
            // As a result, increasing frameCount increases the effective
            // gradient scale for frequency parameters, requiring a smaller
            // learning rate to maintain stability. This is independent of the
            // particular loss (e.g., spectral loss) and comes from the time-
            // weighted nature of d(phase)/d(freq).
            //
            // TODO - use actual sampleRate in system
            let sampleRate = b.constant(44100.0)
            let currentTime = b.frameIndex(nodeId)

            // Compute this timestep's frequency gradient
            // d(phase)/d(freq) = time / sampleRate (since phase accumulates as: phase += freq/sampleRate)
            let gradFreq = b.value(gradOutput) * currentTime / sampleRate

            // Write gradient directly (no accumulation needed - test will sum across frames)
            b.grad(node.inputs[0], value: gradFreq.lazy)
        case .output(_):
            // Output just passes gradient through to its input
            guard inputs.count == 1 else { fatalError("output requires 1 input") }
            b.grad(node.inputs[0], value: gradOutput)
        case .input(_):
            break
        case .seq:
            // TODO: take another look at this, all the computation in each input is executed, even though the final one is returned
            // Gradient flows only to the last input (the one whose value is returned)
            guard node.inputs.count >= 2 else { fatalError("seq requires at least 2 inputs") }
            // Zero gradients for all inputs except the last
            for i in 0..<(node.inputs.count - 1) {
                b.grad(node.inputs[i], value: ctx.useConstant(src: nil, value: 0.0))
            }
            // Pass gradient to the last input
            if let lastInput = node.inputs.last {
                b.grad(lastInput, value: gradOutput)
            }

        // MARK: - Tensor Operation Gradients

        case .conv1d(_):
            // Conv1d gradient: similar to conv2d, would require transposed convolution
            // For now, we provide zero gradients (TODO: implement full backprop)
            guard inputs.count == 2 else { fatalError("conv1d requires 2 inputs") }
            b.grad(node.inputs[0], value: ctx.useConstant(src: nil, value: 0.0))
            b.grad(node.inputs[1], value: ctx.useConstant(src: nil, value: 0.0))

        case .conv2d(_):
            // Conv2d gradient: gradient w.r.t. input = conv2d(grad_output, flipped_kernel)
            // gradient w.r.t. kernel = conv2d(input, grad_output)
            // This is complex - for now, we provide zero gradients (TODO: implement full backprop)
            guard inputs.count == 2 else { fatalError("conv2d requires 2 inputs") }
            // For membrane simulation, we often don't need kernel gradients (fixed Laplacian)
            // Input gradient would require transposed convolution
            b.grad(node.inputs[0], value: ctx.useConstant(src: nil, value: 0.0))
            b.grad(node.inputs[1], value: ctx.useConstant(src: nil, value: 0.0))

        case .sum:
            // d(sum(x))/dx[i] = 1 for all i
            // Gradient broadcasts from scalar back to tensor
            guard inputs.count == 1 else { fatalError("sum requires 1 input") }
            let inputNodeId = node.inputs[0]

            // Check if input is a tensor
            guard let inputNode = g.nodes[inputNodeId],
                  case .tensor(let inputShape) = inputNode.shape else {
                // Scalar input: just pass gradient through
                b.grad(node.inputs[0], value: gradOutput)
                break
            }

            let size = inputShape.reduce(1, *)

            // Allocate tensor gradients for input
            let _ = ctx.useTensorGradient(src: inputNodeId, size: size)

            // Broadcast: dL/dx[i] = dL/d(sum) for all i
            b.parallelRange(size) { idx in
                b.tensorGrad(inputNodeId, index: idx, value: gradOutput)
            }

        case .sumAxis(let axis):
            // d(sumAxis(x, axis))/dx[i] = dL/d(output[reduced_index])
            // Each input element's gradient = gradient of corresponding output element
            guard inputs.count == 1 else { fatalError("sumAxis requires 1 input") }
            let inputNodeId = node.inputs[0]

            guard let inputNode = g.nodes[inputNodeId],
                  case .tensor(let inputShape) = inputNode.shape,
                  case .tensor(let outputShape) = node.shape else {
                // Fallback for non-tensor case
                b.grad(node.inputs[0], value: gradOutput)
                break
            }

            let inputSize = inputShape.reduce(1, *)
            let normalizedAxis = axis < 0 ? inputShape.count + axis : axis

            // Allocate tensor gradients for input and ensure we have them for output
            let _ = ctx.useTensorGradient(src: inputNodeId, size: inputSize)
            let outputSize = outputShape.reduce(1, *)
            let _ = ctx.useTensorGradient(src: node.id, size: outputSize)

            // Compute strides for index mapping
            let inputStrides = Tensor.computeRowMajorStrides(inputShape)
            let outputStrides = Tensor.computeRowMajorStrides(outputShape)

            // Each input element maps to an output element by removing the axis dimension
            b.parallelRange(inputSize) { flatIdx in
                // Convert flat index to multi-dimensional index
                // Then remove the axis dimension to get output index

                // Build output flat index by skipping the axis dimension
                var outputFlatIdx = b.int(0)
                var outDim = 0
                for dim in 0..<inputShape.count {
                    if dim == normalizedAxis {
                        continue  // Skip reduced dimension
                    }
                    // coordAtDim = (flatIdx / inputStrides[dim]) % inputShape[dim]
                    let strideExpr = b.int(inputStrides[dim])
                    let shapeExpr = b.int(inputShape[dim])
                    let coordAtDim = b.mod(b.div(flatIdx, strideExpr), shapeExpr)

                    // outputFlatIdx += coordAtDim * outputStrides[outDim]
                    let outStrideExpr = b.int(outputStrides[outDim])
                    outputFlatIdx = b.add(outputFlatIdx, b.mul(coordAtDim, outStrideExpr))
                    outDim += 1
                }

                // Load the gradient from the output tensor
                let gradVal = b.loadTensorGrad(node.id, index: b.cast(outputFlatIdx, to: .float))
                b.tensorGrad(inputNodeId, index: flatIdx, value: gradVal.lazy)
            }

        case .reshape(let newShape):
            // Reshape is metadata-only, gradient flows through unchanged
            // Flat indices are the same between input and output
            guard inputs.count == 1 else { fatalError("reshape requires 1 input") }
            let inputNodeId = node.inputs[0]

            guard let inputNode = g.nodes[inputNodeId],
                  case .tensor(let inputShape) = inputNode.shape else {
                b.grad(node.inputs[0], value: gradOutput)
                break
            }

            let size = inputShape.reduce(1, *)

            // Allocate tensor gradients for both input and output (they share flat indices)
            let _ = ctx.useTensorGradient(src: inputNodeId, size: size)
            let _ = ctx.useTensorGradient(src: node.id, size: newShape.reduce(1, *))

            // Pass through: dL/dx[i] = dL/dy[i] (same flat index)
            b.parallelRange(size) { idx in
                let gradVal = b.loadTensorGrad(node.id, index: idx)
                b.tensorGrad(inputNodeId, index: idx, value: gradVal.lazy)
            }

        case .transpose(let perm):
            // Transpose permutes dimensions, need inverse permutation for gradients
            guard inputs.count == 1 else { fatalError("transpose requires 1 input") }
            let inputNodeId = node.inputs[0]

            guard let inputNode = g.nodes[inputNodeId],
                  case .tensor(let inputShape) = inputNode.shape,
                  case .tensor(let outputShape) = node.shape else {
                b.grad(node.inputs[0], value: gradOutput)
                break
            }

            let size = inputShape.reduce(1, *)

            // Allocate tensor gradients
            let _ = ctx.useTensorGradient(src: inputNodeId, size: size)
            let _ = ctx.useTensorGradient(src: node.id, size: size)

            // Compute inverse permutation
            var inversePerm = [Int](repeating: 0, count: perm.count)
            for (i, p) in perm.enumerated() {
                inversePerm[p] = i
            }

            let inputStrides = Tensor.computeRowMajorStrides(inputShape)
            let outputStrides = Tensor.computeRowMajorStrides(outputShape)

            // For each output element, compute corresponding input element
            b.parallelRange(size) { outputFlatIdx in
                // Convert output flat index to output multi-dim index
                // Apply inverse perm to get input multi-dim index
                // Convert to input flat index

                var inputFlatIdx = b.int(0)
                for outDim in 0..<outputShape.count {
                    // outCoord = (outputFlatIdx / outputStrides[outDim]) % outputShape[outDim]
                    let outStrideExpr = b.int(outputStrides[outDim])
                    let outShapeExpr = b.int(outputShape[outDim])
                    let outCoord = b.mod(b.div(outputFlatIdx, outStrideExpr), outShapeExpr)

                    // This output coord maps to input dimension inversePerm[outDim]
                    let inDim = inversePerm[outDim]
                    let inStrideExpr = b.int(inputStrides[inDim])
                    inputFlatIdx = b.add(inputFlatIdx, b.mul(outCoord, inStrideExpr))
                }

                let gradVal = b.loadTensorGrad(node.id, index: outputFlatIdx)
                b.tensorGrad(inputNodeId, index: b.cast(inputFlatIdx, to: .float), value: gradVal.lazy)
            }

        case .shrink(let ranges):
            // Shrink only includes elements in the specified range
            // Gradients flow back only to elements within the shrunk region
            guard inputs.count == 1 else { fatalError("shrink requires 1 input") }
            let inputNodeId = node.inputs[0]

            guard let inputNode = g.nodes[inputNodeId],
                  case .tensor(let inputShape) = inputNode.shape,
                  case .tensor(let outputShape) = node.shape else {
                b.grad(node.inputs[0], value: gradOutput)
                break
            }

            let inputSize = inputShape.reduce(1, *)
            let outputSize = outputShape.reduce(1, *)

            // Allocate tensor gradients
            let _ = ctx.useTensorGradient(src: inputNodeId, size: inputSize)
            let _ = ctx.useTensorGradient(src: node.id, size: outputSize)

            // Compute start offsets for each dimension
            var startOffsets = [Int]()
            for range in ranges {
                if let (start, _) = range {
                    startOffsets.append(start)
                } else {
                    startOffsets.append(0)
                }
            }

            let inputStrides = Tensor.computeRowMajorStrides(inputShape)
            let outputStrides = Tensor.computeRowMajorStrides(outputShape)

            // For each output element, compute corresponding input element
            b.parallelRange(outputSize) { outputFlatIdx in
                // Convert output index to input index by adding start offsets
                var inputFlatIdx = b.int(0)
                for dim in 0..<outputShape.count {
                    // outCoord = (outputFlatIdx / outputStrides[dim]) % outputShape[dim]
                    let outStrideExpr = b.int(outputStrides[dim])
                    let outShapeExpr = b.int(outputShape[dim])
                    let outCoord = b.mod(b.div(outputFlatIdx, outStrideExpr), outShapeExpr)

                    // inCoord = outCoord + startOffset[dim]
                    let inCoord = b.add(outCoord, b.int(startOffsets[dim]))
                    let inStrideExpr = b.int(inputStrides[dim])
                    inputFlatIdx = b.add(inputFlatIdx, b.mul(inCoord, inStrideExpr))
                }

                let gradVal = b.loadTensorGrad(node.id, index: outputFlatIdx)
                b.tensorGrad(inputNodeId, index: b.cast(inputFlatIdx, to: .float), value: gradVal.lazy)
            }

        case .pad(let padding):
            // Pad adds zeros around the tensor; only inner region has gradient
            guard inputs.count == 1 else { fatalError("pad requires 1 input") }
            let inputNodeId = node.inputs[0]

            guard let inputNode = g.nodes[inputNodeId],
                  case .tensor(let inputShape) = inputNode.shape,
                  case .tensor(let outputShape) = node.shape else {
                b.grad(node.inputs[0], value: gradOutput)
                break
            }

            let inputSize = inputShape.reduce(1, *)
            let outputSize = outputShape.reduce(1, *)

            // Allocate tensor gradients
            let _ = ctx.useTensorGradient(src: inputNodeId, size: inputSize)
            let _ = ctx.useTensorGradient(src: node.id, size: outputSize)

            let inputStrides = Tensor.computeRowMajorStrides(inputShape)
            let outputStrides = Tensor.computeRowMajorStrides(outputShape)

            // For each input element, compute corresponding output element
            b.parallelRange(inputSize) { inputFlatIdx in
                // Convert input index to output index by adding left padding
                var outputFlatIdx = b.int(0)
                for dim in 0..<inputShape.count {
                    let inStrideExpr = b.int(inputStrides[dim])
                    let inShapeExpr = b.int(inputShape[dim])
                    let inCoord = b.mod(b.div(inputFlatIdx, inStrideExpr), inShapeExpr)

                    // outCoord = inCoord + leftPad[dim]
                    let leftPad = padding[dim].0
                    let outCoord = b.add(inCoord, b.int(leftPad))
                    let outStrideExpr = b.int(outputStrides[dim])
                    outputFlatIdx = b.add(outputFlatIdx, b.mul(outCoord, outStrideExpr))
                }

                let gradVal = b.loadTensorGrad(node.id, index: b.cast(outputFlatIdx, to: .float))
                b.tensorGrad(inputNodeId, index: inputFlatIdx, value: gradVal.lazy)
            }

        case .peek:
            // Peek reads a single value from tensor - gradient only affects that position
            // For now, zero gradients (TODO: implement scatter gradient back to tensor position)
            guard inputs.count == 3 else { fatalError("peek requires 3 inputs") }
            b.grad(node.inputs[0], value: ctx.useConstant(src: nil, value: 0.0))
            b.grad(node.inputs[1], value: ctx.useConstant(src: nil, value: 0.0))
            b.grad(node.inputs[2], value: ctx.useConstant(src: nil, value: 0.0))
        }

        ops.append(contentsOf: b.ops)
        return ops
    }
}

// Two-pass spectral loss backward functions

/// Pass A: Accumulate per-window gradient contributions to memory.
/// Each thread i (window end) computes DFT for its window and stores per-sample contributions.
func u_spectralLossBackwardPass1(
    _ windowSize: Int,
    _ scratchCell: CellID,
    _ sig1: Expr,
    _ sig2: Expr,
    _ upstreamGrad: Expr
) -> (IRBuilder) -> Void {
    return { b in
        let numBins = windowSize / 2 + 1
        let i = b.threadIndex()  // Current frame (window end)

        // For each frequency bin
        b.loop(numBins) { binIndex in
            // DFT accumulators (real and imaginary parts)
            let real1 = b.float(0.0)
            let imag1 = b.float(0.0)
            let real2 = b.float(0.0)
            let imag2 = b.float(0.0)

            // Compute DFT over window samples
            b.loop(windowSize) { n in
                let winSize = b.constant(Float(windowSize))
                let j = i - (winSize - b.constant(1.0)) + b.cast(n, to: .float)

                // Load samples from tape with bounds checking
                let s1 = b.tapeLoad(sig1, at: j)
                let s2 = b.tapeLoad(sig2, at: j)

                // DFT basis: e^(-2πi*k*n/N) = cos(angle) - i*sin(angle)
                let binIndexFloat = b.cast(binIndex, to: .float)
                let nFloat = b.cast(n, to: .float)
                let angle = b.constant(-2.0) * b.pi * binIndexFloat * nFloat / winSize
                let c = b.cos(angle)
                let s = b.sin(angle)

                // Accumulate DFT: Real(X[k]) += x[n]*cos, Imag(X[k]) += x[n]*sin
                real1.accumulate(s1 * c)
                imag1.accumulate(s1 * s)
                real2.accumulate(s2 * c)
                imag2.accumulate(s2 * s)
            }

            // Magnitude: |X[k]| = sqrt(Real² + Imag²)
            let mag1 = b.sqrt(real1.value * real1.value + imag1.value * imag1.value)
            let mag2 = b.sqrt(real2.value * real2.value + imag2.value * imag2.value)

            // Loss gradient for this bin: d/d(mag1) of (mag1-mag2)²
            let magDiff = mag1 - mag2
            let lossGrad = b.constant(2.0) * magDiff

            // For each window offset, compute and store contribution
            b.loop(windowSize) { n in
                let binIndexFloat = b.cast(binIndex, to: .float)
                let nFloat = b.cast(n, to: .float)
                let winSize = b.constant(Float(windowSize))
                let angle_n = b.constant(-2.0) * b.pi * binIndexFloat * nFloat / winSize
                let c_n = b.cos(angle_n)
                let s_n = b.sin(angle_n)

                // ∂mag/∂s[n] = (real*cos + imag*sin) / mag
                let eps = b.constant(1e-8)
                let sampleGrad1 = (real1.value * c_n + imag1.value * s_n) / (mag1 + eps)
                let sampleGrad2 = (real2.value * c_n + imag2.value * s_n) / (mag2 + eps)

                // Chain rule: ∂L/∂s = (∂L/∂mag) * (∂mag/∂s) * upstreamGrad
                let contrib1 = lossGrad * sampleGrad1 * upstreamGrad
                let contrib2 = (b.constant(0.0) - lossGrad) * sampleGrad2 * upstreamGrad

                // Write to memory: memory[scratchCell + (i * windowSize * 2) + (n * 2) + component]
                let winSizeConst = b.constant(Float(windowSize))
                let offset1 =
                    i * winSizeConst * b.constant(2.0) + b.cast(n, to: .float) * b.constant(2.0)
                let offset2 = offset1 + b.constant(1.0)

                _ = b.memoryWrite(scratchCell, b.cast(offset1, to: .int), contrib1)
                _ = b.memoryWrite(scratchCell, b.cast(offset2, to: .int), contrib2)
            }
        }
    }
}
