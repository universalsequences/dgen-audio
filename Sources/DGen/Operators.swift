import Foundation

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

// frontend
public enum LazyOp {
    case add, sub, div, mul, abs, sign, sin, cos, tan, tanh, exp, log, log10, sqrt, atan2, gt, gte,
        lte,
        lt, eq,
        gswitch, mix, pow, floor, ceil, round, mod, min, max, and, or, xor
    case mse  // mean squared error per-sample: (a-b)^2
    case spectralLossPass1(Int, CellID)  // Pass 1: compute loss & store DFT contributions
    case spectralLossPass2(Int, CellID)  // Pass 2: reduce contributions to gradients (no-op in forward)

    // FFT-based spectral loss with backprop support
    case spectralLossFFT(
        windowSize: Int,
        useHann: Bool,
        windowCell: CellID,
        fft1Cell: CellID,
        fft2Cell: CellID,
        mag1Cell: CellID,
        mag2Cell: CellID,
        scratchCell: CellID
    )
    case spectralLossFFTGradSpec(
        windowSize: Int,
        fft1Cell: CellID,
        fft2Cell: CellID,
        mag1Cell: CellID,
        mag2Cell: CellID,
        gradSpec1Cell: CellID,
        gradSpec2Cell: CellID
    )
    case spectralLossFFTGradIFFT(
        windowSize: Int,
        gradSpec1Cell: CellID,
        gradSpec2Cell: CellID,
        gradTime1Cell: CellID,
        gradTime2Cell: CellID,
        windowCell: CellID
    )
    // Inline gradient computation for spectralLossFFT - recomputes DFT to avoid race conditions
    case spectralLossFFTGradInline(
        windowSize: Int,
        useHann: Bool,
        windowCell: CellID,
        gradTime1Cell: CellID,
        gradTime2Cell: CellID
    )
    // Read gradient from frame-indexed storage (returns grad1)
    case spectralLossFFTGradRead(
        windowSize: Int,
        gradTime1Cell: CellID,
        gradTime2Cell: CellID
    )
    // Read second gradient from frame-indexed storage (returns grad2)
    case spectralLossFFTGradRead2(
        windowSize: Int,
        gradTime2Cell: CellID
    )
    // selectRow: extract a single row from a 2D tensor using dynamic index
    // Input: [tensor2D, rowIndex] where rowIndex is floored to int
    // Output: 1D tensor [numCols]
    case selectRow
    // peekRowInline: interpolated row extraction with frame-local computation
    // Input: [tensor2D, rowIndex] where rowIndex is interpolated between floor and ceil
    // Output: 1D tensor [numCols] - uses frame-indexed storage for SIMD safety
    // scratchCell stores frame-indexed outputs: frame * numCols + col
    case peekRowInline(scratchCell: CellID, numRows: Int, numCols: Int)
    // peekRowGradWrite: write gradients for both floor and ceil rows to frame-indexed storage
    // Input: [gradOutput (1D tensor), rowIndex]
    case peekRowGradWrite(
        floorGradCell: CellID, ceilGradCell: CellID, rowIdxCell: CellID, fracCell: CellID,
        numRows: Int, numCols: Int)
    // peekRowGradReduce: sum gradient contributions from all frames for each tensor position
    case peekRowGradReduce(
        floorGradCell: CellID, ceilGradCell: CellID, rowIdxCell: CellID, fracCell: CellID,
        gradCell: CellID, numRows: Int, numCols: Int, maxFrameCount: Int)
    // selectRowGradWrite: write gradient to frame-indexed storage (deterministic, no atomics)
    // Input: [gradOutput (1D tensor), rowIndex]
    // Writes to gradWriteCell[frame * numCols + col] and rowIdxCell[frame]
    case selectRowGradWrite(gradWriteCell: CellID, rowIdxCell: CellID, numRows: Int, numCols: Int)
    // selectRowGradReduce: sum contributions from all frames for each tensor position
    // Input: [gradWritePass] (for ordering)
    // Reads from frame-indexed storage and accumulates to gradCell
    case selectRowGradReduce(
        gradWriteCell: CellID, rowIdxCell: CellID, gradCell: CellID, numRows: Int, numCols: Int,
        maxFrameCount: Int)
    case parallelMap2DTestPass1(Int, CellID)  // Pass 1: write per-bin values to scratch
    case parallelMap2DTestPass2(Int, CellID)  // Pass 2: reduce per-bin values to scalar
    case selector  // selector(mode, options[])
    case memoryRead(CellID)
    case memoryWrite(CellID)
    case memoryAccumulate(CellID)  // Atomic add to memory cell
    case memoryCellSum(CellID, Int)  // Sum all elements in a memory cell (cell, size)
    case tensorAccumulate(CellID)  // Atomic add tensor elements to memory region
    case historyWrite(CellID)
    case historyReadWrite(CellID)
    case param(CellID)
    case latch(CellID)
    case click(CellID)
    case historyRead(CellID)
    case phasor(CellID)
    case deterministicPhasor  // Stateless phasor for constant frequency - parallelizable
    case accum(CellID)
    case noise(CellID)
    case constant(Float)
    case output(Int)
    case input(Int)
    case tensorRef(TensorID)
    case seq  // Sequential execution - returns value of last input

    // Tensor operations (historyRead/historyWrite handle tensors automatically based on cell size)
    case conv1d(Int)  // 1D convolution, Int is kernel size
    case conv2d(Shape)  // 2D convolution, Shape is kernel shape [kH, kW]
    case sum  // Reduce tensor to scalar by summing all elements
    case sumAxis(Int)  // Reduce along a specific axis
    case reshape(Shape)  // Reshape tensor (metadata only, no data movement)
    case transpose([Int])  // Transpose/permute axes (metadata only)
    case shrink([(Int, Int)?])  // Shrink/slice tensor (metadata only, no data movement)
    case pad([(Int, Int)])  // Pad tensor with zeros (virtual view, conditional reads)
    case peek  // Read from 2D tensor at (index, channel) with interpolation - lazy version
    case fft(Int, Int, CellID, CellID, CellID, CellID)  // FFT transform: windowSize, hopSize, scratchCell, ringBufferCell, writePosCell, counterCell
    case ifft(Int, Int, CellID, CellID, CellID, CellID)  // IFFT transform: windowSize, hopSize, scratchCell, outputRingCell, readPosCell, counterCell

    // Gradient-specific operations (used by Gradients.swift)
    case neg  // Unary negation: -x
    case expand(Shape)  // Broadcast scalar to tensor shape (sum backward)
    case expandAxis(Shape, Int)  // Broadcast along a specific axis (sumAxis backward)
    case gradPhasor(NodeID)  // Gradient for phasor: needs frame index context
    case gradDeterministicPhasor  // Gradient for deterministic phasor

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
                let fullInputs: [Lazy?] = node.inputs.map { ctx.values[$0] }
                print(
                    "mul failing \(node.id) nilIndex=\(fullInputs.firstIndex {$0 == nil}) fullInputs=\(fullInputs) node.inputs=\(node.inputs)"
                )
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

        case .neg:
            guard inputs.count == 1 else {
                throw DGenError.insufficientInputs(
                    operator: "neg", expected: 1, actual: inputs.count)
            }
            try emitUnaryOp(b: b, g: g, node: node, inputs: inputs) { b.neg($0) }
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
        case .memoryAccumulate(let cellId):
            guard inputs.count == 2 else {
                throw DGenError.insufficientInputs(
                    operator: "memoryAccumulate", expected: 2, actual: inputs.count)
            }
            b.use(val: b.memoryAccumulate(cellId, b.value(inputs[0]), b.value(inputs[1])))
        case .memoryCellSum(let cellId, let size):
            // Sum all elements in a memory cell
            let acc = b.float(0.0)
            b.loop(size) { i in
                let val = b.memoryRead(cellId, b.cast(i, to: .int))
                acc.accumulate(val)
            }
            b.use(val: acc.value)
        case .tensorAccumulate(let cellId):
            // Input is a tensor node - atomically add each element to cell
            guard node.inputs.count == 1 else {
                throw DGenError.insufficientInputs(
                    operator: "tensorAccumulate", expected: 1, actual: node.inputs.count)
            }

            let tensorInput = node.inputs[0]
            guard let inputNode = g.nodes[tensorInput],
                case .tensor(let shape) = inputNode.shape,
                let tensorId = g.nodeToTensor[tensorInput],
                let tensor = g.tensors[tensorId]
            else {
                throw DGenError.tensorError(op: "tensorAccumulate", reason: "requires tensor input")
            }

            let size = shape.reduce(1, *)

            // Check if input tensor is frame-aware (has per-frame storage)
            if ctx.frameAwareTensorCells.contains(tensor.cellId),
               let (tensorSize, frameCount) = ctx.g.frameAwareCells[tensor.cellId]
            {
                // Frame-aware: sum across all frames into the accumulation cell
                // Each frame has its own copy at frameIdx * tensorSize + elemIdx
                let tensorSizeFloat = b.constant(Float(tensorSize))
                b.parallelRange(size) { elemIdx in
                    let elemIdxFloat = b.cast(elemIdx, to: .float)
                    b.loop(frameCount) { frameIdx in
                        let frameIdxFloat = b.cast(frameIdx, to: .float)
                        let readPos = frameIdxFloat * tensorSizeFloat + elemIdxFloat
                        let val = b.memoryRead(tensor.cellId, b.cast(readPos, to: .int))
                        _ = b.memoryAccumulate(cellId, b.cast(elemIdx, to: .int), val)
                    }
                }
            } else {
                // Non-frame-aware: existing linear read
                b.parallelRange(size) { idx in
                    let val = b.memoryRead(tensor.cellId, b.cast(idx, to: .int))
                    _ = b.memoryAccumulate(cellId, b.cast(idx, to: .int), val)
                }
            }

            ctx.values[nodeId] = .empty
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

        case .spectralLossPass1(let windowSize, let scratchCell):
            guard inputs.count == 2 else {
                throw DGenError.insufficientInputs(
                    operator: "spectralLossPass1", expected: 2, actual: inputs.count)
            }
            let (sig1, sig2) = b.values(inputs, count: 2)
            // Forward: compute spectral loss normally (Pass1 does the actual work)
            b.use(
                val: u_spectralLoss(
                    sig1: sig1, sig2: sig2, windowSize: windowSize, scratchCell: scratchCell)(b))

        case .spectralLossPass2(let windowSize, let scratchCell):
            guard inputs.count == 1 else {
                throw DGenError.insufficientInputs(
                    operator: "spectralLossPass2", expected: 1, actual: inputs.count)
            }
            // Forward: reduce per-bin errors written by Pass1
            let _ = b.value(inputs[0])
            let numBins = windowSize / 2 + 1
            let winSize = b.constant(Float(windowSize))
            let frameIdx = b.threadIndex()
            let baseOffset = frameIdx * winSize * b.constant(2.0)
            let totalError = b.float(0.0)

            b.loop(numBins) { binIndex in
                let binIndexFloat = b.cast(binIndex, to: .float)
                let offset = baseOffset + binIndexFloat
                let binError = b.memoryRead(scratchCell, b.cast(offset, to: .int))
                totalError.accumulate(binError)
            }

            b.use(val: totalError.value)

        case .spectralLossFFT(
            let windowSize, let useHann, let windowCell,
            let fft1Cell, let fft2Cell, let mag1Cell, let mag2Cell, let scratchCell):
            // FFT-based spectral loss: forward pass
            // 1. Apply optional Hann window
            // 2. Compute FFT of both signals via Cooley-Tukey
            // 3. Compute magnitudes
            // 4. Sum squared differences as loss
            guard inputs.count == 2 else {
                throw DGenError.insufficientInputs(
                    operator: "spectralLossFFT", expected: 2, actual: inputs.count)
            }

            b.markRequiresScalar()

            let numBins = windowSize / 2 + 1
            let numStages = Int(log2(Double(windowSize)))
            let imagOffset = windowSize  // Scratch layout: real[0..<N], imag[N..<2N]

            let sig1 = b.value(inputs[0])
            let sig2 = b.value(inputs[1])
            let winSizeFloat = b.constant(Float(windowSize))
            let zero = b.constant(0.0)
            let one = b.constant(1.0)
            let frameIdx = b.threadIndex()

            // Use b.if_ to force scalar mode for the entire computation block
            // This prevents SIMD variable generation issues in nested loops
            let alwaysTrue = one > zero
            let loss = b.float(0.0)
            b.if_(alwaysTrue) {
                // 1. Generate Hann window coefficients (parallelizable - each index is independent)
                if useHann {
                    b.parallelRange(windowSize) { n in
                        let nFloat = b.cast(n, to: .float)
                        // w[n] = 0.5 * (1 - cos(2*pi*n / (N-1)))
                        let angle =
                            b.constant(2.0) * b.pi * nFloat / b.constant(Float(windowSize - 1))
                        let w = b.constant(0.5) * (one - b.cos(angle))
                        _ = b.memoryWrite(windowCell, b.cast(n, to: .int), w)
                    }
                } else {
                    // Rectangular window (all ones)
                    b.parallelRange(windowSize) { n in
                        _ = b.memoryWrite(windowCell, b.cast(n, to: .int), one)
                    }
                }

                // 2. Load windowed samples from tape into FFT scratch cells (parallelizable)
                b.parallelRange(windowSize) { n in
                    let nFloat = b.cast(n, to: .float)
                    let w = b.memoryRead(windowCell, b.cast(n, to: .int))

                    // Load from tape: samples from frameIdx - windowSize + 1 + n to frameIdx
                    let j = frameIdx - (winSizeFloat - one) + nFloat

                    let s1 = b.tapeLoad(sig1, at: j)
                    let s2 = b.tapeLoad(sig2, at: j)

                    // Apply window and store as real part; imag = 0
                    _ = b.memoryWrite(fft1Cell, b.cast(n, to: .int), s1 * w)
                    _ = b.memoryWrite(
                        fft1Cell, b.cast(n, to: .int) + b.constant(Float(imagOffset)), zero)
                    _ = b.memoryWrite(fft2Cell, b.cast(n, to: .int), s2 * w)
                    _ = b.memoryWrite(
                        fft2Cell, b.cast(n, to: .int) + b.constant(Float(imagOffset)), zero)
                }

                // 3. In-place FFT via Cooley-Tukey (bit-reversal + butterfly stages)
                // Helper function to emit FFT for a single cell
                func emitFFTInPlace(_ fftCell: CellID) {
                    // Bit-reversal permutation
                    b.loop(windowSize) { i in
                        var rev = b.constant(0.0)
                        var n = b.cast(i, to: .float)
                        for _ in 0..<numStages {
                            rev = rev * b.constant(2.0) + (n % b.constant(2.0))
                            n = b.floor(n / b.constant(2.0))
                        }

                        let iFloat = b.cast(i, to: .float)
                        let shouldSwap = iFloat < rev
                        let iInt = b.cast(i, to: .int)
                        let revInt = b.cast(rev, to: .int)

                        let tempRealI = b.memoryRead(fftCell, iInt)
                        let tempImagI = b.memoryRead(fftCell, iInt + b.constant(Float(imagOffset)))
                        let tempRealRev = b.memoryRead(fftCell, revInt)
                        let tempImagRev = b.memoryRead(
                            fftCell, revInt + b.constant(Float(imagOffset)))

                        let newRealI = b.gswitch(shouldSwap, tempRealRev, tempRealI)
                        let newImagI = b.gswitch(shouldSwap, tempImagRev, tempImagI)
                        let newRealRev = b.gswitch(shouldSwap, tempRealI, tempRealRev)
                        let newImagRev = b.gswitch(shouldSwap, tempImagI, tempImagRev)

                        _ = b.memoryWrite(fftCell, iInt, newRealI)
                        _ = b.memoryWrite(fftCell, iInt + b.constant(Float(imagOffset)), newImagI)
                        _ = b.memoryWrite(fftCell, revInt, newRealRev)
                        _ = b.memoryWrite(
                            fftCell, revInt + b.constant(Float(imagOffset)), newImagRev)
                    }

                    // Butterfly stages - each stage must complete before the next
                    // But within each stage, all butterflies are independent and parallelizable
                    for stage in 0..<numStages {
                        let butterflySize = 1 << (stage + 1)
                        let halfSize = butterflySize / 2
                        let numGroups = windowSize / butterflySize
                        let numButterflies = numGroups * halfSize  // Total butterflies in this stage

                        // Parallelize all butterflies in this stage
                        b.parallelRange(numButterflies) { flatIdx in
                            // Compute group and k from flat index
                            let flatFloat = b.cast(flatIdx, to: .float)
                            let halfSizeFloat = b.constant(Float(halfSize))
                            let butterflySizeFloat = b.constant(Float(butterflySize))

                            let group = b.floor(flatFloat / halfSizeFloat)
                            let k = flatFloat - (group * halfSizeFloat)

                            let i = group * butterflySizeFloat + k
                            let j = i + halfSizeFloat

                            // Twiddle factor: W = e^(-2*pi*i*k/butterflySize)
                            let angle = b.constant(-2.0) * b.pi * k / butterflySizeFloat
                            let wr = b.cos(angle)
                            let wi = b.sin(angle)

                            let iInt = b.cast(i, to: .int)
                            let jInt = b.cast(j, to: .int)

                            let ar = b.memoryRead(fftCell, iInt)
                            let ai = b.memoryRead(fftCell, iInt + b.constant(Float(imagOffset)))
                            let br = b.memoryRead(fftCell, jInt)
                            let bi = b.memoryRead(fftCell, jInt + b.constant(Float(imagOffset)))

                            // Butterfly: (ar,ai) + W*(br,bi) and (ar,ai) - W*(br,bi)
                            let tr = wr * br - wi * bi
                            let ti = wr * bi + wi * br

                            _ = b.memoryWrite(fftCell, iInt, ar + tr)
                            _ = b.memoryWrite(
                                fftCell, iInt + b.constant(Float(imagOffset)), ai + ti)
                            _ = b.memoryWrite(fftCell, jInt, ar - tr)
                            _ = b.memoryWrite(
                                fftCell, jInt + b.constant(Float(imagOffset)), ai - ti)
                        }
                    }
                }

                // Apply FFT to both cells
                emitFFTInPlace(fft1Cell)
                emitFFTInPlace(fft2Cell)

                // 4. Compute magnitudes in parallel and store for backward pass
                // This loop is fully parallelizable - each bin is independent
                b.parallelRange(numBins) { k in
                    let kInt = b.cast(k, to: .int)

                    let real1 = b.memoryRead(fft1Cell, kInt)
                    let imag1 = b.memoryRead(fft1Cell, kInt + b.constant(Float(imagOffset)))
                    let mag1 = b.sqrt(real1 * real1 + imag1 * imag1)
                    _ = b.memoryWrite(mag1Cell, kInt, mag1)

                    let real2 = b.memoryRead(fft2Cell, kInt)
                    let imag2 = b.memoryRead(fft2Cell, kInt + b.constant(Float(imagOffset)))
                    let mag2 = b.sqrt(real2 * real2 + imag2 * imag2)
                    _ = b.memoryWrite(mag2Cell, kInt, mag2)

                    // Store squared difference in scratch for parallel reduction
                    let diff = mag1 - mag2
                    _ = b.memoryWrite(scratchCell, kInt, diff * diff)
                }

                // 5. Sequential reduction of loss (could be improved with parallel reduction)
                b.loop(numBins) { k in
                    let kInt = b.cast(k, to: .int)
                    let diffSq = b.memoryRead(scratchCell, kInt)
                    loss.accumulate(diffSq)
                }
            }

            b.use(val: loss.value)

        case .spectralLossFFTGradSpec(
            let windowSize, let fft1Cell, let fft2Cell,
            let mag1Cell, let mag2Cell, let gradSpec1Cell, let gradSpec2Cell):
            // Compute gradient w.r.t. complex spectrum
            // ∂L/∂X.real = ∂L/∂mag * (real / mag)
            // ∂L/∂X.imag = ∂L/∂mag * (imag / mag)
            guard inputs.count == 1 else {
                throw DGenError.insufficientInputs(
                    operator: "spectralLossFFTGradSpec", expected: 1, actual: inputs.count)
            }

            b.markRequiresScalar()

            let numBins = windowSize / 2 + 1
            let imagOffset = windowSize
            let gradOutput = b.value(inputs[0])
            let eps = b.constant(1e-8)

            // Compute gradient spectrum in parallel (each bin is independent)
            b.parallelRange(numBins) { k in
                let kInt = b.cast(k, to: .int)

                // Read stored values
                let mag1 = b.memoryRead(mag1Cell, kInt)
                let mag2 = b.memoryRead(mag2Cell, kInt)
                let real1 = b.memoryRead(fft1Cell, kInt)
                let imag1 = b.memoryRead(fft1Cell, kInt + b.constant(Float(imagOffset)))
                let real2 = b.memoryRead(fft2Cell, kInt)
                let imag2 = b.memoryRead(fft2Cell, kInt + b.constant(Float(imagOffset)))

                // ∂L/∂mag = 2 * (mag1 - mag2) * gradOutput
                let gradMag1 = b.constant(2.0) * (mag1 - mag2) * gradOutput
                let gradMag2 = b.constant(-2.0) * (mag1 - mag2) * gradOutput

                // Handle division by zero with epsilon
                let safeMag1 = b.max(mag1, eps)
                let safeMag2 = b.max(mag2, eps)

                // ∂L/∂X = gradMag * (X / |X|) = gradMag * (real/mag, imag/mag)
                let gradReal1 = gradMag1 * real1 / safeMag1
                let gradImag1 = gradMag1 * imag1 / safeMag1
                let gradReal2 = gradMag2 * real2 / safeMag2
                let gradImag2 = gradMag2 * imag2 / safeMag2

                // Store gradient spectrum
                _ = b.memoryWrite(gradSpec1Cell, kInt, gradReal1)
                _ = b.memoryWrite(gradSpec1Cell, kInt + b.constant(Float(imagOffset)), gradImag1)
                _ = b.memoryWrite(gradSpec2Cell, kInt, gradReal2)
                _ = b.memoryWrite(gradSpec2Cell, kInt + b.constant(Float(imagOffset)), gradImag2)
            }

            // Fill in conjugate symmetric part for IFFT (bins numBins to windowSize-1)
            // X[N-k] = conj(X[k]) for k = 1 to N/2-1
            // This is parallelizable since each k writes to a unique index
            b.parallelRange(windowSize / 2 - 1) { k in
                let kPlusOne = b.cast(k, to: .float) + b.constant(1.0)
                let kIdx = b.cast(kPlusOne, to: .int)
                let conjIdx = b.constant(Float(windowSize)) - kPlusOne

                // Signal 1
                let real1 = b.memoryRead(gradSpec1Cell, kIdx)
                let imag1 = b.memoryRead(gradSpec1Cell, kIdx + b.constant(Float(imagOffset)))
                _ = b.memoryWrite(gradSpec1Cell, b.cast(conjIdx, to: .int), real1)
                _ = b.memoryWrite(
                    gradSpec1Cell, b.cast(conjIdx, to: .int) + b.constant(Float(imagOffset)),
                    b.constant(0.0) - imag1)

                // Signal 2
                let real2 = b.memoryRead(gradSpec2Cell, kIdx)
                let imag2 = b.memoryRead(gradSpec2Cell, kIdx + b.constant(Float(imagOffset)))
                _ = b.memoryWrite(gradSpec2Cell, b.cast(conjIdx, to: .int), real2)
                _ = b.memoryWrite(
                    gradSpec2Cell, b.cast(conjIdx, to: .int) + b.constant(Float(imagOffset)),
                    b.constant(0.0) - imag2)
            }

            b.use(val: b.constant(0.0))  // Side-effect only

        case .spectralLossFFTGradIFFT(
            let windowSize, let gradSpec1Cell, let gradSpec2Cell,
            let gradTime1Cell, let gradTime2Cell, let windowCell):
            // IFFT to scatter frequency-domain gradients back to time domain
            // Then multiply by window coefficients for Hann backprop
            guard inputs.count == 1 else {
                throw DGenError.insufficientInputs(
                    operator: "spectralLossFFTGradIFFT", expected: 1, actual: inputs.count)
            }

            b.markRequiresScalar()

            let numStages = Int(log2(Double(windowSize)))
            let imagOffset = windowSize
            let invN = b.constant(1.0 / Float(windowSize))

            // Helper function to emit IFFT for a single gradient spectrum cell
            func emitIFFTInPlace(_ gradSpecCell: CellID, _ gradTimeCell: CellID) {
                // Bit-reversal permutation (has data dependencies - keep sequential)
                b.loop(windowSize) { i in
                    var rev = b.constant(0.0)
                    var n = b.cast(i, to: .float)
                    for _ in 0..<numStages {
                        rev = rev * b.constant(2.0) + (n % b.constant(2.0))
                        n = b.floor(n / b.constant(2.0))
                    }

                    let iFloat = b.cast(i, to: .float)
                    let shouldSwap = iFloat < rev
                    let iInt = b.cast(i, to: .int)
                    let revInt = b.cast(rev, to: .int)

                    let tempR = b.memoryRead(gradSpecCell, iInt)
                    let tempI = b.memoryRead(gradSpecCell, iInt + b.constant(Float(imagOffset)))
                    let revR = b.memoryRead(gradSpecCell, revInt)
                    let revI = b.memoryRead(gradSpecCell, revInt + b.constant(Float(imagOffset)))

                    let newIR = b.gswitch(shouldSwap, revR, tempR)
                    let newII = b.gswitch(shouldSwap, revI, tempI)
                    let newRevR = b.gswitch(shouldSwap, tempR, revR)
                    let newRevI = b.gswitch(shouldSwap, tempI, revI)

                    _ = b.memoryWrite(gradSpecCell, iInt, newIR)
                    _ = b.memoryWrite(gradSpecCell, iInt + b.constant(Float(imagOffset)), newII)
                    _ = b.memoryWrite(gradSpecCell, revInt, newRevR)
                    _ = b.memoryWrite(gradSpecCell, revInt + b.constant(Float(imagOffset)), newRevI)
                }

                // Butterfly stages with POSITIVE twiddle angles (IFFT)
                // Stages are sequential but butterflies within each stage are parallel
                var butterflySize = 2
                for _ in 0..<numStages {
                    let halfSize = butterflySize / 2
                    let numGroups = windowSize / butterflySize
                    let numButterflies = numGroups * halfSize

                    b.parallelRange(numButterflies) { flatIdx in
                        let flatFloat = b.cast(flatIdx, to: .float)
                        let halfSizeFloat = b.constant(Float(halfSize))
                        let butterflySizeFloat = b.constant(Float(butterflySize))

                        let group = b.floor(flatFloat / halfSizeFloat)
                        let k = flatFloat - (group * halfSizeFloat)

                        let i = group * butterflySizeFloat + k
                        let j = i + halfSizeFloat

                        // IFFT twiddle: W = e^(+2πi*k/butterflySize) - POSITIVE angle
                        let angle = b.constant(2.0) * b.pi * k / butterflySizeFloat
                        let wr = b.cos(angle)
                        let wi = b.sin(angle)

                        let iInt = b.cast(i, to: .int)
                        let jInt = b.cast(j, to: .int)

                        let ar = b.memoryRead(gradSpecCell, iInt)
                        let ai = b.memoryRead(gradSpecCell, iInt + b.constant(Float(imagOffset)))
                        let br = b.memoryRead(gradSpecCell, jInt)
                        let bi = b.memoryRead(gradSpecCell, jInt + b.constant(Float(imagOffset)))

                        // Complex multiply and butterfly
                        let tr = wr * br - wi * bi
                        let ti = wr * bi + wi * br

                        _ = b.memoryWrite(gradSpecCell, iInt, ar + tr)
                        _ = b.memoryWrite(
                            gradSpecCell, iInt + b.constant(Float(imagOffset)), ai + ti)
                        _ = b.memoryWrite(gradSpecCell, jInt, ar - tr)
                        _ = b.memoryWrite(
                            gradSpecCell, jInt + b.constant(Float(imagOffset)), ai - ti)
                    }
                    butterflySize *= 2
                }

                // Scale by 1/N and multiply by window (Hann backprop), write to time-domain gradient cell
                // This is fully parallelizable - each index is independent
                b.parallelRange(windowSize) { n in
                    let nInt = b.cast(n, to: .int)
                    let realVal = b.memoryRead(gradSpecCell, nInt) * invN
                    let w = b.memoryRead(windowCell, nInt)
                    _ = b.memoryWrite(gradTimeCell, nInt, realVal * w)
                }
            }

            // Apply IFFT to both gradient cells
            emitIFFTInPlace(gradSpec1Cell, gradTime1Cell)
            emitIFFTInPlace(gradSpec2Cell, gradTime2Cell)

            b.use(val: b.constant(0.0))  // Side-effect only

        case .spectralLossFFTGradInline(
            let windowSize, let useHann, let windowCell,
            let gradTime1Cell, let gradTime2Cell):
            // Inline gradient computation that recomputes DFT to avoid race conditions
            // Uses frame-indexed storage to prevent race conditions between parallel frames
            // Inputs: [gradOutput, sig1, sig2]
            guard inputs.count == 3 else {
                throw DGenError.insufficientInputs(
                    operator: "spectralLossFFTGradInline", expected: 3, actual: inputs.count)
            }

            b.markRequiresScalar()

            let numBins = windowSize / 2 + 1
            let gradOutput = b.value(inputs[0])
            let sig1 = b.value(inputs[1])
            let sig2 = b.value(inputs[2])
            let winSizeFloat = b.constant(Float(windowSize))
            let zero = b.constant(0.0)
            let one = b.constant(1.0)
            let eps = b.constant(1e-8)
            let frameIdx = b.threadIndex()

            // Frame-indexed base offset for this frame's gradient storage
            let frameBase = frameIdx * winSizeFloat

            // Use if_ to force scalar mode
            let alwaysTrue = one > zero
            b.if_(alwaysTrue) {
                // 1. Generate Hann window coefficients (same as forward)
                if useHann {
                    b.parallelRange(windowSize) { n in
                        let nFloat = b.cast(n, to: .float)
                        let angle =
                            b.constant(2.0) * b.pi * nFloat / b.constant(Float(windowSize - 1))
                        let w = b.constant(0.5) * (one - b.cos(angle))
                        _ = b.memoryWrite(windowCell, b.cast(n, to: .int), w)
                    }
                } else {
                    b.parallelRange(windowSize) { n in
                        _ = b.memoryWrite(windowCell, b.cast(n, to: .int), one)
                    }
                }

                // 2. Zero the gradient cells at frame-indexed positions
                b.loop(windowSize) { n in
                    let idx = b.cast(frameBase + b.cast(n, to: .float), to: .int)
                    _ = b.memoryWrite(gradTime1Cell, idx, zero)
                    _ = b.memoryWrite(gradTime2Cell, idx, zero)
                }

                // 3. For each bin, recompute DFT and accumulate gradients to time domain
                // This is the key: we compute the DFT inline using tapeLoad, avoiding shared cells
                b.loop(numBins) { kIdx in
                    let k = b.cast(kIdx, to: .float)

                    // Recompute DFT for this bin
                    let real1 = b.float(0.0)
                    let imag1 = b.float(0.0)
                    let real2 = b.float(0.0)
                    let imag2 = b.float(0.0)

                    b.loop(windowSize) { n in
                        let nFloat = b.cast(n, to: .float)
                        let j = frameIdx - (winSizeFloat - one) + nFloat
                        let w = b.memoryRead(windowCell, b.cast(n, to: .int))

                        let s1 = b.tapeLoad(sig1, at: j) * w
                        let s2 = b.tapeLoad(sig2, at: j) * w

                        // DFT basis: e^(-2πi*k*n/N)
                        let angle = b.constant(-2.0) * b.pi * k * nFloat / winSizeFloat
                        let c = b.cos(angle)
                        let s = b.sin(angle)

                        real1.accumulate(s1 * c)
                        imag1.accumulate(s1 * s)
                        real2.accumulate(s2 * c)
                        imag2.accumulate(s2 * s)
                    }

                    // Compute magnitudes
                    let mag1 = b.sqrt(real1.value * real1.value + imag1.value * imag1.value)
                    let mag2 = b.sqrt(real2.value * real2.value + imag2.value * imag2.value)

                    // Gradient: ∂L/∂mag = 2 * (mag1 - mag2) * gradOutput
                    let gradMag1 = b.constant(2.0) * (mag1 - mag2) * gradOutput
                    let gradMag2 = b.constant(-2.0) * (mag1 - mag2) * gradOutput

                    // Handle division by zero
                    let safeMag1 = b.max(mag1, eps)
                    let safeMag2 = b.max(mag2, eps)

                    // ∂L/∂X = gradMag * (real/mag, imag/mag)
                    let gradReal1 = gradMag1 * real1.value / safeMag1
                    let gradImag1 = gradMag1 * imag1.value / safeMag1
                    let gradReal2 = gradMag2 * real2.value / safeMag2
                    let gradImag2 = gradMag2 * imag2.value / safeMag2

                    // Scatter gradient to time domain: ∂L/∂x[n] += gradReal * cos + gradImag * sin
                    // This is the transpose of the DFT (IDFT without normalization)
                    // Uses frame-indexed storage to avoid race conditions
                    b.loop(windowSize) { n in
                        let nFloat = b.cast(n, to: .float)
                        let angle = b.constant(-2.0) * b.pi * k * nFloat / winSizeFloat
                        let c = b.cos(angle)
                        let s = b.sin(angle)
                        let w = b.memoryRead(windowCell, b.cast(n, to: .int))

                        // Frame-indexed position
                        let idx = b.cast(frameBase + nFloat, to: .int)

                        // Accumulate gradient (window backprop included)
                        let grad1Contrib = (gradReal1 * c + gradImag1 * s) * w
                        let grad2Contrib = (gradReal2 * c + gradImag2 * s) * w
                        _ = b.memoryAccumulate(gradTime1Cell, idx, grad1Contrib)
                        _ = b.memoryAccumulate(gradTime2Cell, idx, grad2Contrib)
                    }
                }
            }

            b.use(val: zero)  // Side-effect only

        case .spectralLossFFTGradRead(let windowSize, let gradTime1Cell, _):
            // Read gradient for signal 1 from frame-indexed storage
            // Sample at position p appears in windows at frames p, p+1, ..., p+windowSize-1
            // We must sum contributions from all these windows
            guard inputs.count == 1 else {
                throw DGenError.insufficientInputs(
                    operator: "spectralLossFFTGradRead", expected: 1, actual: inputs.count)
            }
            // Force dependency on gradPass by reading its value (should be 0, just for ordering)
            let _ = b.value(inputs[0])

            let frameIdx = b.threadIndex()
            let winSizeFloat = b.constant(Float(windowSize))
            let p = frameIdx  // absolute sample position

            // Sum contributions from all windows that contain sample p
            // Window at frame w contains sample p at offset (p - w + windowSize - 1)
            // Gradient stored at w * windowSize + (p - w + windowSize - 1)
            let gradSum = b.float(0.0)
            b.loop(windowSize) { i in
                let iFloat = b.cast(i, to: .float)
                let w = p + iFloat  // window frame index
                let offset = winSizeFloat - b.constant(1.0) - iFloat  // offset in that window
                let idx = w * winSizeFloat + offset
                let contrib = b.memoryRead(gradTime1Cell, b.cast(idx, to: .int))
                gradSum.accumulate(contrib)
            }
            b.use(val: gradSum.value)

        case .spectralLossFFTGradRead2(let windowSize, let gradTime2Cell):
            // Read gradient for signal 2 from frame-indexed storage
            // Sample at position p appears in windows at frames p, p+1, ..., p+windowSize-1
            // We must sum contributions from all these windows
            guard inputs.count == 1 else {
                throw DGenError.insufficientInputs(
                    operator: "spectralLossFFTGradRead2", expected: 1, actual: inputs.count)
            }
            // Force dependency on gradPass by reading its value (should be 0, just for ordering)
            let _ = b.value(inputs[0])

            let frameIdx = b.threadIndex()
            let winSizeFloat = b.constant(Float(windowSize))
            let p = frameIdx  // absolute sample position

            // Sum contributions from all windows that contain sample p
            // Window at frame w contains sample p at offset (p - w + windowSize - 1)
            // Gradient stored at w * windowSize + (p - w + windowSize - 1)
            let gradSum = b.float(0.0)
            b.loop(windowSize) { i in
                let iFloat = b.cast(i, to: .float)
                let w = p + iFloat  // window frame index
                let offset = winSizeFloat - b.constant(1.0) - iFloat  // offset in that window
                let idx = w * winSizeFloat + offset
                let contrib = b.memoryRead(gradTime2Cell, b.cast(idx, to: .int))
                gradSum.accumulate(contrib)
            }
            b.use(val: gradSum.value)

        case .selectRow:
            // selectRow: extract a single row from a 2D tensor using dynamic index
            // Inputs: [tensor2D, rowIndex]
            // Output: 1D tensor [numCols]
            guard node.inputs.count == 2 else {
                throw DGenError.insufficientInputs(
                    operator: "selectRow", expected: 2, actual: node.inputs.count)
            }

            let tensorInput = node.inputs[0]

            // Get tensor shape from the input node
            guard let inputNode = g.nodes[tensorInput],
                case .tensor(let shape) = inputNode.shape,
                shape.count == 2
            else {
                throw DGenError.tensorError(op: "selectRow", reason: "requires 2D tensor input")
            }

            let numRows = shape[0]
            let numCols = shape[1]

            // Get input and output tensors
            guard let inTensorId = g.nodeToTensor[tensorInput],
                let inTensor = g.tensors[inTensorId],
                let outTensorId = g.nodeToTensor[node.id],
                let outTensor = g.tensors[outTensorId]
            else {
                throw DGenError.tensorError(op: "selectRow", reason: "missing tensor")
            }

            // Read rowIndex input and floor it
            let rowIndex = try b.readInput(node, inputs, at: 1)
            let numRowsFloat = b.constant(Float(numRows))
            let zero = b.constant(0.0)

            // Wrap rowIndex using modulo for wrapping behavior, then floor
            let wrappedIndex = b.mod(rowIndex, numRowsFloat)
            let isNegative = wrappedIndex < zero
            let positiveIndex = b.gswitch(isNegative, wrappedIndex + numRowsFloat, wrappedIndex)
            let floorIndex = b.floor(positiveIndex)

            // Read the selected row and write to output
            // Column-major layout: offset = col * numRows + row
            b.parallelRange(numCols) { colIdx in
                let colIdxFloat = b.cast(colIdx, to: .float)
                let readPos = colIdxFloat * numRowsFloat + floorIndex
                let value = b.memoryRead(inTensor.cellId, b.cast(readPos, to: .int))
                _ = b.memoryWrite(outTensor.cellId, b.cast(colIdx, to: .int), value)
            }

            ctx.values[nodeId] = .empty

        case .peekRowInline(let scratchCell, let numRows, let numCols):
            // Interpolated row extraction with frame-indexed storage for SIMD safety
            // Inputs: [tensor2D, rowIndex]
            // Output: 1D tensor [numCols] stored at scratchCell[frame * numCols + col]
            guard node.inputs.count == 2 else {
                throw DGenError.insufficientInputs(
                    operator: "peekRowInline", expected: 2, actual: node.inputs.count)
            }

            let tensorInput = node.inputs[0]

            // Get input tensor
            guard let inTensorId = g.nodeToTensor[tensorInput],
                let inTensor = g.tensors[inTensorId],
                let outTensorId = g.nodeToTensor[node.id],
                let outTensor = g.tensors[outTensorId]
            else {
                throw DGenError.tensorError(op: "peekRowInline", reason: "missing tensor")
            }

            // Read rowIndex input
            let rowIndex = try b.readInput(node, inputs, at: 1)
            let numRowsFloat = b.constant(Float(numRows))
            let numColsFloat = b.constant(Float(numCols))
            let zero = b.constant(0.0)
            let one = b.constant(1.0)

            // Wrap rowIndex using modulo for wrapping behavior
            let wrappedIndex = b.mod(rowIndex, numRowsFloat)
            let isNegative = wrappedIndex < zero
            let positiveIndex = b.gswitch(isNegative, wrappedIndex + numRowsFloat, wrappedIndex)

            // Compute floor and ceil indices
            let floorIndex = b.floor(positiveIndex)
            let ceilIndex = floorIndex + one
            let ceilWrapped = b.gswitch(ceilIndex >= numRowsFloat, zero, ceilIndex)

            // Compute frac for interpolation
            let frac = positiveIndex - floorIndex
            let oneMinusFrac = one - frac

            // Get frame index for frame-indexed output storage
            // Use currentFrameIndex() which returns the correct frame index in both normal
            // and frame-aware tensor blocks (where threadIndex() would return the flat index)
            let frameIdx = b.currentFrameIndex()
            let frameBase = frameIdx * numColsFloat

            // Compute interpolated values and write to frame-indexed positions
            // Column-major layout for input: offset = col * numRows + row
            b.parallelRange(numCols) { colIdx in
                let colIdxFloat = b.cast(colIdx, to: .float)
                // Read floor row value
                let floorPos = colIdxFloat * numRowsFloat + floorIndex
                let floorValue = b.memoryRead(inTensor.cellId, b.cast(floorPos, to: .int))
                // Read ceil row value
                let ceilPos = colIdxFloat * numRowsFloat + ceilWrapped
                let ceilValue = b.memoryRead(inTensor.cellId, b.cast(ceilPos, to: .int))
                // Interpolate: (1 - frac) * floor + frac * ceil
                let interpolated = oneMinusFrac * floorValue + frac * ceilValue
                // Write to frame-indexed position in scratch cell
                let writePos = frameBase + colIdxFloat
                _ = b.memoryWrite(scratchCell, b.cast(writePos, to: .int), interpolated)
                // Also write to output tensor
                // If the output cell is frame-aware, use frame-indexed addressing
                // This matches how tensorRead will read from it
                if ctx.frameAwareTensorCells.contains(outTensor.cellId) {
                    _ = b.memoryWrite(outTensor.cellId, b.cast(writePos, to: .int), interpolated)
                } else {
                    // Legacy mode: non-frame-indexed (will be overwritten each frame)
                    _ = b.memoryWrite(outTensor.cellId, b.cast(colIdx, to: .int), interpolated)
                }
            }

            ctx.values[nodeId] = .empty

        case .selectRowGradWrite(let gradWriteCell, let rowIdxCell, let numRows, let numCols):
            // Write gradient to frame-indexed storage (deterministic, no atomics)
            // Inputs: [gradOutput (1D tensor), rowIndex]
            // Writes to gradWriteCell[frame * numCols + col] and rowIdxCell[frame]
            guard node.inputs.count == 2 else {
                throw DGenError.insufficientInputs(
                    operator: "selectRowGradWrite", expected: 2, actual: node.inputs.count)
            }

            let gradTensorInput = node.inputs[0]

            // Get the gradient tensor's cell
            guard let gradTensorId = g.nodeToTensor[gradTensorInput],
                let gradTensor = g.tensors[gradTensorId]
            else {
                throw DGenError.tensorError(
                    op: "selectRowGradWrite", reason: "missing gradient tensor")
            }

            // Read rowIndex input and floor it
            let rowIndex = try b.readInput(node, inputs, at: 1)
            let numRowsFloat = b.constant(Float(numRows))
            let numColsFloat = b.constant(Float(numCols))
            let zero = b.constant(0.0)

            // Wrap rowIndex using modulo for wrapping behavior, then floor
            let wrappedIndex = b.mod(rowIndex, numRowsFloat)
            let isNegative = wrappedIndex < zero
            let positiveIndex = b.gswitch(isNegative, wrappedIndex + numRowsFloat, wrappedIndex)
            let floorIndex = b.floor(positiveIndex)

            // Get frame index for frame-indexed storage
            // Use currentFrameIndex for correct behavior in frame-aware tensor blocks
            let frameIdx = b.currentFrameIndex()

            // Write the floored row index for this frame
            _ = b.memoryWrite(rowIdxCell, b.cast(frameIdx, to: .int), floorIndex)

            // Write each gradient element to frame-indexed position
            // Layout: gradWriteCell[frame * numCols + col]
            let frameBase = frameIdx * numColsFloat
            b.parallelRange(numCols) { colIdx in
                let colIdxFloat = b.cast(colIdx, to: .float)
                // Read gradient element from gradOutput tensor
                let gradValue = b.memoryRead(gradTensor.cellId, b.cast(colIdx, to: .int))
                // Write to frame-indexed position
                let writePos = frameBase + colIdxFloat
                _ = b.memoryWrite(gradWriteCell, b.cast(writePos, to: .int), gradValue)
            }

            b.use(val: zero)  // Side-effect only

        case .selectRowGradReduce(
            let gradWriteCell, let rowIdxCell, let gradCell,
            let numRows, let numCols, let maxFrameCount):
            // Sum contributions from all frames for each tensor position
            // Input: [gradWritePass] (for ordering)
            // Reads from frame-indexed storage and accumulates to gradCell
            guard node.inputs.count == 1 else {
                throw DGenError.insufficientInputs(
                    operator: "selectRowGradReduce", expected: 1, actual: node.inputs.count)
            }

            // Force dependency on write pass
            let _ = b.value(inputs[0])

            let numColsFloat = b.constant(Float(numCols))
            let zero = b.constant(0.0)

            // For each position in the 2D tensor, sum contributions from all frames
            // that selected this row
            // Column-major layout: offset = col * numRows + row
            b.parallelRange(numRows) { rowIdx in
                let rowFloat = b.cast(rowIdx, to: .float)

                b.parallelRange(numCols) { colIdx in
                    let colFloat = b.cast(colIdx, to: .float)
                    let gradSum = b.float(0.0)

                    // Loop over all frames
                    b.loop(maxFrameCount) { frameIdx in
                        let frameFloat = b.cast(frameIdx, to: .float)
                        // Read which row this frame selected
                        let selectedRow = b.memoryRead(rowIdxCell, b.cast(frameIdx, to: .int))
                        // Check if this frame selected the current row
                        let isMatch = b.abs(selectedRow - rowFloat) < b.constant(0.5)
                        // Read gradient value from frame-indexed storage
                        let readPos = frameFloat * numColsFloat + colFloat
                        let gradValue = b.memoryRead(gradWriteCell, b.cast(readPos, to: .int))
                        // Conditionally accumulate
                        let contribution = b.gswitch(isMatch, gradValue, zero)
                        gradSum.accumulate(contribution)
                    }

                    // Write accumulated gradient to output cell
                    // Column-major layout: offset = col * numRows + row
                    let destPos = colFloat * b.constant(Float(numRows)) + rowFloat
                    _ = b.memoryAccumulate(gradCell, b.cast(destPos, to: .int), gradSum.value)
                }
            }

            b.use(val: zero)  // Side-effect only

        case .peekRowGradWrite(
            let floorGradCell, let ceilGradCell, let rowIdxCell, let fracCell,
            let numRows, let numCols):
            // Write gradients for both floor and ceil rows to frame-indexed storage
            // Inputs: [gradOutput (scalar or 1D tensor), rowIndex]
            // Note: gradOutput can be scalar (from sum reduction) - same value for all elements
            guard node.inputs.count == 2 else {
                throw DGenError.insufficientInputs(
                    operator: "peekRowGradWrite", expected: 2, actual: node.inputs.count)
            }

            let gradTensorInput = node.inputs[0]

            // Check if gradient input is a tensor or scalar
            let gradCellId: CellID?
            if let gradTensorId = g.nodeToTensor[gradTensorInput],
                let gradTensor = g.tensors[gradTensorId]
            {
                gradCellId = gradTensor.cellId
            } else {
                gradCellId = nil  // Scalar gradient - will use b.value()
            }

            // Read gradient as scalar (works for both scalar and will be used as broadcast value)
            let scalarGrad = b.value(inputs[0])

            // Read rowIndex input
            let rowIndex = try b.readInput(node, inputs, at: 1)
            let numRowsFloat = b.constant(Float(numRows))
            let numColsFloat = b.constant(Float(numCols))
            let zero = b.constant(0.0)
            let one = b.constant(1.0)

            // Wrap rowIndex using modulo
            let wrappedIndex = b.mod(rowIndex, numRowsFloat)
            let isNegative = wrappedIndex < zero
            let positiveIndex = b.gswitch(isNegative, wrappedIndex + numRowsFloat, wrappedIndex)

            // Compute floor and ceil indices
            let floorIndex = b.floor(positiveIndex)
            let ceilIndex = floorIndex + one
            let ceilWrapped = b.gswitch(ceilIndex >= numRowsFloat, zero, ceilIndex)

            // Compute frac
            let frac = positiveIndex - floorIndex

            // Get frame index - use frameIndex() which respects setFrameIndex
            let frameIdx = b.frameIndex()

            // Write row indices and frac for this frame
            _ = b.memoryWrite(rowIdxCell, b.cast(frameIdx, to: .int), floorIndex)
            _ = b.memoryWrite(fracCell, b.cast(frameIdx, to: .int), frac)
            // Write ceil index to a separate slot (frame + maxFrameCount)
            let ceilSlot = frameIdx + b.constant(4096.0)  // maxFrameCount offset
            _ = b.memoryWrite(rowIdxCell, b.cast(ceilSlot, to: .int), ceilWrapped)

            // Write weighted gradients for floor and ceil
            // floor gets grad * (1 - frac), ceil gets grad * frac
            let oneMinusFrac = one - frac
            let frameBase = frameIdx * numColsFloat

            b.parallelRange(numCols) { colIdx in
                let colIdxFloat = b.cast(colIdx, to: .float)
                // Get gradient value - from tensor cell if available, otherwise use scalar
                let gradValue: Expr
                if let cellId = gradCellId {
                    // Frame-aware tensor: read from frameIdx * numCols + colIdx
                    let readPos = frameBase + colIdxFloat
                    gradValue = b.memoryRead(cellId, b.cast(readPos, to: .int))
                } else {
                    gradValue = scalarGrad  // Broadcast scalar to all elements
                }
                let writePos = frameBase + colIdxFloat
                // Floor gradient: grad * (1 - frac)
                let floorGrad = gradValue * oneMinusFrac
                _ = b.memoryWrite(floorGradCell, b.cast(writePos, to: .int), floorGrad)
                // Ceil gradient: grad * frac
                let ceilGrad = gradValue * frac
                _ = b.memoryWrite(ceilGradCell, b.cast(writePos, to: .int), ceilGrad)
            }

            b.use(val: zero)

        case .peekRowGradReduce(
            let floorGradCell, let ceilGradCell, let rowIdxCell, _,
            let gradCell, let numRows, let numCols, let maxFrameCount):
            // Sum gradient contributions from all frames for each tensor position
            guard node.inputs.count == 1 else {
                throw DGenError.insufficientInputs(
                    operator: "peekRowGradReduce", expected: 1, actual: node.inputs.count)
            }

            // Force dependency on write pass
            let _ = b.value(inputs[0])

            let numColsFloat = b.constant(Float(numCols))
            let zero = b.constant(0.0)
            let maxFrameCountFloat = b.constant(Float(maxFrameCount))

            // For each position in the 2D tensor, sum contributions from all frames
            b.parallelRange(numRows) { rowIdx in
                let rowFloat = b.cast(rowIdx, to: .float)

                b.parallelRange(numCols) { colIdx in
                    let colFloat = b.cast(colIdx, to: .float)
                    let gradSum = b.float(0.0)

                    // Check both floor and ceil contributions from each frame
                    b.loop(maxFrameCount) { frameIdx in
                        let frameFloat = b.cast(frameIdx, to: .float)
                        // Check floor row match
                        let floorRow = b.memoryRead(rowIdxCell, b.cast(frameIdx, to: .int))
                        let isFloorMatch = b.abs(floorRow - rowFloat) < b.constant(0.5)
                        let readPos = frameFloat * numColsFloat + colFloat
                        let floorGrad = b.memoryRead(floorGradCell, b.cast(readPos, to: .int))
                        let floorContrib = b.gswitch(isFloorMatch, floorGrad, zero)
                        gradSum.accumulate(floorContrib)

                        // Check ceil row match (stored at frame + maxFrameCount)
                        let ceilSlot = frameFloat + maxFrameCountFloat
                        let ceilRow = b.memoryRead(rowIdxCell, b.cast(ceilSlot, to: .int))
                        let isCeilMatch = b.abs(ceilRow - rowFloat) < b.constant(0.5)
                        let ceilGrad = b.memoryRead(ceilGradCell, b.cast(readPos, to: .int))
                        let ceilContrib = b.gswitch(isCeilMatch, ceilGrad, zero)
                        gradSum.accumulate(ceilContrib)
                    }

                    // Write to gradient cell (column-major layout)
                    let destPos = colFloat * b.constant(Float(numRows)) + rowFloat
                    _ = b.memoryAccumulate(gradCell, b.cast(destPos, to: .int), gradSum.value)
                }
            }

            b.use(val: zero)

        case .parallelMap2DTestPass1(let bins, let scratchCell):
            guard inputs.isEmpty else {
                throw DGenError.insufficientInputs(
                    operator: "parallelMap2DTestPass1", expected: 0, actual: inputs.count)
            }
            u_parallelMap2DTestPass1(bins: bins, scratchCell: scratchCell)(b)
            b.use(val: b.constant(0.0))

        case .parallelMap2DTestPass2(let bins, let scratchCell):
            guard inputs.count == 1 else {
                throw DGenError.insufficientInputs(
                    operator: "parallelMap2DTestPass2", expected: 1, actual: inputs.count)
            }
            let _ = b.value(inputs[0])
            let binsFloat = b.constant(Float(bins))
            let frameIdx = b.threadIndex()
            let acc = b.float(0.0)
            b.loop(bins) { binIdx in
                let binIdxFloat = b.cast(binIdx, to: .float)
                let offset = frameIdx * binsFloat + binIdxFloat
                let val = b.memoryRead(scratchCell, b.cast(offset, to: .int))
                acc.accumulate(val)
            }
            b.use(val: acc.value)

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
                let sampleRate = b.constant(b.ctx.g.sampleRate)
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
        case .deterministicPhasor:
            // Stateless phasor for constant frequency - fully parallelizable
            // phase = fmod(freq / sampleRate * threadIndex, 1.0)
            guard inputs.count == 1 else {
                throw DGenError.insufficientInputs(
                    operator: "deterministicPhasor", expected: 1, actual: inputs.count)
            }
            // Use emitUnaryOp to handle both scalar and tensor cases
            try emitUnaryOp(b: b, g: g, node: node, inputs: inputs) { freq in
                let sampleRate = b.constant(g.sampleRate)
                // Use currentFrameIndex which returns the correct frame index in both normal
                // and frame-aware tensor blocks
                let frameIdx = b.currentFrameIndex()
                // Compute phase directly: (freq / sampleRate * frameIndex) mod 1.0
                let phaseIncrement = freq / sampleRate
                let rawPhase = phaseIncrement * frameIdx
                return rawPhase - b.floor(rawPhase)  // fmod equivalent for positive values
            }
        case .gradDeterministicPhasor:
            // Gradient for deterministic phasor: d(phase)/d(freq) = frameIndex / sampleRate
            // inputs: [gradOutput, sampleRate]
            // Use currentFrameIndex() - matches forward pass which uses currentFrameIndex() for tensor contexts
            guard inputs.count == 2 else { fatalError("gradDeterministicPhasor requires 2 inputs") }
            try emitBinaryOp(b: b, g: g, node: node, inputs: inputs) { gradOut, sampleRate in
                let frameIdx = b.currentFrameIndex()
                let gradFreq = gradOut * frameIdx / sampleRate
                return gradFreq
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
                throw DGenError.tensorError(
                    op: "conv1d", reason: "requires 1D input/output tensors")
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
                    let inVal = b.gswitch(
                        inBounds, b.memoryRead(inTensor.cellId, safeIdx), b.constant(0))

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

                        let rawIdx = b.tensorMemoryIndex(
                            inTensor, indices: [b.cast(inY, to: .int), b.cast(inX, to: .int)])
                        let safeIdx = b.gswitch(inBounds, rawIdx, b.constant(0))
                        let inVal = b.gswitch(
                            inBounds, b.memoryRead(inTensor.cellId, safeIdx), b.constant(0))

                        let kMemIdx = b.tensorMemoryIndex(
                            kTensor, indices: [b.cast(ky, to: .int), b.cast(kx, to: .int)])
                        let kVal = b.memoryRead(kTensor.cellId, kMemIdx)

                        acc.accumulate(inVal * kVal)
                    }
                }
                _ = b.memoryWrite(outCell, b.cast(flatIdx, to: .int), acc.value)
            }

        case .sum:
            if let scratch = ctx.frameTensorChainScratch[nodeId] {
                let acc = b.float(0.0)
                // Use currentFrameIndex for correct behavior in frame-aware tensor blocks
                let frameIdx = b.currentFrameIndex()
                let sizeExpr = b.constant(Float(scratch.tensorSize))
                b.loop(scratch.tensorSize) { i in
                    let idx = frameIdx * sizeExpr + b.cast(i, to: .float)
                    let val = b.memoryRead(scratch.cellId, b.cast(idx, to: .int))
                    acc.accumulate(val)
                }
                b.use(val: acc.value)
                break
            }
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
                throw DGenError.insufficientInputs(
                    operator: "peek", expected: 3, actual: node.inputs.count)
            }

            let tensorInput = node.inputs[0]

            // Get tensor shape from the input node
            guard let inputNode = g.nodes[tensorInput],
                case .tensor(let shape) = inputNode.shape,
                shape.count >= 2
            else {
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
            let clampedChannel = b.floor(
                b.max(zero, b.min(channel, b.constant(Float(numChannels - 1)))))
            let channelOffset = channelSizeFloat * clampedChannel

            // Calculate final read position
            let finalReadPos = channelOffset + positiveIndex

            // Linear interpolation for fractional indices
            let flooredPos = b.floor(finalReadPos)
            let frac = finalReadPos - flooredPos

            // Get tensor cellId - either from concrete tensor or from input tensor
            let cellId: CellID
            if let tensorId = g.nodeToTensor[tensorInput],
                let tensor = g.tensors[tensorId]
            {
                cellId = tensor.cellId
            } else {
                throw DGenError.tensorError(
                    op: "peek",
                    reason: "frame-based tensor peek requires tensor context - not yet implemented")
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

        case .fft(
            let windowSize, let hopSize, let scratchCell, let ringBufferCell, let writePosCell,
            let counterCell):
            // FFT using Cooley-Tukey algorithm with ring buffer for sample history
            // Input: signal (scalar per frame)
            // Output: tensor [numBins, 2] where numBins = windowSize/2 + 1
            // hopSize: only compute FFT every hopSize frames (reduces CPU)

            // Mark as scalar to avoid SIMD variable naming collisions in C renderer
            b.markRequiresScalar()

            guard inputs.count == 1 else {
                throw DGenError.insufficientInputs(
                    operator: "fft", expected: 1, actual: inputs.count)
            }
            guard windowSize > 0 && (windowSize & (windowSize - 1)) == 0 else {
                throw DGenError.tensorError(op: "fft", reason: "windowSize must be a power of 2")
            }

            let numBins = windowSize / 2 + 1
            let numStages = Int(log2(Double(windowSize)))

            // Get output tensor
            guard let outTensorId = g.nodeToTensor[node.id],
                let outTensor = g.tensors[outTensorId]
            else {
                throw DGenError.tensorError(op: "fft", reason: "missing output tensor")
            }

            let sig = b.value(inputs[0])

            // Scratch memory layout:
            // scratch[0..<windowSize] = real components
            // scratch[windowSize..<windowSize*2] = imaginary components
            let imagOffset = windowSize

            let winSizeFloat = b.constant(Float(windowSize))
            let hopSizeFloat = b.constant(Float(hopSize))
            let zero = b.constant(0.0)
            let one = b.constant(1.0)

            // Load write position and hop counter
            let writePos = b.memoryRead(writePosCell, zero)
            let counter = b.memoryRead(counterCell, zero)

            // Write current sample to ring buffer at writePos
            _ = b.memoryWrite(ringBufferCell, b.cast(writePos, to: .int), sig)

            // Update write position: (writePos + 1) % windowSize
            let nextWritePos = writePos + one
            let wrappedWritePos = b.gswitch(nextWritePos >= winSizeFloat, zero, nextWritePos)
            _ = b.memoryWrite(writePosCell, zero, wrappedWritePos)

            // Update counter: (counter + 1) % hopSize
            let nextCounter = counter + one
            let wrappedCounter = b.gswitch(nextCounter >= hopSizeFloat, zero, nextCounter)
            _ = b.memoryWrite(counterCell, zero, wrappedCounter)

            // Only compute FFT when counter == 0 (every hopSize frames)
            let shouldCompute = counter == zero

            // Wrap entire FFT computation in if-statement for efficiency
            // This skips all expensive loops when shouldCompute is false
            b.if_(shouldCompute) {
                // 1. Load samples from ring buffer into scratch (imaginary = 0)
                // Read from oldest to newest: start from wrappedWritePos (oldest) and go around
                b.loop(windowSize) { n in
                    let nFloat = b.cast(n, to: .float)
                    // Read position: (writePos + n) % windowSize gives oldest to newest
                    // Use wrappedWritePos which now points to oldest sample
                    let readIdx = wrappedWritePos + nFloat
                    let wrappedReadIdx = b.gswitch(
                        readIdx >= winSizeFloat, readIdx - winSizeFloat, readIdx)
                    let sample = b.memoryRead(ringBufferCell, b.cast(wrappedReadIdx, to: .int))
                    _ = b.memoryWrite(scratchCell, b.cast(n, to: .int), sample)
                    _ = b.memoryWrite(
                        scratchCell, b.cast(n, to: .int) + b.constant(Float(imagOffset)),
                        b.constant(0.0))
                }

                // 2. Bit-reversal permutation
                b.loop(windowSize) { i in
                    // Compute bit-reversed index
                    var rev = b.constant(0.0)
                    var n = b.cast(i, to: .float)
                    for _ in 0..<numStages {
                        rev = rev * b.constant(2.0) + (n % b.constant(2.0))
                        n = b.floor(n / b.constant(2.0))
                    }

                    let iFloat = b.cast(i, to: .float)
                    // Swap if i < rev (avoid double-swap)
                    let shouldSwap = iFloat < rev
                    let iInt = b.cast(i, to: .int)
                    let revInt = b.cast(rev, to: .int)

                    // Load values at i and rev
                    let tempRealI = b.memoryRead(scratchCell, iInt)
                    let tempImagI = b.memoryRead(scratchCell, iInt + b.constant(Float(imagOffset)))
                    let tempRealRev = b.memoryRead(scratchCell, revInt)
                    let tempImagRev = b.memoryRead(
                        scratchCell, revInt + b.constant(Float(imagOffset)))

                    // Conditionally swap
                    let newRealI = b.gswitch(shouldSwap, tempRealRev, tempRealI)
                    let newImagI = b.gswitch(shouldSwap, tempImagRev, tempImagI)
                    let newRealRev = b.gswitch(shouldSwap, tempRealI, tempRealRev)
                    let newImagRev = b.gswitch(shouldSwap, tempImagI, tempImagRev)

                    _ = b.memoryWrite(scratchCell, iInt, newRealI)
                    _ = b.memoryWrite(scratchCell, iInt + b.constant(Float(imagOffset)), newImagI)
                    _ = b.memoryWrite(scratchCell, revInt, newRealRev)
                    _ = b.memoryWrite(
                        scratchCell, revInt + b.constant(Float(imagOffset)), newImagRev)
                }

                // 3. Butterfly stages
                for stage in 0..<numStages {
                    let butterflySize = 1 << (stage + 1)  // 2, 4, 8, ...
                    let halfSize = butterflySize / 2
                    let numGroups = windowSize / butterflySize

                    b.loop(numGroups) { group in
                        b.loop(halfSize) { k in
                            let groupFloat = b.cast(group, to: .float)
                            let kFloat = b.cast(k, to: .float)
                            let butterflySizeFloat = b.constant(Float(butterflySize))
                            let halfSizeFloat = b.constant(Float(halfSize))

                            let i = groupFloat * butterflySizeFloat + kFloat
                            let j = i + halfSizeFloat

                            // Twiddle factor: W = e^(-2*pi*i*k/butterflySize)
                            let angle = b.constant(-2.0) * b.pi * kFloat / butterflySizeFloat
                            let wr = b.cos(angle)
                            let wi = b.sin(angle)

                            let iInt = b.cast(i, to: .int)
                            let jInt = b.cast(j, to: .int)

                            // Load values
                            let ar = b.memoryRead(scratchCell, iInt)
                            let ai = b.memoryRead(scratchCell, iInt + b.constant(Float(imagOffset)))
                            let br = b.memoryRead(scratchCell, jInt)
                            let bi = b.memoryRead(scratchCell, jInt + b.constant(Float(imagOffset)))

                            // Butterfly: (ar,ai) + W*(br,bi) and (ar,ai) - W*(br,bi)
                            // W*(br,bi) = (wr*br - wi*bi, wr*bi + wi*br)
                            let tr = wr * br - wi * bi
                            let ti = wr * bi + wi * br

                            _ = b.memoryWrite(scratchCell, iInt, ar + tr)
                            _ = b.memoryWrite(
                                scratchCell, iInt + b.constant(Float(imagOffset)), ai + ti)
                            _ = b.memoryWrite(scratchCell, jInt, ar - tr)
                            _ = b.memoryWrite(
                                scratchCell, jInt + b.constant(Float(imagOffset)), ai - ti)
                        }
                    }
                }

                // 4. Copy first numBins to output tensor [numBins, 2]
                // Output layout: column-major for peek compatibility
                // Channel 0 (real): offsets 0, 1, 2, ... numBins-1
                // Channel 1 (imag): offsets numBins, numBins+1, ... 2*numBins-1
                b.loop(numBins) { k in
                    let kInt = b.cast(k, to: .int)
                    let real = b.memoryRead(scratchCell, kInt)
                    let imag = b.memoryRead(scratchCell, kInt + b.constant(Float(imagOffset)))
                    _ = b.memoryWrite(outTensor.cellId, kInt, real)
                    _ = b.memoryWrite(outTensor.cellId, kInt + b.constant(Float(numBins)), imag)
                }
            }

            // Register output for downstream ops
            ctx.values[nodeId] = .empty

        case .ifft(
            let windowSize, let hopSize, let scratchCell, let outputRingCell, let readPosCell,
            let counterCell):
            // IFFT using Cooley-Tukey algorithm with overlap-add for reconstruction
            // Input: spectrum tensor [numBins, 2] where numBins = windowSize/2 + 1
            // Output: scalar (one sample per frame via overlap-add)

            b.markRequiresScalar()

            guard inputs.count == 1 else {
                throw DGenError.insufficientInputs(
                    operator: "ifft", expected: 1, actual: inputs.count)
            }
            guard windowSize > 0 && (windowSize & (windowSize - 1)) == 0 else {
                throw DGenError.tensorError(op: "ifft", reason: "windowSize must be a power of 2")
            }

            let numBins = windowSize / 2 + 1
            let numStages = Int(log2(Double(windowSize)))
            let imagOffset = windowSize  // Scratch layout: real[0..<N], imag[N..<2N]

            // Get input tensor (spectrum)
            guard let inputNodeId = node.inputs.first,
                let inputTensorId = g.nodeToTensor[inputNodeId],
                let inputTensor = g.tensors[inputTensorId]
            else {
                throw DGenError.tensorError(op: "ifft", reason: "missing input spectrum tensor")
            }

            let winSizeFloat = b.constant(Float(windowSize))
            let hopSizeFloat = b.constant(Float(hopSize))
            let zero = b.constant(0.0)
            let one = b.constant(1.0)
            let numBinsFloat = b.constant(Float(numBins))

            // Load read position and hop counter
            let readPos = b.memoryRead(readPosCell, zero)
            let counter = b.memoryRead(counterCell, zero)

            // Output current sample from ring buffer, then clear it for next overlap-add cycle
            let outputSample = b.memoryRead(outputRingCell, b.cast(readPos, to: .int))
            _ = b.memoryWrite(outputRingCell, b.cast(readPos, to: .int), zero)

            // Update read position: (readPos + 1) % windowSize
            let nextReadPos = readPos + one
            let wrappedReadPos = b.gswitch(nextReadPos >= winSizeFloat, zero, nextReadPos)
            _ = b.memoryWrite(readPosCell, zero, wrappedReadPos)

            // Update counter: (counter + 1) % hopSize
            let nextCounter = counter + one
            let wrappedCounter = b.gswitch(nextCounter >= hopSizeFloat, zero, nextCounter)
            _ = b.memoryWrite(counterCell, zero, wrappedCounter)

            // Only compute IFFT when counter == 0
            let shouldCompute = counter == zero
            b.if_(shouldCompute) {
                // 1. Load spectrum into scratch with conjugate symmetry to get full N points
                // Input layout (column-major): real[0..<numBins], imag[numBins..<2*numBins]
                // For real signal: X[N-k] = conj(X[k]) for k = 1 to N/2-1
                // DC (k=0) and Nyquist (k=N/2) have no imaginary part in output

                b.loop(windowSize) { n in
                    let nInt = b.cast(n, to: .int)
                    let halfN = b.constant(Float(windowSize / 2))

                    // Determine which input bin to read from
                    let isFirstHalf = n <= halfN
                    let inputBin = b.gswitch(isFirstHalf, n, winSizeFloat - n)
                    let inputBinInt = b.cast(inputBin, to: .int)

                    // Read real and imag from input tensor (column-major layout)
                    let inReal = b.memoryRead(inputTensor.cellId, inputBinInt)
                    let inImag = b.memoryRead(
                        inputTensor.cellId, inputBinInt + b.constant(Float(numBins)))

                    // For second half (conjugate), negate imaginary part
                    let realVal = inReal
                    let imagVal = b.gswitch(isFirstHalf, inImag, zero - inImag)

                    // Write to scratch
                    _ = b.memoryWrite(scratchCell, nInt, realVal)
                    _ = b.memoryWrite(scratchCell, nInt + b.constant(Float(imagOffset)), imagVal)
                }

                // 2. Bit-reversal permutation (same as FFT)
                b.loop(windowSize) { i in
                    var rev = b.constant(0.0)
                    var n = i
                    for _ in 0..<numStages {
                        rev = rev * b.constant(2.0) + b.mod(n, b.constant(2.0))
                        n = b.floor(n / b.constant(2.0))
                    }

                    let iFloat = i
                    let shouldSwap = b.and(iFloat < rev, shouldCompute)

                    let iInt = b.cast(i, to: .int)
                    let revInt = b.cast(rev, to: .int)

                    let tempR = b.memoryRead(scratchCell, iInt)
                    let tempI = b.memoryRead(scratchCell, iInt + b.constant(Float(imagOffset)))
                    let revR = b.memoryRead(scratchCell, revInt)
                    let revI = b.memoryRead(scratchCell, revInt + b.constant(Float(imagOffset)))

                    let newIR = b.gswitch(shouldSwap, revR, tempR)
                    let newII = b.gswitch(shouldSwap, revI, tempI)
                    let newRevR = b.gswitch(shouldSwap, tempR, revR)
                    let newRevI = b.gswitch(shouldSwap, tempI, revI)

                    _ = b.memoryWrite(scratchCell, iInt, newIR)
                    _ = b.memoryWrite(scratchCell, iInt + b.constant(Float(imagOffset)), newII)
                    _ = b.memoryWrite(scratchCell, revInt, newRevR)
                    _ = b.memoryWrite(scratchCell, revInt + b.constant(Float(imagOffset)), newRevI)
                }

                // 3. Butterfly stages (IFFT uses POSITIVE twiddle angles)
                var butterflySize = 2
                for stage in 0..<numStages {
                    let halfSize = butterflySize / 2
                    let numGroups = windowSize / butterflySize

                    b.loop(numGroups) { group in
                        b.loop(halfSize) { k in
                            let i = group * b.constant(Float(butterflySize)) + k
                            let j = i + b.constant(Float(halfSize))

                            // IFFT twiddle: W = e^(+2πi*k/butterflySize) - POSITIVE angle
                            let angle =
                                b.constant(2.0) * b.constant(Float.pi) * k
                                / b.constant(Float(butterflySize))
                            let wr = b.cos(angle)
                            let wi = b.sin(angle)

                            let iInt = b.cast(i, to: .int)
                            let jInt = b.cast(j, to: .int)

                            let ar = b.memoryRead(scratchCell, iInt)
                            let ai = b.memoryRead(scratchCell, iInt + b.constant(Float(imagOffset)))
                            let br = b.memoryRead(scratchCell, jInt)
                            let bi = b.memoryRead(scratchCell, jInt + b.constant(Float(imagOffset)))

                            // Complex multiply: (wr + i*wi) * (br + i*bi)
                            let tr = wr * br - wi * bi
                            let ti = wr * bi + wi * br

                            // Butterfly
                            _ = b.memoryWrite(scratchCell, iInt, ar + tr)
                            _ = b.memoryWrite(
                                scratchCell, iInt + b.constant(Float(imagOffset)), ai + ti)
                            _ = b.memoryWrite(scratchCell, jInt, ar - tr)
                            _ = b.memoryWrite(
                                scratchCell, jInt + b.constant(Float(imagOffset)), ai - ti)
                        }
                    }
                    butterflySize *= 2
                }

                // 4. Divide by N and add to output ring buffer (overlap-add)
                let invN = b.constant(1.0 / Float(windowSize))
                b.loop(windowSize) { n in
                    let nInt = b.cast(n, to: .int)
                    let realVal = b.memoryRead(scratchCell, nInt) * invN

                    // Calculate output position with wrap-around
                    let outPos = readPos + n
                    let wrappedOutPos = b.gswitch(
                        outPos >= winSizeFloat, outPos - winSizeFloat, outPos)
                    let outPosInt = b.cast(wrappedOutPos, to: .int)

                    // Overlap-add: accumulate into output buffer
                    let existing = b.memoryRead(outputRingCell, outPosInt)
                    _ = b.memoryWrite(outputRingCell, outPosInt, existing + realVal)
                }
            }

            // Use the output sample
            b.use(val: outputSample)

        case .expand(let targetShape):
            // Broadcast scalar to tensor shape (for sum backward)
            // Input is scalar, output is tensor where all elements = input
            guard inputs.count == 1 else { fatalError("expand requires 1 input") }

            // Get or create output tensor
            guard let outTensor = g.nodeToTensor[nodeId].flatMap({ g.tensors[$0] }) else {
                // Fallback: just pass through the scalar
                b.use(val: b.value(inputs[0]))
                break
            }

            let size = targetShape.reduce(1, *)
            let inputNodeId = node.inputs[0]

            // Check if output tensor is frame-aware (needs per-frame storage)
            let isFrameAware = ctx.frameAwareTensorCells.contains(outTensor.cellId)

            // Check if input has a tensor cell we can read from (for cross-scope access)
            if let inputTensorId = g.nodeToTensor[inputNodeId],
                let inputTensor = g.tensors[inputTensorId]
            {
                // Input is a tensor - read element 0 (assumes scalar stored in tensor)
                if isFrameAware {
                    // Frame-aware output: write to frame-indexed positions
                    let frameIdx = b.currentFrameIndex()
                    let frameBase = frameIdx * b.constant(Float(size))
                    b.parallelRange(size) { idx in
                        let scalarVal = b.memoryRead(inputTensor.cellId, b.int(0))
                        let writePos = frameBase + b.cast(idx, to: .float)
                        _ = b.memoryWrite(outTensor.cellId, b.cast(writePos, to: .int), scalarVal)
                    }
                } else {
                    // Non-frame-aware: existing linear write
                    b.parallelRange(size) { idx in
                        let scalarVal = b.memoryRead(inputTensor.cellId, b.int(0))
                        _ = b.memoryWrite(outTensor.cellId, b.cast(idx, to: .int), scalarVal)
                    }
                }
            } else {
                // Input is a scalar computed in current scope
                let scalarVal = b.value(inputs[0])

                if isFrameAware {
                    // Frame-aware output: write to frame-indexed positions
                    // Use scalar directly without write-read-back to avoid race condition
                    let frameIdx = b.currentFrameIndex()
                    let frameBase = frameIdx * b.constant(Float(size))
                    b.parallelRange(size) { idx in
                        let writePos = frameBase + b.cast(idx, to: .float)
                        _ = b.memoryWrite(outTensor.cellId, b.cast(writePos, to: .int), scalarVal)
                    }
                } else {
                    // Non-frame-aware: use write-read-back pattern for scope capture
                    _ = b.memoryWrite(outTensor.cellId, b.int(0), scalarVal)
                    b.parallelRange(size) { idx in
                        let storedVal = b.memoryRead(outTensor.cellId, b.int(0))
                        _ = b.memoryWrite(outTensor.cellId, b.cast(idx, to: .int), storedVal)
                    }
                }
            }

        case .expandAxis(let targetShape, let axis):
            // Broadcast along a specific axis (for sumAxis backward)
            guard inputs.count == 1 else { fatalError("expandAxis requires 1 input") }

            guard let inTensor = g.nodeToTensor[node.inputs[0]].flatMap({ g.tensors[$0] }),
                let outTensor = g.nodeToTensor[nodeId].flatMap({ g.tensors[$0] })
            else {
                // Fallback
                if let s = inputs.first { b.use(val: b.value(s)) }
                break
            }

            let outSize = targetShape.reduce(1, *)
            let normalizedAxis = axis < 0 ? targetShape.count + axis : axis

            // Compute strides for output shape (excluding the expanded axis)
            var inputShape = targetShape
            inputShape.remove(at: normalizedAxis)
            let inputStrides = Tensor.computeRowMajorStrides(inputShape)
            let outputStrides = Tensor.computeRowMajorStrides(targetShape)

            b.parallelRange(outSize) { outIdx in
                // Map output index to input index (skip the expanded axis dimension)
                var inputFlatIdx = b.int(0)
                var inDim = 0
                for dim in 0..<targetShape.count {
                    if dim == normalizedAxis { continue }
                    let coord = b.mod(
                        b.floorDiv(outIdx, b.int(outputStrides[dim])), b.int(targetShape[dim]))
                    inputFlatIdx = b.add(inputFlatIdx, b.mul(coord, b.int(inputStrides[inDim])))
                    inDim += 1
                }

                let val = b.memoryRead(inTensor.cellId, inputFlatIdx)
                _ = b.memoryWrite(outTensor.cellId, b.cast(outIdx, to: .int), val)
            }

        case .gradPhasor(_):
            // Gradient for phasor: d(phase)/d(freq) = frameIndex / sampleRate
            // inputs: [gradOutput, sampleRate]
            // Use threadIndex() - the actual sample index, not decomposed frame index
            guard inputs.count == 2 else { fatalError("gradPhasor requires 2 inputs") }
            let gradOut = b.value(inputs[0])
            let sampleRate = b.value(inputs[1])
            let frameIdx = b.threadIndex()
            let gradFreq = gradOut * frameIdx / sampleRate
            b.use(val: gradFreq)

        }
        ops.append(contentsOf: b.ops)
        return ops
    }
}
