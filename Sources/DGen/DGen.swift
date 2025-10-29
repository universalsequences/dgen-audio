import Foundation

public class IRContext {
    private var varIdx = 0
    private var gradIdx = 0
    private var constantIdx = 0
    // Reuse constant IDs for identical values to reduce duplicate vdupq constants
    private var constantIdByValue: [Float: ConstantID] = [:]

    public init() {}

    public var globals: Set<VarID> = []

    // map of nodeId -> Lazy value (variable or constant)
    public var values: [NodeID: Lazy] = [:]
    public var gradients: [NodeID: GradID] = [:]
    public var constants: [ConstantID: Float] = [:]
    public var variables: [VarID: NodeID] = [:]
    public var tapeIndex: [NodeID: Int] = [:]

    public func useConstant(src: NodeID?, value: Float) -> Lazy {
        if let existing = constantIdByValue[value] {
            let constant = Lazy.constant(existing, value)
            if let srcId = src { self.values[srcId] = constant }
            return constant
        }

        let constantId = self.constantIdx + 1
        self.constantIdx = constantId
        self.constants[constantId] = value
        constantIdByValue[value] = constantId

        let constant = Lazy.constant(constantId, value)
        if let srcId = src { self.values[srcId] = constant }
        return constant
    }

    public func useGradient(src: NodeID) -> GradID {
        if let gradId = self.gradients[src] {
            return gradId
        }
        let gradId = self.gradIdx + 1
        self.gradIdx = gradId
        return gradId
    }

    public func useVariable(src: NodeID?, trackInValues: Bool = true) -> Lazy {
        let varId = self.varIdx + 1
        self.varIdx = varId
        let variable = Lazy.variable(varId, src)
        if let srcNodeId = src, trackInValues {
            self.values[srcNodeId] = variable
            self.variables[varId] = srcNodeId
        }
        return variable
    }
}

public struct Node {
    public let id: NodeID
    public var op: LazyOp
    public let inputs: [NodeID]
    public var temporalDependencies: [NodeID] = []

    /// Returns all dependencies (both regular inputs and temporal dependencies)
    public var allDependencies: [NodeID] {
        return inputs + temporalDependencies
    }
}

public final class Graph {
    private var next = 0
    public var nodes: [NodeID: Node] = [:]
    private var nextCellId = 0
    public var nextTensorId = 0
    public var tensors: [TensorID: Tensor] = [:]

    public init() {}

    /// Returns the total number of allocated memory cells
    public var totalMemoryCells: Int { nextCellId }

    @discardableResult public func n(_ op: LazyOp, _ ins: NodeID...) -> NodeID {
        return n(op, ins)
    }

    @discardableResult public func n(_ op: LazyOp, _ ins: [NodeID]) -> NodeID {
        let id = next
        next += 1
        nodes[id] = Node(id: id, op: op, inputs: ins)

        // Handle seq operator: find root dependencies of B and make them depend on A
        if case .seq = op, ins.count >= 2 {
            let a = ins[0]  // First input (e.g., writeOp)
            let b = ins[1]  // Second input (e.g., interpolated)

            // For seq(a, b), find all nodes in B's dependency tree that should wait for A
            // We traverse B's dependencies and find memory operations that should depend on A
            var visited = Set<NodeID>()
            var queue = [b]

            while !queue.isEmpty {
                let currentId = queue.removeFirst()
                if visited.contains(currentId) { continue }
                visited.insert(currentId)

                guard let node = nodes[currentId] else { continue }

                // Check if this node is a memory operation that should depend on A
                switch node.op {
                case .memoryRead(_), .historyRead(_):
                    // Memory reads should depend on the write
                    if var currentNode = nodes[currentId] {
                        currentNode.temporalDependencies.append(a)
                        nodes[currentId] = currentNode
                    }
                default:
                    // For other nodes, continue traversing
                    queue.append(contentsOf: node.inputs)
                }
            }
        }

        return id
    }

    /// Allocate a new cell ID for memory-based operations like phasor, latch, etc.
    /// For vector operations, this will allocate consecutive slots
    public func alloc(vectorWidth: Int = 1) -> CellID {
        let cellId = nextCellId
        nextCellId += vectorWidth
        return cellId
    }

    /// Allocate a single cell (backward compatibility)
    public func alloc() -> CellID {
        return alloc(vectorWidth: 1)
    }

    /// Find all root dependencies of a node (nodes with no inputs in the dependency tree)
    private func findRootDependencies(of nodeId: NodeID) -> Set<NodeID> {
        var roots = Set<NodeID>()
        var visited = Set<NodeID>()
        var queue = [nodeId]

        while !queue.isEmpty {
            let currentId = queue.removeFirst()
            if visited.contains(currentId) { continue }
            visited.insert(currentId)

            guard let node = nodes[currentId] else { continue }

            if node.inputs.isEmpty {
                // This is a root node (no dependencies)
                roots.insert(currentId)
            } else {
                // Add all inputs to the queue to explore
                queue.append(contentsOf: node.inputs)
            }
        }

        return roots
    }

    public func seq(a: NodeID, b: NodeID) -> NodeID {
        return n(.seq, a, b)
    }

    // Higher-level compound operations

    /// Delta: returns the difference between current and previous input value
    /// delta(input) = input - history(input)
    public func delta(_ input: NodeID) -> NodeID {
        let cellId = alloc()
        let historyRead = n(.historyRead(cellId))
        _ = n(.historyWrite(cellId), input)
        return n(.sub, input, historyRead)
    }

    /// Change: returns the sign of the difference between current and previous input
    /// change(input) = sign(input - history(input))
    public func change(_ input: NodeID) -> NodeID {
        let deltaNode = delta(input)
        return n(.sign, deltaNode)
    }

    /// RampToTrig: detects discontinuities in a ramp signal and outputs triggers
    /// Detects when a ramp resets (wraps around) by analyzing the rate of change
    public func rampToTrig(_ ramp: NodeID) -> NodeID {
        let cellId = alloc()
        let historyRead = n(.historyRead(cellId))
        _ = n(.historyWrite(cellId), ramp)

        // Calculate abs(ramp - history) / (ramp + history)
        let diff = n(.sub, ramp, historyRead)
        let sum = n(.add, ramp, historyRead)
        let ratio = n(.div, diff, sum)
        let absRatio = n(.abs, ratio)

        // Check if ratio > 0.5 (indicates a wrap-around)
        let halfConst = n(.constant(0.5))
        let isLargeJump = n(.gt, absRatio, halfConst)

        // Get the change/sign of this boolean result and convert to positive triggers only
        let changeResult = change(isLargeJump)
        // Note: Using abs() here loses distinction between rising/falling edges,
        // but provides clean 0/1 triggers suitable for accumulation and most trigger uses
        return n(.abs, changeResult)
    }

    /// Scale: maps a value from one range to another with optional exponential curve
    /// Based on Max MSP gen~ scale function
    /// scale(value, min1, max1, min2, max2, exponent = 1.0)
    /// - value: input value to scale
    /// - min1: minimum of input range
    /// - max1: maximum of input range
    /// - min2: minimum of output range
    /// - max2: maximum of output range
    /// - exponent: exponential curve factor (1.0 = linear, >1.0 = exponential, <1.0 = logarithmic)
    public func scale(
        _ value: NodeID, _ min1: NodeID, _ max1: NodeID, _ min2: NodeID, _ max2: NodeID,
        _ exponent: NodeID? = nil
    ) -> NodeID {
        let exp = exponent ?? n(.constant(1.0))

        // Calculate range1 = max1 - min1
        let range1 = n(.sub, max1, min1)

        // Calculate range2 = max2 - min2
        let range2 = n(.sub, max2, min2)

        // Calculate normalized value = (value - min1) / range1
        // Handle division by zero by checking if range1 == 0
        let valueMinusMin1 = n(.sub, value, min1)
        let zero = n(.constant(0.0))
        let range1IsZero = n(.eq, range1, zero)
        let normVal = n(.gswitch, range1IsZero, zero, n(.div, valueMinusMin1, range1))

        // Apply exponential curve: pow(normVal, exponent)
        let curvedVal = n(.pow, normVal, exp)

        // Scale to output range: min2 + range2 * curvedVal
        let scaledRange = n(.mul, range2, curvedVal)
        return n(.add, min2, scaledRange)
    }

    /// Triangle: converts a ramp signal to a triangle wave
    /// Based on Max MSP gen~ triangle function
    /// triangle(ramp, duty = 0.5)
    /// - ramp: input ramp signal (0-1)
    /// - duty: duty cycle (0-1), determines the peak position (0.5 = symmetric triangle)
    /// When ramp < duty: output scales from 0 to 1
    /// When ramp >= duty: output scales from 1 to 0
    public func triangle(_ ramp: NodeID, _ duty: NodeID? = nil) -> NodeID {
        let dutyValue = duty ?? n(.constant(0.5))

        // Check if ramp < duty
        let isRising = n(.lt, ramp, dutyValue)

        // Rising portion: scale(ramp, 0, duty, 0, 1)
        let zero = n(.constant(0.0))
        let one = n(.constant(1.0))
        let risingOutput = scale(ramp, zero, dutyValue, zero, one)

        // Falling portion: scale(ramp, duty, 1, 1, 0)
        let fallingOutput = scale(ramp, dutyValue, one, one, zero)

        // Switch between rising and falling based on comparison
        return n(.gswitch, isRising, risingOutput, fallingOutput)
    }

    public func wrap(_ input: NodeID, _ min: NodeID, _ max: NodeID) -> NodeID {
        let range = n(.sub, max, min)
        let normalized = n(.mod, n(.sub, input, min), range)
        return n(
            .gswitch, n(.gte, normalized, n(.constant(0))), n(.add, normalized, min),
            n(.add, range, n(.add, normalized, min)))
    }

    public func clip(_ input: NodeID, _ min: NodeID, _ max: NodeID) -> NodeID {
        return n(.max, min, n(.max, input, max))
    }

    public func selector(_ cond: NodeID, _ args: NodeID...) -> NodeID {
        // Use the new first-class selector operation
        return n(.selector, [cond] + args)
    }

    public func biquad(
        _ in1: NodeID, _ cutoff: NodeID, _ resonance: NodeID, _ gain: NodeID, _ mode: NodeID
    ) -> NodeID {
        // Allocate history cells
        let history0Cell = alloc()
        let history1Cell = alloc()
        let history2Cell = alloc()
        let history3Cell = alloc()

        // History reads
        let history0Read = n(.historyRead(history0Cell))
        let history1Read = n(.historyRead(history1Cell))
        let history2Read = n(.historyRead(history2Cell))
        let history3Read = n(.historyRead(history3Cell))

        // History writes and chaining
        let history0Write = n(.historyWrite(history0Cell), history1Read)
        let history00 = history0Read

        let history2Write = n(.historyWrite(history2Cell), in1)
        let history21 = history2Read

        let history3Write = n(.historyWrite(history3Cell), history2Read)
        let history22 = history2Read

        // Core filter computation following TypeScript exactly
        let param3 = mode
        let add4 = n(.add, param3, n(.constant(1)))
        let param5 = cutoff
        let abs6 = n(.abs, param5)
        let mult7 = n(.mul, abs6, n(.constant(0.00014247585730565955)))
        let cos8 = n(.cos, mult7)
        let mult9 = n(.mul, cos8, n(.constant(-1)))
        let add10 = n(.add, mult9, n(.constant(1)))
        let div11 = n(.div, add10, n(.constant(2)))
        let sub12 = n(.sub, mult9, n(.constant(1)))
        let div13 = n(.div, sub12, n(.constant(-2)))
        let sin14 = n(.sin, mult7)
        let mult15 = n(.mul, sin14, n(.constant(0.5)))
        let param16 = resonance
        let abs17 = n(.abs, param16)
        let div18 = n(.div, mult15, abs17)
        let mult19 = n(.mul, div18, n(.constant(-1)))
        let mult20 = n(.mul, div18, abs17)
        let mult21 = n(.mul, mult20, n(.constant(-1)))
        let add22 = n(.add, div18, n(.constant(1)))
        let selector23 = selector(add4, div11, div13, mult19, mult21, n(.constant(1)), add22)
        let reciprical24 = n(.div, n(.constant(1)), add22)
        let param25 = gain
        let mult26 = n(.mul, reciprical24, param25)
        let b2 = n(.mul, selector23, mult26)
        let mult28 = n(.mul, history3Read, b2)
        let param29 = in1
        let history330 = history3Read
        let mult31 = n(.mul, mult9, n(.constant(2)))
        let selector32 = selector(
            add4, add10, sub12, n(.constant(0)), n(.constant(0)), mult31, mult31)
        let b1 = n(.mul, selector32, mult26)
        let mult34 = n(.mul, history2Read, b1)
        let not_sub35 = n(.sub, n(.constant(1)), div18)
        let selector36 = selector(add4, div11, div13, div18, mult20, n(.constant(1)), not_sub35)
        let b0 = n(.mul, selector36, mult26)
        let mult38 = n(.mul, in1, b0)

        // Final computation - following TypeScript s() pattern
        let add39 = n(.add, n(.add, mult28, mult34), mult38)

        let history040 = history0Read
        let mult41 = n(.mul, not_sub35, reciprical24)
        let mult42 = n(.mul, history040, mult41)
        let mult43 = n(.mul, mult31, reciprical24)
        let mult44 = n(.mul, history1Read, mult43)
        let add45 = n(.add, mult42, mult44)
        let sub46 = n(.sub, add39, add45)

        // Write final result to history1
        let history148 = n(.historyWrite(history1Cell), sub46)

        // Return history148 (last arg of s() equivalent: s(history00, history330, history21, history148))
        return sub46
    }

    private func amp2db(_ amp: NodeID) -> NodeID {
        let absAmp = n(.abs, amp)
        let maxAmp = n(.max, absAmp, n(.constant(0.00001)))
        let log10Val = n(.log10, maxAmp)
        return n(.mul, n(.constant(20.0)), log10Val)
    }

    private func db2amp(_ db: NodeID) -> NodeID {
        let divBy20 = n(.div, db, n(.constant(20.0)))
        return n(.pow, n(.constant(10.0)), divBy20)
    }

    public func compressor(
        _ in1: NodeID, _ ratio: NodeID, _ threshold: NodeID, _ knee: NodeID, _ attack: NodeID,
        _ release: NodeID, _ isSideChain: NodeID, _ sidechainIn: NodeID
    ) -> NodeID {

        // Detect level from either main or sidechain input
        let detectorDb = amp2db(n(.gswitch, isSideChain, sidechainIn, in1))

        // But always process the main input
        let inDb = amp2db(in1)

        // Attack and release coefficient calculations
        let log001 = n(.log, n(.constant(0.01)))
        let sampleRate = n(.constant(44100.0))

        let attackSamples = n(.mul, attack, sampleRate)
        let attackCoef = n(.exp, n(.div, log001, attackSamples))

        let releaseSamples = n(.mul, release, sampleRate)
        let releaseCoef = n(.exp, n(.div, log001, releaseSamples))

        // Knee calculations
        let kneeHalf = n(.div, knee, n(.constant(2.0)))
        let kneeStart = n(.sub, threshold, kneeHalf)
        let kneeEnd = n(.add, threshold, kneeHalf)

        // Position within knee and interpolated ratio
        let positionWithinKnee = n(.div, n(.sub, detectorDb, kneeStart), knee)
        let ratioMinus1 = n(.sub, ratio, n(.constant(1.0)))
        let interpolatedRatio = n(.add, n(.constant(1.0)), n(.mul, positionWithinKnee, ratioMinus1))

        let thresholdMinusKneeStart = n(.sub, threshold, kneeStart)
        let overThreshold = n(
            .sub, detectorDb,
            n(.add, kneeStart, n(.mul, positionWithinKnee, thresholdMinusKneeStart)))

        // Gain reduction calculation using nested gswitch (zswitch equivalent)
        let belowKneeStart = n(.lte, detectorDb, kneeStart)
        let aboveKneeEnd = n(.gte, detectorDb, kneeEnd)

        let hardRatio = n(.sub, n(.constant(1.0)), n(.div, n(.constant(1.0)), ratio))
        let softRatio = n(.sub, n(.constant(1.0)), n(.div, n(.constant(1.0)), interpolatedRatio))

        let hardCompression = n(.mul, n(.sub, detectorDb, threshold), hardRatio)
        let softCompression = n(.mul, overThreshold, softRatio)

        let gr = n(
            .gswitch, belowKneeStart, n(.constant(0.0)),
            n(.gswitch, aboveKneeEnd, hardCompression, softCompression))

        // History cell for previous gain reduction (equivalent to history() in TypeScript)
        let grHistoryCell = alloc()
        let prevGr = n(.historyRead(grHistoryCell))

        // Smooth the gain reduction (attack/release)
        let grIsIncreasing = n(.gt, gr, prevGr)
        let attackSmooth = n(
            .add, n(.mul, attackCoef, prevGr), n(.mul, gr, n(.sub, n(.constant(1.0)), attackCoef)))
        let releaseSmooth = n(
            .add, n(.mul, releaseCoef, prevGr), n(.mul, gr, n(.sub, n(.constant(1.0)), releaseCoef))
        )

        let smoothedGr = n(.gswitch, grIsIncreasing, attackSmooth, releaseSmooth)

        // Write smoothed gain reduction to history
        _ = n(.historyWrite(grHistoryCell), smoothedGr)

        // Apply gain reduction to main input
        let signIn1 = n(.sign, in1)
        let attenuatedDb = n(.sub, inDb, smoothedGr)
        let attenuatedAmp = db2amp(attenuatedDb)

        return n(.mul, signIn1, attenuatedAmp)
    }

    /// Delay: delays the input signal by a variable amount of time with linear interpolation
    /// delay(input, delayTimeInSamples)
    /// - input: input signal to delay
    /// - delayTimeInSamples: delay time in samples (0 to MAX_DELAY)
    /// Implements a circular buffer with linear interpolation for fractional delay times
    public func delay(_ input: NodeID, _ delayTimeInSamples: NodeID) -> NodeID {
        let MAX_DELAY = 88000
        let bufferBase = alloc(vectorWidth: MAX_DELAY)  // Allocate proper buffer space
        let writePosCellId = alloc()

        // Constants
        let one = n(.constant(1.0))
        let zero = n(.constant(0.0))
        let maxDelay = n(.constant(Float(MAX_DELAY)))

        // Accumulator for write position (0 to MAX_DELAY-1, wraps around)
        let writePos = n(.floor, n(.accum(writePosCellId), one, zero, zero, maxDelay))

        // Calculate delay read position: writePos - delayTimeInSamples
        let readPos = n(.sub, writePos, delayTimeInSamples)

        // Wrap-around logic matching TypeScript implementation:
        // if (readPos < 0) readPos += MAX_DELAY
        let isNegative = n(.lt, readPos, zero)
        let afterNegWrap = n(
            .gswitch, isNegative,
            n(.add, readPos, maxDelay),
            readPos)

        // if (readPos >= MAX_DELAY) readPos -= MAX_DELAY
        let isTooLarge = n(.gte, afterNegWrap, maxDelay)
        let finalReadPos = n(
            .gswitch, isTooLarge,
            n(.sub, afterNegWrap, maxDelay),
            afterNegWrap)

        // Use seq to ensure write happens before reads
        let writeOp = n(.memoryWrite(bufferBase), writePos, input)

        // Read with linear interpolation
        let flooredPos = n(.floor, finalReadPos)
        let frac = n(.sub, finalReadPos, flooredPos)

        // Read two samples for interpolation
        let sample1 = n(.memoryRead(bufferBase), flooredPos)
        let nextPos = n(.add, flooredPos, one)

        // Wrap nextPos if needed (handle nextPos >= MAX_DELAY)
        let nextPosWrapped = n(.gswitch, n(.gte, nextPos, maxDelay), zero, nextPos)

        let sample2 = n(.memoryRead(bufferBase), nextPosWrapped)

        // Linear interpolation: (1-frac)*sample1 + frac*sample2
        let interpolated = n(.mix, sample1, sample2, frac)

        // Use seq to ensure write happens before interpolation
        return n(.seq, writeOp, interpolated)
    }
}

extension Lazy {
    public var varId: VarID? {
        switch self {
        case let .variable(id, _):
            return id
        default:
            return nil
        }
    }
}

extension Op {
    public var operands: [Lazy] {
        switch self {
        case let .add(a, b):
            return [a, b]
        case let .mul(a, b):
            return [a, b]
        case let .sub(a, b):
            return [a, b]
        case let .div(a, b):
            return [a, b]
        case let .abs(a):
            return [a]
        case let .sign(a):
            return [a]
        case let .gt(a, b):
            return [a, b]
        case let .lt(a, b):
            return [a, b]
        case let .store(_, b):
            return [b]
        case .load(_):
            return []
        case let .beginIf(a):
            return [a]
        case let .mutate(a, b):
            return [a, b]
        default:
            return []
        }
    }
}
