import Foundation

// MARK: - Parameter

/// A learnable parameter in the computation graph
/// Wraps a CellID and tracks its value, gradients, and optimizer state
public class Parameter {
    let cellId: CellID
    let nodeId: NodeID
    public var value: Float
    let name: String?

    // Internal: set by TrainingContext after compilation
    var gradId: GradID?

    public init(graph: Graph, value: Float, name: String? = nil) {
        self.cellId = graph.alloc()
        self.value = value
        self.name = name
        self.nodeId = graph.n(.param(cellId))
    }

    public func node() -> NodeID {
        return nodeId
    }
}

// MARK: - Optimizer Protocol

public protocol Optimizer {
    mutating func step(parameters: inout [Parameter], gradients: [Float])
    func zeroGrad()
}

// MARK: - SGD Optimizer

public struct SGD: Optimizer {
    public let learningRate: Float

    public init(lr: Float = 0.01) {
        self.learningRate = lr
    }

    public mutating func step(parameters: inout [Parameter], gradients: [Float]) {
        for i in 0..<parameters.count {
            parameters[i].value -= learningRate * gradients[i]
        }
    }

    public func zeroGrad() {
        // No state to clear in SGD
    }
}

// MARK: - Adam Optimizer

public struct Adam: Optimizer {
    public let learningRate: Float
    public let beta1: Float
    public let beta2: Float
    public let epsilon: Float

    private var m: [Float] = []  // First moment
    private var v: [Float] = []  // Second moment
    private var t: Int = 0       // Timestep

    public init(lr: Float = 0.001, beta1: Float = 0.9, beta2: Float = 0.999, epsilon: Float = 1e-8) {
        self.learningRate = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
    }

    public mutating func step(parameters: inout [Parameter], gradients: [Float]) {
        // Initialize momentum buffers on first call
        if m.isEmpty {
            m = Array(repeating: 0.0, count: gradients.count)
            v = Array(repeating: 0.0, count: gradients.count)
        }

        t += 1

        for i in 0..<parameters.count {
            // Update biased first moment estimate
            m[i] = beta1 * m[i] + (1 - beta1) * gradients[i]

            // Update biased second raw moment estimate
            v[i] = beta2 * v[i] + (1 - beta2) * gradients[i] * gradients[i]

            // Compute bias-corrected first moment estimate
            let mHat = m[i] / (1 - pow(beta1, Float(t)))

            // Compute bias-corrected second raw moment estimate
            let vHat = v[i] / (1 - pow(beta2, Float(t)))

            // Update parameters
            parameters[i].value -= learningRate * mHat / (sqrt(vHat) + epsilon)
        }
    }

    public func zeroGrad() {
        // Momentum is persistent across steps - don't clear
    }
}

// MARK: - Training Context

/// Manages learnable parameters, memory, and optimization during training
public class TrainingContext {
    public var parameters: [Parameter]
    private var optimizer: Optimizer
    private var memory: UnsafeMutableRawPointer?
    private var cellAllocations: CellAllocations?
    private var runtime: MetalCompiledKernel?
    private var frameCount: Int = 0
    private var context: IRContext?
    private let lossNode: NodeID?

    /// Initialize training context
    /// - Parameters:
    ///   - parameters: Learnable parameters to optimize
    ///   - optimizer: Optimization algorithm (SGD, Adam, etc.)
    ///   - lossNode: The loss node to seed gradients from (optional, can be inferred if only one output)
    public init(parameters: [Parameter], optimizer: Optimizer, lossNode: NodeID? = nil) {
        self.parameters = parameters
        self.optimizer = optimizer
        self.lossNode = lossNode
    }

    /// Initialize memory and set up parameter mappings after compilation
    /// Must be called before first forward pass
    public func initializeMemory(
        runtime: MetalCompiledKernel,
        cellAllocations: CellAllocations,
        context: IRContext,
        frameCount: Int
    ) {
        self.runtime = runtime
        self.cellAllocations = cellAllocations
        self.frameCount = frameCount
        self.context = context

        // Mark loss node as seed gradient if specified
        if let lossNode = lossNode {
            let _ = context.useGradient(src: lossNode, seed: true)
        }

        // Verify we have seed gradients (loss nodes)
        guard !context.seedGradients.isEmpty else {
            fatalError("""
                No seed gradients found!

                You must either:
                1. Pass lossNode to TrainingContext init:
                   TrainingContext(parameters: [...], optimizer: Adam(), lossNode: loss)

                2. Or manually mark it before compilation:
                   let _ = context.useGradient(src: loss, seed: true)
                """)
        }

        print("🌱 Found \(context.seedGradients.count) seed gradient(s): \(context.seedGradients)")

        // Allocate persistent memory
        self.memory = runtime.allocateNodeMemory()
        guard let memory = self.memory else {
            fatalError("Failed to allocate node memory")
        }

        let memPtr = memory.assumingMemoryBound(to: Float.self)

        // Initialize parameter values in memory
        for param in parameters {
            let physicalCell = cellAllocations.cellMappings[param.cellId] ?? param.cellId
            memPtr[physicalCell] = param.value

            // Look up and store the GradID for this parameter
            param.gradId = context.gradients[param.nodeId]
        }

        // Verify all parameters have gradients
        for param in parameters {
            guard param.gradId != nil else {
                let name = param.name ?? "unnamed"
                fatalError("Parameter '\(name)' (node \(param.nodeId)) has no gradient. Did you set backwards: true?")
            }
        }
    }

    /// Extract and reduce gradients from the gradients buffer
    /// Gradients are stored as: gradients[frameCount * gradId + frameIndex]
    /// We sum across all frames to get the total gradient
    public func extractGradients() -> [Float] {
        guard let runtime = runtime else {
            fatalError("Runtime not initialized")
        }

        let gradPtr = runtime.getGradientsBuffer()

        // DEBUG: Check if seed gradient is still 1.0 after backward pass
        if let firstSeed = context?.seedGradients.first {
            let seedIdx = frameCount * firstSeed
            print("   [DEBUG] After backward pass, seed gradient[\(seedIdx)] = \(gradPtr[seedIdx])")

            // Check all gradIds to see where gradients are
            print("   [DEBUG] Checking all gradient IDs:")
            for gradId in 0...7 {
                let idx = frameCount * gradId
                var sum: Float = 0.0
                for i in 0..<min(4, frameCount) {
                    sum += gradPtr[idx + i]
                }
                print("      gradId=\(gradId): first4_sum=\(sum), [0]=\(gradPtr[idx])")
            }
        }

        return parameters.map { param in
            guard let gradId = param.gradId else {
                fatalError("Parameter has no gradient ID")
            }

            // Sum gradients across all frames
            var totalGrad: Float = 0.0
            let baseIndex = frameCount * gradId

            for frameIndex in 0..<frameCount {
                totalGrad += gradPtr[baseIndex + frameIndex]
            }

            // Use mean instead of sum (average over frames)
            let meanGrad = totalGrad / Float(frameCount)

            // Debug first extraction
            if context?.seedGradients.isEmpty == false {
                let name = param.name ?? "param\(param.nodeId)"
                print("   [DEBUG] Extracting \(name): gradId=\(gradId), baseIndex=\(baseIndex), totalGrad=\(totalGrad), meanGrad=\(meanGrad)")
            }

            return meanGrad
        }
    }

    /// Zero out all gradients in the gradient buffer
    /// Must be called before each backward pass
    public func zeroGrad() {
        guard let runtime = runtime else {
            fatalError("Runtime not initialized")
        }

        runtime.resetGradientBuffers(numFrames: frameCount)

        // DEBUG: Verify seed gradients were set
        if let firstSeed = context?.seedGradients.first {
            let gradPtr = runtime.getGradientsBuffer()
            let checkIdx = frameCount * firstSeed
            print("   [DEBUG] After reset, checking gradients[\(checkIdx)] = \(gradPtr[checkIdx])")
        }

        optimizer.zeroGrad()
    }

    /// Update parameters using gradients and optimizer
    public func step() {
        // Extract gradients from gradient buffer
        let gradients = extractGradients()

        // Debug: Print gradients
        if gradients.contains(where: { $0 != 0.0 }) {
            print("   [DEBUG] Gradients: \(gradients.map { String(format: "%.6f", $0) })")
        } else {
            print("   [DEBUG] WARNING: All gradients are zero!")
        }

        // Update parameters via optimizer
        optimizer.step(parameters: &parameters, gradients: gradients)

        // Write updated values back to memory
        // For Metal, we need to write to the GPU's memory buffer directly
        guard let runtime = runtime, let cellAllocations = cellAllocations else {
            fatalError("Memory not initialized")
        }

        // Get the actual GPU memory buffer (not the external pointer)
        guard let memBuffer = runtime.getBuffer(name: "memory") else {
            fatalError("No memory buffer found")
        }

        let memPtr = memBuffer.contents().assumingMemoryBound(to: Float.self)
        for param in parameters {
            let physicalCell = cellAllocations.cellMappings[param.cellId] ?? param.cellId
            memPtr[physicalCell] = param.value
        }
    }

    /// Get the memory pointer for passing to runWithMemory
    public func getMemory() -> UnsafeMutableRawPointer {
        guard let memory = memory else {
            fatalError("Memory not initialized. Call initializeMemory() first.")
        }
        return memory
    }

    deinit {
        if let mem = memory, let rt = runtime {
            rt.deallocateNodeMemory(mem)
        }
    }
}

// MARK: - Metal Backend Extensions

extension MetalCompiledKernel {
    /// Get direct access to the gradients buffer for parameter updates
    public func getGradientsBuffer() -> UnsafeMutablePointer<Float> {
        guard let buffer = getBuffer(name: "gradients") else {
            fatalError("No gradients buffer found. Ensure backwards: true in compilation options.")
        }
        return buffer.contents().assumingMemoryBound(to: Float.self)
    }
}
