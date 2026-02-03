// Optimizers - Parameter update algorithms
//
// Implements Adam, SGD optimizers that work with DGenLazy tensors.

import DGen

// MARK: - Optimizer Protocol

/// Protocol for parameter optimizers
public protocol Optimizer {
    /// Update parameters using computed gradients
    func step()

    /// Clear all gradients
    func zeroGrad()
}

// MARK: - SGD Optimizer

/// Stochastic Gradient Descent optimizer
///
/// ```swift
/// let opt = SGD(params: [w1, w2], lr: 0.01)
/// loss.backward()
/// opt.step()
/// opt.zeroGrad()
/// ```
public class SGD: Optimizer {
    /// Learning rate
    public let lr: Float

    /// Momentum factor (0 = no momentum)
    public let momentum: Float

    /// Weight decay (L2 regularization)
    public let weightDecay: Float

    /// Whether to use Nesterov momentum
    public let nesterov: Bool

    /// Parameters to optimize (stored as Any to handle Tensor/Signal)
    private var params: [any LazyValue]

    /// Momentum buffers
    private var velocities: [[Float]]

    public init(
        params: [any LazyValue],
        lr: Float = 0.001,
        momentum: Float = 0.0,
        weightDecay: Float = 0.0,
        nesterov: Bool = false
    ) {
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.weightDecay = weightDecay
        self.nesterov = nesterov
        self.velocities = []
    }

    public func step() {
        // TODO: Implement parameter updates
        // This will:
        // 1. Read .grad from each parameter
        // 2. Apply momentum if enabled
        // 3. Update parameter values in the graph's memory
        fatalError("SGD.step() not yet implemented")
    }

    public func zeroGrad() {
        // TODO: Clear gradients
        // Set .grad = nil on all parameters
        fatalError("SGD.zeroGrad() not yet implemented")
    }
}

// MARK: - Adam Optimizer

/// Adam optimizer
///
/// ```swift
/// let opt = Adam(params: [w1, w2], lr: 0.001)
/// loss.backward()
/// opt.step()
/// opt.zeroGrad()
/// ```
public class Adam: Optimizer {
    /// Learning rate
    public let lr: Float

    /// First moment decay rate
    public let beta1: Float

    /// Second moment decay rate
    public let beta2: Float

    /// Small constant for numerical stability
    public let eps: Float

    /// Parameters to optimize
    private var params: [any LazyValue]

    /// First moment estimates
    private var m: [[Float]]

    /// Second moment estimates
    private var v: [[Float]]

    /// Time step
    private var t: Int

    public init(
        params: [any LazyValue],
        lr: Float = 0.001,
        beta1: Float = 0.9,
        beta2: Float = 0.999,
        eps: Float = 1e-8
    ) {
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = []
        self.v = []
        self.t = 0
    }

    public func step() {
        // TODO: Implement Adam update
        // This will:
        // 1. Increment t
        // 2. For each parameter:
        //    - Update biased first moment: m = beta1 * m + (1 - beta1) * grad
        //    - Update biased second moment: v = beta2 * v + (1 - beta2) * grad^2
        //    - Compute bias-corrected estimates
        //    - Update parameter: param -= lr * m_hat / (sqrt(v_hat) + eps)
        fatalError("Adam.step() not yet implemented")
    }

    public func zeroGrad() {
        // TODO: Clear gradients
        fatalError("Adam.zeroGrad() not yet implemented")
    }
}
