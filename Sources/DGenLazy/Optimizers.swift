// Optimizers - Parameter update algorithms
//
// Implements Adam, SGD optimizers that work with DGenLazy tensors.

import DGen
import Foundation

// MARK: - Optimizer Protocol

/// Protocol for parameter optimizers
public class Optimizer {
  /// Parameters to optimize (stored as Any to handle Tensor/Signal)
  public var params: [any LazyValue]

  public init(params: [any LazyValue]) {
    self.params = params
  }

  /// Update parameters using computed gradients
  public func step() {}

  /// Clear all gradients
  public func zeroGrad() {
    for param in params {
      if let tensor = param as? Tensor {
        tensor.grad = nil
      } else if let signal = param as? Signal {
        signal.grad = nil
      }
    }
  }
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

  /// Momentum buffers
  private var velocities: [[Float]]

  public init(
    params: [any LazyValue],
    lr: Float = 0.001,
    momentum: Float = 0.0,
    weightDecay: Float = 0.0,
    nesterov: Bool = false
  ) {
    // Initialize subclass properties BEFORE super.init()
    self.lr = lr
    self.momentum = momentum
    self.weightDecay = weightDecay
    self.nesterov = nesterov
    self.velocities = []
    super.init(params: params)
  }

  public override func step() {
    for (paramIdx, parameter) in params.enumerated() {
      if let tensor = parameter as? Tensor {
        guard let gradTensor = tensor.grad,
          let gradData = gradTensor.getData(),
          let data: [Float] = tensor.getData()
        else {
          continue  // Skip if no gradient computed
        }

        // Initialize velocity buffer on first step
        if velocities.count <= paramIdx {
          velocities.append([Float](repeating: 0.0, count: data.count))
        }

        var newData = data
        for (j, (param, grad)) in zip(data, gradData).enumerated() {
          velocities[paramIdx][j] = momentum * velocities[paramIdx][j] + grad + weightDecay * param
          newData[j] = param - lr * velocities[paramIdx][j]
        }

        tensor.updateDataLazily(newData)
      } else if let sig = parameter as? Signal {
        guard let paramValue = sig.data,
              let gradSignal = sig.grad,
              let gradValue = gradSignal.data else {
          continue
        }

        // Simple SGD for scalar: param = param - lr * grad
        // (No momentum for scalars for now - could add if needed)
        let newValue = paramValue - lr * gradValue
        sig.updateDataLazily(newValue)
      }
    }
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
  /// Learning rate (mutable for LR scheduling)
  public var lr: Float

  /// First moment decay rate
  public let beta1: Float

  /// Second moment decay rate
  public let beta2: Float

  /// Small constant for numerical stability
  public let eps: Float

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
    // Initialize subclass properties BEFORE super.init()
    self.lr = lr
    self.beta1 = beta1
    self.beta2 = beta2
    self.eps = eps
    self.m = []
    self.v = []
    self.t = 0
    super.init(params: params)
  }

  public override func step() {
    t += 1

    for (paramIdx, parameter) in params.enumerated() {
      if let tensor = parameter as? Tensor {
        guard let gradTensor = tensor.grad,
          let gradData = gradTensor.getData(),
          let data: [Float] = tensor.getData()
        else {
          continue
        }

        // Initialize moment buffers on first step (or fix size if pre-filled by scalar handler)
        if m.count <= paramIdx {
          m.append([Float](repeating: 0.0, count: data.count))
          v.append([Float](repeating: 0.0, count: data.count))
        } else if m[paramIdx].count != data.count {
          m[paramIdx] = [Float](repeating: 0.0, count: data.count)
          v[paramIdx] = [Float](repeating: 0.0, count: data.count)
        }

        // Bias correction factors
        let biasCorrection1 = 1.0 - pow(beta1, Float(t))
        let biasCorrection2 = 1.0 - pow(beta2, Float(t))

        var newData = data
        for j in 0..<data.count {
          let grad = gradData[j]

          // Update biased first moment estimate
          m[paramIdx][j] = beta1 * m[paramIdx][j] + (1.0 - beta1) * grad

          // Update biased second moment estimate
          v[paramIdx][j] = beta2 * v[paramIdx][j] + (1.0 - beta2) * grad * grad

          // Compute bias-corrected estimates
          let mHat = m[paramIdx][j] / biasCorrection1
          let vHat = v[paramIdx][j] / biasCorrection2

          // Update parameter
          newData[j] = data[j] - lr * mHat / (sqrt(vHat) + eps)
        }

        tensor.updateDataLazily(newData)

      } else if let sig = parameter as? Signal {
        guard let paramValue = sig.data,
              let gradSignal = sig.grad,
              let gradValue = gradSignal.data else {
          continue
        }

        // For scalar signals, use simple index 0
        let scalarIdx = params.count + paramIdx  // Offset to avoid collision with tensor indices

        // Initialize moment buffers for scalar
        if m.count <= scalarIdx {
          while m.count <= scalarIdx {
            m.append([0.0])
            v.append([0.0])
          }
        }

        let biasCorrection1 = 1.0 - pow(beta1, Float(t))
        let biasCorrection2 = 1.0 - pow(beta2, Float(t))

        // Update moments
        m[scalarIdx][0] = beta1 * m[scalarIdx][0] + (1.0 - beta1) * gradValue
        v[scalarIdx][0] = beta2 * v[scalarIdx][0] + (1.0 - beta2) * gradValue * gradValue

        // Bias-corrected estimates
        let mHat = m[scalarIdx][0] / biasCorrection1
        let vHat = v[scalarIdx][0] / biasCorrection2

        // Update parameter
        let newValue = paramValue - lr * mHat / (sqrt(vHat) + eps)
        sig.updateDataLazily(newValue)
      }
    }
  }
}
