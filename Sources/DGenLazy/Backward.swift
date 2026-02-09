// Backward - Gradient computation for lazy types
//
// Implements backward() for automatic differentiation using
// DGen's computeGradients infrastructure.

import DGen
import Foundation

// MARK: - Parameter Registry

/// Tracks parameters that need gradients in the current graph
class ParameterRegistry {
  /// All tensors with requiresGrad=true
  var tensors: [Tensor] = []

  /// All signals with requiresGrad=true
  var signals: [Signal] = []

  /// Gradient accumulation cells for tensors (nodeId -> cellId)
  var tensorGradCells: [NodeID: CellID] = [:]

  /// Gradient accumulation cells for signals (nodeId -> cellId)
  var signalGradCells: [NodeID: CellID] = [:]

  func register(_ tensor: Tensor) {
    if tensor.requiresGrad && !tensors.contains(where: { $0.nodeId == tensor.nodeId }) {
      tensors.append(tensor)
    }
  }

  func register(_ signal: Signal) {
    if signal.requiresGrad && !signals.contains(where: { $0.nodeId == signal.nodeId }) {
      signals.append(signal)
    }
  }

  func clear() {
    tensors.removeAll()
    signals.removeAll()
    tensorGradCells.removeAll()
    signalGradCells.removeAll()
  }
}

// MARK: - LazyGraph Extensions for Backward

extension LazyGraph {
  /// Registry for parameters that need gradients
  private static var _registry = ParameterRegistry()

  var parameterRegistry: ParameterRegistry {
    return LazyGraph._registry
  }

  /// Register a parameter for gradient computation
  func registerParameter(_ tensor: Tensor) {
    parameterRegistry.register(tensor)
  }

  func registerParameter(_ signal: Signal) {
    parameterRegistry.register(signal)
  }

  /// Compute gradients for all registered parameters
  /// Returns the gradient accumulation cells
  func setupGradients(loss: NodeID, frameCount: Int) -> (
    tensorCells: [NodeID: CellID], signalCells: [NodeID: CellID]
  ) {
    let registry = parameterRegistry

    // Collect all target nodes
    var targets = Set<NodeID>()
    for tensor in registry.tensors {
      targets.insert(tensor.nodeId)
    }
    for signal in registry.signals {
      targets.insert(signal.nodeId)
    }

    guard !targets.isEmpty else {
      return ([:], [:])
    }

    // Record the last forward node ID before adding gradient nodes
    // This is needed for BPTT loop splitting (forward loop vs reverse backward loop)
    graph.lastForwardNodeId = graph.nodes.keys.max() ?? 0

    // Compute gradients using DGen's gradient infrastructure
    let gradients = graph.computeGradients(loss: loss, targets: targets)

    // Use shared gradient setup methods from GradientSetup.swift
    let tensorNodes = registry.tensors.map { ($0.nodeId, $0.size) }
    let tensorGradCells = graph.setupTensorGradients(
      gradients: gradients,
      tensorNodes: tensorNodes
    )

    let signalNodes = registry.signals.map { $0.nodeId }
    let signalGradCells = graph.setupScalarGradients(
      gradients: gradients,
      scalarNodes: signalNodes
    )

    // Store in registry for later retrieval
    registry.tensorGradCells = tensorGradCells
    registry.signalGradCells = signalGradCells

    return (tensorGradCells, signalGradCells)
  }

  /// Read gradient values from memory after execution
  func readGradients(context: ExecutionContext) -> (
    tensorGrads: [NodeID: [Float]], signalGrads: [NodeID: Float]
  ) {
    let registry = parameterRegistry
    var tensorGrads: [NodeID: [Float]] = [:]
    var signalGrads: [NodeID: Float] = [:]

    guard let memPtr = context.runtime.memoryPointer() else {
      return (tensorGrads, signalGrads)
    }

    // Read tensor gradients
    for tensor in registry.tensors {
      guard let gradCell = registry.tensorGradCells[tensor.nodeId] else { continue }
      let physicalCell =
        context.compilationResult.cellAllocations.cellMappings[gradCell] ?? gradCell

      var grads = [Float](repeating: 0, count: tensor.size)
      for i in 0..<tensor.size {
        grads[i] = memPtr[physicalCell + i]
      }
      tensorGrads[tensor.nodeId] = grads
    }

    // Read signal gradients
    for signal in registry.signals {
      guard let gradCell = registry.signalGradCells[signal.nodeId] else { continue }
      let physicalCell =
        context.compilationResult.cellAllocations.cellMappings[gradCell] ?? gradCell
      signalGrads[signal.nodeId] = memPtr[physicalCell]
    }

    return (tensorGrads, signalGrads)
  }
}

// MARK: - Shared Backward Logic

extension LazyGraph {
  /// Core backward pass: set up gradients, compile, run, populate .grad, and return loss values.
  ///
  /// Both `Tensor.backward()` and `Signal.backward()` delegate here to avoid duplication.
  func runBackward(loss: NodeID, frameCount: Int) throws -> [Float] {
    markDirty()
    _ = setupGradients(loss: loss, frameCount: frameCount)

    // Create a single output(0) node. If there are gradient side effects
    // (e.g. tensorAccumulate/memoryAccumulate), chain them via seq so they
    // execute, but the output still carries the loss value through the chain.
    if graph.gradientSideEffects.isEmpty {
      _ = graph.n(.output(0), [loss])
    } else {
      let chainedValue = graph.chainGradientSideEffects(after: loss)
      _ = graph.n(.output(0), [chainedValue])
      graph.gradientSideEffects = []
    }

    // Wrap compilation and execution in autoreleasepool to ensure Metal objects
    // (MTLBuffer, MTLLibrary, MTLComputePipelineState, etc.) are freed each
    // iteration. Without this, autoreleased Obj-C objects from Metal APIs
    // accumulate across training epochs, causing multi-GB memory growth.
    var lossValues: [Float] = []
    try autoreleasepool {
      let context = try compile(frameCount: frameCount)
      run(context: context, preserveState: false)

      let (tensorGrads, signalGrads) = readGradients(context: context)

      // Populate .grad properties (gradients are data-only, not graph participants)
      for tensor in parameterRegistry.tensors {
        if let grads = tensorGrads[tensor.nodeId] {
          tensor.grad = Tensor(nodeId: -1, graph: self, shape: [grads.count], data: grads)
        }
      }
      for signal in parameterRegistry.signals {
        if let gradValue = signalGrads[signal.nodeId] {
          signal.grad = Signal(nodeId: -1, graph: self, data: gradValue)
        }
      }

      lossValues = readOutputs(context: context)
      clearComputationGraph()
      // context goes out of scope here → MetalCompiledKernel freed →
      // autoreleasepool drains → underlying Metal objects released
    }
    return lossValues
  }
}

// MARK: - Backward for Tensor

extension Tensor {
  /// Compute gradients for all tensors with requiresGrad=true
  ///
  /// After calling backward(), access gradients via the `.grad` property.
  /// The graph is automatically cleared after backward to prevent node accumulation.
  ///
  /// ```swift
  /// let w = Tensor.randn([64, 32], requiresGrad: true)
  /// let loss = mse(pred, target)
  /// let lossValue = try loss.backward()  // Returns computed loss
  /// print(w.grad)  // Gradient tensor
  /// optimizer.step()
  /// optimizer.zeroGrad()
  /// // Graph cleared - ready for next iteration
  /// ```
  ///
  /// - Returns: The computed loss values (one per frame)
  @discardableResult
  public func backward(frameCount: Int = DGenConfig.defaultFrameCount) throws -> [Float] {
    return try graph.runBackward(loss: nodeId, frameCount: frameCount)
  }
}

// MARK: - Backward for Signal

extension Signal {
  /// Compute gradients for all signals with requiresGrad=true
  /// - Returns: The computed loss values (one per frame)
  @discardableResult
  public func backward(frames: Int = DGenConfig.defaultFrameCount) throws -> [Float] {
    return try graph.runBackward(loss: nodeId, frameCount: frames)
  }
}

// MARK: - Auto-registration of parameters

extension Tensor {
  /// Called when a tensor with requiresGrad is used in an operation
  func registerIfNeeded() {
    if requiresGrad {
      graph.registerParameter(self)
    }
  }
}

extension Signal {
  /// Called when a signal with requiresGrad is used in an operation
  func registerIfNeeded() {
    if requiresGrad {
      graph.registerParameter(self)
    }
  }
}
