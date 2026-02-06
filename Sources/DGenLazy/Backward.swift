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

    guard let memBuffer = context.runtime.getBuffer(name: "memory") else {
      return (tensorGrads, signalGrads)
    }
    let memPtr = memBuffer.contents().assumingMemoryBound(to: Float.self)

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
    // Set up gradient computation before adding output
    let _ = graph.setupGradients(loss: nodeId, frameCount: frameCount)

    // Create a single output(0) node. If there are gradient side effects
    // (e.g. tensorAccumulate), chain them via seq so they execute,
    // but the output still carries the loss value through the seq chain.
    if !graph.graph.gradientSideEffects.isEmpty {
      let chainedValue = graph.graph.chainGradientSideEffects(after: nodeId)
      let _ = graph.graph.n(.output(0), [chainedValue])
    } else {
      let _ = graph.node(.output(0), [nodeId])
    }
    graph.graph.gradientSideEffects = []

    // Compile and run
    let context = try graph.compile(frameCount: frameCount)
    graph.run(context: context, preserveState: false)

    // Read gradients
    let (tensorGrads, signalGrads) = graph.readGradients(context: context)

    // Populate .grad properties (use internal init - gradients are data-only, not graph participants)
    for tensor in graph.parameterRegistry.tensors {
      if let grads = tensorGrads[tensor.nodeId] {
        tensor.grad = Tensor(nodeId: -1, graph: graph, shape: [grads.count], data: grads)
      }
    }

    for signal in graph.parameterRegistry.signals {
      if let gradValue = signalGrads[signal.nodeId] {
        signal.grad = Signal(nodeId: -1, graph: graph, data: gradValue)
      }
    }

    // Read loss values from outputs before clearing
    let lossValues = graph.readOutputs(context: context)

    // Clear graph to prevent node accumulation - tensors refresh automatically
    graph.clearComputationGraph()

    return lossValues
  }
}

// MARK: - Backward for Signal

extension Signal {
  /// Compute gradients for all signals with requiresGrad=true
  /// - Returns: The computed loss values (one per frame)
  @discardableResult
  public func backward(frames: Int = DGenConfig.defaultFrameCount) throws -> [Float] {
    // Set up gradient computation before adding output
    let _ = graph.setupGradients(loss: nodeId, frameCount: frames)

    // Create a single output(0) node. If there are gradient side effects
    // (e.g. memoryAccumulate), chain them via seq so they execute,
    // but the output still carries the loss value through the seq chain.
    if !graph.graph.gradientSideEffects.isEmpty {
      let chainedValue = graph.graph.chainGradientSideEffects(after: nodeId)
      let _ = graph.graph.n(.output(0), [chainedValue])
    } else {
      let _ = graph.node(.output(0), [nodeId])
    }
    graph.graph.gradientSideEffects = []

    // Compile and run
    let context = try graph.compile(frameCount: frames)
    graph.run(context: context, preserveState: false)

    // Read gradients
    let (tensorGrads, signalGrads) = graph.readGradients(context: context)

    // Populate .grad properties (use internal init - gradients are data-only, not graph participants)
    for tensor in graph.parameterRegistry.tensors {
      if let grads = tensorGrads[tensor.nodeId] {
        tensor.grad = Tensor(nodeId: -1, graph: graph, shape: [grads.count], data: grads)
      }
    }

    for signal in graph.parameterRegistry.signals {
      if let gradValue = signalGrads[signal.nodeId] {
        signal.grad = Signal(nodeId: -1, graph: graph, data: gradValue)
      }
    }

    // Read loss values from outputs before clearing
    let lossValues = graph.readOutputs(context: context)

    // Clear graph to prevent node accumulation - signals refresh automatically
    graph.clearComputationGraph()

    return lossValues
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
