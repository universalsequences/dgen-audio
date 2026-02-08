// Signal - Lazy frame-based scalar
//
// A scalar value that varies per audio frame (sample). Used for audio
// signal processing with automatic differentiation support.

import DGen

// MARK: - Signal

/// A lazy frame-based scalar signal
///
/// Signals represent values that vary per audio frame. They're the building
/// blocks for audio signal processing in DGenLazy.
///
/// ```swift
/// let freq = Signal.param(440.0)
/// let osc = Signal.phasor(freq)
/// let audio = sin(osc * 2 * .pi)
/// let samples = audio.realize(frames: 1024)
/// ```
public class Signal: LazyValue {
  // MARK: - Properties

  /// The underlying node ID in the lazy graph (mutable for lazy recreation after graph clear)
  private var _nodeId: NodeID

  /// The graph this signal belongs to
  public let graph: LazyGraph

  /// Whether this signal requires gradient computation
  public let requiresGrad: Bool

  /// Gradient signal (populated after backward())
  public var grad: Signal?

  public var data: Float?

  /// Memory cell for stateful signals (phasor, history, etc.)
  internal var cellId: CellID?

  /// Track which graph generation this signal belongs to
  private var graphId: Int = -1

  /// Computed property for nodeId
  public var nodeId: NodeID {
    return _nodeId
  }

  /// Refresh signal after graph clear - recreates node in the fresh graph
  public func refresh() {
    let graph = LazyGraphContext.current
    guard let value = data else { return }

    if graph.id != graphId {
      graphId = graph.id
      if requiresGrad {
        // Param signal - allocate new cell and create param node
        let cell = graph.alloc()
        cellId = cell
        _nodeId = graph.node(.param(cell), [])
      } else {
        // Constant signal - just create constant node
        _nodeId = graph.node(.constant(value), [])
      }
    }
  }

  // MARK: - Initializers

  /// Create a constant signal
  /// - Parameters:
  ///   - value: The constant value
  ///   - requiresGrad: Whether to compute gradients
  public init(_ value: Float, requiresGrad: Bool = false) {
    let graph = LazyGraphContext.current

    // For learnable signals, use a param node with memory cell
    let nodeId: NodeID
    let cellId: CellID?
    if requiresGrad {
      let cell = graph.alloc()
      nodeId = graph.node(.param(cell))
      cellId = cell
    } else {
      nodeId = graph.node(.constant(value))
      cellId = nil
    }

    self._nodeId = nodeId
    self.graph = graph
    self.requiresGrad = requiresGrad
    self.cellId = cellId
    self.grad = nil
    self.data = value  // Always store the value for later access
    self.graphId = graph.id  // Track which graph generation we belong to

    // Auto-register for gradient tracking
    if requiresGrad {
      graph.registerParameter(self)
    }
    graph.registerSignal(self)  // Track for refresh after clear
  }

  /// Internal initializer for creating signals from operations
  internal init(
    nodeId: NodeID, graph: LazyGraph, requiresGrad: Bool = false, cellId: CellID? = nil,
    data: Float? = nil, graphId: Int = -1
  ) {
    self._nodeId = nodeId
    self.graph = graph
    self.requiresGrad = requiresGrad
    self.cellId = cellId
    self.grad = nil
    self.data = data
    self.graphId = graphId
  }

  // MARK: - Factory Methods

  /// Create a constant signal (explicit factory)
  public static func constant(_ value: Float) -> Signal {
    return Signal(value, requiresGrad: false)
  }

  /// Optional bounds for parameter clamping (applied after optimizer step)
  public var minBound: Float?
  public var maxBound: Float?

  /// Create a learnable parameter signal
  /// - Parameters:
  ///   - value: Initial value
  ///   - min: Optional lower bound (clamped after each optimizer step)
  ///   - max: Optional upper bound (clamped after each optimizer step)
  public static func param(_ value: Float, min: Float? = nil, max: Float? = nil) -> Signal {
    let graph = LazyGraphContext.current
    let cellId = graph.alloc()

    let nodeId = graph.node(.param(cellId), [])
    let signal = Signal(nodeId: nodeId, graph: graph, requiresGrad: true, cellId: cellId, data: value, graphId: graph.id)
    signal.minBound = min
    signal.maxBound = max
    graph.registerParameter(signal)  // Register for gradient tracking
    graph.registerSignal(signal)  // Track for refresh after clear
    return signal
  }

  /// Update the signal's parameter data for the next forward pass
  /// Used by optimizers to apply parameter updates
  public func updateDataLazily(_ newValue: Float) {
    var clamped = newValue
    if let lo = minBound { clamped = Swift.max(clamped, lo) }
    if let hi = maxBound { clamped = Swift.min(clamped, hi) }
    self.data = clamped
    graph.markDirty()  // Invalidate caches so next realize uses new data
  }

  /// Create an audio input signal
  /// - Parameter channel: Input channel index (default: 0)
  public static func input(_ channel: Int = 0) -> Signal {
    let graph = LazyGraphContext.current
    let nodeId = graph.node(.input(channel))
    return Signal(nodeId: nodeId, graph: graph, requiresGrad: false)
  }

  /// Create a phasor (ramp oscillator)
  /// - Parameters:
  ///   - freq: Frequency in Hz (can be Signal, Tensor, or Float)
  ///   - reset: Optional reset trigger
  /// - Returns: Signal (if freq is scalar/Signal) or SignalTensor (if freq is Tensor)
  public static func phasor(_ freq: Signal, reset: Signal? = nil) -> Signal {
    let graph = freq.graph
    let cellId = graph.alloc()

    let resetNode = reset?.nodeId ?? graph.node(.constant(0.0))
    let nodeId = graph.node(.phasor(cellId), [freq.nodeId, resetNode])

    // Propagate requiresGrad from inputs
    let needsGrad = freq.requiresGrad || (reset?.requiresGrad ?? false)
    return Signal(nodeId: nodeId, graph: graph, requiresGrad: needsGrad, cellId: cellId)
  }

  /// Create a phasor with a constant frequency
  public static func phasor(_ freq: Float, reset: Signal? = nil) -> Signal {
    return phasor(Signal.constant(freq), reset: reset)
  }

  /// Create a phasor with a tensor of frequencies (returns SignalTensor)
  public static func phasor(_ freqs: Tensor, reset: Signal? = nil) -> SignalTensor {
    return SignalTensor.phasor(freqs, reset: reset)
  }

  /// Create a white noise signal
  public static func noise() -> Signal {
    let graph = LazyGraphContext.current
    let cellId = graph.alloc()
    let nodeId = graph.node(.noise(cellId))
    return Signal(nodeId: nodeId, graph: graph, requiresGrad: false)
  }

  /// Create a click/impulse signal (outputs 1.0 on first frame, then 0.0)
  public static func click() -> Signal {
    let graph = LazyGraphContext.current
    let cellId = graph.alloc()
    let nodeId = graph.node(.click(cellId))
    return Signal(nodeId: nodeId, graph: graph, requiresGrad: false)
  }

  // MARK: - Stateful Operations

  /// Create a history cell and return a reader for it
  /// Use this to build custom feedback loops
  /// - Returns: Tuple of (read: Signal, write: (Signal) -> Void)
  public static func history() -> (read: Signal, write: (Signal) -> Void) {
    let graph = LazyGraphContext.current
    let cellId = graph.alloc()

    let readNode = graph.node(.historyRead(cellId))
    let readSignal = Signal(nodeId: readNode, graph: graph, requiresGrad: false, cellId: cellId)

    let writeFunc: (Signal) -> Void = { value in
      let _ = graph.node(.historyWrite(cellId), [value.nodeId])
    }

    return (read: readSignal, write: writeFunc)
  }

  /// Create an accumulator
  /// - Parameters:
  ///   - increment: Value to add each frame
  ///   - reset: Reset trigger (resets to min when > 0)
  ///   - min: Minimum value (wraps to this after max)
  ///   - max: Maximum value (wraps to min after reaching this)
  /// - Returns: Accumulated value
  public static func accum(
    _ increment: Signal,
    reset: Signal? = nil,
    min: Signal? = nil,
    max: Signal? = nil
  ) -> Signal {
    let graph = increment.graph
    let cellId = graph.alloc()

    let resetNode = reset?.nodeId ?? graph.node(.constant(0.0))
    let minNode = min?.nodeId ?? graph.node(.constant(0.0))
    let maxNode = max?.nodeId ?? graph.node(.constant(1.0))

    let nodeId = graph.node(.accum(cellId), [increment.nodeId, resetNode, minNode, maxNode])

    let needsGrad =
      increment.requiresGrad || (reset?.requiresGrad ?? false) || (min?.requiresGrad ?? false)
      || (max?.requiresGrad ?? false)

    return Signal(nodeId: nodeId, graph: graph, requiresGrad: needsGrad, cellId: cellId)
  }

  /// Accumulator with Float parameters
  public static func accum(
    _ increment: Signal,
    reset: Float = 0.0,
    min: Float = 0.0,
    max: Float = 1.0
  ) -> Signal {
    return accum(
      increment,
      reset: Signal.constant(reset),
      min: Signal.constant(min),
      max: Signal.constant(max)
    )
  }

  /// Create a latch (sample-and-hold)
  /// Captures value when condition becomes true
  /// - Parameters:
  ///   - value: Value to sample
  ///   - condition: When > 0, captures the current value
  /// - Returns: Latched value
  public static func latch(_ value: Signal, when condition: Signal) -> Signal {
    let graph = value.graph
    let cellId = graph.alloc()

    let nodeId = graph.node(.latch(cellId), [value.nodeId, condition.nodeId])
    let needsGrad = value.requiresGrad || condition.requiresGrad

    return Signal(nodeId: nodeId, graph: graph, requiresGrad: needsGrad, cellId: cellId)
  }

  /// Linear interpolation between two signals
  /// - Parameters:
  ///   - a: Start value (returned when t = 0)
  ///   - b: End value (returned when t = 1)
  ///   - t: Interpolation factor (0 to 1)
  /// - Returns: Interpolated value: a * (1-t) + b * t
  public static func mix(_ a: Signal, _ b: Signal, _ t: Signal) -> Signal {
    let graph = a.graph
    let nodeId = graph.node(.mix, [a.nodeId, b.nodeId, t.nodeId])
    let needsGrad = a.requiresGrad || b.requiresGrad || t.requiresGrad
    return Signal(nodeId: nodeId, graph: graph, requiresGrad: needsGrad)
  }

  /// Mix with Float interpolation factor
  public static func mix(_ a: Signal, _ b: Signal, _ t: Float) -> Signal {
    return mix(a, b, Signal.constant(t))
  }

  // MARK: - Buffer

  /// Buffer the last N samples of this signal into a [1, N] tensor.
  /// Zero-copy ring buffer — the tensor IS the buffer. Composes with conv2d, sum, etc.
  /// - Parameters:
  ///   - size: Number of samples to buffer
  ///   - hop: If specified, downstream ops only execute every `hop` frames
  /// - Returns: SignalTensor of shape [1, size] backed by a ring buffer
  public func buffer(size: Int, hop: Int? = nil) -> SignalTensor {
    let nodeId = graph.graph.bufferView(self.nodeId, size: size, hopSize: hop)
    return SignalTensor(
      nodeId: nodeId, graph: graph, shape: [1, size],
      requiresGrad: self.requiresGrad)
  }
}
