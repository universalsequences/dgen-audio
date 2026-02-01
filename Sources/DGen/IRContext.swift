import Foundation

public class IRContext {
  public var g: Graph
  private var varIdx = 0
  private var gradIdx = 0
  private var constantIdx = 0

  // Maps node ids to what the current "index" is for tensor-based computation
  // E.g we are on the 2nd element in tensor in a loop
  public var tensorIndices: [NodeID: Lazy] = [:]

  // Maximum gradient ID allocated (for buffer sizing)
  public var maxGradId: Int { return gradIdx }
  // Reuse constant IDs for identical values to reduce duplicate vdupq constants
  private var constantIdByValue: [Float: ConstantID] = [:]

  // Tensor register optimization:
  // - outboundTensorCells: tensor cells that must be written to memory (needed by later blocks)
  // - tensorCellToVar: maps cell IDs to computed Lazy values (register variables) within current block
  // This allows intermediate tensor values to stay in registers instead of going through memory.
  public var outboundTensorCells: Set<CellID> = []
  public var tensorCellToVar: [CellID: Lazy] = [:]

  // Frame-based nodes from temporality analysis (set during compilation)
  public var frameBasedNodes: Set<NodeID> = []

  // Nodes that are part of frame-dependent tensor chains (for SIMD-across-frames optimization)
  public var frameTensorChainNodes: Set<NodeID> = []

  // Scratch buffers for frame-dependent tensor chain reductions (sum)
  public struct FrameTensorChainScratch {
    public let cellId: CellID
    public let tensorSize: Int
  }
  public var frameTensorChainScratch: [NodeID: FrameTensorChainScratch] = [:]

  // Frame-aware tensor nodes: tensors with outbound dependencies in frame-based blocks
  // These get tensorSize × frameCount memory allocation
  public var frameAwareTensorNodes: Set<NodeID> = []
  public var frameAwareTensorCells: Set<CellID> = []

  // Frame-aware tensor block context:
  // When true, we're in a frame-aware tensor block with flat threading (frameCount × tensorSize threads).
  // In this mode, parallelRange should use the pre-computed element index instead of emitting loops.
  public var isInFrameAwareTensorBlock: Bool = false
  // The pre-computed element index for the current frame-aware tensor block
  public var frameAwareTensorElementIndex: Lazy?
  // The pre-computed frame index for the current frame-aware tensor block
  public var frameAwareTensorFrameIndex: Lazy?

  /// Clear tensor register tracking (call at start of each tensor block)
  public func clearTensorRegisters() {
    tensorCellToVar = [:]
  }

  /// Check if a node is part of a frame-dependent tensor chain
  public func isPartOfFrameTensorChain(_ nodeId: NodeID) -> Bool {
    return frameTensorChainNodes.contains(nodeId)
  }

  public init(g: Graph) {
    self.g = g
  }

  // Use Array instead of Set to maintain stable ordering for tape slot assignment
  public var globals: [VarID] = []

  // map of nodeId -> Lazy value (variable or constant)
  public var values: [NodeID: Lazy] = [:]
  public var gradients: [NodeID: GradID] = [:]
  public var constants: [ConstantID: Float] = [:]
  public var variables: [VarID: NodeID] = [:]
  public var tapeIndex: [NodeID: Int] = [:]
  public var seedGradients: [GradID] = []

  // Tensor gradient support: maps tensor nodes to base GradID for contiguous allocation
  public var tensorGradients: [NodeID: GradID] = [:]
  public var tensorGradientSizes: [NodeID: Int] = [:]

  // Track which tensor gradients are frame-based (need frameCount multiplier)
  // GradIDs not in this set are static (only need tensor size)
  public var frameBasedGradients: Set<GradID> = []

  // Track scalar gradients that are frame-based
  public var frameBasedScalarGradients: Set<GradID> = []

  // Track tensor gradients that are frame-aware (need tensorSize × frameCount allocation)
  // Used for gradients of tensors with outbound dependencies in frame-based blocks
  public var frameAwareTensorGradients: Set<GradID> = []

  public func getGlobalId(_ varId: VarID) -> Int {
    if let index = globals.firstIndex(of: varId) {
      return index
    }
    return 0
  }

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

  public func useGradient(src: NodeID, seed: Bool = false) -> GradID {
    if let gradId = self.gradients[src] {
      return gradId
    }
    let gradId = self.gradIdx + 1
    self.gradIdx = gradId
    self.gradients[src] = gradId
    // Auto-detect if this node is frame-based from temporality analysis
    if frameBasedNodes.contains(src) {
      self.frameBasedScalarGradients.insert(gradId)
    }
    if seed {
      self.seedGradients.append(gradId)
    }
    return gradId
  }

  /// Current layout: gradients[(gradId) * frameCount + threadIndex]
  /// So total buffer size = (maxGradId + 1) * frameCount
  public func computeGradientBufferSize(frameCount: Int) -> Int {
    let currentSize = (maxGradId + 1) * frameCount
    return 2 * currentSize  // 2x for safety margin
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
