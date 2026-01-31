import Foundation

/// A learnable parameter in the computation graph
/// Wraps a CellID and tracks its value, gradients, and optimizer state
public class Parameter {
    let cellId: CellID
    let nodeId: NodeID
    public var value: Float
    let name: String?

    // Internal: set by TrainingContext after compilation
    var gradId: GradID?

    // Gradient value (updated after each backward pass)
    public var grad: Float?

    public init(graph: Graph, value: Float, name: String? = nil) {
        self.cellId = graph.alloc()
        self.value = value
        self.name = name
        self.nodeId = graph.n(.param(cellId))
        self.grad = 0.0
    }

    public func node() -> NodeID {
        return nodeId
    }
}

/// A learnable tensor parameter for graph-based training
public class TensorParameter {
    let tensorId: TensorID
    let cellId: CellID
    let nodeId: NodeID
    public let shape: Shape
    public var data: [Float]
    let name: String?
    var baseGradId: GradID?

    /// Total number of elements in the tensor
    public var size: Int { shape.reduce(1, *) }

    /// Per-element gradients (updated after each backward pass)
    public var grads: [Float]

    public init(graph: Graph, shape: Shape, data: [Float]? = nil, name: String? = nil) {
        self.shape = shape
        self.name = name
        let totalSize = shape.reduce(1, *)

        // Initialize data with Xavier initialization if not provided
        if let providedData = data {
            self.data = providedData
            precondition(providedData.count == totalSize, "Data count must match shape size")
        } else {
            // Xavier initialization: N(0, sqrt(2 / (fan_in + fan_out)))
            // For simplicity, use sqrt(1/n) where n is size
            let stddev = sqrt(1.0 / Float(totalSize))
            self.data = (0..<totalSize).map { _ in
                // Box-Muller transform for normal distribution
                let u1 = Float.random(in: 0..<1)
                let u2 = Float.random(in: 0..<1)
                return stddev * sqrt(-2.0 * log(u1)) * cos(2.0 * .pi * u2)
            }
        }

        self.grads = [Float](repeating: 0.0, count: totalSize)

        // Allocate tensor in graph
        self.cellId = graph.alloc(vectorWidth: totalSize)
        self.tensorId = graph.nextTensorId
        graph.nextTensorId += 1
        graph.tensors[tensorId] = Tensor(id: tensorId, shape: shape, cellId: cellId, data: self.data)
        graph.cellToTensor[cellId] = tensorId

        // Create tensorRef node
        self.nodeId = graph.n(.tensorRef(tensorId), [], shape: .tensor(shape))
        graph.nodeToTensor[nodeId] = tensorId
    }

    public func node() -> NodeID {
        return nodeId
    }
}
