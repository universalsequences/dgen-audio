// GradientSetup.swift - Shared gradient setup utilities
//
// Provides reusable methods for setting up gradient accumulation that both
// GraphTrainingContext and DGenLazy can use, ensuring consistent behavior
// and avoiding code duplication.

import Foundation

// MARK: - Graph Extension for Gradient Setup

extension Graph {

    /// Set up gradient accumulation for tensor parameters.
    ///
    /// Creates gradient accumulation cells and `tensorAccumulate` ops for each tensor
    /// that has a computed gradient.
    ///
    /// - Parameters:
    ///   - gradients: Dictionary mapping forward node IDs to their gradient node IDs
    ///   - tensorNodes: List of (nodeId, size) tuples for tensors needing gradients
    /// - Returns: Dictionary mapping nodeId -> gradient cell ID
    public func setupTensorGradients(
        gradients: [NodeID: NodeID],
        tensorNodes: [(nodeId: NodeID, size: Int)]
    ) -> [NodeID: CellID] {
        var result: [NodeID: CellID] = [:]

        for (nodeId, size) in tensorNodes {
            guard let gradNode = gradients[nodeId] else {
                continue
            }

            // Allocate gradient accumulation cell
            let gradCell = alloc(vectorWidth: size)
            tensorGradCells[nodeId] = gradCell
            result[nodeId] = gradCell

            // Create tensor accumulate op - this pulls gradNode into the graph
            let accumOp = n(.tensorAccumulate(gradCell), [gradNode])
            addGradientSideEffect(accumOp)
        }

        return result
    }

    /// Set up gradient accumulation for scalar parameters.
    ///
    /// Creates gradient accumulation cells and `memoryAccumulate` ops for each scalar
    /// parameter that has a computed gradient.
    ///
    /// - Parameters:
    ///   - gradients: Dictionary mapping forward node IDs to their gradient node IDs
    ///   - scalarNodes: List of node IDs for scalar parameters needing gradients
    /// - Returns: Dictionary mapping nodeId -> gradient cell ID
    public func setupScalarGradients(
        gradients: [NodeID: NodeID],
        scalarNodes: [NodeID]
    ) -> [NodeID: CellID] {
        let zero = n(.constant(0.0))
        var result: [NodeID: CellID] = [:]

        for nodeId in scalarNodes {
            guard let gradNode = gradients[nodeId] else {
                continue
            }

            // Allocate gradient accumulation cell (scalar = 1 cell)
            let gradCell = alloc()
            result[nodeId] = gradCell

            // Atomic accumulation for per-frame gradients
            // memoryAccumulate(cell) takes (offset, value) - offset=0 for scalar
            let accumOp = n(.memoryAccumulate(gradCell), [zero, gradNode])
            addGradientSideEffect(accumOp)
        }

        return result
    }

    /// Chain all gradient side effects to ensure they execute before the given value node.
    ///
    /// This is used to ensure gradient accumulation operations are scheduled as part
    /// of the computation graph by making the output depend on them via `seq` nodes.
    ///
    /// - Parameter valueNode: The node whose value should be returned after side effects
    /// - Returns: A new node that sequences all side effects before returning valueNode
    public func chainGradientSideEffects(after valueNode: NodeID) -> NodeID {
        guard !gradientSideEffects.isEmpty else {
            return valueNode
        }

        var chainedValue = valueNode
        for sideEffect in gradientSideEffects {
            chainedValue = n(.seq, [sideEffect, chainedValue])
        }
        return chainedValue
    }
}
