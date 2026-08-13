// GraphIntrospection.swift — read-only analysis queries for training
// lowering policy (dgenlisp train plan event).
//
// Works on the realized lazy graph, after lisp evaluation, so it sees
// through defs and macro expansion: a phasor buried three macros deep
// still reports which param cells feed its frequency input.

import DGen
import Foundation

public struct PhasorFrequencyAnalysis {
    /// Cell ids of `.param` nodes that are ancestors of any phasor node's
    /// frequency input. Training must freeze these (swept-f0 /
    /// trainable-statefulPhasor-frequency adjoints are unreliable).
    public let frequencyParamCells: Set<CellID>
    /// Phasor nodes whose reset input is driven by another oscillator
    /// (oscillator sync — a declared non-goal; refuse to train).
    public let syncPhasorNodeIds: [NodeID]
}

extension LazyGraph {
    public func analyzePhasorFrequencies() -> PhasorFrequencyAnalysis {
        var freqCells: Set<CellID> = []
        var syncNodes: [NodeID] = []
        for (id, node) in graph.nodes {
            guard case .phasor = node.op else { continue }
            if node.inputs.count >= 1 {
                freqCells.formUnion(paramCellAncestors(of: node.inputs[0]))
            }
            // inputs[1] is the reset; a reset that itself derives from an
            // oscillator is hard sync. Trigger/click/constant resets are
            // ordinary voice behavior and stay supported.
            if node.inputs.count >= 2, ancestorsContainPhasor(of: node.inputs[1]) {
                syncNodes.append(id)
            }
        }
        return PhasorFrequencyAnalysis(
            frequencyParamCells: freqCells,
            syncPhasorNodeIds: syncNodes.sorted())
    }

    private func paramCellAncestors(of root: NodeID) -> Set<CellID> {
        var cells: Set<CellID> = []
        visitAncestors(of: root) { node in
            if case .param(let cell) = node.op {
                cells.insert(cell)
            }
        }
        return cells
    }

    private func ancestorsContainPhasor(of root: NodeID) -> Bool {
        var found = false
        visitAncestors(of: root) { node in
            if case .phasor = node.op { found = true }
        }
        return found
    }

    private func visitAncestors(of root: NodeID, _ visit: (Node) -> Void) {
        var stack = [root]
        var seen: Set<NodeID> = []
        while let id = stack.popLast() {
            guard seen.insert(id).inserted, let node = graph.nodes[id] else { continue }
            visit(node)
            stack.append(contentsOf: node.allDependencies)
        }
    }
}
