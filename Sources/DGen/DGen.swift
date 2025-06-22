import Foundation;

public typealias NodeID = Int;
public typealias VarID = Int;
public typealias CellID = Int;

public enum Lazy {
    case constant(Float)
    case variable(VarID, NodeID?)
}

public class IRContext {
    private var varIdx = 0

    // map of node -> variable
    public var values: [NodeID: Lazy] = [:]

    public func useConstant(src: NodeID, value: Float) -> Lazy {
        let constant = Lazy.constant(value)
        self.values[src] = constant
        return constant
    }
    
    public func useVariable(src: NodeID?, trackInValues: Bool = true) -> Lazy {
        let varId = self.varIdx + 1
        self.varIdx = varId
        let variable = Lazy.variable(varId, src)
        if let srcNodeId = src, trackInValues {
            self.values[srcNodeId] = variable 
        }
        return variable
    }
}

public struct Node { public let id: NodeID; public let op: LazyOp; public let inputs: [NodeID] }

public final class Graph {
    private var next = 0; public var nodes: [NodeID: Node] = [:]

    public init() {}

    @discardableResult public func n(_ op: LazyOp, _ ins: NodeID...) -> NodeID {
        let id = next; next += 1; nodes[id] = Node(id: id, op: op, inputs: ins); return id
    }
}

