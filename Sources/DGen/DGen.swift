import Foundation

public class IRContext {
    private var varIdx = 0
    private var constantIdx = 0
    
    public init() {}

    public var globals: Set<VarID> = []

    // map of nodeId -> Lazy value (variable or constant)
    public var values: [NodeID: Lazy] = [:]
    public var constants: [ConstantID: Float] = [:]
    public var variables: [VarID: NodeID] = [:]

    public func useConstant(src: NodeID?, value: Float) -> Lazy {
        let constantId = self.constantIdx + 1
        self.constantIdx = constantId;
        self.constants[constantId] = value
        let constant = Lazy.constant(constantId, value)
        if let srcId = src {
            self.values[srcId] = constant
        }
        return constant
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

public struct Node { public let id: NodeID; public let op: LazyOp; public let inputs: [NodeID] }

public final class Graph {
    private var next = 0; public var nodes: [NodeID: Node] = [:]

    public init() {}

    @discardableResult public func n(_ op: LazyOp, _ ins: NodeID...) -> NodeID {
        let id = next; next += 1; nodes[id] = Node(id: id, op: op, inputs: ins); return id
    }
}

public extension Lazy {
    var varId: VarID? {
        switch self {
        case let .variable(id, _):
            return id
        default:
            return nil
        }
    }
}

public extension Op {
    var operands: [Lazy] {
        switch self {
        case let .add(a, b):
            return [a, b]
        case let .mul(a, b):
            return [a, b]
        case let .sub(a, b):
            return [a, b]
        case let .div(a, b):
            return [a, b]
        case let .gt(a, b):
            return [a, b]
        case let .lt(a, b):
            return [a, b]
        case let .store(a, b):
            return [b]
        case let .load(a):
            return []
        case let .beginIf(a):
            return [a]
        case let .mutate(a, b):
            return [a, b]
        default:
            return []
        }
    }
}

