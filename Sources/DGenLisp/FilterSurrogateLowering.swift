// FilterSurrogateLowering.swift — replace the sample-serial patch SVF with
// its frequency-sampled training surrogate. Rendering uses a separate AST.

enum FilterSurrogateLowering {
    static func lower(nodes: [ASTNode], window: Int, hop: Int) -> [ASTNode] {
        nodes.map { rewrite($0, window: window, hop: hop) }
    }

    private static func rewrite(_ node: ASTNode, window: Int, hop: Int) -> ASTNode {
        guard case .list(let elements) = node, !elements.isEmpty else { return node }

        // Keep the patch's declaration intact. Calls are rewritten before
        // evaluation, so this macro simply becomes dead code.
        if elements.count >= 2,
           case .atom("defmacro") = elements[0],
           case .atom("svf") = elements[1]
        {
            return node
        }

        let children = elements.map { rewrite($0, window: window, hop: hop) }
        guard case .atom("svf") = children[0] else { return .list(children) }
        return .list(
            [.atom("svf-freq")] + Array(children.dropFirst())
            + [.atom("@window"), .atom(String(window)), .atom("@hop"), .atom(String(hop))])
    }
}
