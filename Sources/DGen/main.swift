let g = Graph()

let ph = g.n(.phasor(0),
  g.n(.constant(430)),
  g.n(.constant(0.3)));

g.n(.mix,
    ph,
    g.n(.constant(333)),
    g.n(.constant(0.7)))
    
let sorted = topo(g)

var ctx = IRContext()

var uops: [UOp] = []
for nodeId in sorted {
    if let node = g.nodes[nodeId] {
        node.op.emit(ctx: ctx, g: g, nodeId: nodeId).forEach {
            print($0) 
        }
    }
}
