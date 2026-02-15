# Simplification: Defer `block.tensorIndex` Minting to Emission

## Status: Proposal

## Problem

`block.tensorIndex` (a `Lazy?` on `Block`) is a variable ID that represents "which
tensor element is the current execution processing." It's minted in **BlockFormation**
but only meaningfully bound in **BlockEmission** — and 3 of 4 binding paths ignore the
minted value entirely, creating their own.

This makes the lifecycle harder to follow than it needs to be: a variable is allocated
in one compilation phase, conditionally used or discarded in another, and the reader has
to trace through multiple code paths to figure out which value actually ends up in
`ctx.tensorIndices[nodeId]`.

## Current Lifecycle

```
Phase 1 — BlockFormation (Minting)
    assignTensorIndexFromFirstTensorNode()
    block.tensorIndex = ctx.useVariable(src: nil)   ← allocate VarID
    block.shape = shape

Phase 2 — BlockEmission (Binding + Propagation)
    One of 4 paths runs, each binding the element index differently:

    ┌─────────────────────────────────────────────────────────────────────┐
    │ Path                  │ Uses block.tensorIndex? │ What it does      │
    ├───────────────────────┼────────────────────────-┼───────────────────┤
    │ Standard              │ YES                     │ Copies it to      │
    │ (line 220-228)        │                         │ ctx.tensorIndices  │
    │                       │                         │ + beginParallelRng │
    ├───────────────────────┼────────────────────────-┼───────────────────┤
    │ setupFlatThreading    │ NO — creates binIdx     │ Decomposes thread │
    │ (line 207-210)        │ from threadId           │ ID into frame+elem│
    ├───────────────────────┼────────────────────────-┼───────────────────┤
    │ Stateful tensor       │ NO — creates tensorIdx  │ Uses raw          │
    │ (line 196-205)        │ from threadIndex()      │ threadIndex()     │
    ├───────────────────────┼────────────────────────-┼───────────────────┤
    │ Shape-aware regions   │ NO — creates per-region │ Each region gets  │
    │ (RegionEmitter:40-53) │ elemVar                 │ its own loop var  │
    └─────────────────────────────────────────────────────────────────────┘

Phase 3 — Node Emission (Consumption)
    Node emitters read ctx.tensorIndices[nodeId] to index into tensor memory.
    They don't care which path created it.
```

Only the **Standard** path uses the VarID minted in BlockFormation. The other 3 paths
discard it and create their own variable. The Standard path is the one that produces a
`beginParallelRange` UOp, which renders as:
- C backend: `for (uint _pr = 0; _pr < N; _pr++)`
- Metal thread mode: `if (id < N) { uint _pr = id; }`
- Metal loop mode: `for (uint _pr = 0; _pr < N; _pr++)`

## Proposed Change

**Remove `block.tensorIndex` from `Block`. Keep `block.shape` as the "is tensor block"
flag (it already serves this purpose). Mint the variable at emission time, at the point
where it's actually bound.**

### What `block.tensorIndex` currently does

1. **Flag**: "this block iterates over tensor elements" — but `block.shape != nil`
   already says this (they're always set together in BlockFormation)
2. **Variable ID**: the loop/thread binding — but this isn't meaningful until emission,
   and 3 of 4 paths create their own anyway

### After the change

BlockFormation only sets `block.shape`. The emission paths each mint their own variable
at the point of binding:

```
Phase 1 — BlockFormation
    block.shape = shape                    ← just the flag, no variable allocation

Phase 2 — BlockEmission (Mint + Bind + Propagate in one place)
    if block.shape != nil:
        choose binding path based on (backend, statefulTensor, shapeAware)
        mint variable at point of use
        populate ctx.tensorIndices[nodeId] for each node

Phase 3 — Node Emission (unchanged)
    read ctx.tensorIndices[nodeId]
```

Each binding path becomes self-contained: it creates the variable, binds it to a
concrete iteration value, and propagates it to nodes — all in one place.

### Concrete sketch

In `emitStandardBlockBodyUOps`:

```swift
// Before (3 paths that each handle propagation differently)
if backend == .metal, statefulTensorDecision.enabled {
    let tensorIdx = setup.threadIndex()
    for nodeId in block.nodes { ctx.tensorIndices[nodeId] = tensorIdx.lazy }
} else {
    threadScaleUOps = (backend == .metal)
        ? emitThreadCountScaleOpIfNeeded(...)   // sets ctx.tensorIndices internally
        : []
}
// ... later, for nodes not covered above:
if !emittedThreadScale, let tensorIndex = block.tensorIndex {
    ctx.tensorIndices[nodeId] = tensorIndex
}

// After (unified: each path mints + propagates)
let elementIndex: Lazy? = resolveElementIndex(block, backend, statefulTensorDecision, ctx)
if let idx = elementIndex {
    for nodeId in block.nodes where !(g.nodes[nodeId]?.op.isInherentlyScalar ?? false) {
        ctx.tensorIndices[nodeId] = idx
    }
}
```

Where `resolveElementIndex` encapsulates the 3 non-shape-aware paths (standard,
threadScale, statefulTensor) and returns the appropriate `Lazy`. Shape-aware emission
continues to handle its own per-region variables in `RegionEmitter`.

### What stays the same

- `ctx.tensorIndices: [NodeID: Lazy]` — the per-node dictionary must remain because
  shape-aware blocks genuinely assign different indices to different nodes
- `beginParallelRange` — still emitted for the standard path, still rendered by both
  Metal and C renderers in their respective modes
- Node emitters — no changes, they just read `ctx.tensorIndices[nodeId]`
- `block.shape` — stays on Block, serves as the tensor-block flag

### Sites to update

1. **`Block` struct** (`Blocks.swift`): remove `tensorIndex` field
2. **`BlockFormation.swift`**: remove `assignTensorIndexFromFirstTensorNode` and all
   `block.tensorIndex = ctx.useVariable(src: nil)` assignments. Keep `block.shape =`
   assignments.
3. **`BlockEmission.swift`**: consolidate the 4 binding paths. The standard path mints
   its own VarID at the point where it creates `beginParallelRange`.
4. **`determineSIMDPlan`**: replace `block.tensorIndex != nil` checks with
   `block.shape != nil` (lines 310, 323)
5. **`wrapBodyUOpsWithTensorLoopIfNeeded`**: replace `block.tensorIndex` reads with a
   locally-minted variable (or accept it as a parameter from the emission path)
6. **`Block.==`**: remove `tensorIndex` from equality check

### Risks

- **Low risk**: the change is internal to the compilation pipeline. No public API
  changes. Node emitters are unchanged (they read from `ctx.tensorIndices`, not from
  `block.tensorIndex`).
- **Testing**: any existing test that compiles and runs a tensor block exercises this
  path. The generated kernels should be identical — only the internal variable allocation
  timing changes.
