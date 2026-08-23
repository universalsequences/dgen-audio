# DGen - Claude Development Notes

## Spectral Loss

### Key Insight: Frequency Resolution

**Spectral loss requires adequate frequency resolution:**

```
resolution = sampleRate / windowSize
```

If target frequencies are closer together than this resolution, they'll be in the same DFT bin and spectral loss won't work correctly.

#### Example

| Sample Rate | Window Size | Resolution | Can distinguish 440 Hz vs 460 Hz? |
|-------------|-------------|------------|-----------------------------------|
| 44100 Hz    | 64          | 689 Hz     | No (same bin)                     |
| 44100 Hz    | 2048        | 21.5 Hz    | Yes                               |
| 2000 Hz     | 64          | 31.25 Hz   | Yes                               |

#### Practical Guidelines

1. For audio at 44100 Hz sample rate, use window sizes of at least 1024-2048 samples for musical frequency discrimination
2. For lower sample rates (e.g., 2000 Hz for testing), smaller windows work fine
3. The frequency difference between student and teacher should span at least 2-3 bins for reliable gradient direction

### Gradient Accumulation

Spectral loss gradients are summed across all overlapping windows. Each sample at position `p` appears in `windowSize` different windows (at frames `p`, `p+1`, ..., `p+windowSize-1`). The `spectralLossFFTGradRead` operation sums contributions from all these windows to get the correct gradient.

### Local Minima

Some frequency combinations can get stuck in local minima. This is particularly true for:
- Harmonic relationships (e.g., 2:1, 3:2 frequency ratios)
- Frequencies that happen to align with bin boundaries in unexpected ways

If training gets stuck, try:
- Different initialization
- Higher learning rate
- Larger window size for better frequency resolution

## DGenLazy Training Loop

### Gradient Lifecycle (Tinygrad-Style)

DGenLazy uses a tinygrad-inspired pattern where the computation graph is rebuilt each iteration:

1. **`backward()`** - Computes gradients and stores them in `.grad` properties
2. **`step()`** - Reads gradients and updates parameter weights
3. **`zeroGrad()`** - Clears `.grad = nil` to prepare for next iteration
4. **Always capture metrics before `zeroGrad()`** if you need them later

### Graph Rebuilding

After `backward()`, the graph is cleared to prevent node accumulation. Parameters (created with `Tensor.param()`) survive because their data is stored locally and nodeIds are lazily recreated. Computed nodes like `loss` must be rebuilt each iteration:

```swift
for epoch in 0..<epochs {
    let loss = buildLoss()  // Rebuild graph fresh
    try loss.backward(frames: frameCount)
    optimizer.step()
    optimizer.zeroGrad()
}
```

## Testing: Tensor/Signal Creation Order Matters

**Always create `Tensor` objects AFTER `LazyGraphContext.reset()`, not before.**

`Tensor` and `Signal` objects store an internal `nodeId` pointing to their graph node. When `LazyGraphContext.reset()` creates a new graph, objects created before the reset hold stale `nodeId`s. The `refresh()` mechanism re-creates them in the new graph, but only when they're used in an operator (e.g., `*`, `+`). If a `Tensor` is created before `reset()` and the `refresh()` fires correctly, it gets a new valid `nodeId`. But if anything goes wrong with refresh detection, the stale `nodeId` silently aliases an unrelated node in the new graph (e.g., the audio input), causing operations to produce wrong results with no error.

```swift
// BAD — hannWindow created before reset, may alias wrong node
let hannWindow = Tensor(hannData)
LazyGraphContext.reset()
let flat = sig.buffer(size: N, hop: hop).reshape([N])
let windowed = flat * hannWindow  // hannWindow.nodeId may be stale!

// GOOD — create after reset
LazyGraphContext.reset()
let hannWindow = Tensor(hannData)
let flat = sig.buffer(size: N, hop: hop).reshape([N])
let windowed = flat * hannWindow  // hannWindow.nodeId is valid
```

**Symptoms of stale nodeId**: Operations compile and run without errors but produce wrong results. A multiplication by a tensor window becomes a no-op (multiply by 1.0). The generated kernel shows missing operations with no obvious cause.

## Metal GPU Synchronization

1. **`atomic_thread_fence` does NOT sync between threads** - it only orders memory operations within a single thread. For cross-thread synchronization, split into separate kernels.

2. **Reduction ops need kernel boundaries** - If a write phase stores per-frame data and a reduce phase reads from ALL frames, they MUST be in separate kernels. Add the op to `isReductionOp()` in Blocks.swift.

3. **Global reduces should skip thread scaling** - Ops like `peekRowGradReduce` that loop over all frames internally should NOT get `threadCountScale`. Check `splitReduceBlocks()` to exclude them from shape assignment.

## Memory Allocation & Cell IDs

### Cell IDs ≠ Memory Addresses

Cell IDs are **logical identifiers**, not memory offsets. The actual memory layout is computed by `remapVectorMemorySlots` in CompilationPipeline.swift.

```
Cell ID 0  → memory[84..99]   (after remapping)
Cell ID 16 → memory[100..103]
Cell ID -4 → memory[80..95]   (lazy cell)
```

To debug memory issues, check `cellAllocations.cellMappings` after compilation.

### Lazy Cells (Negative IDs)

Tensors created during graph construction get **lazy cells** (negative IDs like -1, -2, etc.) via `reserveLazyCellId()`. These are placeholders until we know if the tensor needs:
- Frame-aware allocation (tensorSize × frameCount)
- Outbound allocation (crosses block boundaries)

**Critical**: `allocateTensorMemory` in TypeChecker.swift must register sizes for ALL lazy cells in `cellAllocationSizes`, even non-outbound ones. Otherwise `remapVectorMemorySlots` defaults to size=1, causing memory overlap.

### Debugging Memory Corruption

If gradients explode or have wrong values:
1. Check generated Metal kernel for memory indices (e.g., `memory[80 + ...]`)
2. Look for overlapping ranges between different tensors
3. Verify `cellAllocationSizes` has correct sizes for all cells used
4. Add debug output in GraphTrainingContext to print `cellAllocations.cellMappings`


<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:1105d646 -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/core-concepts/sync-concepts.md for details and anti-patterns.

## Agent Context Profiles

The managed Beads block is task-tracking guidance, not permission to override repository, user, or orchestrator instructions.

- **Conservative (default)**: Use `bd` for task tracking. Do not run git commits, git pushes, or Dolt remote sync unless explicitly asked. At handoff, report changed files, validation, and suggested next commands.
- **Minimal**: Keep tool instruction files as pointers to `bd prime`; use the same conservative git policy unless active instructions say otherwise.
- **Team-maintainer**: Only when the repository explicitly opts in, agents may close beads, run quality gates, commit, and push as part of session close. A current "do not commit" or "do not push" instruction still wins.

## Session Completion

This protocol applies when ending a Beads implementation workflow. It is subordinate to explicit user, repository, and orchestrator instructions.

1. **File issues for remaining work** - Create beads for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **Handle git/sync by active profile**:
   ```bash
   # Conservative/minimal/default: report status and proposed commands; wait for approval.
   git status

   # Team-maintainer opt-in only, unless current instructions forbid it:
   git pull --rebase
   git push
   git status
   ```
5. **Hand off** - Summarize changes, validation, issue status, and any blocked sync/commit/push step

**Critical rules:**
- Explicit user or orchestrator instructions override this Beads block.
- Do not commit or push without clear authority from the active profile or the current user request.
- If a required sync or push is blocked, stop and report the exact command and error.
<!-- END BEADS INTEGRATION -->
