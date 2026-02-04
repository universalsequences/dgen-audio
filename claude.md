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
