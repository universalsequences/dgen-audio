# Plan: Parallelize SpectralLoss Across Frequency Bins

## Current Performance Problem

Looking at the generated Metal kernels (`/tmp/static_mlp_piano_kernels.metal`):

**kernel_8** (lines 373-643) is the main bottleneck:
- `ThreadGroupSize: Optional(1)` - runs on a **single GPU thread**
- Sequential frame loop: `for (uint i = 0; i < frameCount; i++)`
- Nested DFT loops: `for t223 < 33` × `for t228 < 64` = 2,112 ops/frame
- Then another set for backward: `for t286 < 33` × `for t323 < 64`

**Total: ~4,224 sequential operations per frame on ONE thread**

## Root Cause Analysis

The current implementation in `Definitions.swift:140-189` and `Operators.swift:2520-2599`:

```swift
// Forward - sequential bin loop
b.loop(numBins) { binIndex in      // 33 bins for window=64
  b.loop(windowSize) { n in        // 64 samples
    // DFT accumulation
  }
  // Compute magnitude and accumulate error
}

// Backward Pass1 - sequential bin loop
b.loop(numBins) { binIndex in
  b.loop(windowSize) { n in ... }  // Compute DFT
  b.loop(windowSize) { n in ... }  // Write gradients
}
```

### Bug in Current Backward Implementation

The current backward pass has a subtle bug: multiple bins contribute to the same sample's gradient, but the code **overwrites** rather than **accumulates**:

```swift
// Line 2594-2595: This OVERWRITES per-bin, should ACCUMULATE
_ = b.memoryWrite(scratchCell, offset1, contrib1)  // Later bins overwrite earlier!
```

The fix must accumulate contributions from all bins.

---

## Proposed Solution: Tensor-Based Parallel DFT

### Mathematical Foundation

The DFT can be expressed as matrix multiplication:

```
Forward:
  X_real[k] = Σ_n signal[n] * cos(-2πkn/N)
  X_imag[k] = Σ_n signal[n] * sin(-2πkn/N)

  Or: X = DFT_matrix @ signal  (where DFT_matrix is [numBins × windowSize])

Backward:
  ∂L/∂signal[n] = Σ_k ∂L/∂X[k] * ∂X[k]/∂signal[n]
                = Σ_k grad_magnitude[k] * DFT_basis[k,n]

  Or: grad_signal = DFT_matrix^T @ grad_magnitude
```

### Key Insight: Separate Bin Computation from Cross-Bin Reduction

**Forward Pass:**
1. **Parallel across bins**: Each thread computes DFT for one bin
2. **Per-bin output**: Store `|X1[k]| - |X2[k]|` squared to tensor `[numBins]`
3. **Reduction**: Sum tensor to get total loss (existing tensor.sum() infrastructure)

**Backward Pass:**
1. **Parallel across bins**: Compute `grad_per_bin[k] = 2*(mag1-mag2) * (real*cos + imag*sin) / mag`
2. **Store intermediate**: Write to tensor `[numBins × windowSize × 2]`
3. **Parallel across samples**: Reduce bins to get per-sample gradient

---

## Implementation Plan

### Phase 1: Create Precomputed DFT Basis Tensors (Static)

Create tensors for the DFT basis functions that can be computed once:

```swift
// In spectralLoss() setup:
let cosBasis = tensor(shape: [numBins, windowSize])  // cos(-2πkn/N)
let sinBasis = tensor(shape: [numBins, windowSize])  // sin(-2πkn/N)
// Initialize with DFT basis values
```

This moves trig computation out of the per-frame loop.

### Phase 2: Forward Pass - Parallel Bin Computation

```swift
func spectralLossForward_v2(sig1: Tensor, sig2: Tensor, windowSize: Int) -> (loss: NodeID, intermediates: Intermediates) {
  let numBins = windowSize / 2 + 1

  // Tensors for DFT results [numBins]
  let real1 = zeros([numBins])
  let imag1 = zeros([numBins])
  let real2 = zeros([numBins])
  let imag2 = zeros([numBins])

  // Parallel across bins
  parallelRange(numBins) { k in
    // For each bin k, compute DFT using tapeLoad
    loop(windowSize) { n in
      let sample1 = tapeLoad(sig1, at: frameIdx - windowSize + 1 + n)
      let sample2 = tapeLoad(sig2, at: frameIdx - windowSize + 1 + n)

      // Use precomputed basis or compute inline
      let angle = -2π * k * n / windowSize
      real1[k] += sample1 * cos(angle)
      imag1[k] += sample1 * sin(angle)
      // ... same for sig2
    }
  }

  // Parallel across bins: compute magnitudes
  let mag1 = sqrt(real1*real1 + imag1*imag1)  // [numBins]
  let mag2 = sqrt(real2*real2 + imag2*imag2)  // [numBins]

  // Loss: sum of squared differences
  let diff = mag1 - mag2                       // [numBins]
  let sqDiff = diff * diff                     // [numBins]
  let loss = sqDiff.sum()                      // scalar

  return (loss, Intermediates(real1, imag1, real2, imag2, mag1, mag2, diff))
}
```

### Phase 3: Backward Pass1 - Parallel Gradient Computation

```swift
func spectralLossBackward_v2(intermediates: Intermediates, upstreamGrad: NodeID) {
  let numBins = windowSize / 2 + 1

  // Gradient tensor: [numBins × windowSize × 2] for sig1 and sig2
  let binGrads = zeros([numBins, windowSize, 2])

  // Parallel across bins
  parallelRange(numBins) { k in
    let lossGrad = 2.0 * intermediates.diff[k] * upstreamGrad

    loop(windowSize) { n in
      let angle = -2π * k * n / windowSize
      let c = cos(angle)
      let s = sin(angle)

      // ∂mag/∂signal = (real*cos + imag*sin) / mag
      let sampleGrad1 = (intermediates.real1[k] * c + intermediates.imag1[k] * s)
                        / (intermediates.mag1[k] + eps)
      let sampleGrad2 = (intermediates.real2[k] * c + intermediates.imag2[k] * s)
                        / (intermediates.mag2[k] + eps)

      binGrads[k, n, 0] = lossGrad * sampleGrad1
      binGrads[k, n, 1] = -lossGrad * sampleGrad2
    }
  }

  // Store binGrads for Pass2 reduction
}
```

### Phase 4: Backward Pass2 - Parallel Sample Gradient Reduction

```swift
func spectralLossBackwardPass2_v2(binGrads: Tensor, windowSize: Int) -> (grad1: Tensor, grad2: Tensor) {
  let numBins = windowSize / 2 + 1

  // Output: per-sample gradients
  let grad1 = zeros([windowSize])
  let grad2 = zeros([windowSize])

  // Parallel across samples
  parallelRange(windowSize) { n in
    // Sum contributions from all bins (this is the reduction)
    loop(numBins) { k in
      grad1[n] += binGrads[k, n, 0]
      grad2[n] += binGrads[k, n, 1]
    }
  }

  return (grad1, grad2)
}
```

### Phase 5: Window Overlap Handling

The existing Pass2 gathers from overlapping windows. This stays the same but now uses the reduced per-sample gradients:

```swift
// For each sample j in the output
parallelRange(frameCount) { j in
  // Gather from all windows that include sample j
  loop(windowSize) { offset in
    let windowEnd = j + offset
    let n = j - windowEnd + (windowSize - 1)

    // Read from reduced gradient tensor instead of raw scratch
    finalGrad1[j] += windowGrads[windowEnd, n, 0]
    finalGrad2[j] += windowGrads[windowEnd, n, 1]
  }
}
```

---

## Memory Layout Changes

### Current Layout (Problematic)
```
scratchCell: [frameCount × windowSize × 2]
  - Indexed by: frame_idx * windowSize * 2 + sample_in_window * 2 + signal_component
  - Problem: No bin dimension, bins overwrite each other
```

### New Layout Option A: Full Intermediate Storage
```
binGradients: [frameCount × numBins × windowSize × 2]
  - Full storage of all bin contributions
  - Memory: frameCount × 33 × 64 × 2 × 4 bytes = ~17MB for 1024 frames
  - Pro: No contention, clean reduction
  - Con: More memory
```

### New Layout Option B: Reduced Storage with Atomic Adds
```
sampleGradients: [frameCount × windowSize × 2]
  - Use atomic_add when accumulating across bins
  - Memory: frameCount × 64 × 2 × 4 bytes = ~0.5MB for 1024 frames
  - Pro: Less memory
  - Con: Atomic contention on same memory locations
```

**Recommendation**: Use Option A for correctness first, optimize later if memory is an issue.

---

## Files to Modify

1. **`Sources/DGen/HigherOps.swift`** (spectralLoss function)
   - Add intermediate tensor allocations
   - Update operator structure

2. **`Sources/DGen/Definitions.swift`** (u_spectralLoss forward)
   - Rewrite using `parallelRange` instead of `loop` for bins
   - Store per-bin results to tensor

3. **`Sources/DGen/Operators.swift`** (backward passes)
   - Rewrite u_spectralLossBackwardPass1 with `parallelRange`
   - Add new reduction kernel for bin→sample gradient accumulation
   - Update scratch memory layout

4. **`Sources/DGen/Blocks.swift`** (if kernel splitting needed)
   - May need to ensure bin-parallel operations get their own kernel

5. **Tests** - Update to verify parallel version matches sequential

---

## Execution Order (Kernel Structure)

### Current (All Sequential on 1 Thread)
```
kernel_8: [ThreadGroup=1]
  for each frame:
    forward DFT (bins × samples)
    backward DFT (bins × samples)
    backward gradient write (bins × samples)
```

### New (Parallel)
```
kernel_forward_dft: [ThreadGroup=numBins]
  - Each thread: one bin's DFT computation

kernel_forward_reduce: [ThreadGroup=1 or parallel reduction]
  - Sum per-bin losses to total

kernel_backward_bingrads: [ThreadGroup=numBins]
  - Each thread: one bin's gradient computation
  - Write to [numBins × windowSize × 2] tensor

kernel_backward_reduce: [ThreadGroup=windowSize]
  - Each thread: one sample
  - Sum across bins for that sample

kernel_backward_gather: [ThreadGroup=frameCount]
  - Each thread: one output sample
  - Gather from overlapping windows
```

---

## Expected Speedup

- Current: 33 bins × 64 samples × 2 (forward+backward) = ~4,224 ops sequential
- After:
  - DFT: 64 samples × 2 sequential per thread, 33 threads parallel → ~128 ops
  - Reduction: 33 bins per thread, 64 threads → ~33 ops
  - **~30-50× speedup expected** for the spectral loss computation

---

## Implementation Priority

1. **Fix the bug first**: Ensure backward pass accumulates across bins correctly
2. **Implement parallel forward**: Easiest to verify correctness
3. **Implement parallel backward**: More complex, test thoroughly
4. **Optimize memory layout**: Once correctness is verified

---

## Testing Strategy

1. **Numerical verification**: Compare parallel vs sequential results
2. **Gradient checking**: Finite difference verification of backward pass
3. **Performance benchmarking**: Measure kernel execution times before/after
4. **Integration test**: Run the MLP piano test and verify learning still works
