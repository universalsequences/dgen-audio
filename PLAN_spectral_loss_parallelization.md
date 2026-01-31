# Plan: Proper Spectral Loss Parallelization

## Background

Spectral loss computes frequency-domain difference between two signals using sliding-window DFT:
```
L = Σ_windows Σ_bins (|X1[k]| - |X2[k]|)²
```

For backprop, each sample j needs gradient contributions from ALL windows containing it and ALL frequency bins.

## What We Tried (FAILED)

### Attempt: Inline DFT Recomputation in Pass2

Changed `u_spectralLossBackwardPass2` to recompute DFT inline instead of reading from scratch memory:

```swift
// BROKEN - Do not use
b.loop(windowSize) { offsetFromJ in
    b.loop(numBins) { binIndex in
        // Recompute entire DFT for this window...
        // Compute gradient contribution...
        grad1.accumulate(contrib1)
    }
}
```

**Why it failed**: Unknown - mathematically should be equivalent, but produced incorrect/lower gradients. Possible causes:
- Subtle numerical differences in recomputation vs stored values
- Bug in the n-position calculation
- Different accumulation order affecting floating point precision

**Lesson**: Avoid recomputing DFT in Pass2 - stick with memory-based approach.

---

## Current Implementation Analysis

### Bug in Current Pass1!

Looking at `u_spectralLossBackwardPass1` (Operators.swift:2530-2594):

```swift
b.loop(numBins) { binIndex in        // <- Outer loop over bins
    // ... compute DFT for this bin ...

    b.loop(windowSize) { n in        // <- Inner loop over samples
        // Compute contribution for THIS bin...

        // BUG: Writing to same offset for ALL bins - only last bin survives!
        let offset1 = i * winSizeConst * 2.0 + n * 2.0  // No bin index!
        _ = b.memoryWrite(scratchCell, offset1, contrib1)
    }
}
```

The memory offset is `(frame * windowSize * 2) + (sample * 2)` - it doesn't include `binIndex`!

Each bin OVERWRITES the previous bin's contribution. Only the last bin (k = numBins-1) survives.

**Why it still converges**:
- Forward loss is computed correctly (sums all bins)
- Backward gradient from last bin gives SOME signal in the right direction
- Learning rate compensates for magnitude being wrong

---

## Correct Parallelization Approaches

### Option A: Fix Bin Accumulation (Recommended - Minimal Change)

Swap loop order and accumulate across bins BEFORE writing:

```swift
func u_spectralLossBackwardPass1_fixed(...) {
    let i = b.threadIndex()  // Frame/window index

    // For each sample position in window
    b.loop(windowSize) { n in
        let contrib1 = b.float(0.0)  // Accumulator for all bins
        let contrib2 = b.float(0.0)

        // Sum contributions across ALL bins
        b.loop(numBins) { binIndex in
            // Compute DFT for this bin (need to recompute per bin)
            // ... DFT computation ...

            let binContrib1 = lossGrad * sampleGrad1 * upstreamGrad
            let binContrib2 = -lossGrad * sampleGrad2 * upstreamGrad

            contrib1.accumulate(binContrib1)
            contrib2.accumulate(binContrib2)
        }

        // Write ONCE per (frame, sample) - summed over all bins
        let offset1 = i * winSize * 2.0 + n * 2.0
        _ = b.memoryWrite(scratchCell, offset1, contrib1.value)
        _ = b.memoryWrite(scratchCell, offset2, contrib2.value)
    }
}
```

**Challenge**: DFT coefficients (real1, imag1, mag1, etc.) must be computed per-bin inside the inner loop.

### Option B: Expand Memory Layout (More Memory, Simpler Logic)

Allocate memory for ALL bins:
```
scratch[frame][sample][bin][component]
= scratch[i * windowSize * numBins * 2 + n * numBins * 2 + k * 2 + c]
```

Pass1 writes per-bin contributions, Pass2 sums across bins while gathering.

**Pros**: Current loop structure mostly works
**Cons**: Memory = O(frames * windowSize * numBins) instead of O(frames * windowSize)

### Option C: Parallelize Across Bins (GPU-Friendly)

Use `parallelRange(numBins)` to parallelize bin computation:

```swift
// Each thread handles one bin
b.parallelRange(numBins) { binIndex in
    // Compute DFT for this bin across all windows
    // Use atomic adds or reduction for gradient accumulation
}
```

**Challenge**: Need atomic operations or careful reduction for gradient accumulation.

---

## Recommended Implementation Plan

### Phase 1: Fix the Bin Accumulation Bug (Option A)

1. Restructure `u_spectralLossBackwardPass1`:
   - Outer loop: samples (n)  
   - Inner loop: bins (k)
   - Accumulate bin contributions before writing

2. For each sample n, need to compute DFT for ALL bins. Structure:
   ```swift
   b.loop(windowSize) { n in
       let totalContrib1 = b.float(0.0)
       let totalContrib2 = b.float(0.0)
       
       b.loop(numBins) { binIndex in
           // Compute DFT accumulators for this bin
           let real1 = b.float(0.0), imag1 = b.float(0.0)
           let real2 = b.float(0.0), imag2 = b.float(0.0)
           
           b.loop(windowSize) { m in
               // DFT summation over window
           }
           
           // Compute magnitude, loss gradient, sample gradient
           // Accumulate this bin's contribution
           totalContrib1.accumulate(binContrib1)
           totalContrib2.accumulate(binContrib2)
       }
       
       // Write once, summed over all bins
       b.memoryWrite(...)
   }
   ```

3. Keep Pass2 unchanged (memory gather still works)

### Phase 2: Verify Correctness

1. Add numerical gradient checking test
2. Compare gradients against finite-difference approximation
3. Verify loss converges faster/to lower values than buggy version

### Phase 3: Optional - True Parallelization

Only if Phase 1 is still too slow, consider bin-level parallelization with atomic ops.

---

## Files to Modify

- `Sources/DGen/Operators.swift`: `u_spectralLossBackwardPass1`
- `Tests/DGenTests/`: Add gradient verification test
