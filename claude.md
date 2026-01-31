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
