IMPORTANT: You MUST use the Task tool to spawn a subagent. Do NOT read the kernel file in the main conversation - this preserves context.

Spawn using:
- subagent_type: "general-purpose"
- description: "Analyze Metal kernel perf"
- prompt: See below with file path substituted

File: $ARGUMENTS

---
SUBAGENT PROMPT:

Read the Metal kernel file above. This is a DGen-generated kernel.

## DGen-Specific Checks

1. **Static vs Dynamic**: Are static computations recomputed per-frame?
2. **Gradient Loops**: Inefficient `for (_gfi = 0; _gfi < frameCount; _gfi++)` accumulation
3. **Index Math**: Expensive floor/modulo chains for tensor indexing
4. **Broadcast Patterns**: Redundant memory reads from broadcast access
5. **Frame Structure**: SIMD-across-frames efficiency

## General GPU Checks

1. **Thread Divergence**: SIMD divergence from conditionals
2. **Memory Coalescing**: Access patterns
3. **Sync Barriers**: atomic_thread_fence count
4. **Register Pressure**: Float temporaries per thread

## Output (under 350 words)

**Summary**: [1-2 sentences]

**Kernels**:
| Kernel | Threads | Op | Issue |
|--------|---------|-----|-------|

**Top 3 Bottlenecks**:
1. [kernel_N line X: issue -> fix]

**Quick Wins**: [bullets]

---

Relay findings after subagent returns.
