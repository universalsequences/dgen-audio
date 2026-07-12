# SynthID Runtime and GPU Refinement Notes

## Summary

**Updated 2026-07-12 with measured numbers.** The original version of this
document estimated a 25–30 minute rung-3 run and identified the Python
refiner's sequential rendering as the main architectural gap. Direct
measurement on `Assets/909kick.wav` shows the picture has changed:

1. Per-epoch cost quadrupled (0.27 s → 1.04 s) when the 40 TR-909
   harmonic-correction oscillators were added. A default full run at current
   settings is ~90+ minutes, not 25–30.
2. The Python refiner's cost also grew (~51 s → ~8 min) after the harmonic
   coordinate-search stages were added, but it is still not the dominant cost.
3. The "persistent compiled graphs" item is already implemented:
   `Realize.swift` has a compilation fingerprint fast-path and an MTLLibrary
   cache keyed by kernel-source hash, so steady-state epochs do not recompile.

The largest wins now are: keep the harmonic corrections out of the graph
during restart search, run release builds, hoist pitch extraction, early-stop,
and vectorize the Python refiner's harmonic stages. GPU-native batched
refinement remains a valid longer-term idea but attacks a minority of the
runtime.

## Measured workload (2026-07-12, Assets/909kick.wav, current tree)

Method: full pipeline runs at 40 vs 240 epochs (1 restart, no pitch refine,
`--no-refine true`), so fixed costs cancel in the difference. Verified on an
otherwise idle machine.

| Quantity                          | Debug   | Release |
|-----------------------------------|---------|---------|
| Steady-state training epoch       | ~1.18 s | ~1.04 s |
| Fixed cost per restart (pitch fit + first compile) | ~41 s | ~3 s |

Additional measurements:

- The v3 run (before the 40 harmonic-correction terms existed — its
  checkpoints contain only the 16 base params) trained at **0.27 s/epoch**.
  The 4× regression is the harmonic-correction terms; epoch time is
  GPU/dispatch dominated, which is why release barely helps the loop itself.
- The current `refine_rung3.py` (with the 909 harmonic stages: ~30 passes ×
  40 params × 17 steps × 2 rounds) takes **6.5–9 minutes**. The 51 s figure
  from the v3 run predates those stages.
- Loss curves: every v3 restart was within 1% of its final best loss by epoch
  ~390–460 of 900. The second half of the epoch budget buys under 1%.
- Projected full run at current defaults (5 restarts × 900 epochs + stitch +
  refine): **~90–100 minutes** in release.

## Current workload structure

The default rung-3 configuration performs approximately:

```text
5 restarts × (600 training epochs + 300 pitch epochs) = 4,500 epochs
stitched run: 200 training epochs + 300 pitch epochs  =   500 epochs
                                                       ------------
                                                        ~5,000 epochs
```

Each epoch renders 22,528 audio samples and backpropagates through overlapping
STFT losses with window sizes 256, 512, 1024, and 2048.

After GPU training, `scripts/refine_rung3.py` performs a gradient-free search
against the independent acceptance metric. For each parameter, it renders many
candidate patches sequentially in NumPy, computes four STFT distances, keeps
the best candidate, and repeats with progressively narrower search ranges.

## Why Python refinement exists

The differentiable training loss and the final acceptance metric are related,
but not identical. The final metric is:

```text
distance = sum over windows(
  mean(abs(
    log(abs(STFT(rendered)) / hannScale + 0.001)
    - log(abs(STFT(target)) / hannScale + 0.001)
  ))
)
```

Before the STFT, the target, initialization, and learned audio receive the same
zero-phase 30 Hz high-pass. The reported improvement is:

```text
1 - learnedDistance / initialDistance
```

Python refinement directly optimizes this declared metric. It also provides an
implementation independent of DGen/Metal, which helps detect renderer or loss
bugs. On the original corrected 909 run it reduced the distance from about
`0.038771` to `0.034797`, raising improvement from approximately 65.45% to
69.00%.

## Ranked runtime improvements

Ordered by measured impact; none require a new GPU abstraction.

### 1. Freeze the 40 harmonic corrections OUT of the graph during restart search

They are zero-default corrections whose job is the last few points of the
metric, yet every one of the ~4,500 restart epochs pays sin/cos + envelope +
backward for all 40 terms. Skipping the *optimizer step* is not enough — the
terms must be absent from the graph (as when `bodyAsymmetry`-style terms are
inert). Train them on the winner only, for a few hundred epochs. This alone
takes 5 × 900 epochs from ~78 min back to ~20 min.

### 2. Group harmonic terms by decay when they are trained

There are only 4 distinct decay values (0, 15, 60, 240), but `Patch.swift`
builds `bodyEnv * bodyAmp * exp(-decay·t)` per term. Factoring into 4 group
envelopes × Σ(cₖ·waveₖ) turns 40 exp+mul chains into 4, in forward and
backward.

### 3. Use release builds and hoist pitch extraction

Recent runs were debug builds (`Building for debugging...` in the run logs).
Release cuts per-restart fixed cost 41 s → 3 s. Independently,
`PitchTrack.extract`/`fit` is recomputed inside every `train()` call on the
identical target (6× per full run) — compute it once in `main.swift` and pass
it in.

### 4. Vectorize the Python refiner's harmonic stages

That is where its ~8 minutes go. When only harmonic coefficient cₖ changes,
the pre-drive mix is `cached_fixed_mix + cₖ·(waveₖ·env)` — one FMA instead of
a full re-render. All 17 candidates for one coordinate can then go through the
high-pass and the four STFTs as a single batched NumPy call (`rfft` over a
stacked axis). Expected 10×+ on the refiner with no DGen/Metal work.

### 5. Early stopping with a compressed LR schedule

Given the loss curves, ~450–500 epochs with the cosine schedule compressed to
that length should match 900-epoch quality. Combine with reducing restarts
(see below) to halve the epoch count again.

### 6. Reduce redundant restarts and run pitch refinement winner-only

Five restarts repeat nearly the entire forward and backward workload, and the
300 pitch-only epochs run for every restart even though only one is selected.
Reduce to two restarts after validating the deterministic pitch initialization
brackets, and refine only the restart selected by a cheaper preliminary score.

### 7. Parallelize restarts as subprocesses

Restarts are fully independent and the trainer is single-process. Launching
them as separate `SynthID` processes overlaps host-side graph construction
with the GPU work of the others; realistic 2–3× wall clock on the restart
phase.

### 8. Stop recompiling in the stitch phase

The stitch loop makes ~46 `evaluateLoss` calls, each doing
`LazyGraphContext.reset()` → full pipeline recompile → full *backward* pass,
when it only needs a forward loss with swapped parameter values on the cached
compiled graph.

Combined, items 1–6 plausibly put a full run near 10–15 minutes without any
new GPU abstraction.

Known CLI paper cut: `--no-refine` is not registered in the boolean-flags set
in `main.swift`, so it silently requires a dummy value (`--no-refine true`);
bare `--no-refine` errors out.

## What DGen needs for GPU-native refinement (longer term)

Note: with the refiner at ~8 minutes (and ~1 minute once vectorized per item 4
above), this section attacks a minority of the runtime. Build it only if the
NumPy-vectorized refiner is still a bottleneck.

The central missing feature is a parameter-candidate batch dimension. Instead
of rendering one signal shaped as `[audioFrame]`, DGen should support a search
batch shaped conceptually as:

```text
[candidate, audioFrame]
```

One GPU operation could then render a coordinate grid, compute each candidate's
four-resolution distance, and return a vector containing one loss per candidate.

Supporting this well requires:

- Batched parameter buffers and broadcasting through the signal graph.
- Batched overlapping FFTs whose reductions preserve the candidate dimension.
- The exact gate loss on GPU: Hann coherent-gain normalization, log epsilon,
  mean absolute difference, and equal weighting of the four resolutions.
- A comparator-equivalent zero-phase high-pass. DGen's ordinary biquad is
  causal, whereas the comparator uses forward/backward filtering.
- Candidate-local reductions so no STFT frames or losses leak between candidates.
- A GPU coordinate-search driver that generates a candidate grid, evaluates it,
  selects the minimum, and contracts the search interval.
- NumPy-versus-Metal parity tests for rendered audio and the final loss.

(Persistent compiled graphs — previously listed here — already exist: see
`Realize.swift`'s fingerprint fast-path and kernel-source-hash runtime cache.)

The desired execution shape is:

```text
parameter grid
    -> batched render
    -> batched zero-phase high-pass
    -> batched MR-STFT
    -> one distance per candidate
    -> select winner
```

This turns the most parallel part of Python refinement into a small number of
GPU dispatches. A 17-value coordinate grid should cost much closer to one render
than to 17 sequential renders.

## Success criteria

An optimized workflow should satisfy all of the following:

- End-to-end rung-3 runtime near or below 10 minutes on the development machine.
- No weakening or redefinition of the independent 80% gate.
- Final NumPy and GPU MR-STFT distances agree within a documented tolerance.
- The selected patch rerenders deterministically from its parameter JSON.
- The 808 profile remains numerically unchanged unless explicitly opted into the
  new refinement path.
