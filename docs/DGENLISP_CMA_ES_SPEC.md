# Spec: CMA-ES global search for `dgenlisp train`

Status: PROPOSED — 2026-08-14

## Summary

Add a product-facing, gradient-free global-search mode to `dgenlisp train`
using Covariance Matrix Adaptation Evolution Strategy (CMA-ES), evaluated by
the generic tensor-lane Lisp voice path.

The intended product pipeline is:

```text
user seed + declared parameter bounds
  -> tensor-batched CMA-ES basin search
  -> retain the supplied seed and several global candidates
  -> optional local, top-K, and winner-only Adam refinement
  -> independently score and report enabled local versus global outcomes
```

CMA-ES is the primary basin finder. Gradients become an optional final polish,
not a prerequisite for useful parameter fitting.

## Motivation and measured anchor

The existing direction trainer runs one seeded Adam trajectory and one
transformed-midpoint trajectory. That is a weak global policy for bounded,
coupled, non-convex synthesizer parameters.

The tensor-lane multistart prototype in commit `bb44b82` established that an
unmodified complicated Lisp patch can be evaluated as independent candidates:

- voice: real Korg1, including phasors, PolyBLEP oscillators, analytic ADSRs,
  dual coupled-history SVFs, saturation, noise, and 34 learnable parameters;
- render length: 65,536 frames;
- population: 64 candidates in B=16 chunks;
- forward render plus independent CPU MR-STFT scoring: **1.44 s total**,
  **22.6 ms/candidate**;
- five B=16 Adam steps: **45.55 s**;
- seed score: 0.5654;
- best initial population score: 0.4855;
- best score after five Adam steps: 0.4812.

At this operating point, thousands of forward fitness evaluations cost about
the same as a handful of batched backward steps. Forward and backward now both
support the complete phasor + SVF topology after `cd3aa3b`; the earlier tensor
temporal-gradient detach workaround has been removed and composition is pinned
by `TensorTemporalGradientCompositionTests`.

A covariance-adapting forward optimizer is therefore attractive because of
its search economics and global behavior, not because gradients are incomplete.
Complete gradients remain available as the final local-polish stage.

## Goals

1. Search arbitrary bounded continuous dgenlisp parameters without requiring
   gradients.
2. Evaluate one population through the existing generic tensor-lane lowering,
   with isolated recurrent state per lane.
3. Learn coupled parameter directions such as drive/gain,
   cutoff/envelope-amount, and resonance/shape.
4. Always preserve and report the user's supplied seed score; separately run
   and report its local Adam trajectory when `--local-epochs` is nonzero.
5. Optionally refine the best global candidates with complete-gradient scalar
   or tensor-batched Adam.
6. Make runs deterministic and leave a complete artifact trail.
7. Compare against seeded+midpoint and random/stratified multistart at equal
   wall-clock and equal forward-evaluation budgets.

## Non-goals for v1

- Discrete selector optimization.
- Categorical patch topology search.
- Polyphonic or multi-note fitness suites in one run.
- Noisy per-lane excitation or per-lane random streams.
- Replacing Adam for final local convergence.
- GPU implementation of the CMA covariance update; its CPU cost is negligible
  at the expected 20-100 dimensions.
- Large-dimensional neural-network training.

## User-facing interface

Proposed options:

```text
dgenlisp train ...
  --search cma-es
  --cma-generations 12
  --cma-population 64
  --cma-sigma 0.20
  --cma-seed 1
  --cma-forward-batch 64
  --local-epochs 300
  --cma-continue 3
  --cma-refine-epochs 50
  --cma-refine-mode auto
  --cma-final-epochs 0
```

Defaults when `--search cma-es` is selected:

| Option | Default | Meaning |
|---|---:|---|
| `--cma-generations` | 12 | Maximum generations |
| `--cma-population` | `max(32, 4 + floor(3*log(D)))`, rounded for batching | Candidates per generation |
| `--cma-sigma` | 0.20 | Initial standard deviation in normalized transformed coordinates |
| `--cma-seed` | 1 | Deterministic random seed |
| `--cma-forward-batch` | population size, capped by safe memory policy | Tensor lanes per render |
| `--local-epochs` | value of `--epochs` | Independent seeded Adam fallback; zero disables it |
| `--cma-continue` | 3 | Number of diverse global candidates receiving short continuation |
| `--cma-refine-epochs` | value of `--epochs` | Adam epochs per continued candidate; zero disables top-K refinement |
| `--cma-refine-mode` | `auto` | `scalar`, `batched`, or occupancy-aware automatic selection |
| `--cma-final-epochs` | 0 | Long scalar Adam continuation of the selected global winner; zero disables it |

`--search legacy` preserves seeded+midpoint behavior. The existing
`--multistart-candidates` prototype remains an experimental baseline until CMA
supersedes or incorporates it.

## Parameter coordinates

CMA operates in the trainer's normalized transformed coordinates:

```text
z_i in [0, 1]
```

For a linearly transformed parameter:

```text
natural = min + z * (max - min)
```

For a wide positive parameter (`min > 0 && max/min >= 8`):

```text
natural = exp(log(min) + z * (log(max) - log(min)))
```

This gives every knob comparable scale and avoids treating 20-18,000 Hz as a
linear search interval.

Only parameters classified as learnable by `TrainPlanner` participate.
Generated, hidden, unbounded, unreachable, and unsupported parameters remain
frozen exactly as in normal training.

## Boundary handling

Do not clamp samples directly; clipping creates probability mass on parameter
bounds and distorts covariance adaptation.

Use repeated reflection into `[0,1]`:

```text
... -0.2 -> 0.2
...  1.3 -> 0.7
```

After reflection, apply a small numerical inset only where the natural
parameter's downstream graph is undefined exactly at a bound.

Record the fraction of reflected coordinates per generation. Excessive
reflection is a diagnostic that sigma or the mean is poorly placed.

## CMA-ES algorithm

Implement canonical full-covariance CMA-ES in `Float64` on CPU. Candidate
vectors convert to `Float` only at the tensor render boundary.

State for dimension `D`:

- mean `m[D]`;
- global step size `sigma`;
- covariance `C[D,D]`;
- covariance factor/eigendecomposition `B`, `D_sqrt`;
- evolution paths `p_c[D]`, `p_sigma[D]`;
- generation counter and deterministic RNG.

For each generation:

1. Draw `lambda` standard-normal vectors `y_k`.
2. Form `x_k = m + sigma * B * D_sqrt * y_k`.
3. Reflect `x_k` into `[0,1]`.
4. Inject required anchors, replacing the last sampled slots:
   - generation 0: exact user seed and transformed midpoint;
   - every generation: all-time best candidate and current mean;
   - anchors are eligible for ranking and recombination.
5. Tensor-render and independently score all candidates.
6. Stable-sort by `(fitness, candidateIndex)`.
7. Recombine the best `mu` candidates with standard positive logarithmic
   weights.
8. Update `p_sigma` and cumulative step-size adaptation.
9. Update `p_c`.
10. Apply rank-one and rank-mu covariance updates.
11. Symmetrize `C`, floor eigenvalues to a small positive epsilon, and refresh
    its eigendecomposition at the canonical amortized cadence.
12. Update `sigma`, with finite-value and maximum-radius guards.

Use the standard Hansen CMA-ES coefficient formulas derived from `D`,
`lambda`, `mu`, and `mu_eff`. Do not invent synthesizer-specific covariance
rules in v1.

### Numerical safeguards

- All CMA state and score ordering use `Double`.
- Reject non-finite candidates before rendering.
- Assign non-finite audio or fitness `+infinity` without aborting the whole
  generation.
- Enforce covariance symmetry after every update.
- Floor covariance eigenvalues at `1e-14` and cap condition number at `1e14`.
- Stop if sigma is non-finite or all covariance axes collapse below tolerance.

## Initialization policy

Default mean is the supplied seed, not midpoint. This respects the semantic
meaning of direction learning and gives CMA a useful prior.

The first generation still injects:

- exact seed;
- transformed midpoint;
- several deterministic stratified immigrants if population size permits.

Optional future policies may initialize from multiple known presets or a prior
bank, but they must not silently replace the user's seed.

Initial covariance is identity in normalized transformed coordinates. The
single global sigma therefore initially means the same fraction of every
knob's declared range.

## Fitness

### v1 objective

Use `TrainSpectralScorer`, the independent vDSP multi-resolution log-magnitude
STFT score:

- windows 256, 512, 1024, 2048 where they fit;
- hop = window/4;
- symmetric Hann;
- `log(magnitude + 1e-3)`;
- L1 mean over bins and frames, summed over resolutions.

This scorer is independent of autograd and returns one fitness per lane.

### Determinism

Fitness must be stable across generations:

- all lanes receive the same deterministic noise source;
- target preparation and excitation remain fixed;
- recurrent state starts from zero for every candidate render;
- no candidate may inherit another lane's state;
- rerendering the same candidate in a different batch position must match
  within the existing forward-conformance tolerance.

### Future product objectives

The artifact schema should permit adding weighted components later:

- waveform/envelope distance;
- pitch error;
- loudness trajectory;
- transient alignment;
- regularization toward the user seed;
- multi-note or multi-velocity aggregate scores.

CMA sees only the final scalar fitness, so these do not require optimizer
changes.

## Tensor-batched evaluation

Reuse the batch lowering added in `bb44b82`:

- each parameter is a `[B]` tensor;
- scalar inlets and deterministic excitation broadcast to lanes;
- every phasor has independent `[B]` phase state;
- every Lisp `make-history` cell becomes `TensorHistory(shape: [B])`;
- arithmetic and shaping remain in the SignalTensor domain;
- output is frame-major `[frames, B]` and deinterleaved for scoring.

Population sizes larger than the safe lane count are evaluated in fixed-size
chunks. Pad only the final chunk, ignore padded scores, and keep compilation
shape stable for cache reuse.

Add separate timing for:

- graph build/compile;
- GPU forward render;
- readback/deinterleave;
- CPU scoring;
- CMA update.

The current 1.44 s number combines all forward rendering and CPU scoring and
must not be presented as GPU-only throughput.

## Hybrid gradient refinement

After CMA terminates:

1. Optionally preserve an independent scalar seeded trajectory for
   `--local-epochs`; zero gives a genuinely gradient-free CMA-only run.
2. Select `--cma-continue K` globally best candidates with a diversity floor.
3. Optionally run each through scalar or batched Adam for
   `--cma-refine-epochs` and independently reject regressions.
4. Independently select the global winner.
5. Optionally run that winner through scalar Adam for
   `--cma-final-epochs`, again retaining its pre-Adam input on regression.
6. Independently compare the global winner with the local fallback, when
   enabled.
7. Report:
   - original seed;
   - locally refined seed;
   - CMA best before Adam;
   - each CMA candidate after Adam;
   - globally selected final result.

Both refinement paths support complete phasor + SVF and accumulator + SVF
temporal gradients after `cd3aa3b`. Use scalar refinement for a small top-K;
use tensor-batched refinement when enough diverse elites/restarts can be packed
to amortize backward execution. Selection must still use the independent
forward fitness, not the reduced batched training loss.

If refinement is disabled, CMA's best candidate is still a valid final product
result.

## Stopping and restart policy

Stop on the first of:

- generation limit;
- wall-clock budget;
- target fitness;
- no meaningful best-fitness improvement for 5 generations;
- sigma/covariance collapse;
- unrecoverable numerical failure.

A first implementation may stop without restart. A follow-up may add IPOP-CMA
restart behavior: restart from the all-time best or a broad immigrant mean with
larger population. Restart policy must be explicit in artifacts.

## Reporting and artifacts

Write `cma_es_report.json` containing:

```json
{
  "algorithm": "cma-es",
  "version": 1,
  "dimension": 34,
  "population": 64,
  "generations_completed": 12,
  "evaluations": 768,
  "seed": 1,
  "initial_sigma": 0.2,
  "stop_reason": "generation_limit",
  "generation_trace": [
    {
      "generation": 0,
      "best": 0.48,
      "median": 0.71,
      "mean": 0.83,
      "sigma": 0.19,
      "condition_number": 3.2,
      "reflected_fraction": 0.04,
      "forward_seconds": 1.4,
      "cma_update_ms": 0.2
    }
  ],
  "seed_score": 0.56,
  "cma_best_score": 0.31,
  "best_z": [],
  "best_params": {},
  "continued_candidates": [],
  "local_seed_outcome": {},
  "global_outcome": {}
}
```

Also preserve the best candidate from every generation or whenever all-time
fitness improves. A run must be resumable from serialized CMA state without
changing subsequent samples or rankings.

The existing NDJSON protocol may emit a `stage` named `cma-es`. Do not overload
`EpochEvent` with an entire population. Product UI protocol changes, if needed,
should be a separately versioned event type.

## Correctness tests

### CMA math

1. Deterministic sampling golden for fixed state and seed.
2. Sphere function convergence.
3. Rotated ellipsoid convergence, proving covariance adaptation.
4. Reflection never emits values outside `[0,1]`.
5. State serialize/resume produces byte-identical subsequent candidates.
6. Covariance remains symmetric positive definite.

### Batched voice

1. B=1 batch output matches scalar post-lowering output.
2. B=8 identical candidates produce identical audio.
3. B=8 distinct candidates match eight independent scalar forward renders.
4. Candidate score is invariant to lane position and chunk size.
5. Tensor-history state is isolated across lanes.
6. Korg1 forward conformance includes phasor + dual SVF topology.

### End-to-end recovery

Use known-config round trips for at least:

- simple oscillator/envelope patch;
- TB303-basic;
- Korg1;
- one patch with drive/gain and cutoff/envelope ridges.

For each, record parameter recovery, independent score, render quality, wall
clock, and evaluation count.

## Benchmark matrix

Compare at equal wall-clock budgets and, separately, equal forward-evaluation
budgets:

1. seeded Adam only;
2. seeded + midpoint Adam;
3. one-shot stratified population;
4. stratified population + short batched Adam;
5. CMA-ES only;
6. CMA-ES + scalar Adam polish.

Run at least five deterministic hidden configurations per patch and three
optimizer seeds per configuration. Report median, quartiles, failure rate, and
best/worst cases—not only the best run.

Primary product metric:

```text
independent final score versus wall-clock time
```

Secondary metrics:

- fraction beating the supplied seed;
- fraction recovering the correct basin;
- known-parameter error in transformed coordinates;
- initial versus final score correlation;
- candidate evaluations per second;
- audible or render-level regressions.

## Performance targets

On the machine that produced the Korg1 anchor:

- 64-candidate, 65,536-frame generation including scoring: <= 2.0 s;
- CMA CPU update for D <= 64: < 10 ms/generation;
- no per-generation Metal recompilation after first fixed-shape batch;
- B=64 candidate output remains within a documented memory budget;
- 768-evaluation 12-generation Korg1 search: target <= 30 s before optional
  scalar polish.

The last target requires separating and reducing CPU scoring/readback overhead;
it is a goal, not an extrapolated result.

## Implementation plan

### M1 — Standalone optimizer

- Add a tested `CMAES` state type with canonical updates, bounds reflection,
  deterministic RNG, and serialization.
- Validate on sphere and rotated ellipsoid functions.

### M2 — Forward-search harness

- Drive generic batched Lisp evaluation from CMA candidates.
- Split render, readback, and CPU-score timing.
- Produce `cma_es_report.json` and generation checkpoints.
- Pass simple-patch and Korg1 forward conformance.

### M3 — `dgenlisp train` integration

- Add `--search cma-es` and options.
- Preserve local seed versus global candidate semantics.
- Add top-K scalar/batched continuation and final independent selection.
- Support cancellation and wall-clock limits.

### M4 — Product decision benchmark

- Run the full benchmark matrix.
- Select production defaults by median score/time, not one favorable target.
- Decide whether legacy midpoint and one-shot multistart remain exposed.

## Acceptance criteria

Done when:

1. Canonical CMA-ES passes deterministic and rotated-objective tests.
2. Korg1 runs through at least 10 generations without gradients or patch-
   specific voice code.
3. Every candidate remains lane-isolated and score-invariant to packing.
4. Local seed and global search outcomes are separately reported.
5. Optional scalar Adam improves or preserves independently scored CMA output;
   regressions select the pre-Adam candidate instead.
6. Equal-wall-clock results across the known-config matrix show whether CMA-ES
   materially improves median final score or time-to-quality.
7. The result, including failures and negative findings, is documented in the
   bead and this spec.
