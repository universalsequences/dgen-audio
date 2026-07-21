# E3 gate declaration (frozen BEFORE unsealing seeds 8/9)

Declared 2026-07-21, after the schedule-continuation experiments
(`SCHEDULE_FINDING.md`) and before generating or scoring any seed-8/9
artifact. User-approved revision of the 2% ratio-only gate.

## Pass criterion (per seed)

A seed PASSES if its best candidate satisfies **either**:

1. **Ratio bar**: production `finalLoss / coldBaseline <= 2%`, where the
   cold baseline is the production loss of that seed's own
   `initial_params.json` against its target (method validated by
   reproducing seed-6's recorded 2.502263), **or**
2. **Recovery bar**: all 12 subtractive-bass surface params within the
   declared per-param spec tolerances (transformed-space abs error / span,
   `KickParamSpecs` tolerances — the same check `batch-refine
   --true-params` reports).

Rationale (evidence from the design seeds, both fully documented before
this declaration): seed 7 passes bar 1 outright (0.1393%); seed 6 fails
bar 1 at 7.70% solely on the res/shape compensation ridge while passing
bar 2 (12/12) with best-lane audio the user could not distinguish from the
target in A/B. The loss ratio alone is stricter than the perceptual bar in
ridge-bearing corners of the parameter space.

## E3 protocol (untouched pipeline, no per-seed tuning)

Per seed s in {8, 9}:

1. Generate target + hidden truth with the standard rung1 generation path
   (`PatchValues.sample(seed: s)`, parameter-backed render, standard
   normalization). Truth is not read until scoring.
2. Cold baseline: production loss of the seed's `initial_params.json`.
3. Basin search: the frozen v2 stratified policy, defaults unchanged,
   `--seed s` only.
4. Refinement: `batch-refine --mode polish --schedule s4` (LR ×2.0 group
   multiplier, 150 smooth + 500 production, cosine per phase), 6 restarts
   per elite at jitter σ0.05, per-lane CPU mrstft selection, production
   re-scoring of winners.
5. Verdict per the pass criterion above. E3 passes if BOTH seeds pass.

No parameter of any stage may be adjusted after seeing seed-8/9 results;
a failure is reported as a failure.
