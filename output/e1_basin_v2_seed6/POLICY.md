# Predeclared stratified basin-search policy v2 (frozen before execution)

Declared 2026-07-17, before scoring any seed-6 candidate at production scale
under v2. Target-independent: no hidden-parameter access at any stage; every
setting below is fixed for all seeds.

Policy v1 (uniform raw-unit prior, single round; see
`output/e1_basin_search_seed6/`) FAILED its gate on seed 6: best elite
17.0982% vs the 2% gate. v2 changes exactly two things, both motivated by
target-independent facts:
 1. Candidates are sampled uniformly in TRANSFORMED coordinates over the full
    declared bounds — the coordinate system the declared E1 generative prior
    (PatchValues.sample) draws from. v1's raw-unit prior made low-fAmt truths
    ~0.2% tail events.
 2. Two stratified Gaussian resampling rounds (derivative-free local search
    across the coupled shape/filter ridge that Adam cannot cross).

## Stage 1 — batched iterated scan
- Round 0: 8192 candidates uniform in transformed coordinates, SplitMix64
  stream `seed * 0x9E3779B97F4A7C15 + 0xE1BA51`.
- Rounds 1-2: per stratum, top-8 parents each spawn 96 Gaussian children,
  sigma = 0.08 then 0.04 of each transformed span, clamped to bounds
  (10 strata x 8 x 96 = 7680 children per round; ~23.5k evaluations total).
- Rendered through the batched subtractive voice at B=256 (validated in
  `output/batch_bench`, max abs diff 6.17e-5 vs serial).
- Scored on CPU with the compare.py `mrstft` objective (windows
  256/512/1024/2048, hop w/4, Hann, log(mag+1e-3), L1). Swift port verified
  against compare.py to 7 significant digits on seed-6 audio.

## Stage 2 — stratified elite selection
- Strata: fBase octave bands [60,120,240,480,960,2000] x shape halves
  (<0.5, >=0.5) = 10 strata; best archive candidate per stratum, plus the 2
  best overall not already selected (up to 12 elites).
- Rationale: the E1 policy audit showed global ranking collapses into the
  dominant bright/compensated basin; strata guarantee dark corners survive.

## Stage 3 — refinement (audit's best schedule, unchanged)
- Per elite: 600 smooth-loss epochs from the elite, then 800 production
  epochs from the smooth result (`scripts/refine_elites.sh`).
- Final selection by production `finalLoss` only.

## Gate
- `finalLoss / coldBaseline <= 0.02`, cold baseline unchanged from the
  audit: seed 6 = 2.502263, seed 7 = 19.828098.
- Order: seed 6 first; if PASS, seed 7 with identical settings; only if both
  pass, fresh seeds 8 and 9 (still unopened), then E3.
