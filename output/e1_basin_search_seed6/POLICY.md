# Predeclared stratified basin-search policy (frozen before execution)

Declared 2026-07-16, before scoring any seed-6 candidate at production scale.
Target-independent: no hidden-parameter access at any stage; every setting
below is fixed for all seeds.

## Stage 1 — batched wide scan
- 8192 candidates from `BatchBench.randomCandidates` (the declared sampling
  prior), SplitMix64 stream `seed * 0x9E3779B97F4A7C15 + 0xE1BA51`.
- Rendered through the batched subtractive voice at B=256 (validated in
  `output/batch_bench`, max abs diff 6.17e-5 vs serial).
- Scored on CPU with the compare.py `mrstft` objective (windows
  256/512/1024/2048, hop w/4, Hann, log(mag+1e-3), L1). Swift port verified
  against compare.py to 7 significant digits on seed-6 audio.

## Stage 2 — stratified elite selection
- Strata: fBase octave bands [60,120,240,480,960,2000] x shape halves
  (<0.5, >=0.5) = 10 strata; best candidate per stratum, plus the 2 best
  overall not already selected (up to 12 elites).
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
