# E3 — sealed-seed validation (seeds 8/9): FAIL on the declared gate

Status: COMPLETE 2026-07-21. Gate declared in `E3_GATE.md` BEFORE any
seed-8/9 artifact existed; pipeline run untouched, no per-seed tuning, no
re-runs. Artifacts: `output/e3_seeds_8_9/` (generation),
`output/schedule_spec_e3/` (search + refinement + verdict checks).

## Verdict

| Seed | Cold baseline | Best ratio | Params in tolerance | Bar 1 (≤2%) | Bar 2 (12/12) | Seed verdict |
|---|---:|---:|---:|---|---|---|
| 8 | 3.006385 | 2.2692% (elite-09) | 8/12 | FAIL (by 0.27 pt) | FAIL | **FAIL** |
| 9 | 2.669603 | **0.2310%** (elite-07) | 11/12 | **PASS** | fail | **PASS** |

E3 requires both seeds to pass. **E3 FAILS as declared.** Reported as a
failure per the declaration's no-adjustment rule.

## Seed 8 detail (the miss)

Global best (elite-09 lane 59, prod loss 0.068219): shape/pw/res/outGain/
fBase/releaseTime recovered to 3–4 digits; the four misses are ONE
compensation cluster — fAmt 15.4 vs truth 137.9 traded against fDecay
0.204 vs 0.016 (the filter envelope collapsed toward a quasi-static
cutoff), plus a sustain 0.542-vs-0.710 / decayTime trade in the VCA. This
is the fAmt·fDecay product degeneracy class, the same family as seed 7's
(passing) residual and seed 6's res/shape ridge: a near-null direction of
the production objective holding a small loss (2.3% of baseline).
A/B material: `output/schedule_spec_e3/seed-8/best_lane.wav` vs
`target_ref.wav` (rendered before the verdict; ear evidence, not gate
input).

## Seed 9 detail

Global best (elite-07 lane 43, prod loss 0.006166 = 0.2310%): everything
except sustain within tolerance, most params to 4 digits (fAmt 711.76 vs
711.66, fBase 195.64 vs 195.70, pw exact). Three elites under 1%. The
population mechanism again did real work (elite-07's unjittered lane
0.00275 → jitter 0.00017; elite-04 similar).

## Aggregate picture across all four evaluated seeds

Untouched frozen pipeline (v2 basin search + S4 batched refinement):

| Seed | Best ratio | Recovery | Character |
|---|---:|---:|---|
| 6 (design) | 7.70% | 12/12 | res/shape ridge, audio indistinguishable per user A/B |
| 7 (design) | 0.1393% | near-exact | clean pass |
| 8 (sealed) | 2.2692% | 8/12 | fAmt·fDecay + sustain/decayTime compensation |
| 9 (sealed) | 0.2310% | 11/12 | clean pass |

Two clean passes, two near-misses, and every miss is the same phenomenon:
quasi-degenerate parameter directions (res/shape, fAmt·fDecay,
sustain/decayTime) where the production objective — and, on seed 6,
verified human ears — barely distinguish the compensated point from
truth. No seed shows search failure (wrong basin) or optimizer failure
(divergence, stall above 8%): basin discovery and refinement are solved;
what remains is objective identifiability of near-null directions.

## What would move the number (not run — E3 is closed)

- Identifiability-aware losses: envelope-trajectory or time-domain terms
  that separate fAmt·fDecay (the spec's S2 idea, unbuilt), or a
  high-resolution spectral term (blocked on a chunked/half-precision
  Metal FFT for 4096+ windows).
- Or: accept that param-exact recovery of near-null directions is the
  wrong bar for the product goal ("train patched synth to sample"), where
  the deliverable is the sound, not the hidden parameters — a bar three
  of four seeds (and arguably all four, by ear) already meet.

That choice belongs to the next phase, with this failure honestly on the
record.
