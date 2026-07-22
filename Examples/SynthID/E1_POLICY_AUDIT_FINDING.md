# SynthID E1 Optimizer-Policy Audit Finding

## Status: BLOCKED — development seed 6 remains outside the gate

Investigated 2026-07-13 after the frozen E1 policy failed fresh seeds 6 and 7.
The formal, predeclared E1 result remains PASS (3/5). This audit asks the
stronger question required before E3: can a target-independent basin policy
clear the two known development failures and then generalize to new untouched
seeds?

It cannot yet. The best fixed policy attempted on seed 6 reduced production
loss from `1.359184` to `0.327534`, but its loss ratio is `13.0895%` against
the unchanged cold baseline. The readiness gate is `2%`. Per protocol, the
policy was rejected, seed 7 was not used to tune it further, seeds 8 and 9
remain unopened, and E3 was not started.

## Gate evidence

All values below are the unchanged production log-L1 plus linear-L1 loss.
Ratios use each seed's canonical cold baseline (`2.502263` for seed 6 and
`19.828098` for seed 7).

| Seed | Policy/candidate | Final loss | Ratio | Gate |
|---:|---|---:|---:|:---:|
| 6 | frozen E1 canonical policy | 1.359184 | 54.3182% | FAIL |
| 6 | low-filter anchor, freeze/release/smooth | 1.358517 | 54.2915% | FAIL |
| 6 | spectral-core anchor, joint shape/pw grid, smooth | 0.721602 | 28.8380% | FAIL |
| 6 | preceding candidate plus alternating VCA block | 0.710222 | 28.3832% | FAIL |
| 6 | best of additional deterministic restarts 3–6 | 1.243728 | 49.7041% | FAIL |
| 6 | population search candidate | 0.903712 | 36.1158% | FAIL |
| 6 | population candidate plus 600 smooth epochs | 0.387583 | 15.4893% | FAIL |
| 6 | preceding candidate plus 800 production epochs | **0.327534** | **13.0895%** | **FAIL** |
| 7 | retained high-filter restart, full pw scan/tune/smooth | 2.192974 | 11.0599% | FAIL |

The population experiment was deterministic and audio-only. It searched all
12 transformed parameter coordinates with a diagonal cross-entropy method:
population 48, eight elites, 30 generations, fixed `SplitMix64` seed derived
from the self-inversion seed, initial normalized standard deviation 0.30,
and no hidden-parameter access. Candidate ranking used the already-frozen
smooth log-magnitude L2 objective; selection and every number above used the
production objective. Its implementation was deliberately not promoted after
the failed gate.

## Root cause

This is an optimizer basin-selection failure, not an E0 adjoint failure, an
E2 renderer mismatch, or a voice-capacity failure.

1. Scoring seed 6's hidden parameters through the same production path gives
   `9.023061e-7`, so the exact solution exists in the deployed topology.
2. Seed 6 sits in a dark, tightly coupled corner: hidden `fBase=86.44 Hz`,
   `fAmt=7.19 Hz`, `res=3.04`, `shape=0.6679`, and `pw=0.4647`. The canonical
   policy instead selected a compensated bright/filter basin with
   `fBase=265.97 Hz`, `fAmt=207.16 Hz`, `shape=0.9893`.
3. One-dimensional shape and pw sweeps did not escape that ridge. A joint
   shape/pw grid lowered a spectral-core candidate from `1.406988` to
   `0.909673`, proving the variables must move together, but subsequent
   optimization stopped at `0.710222`.
4. Replacing only the compensated solution's filter with the hidden filter
   worsened loss to `1.479785`; replacing its oscillator and filter worsened
   it to `3.213378`. The target coordinates are not independently beneficial
   while the VCA/drive/filter compensation remains in place.
5. The global population candidate did recover the correct neighborhoods for
   `fBase` (`81.35 Hz`) and `pw` (`0.46565`), yet converged with
   `shape=0.7898`, `res=2.262`, and `fDecay=0.01968 s` and stopped at a 13.09%
   production ratio. A broader initialization distribution alone therefore
   does not solve the coupled ridge.
6. Seed 7 shows the complementary failure. A discarded restart retained its
   correct high-filter basin; a full pw scan moved `pw` to `0.17137` versus
   hidden `0.18205`, but fixed follow-up training still stopped at 11.06%.

The practical conclusion is narrower than “gradient optimization cannot do
this.” Local gradients are correct and real optimization already recovers
three seeds exactly enough to pass. What is missing is a target-independent
continuation or objective that breaks the oscillator/filter/VCA compensation
without knowing the hidden coordinates.

## Experiments rejected

- Low- and high-filter target-independent anchors, with filter-frozen and
  filter-released stages.
- Smooth-first continuation from those anchors.
- Freezing oscillator and filter controls while fitting the spectral core,
  followed by joint release.
- Full one-dimensional shape/pw sweeps and two-/three-dimensional
  shape/pw/outGain grids.
- Alternating oscillator/filter and VCA/drive blocks.
- Four additional deterministic cold restarts.
- Retaining seed 7's discarded high-filter basin and rescanning pulse width.
- Deterministic full-parameter population search followed by both smooth and
  production Adam refinement.

These are negative results, not additions to the canonical policy.

## Reproduction commands

The fresh-seed failure that started the audit:

```sh
.build/release/SynthID rung1 \
  --profile subtractive-bass --seeds 6,7 \
  --out output/e1_subtractive_fresh_seeds_6_7 \
  --epochs 600 --restarts 3 \
  --log-every 600 --checkpoint-every 300 --allow-fail
```

Representative independent basin diagnostics, using the retained restart
index and multidimensional loss-sweep instruments:

```sh
.build/release/SynthID train \
  --profile subtractive-bass --seed 6 --restart-index 5 \
  --target output/e1_subtractive_fresh_seeds_6_7/seed-6/target.wav \
  --out output/e1_policy_diagnostics/seed6_restart5_300 \
  --epochs 300 --log-every 300 --checkpoint-every 300

.build/release/SynthID loss-sweep \
  --profile subtractive-bass \
  --target output/e1_subtractive_fresh_seeds_6_7/seed-6/target.wav \
  --params output/e1_policy_diagnostics/seed6_core_anchor_release/recovered_params.json \
  --param shape --radius 0.25 --points 17 \
  --param2 pw --radius2 0.47 --points2 65 \
  --out output/e1_policy_diagnostics/seed6_core_osc_grid.csv
```

The best population candidate was refined and scored with:

```sh
# Executed from the transient diagnostic revision. The rejected flag was
# removed before commit; its fixed algorithm/settings are recorded above.
.build/release/SynthID train \
  --profile subtractive-bass --seed 6 \
  --target output/e1_subtractive_fresh_seeds_6_7/seed-6/target.wav \
  --out output/e1_policy_diagnostics/seed6_population_search \
  --epochs 1 --smooth-training-loss --population-basin-search \
  --log-every 1 --checkpoint-every 1

.build/release/SynthID train \
  --profile subtractive-bass --seed 6 \
  --target output/e1_subtractive_fresh_seeds_6_7/seed-6/target.wav \
  --initial-params output/e1_policy_diagnostics/seed6_population_search/recovered_params.json \
  --out output/e1_policy_diagnostics/seed6_population_refine_smooth \
  --epochs 600 --smooth-training-loss \
  --log-every 100 --checkpoint-every 600

.build/release/SynthID train \
  --profile subtractive-bass --seed 6 \
  --target output/e1_subtractive_fresh_seeds_6_7/seed-6/target.wav \
  --initial-params output/e1_policy_diagnostics/seed6_population_refine_smooth/recovered_params.json \
  --out output/e1_policy_diagnostics/seed6_population_refine_production \
  --epochs 800 --log-every 100 --checkpoint-every 800

.build/release/SynthID score \
  --profile subtractive-bass \
  --target output/e1_subtractive_fresh_seeds_6_7/seed-6/target.wav \
  --params output/e1_policy_diagnostics/seed6_population_refine_production/recovered_params.json \
  --true-params output/e1_subtractive_fresh_seeds_6_7/seed-6/true_params.json \
  --initial-params output/e1_subtractive_fresh_seeds_6_7/seed-6/initial_params.json \
  --out output/e1_policy_diagnostics/seed6_population_refine_production_score
# score pass=false loss=0.327534 ratio=0.130895
```

## Decision and next experiment

The attempted policy is rejected. Seeds 8 and 9 are still valid holdouts and
must remain sealed. E3 remains blocked.

The next policy hypothesis should explicitly break the compensation ridge,
for example a fixed multi-resolution continuation that first fits observable
summary trajectories (cutoff/brightness and output envelope) before enabling
the full waveform loss. It must be designed and frozen using seeds 6 and 7
only, clear both at the existing 2%/10% bars, and then pass the untouched
seeds 8 and 9 without modification. Merely adding more random restarts or a
wider population is not supported by this audit.
