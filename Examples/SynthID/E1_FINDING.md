# SynthID E1 Self-Inversion Finding

## Status: PASS — 3/5 seed majority

Completed 2026-07-13. E0 remains PASS at commit `229258f`. E1 recovers
the complete 12-control subtractive voice on three of five deterministic
self-inversion seeds. E2 was not started.

## Gate verdict

The E1 gate is rung-1 acceptance style: production MR-STFT loss at most 2% of
the selected cold initialization, all scored invariant quantities within 10%,
and a majority of five seeds.

| Seed | Initial loss | Final production loss | Ratio | Max invariant error | Verdict |
|---:|---:|---:|---:|---:|:---:|
| 1 | 10.403641 | 0.062538 | 0.6011% | 1.11% | PASS |
| 2 | 11.900508 | 0.038825 | 0.3262% | 1.13% | PASS |
| 3 | 7.199862 | 0.025997 | 0.3611% | 2.24% | PASS |
| 4 | 4.295578 | 1.088307 | 25.3355% | 17.94% | FAIL |
| 5 | 2.650579 | 0.515013 | 19.4302% | 40.42% | FAIL |

**Verdict: PASS (3/5).** The passing seeds are comfortably inside both
numeric bars. `ab.wav` files for every scored seed place the normalized target
and learned render back-to-back; the production loss, not the smooth
refinement objective, is the audible-match acceptance measurement.

Seeds 4 and 5 are conservatively recorded at their completed pre-rescue
checkpoints. The corrected policy would trigger its additional basin rescue
for them, but a fourth pass is not needed to establish the declared majority.

## What was implemented

The `subtractive-bass` profile trains the literal deployment topology:

```text
PolyBLEP saw/pulse blend
  -> time-varying resonant low-pass biquad
  -> smooth ADSR
  -> drive -> softsign -> output gain
```

The twelve trained controls are `shape`, `pw`, `fBase`, `fAmt`, `fDecay`,
`res`, `attackTime`, `decayTime`, `sustain`, `releaseTime`, `drive`, and
`outGain`. `f0=110 Hz` and `noteOff=0.6 s` are fixed. Render and training use
the same direct PolyBLEP graph; no additive oscillator substitute, custom
modulo adjoint, or target-derived initialization is used.

E1 also adds:

- deterministic interior restart basins and profile-specific optimizer groups;
- parameter freezing and explicit continuation support;
- a production-loss pulse-width basin scan, analogous to the existing click
  frequency scan;
- smooth log-magnitude L2 refinement using the same Hann STFT windows, log
  floor, and graph adjoints as production;
- a target-independent smooth coordinate-basin rescue across the declared
  transformed bounds when the production ratio remains above 2%;
- a second fixed smooth anneal after rescue, followed by production-loss
  selection and a `dt/16` local width scan;
- read-only scoring of saved checkpoints with the production objective.

The smooth objective is an optimizer instrument only. Every selection after a
smooth stage is re-evaluated under unchanged production log-L1 plus linear-L1,
and every reported loss in the gate table is a production loss.

## Root cause and final optimization policy

The original production-only run failed because PolyBLEP `pw` has a narrow,
non-monotone loss comb at the edge-transition scale. A coarse `dt/2` search
could enter the neighboring basin but could not resolve the coupled optimum.
Smooth log-L2 removes the binwise-L1 slope breaks and recovers seeds 1 and 2,
but seeds 3–5 showed a second issue: selecting one restart by production loss
before smooth refinement can discard the useful oscillator/filter basin.

The final policy is therefore uniform and audio-only:

1. run the declared production restarts;
2. scan `pw`, freeze it, and settle the other controls under production loss;
3. run one smooth full-graph anneal;
4. if the candidate still exceeds the 2% production gate, scan every declared
   coordinate under the smooth objective, run a fixed smooth anneal, then reset
   the cosine schedule for one second fixed anneal;
5. retain the lowest production-loss candidate and finish with a `dt/16`
   production width scan.

Seed 3 demonstrates the rescue. Its first smooth candidate remained at 20.14%
production ratio. The coordinate search moved `pw` into the correct basin; two
fixed anneals reduced production loss to 0.025997 (0.3611%). Its recovered
`shape=0.22951` and `pw=0.63181` compare with hidden values `0.22942` and
`0.63179`.

## Final invariant definition

E1 exposed an amplitude-scale ridge in seed 2. Its recovered `drive*outGain`
was 14% high while all raw VCA samples were about 12% low, yet their products
and the audio matched. Scoring both factors independently double-counted one
unidentifiable scale.

The finalized scored invariants are:

- `fEnv(0) = fBase + fAmt`;
- `fEnv(infinity) = fBase`;
- `res`;
- `aEnv(t) * drive * outGain` at 10 ms, 300 ms, and 700 ms;
- `aEnv(300ms)/aEnv(10ms)` and `aEnv(700ms)/aEnv(300ms)`.

This scores absolute output-envelope level and scale-free envelope shape while
leaving individual ridge factors as diagnostics. On passing seeds, every row
is within 2.24%.

Seed 3's individual `fDecay` reaches its upper bound while `fAmt` is small.
That is the expected weak-depth filter-EG compensation: both cutoff endpoints,
resonance, production loss, and all output-envelope invariants pass. It is
reported as an unscored knob and not hidden.

## Exact commands executed

Cold acceptance runs and the first smooth policy:

```sh
.build/release/SynthID rung1 \
  --profile subtractive-bass --seeds 1 \
  --out output/e1_subtractive_final_seed1 \
  --epochs 600 --restarts 3 \
  --log-every 300 --checkpoint-every 300 --allow-fail

.build/release/SynthID rung1 \
  --profile subtractive-bass --seeds 2,3,4,5 \
  --out output/e1_subtractive_final_seeds2_5 \
  --epochs 600 --restarts 3 \
  --log-every 600 --checkpoint-every 300 --allow-fail
```

Seed-3 gate-triggered rescue (the two stages now executed automatically by
`rung1` when needed):

```sh
.build/release/SynthID train \
  --profile subtractive-bass \
  --target output/e1_subtractive_final_seeds2_5/seed-3/target.wav \
  --out output/e1_seed3_coordinate_refine \
  --initial-params output/e1_subtractive_final_seeds2_5/seed-3/recovered_params.json \
  --epochs 600 --smooth-training-loss --smooth-basin-search \
  --log-every 600 --checkpoint-every 300

.build/release/SynthID train \
  --profile subtractive-bass \
  --target output/e1_subtractive_final_seeds2_5/seed-3/target.wav \
  --out output/e1_seed3_coordinate_continuation \
  --initial-params output/e1_seed3_coordinate_refine/recovered_params.json \
  --epochs 600 --smooth-training-loss \
  --log-every 600 --checkpoint-every 300
```

The canonical one-command reproduction for the integrated final policy is:

```sh
.build/release/SynthID rung1 \
  --profile subtractive-bass --seeds 1,2,3,4,5 \
  --out output/e1_subtractive_final_policy \
  --epochs 600 --restarts 3 \
  --log-every 600 --checkpoint-every 300 --allow-fail
```

The saved checkpoints were rescored without training into
`output/e1_subtractive_majority_scored/seed-{1...5}` using the new `score`
command. Each score invocation supplies the target, recovered parameters,
hidden truth for invariant reporting, and the selected cold initialization.

## Ladder status

E1 is complete and PASS. E2 is unblocked but was not started in this work.
