# `dgenlisp train` — implementation notes, deviations, punts

Status: Phases A–C landed (protocol layer, real plan event, E4 direction
trainer). Companion to the eseq repo's `docs/patch-learn-spec.md` (rev 1).
Items marked **SPEC-SYNC** need the spec updated to match (or the code
changed after discussion).

## Spec deviations (flag loudly)

1. **SPEC-SYNC — per-group LRs**: spec §7 says "Adam with per-group LRs in
   transformed coordinates". Generic lisp patches have arbitrary param names,
   so there are no groups to key off. v1 uses a single global transformed-
   coordinate LR of 2e-2 (= 2x the legacy production toneLR, the
   BATCH_REFINE_FINDING recommendation), log-reparam for params with
   `min > 0 && max/min >= 8`, per-param grad clip 1.0, cosine decay, bounds
   projection. Grouping could later come from `@group`/`@unit` attributes.
2. **SPEC-SYNC — unsupported element schema**: spec §4 shows
   `"unsupported":[]` without an element shape. Implemented as
   `{name, reason}` (same as frozen). Also used for inlets outside the
   excitation convention (reason `input-not-in-excitation-convention`).
3. **SPEC-SYNC — abs_distance source**: v1 reports the best training loss
   (MR-STFT, frozen SPEC.md §4 config) as `abs_distance`, not an independent
   CPU-scorer re-evaluation. Porting `CPUSpectralScorer` (BasinSearch.swift,
   vDSP) into the shared target would give the independent number; the host's
   own round-trip verification (spec §8) is the real defense either way.
4. **Basin check is serial, not background**: the cold restart (deterministic
   transformed-midpoint init) runs after the seeded run at the same epoch
   budget; `wrong_neighborhood` iff cold best < 0.75 x seeded best. Threshold
   was not pinned by the spec.
5. **`--plan-only`** (extension): emits the real plan then terminates with an
   `error` event ("plan-only: no training performed"), exit 1 — a `result`
   is reserved for actual training. Useful as a host preflight.

## Known limitations / punts (filed here, not silently dropped)

- **C backend cannot train**: spectral-loss BPTT kernels fail to compile on
  the C backend (`use of undeclared identifier 'tape'` — the C renderer never
  declares the tape buffer this configuration needs; same family as the
  pre-existing `testShrinkWithScalarOp` C failure). `train --backend c`
  reports the compile error as a protocol-clean error event. Training and the
  E2E test are Metal-only; the protocol layer (Phase A) and plan layer
  (Phase B) are Metal-free. `train-render` (forward only) works on both
  backends.
- **Per-epoch recompile**: each epoch does `LazyGraphContext.reset()` +
  re-evaluate + full Metal recompile (~0.5 s/epoch at 8192 frames). Correct
  by construction (no stale-nodeId class of bugs) but well above the
  ~0.27 s/epoch production trainer. Fix later by fingerprint-caching across
  resets or reusing one graph per phase.
- **Stereo targets**: mono-summed via `AudioFile.load(mono: true)` (spec §9
  open question; pick made explicit here).
- **Poly patches**: nothing voice-aware; the patch is evaluated exactly once
  (1 voice). Spec §9 open question stands.
- **Ring mod is not detected**: only phasor-sync (reset driven by another
  oscillator) is refused. Ring mod is multiplication of oscillators and is
  indistinguishable from legitimate AM at graph level; per the feasibility
  doc it stays a declared non-goal rather than a detected refusal.
- **Tensor/batched phasors**: the freeze analysis walks scalar `.phasor`
  nodes. Tensor-lane phasors (batch lowering) are not classified — fine for
  v1 single-voice patches.
- **Checkpoint renders spawn `train-render`**: realize() must not interleave
  with backward() in one process (SPEC.md §5 hard-won rule), so preview WAVs
  re-invoke the executable on `lowered.lisp`. A failed render skips the
  checkpoint event (logged to stderr) rather than killing the job; the final
  render failing IS fatal.
- **`in` channel-only inlets**: inlets are matched by `@name`
  (gate/trigger/pitch/velocity). Unnamed `(in N)` inlets are refused; a
  channel-number convention could be added if eseq patches rely on it.

## Test map

- `Tests/DGenTrainProtocolTests` — Metal-free: event round-trip/goldens,
  fake-trainer transcript golden (`Fixtures/fake_transcript.golden.ndjson`,
  regenerate with `DGEN_UPDATE_GOLDEN=1`), excitation measurement on
  synthesized drum/sustained assets.
- `Tests/DGenLispTests/TrainCLITests.swift` — subprocess protocol tests:
  happy path, poisoned stdout, crash path, SIGTERM, plan-only, and the
  Python mock host (`scripts/consume_train_stream.py`).
- `Tests/DGenLispTests/TrainPlannerTests.swift` — lowering verdict incl.
  macro transparency and inlet rewriting.
- `Tests/DGenLispTests/TrainE2ETests.swift` — Metal-gated rung-1-style
  self-consistency run through the real trainer.
