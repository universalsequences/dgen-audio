# DDSP Paper Alignment Plan (DGEN)

## Why this doc
This is a concrete gap analysis between:
- the original DDSP setup from the Google Magenta ICLR 2020 work, and
- the current `Examples/DDSPE2E` implementation in this repo.

It also proposes a multi-stage implementation plan that is DDSP-faithful where it matters, while using DGEN-native temporal/state operators before jumping to a full GRU.

## Ground truth: what original DDSP actually used

### 1) Decoder architecture
Paper-era Magenta config and code use:
- `RnnFcDecoder`: per-input FC stacks + recurrent core + output stack.
- Typical config for paper-style runs:
  - `rnn_channels = 512`
  - `rnn_type = 'gru'`
  - `ch = 512`
  - `layers_per_stack = 3`
  - `input_keys = ('ld_scaled', 'f0_scaled')` (or with `z` for autoencoder)
- Source refs:
  - `ddsp/training/gin/models/solo_instrument.gin`
  - `ddsp/training/gin/models/ae.gin`
  - `ddsp/training/decoders.py`

### 2) Synth controls and parameterization
- Harmonic synth:
  - predicts `amps` and `harmonic_distribution`
  - both pass through `exp_sigmoid`
  - then harmonic distribution is normalized, with above-Nyquist harmonics zeroed before renorm.
- Noise synth:
  - predicts `noise_magnitudes` (often 65 bins)
  - magnitudes use `exp_sigmoid(magnitudes + initial_bias)`
  - filtered noise branch is always part of the paper model.
- Source refs:
  - `ddsp/synths.py`
  - `ddsp/core.py` (`exp_sigmoid`, `remove_above_nyquist`, `normalize_harmonics`)

### 3) Loss
- Core loss is multi-scale spectral loss over several FFT sizes.
- Critically: both linear magnitude and log-magnitude terms are used in common paper configs.
- Default spectral-loss class includes:
  - `fft_sizes=(2048, 1024, 512, 256, 128, 64)`
  - `loss_type='L1'`
  - `mag_weight=1.0`
  - `logmag_weight` often set to `1.0` in model gin files.
- Source refs:
  - `ddsp/losses.py`
  - `ddsp/training/gin/models/ae.gin`

### 4) Optimization defaults
- Adam + exponential LR decay, not huge LR.
- Typical defaults:
  - `learning_rate = 3e-4`
  - decay steps/rate configured
  - gradient clip norm around `3.0`
- Source refs:
  - `ddsp/training/gin/optimization/base.gin`
  - `ddsp/training/trainers.py`

### 5) Conditioning/preprocessing
- f0 and loudness are carefully scaled (`f0_scaled`, `ld_scaled`) and often use CREPE-derived f0/confidence in the prep pipeline.
- Source refs:
  - `ddsp/training/preprocessing.py`
  - `ddsp/training/data_preparation/ddsp_prepare_tfrecord.py`

## Current DDSPE2E status vs paper

### Big deltas (high impact)
1. Decoder is framewise MLP, no recurrent core.
- Current: `Examples/DDSPE2E/ModelDecoder.swift` (tanh trunk + heads).
- This reduces temporal context and can encourage static/control-collapse solutions.

2. Harmonic parameterization differs from DDSP reference path.
- Current supports `legacy`, `normalized`, `softmax-db` heads.
- DDSP reference behavior is closer to `exp_sigmoid + nyquist-aware renorm`.
- Current synth path does not yet explicitly perform Nyquist masking/renorm of harmonic distribution at render time.

3. Spectral loss formulation is not the same as paper loss.
- Current spectral op is squared-magnitude FFT difference (with optional normalization), not the DDSP-style `L1(mag) + L1(logmag)` stack.
- Files:
  - `Examples/DDSPE2E/Losses.swift`
  - `Sources/DGenLazy/Functions.swift`

4. Training hyperparameters are much more aggressive than DDSP defaults.
- Example scripts use large LR (e.g. `0.02`, `2.1`) and can push the model to degenerate attractors quickly.
- Files:
  - `train4.sh`
  - `train_flute.sh`

5. Feature pipeline is simpler than DDSP’s standard prep.
- Current f0/loudness extraction is intentionally lightweight and may inject noise/bias that makes collapse easier.

## Why collapse can happen here
Likely combined mechanism:
- Head parameterization + loss mismatch allows cheap local minima (single dominant harmonic + very low/flat gain envelope).
- No temporal model means each frame independently seeks local loss reductions.
- High LR accelerates early movement into collapsed basin.
- Missing log-magnitude term weakens gradients on quieter spectral structure.

## DGEN-native strategy: temporal context without full GRU first

DGEN has first-class state operators (`historyRead`/`historyWrite`, `accum`, stateful ops, biquad-style recurrences). We can exploit this to add temporal inductive bias with lower implementation/perf risk than immediately adding a large GRU.

### Recurrent-lite building blocks
1. One-pole smoothing on control streams (per control dimension)
- Form: `y_t = a * y_{t-1} + (1-a) * x_t`
- Implement with history cells in graph-level ops.
- Applies to:
  - harmonic gain
  - noise gain
  - harmonic distribution logits or post-activation distribution

2. AR residual control path
- Predict `delta_t` and integrate with state:
  - `state_t = state_{t-1} + delta_t` (or leaky form)
- This biases controls toward continuity and prevents framewise thrash.

3. Causal trend features from history
- Add `delta(f0_scaled)` and `delta(ld_scaled)` features (already easy via history-based `delta` op pattern).
- Gives model short-term dynamics without full RNN.

4. Optional stronger smoothing mode
- Reuse biquad-style recurrence for control envelopes when one-pole is insufficient.

This gives a "tiny-RNN" effect with explicit DSP semantics and good transparency in debugging plots.

## Multi-stage implementation plan

### Stage 0: Freeze baseline and instrumentation
Goal: make every future change measurable.

Tasks:
- Keep a locked baseline script (single-chunk overfit + multi-chunk flute).
- Add consistent logging of:
  - per-term losses (mag, logmag once added)
  - control stats (harmonic entropy, amp sum, gain ranges)
  - collapse indicators (top-harmonic concentration).
- Ensure `dump-controls-every` snapshots include settings used in run metadata.

Exit criteria:
- We can reproduce current `~7e-4 / ~5.7e-4` floors with saved plots.

### Stage 1: DDSP control/loss parity (no recurrence yet)
Goal: match the most important paper mechanics first.

Tasks:
1. Add `exp_sigmoid` activation mode for control heads.
- New head mode option (or sub-option) with same shape contract.
- Keep existing modes for A/B.

2. Add Nyquist masking + renorm for harmonic distribution.
- Compute per-frame harmonic frequencies from `f0 * harmonic_index`.
- Zero bins above Nyquist.
- Renormalize remaining bins with epsilon protection.

3. Add DDSP-style log-magnitude spectral term.
- Keep current mag loss term.
- Add `logmag_weight` flag and schedule if needed.
- Use safe-log epsilon.

4. Use DDSP-like conservative optimizer defaults for parity run.
- LR around `1e-4..3e-4`, exp decay, global clip.

Exit criteria:
- On single-example overfit: avoids immediate collapse-to-one-harmonic and beats current floor.

### Stage 2: Recurrent-lite controls (DGEN-first)
Goal: inject temporal memory while preserving speed and graph simplicity.

Tasks:
1. Add control smoothing flags.
- `--control-smoothing-mode <none|onepole|biquad>`
- `--control-smoothing-alpha <float>` (one-pole coefficient)
- `--control-smoothing-targets <harm_gain,noise_gain,harm_dist>`

2. Implement one-pole history path in synth/control graph.
- At minimum smooth gain streams.
- Optional smoothing of harmonic logits before normalization.

3. Add delta-conditioned inputs.
- Expand conditioning from `[f0Norm, loudNorm, uv]` to include short-term deltas.
- Keep flag-gated for A/B.

4. Add anti-collapse regularizer aligned with smoothing.
- Keep entropy/temperature controls, but apply with sane schedules and clear defaults.

Exit criteria:
- Plot trajectories show non-trivial harmonic distribution over longer steps.
- Lower final floor and better rendered timbre movement than Stage 1 parity run.

### Stage 3: Capacity bump with minimal recurrence complexity
Goal: increase expressiveness before full GRU.

Tasks:
- Increase trunk depth/width with stable init.
- Optional per-input projection stacks (f0/loudness/uv separate then concat).
- Keep recurrent-lite smoothing from Stage 2.

Exit criteria:
- Better fit on multi-chunk data without reverting to collapse patterns.

### Stage 4: Hybrid micro-RNN cell (optional, if Stage 2/3 not enough)
Goal: introduce explicit learned state update, smaller than GRU.

Tasks:
- Add lightweight gated memory cell implemented with DGEN state ops.
- Example form:
  - `z_t = sigmoid(Wz x_t + Uz h_{t-1})`
  - `h_t = z_t * h_{t-1} + (1-z_t) * tanh(Wx x_t + Uh h_{t-1})`
- Keep hidden size small (e.g., 16-64) to limit cost.

Exit criteria:
- Measurable win over Stage 3 with manageable step-time increase.

### Stage 5: Full GRU parity path (last resort / parity-complete)
Goal: only if needed for final paper-faithful reproduction.

Tasks:
- Implement GRU decoder block with chunked sequence processing.
- Maintain existing profiler checks to ensure backward kernels are acceptable.
- Use this only if micro-RNN + recurrent-lite cannot reach quality target.

Exit criteria:
- Clear quality gain that justifies runtime overhead.

## Recommended immediate execution order
1. Stage 1 (control/loss parity) first.
2. Stage 2 (one-pole/delta recurrent-lite) second.
3. Only then decide whether Stage 4/5 is needed.

This sequence gives maximum odds of fixing collapse with minimal perf regression.

## Concrete file-level implementation map

### Stage 1
- `Examples/DDSPE2E/ModelDecoder.swift`
  - add `exp_sigmoid` head option and config plumbing
- `Examples/DDSPE2E/Synth.swift`
  - Nyquist masking + harmonic renorm path
- `Examples/DDSPE2E/Losses.swift`
  - add `logmag` term and weighting
- `Examples/DDSPE2E/Config.swift`
  - add flags for new loss/control options
- `train4.sh`, `run_train4_100_and_plot.sh`, `train_flute.sh`
  - add parity presets

### Stage 2
- `Examples/DDSPE2E/Synth.swift` and/or graph helper ops
  - one-pole smoothing state ops on controls
- `Examples/DDSPE2E/Trainer.swift`
  - conditioning deltas + expanded feature vector
- `Examples/DDSPE2E/scripts/plot_controls.py`
  - add smoothing-state diagnostics

## Experiment matrix (minimum)
For each stage, run:
1. Single-chunk fixed-batch overfit (`100-500` steps) with control plots every `20` steps.
2. Multi-chunk fixed-batch (`batch 16/32`) with timing + collapse metrics.
3. Multi-chunk shuffled training with render snapshots.

Track:
- best loss
- loss slope after early fast drop
- harmonic entropy trend
- rendered-audio diversity (subjective + simple stats)
- step time and backward time

## Decision gates
- If Stage 1 alone resolves collapse/floor: do not add heavy recurrence.
- If Stage 1 + Stage 2 resolves collapse: keep DGEN-native recurrent-lite as default.
- If still stuck, move to Stage 4 micro-RNN; GRU parity only after that.

## Bottom line
The most likely path to materially better results is not "add GRU immediately". The highest ROI path is:
1. match DDSP control/loss mechanics,
2. leverage DGEN state primitives for lightweight temporal smoothing,
3. escalate to heavier recurrence only if evidence says it is necessary.

## Transformer GRU-Replacement Actionable Checklist
Goal: add a transformer temporal backbone where DDSP originally used GRU, while preserving current synth and control heads for clear A/B comparisons.

### Phase 0 - Baseline lock (required before edits)
- [ ] Save one reproducible baseline run with current MLP decoder.
- [ ] Export control dumps (`--dump-controls-every 20`) and render snapshots for reference.
- [ ] Record baseline metrics in a short run note:
  - [ ] total loss floor
  - [ ] spectral mag term
  - [ ] spectral logmag term
  - [ ] harmonic entropy/concentration trends

Exit gate:
- [ ] Baseline can be rerun with matching behavior before transformer code is introduced.

### Phase 1 - Config and CLI plumbing
- [ ] In `Examples/DDSPE2E/Config.swift`, add decoder backbone selection:
  - [ ] `decoderBackbone: mlp|transformer`
  - [ ] `transformerDModel`
  - [ ] `transformerLayers`
  - [ ] `transformerFFMultiplier`
  - [ ] `transformerCausal`
  - [ ] `transformerUsePositionalEncoding`
- [ ] Add validation rules for all new transformer config values.
- [ ] Add CLI override parsing for new fields in `Examples/DDSPE2E/Config.swift`.
- [ ] Update CLI help text in `Examples/DDSPE2E/main.swift`.
- [ ] Add new config values to run metadata logging in `Examples/DDSPE2E/Trainer.swift`.

Exit gate:
- [ ] `dump-config` includes transformer fields.
- [ ] `train` accepts new flags and writes them to `resolved_config.json` + `run_meta.json`.

### Phase 2 - Decoder transformer trunk
- [ ] In `Examples/DDSPE2E/ModelDecoder.swift`, keep existing heads unchanged.
- [ ] Add an input projection from `[F,5] -> [F,d_model]`.
- [ ] Implement transformer block(s) with:
  - [ ] pre-norm LayerNorm
  - [ ] scaled dot-product attention
  - [ ] residual connection
  - [ ] feed-forward sublayer + residual
- [ ] Implement optional causal attention mask tensor.
- [ ] Implement optional positional encoding addition.
- [ ] Route trunk output into existing harmonic/noise heads.
- [ ] Keep `mlp` path functional for direct A/B runs.

Exit gate:
- [ ] Forward pass shape checks pass for both `mlp` and `transformer`.
- [ ] No NaN/Inf in decoder outputs on a short dry run.

### Phase 3 - Training integration and defaults
- [ ] In transformer mode, default to `batchSize=1` and use `gradAccumSteps` to recover effective batch.
- [ ] In transformer mode, default `controlSmoothingMode=off` to avoid confounding FIR and temporal modeling.
- [ ] Preserve existing loss stack and synth behavior during first transformer experiments.
- [ ] Add an explicit warning in logs if transformer mode runs with settings likely to cause OOM or quadratic blowup.

Exit gate:
- [ ] 100-step single-chunk run completes in transformer mode with stable loss and finite gradients.

### Phase 4 - Tests
- [ ] Extend `Tests/DGenLazyTests/TransformerOpsTests.swift` or add decoder-specific tests for:
  - [ ] shape contract (`[F,5] -> controls`)
  - [ ] gradient flow through transformer params
  - [ ] causal property (future frames do not alter earlier outputs)
  - [ ] numerical stability (no NaN/Inf over repeated optimization steps)
- [ ] Add checkpoint compatibility coverage:
  - [ ] old MLP checkpoints still load
  - [ ] missing transformer params are handled safely

Exit gate:
- [ ] Transformer-related tests pass with `swift test`.

### Phase 5 - Experiment matrix (decide if transformer replaces GRU role)
- [ ] Run A: MLP + current smoothing policy.
- [ ] Run B: MLP + smoothing off.
- [ ] Run C: Transformer (causal) + smoothing off.
- [ ] Run D: Transformer (causal) + minimal smoothing (optional).
- [ ] For each run, collect:
  - [ ] best loss
  - [ ] final loss slope
  - [ ] harmonic entropy/concentration traces
  - [ ] render snapshots at fixed steps
  - [ ] step time and backward time

Exit gate:
- [ ] Transformer shows clear temporal-control benefit (less collapse and/or better audio movement) relative to MLP baselines.

### Phase 6 - Optional batched transformer path
- [ ] If needed, add block-diagonal + causal masking for `batchSize > 1` to prevent cross-chunk attention leakage.
- [ ] Add guardrails for sequence length (`B*F`) and memory costs.
- [ ] Re-run experiment matrix in batched mode.

Exit gate:
- [ ] Batched transformer is correct (no cross-sample leakage) and performant enough to keep.

### Final decision checklist
- [ ] Keep transformer path as default temporal backbone if it consistently beats MLP across target datasets.
- [ ] Keep MLP path available as fallback/debug baseline.
- [ ] Document recommended transformer preset in `Examples/DDSPE2E/README.md` and training scripts.
