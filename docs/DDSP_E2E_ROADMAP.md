# DDSP End-to-End Roadmap (DGen)

## Goal
Build a **credible end-to-end DDSP-style demonstration** in this repo:
- Input: raw WAV dataset
- Pipeline: feature extraction (`f0`, loudness) -> neural controls -> differentiable synth
- Output: reconstructed audio + checkpoints + reproducible training run
- Quality bar: consistent loss reduction across dataset and listenable reconstructions

This roadmap prioritizes shipping a working system quickly, then hardening it to paper-like completeness.

## Success Criteria
1. One command can train and export periodic audio samples.
2. Training runs on more than one clip and converges (not just a toy single-example fit).
3. Inference can reconstruct unseen validation clips from conditioning features.
4. The whole pipeline can run in Swift (Python optional, not required).

## Scope Boundaries
### In Scope
- Monophonic audio first
- Decoder-only DDSP first (`f0` + loudness conditioning)
- Harmonic + noise synthesis branches
- Spectral training losses (multi-scale)
- Checkpointing and reproducible config

### Out of Scope (for first complete demo)
- Production-grade pitch tracker parity with CREPE
- Full dataset tooling UX polish
- Real-time plugin integration

---

## Milestones

## M0 - Project Skeleton and Reproducible Runner
### Deliverables
- `Examples/DDSPE2E/` module with:
  - `main.swift` training entrypoint
  - config struct (JSON + CLI overrides)
  - deterministic seed handling
- `Package.swift` updates:
  - add `.executableTarget(name: "DDSPE2E", ...)`
  - add executable product entry (`DDSPE2E`) for discoverability
- Output dirs: `checkpoints/`, `renders/`, `logs/`

### Exit Criteria
- `swift run DDSPE2E --help` works (target is wired in `Package.swift`)
- `swift run DDSPE2E train --config ...` starts and writes a run directory

### Notes
- Keep this out of `Tests/`; this is an example/training program.

---

## M1 - Data + Feature Pipeline (Swift-first)
### Deliverables
- Dataset loader:
  - scans WAV files
  - resamples to training SR
  - mono conversion + normalization
  - chunking into fixed-length training segments
- Feature extractor:
  - frame-wise loudness (RMS or log-RMS)
  - initial `f0` extractor (YIN/autocorr)
  - voiced/unvoiced confidence mask (basic)
- Feature cache format (binary or JSON) to avoid recomputation

### Exit Criteria
- Can preprocess a folder of WAV files and save feature cache
- Can iterate chunks and retrieve `{audio, f0, loudness, uvMask}` tensors/signals

### Risks
- `f0` quality on noisy/percussive material; start with monophonic sustained sources.

---

## M2 - Decoder-Only DDSP Baseline (No Encoder Yet)
### Deliverables
- Neural control network:
  - input: per-frame `f0`, loudness
  - output: controls for
    - global amp envelope
    - harmonic distribution (K harmonics)
    - noise magnitudes (B bands)
- Differentiable synth graph:
  - harmonic additive branch
  - filtered noise branch (STFT/filterbank-style)
  - branch mixing and output gain

### Exit Criteria
- Forward render works for batch/chunk inputs
- Backward pass updates parameters with non-zero gradients
- Short training run decreases loss on a small dataset

### Notes
- This is the first meaningful “DDSP working” milestone.

---

## M3 - Training Stack and Losses
### Deliverables
- Multi-scale spectral loss (several FFT sizes, weighted sum)
- Optional waveform loss term (small weight)
- Optimizer + gradient clipping + LR schedule
- Train/val loop with metrics logging

### Exit Criteria
- Loss trends are stable (no frequent NaN/Inf)
- Validation metrics improve over baseline
- Periodic renders export without manual intervention

### Risks
- Single-window spectral loss may underconstrain; multi-scale is required.

---

## M4 - End-to-End Demo Quality Bar
### Deliverables
- Reproducible experiment config committed to repo
- Training script that reaches target quality in finite time
- Export script for A/B clips (`target`, `recon`, optionally `harmonic-only`, `noise-only`)

### Exit Criteria
- Clean run from raw WAV folder -> trained checkpoint -> validation renders
- README/doc section with exact commands
- Subjective listening pass: recon is recognizable and not collapsed/noisy

---

## M5 - Encoder Integration (Paper-Complete Direction)
### Deliverables
- Audio encoder network producing latent `z` from target audio features
- Decoder updated to consume `{f0, loudness, z}`
- Joint training objective for encoder+decoder

### Exit Criteria
- Encoder path trains stably
- With fixed conditioning, changing `z` changes timbre as expected
- Reconstruction improves over decoder-only baseline on validation set

### Notes
- Do not block M2-M4 on this milestone.

---

## M6 - Optional Reverb / Effects Head
### Deliverables
- Optional reverb/effect parameter head from decoder
- Differentiable reverb block in synthesis chain

### Exit Criteria
- Reverb head can be toggled in config
- Improves quality on relevant material without destabilizing training

---

## M7 - Full Swift Pipeline (Remove Python Dependency)
### Deliverables
- Replace remaining Python preprocessing with Swift tools
- Single Swift command for preprocess + train + eval

### Exit Criteria
- No Python required for default workflow
- Docs include only Swift commands for core path

---

## Implementation Order (Recommended)
1. M0 skeleton
2. M1 data + loudness + basic `f0`
3. M2 decoder-only synth
4. M3 training stability + multi-scale loss
5. M4 reproducible demo
6. M5 encoder
7. M6/M7 optional hardening

This order de-risks the project by proving value before high-complexity architecture work.

---

## Architecture Sketch
- **Preprocess phase**
  - WAV -> resample/mono -> chunk
  - chunk -> features (`f0`, loudness, uvMask)
  - cache features + chunk metadata
- **Train phase**
  - loader yields mini-batch chunks
  - decoder (and later encoder) predicts synth controls
  - differentiable synth renders waveform
  - multi-scale spectral loss + backward + step
  - checkpoint + render samples every N steps
- **Eval phase**
  - load checkpoint
  - run validation split
  - export metrics and WAV comparisons

---

## Minimal File Plan
- `Package.swift` (add DDSPE2E executable target/product)
- `Examples/DDSPE2E/main.swift`
- `Examples/DDSPE2E/Config.swift`
- `Examples/DDSPE2E/Dataset.swift`
- `Examples/DDSPE2E/Features.swift`
- `Examples/DDSPE2E/ModelDecoder.swift`
- `Examples/DDSPE2E/Synth.swift`
- `Examples/DDSPE2E/Losses.swift`
- `Examples/DDSPE2E/Trainer.swift`
- `Examples/DDSPE2E/Checkpoint.swift`
- `docs/DDSP_E2E_ROADMAP.md` (this file)

---

## Tooling Requirements Checklist
### Required
- Swift toolchain + Metal-capable environment
- WAV IO (already present in `DGenLazy.AudioFile`)
- Feature extraction in Swift (`f0`, loudness)
- Training/eval CLI entrypoints

### Nice-to-Have
- TensorBoard-like logger adapter
- Better pitch estimator backend (optional)
- Dataset manifest tooling

---

## Validation Protocol
1. **Smoke**: one clip overfit (very low loss).
2. **Small dataset**: 10-50 clips; verify general trend and no divergence.
3. **Held-out**: unseen clips reconstructed from features.
4. **Ablation**:
   - harmonic-only
   - harmonic+noise
   - harmonic+noise+reverb

---

## Risks and Mitigations
- Unstable gradients on long chunks:
  - use shorter chunks first, gradient clipping, lower LR, warmup
- Weak `f0` on difficult material:
  - begin with monophonic voiced data, improve tracker later
- Loss mismatch to perception:
  - use multi-scale spectral + small waveform term + listening checks
- Scope creep:
  - require each milestone exit criteria before next milestone

---

## Definition of Done (Ultimate Demo)
A new user can:
1. preprocess a WAV dataset,
2. train a DDSP-style model,
3. load a checkpoint,
4. render reconstructions,
5. and reproduce documented results,

using repo-native tooling with no hidden manual steps.
