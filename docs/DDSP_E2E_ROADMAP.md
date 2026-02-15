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

## Current Status (Feb 2026)

### What's Working
- **Full pipeline**: preprocess WAV -> extract features -> train decoder -> render audio
- **Harmonic additive synth**: stateful phasor, K harmonics, per-frame amplitude + gain from MLP
- **Multi-scale spectral loss**: multiple FFT window sizes, hop-aligned, with warmup/ramp scheduling
- **MSE loss**: available but currently hurts convergence when combined with spectral
- **Deterministic evaluation**: spectral loss FFT race condition fixed (ThreadCountScale on isolated spectral blocks)
- **Pre-allocated tensor pattern**: data tensors created once, `updateDataLazily()` each iteration
- **Checkpointing**: save/load model weights, periodic render exports
- **Gradient clipping**: element-wise and global modes
- **Adam optimizer**: working, per-parameter group LR supported

### Current Limitations
- **Spectral-only loss plateaus at ~10% reduction** on single audio clip (harmonic-only synth, no noise)
- **Noise branch is static**: fixed FIR kernel, not conditioned by the network (unlike DDSP paper)
- **Single-layer MLP**: `[3] -> [H] -> heads` — likely underpowered for complex timbres
- **No LR scheduling**: fixed learning rate throughout training
- **No validation renders during training** for quick listening checks

---

## Milestones

## M0 - Project Skeleton and Reproducible Runner -- DONE
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

## M1 - Data + Feature Pipeline (Swift-first) -- DONE
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

## M2 - Decoder-Only DDSP Baseline (No Encoder Yet) -- DONE (harmonic branch)
### Deliverables
- Neural control network:
  - input: per-frame `f0`, loudness, uv
  - output: controls for
    - global amp envelope
    - harmonic distribution (K harmonics)
    - ~~noise magnitudes (B bands)~~ (static FIR only — conditioned noise is M3a)
- Differentiable synth graph:
  - harmonic additive branch
  - ~~filtered noise branch (STFT/filterbank-style)~~ (static FIR placeholder)
  - branch mixing and output gain

### Exit Criteria
- Forward render works for batch/chunk inputs
- Backward pass updates parameters with non-zero gradients
- Short training run decreases loss on a small dataset

### What Shipped
- Single-layer MLP decoder: `[f0Norm, loudNorm, uv]` -> tanh hidden -> sigmoid heads
- Harmonic synth with stateful phasor, K harmonics, peekRow amplitude lookup
- Static FIR noise branch (optional, disabled by default)
- Pre-allocated tensor pattern for training loop efficiency

---

## M3 - Training Stack and Losses -- IN PROGRESS

### M3a - Conditioned Noise Filter (next up)
The DDSP paper uses a **frequency-domain filtered noise** branch where the network predicts
**65 noise magnitudes** (frequency-sampled filter banks) per frame. These are converted to an
impulse response via `frequency_impulse_response` and convolved with white noise using FFT.
The current static FIR kernel is a placeholder.

**Approach options** (in order of fidelity to the original paper):
1. **Time-domain FIR with predicted coefficients**: MLP head outputs `[F, K_noise]` filter taps,
   convolved with white noise per frame. Simple, but limited frequency resolution for short filters.
2. **Frequency-domain filtering** (paper approach): MLP predicts `[F, N_bands]` magnitude
   envelope (65 bands in the paper), convert to impulse response via frequency sampling,
   apply via FFT convolution. Matches the paper.
3. **Subtractive synthesis shortcut**: predict per-band gains for a fixed filterbank. Less
   flexible but numerically simpler.

**Recommendation**: Start with option 1 (predicted FIR taps) since `conv2d` and `buffer` already
work. Graduate to option 2 if frequency resolution is insufficient.

### M3b - Deeper Model + Single-Clip Convergence
Before scaling to multiple files, prove the system can substantially reduce loss on one clip.

**Model capacity**: The current single-layer MLP is far smaller than the paper's architecture.
The DDSP paper (Engel et al., ICLR 2020) uses:
- Per-input MLPs: each of `f0`, `loudness`, (and optionally `z`) gets its own **3-layer MLP
  with 512 units** and layer norm + ReLU
- Concatenation of MLP outputs
- **512-unit GRU** (recurrent — captures temporal context across frames)
- **Output MLP** mapping GRU output to synth controls: `amps(1) + harmonic_distribution(60)
  + noise_magnitudes(65)`

We don't need the full paper architecture to make progress, but the current `[3] -> [32] -> heads`
is almost certainly the convergence bottleneck. Incremental steps:
1. **Larger hidden size**: 32 -> 128 or 256 (`--model-hidden 128`)
2. **2-layer trunk**: `--model-layers 2` gives `[3] -> [H] -> tanh -> [H] -> tanh -> heads`
   (the paper uses 3 layers with layer norm, but 2 is a reasonable step)
3. **Per-input MLPs** (optional): separate `f0` and `loudness` embeddings before concatenation
4. **GRU** (if nothing else works): the paper uses a 512-unit GRU for temporal smoothing across
   frames, but our architecture already produces smooth sample-rate envelopes via `peekRow`
   linear interpolation between control frames. The GRU's temporal context is less critical
   when the synth path itself handles frame-to-sample smoothing. Could also add simple
   low-pass filtering on the control signals for similar effect without recurrence.

**Other convergence levers**:
- LR scheduling: warmup + cosine/exponential decay
- Spectral loss tuning: window size selection, weighting between scales
- Loudness loss: cheap RMS-in-dB term that helps the overall envelope converge
- Confirm harmonic amplitudes and gain converge to reasonable values

**Exit criteria**: >50% spectral loss reduction on a single sustained monophonic clip.

### M3c - Multi-File Training
- Verify training loop handles varying chunk conditioning gracefully
- Shuffle chunks across files each epoch
- Monitor per-file loss to catch dataset imbalance
- Validate on held-out chunks

### Existing M3 Deliverables (partially done)
- [x] Multi-scale spectral loss (several FFT sizes, weighted sum)
- [x] Optimizer + gradient clipping
- [ ] LR schedule (warmup + cosine or exponential decay)
- [ ] Loudness loss (RMS-in-dB per frame — cheap and helps envelope convergence)
- [ ] ~~Optional waveform loss term~~ (MSE hurts — revisit later with proper weighting)
- [ ] Train/val loop with metrics logging
- [ ] Periodic validation renders for listening checks

### Exit Criteria
- Conditioned noise branch produces audibly different output than harmonic-only
- Loss trends are stable (no frequent NaN/Inf)
- Validation metrics improve over baseline
- Periodic renders export without manual intervention

### Risks
- Single-window spectral loss may underconstrain; multi-scale is required.
- MSE loss currently destabilizes training — may need careful weighting or a perceptual alternative.

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

## M6 - Learnable Convolution Reverb
### Deliverables
- Learnable impulse response (IR) parameter tensor
- FFT-based convolution: `signal -> FFT -> multiply IR spectrum -> IFFT -> overlap-add`
- IR head from decoder (optional: fixed IR or per-frame modulated)
- Config toggle to enable/disable reverb

### Exit Criteria
- Reverb head can be toggled in config
- IR converges to a reasonable shape (early reflection + decay)
- Improves quality on reverberant material without destabilizing training

### Notes
- The FFT/IFFT/overlapAdd machinery already exists in DGen. The main work is:
  1. Making the IR a learnable `Tensor.param`
  2. FFT-domain multiplication of signal spectrum with IR spectrum
  3. Proper overlap-add of the convolved output
- Start with a fixed-length IR (e.g., 4096 samples at 16kHz = 256ms)
- Gradient flows through the IR spectrum multiplication

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

```
M0 skeleton                     -- DONE
M1 data + loudness + basic f0   -- DONE
M2 decoder-only synth           -- DONE (harmonic branch)
M3a conditioned noise filter    <-- NEXT
M3b deeper single-clip convergence
M3c multi-file training
M4 reproducible demo
M5 encoder
M6 learnable reverb
M7 optional hardening
```

This order de-risks the project by proving value before high-complexity architecture work.

---

## Key Lessons Learned

### Spectral Loss
- **Spectral-only loss works better than MSE+spectral** at this stage. MSE pulls gradients
  toward sample-level matching which conflicts with the spectral objective.
- **Frequency resolution matters**: `resolution = sampleRate / windowSize`. Use windows large
  enough that target harmonics land in distinct bins.
- **Multi-scale is important**: use 2-3 window sizes (e.g., 512 + 1024) to capture both
  fine time resolution and fine frequency resolution.

### Training Stability
- **Spectral loss FFT must run without ThreadCountScale**: when matmul sets ThreadCountScale
  on a parent block, isolated spectral blocks must clear it. FFT butterflies are single-threaded
  per frame. (Fixed Feb 2026 — was causing 30%+ loss oscillation.)
- **Pre-allocate data tensors**: creating new Tensor objects each iteration accumulates weak
  refs and triggers unnecessary graph refresh. Use `updateDataLazily()` instead.
- **Gradient clipping is essential**: without it, spectral loss gradients can spike on
  certain frequency alignments.

### What Didn't Work
- **MSE loss weight > 0**: even small MSE weight (0.1) caused training instability when
  combined with spectral loss. Revisit with proper loss balancing later.
- **Very low learning rates** (1e-7): too slow to escape initial random parameter regime.
  LR ~1e-3 to 3e-4 works better with gradient clipping.

---

## Architecture Sketch
- **Preprocess phase**
  - WAV -> resample/mono -> chunk
  - chunk -> features (`f0`, loudness, uvMask)
  - cache features + chunk metadata
- **Train phase**
  - loader yields mini-batch chunks
  - decoder predicts synth controls
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
  - use multi-scale spectral + listening checks; MSE is counterproductive for now
- Scope creep:
  - require each milestone exit criteria before next milestone
- Model capacity too low:
  - single-layer MLP may plateau early; add depth before blaming the loss

---

## Definition of Done (Ultimate Demo)
A new user can:
1. preprocess a WAV dataset,
2. train a DDSP-style model,
3. load a checkpoint,
4. render reconstructions,
5. and reproduce documented results,

using repo-native tooling with no hidden manual steps.
