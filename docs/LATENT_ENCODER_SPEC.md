# Latent Encoder Spec (audio → latent code, trained in dgen)

Status: phase 1 implemented (branch `latent-encoder`, see §7) · audited against `main @ 11cd7ca` (2026-11-02)

## Goal

Given a trained latent instrument (decoder + voice, e.g. *Latent Kick (DOOM + Neptunes)*),
train an **encoder** that maps a drum hit's audio to its latent code `z` in one forward pass,
using dgen's own autograd. The same encoder patch compiles (C) so eseq and the browser
(wasm) can drop a WAV in and get a playable code instantly.

Success: on held-out real hits the encoder never saw, `voice(decoder(enc(x)))` scores close
to `voice(decoder(z_learned))` (the auto-decoder's own fit) on the band / MR metrics, with no
loudness blow-ups (peak within the training set's range).

## Background: today's models are auto-decoders

`train-dataset` (TrainDataset.swift) trains a `latent-table` of free code rows `z:<table>`
[N,D] jointly with the decoder (`pretrain`: regress U(codes) onto fitted rows; `audio`: MR-STFT
through the voice back to codes and weights). Nothing maps audio → code; a hit has a code only
because it was in the training set. The encoder adds that map. The learned codes
(`export/<table>.json`, `{names, codes, dim}`) become supervision targets for free.

Measured motivation (browser demo, this session): decoding codes off the data is unsafe:
a kick 0.5 latent units from any real hit can peak 3× louder than every training hit, 17×
at the map corner. The encoder must stay on the code cloud (prior term, §5 stage C).

---

## 1. What exists (verified)

Legend: ✅ works with gradients · ◐ partial · ❌ missing. G = Sources/DGen/Gradients.swift,
LE = Sources/DGenLisp/LispEvaluator.swift, TD = Sources/DGenLisp/TrainDataset.swift.

### Tensor / NN ops

| Need | Status | Notes |
|---|---|---|
| matmul | ◐ | `(matmul a b)` **2-D only** (LE:2386). Lowers to broadcast mul + sumAxis; grads via primitives (tested). GEMM rewrite is Metal-only and post-gradient. Batched: reshape `[B,T,F]→[B·T,F]`. |
| conv1d | ❌ for training | fused `.conv1d` op: 1-D × 1-D, no channels/batch/stride; **backward returns nil** (G:981). |
| conv2d | ◐ | two paths. Fused `.conv2d` (`conv2dSame`, `@padding same`): backward nil. **Lowered** `conv2dView` (valid): `pool` → `asStrided` windows → broadcast mul → sumAxis. Kernel grad ✅ (Conv2DLearningTests); **input grad wrong**: asStrided backward reshapes the [oH,oW,kH,kW] window grad back to [H,W] (G:700) instead of scatter-adding overlaps. Invisible in single-layer tests (input is data); breaks stacked layers. |
| pooling | ◐ | `mean-axis` / `max-axis` / `sum-axis` ✅ (G:651/661/678). No strided pool form; `Graph.pool` exists core-only (Tensor.swift:909) with the same asStrided grad problem. |
| activations | ✅ | relu, sigmoid, tanh, softmax (tensor only). softplus/gelu/leaky: compose. |
| normalization | ◐ | none built-in; compose with mean-axis + reshape (no `@keepdims` on mean-axis) + broadcasting (grads reduce correctly, G:1569). |
| views | ✅ | reshape, transpose, shrink, pad, expandView: correct grads. ⚠ `repeatView` and `.expand` grads are TODO passthroughs (G:745, 772). No concat (pad + add works). `gather` no grad. |
| stop-gradient | ❌ | only special-case detaches (`detachPhasorFrequency`, `detachAccumInputs`). |
| learnable tensors | ✅ | `(tensor @shape [...] @learnable true @file … @init glorot\|zeros\|small\|file)` any rank (DatasetForms.swift:198). ⚠ glorot fan-in = shape[0] (TD:447): wrong for 3-D kernels. |
| random | ◐ | `randn` is fixed at graph build; `noise @size` per frame; no dropout. |
| optimizer | ◐ | trainer's `AdamMoments` (TD:261): fixed betas, cosine LR w/ 5 % floor, non-finite grads zeroed. Per-tensor clip exists but **is never passed** (TD:706). No weight decay. |

### Spectral

| Need | Status | Notes |
|---|---|---|
| framing `[B,N]→[B,T,W]` | ◐ | `Graph.pool(kernel:[1024], stride:[256])` does it as a zero-copy view, batch dims kept, **not exposed** to DGenLazy/Lisp. Lisp `windows` is 2-D, stride 1. |
| FFT on tensors | ◐ | `fft` / `polar-fft` / `hann` exist; `tensorFFT` requires exactly `[N]` (HigherOps+FFT.swift:16): **no batch dims**. Butterfly views are differentiable. `acceleratedFFT`: C only, no grad. |
| spectrogram exposed | ❌ | `spectralLossFFT(Batched)` computes STFTs in threadgroup scratch (N ≤ 2048 on Metal) and reduces to a scalar; magnitudes are private cells. Uses a **symmetric** Hann; `hann` form is periodic. |
| host log-mel | ✅ | `Examples/DDSPE2E/Features.swift:160 timbreLogMelFrames` (vDSP, mel filters, log). |
| log / sqrt / pow grads | ✅ | G:422-561 (add ε before sqrt/log). |
| real-time buffering (inference) | ✅ | `buffer(size, hop)` ring view (Signal.swift:387) → ×hann → FFT; proven in STFTIdentityTests (n=512). `latch` / `event-hold` / `hop-hold` / tensor history exist. |

### Training infrastructure

| Need | Status | Notes |
|---|---|---|
| per-row **data** input | ❌ | no form feeds non-learnable per-row data. Targets are host-prepared (`DatasetTargets.prepare`: mono, onset −40 dB −2 ms, peak 0.9, crop) and only reach the **loss**, never the patch. |
| select trainable leaves | ◐ | leaves `w:<tensor>`, `z:<table>`, `row:<param-row>`; `trainable:` closures are hardcoded per stage (TD:793/802). `--freeze` masks row *segments*, and only zeroes grads in refine (doc at TD:61 says audio too; code doesn't). |
| gradient into an upstream encoder | ✅ (INFERRED) | `audio` stage's `latchStep` pass 3 is ordinary reverse-mode over the static side, so an encoder upstream of the codes gets dL/dθ, provided its input is a **static** tensor. ⚠ `codeL2` force-unwraps `z:` (TD:900). |
| batching | ❌ | B = N, one evaluation, no minibatch/shuffle/val split. Overflow ("32-bit index range") is handled **outside dgen** by eseq's runner halving the rows (eseq tools/modal/train/run.py:131). |
| render arbitrary z | ◐ | `--stage render --z-rows` renders a batch to WAVs; preset key hardcoded `z:codes` (TD:609); no rows.json export. |
| PathSelect | ✅ | random in-between latents, resampled each step (TD:807). |
| composition | ❌ | no include/import in dgenlisp; `defmacro` per file. Training reads channel 0 / first tensor output only. |
| compile of the encoder | ✅ | learnable tensors become `@file` constants; matmul/elementwise/reductions emit on C. |

---

## 2. Design

Two patches, one trainer:

- **`encoder.lisp`**: features → z. Its input is a new `row-data` form: in training it is the
  per-row feature tensor; in a compile it is built from `(in 1)` by the streaming feature chain.
- **`voice.lisp`**: the existing trained instrument (decoder + voice), unchanged, frozen.
- **`train-dataset --stage encode|encode-audio --encoder encoder.lisp`**: the trainer evaluates
  the encoder under the same `DatasetContext` and **binds its output into the voice's
  `latent-table`** (via the `latentRows` closure) instead of the `z:` leaf. That avoids an
  include mechanism.

### New forms / options (phase 1 minimal set)

```lisp
(row-data @name feats @shape [T F] X)
;; train: persistent non-param [B, T·F] tensor, filled by the trainer per step (minibatch)
;; compile: evaluates X (the streaming feature chain), shape [1, T·F]

(latent-out Z)            ;; marks the encoder's [B, D] output (compile: D signals / outputs)
```

Trainer options: `--encoder PATH`, `--features feats.json | --features-from host-logmel`,
`--codes export/codes.json`, `--batch B`, `--val K`, `--train-only w:enc_` (prefix filter),
`--clip X`, `--weight-decay X`, `--prior X`.

### Features (phase 1: host, with a compile-time twin)

Training features are computed **on the host** (no gradient is needed into data): reuse
`timbreLogMelFrames`, but with a **periodic** Hann to match `hann`. Fixed recipe: 48 kHz,
onset-aligned crop of 0.5 s, window 1024, hop 256 → T = 90, 64 mel bands, `log(mel + 1e-5)`,
per-band mean/std from the training set (exported as `feat_stats.json`).

The compiled encoder rebuilds the same features from audio: `buffer(1024, 256)` → ×`hann` →
FFT (`acceleratedFFT` on C) → `polar-fft` mag → shrink to 513 → `matmul` mel [513, 64] → log →
normalize, with the last T frames kept in tensor history and the encoder gated by `latch` on the
trigger. **Parity test required** (§4): host features vs compiled-patch features on the same
WAVs, frame alignment included (`pool` starts at sample 0; `buffer` hop phase must match).

Phase 3 moves features in-graph (needed only for online synthetic sampling): expose
`(frames x @size @hop)` over `Graph.pool` (forward only is fine: data has no grad) and either
batch `tensorFFT` over leading dims or add a dedicated `stftMagnitude` op reusing
`emitFFTInScratch` from Emit+SpectralLoss.swift (one thread per (b, t), N ≤ 2048).

### Fix first: `asStrided` backward (unlocks the lowered conv path)

The lowered conv is already the right shape for a real encoder; only its input gradient is
wrong. Correct rule: **scatter-add**: every window element accumulates its grad into the input
position it read (`offset + Σ idx·stride`), overlaps summing.

- **Phase 1 (composed, pool-shaped views):** for a 1-D pool with kernel k and stride s, window
  offset j contributes G[…, t, j] at input position j + s·t, so
  `grad_x = Σⱼ pad(dilate_s(G[…, j]), (j, rest))` with dilate = reshape + pad + reshape.
  No new kernel; built from ops with correct grads; k terms (k = 3–7 for conv; the STFT's
  1024-wide frames never need a grad). Same construction per axis for 2-D pools.
- **Guard:** any other `asStrided` view on a gradient path → hard error (not a silent reshape).
- **Later (general):** a `scatterAdd` op (atomic accumulate on Metal, like `tensorAccumulate`)
  for arbitrary strided views.
- **Test first:** finite-difference grad check through two stacked conv layers (input grad of
  layer 2 = output grad of layer 1). Expected to fail on `main` today; it gates the fix.

### Encoder architecture: im2col conv over `pool` + `matmul`

With the fix, multi-channel, batched, strided conv1d is the standard im2col, all existing ops:

```
[B, Cin, T] → pool(kernel:[k], stride:[s]) → [B, Cin, T′, k]
            → transpose → [B, T′, Cin, k] → reshape [B·T′, Cin·k] → matmul [Cin·k, Cout] → [B, T′, Cout]
```

Kernels are 2-D `[Cin·k, Cout]`, so glorot fan-in is right as-is. Needed surface: expose `pool`
as a dgenlisp form, e.g. `(windows1d x @size k @stride s)` (the existing `windows` is 2-D, stride 1).

```lisp
;; feats: [B, 64, 90]   (channels = mel bands, time last)
(defmacro conv1d-ch (x cin cout k s name)
  (def w (windows1d x @size k @stride s))                         ;; [B, cin, T', k]
  (def cols (reshape (transpose w @axes [0 2 1 3]) [-1 (* cin k)]))
  (matmul cols (tensor @shape [(* cin k) cout] @learnable true @name name)))  ;; [B·T', cout]
(def h1 (relu (conv1d-ch feats 64 96 5 1 c1)))        ;; → [B, 86, 96] after reshape
(def h2 (relu (conv1d-ch (to-ct h1) 96 128 5 2 c2)))  ;; strided: [B, 41, 128]
(def g  (mean-axis h2 1))                             ;; [B, 128] time average
(def h3 (tanh (+ (matmul g W1) b1)))                  ;; [128 → 128]
(latent-out (+ (matmul h3 W2) b2))                    ;; [B, D]
```

Fallback if the fix slips: **shift-and-matmul**, y = Σⱼ shrink(x, j…j+T′) @ Wⱼ (shrink, matmul,
add only), with pooling as reshape + mean-axis. Same math, k matmuls per layer.

≈ 64·96·5 + 96·128·5 + 128·128 + 128·D ≈ 110 k params; small enough for CPU inference in wasm.

---

## 3. Training pipeline

```
data:   real   = 94 targets (prepare()), labels = learned codes; hold out K = 10 for val
        synth  = render stage over sampled z (code + N(0, 0.3²), PathSelect-style mixes),
                 WAVs + z labels written next to each other; host features for both
        augment (host, per step): gain ±, onset jitter ±3 ms, ±1 st, EQ tilt ±6 dB, noise −60…−35 dB

A  encode (supervised, synthetic):   loss = ‖enc(f(aug(x_synth))) − z‖²
                                          + λ_u ‖U(enc(·)) − U(z)‖² (decoded-param term, capture mode)
B  encode (supervised, real anchor): same loss on real ∪ synth minibatches (50/50)
C  encode-audio (through the synth): x → enc → FROZEN decoder → FROZEN voice → MR-STFT vs x
                                          + λ_p prior(z)   (distance to the code cloud)
                                     uses latchStep unchanged; trainable = w:enc_* only
eval    per val hit: code error; MR/band of voice(dec(enc(x))) vs x; same for the learned code
        (ceiling); peak vs training peak range
export  encoder @file tensors + feat_stats.json + mel filterbank; compile encoder.lisp → C → wasm
```

A and B need no voice render (capture mode stops at `param-row`), so they can run large
minibatches; C renders the voice and inherits today's batch limits (keep B ≤ 16 until
minibatching handles the 32-bit index overflow inside dgen).

---

## 4. Implementation plan

**Phase 1: supervised encoder, host features** (smallest end-to-end proof)
0. Stacked-conv finite-difference grad test (fails today) → composed scatter-add backward for
   pool-shaped `asStrided` + hard error for other views → expose `windows1d` (pool) in dgenlisp.
1. `row-data` form + `DatasetContext.rowData` closure; trainer fills it per step
   (persistent tensor + `updateDataLazily`, like `PathSelect`).
2. `--encoder`: evaluate encoder patch, bind `latent-out` into `latentRows`.
3. Trainable filter by prefix (`--train-only`); guard `codeL2` when no `z:` leaf.
4. `encode` stage (capture mode) with code-MSE and decoded-param loss; `--codes`.
5. Minibatch sampling + `--val K` held-out split + val metrics in the summary.
6. Wire `clip` into `AdamMoments.step`; add decoupled weight decay.
7. Host log-mel (periodic Hann) + `feat_stats.json` export.
8. Synthetic data: render stage follows the table name (not hardcoded `z:codes`) and writes
   `labels.json` beside the WAVs.

Acceptance: on 10 held-out real kicks, median MR of `voice(dec(enc(x)))` within 15 % of the
learned-code ceiling; no val render peaks above 1.2× the loudest training hit.

**Phase 2: through-synth fine-tuning + compiled inference**
9. `encode-audio` stage: `audio` stage with the encoder bound; prior term.
10. Streaming feature chain in `encoder.lisp` (compile path) + parity test vs host features
    (max abs diff on normalized log-mel < 1e-3 after alignment).
11. eseq / browser: compile encoder → C → wasm; drop WAV → z.

**Phase 3: in-graph features + online synthetic rows**
12. `(frames …)` over `Graph.pool`; batched `tensorFFT` or `stftMagnitude` op.
13. General `stop-gradient` op (needed once synthetic audio is rendered in the same graph).
14. Optional: in-dgen minibatching of voice renders (retire the eseq-side halving).

---

## 5. Risks

- **Real vs render gap**: renders are clean; breaks have room, bleed, compression. Mitigation:
  augmentation, real-anchor stage B, through-synth stage C (trains on real audio directly).
- **Off-cloud codes are loud**: prior term in C; eval gate on peak; host-side tether in the UI.
- **Small real set** (85 / 94): val split of 10 is noisy; report per-hit, not only medians.
- **Feature parity** between host training features and the compiled patch (window type,
  frame phase, normalization): explicit parity test before shipping.
- **im2col memory**: `[B·T′, Cin·k]` columns materialize for the matmul (e.g. 64·86 × 480
  floats per layer at B = 64): fine at this size; the shift-and-matmul fallback avoids it.

## 6. Incidental findings (worth separate issues)

- `asStrided` backward is a plain reshape (G:700): wrong for overlapping windows, affects
  conv2d input grads and any `pool`/`windows` on a trainable path (fixed in phase 1, step 0).
- `.expand` / `.expandAxis` and `repeatView` gradients are TODO passthroughs (G:745, 772).
- `AdamMoments` gradient clipping is implemented but never enabled (TD:706).
- `--freeze` doc (TD:61) says audio zeroes frozen segments; code only does so in refine.
- `--init-from` silently re-initialises a leaf on size mismatch (e.g. `z:` across datasets).
- Glorot fan-in uses shape[0] for any rank (TD:447).
- `DGEN_FORCE_SCALAR=1` emits invalid C for the Latent Break Snare v5 Morph patch: vector
  results assigned to scalars (`l438 = vminq_f32(simd437, c13)`, 12 errors).

## 7. Implementation status (phase 1)

Landed: steps 0–8. Beads epic `dgen-k2f`; follow-ups `dgen-1rd` (encode-audio), `dgen-5q1`
(streaming features + parity), `dgen-1jj` (phase 3).

- **Step 0.** `asStrided` backward is a composed scatter-add for pool-shaped views
  (`Graph.poolViewGradient`, Gradients.swift); any other strided view on a gradient path is a
  `fatalError`. The stacked-conv FD test (`PoolGradientTests`) exposed **two more forward bugs**
  on the same path, both fixed: `composeTransforms` treated `asStrided` strides as absolute
  (wrong over a transposed concrete tensor: `x.transpose().pool()` read the untransposed base),
  and `GEMMPass` stripped any view chain (pool / pad / shrink included) and read the source's
  raw memory (`matmul` of im2col columns was wrong). `windows` now uses logical strides too.
  `(windows1d X @size k [@stride s])` exposes `pool`; Lisp `reshape` infers one `-1`.
- **Steps 1–2.** `(row-data @name N @shape [bands frames] X)`, `(latent-out Z)`,
  `DatasetContext.rowData`; the encoder is evaluated per step and bound into the voice's
  `latent-table` through `evaluate(…, latent:)`.
- **Steps 3–6.** `--train-only` prefixes (default: the leaves the encoder created); `codeL2`
  guarded; `encode` stage (code MSE + `--u-weight` decoded-param term + optional `--prior`
  nearest-code term); `--batch`, `--val` (held out from the first `--dataset`; several datasets
  are sampled evenly per minibatch); `--clip` wired; `--weight-decay` (decoupled).
- **Step 7.** Host features (EncoderFeatures.swift) follow the *compiled* chain rather than
  `timbreLogMelFrames` (which energy-normalises bands and subtracts a per-chunk mean, neither
  reproducible in-graph): frames from sample 0, periodic Hann, |DFT| (sum convention),
  `@ M` (export/mel_filters.json, [513, 64]), `log(· + 1e-5)`, per-band stats
  (export/feat_stats.json). `--augment`: gain ±6 dB, onset ±3 ms, ±1 st, tilt ±6 dB,
  noise −60…−35 dB.
- **Step 8.** Render keys the code table by the patch's table name and writes
  `renders/labels.json` (the codes.json format, so it feeds `--codes` directly).
- Incidental (§6) fixed: glorot fan-in (all axes but the last), `--freeze` doc, non-finite
  grads zeroed before clipping. Example patch: Examples/LatentEncoder/encoder.lisp.
- **Peak gate**: `peak_ratio_max` is relative to the loudest *prepared* training target (0.9
  after `prepare()`); `ceiling_peak_ratio_max` reports the learned codes' own renders for
  context.
