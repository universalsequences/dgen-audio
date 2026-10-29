# ModalSnareAD: a differentiable VILLAIN Snare and the Latent Snare instruments

This example trains a learned latent space for snare drums with dgen autograd, then distills it into
two realtime eseq instruments:

- **Latent Snare**: an MLP decoder (4 latent knobs + velocity -> 360 voice parameters) that runs once
  per hit and drives a closed-form 32-mode modal head, 16 noise bands and a 2-strand snare-wire
  model. Two snare picks plus a blend knob walk a straight line in latent space, with a perceptual
  warp so equal knob steps sound like equal steps.
- **Latent Snare Morph**: the same decoder re-run every 128 samples, so the snare morphs from a START
  to an END latent during the hit.

The teacher is eseq's VILLAIN Snare: 32 snares rendered with stems, fitted through a dgen renderer
that matches its head stem to < 7e-4 relative RMS.

The Swift sources are the trainer (`swift build -c release --product ModalSnareAD`). Everything
else, from rendering the dataset to building the installed instruments, is in `scripts/`. The
companion kick example is `../ModalKickAD`; it shares `scripts/modal_ad.py` and a few modules
from here.

## Swift sources

| File | What it is |
|---|---|
| `Head.swift`, `SnareModel.swift`, `Data.swift` | E1: VILLAIN Snare's head modal bank, band noise and output stage as a batched `[B, K]` frame-parallel DGenLazy renderer, plus dataset IO |
| `Fit.swift`, `FDCheck.swift`, `Parity.swift`, `PhysHead.swift`, `BestMetric.swift` | E1: per-snare fits, finite-difference gradient checks, stem parity, a Bessel-membrane head (a negative result), the MR-STFT scorer |
| `Latent.swift`, `LatentCommands.swift` | E2: the latent autodecoder (`latent-pretrain`, `latent-train`, `latent-fitz`, `latent-render`) |
| `LatentPath.swift` | E6: `latent-path`, a decoder fine-tune that evens out blend paths (analytic band-envelope surrogate, no audio) |
| `ModalSnareADCLI.swift` | subcommand dispatch |

## scripts/

Each stage directory holds the scripts of one research step. The step names (E0 to E8) are the
order the work was done in and are used throughout the code and the data layout.

| Directory | Step |
|---|---|
| `dataset/` | E0: the VILLAIN Snare stem-tap copy, the 156-render dataset (32 snares x 3 velocities, 40 blends, 20 PC points) and its exact mode table |
| `e1/`, `e1/physical/` | E1: the fixed noise-band table, noise-envelope init, metric floors, fit evaluation; Bessel-membrane analysis |
| `e2/` | E2: teacher targets, pretrain data, the numpy twin of the voice (`template.py`, `latent_np.py`), evaluation, export to an instrument |
| `e3proto/` | E3: the realtime instrument generator (`gen_dsp.py`), weight export, the CPU bench (`bench.py`, `bench_driver.c`), the noise-band spec (`NOISE_BANDS.md`), and `Latent Snare/` (instrument.json + ui.lisp templates) |
| `e4/` | E4: latent family-map envelopes for the panel (`build_family.py`, imported by `e6/build_panel.py` and the kick builders; run on its own it writes `e4/Latent Snare/data/family.json`) |
| `e5/` | E5: the snare-wire model (`wire_splice.py`, C twin `wires.c`), strand targets (`energy.py`), VILLAIN with an injected strand signal (`make_inject_copy.py`, `inject.py`) |
| `e7/` | E7: fit and distill the wire knobs into the decoder, splice the wire layer into the E2 instrument |
| `e6/` | E6: blend-path fine-tune inputs, re-anchoring, the per-pair perceptual warp, the installed **Latent Snare** (`build_instrument.py`, `build_panel.py`, `ui_template.lisp`, `write_presets.py`) |
| `e8/` | E8: the installed **Latent Snare Morph** (`gen_morph.py`, `build_stage.py`, `ui.tmpl.lisp`, `presets_renders.py`) and its checks |
| `modal_ad.py` | where code and data live (imported by every script) |
| `make_dgenlisp_tool.sh` | builds this checkout's DGenLisp into `$MODAL_AD_ROOT/tools/dgenlisp` |

E6 comes after E7 because it fine-tunes the E7 decoder.

## The data directory

Nothing the scripts produce is checked in: audio, renders, targets, weights and staged instruments
all go to `$MODAL_AD_ROOT`, which mirrors the stage directories (`$MODAL_AD_ROOT/data`,
`$MODAL_AD_ROOT/e1` ... `e8`, `$MODAL_AD_ROOT/kick`). When a script's docstring names a path like
`e2/runs/outbox_d4` or `e7/Latent Snare/`, it is relative to `$MODAL_AD_ROOT`, and the pipeline
step below that creates it is listed. The Swift binaries read the same directory (`MODAL_AD_ROOT`
must be set; they exit with a message otherwise).

## Requirements

- macOS on Apple Silicon. Training stages run on Metal; renders and checks use the C backend.
- Python 3 with numpy. scipy is optional: `e1/make_noise_bands.py` uses it when present.
- An eseq checkout (`ESEQ_ROOT`) with `./scripts/fetch_dgenlisp.sh` and
  `./scripts/fetch_dgen_toolchain.sh` run once. The scripts read VILLAIN Snare's sources from it,
  render through its `tools/audition`, and link with its hermetic clang/lld stage. `eseq instrument
  check` (from an eseq build) is used to validate installed instruments.

## Setup

```bash
cd <this dgen checkout>
export MODAL_AD_ROOT=~/modal-ad          # any empty directory
export ESEQ_ROOT=<eseq checkout>
S=$PWD/Examples/ModalSnareAD/scripts
B=$PWD/.build/release/ModalSnareAD
export DGEN_RUNTIME_INCLUDE=$PWD/toolchain/include DGEN_BINARY_AUDIT_TOOL=$PWD/scripts/audit-dgen-dylib.sh
swift build -c release --product ModalSnareAD
$S/make_dgenlisp_tool.sh                 # -> $MODAL_AD_ROOT/tools/dgenlisp/DGenLisp
export ESEQ_DGENLISP_TOOL=$MODAL_AD_ROOT/tools/dgenlisp/DGenLisp
```

The Latent instruments need this checkout's DGenLisp. The compiler eseq pins predates the dgen
fixes they rely on. VILLAIN Snare renders bit-identically with either compiler.

## Pipeline: from nothing to the installed instruments

Each command lists what it creates under `$MODAL_AD_ROOT`. Commands that take relative run paths
are run from the stage's data directory, as shown.

### E0: dataset

```bash
python3 $S/dataset/make_stems_copy.py        # villain_stems/ (VILLAIN Snare + head/source/strands taps)
python3 $S/dataset/render_villain_dataset.py # data/villain/ (156 renders + stems, manifest.json), data/villain_modes.json
python3 $S/dataset/villain_stats.py          # data/villain/stats.json (optional)
```

### E1: differentiable renderer and per-snare fits

```bash
python3 $S/e1/make_noise_bands.py            # e1/noise_bands/ (bands.json, bands.f32)
python3 $S/e1/noise_init.py                  # e1/noise_init.json
$B parity --ids all                          # head vs stems: worst 6.7e-4 relative RMS
$B fdcheck --head true --out $MODAL_AD_ROOT/e1/fdcheck_mse.json                       # optional
$B fdcheck --head true --loss spectral --out $MODAL_AD_ROOT/e1/fdcheck_spectral.json  # optional
$B fit --steps 400 --lr 0.03 --freeze-output always --init $MODAL_AD_ROOT/e1/noise_init.json --out $MODAL_AD_ROOT/e1/runs/r4_noise
$B fit --steps 200 --lr 0.01 --unfreeze-head 0 --head-l2 1e-3 --init $MODAL_AD_ROOT/e1/runs/r4_noise/params.json --out $MODAL_AD_ROOT/e1/fits
python3 $S/e1/noise_floor.py                 # e1/noise_floor/, e1/noise_floor.json
python3 $S/e1/eval_run.py $MODAL_AD_ROOT/e1/fits       # e1/fits/eval.json
python3 $S/e1/oracle_scores.py $MODAL_AD_ROOT/e1/fits  # e1/oracle_scores.json (optional)
```

E2 reads `e1/fits/params.json` (teacher noise envelopes) and `e1/fits/eval.json`. The Bessel
analysis is optional; `physfit` reads its outputs:

```bash
python3 $S/e1/physical/bessel_modes.py && python3 $S/e1/physical/bessel_poly.py   # e1/physical/bessel_{modes,poly}.json
python3 $S/e1/physical/table_vs_bessel.py    # e1/physical/table_vs_bessel.json
for n in 8 12 24 64; do python3 $S/e1/physical/bessel_null.py $n; done
$B physfit --steps 300 --lambda 1 --out $MODAL_AD_ROOT/e1/physical/assign_l1
python3 $S/e1/physical/summarize.py
```

### E2: latent autodecoder

```bash
cd $MODAL_AD_ROOT/e2
python3 $S/e3proto/gen_dsp.py --out "$MODAL_AD_ROOT/e2/Latent Snare" --random-weights --layout $MODAL_AD_ROOT/e2/decoder_layout.json
python3 $S/e2/targets.py                     # e2/targets.json (teacher rows)
python3 $S/e2/calib.py                       # e2/calib.json, e2/floor/ (metric floors for all 156)
python3 $S/e2/prep_pretrain.py               # e2/pretrain.json
$B latent-pretrain --d 4 --steps 12000 --lr 0.003 --zl2 1e-4 --box 3 --out $MODAL_AD_ROOT/e2/runs/prebox3_d4
export DGEN_METAL_KERNELS_PER_COMMAND_BUFFER=1   # split command buffers: avoids the display-GPU watchdog
$B latent-train --stage head --init runs/prebox3_d4/checkpoint.json --steps 120 --lr 3e-4 --anchor 10 --zl2 1e-4 --box 3 --eval-every 40 --group 24 --out $MODAL_AD_ROOT/e2/runs/headbox_d4
$B latent-train --stage out --init runs/headbox_d4/checkpoint.json --steps 360 --lr 3e-4 --anchor 10 --anchor-end 3 --zl2 1e-4 --box 3 --eval-every 60 --group 16 --out $MODAL_AD_ROOT/e2/runs/outbox_d4
python3 $S/e2/eval_e2.py runs/outbox_d4      # runs/outbox_d4/eval.json
python3 $S/e2/export_instrument.py runs/outbox_d4 "Latent Snare"   # e2/Latent Snare/ (weights, codes_std.json, dsp.lisp)
python3 $S/e2/swift_parity.py runs/outbox_d4                       # numpy twin vs the Swift trainer
python3 $S/e2/instrument_parity.py runs/outbox_d4 "Latent Snare"   # compiled instrument vs the trainer
```

The blend-line prior (`targets_blend.py`, `render_blend_targets.py` -> `e2/blend_targets/`,
`prep_pretrain.py --blend`) was tried and rejected; it is kept because `latent-train` can read it.

### E5 and E7: snare wires

```bash
python3 $S/e5/energy.py                      # e5/targets/<id>.{strand_out,out_r0}.wav, e5/energy.json
python3 $S/e5/make_inject_copy.py            # e5/villain_inject/
python3 $S/e7/fit_wires.py                   # e7/fit_wires.json, e7/rows/
python3 $S/e7/distill.py                     # e7/wire_targets.json, e7/weights_wires.json, e7/weights_wires_rebal.json
python3 $S/e7/build_instrument.py            # e7/Latent Snare/ (+ e7/variants/)
python3 $S/e7/parity.py                      # compiled vs twin (wires are chaotic: judged by band)
python3 $S/e7/eval_distill.py                # optional
```

### E6: even blends and the installed Latent Snare

```bash
python3 $S/e6/make_base_e7.py                # e6/runs/base_e7/ (E2 checkpoint + E7 readout)
cd $MODAL_AD_ROOT/e6
python3 $S/e6/prep_path.py runs/base_e7 runs/path_data_e7.json
python3 $S/e6/validate_surrogate.py $MODAL_AD_ROOT/e2/runs/outbox_d4 24   # optional
$S/e6/sweep.sh F5 --lr 1e-2 --even 10 --len 0.3 --steps 4000 --readout-null --rec 0 --lip 0.05 --lip-cap 1.6 --pairs 16
                                             # runs/F5 (latent-path), runs/F5_ra (re-anchored), runs/w100_F5_ra.json
python3 $S/e6/warp.py runs/F5_ra runs/warp_F5.json --workers 4
python3 $S/e6/measure_walks.py runs/F5_ra runs/walks_after.json --workers 4
python3 $S/e6/build_instrument.py runs/F5_ra runs/warp_F5.json "stage/Latent Snare"
python3 $S/e6/build_panel.py runs/F5_ra "stage/Latent Snare"
python3 $S/e6/write_presets.py runs/F5_ra runs/warp_F5.json runs/walks_after.json "stage/Latent Snare.presets" 10
python3 $S/e6/parity.py runs/F5_ra runs/warp_F5.json "stage/Latent Snare"
```

`e6/stage/Latent Snare/` and `Latent Snare.presets` are the instrument. To install, copy both into
the `Research/` folder of your eseq user instruments (`eseq paths` prints `user-instruments`), then
`eseq instrument check "Research/Latent Snare"` with `ESEQ_DGENLISP_TOOL` still exported.

### E8: Latent Snare Morph

```bash
python3 $S/e8/build_stage.py                 # e8/Latent Snare Morph/ (from e6/stage/Latent Snare)
python3 $S/e8/presets_renders.py             # e8/Latent Snare Morph.presets, e8/renders/
python3 $S/e8/build_stage.py --install "<eseq user-instruments>/Research"
python3 $S/e8/make_refs.py                   # e8/ref/, e8/build/ (check builds)
python3 $S/e8/t_parity.py                    # START = END vs the Latent Snare
python3 $S/e8/t_click.py                     # hop-boundary clicks vs a build that decodes every sample
python3 $S/e8/bench8.py                      # CPU, scripts/e3proto/bench.py method
```

## Reproducibility

The instrument builders are deterministic. Re-running `e2/export_instrument.py`,
`e7/build_instrument.py`, `e6/build_instrument.py` + `build_panel.py` + `write_presets.py` and
`e8/build_stage.py` on the trained runs reproduces the shipped Latent Snare and Latent Snare Morph
exactly, apart from comments. GPU training (`latent-pretrain`, `latent-train`, `latent-path`) is
not bit-reproducible, so expect a retrained decoder to land close to the published numbers but not
identical. Trained weights are not checked in.

## Renderer notes

**E1** (`fit`, `parity`, `fdcheck`, `physfit`): the head is VILLAIN Snare's analytic modal bank,
mode for mode: amplitude, decay, attack, onset `t0`, phase, and an exponentially decaying pitch
glide with its phase integrated in closed form, the Nyquist fade, the head and drum gates and
`head_v`. The noise and wire part is 16 fixed band-noise signals (`e1/noise_bands`, from
`scripts/e1/make_noise_bands.py`; the filter bank follows `scripts/e3proto/NOISE_BANDS.md`), each
scaled by a closed-form two-stage envelope, plus a head-coupled term `C_b * E(t)` with
`E(t) = sqrt(sum_k env_k(t)^2)`. The output stage is VILLAIN's drive mix
`x + d_amt (tanh(dv x) d_norm - x)`, the length fade and the output gain. All parameters are
bounded (log/sigmoid maps). `fit` trains 32 snares in one batched pass (MR-STFT lin + log, windows
128-2048, hop w/4) against `out.wav` and writes per-snare `*.fit.wav`, `*.target.wav`,
`*.fit_headpart.wav`, `*.fit_noisepart.wav`, `summary.json`, `params.json` and `loss.csv`.

**E2** (`latent-*`): per-snare latent codes z (D) plus velocity go through a tanh MLP
(D+1 -> 128 -> 128 -> P) into the segments of the realtime template (`Latent.swift`; layout
`$MODAL_AD_ROOT/e2/decoder_layout.json`, written by `scripts/e3proto/gen_dsp.py --layout`; numpy
twin `scripts/e2/template.py`). The renderer is the same math as the instrument: 32 closed-form
head modes, 2 head gates, 16 band-noise envelopes, drive, soft clipper, length fade.

- Audio stages backprop in two passes (`twoPass`): first the audio loss to a leaf u [B, P]
  (frame-summed dL/du), then sum(u * dL/du) plus anchors through the static decoder. A single
  graph would carry per-frame tapes through the MLP ([frames, H, P], more than 2^31 floats even at
  B = 12).
- `--train readout` (default) moves only W3/b3, with the `mode_freq` columns masked: decoder-wide
  Adam steps move every frequency by several cents.
- Batch tensors are persistent and updated in place per chunk. A fresh data tensor per step or per
  chunk compiles new kernels, and kept one runtime plus its buffers alive per chunk.
- `--box` adds a range hinge on random latent samples, which keeps off-code decodes inside the
  corpus parameter box.

**E6** (`latent-path`): fine-tunes W3/b3 inside the null space of the 96 anchor rows' readout
features (`--readout-null`), so reconstruction at the 32 snares stays exact while blend paths even
out under an analytic band-envelope surrogate (`scripts/e6/common.py`).

## Not checked in

All audio (dataset renders, stems, targets, fit and demo renders), noise tables (`bands.f32`),
pretrain data, trained runs and weights, and staged instruments. They are regenerated by the steps
above. The only data files in `scripts/` are small contracts and templates: the frozen E3 decoder
layout (`e3proto/decoder_layout.json`, 8 KB) and the E3 instrument templates
(`e3proto/Latent Snare/`).
