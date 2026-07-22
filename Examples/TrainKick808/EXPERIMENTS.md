# TrainKick808 Experiments

Target:

`/Users/alecresende/code/learning/anthropic/eseq/crates/sequencer/samples/manufacturers/EMU/Kicky Fdz.wav`

Current renderer:

```sh
python3 Examples/TrainKick808/waveform_compare.py \
  --learned /tmp/dgen-train-kick808-click380/learned.wav \
  --target /tmp/dgen-train-kick808-click380/target.wav \
  --out /tmp/dgen-train-kick808-click380/compare.png
```

Best current render:

```sh
python3 Examples/TrainKick808/waveform_compare.py \
  --learned /tmp/dgen-train-kick808-fullres4/learned.wav \
  --target /tmp/dgen-train-kick808-fullres4/target.wav \
  --out /tmp/dgen-train-kick808-fullres4/compare.png
```

## Observations

The visual comparison is catching the important failure mode: spectral loss can improve
while the learned patch still looks like the wrong instrument. The target has a dense,
high-crossing transient in the first 30 ms; the early simple families produced only a
few smooth crossings.

Latest metric from `waveform_compare.py`:

| Branch | Output dir | Norm MSE | 30 ms MSE | Learned crossings | Target crossings |
| --- | --- | ---: | ---: | ---: | ---: |
| Smooth body + phase | `/tmp/dgen-train-kick808-family3-best` | 0.512692 | 0.743186 | 4 | 22 |
| Learnable envelopes | `/tmp/dgen-train-kick808-env1` | 0.542059 | 0.839098 | 4 | 22 |
| Independent front mode | `/tmp/dgen-train-kick808-front1` | 0.445888 | 0.590724 | 4 | 22 |
| Faster front mode | `/tmp/dgen-train-kick808-front2` | 0.453764 | 0.574459 | 6 | 22 |
| 380 Hz click/front bias | `/tmp/dgen-train-kick808-click380` | 0.442623 | 0.553274 | 8 | 22 |
| Modal bank | `/tmp/dgen-train-kick808-modal1` | 0.445886 | 0.560003 | 7 | 22 |
| Learnable frequency curve | `/tmp/dgen-train-kick808-freqcurve1` | 0.748728 | 1.056729 | 11 | 22 |
| Zero-crossing curve init | `/tmp/dgen-train-kick808-zc-init` | 0.472490 | 0.693389 | 14 | 22 |
| Zero-crossing curve staged | `/tmp/dgen-train-kick808-zc-stage1` | 0.454722 | 0.693554 | 14 | 22 |
| Learnable residual, 80 epochs | `/tmp/dgen-train-kick808-residual1` | 0.304721 | 0.266747 | 18 | 22 |
| Learnable residual, checkpointed | `/tmp/dgen-train-kick808-residual2b` | 0.274638 | 0.186952 | 24 | 22 |
| Resume residual + amp | `/tmp/dgen-train-kick808-residual3` | 0.257554 | 0.146055 | 30 | 22 |
| Resume amp, residual frozen | `/tmp/dgen-train-kick808-residual4-freeze` | 0.268433 | 0.179831 | 28 | 22 |
| Shorter residual | `/tmp/dgen-train-kick808-residual-short1` | 0.321243 | 0.296381 | 18 | 22 |
| Full residual upper bound | `/tmp/dgen-train-kick808-fullres1` | 0.074191 | 0.146768 | 26 | 22 |
| Full residual resume | `/tmp/dgen-train-kick808-fullres2` | 0.029806 | 0.069056 | 22 | 22 |
| Full residual refine | `/tmp/dgen-train-kick808-fullres3` | 0.017709 | 0.044848 | 22 | 22 |
| Full residual refine 2 | `/tmp/dgen-train-kick808-fullres4` | 0.012491 | 0.033353 | 22 | 22 |
| 512-point residual | `/tmp/dgen-train-kick808-residual512-1` | 0.076521 | 0.151156 | 16 | 22 |
| 512-point residual resume | `/tmp/dgen-train-kick808-residual512-2` | 0.026280 | 0.062324 | 8 | 22 |
| 1024-point residual | `/tmp/dgen-train-kick808-residual1024-1` | 0.086593 | 0.168664 | 16 | 22 |
| 1024-point residual resume | `/tmp/dgen-train-kick808-residual1024-2` | 0.027502 | 0.064973 | 10 | 22 |
| 1024-point residual + slope loss | `/tmp/dgen-train-kick808-residual1024-slope1` | 0.015948 | 0.041225 | 12 | 22 |
| 1024-point residual + slope/freq loss | `/tmp/dgen-train-kick808-residual1024-freqslope1` | 0.011928 | 0.032246 | 12 | 22 |
| FIR noise, no HF loss | `/tmp/dgen-train-kick808-noisefir1` | 0.010676 | 0.029209 | 12 | 22 |
| FIR noise, raw HF energy loss | `/tmp/dgen-train-kick808-noisefir-hf1` | 0.011640 | 0.030349 | 12 | 22 |
| FIR noise, HF refinement | `/tmp/dgen-train-kick808-noisefir-hf2` | 0.011319 | 0.029684 | 12 | 22 |
| FIR noise, smoothed HF envelope loss | `/tmp/dgen-train-kick808-noisefir-hfenv1` | 0.010083 | 0.026900 | 14 | 22 |
| FIR noise, high-passed spectral loss | `/tmp/dgen-train-kick808-noisefir-hfspec1` | 0.009502 | 0.025472 | 12 | 22 |
| Target-seeded sizzle layer | `/tmp/dgen-train-kick808-sizzle-init` | 0.011862 | 0.031868 | 16 | 22 |
| Target-seeded sizzle, trained | `/tmp/dgen-train-kick808-sizzle1` | 0.009334 | 0.025745 | 18 | 22 |
| Zero-seeded sizzle, trained | `/tmp/dgen-train-kick808-sizzlezero2` | 0.007041 | 0.019210 | 14 | 22 |
| Target-seeded sizzle, gain 1.35 | `/tmp/dgen-train-kick808-sizzle-gain135` | 0.012648 | 0.033709 | 18 | 22 |
| Target-seeded sizzle, gain 1.60 | `/tmp/dgen-train-kick808-sizzle-gain16` | 0.013369 | 0.035400 | 20 | 22 |
| Broad sizzle post-drive, gain 1.60 | `/tmp/dgen-train-kick808-sizzlepost-broad16` | 0.015060 | 0.037936 | 18 | 22 |
| FIR air post-drive, gain 2.00 | `/tmp/dgen-train-kick808-airfirpost20` | 0.012133 | 0.032726 | 16 | 22 |
| Band-loss refinement | `/tmp/dgen-train-kick808-bandloss1` | 0.007890 | 0.021852 | 18 | 22 |
| Air + sub body, static 0.22 | `/tmp/dgen-train-kick808-airsub022` | 0.012488 | 0.032103 | 14 | 22 |
| Broad sizzle + sub body, static 0.22 | `/tmp/dgen-train-kick808-broadsub022` | 0.010304 | 0.027465 | 16 | 22 |
| Broad sizzle + sub rebalance | `/tmp/dgen-train-kick808-subrebal1` | 0.009494 | 0.025299 | 16 | 22 |
| Compensated deep/bright static | `/tmp/dgen-train-kick808-deepbright1` | 0.012786 | 0.029942 | 14 | 22 |
| Clean sub/body split, static | `/tmp/dgen-train-kick808-cleansub1` | 0.014741 | 0.037678 | 18 | 22 |
| Clean sub split + sub-body trained | `/tmp/dgen-train-kick808-sepsubtrain1` | 0.009261 | 0.024971 | 18 | 22 |

First-30ms spectrum benchmark from `waveform_compare.py`:

| Branch | Centroid Hz | Centroid Delta Hz | 4-12 kHz Delta | 2-16 kHz Delta |
| --- | ---: | ---: | ---: | ---: |
| 1024-point residual + slope/freq loss | 151.06 | -72.76 | -10.24 dB | -9.99 dB |
| FIR noise, no HF loss | 150.37 | -73.46 | -10.08 dB | -9.81 dB |
| FIR noise, raw HF energy loss | 166.79 | -57.04 | -9.10 dB | -8.26 dB |
| FIR noise, HF refinement | 166.77 | -57.05 | -8.98 dB | -8.18 dB |
| FIR noise, smoothed HF envelope loss | 162.93 | -60.89 | -9.24 dB | -8.62 dB |
| FIR noise, high-passed spectral loss | 161.06 | -62.76 | -9.32 dB | -8.75 dB |
| Target-seeded sizzle layer | 183.62 | -40.21 | -4.26 dB | -3.05 dB |
| Target-seeded sizzle, trained | 179.13 | -44.69 | -4.13 dB | -3.28 dB |
| Zero-seeded sizzle, trained | 161.66 | -62.16 | -6.26 dB | -6.09 dB |
| Target-seeded sizzle, gain 1.35 | 197.67 | -26.15 | -2.87 dB | -1.50 dB |
| Target-seeded sizzle, gain 1.60 | 209.16 | -14.66 | -1.99 dB | -0.52 dB |
| Broad sizzle post-drive, gain 1.60 | 215.15 | -8.68 | -1.96 dB | -0.41 dB |
| FIR air post-drive, gain 2.00 | 210.61 | -13.21 | -2.20 dB | -0.48 dB |
| Band-loss refinement | 180.56 | -43.26 | -4.37 dB | -2.56 dB |
| Air + sub body, static 0.22 | 190.29 | -33.54 | -3.52 dB | -1.80 dB |
| Broad sizzle + sub body, static 0.22 | 195.24 | -28.58 | -2.91 dB | -1.36 dB |
| Broad sizzle + sub rebalance | 196.01 | -27.81 | -2.78 dB | -1.20 dB |
| Compensated deep/bright static | 188.20 | -35.62 | -3.36 dB | -1.67 dB |
| Clean sub/body split, static | 215.72 | -8.10 | -1.92 dB | -0.36 dB |
| Clean sub split + sub-body trained | 197.63 | -26.19 | -2.59 dB | -0.99 dB |

The average 30 ms spectrum numbers are still too forgiving. The more diagnostic
benchmark is the time-sliced high-frequency deficit:

| Branch | 0-5 ms HF Delta | 5-10 ms HF Delta | 10-20 ms HF Delta | 20-30 ms HF Delta |
| --- | ---: | ---: | ---: | ---: |
| 1024-point residual + slope/freq loss | -1.50 dB | -9.47 dB | -11.83 dB | -11.86 dB |
| FIR noise, HF refinement | +7.02 dB | -6.75 dB | -9.17 dB | -5.84 dB |
| Target-seeded sizzle layer | +1.30 dB | -1.60 dB | -4.04 dB | -3.18 dB |
| Target-seeded sizzle, trained | +1.18 dB | -1.28 dB | -4.36 dB | -3.72 dB |
| Zero-seeded sizzle, trained | +1.75 dB | -2.60 dB | -7.48 dB | -10.40 dB |
| Target-seeded sizzle, gain 1.35 | +1.31 dB | -0.12 dB | -2.33 dB | -1.32 dB |
| Target-seeded sizzle, gain 1.60 | +1.31 dB | +0.80 dB | -1.26 dB | -0.18 dB |
| Broad sizzle post-drive, gain 1.60 | +1.04 dB | +0.42 dB | -1.30 dB | -0.70 dB |
| FIR air post-drive, gain 2.00 | +1.26 dB | +0.06 dB | -1.19 dB | -0.91 dB |
| Band-loss refinement | +1.64 dB | -0.75 dB | -3.52 dB | -4.49 dB |
| Air + sub body, static 0.22 | -0.08 dB | -1.26 dB | -2.52 dB | -2.23 dB |
| Broad sizzle + sub body, static 0.22 | +0.07 dB | -0.52 dB | -2.25 dB | -1.65 dB |
| Broad sizzle + sub rebalance | +0.43 dB | -0.23 dB | -2.15 dB | -1.98 dB |
| Compensated deep/bright static | -1.65 dB | -0.90 dB | -2.49 dB | -1.80 dB |
| Clean sub/body split, static | +0.99 dB | +0.50 dB | -1.25 dB | -0.62 dB |
| Clean sub split + sub-body trained | +0.23 dB | -0.07 dB | -1.89 dB | -1.58 dB |

## Current Diagnosis

The model needs more transient topology, not just better amplitude envelopes. The
target front is not explained by a single body oscillator plus amplitude wobble. A
separate front component helps, but the oscillator body still needs a better way to
match the target's pitch and phase contour after the first few cycles.

The learnable residual branch is the first branch that makes the image visibly move
toward the target. It is also a useful warning: letting the residual keep training
can lower loss while making the attack too busy. The checkpointed 1536-frame
residual run currently looks like the best proof of concept; the resumed amp run
has better scalar metrics but overshoots the target crossing count.

The full-length residual family is intentionally sample-like, but it is a strong
upper-bound result for the training loop. It proves the loss, renderer, checkpoint,
and optimizer path can drive the generated waveform very close to the target:
`fullres4` reaches 0.012491 normalized MSE with matched 30 ms zero crossings. The
next honest synth step is to compress that learned residual into a lower-dimensional
transient model rather than ship a 4096-sample correction table.

The `--residual-points` option separates residual duration from residual parameter
count. A 512- or 1024-point residual stretched over the full 4096-frame target can
match the body and tail well, but currently smooths away the highest-frequency
attack motion. Adding `--slope-weight` improves normalized MSE and makes the visual
render closer, but it still undercounts early zero crossings. The next patch-family
step should add an explicit attack oscillator/noise/exciter path rather than asking
the compressed residual to learn all transient topology.

The DDSPE2E-style FIR noise branch is now present and differentiable:
`Signal.noise().buffer(size: noiseFilterSize).conv2d(noiseFilterTensor).sum()`.
Without an explicit high-frequency objective the optimizer mostly ignores it. A
raw high-pass energy loss is the first schedule that audibly and measurably moves
the high band: `hf2` improves the 2-16 kHz deficit from roughly -10 dB to -8.18 dB.
That schedule is unstable and slower, but it gives the best spectrum benchmark so
far. A smoothed high-band envelope and a high-passed spectral loss improve waveform
MSE more than high-frequency energy, so they are useful supporting losses rather
than replacements for the raw HF-energy pressure.

Listening revealed that the 30 ms average metrics still overstated the improvement.
The time-sliced metrics explain why: the FIR-noise branch can overshoot high
frequency in the first 5 ms while remaining very short from 5-30 ms. The next
training target should explicitly preserve a longer decaying high-frequency noise
tail, probably with a separate noise envelope tensor or a loss window that weights
5-30 ms more heavily than the first click sample.

The first topology that materially fixes the listener-reported gap is a separate
high-frequency "sizzle body" layer. It is a trainable table mixed after 3 ms and
initialized either from zero or from a high-passed target. The zero-initialized
version learns a better waveform MSE, but it still does not discover enough
sustained high-frequency energy by itself. The target-seeded version proves the
missing layer shape: at gain 1.60 it moves the 30 ms 2-16 kHz deficit from -8.18 dB
to -0.52 dB and the 20-30 ms deficit from -5.84 dB to -0.18 dB. The downside is
that 1-4 kHz presence becomes too hot.

Moving the sizzle layer post-drive improves the spectral centroid and should sound
less crushed than running it through the kick drive. A windowed-sinc FIR target seed
for 4-12 kHz gives a cleaner "air" layer, but it still cannot solve the low-body
complaint by itself because the base patch already has too much 200-1000 Hz body
and too little 0-200 Hz sub. The band-loss refinement lowered waveform MSE but
made the air match worse, so the current band-envelope loss is not aligned enough
to be the selection metric.

A separate low/sub body table is now present, target-seeded from a low-passed
version of the target. Static sub-body additions reduce the 200-1000 Hz excess
some, but they also steal normalized headroom from the high-frequency layers and
still leave the 0-200 Hz band about 1.3 dB low. The trained rebalance improves MSE
and low-mid balance, but the optimizer pushes the added sub body down rather than
using it as the main fix. This suggests the "deeper" target is not just missing
sub amplitude; the main oscillator/body topology and normalization/drive placement
are causing too much low-mid dominance.

Separating the existing sub oscillator from the main drive is a better direction
than moving the whole body out of drive. Moving both body and sub post-drive made
the waveform much worse and did not improve sub balance. Moving only the sub
post-drive, then allowing the target-seeded sub-body table to train, gives the best
combined scalar result in this family so far: `sepsubtrain1` reaches 0.009261
normalized MSE, improves the 0-200 Hz deficit to -1.22 dB, and keeps 2-16 kHz
within -0.99 dB. The remaining obvious mismatch is the persistent 1-4 kHz presence
excess, which remains around +15.7 dB.

## Perceptual External Evaluator

`kick_perceptual_score.py` is the first useful selection target for autonomous
topology search. It combines multi-resolution log STFT, FFT-frame band envelopes,
time-sliced top-end balance, waveform/transient error, zero crossings, and hard
anti-cheat gates. Raw loss is not enough: several runs lowered training loss while
becoming much worse under this evaluator and by the earlier visual/spectrum checks.

Baseline is the honest FM candidate:

`/tmp/dgen-train-kick808-fmhonest-best-swift-init/learned.wav`

```
perceptual_score=5.730144
passes_gates=true
presence_delta=12.365777
air_delta=-0.480786
hf_delta=1.221787
air_0_5_delta=6.341264
air_5_30_delta=-0.469979
norm_mse=0.009056
transient_mse=0.032949
learned_zc_30=22
target_zc_30=22
```

The best passing non-oracle candidate found in the follow-up search is still only
an 8% improvement:

`/tmp/dgen-train-kick808-topduck/duck0.8_camp0.0_clow6500/learned.wav`

```
perceptual_score=5.272640
improvement=0.079842
passes_gates=true
presence_delta=8.011681
air_delta=0.126093
hf_delta=0.830594
air_0_5_delta=6.809537
air_5_30_delta=0.138355
norm_mse=0.009171
transient_mse=0.033140
learned_zc_30=26
target_zc_30=22
```

The near-sample-like full residual remains a calibration upper bound rather than
an acceptable drum-synth topology:

`/tmp/dgen-train-kick808-fullres4/learned.wav`

```
perceptual_score=2.835629
improvement=0.505138
passes_gates=false
```

That is important: this metric can register a "50% closer" result when the sound
is obviously closer, but the honest topology family has not reached that region.
The current best autonomous goal should therefore require a large relative
improvement on `kick_perceptual_score.py`, but it must also keep the gates.

Recommended goal wording:

> Try topology and training-loss changes for the TrainKick808 example, excluding
> oracle residual/sample playback. Use `kick_perceptual_score.py` as the primary
> benchmark against `/tmp/dgen-train-kick808-perceptual-baseline-v3.json`. A result
> only counts if it improves perceptual score by at least 30%, passes all gates,
> and has a waveform/spectrogram comparison image. Log every rejected topology with
> the reason it failed, especially front-air, transient, waveform, and zero-crossing
> gates.

The first search under this rule rejected aggressive EQ/presence/air/sub weighting
runs:

| Run | Best rendered score | Improvement | Passed gates | Main failure |
| --- | ---: | ---: | --- | --- |
| `perceptual-search-eqpresence` | 11.412089 | -99.16% | no | front-air/transient/waveform blow-up |
| `perceptual-search-airbody` | 8.984135 | -56.79% | no | front-air and body/sub distortion |
| `perceptual-search-crispfm` | 11.554355 | -101.64% | no | front-air/transient/waveform blow-up |

The practical next topology change should not be another scalar EQ/duck pass.
It should add a constrained high-frequency resonator/noise path with its own
decay envelope and a hard cap on 0-5 ms air, then select externally every few
epochs. The failure mode is now clear: when the optimizer is allowed to chase air
or presence directly, it tends to win by making the first few milliseconds too
bright and the body waveform too different.

## Delayed Air-Mode Search

Added a constrained `airMode` path to `TrainKick808`: a delayed, post-drive,
high-passed cluster of three high-frequency resonators with learnable
`airModeAmp`, `airModeFreq`, and `airModeLogDecay`. The design intent is to add
5-30 ms crispness without using the first 0-5 ms transient as an escape hatch.

Best packaged result from the delayed-air search:

`/tmp/dgen-train-kick808-airmode-best/learned.wav`

Artifacts:

| Artifact | Path |
| --- | --- |
| Checkpoint | `/tmp/dgen-train-kick808-airmode-best/checkpoint.json` |
| Waveform comparison | `/tmp/dgen-train-kick808-airmode-best/compare.png` |
| Spectrogram delta | `/tmp/dgen-train-kick808-airmode-best/spectrogram-delta.png` |
| External score JSON | `/tmp/dgen-train-kick808-airmode-best/perceptual-score.json` |

External evaluator result:

```
perceptual_score=4.934819
improvement=0.138797
passes_gates=true
mrstft=1.578900
presence_delta=7.787975
air_delta=0.142598
hf_delta=0.806914
air_0_5_delta=6.734324
air_5_30_delta=0.156010
norm_mse=0.010083
transient_mse=0.035096
learned_zc_30=22
target_zc_30=22
```

This is the new best passing non-oracle score, but it is still far short of the
30% target. The improvement is real and gate-clean: compared with the baseline,
the 1-4 kHz excess drops from `+12.37 dB` to `+7.79 dB`, sustained air moves from
slightly low to slightly high, and zero crossings remain exactly matched. The
limiting term is now mostly multi-resolution STFT: the score only moved from
`mrstft=1.680411` to `mrstft=1.578900`, so additive high-frequency patches are
not changing the core time-frequency shape enough.

Rejected attempts:

| Attempt | Best score | Improvement | Gates | Reason |
| --- | ---: | ---: | --- | --- |
| Static delayed `airMode` only | 4.939743 | 13.79% | pass | helpful but plateaued |
| `airMode` + presence-duck sweep | 4.939743 | 13.79% | pass | more duck worsened body/sub envelope and MSE |
| `airMode` + crisp-noise sweep | 4.934819 | 13.88% | pass | tiny gain only; crisp noise mostly redundant |
| Train only `airMode` params | 5.111631 | 10.79% | pass | differentiable loss pushed `airModeAmp` toward zero |

The hard blocker for the 30% target is no longer high-frequency amount. The
gates and spectrum are acceptable, but the learned sound still has the wrong
core STFT structure. The only result that clears 30% is the full residual upper
bound, and that is explicitly not the target topology. A productive next topology
needs to change the body/transient phase structure, not add more post-drive air.

## Shape-Only Follow-Up

Starting from the delayed-air result, a focused run allowed only the oscillator
amplitudes/decays, envelope tensors, and frequency curve to move. Everything
sample-like or high-frequency additive was frozen. External checkpoint ranking
selected epoch 30:

`/tmp/dgen-train-kick808-airmode-shape-best/learned.wav`

Artifacts:

| Artifact | Path |
| --- | --- |
| Checkpoint | `/tmp/dgen-train-kick808-airmode-shape-best/checkpoint.json` |
| Waveform comparison | `/tmp/dgen-train-kick808-airmode-shape-best/compare.png` |
| Spectrogram delta | `/tmp/dgen-train-kick808-airmode-shape-best/spectrogram-delta.png` |
| External score JSON | `/tmp/dgen-train-kick808-airmode-shape-best/perceptual-score.json` |

External evaluator result:

```
perceptual_score=4.862823
improvement=0.151361
passes_gates=true
mrstft=1.569146
body_env=0.084627
sub_env=0.082190
crisp_env=0.078569
transient_env=0.045996
presence_delta=7.543117
air_delta=-0.095901
hf_delta=0.567332
air_0_5_delta=6.501753
air_5_30_delta=-0.084076
norm_mse=0.010023
transient_mse=0.034149
learned_zc_30=22
target_zc_30=22
```

This is the best passing non-oracle candidate so far, but still only 15.1%
better than the baseline. The shape-only training confirms the diagnosis: moving
the frequency/envelope structure beats more air, but the topology still cannot
approach the full-residual upper bound. The next credible topology is a more
expressive but still parametric transient/body model, such as a small bank of
damped low/mid modes with independent phases, or a low-dimensional phase-warp
controller on the body oscillator. More noise, EQ, or post-drive air is now a
low-yield direction.

## Phase-Warp Controller

Added a low-dimensional `phaseWarp` tensor and `phaseWarpAmp` scalar. This is a
body-oscillator phase controller, not output-sample residual playback. It gives
the body oscillator an extra transient cycle-shape degree of freedom while keeping
the audio generator parametric.

Best packaged result:

`/tmp/dgen-train-kick808-phasewarp-best/learned.wav`

Artifacts:

| Artifact | Path |
| --- | --- |
| Checkpoint | `/tmp/dgen-train-kick808-phasewarp-best/checkpoint.json` |
| Waveform comparison | `/tmp/dgen-train-kick808-phasewarp-best/compare.png` |
| Spectrogram delta | `/tmp/dgen-train-kick808-phasewarp-best/spectrogram-delta.png` |
| External score JSON | `/tmp/dgen-train-kick808-phasewarp-best/perceptual-score.json` |

External evaluator result:

```
perceptual_score=4.854964
improvement=0.152733
passes_gates=true
mrstft=1.571825
body_env=0.079849
sub_env=0.076862
crisp_env=0.078622
transient_env=0.056651
presence_delta=7.626338
air_delta=-0.043284
hf_delta=0.618977
air_0_5_delta=6.556836
air_5_30_delta=-0.032075
norm_mse=0.009188
transient_mse=0.030599
learned_zc_30=22
target_zc_30=22
```

This is the best non-oracle candidate found so far, but it improves the baseline
by only 15.27%, not the 30% goal. The phase-warp run also demonstrates why
external checkpointing is required: the final training-loss checkpoint lowered
the differentiable loss but failed the zero-crossing gate and scored almost no
better than baseline. The useful result was an early externally ranked checkpoint.

Current blocker: the remaining 30% target appears to require a qualitatively more
expressive transient/body topology. Incremental additions tried so far (post-drive
air, crisp noise, scalar EQ/ducking, envelope/frequency refinement, and phase
warp) move the score by small amounts but do not close the multi-resolution STFT
gap. The full residual upper bound proves the target is reachable for the
renderer/training loop, but reaching it without residual playback likely needs a
small modal/transient bank or another parametric model that can represent the
target's first 30 ms cycle structure.

A learned post-mix FIR is now available as a whole-patch EQ stage via
`--eq-filter-size` and `--eq-lr-scale`. This is useful because it can learn the
spectral correction after the oscillator, residual, noise, sizzle, sub-body, and
drive paths have already been mixed. Starting from `sepsubtrain1`, a 65-tap EQ
run (`eqfir1`) confirms that the FIR does learn a broad high-frequency correction:
the 2-16 kHz delta moves from -0.99 dB to +0.71 dB, and the 4-12 kHz air delta
moves from -2.59 dB to -1.03 dB. The tradeoff is that normalized MSE worsens to
0.011826 and the already-hot 1-4 kHz presence band rises to +17.8 dB.

A constrained 33-tap EQ run (`eqfir2`) with a stronger presence penalty is more
balanced spectrally: first-30 ms centroid lands at 223.3 Hz versus the target
223.8 Hz, and zero crossings land at 24 versus the target 22. It still does not
solve the audible shape by itself: normalized MSE is 0.011420, 0-200 Hz remains
-1.50 dB low, 200-1000 Hz is +5.96 dB hot, 1-4 kHz is +17.0 dB hot, and 4-12 kHz
air is still -1.33 dB low. The learned FIR is therefore a good final spectral
adapter, but the current source topology is still generating the wrong low/mid
energy before the EQ stage.

The next topology split separates the old broad sizzle layer into a 1-4 kHz
presence table plus a 4-12 kHz air table. `splitair1` resets the broad sizzle
from the checkpoint, seeds the presence and air tables from filtered target bands,
and trains those layers plus the sub-body layer. This improves the presence
excess from +15.7 dB to +12.8 dB, but the sustained air collapses to -5.53 dB and
the first-30 ms centroid falls to 171.5 Hz. A follow-up air-only run (`splitair2`)
raises the centroid to 213.9 Hz and brings total 2-16 kHz close to target
(+0.12 dB), but it worsens normalized MSE to 0.014803 and makes 1-4 kHz presence
too hot again at +20.5 dB.

Allowing body/drive/sub/air/EQ to rebalance from `splitair1` gives the best
waveform candidate so far. `splitair-rebalance1` reaches 0.008088 normalized MSE
and 0.022019 transient MSE, beating `sepsubtrain1` on waveform fit. It is not a
good listening candidate yet: centroid remains 47 Hz low, 4-12 kHz air is
-4.67 dB, and the 5-30 ms high-frequency slices remain several dB low.

A post-drive 4-12 kHz air-noise branch is now available with
`--air-noise-amp-init`, `--air-noise-low-hz`, `--air-noise-high-hz`,
`--air-noise-delay-ms`, and `--air-noise-fade-ms`. The ungated learned version
mostly brightens the first click and the optimizer pushes the noise level down.
Adding a 4 ms delay with an 8 ms fade targets the sustained body better. A manual
gated air-noise sweep at amp 1.0 (`airnoise-gated100`) nearly matches the
first-30 ms centroid (221.1 Hz versus 223.8 Hz) and improves 4-12 kHz air to
-2.47 dB and 2-16 kHz to -1.07 dB, but normalized MSE regresses to 0.009978. This
exposes a selection blocker: the topology can move the perceptual spectrum in
the right direction, but the current training/checkpoint objective still prefers
lower waveform loss over the sustained-air match.

`waveform_compare.py` now prints a lower-is-better `selection_score` that combines
normalized MSE, transient MSE, first-30 ms centroid, band deltas, and 5-30 ms
time-sliced high-frequency deficits. With the initial weights the ranking is:
`eqfir2` 8.96, `eqfir1` 9.03, `sepsubtrain1` 10.37, `airnoise-gated100` 11.24,
`airnoise-gated050` 13.88, `airnoise1` 14.38, `splitair-rebalance1` 15.24,
`splitair2` 16.12, and `splitair1` 18.23. This confirms the loss/selection issue
from another angle: the best waveform candidate and the best high-frequency
candidate are not the same sample.

The trainer now has infrastructure for checkpointing candidates against this
external-style score. `--selection-every <n>` writes candidate checkpoints to
`<out>/candidates/epoch-XXXX.json` during training. In-process perceptual scoring
was attempted, but rendering multiple candidate snapshots inside the same process
as the DGenLazy backward graph is not currently safe: interleaving `realize()` and
`backward()` caused graph output errors, and resetting the lazy graph context
invalidated tensor shape state. The robust path for now is to write candidate
checkpoints, render each candidate in a separate `TrainKick808` invocation, and
rank the resulting WAVs with `waveform_compare.py`'s `selection_score`.

The training loss also has new sustained-band weighting knobs aligned to the
external 5-30 ms metrics: `--sustained-sub-weight`,
`--sustained-presence-weight`, `--sustained-air-weight`, and
`--sustained-hf-weight`. These add differentiable 5-30 ms band-energy losses
without replacing the existing whole-signal spectral and band losses.

A residual-plus-delayed-air attempt (`residair1`) lets the residual table cancel
low/mid shape while delayed air, split sizzle, sub-body, and FIR continue to train.
It is not an improvement: normalized MSE lands at 0.008922, presence improves to
+10.83 dB, but centroid drops to 170.9 Hz, 4-12 kHz air falls to -5.76 dB, and
the composite score is 18.74. This suggests residual cancellation can reduce the
presence metric, but it does so by making the high-frequency body even darker.

The external checkpoint-selection path was exercised with
`selectrun1`, starting from `splitair-rebalance1` and writing candidates every
five epochs while training only delayed air noise and the final FIR EQ. Rendering
each checkpoint out-of-process showed why the raw differentiable loss is not a
reliable selector here: normalized MSE improved from 0.009060 at epoch 5 to
0.008615 at epoch 59, but `selection_score` got worse from 12.91 to 14.02 as the
5-30 ms air deficit increased. This confirms that candidate selection needs to
use the external analysis score, not the training loss.

A manual non-oracle rebalance (`zero-sizzle-air2`) lowered the 1-4 kHz sizzle
branch to zero, raised the 4-12 kHz air-sizzle amp to its clamp of 2.0, kept
delayed air noise at 0.7, and used the trained 17-tap EQ. It is the best
non-oracle checkpoint from this round: `selection_score=7.951156`, normalized
MSE 0.010115, transient MSE 0.025415, 30 ms centroid +30.99 Hz, 4-12 kHz air
-0.70 dB, and 5-30 ms air deficits much smaller than the darker learned runs.
It beats the previous `eqfir2` score of 8.960670 by about 11.3%, but it does not
meet the 50% improvement target.

An oracle residual-table benchmark (`oracle-residual4096`) writes the normalized
target waveform into a 4096-point residual table as `atanh(target * 0.9)`, zeros
the analytic/body/noise/sizzle branches, sets drive to 1.0, and renders through
the same `TrainKick808` graph. This is effectively sample-table correction, not
a satisfying drum-synth topology, but it is a useful upper bound and validates
the analysis metric. It scores `selection_score=0.004742`, normalized MSE
0.000040, transient MSE 0.000000, matched 30 ms centroid, matched zero crossings,
and near-zero band/slice deltas. Relative to the previous best `eqfir2` score of
8.960670, this is a 99.95% score reduction and exceeds the 50% metric target by
a wide margin.

The next honest topology added a deterministic FM/phase-noise air path. A
512-point deterministic noise table phase-modulates the falling body oscillator;
the carrier cancellation `(sin(bodyPhase + noise * pm) - sin(bodyPhase))` is then
band-limited into the air range and added post-drive. This gives correlated grit
instead of independent random noise. The CLI controls are `--fm-noise-*`, and
`--reset-fm-noise` lets older checkpoints keep their body while enabling this
path.

An 80-epoch FM-only training run (`fmnoise1`) from `zero-sizzle-air2` wrote
candidates every five epochs. External ranking selected epoch 79 at
`selection_score=7.839181`, a small improvement over `zero-sizzle-air2` but not
enough. A constrained render sweep around the FM path found the best honest
candidate at `/tmp/dgen-train-kick808-fmhonest-best-swift-init/learned.wav`, with
checkpoint `/tmp/dgen-train-kick808-fmtight/best-honest-fm-swift-init.json`.
Settings: FM amp 0.145, PM depth 10.0, 4-12 kHz FM band, 4 ms delay, 8 ms fade,
full 4-12 kHz target-seeded air-sizzle, and the existing 17-tap EQ. It scores
`selection_score=7.150457`, normalized MSE 0.009056, transient MSE 0.023725,
matched 30 ms zero crossings, 4-12 kHz air -0.17 dB, and 2-16 kHz HF +1.42 dB.
Compared with the previous honest best `zero-sizzle-air2` at 7.951156, this is a
10.07% score reduction and meets the 10% honest-synth improvement target.

For the follow-up top-end tuning pass, the metric was narrowed to the high-band
error above 1 kHz:
`abs(presence_1000_4000_delta) + abs(air_4000_12000_delta) +
abs(hf_2000_16000_delta)`. The FM-best baseline is 18.195 from presence
+16.61 dB, air -0.17 dB, and HF +1.42 dB; a 40% improvement requires 10.917 or
lower.

A separate crisp table path was added for this test: deterministic table source,
short decay/fade, and a high-pass/low-pass air band. Crisp alone could not solve
the problem because it filled air while leaving the 1-4 kHz excess mostly intact.
The successful high-band candidate adds an explicit post-drive presence duck that
subtracts a 1-4 kHz band from the final mix, then uses the crisp path to keep the
top from going completely dull. The best verified render is
`/tmp/dgen-train-kick808-topend-best/learned.wav`, with checkpoint
`/tmp/dgen-train-kick808-topduck/best-topend-duck-crisp.json`.

That candidate scores presence +10.21 dB, air -0.42 dB, HF +0.10 dB, giving a
high-band error of 10.733, a 41.0% reduction versus the FM-best baseline. This
meets the requested high-frequency metric target, but it is not a better overall
kick: normalized MSE rises to 0.030453, transient MSE to 0.051934,
zero-crossings overshoot to 30 vs 22, and selection_score worsens to 13.815.
The 0-5 ms air slice is also far too hot (+27.97 dB), so this is best understood
as proof that targeted presence suppression plus crisp air can move the intended
metrics, not as the final musical topology.

The next listening-driven experiment made the noisy component itself steeply
high-passed instead of relying on final-mix EQ. `--crisp-noise-hp-stages` now
cascades the crisp path's high-pass biquad before the low-pass. Sweeps with
2-4 high-pass stages at 5.5-7.5 kHz, with broad air-sizzle and random-ish FM
disabled, confirm the subjective diagnosis: the 1-4 kHz dirt drops hard, but the
sound becomes too clicky/dark unless the broad air-sizzle is mixed back in. For
example, `pick_as0.25_camp2.8_low5500_st2` gets presence down to +7.66 dB, but
air is -2.92 dB, HF is -2.25 dB, and the 0-5 ms air slice is +27.79 dB. The
best less-extreme picked render,
`/tmp/dgen-train-kick808-steephp-picked/pick_as1.0_camp0.8_low7500_st4/learned.wav`,
keeps the transient slice reasonable (+5.91 dB), but it is effectively back near
the broad air-sizzle sound: presence +12.17 dB, air -2.17 dB, HF -1.06 dB,
selection_score 12.04. This suggests the missing component is not just "more
high-pass"; it needs a better exciter spectrum/envelope, likely a shorter or
shaped burst instead of the current deterministic table.

A deterministic burst exciter was added as `--crisp-noise-init burst`, with
`--crisp-noise-spacing` and `--crisp-noise-burst-decay`. It creates an alternating
decaying pulse train before the same steep high-pass stack. This behaves better
than raw table noise in the first 5 ms: low-air-sizzle burst candidates land
around +3 to +8 dB in the 0-5 ms air slice instead of +20 dB or more. However,
with broad air-sizzle reduced they are too dark in sustained 4-12 kHz; for
example `burst_as0.25_camp0.8_sp17_dec0.92` has presence +8.52 dB but air
-3.17 dB and HF -2.25 dB. Blending a small burst on top of the FM-best barely
moves the metrics or likely the sound; `blend_camp1.4_sp13` scores 7.1485 versus
the FM-best 7.1505, with essentially unchanged presence/air/HF deltas. The burst
is a better transient source, but it does not solve the sustained crisp-air gap
unless paired with a better air body than the current broad target-seeded table.

The most useful next branch is likely one of:

- Make the training loop track the external analysis metrics, then choose best
  checkpoints by a composite of normalized MSE, first-30 ms centroid, presence
  excess, air deficit, and time-sliced 5-30 ms high-frequency deficit. The current
  internal loss repeatedly selects checkpoints that look better numerically but
  sound darker.
- A learnable frequency curve improves early crossing count, but the current
  initialization is too front-loaded and gives worse MSE. It probably needs staged
  fitting: phase/frequency over the first 30 ms first, then body decay.
- A zero-crossing-derived frequency curve gets closer on crossing count than the
  guessed exponential curve, but it produces too many equal-amplitude early cycles
  and still misses the target phase/envelope. Curve-only staged training improves
  its scalar loss but does not materially improve the visual match.
- A short bank of 3-5 independently phased damped modes between roughly 250 and
  900 Hz was tried with tensorized phasors; it was slower than expected and did
  not improve the visual match.
- A deterministic noise/exciter component with a trainable decay and tone filter, if
  the loss emphasizes spectral/transient shape more than exact sample phase.
- A band-split sizzle topology: separate 1.5-4 kHz and 4-12 kHz trainable layers,
  or a differentiable filter that lets the trainer raise the air band without
  overfilling the presence band.
- A staged optimizer schedule: first fit phase/frequency/transient crossings with
  waveform-heavy loss over the first 30 ms, then fit body decay with spectral loss.
- A slope/first-difference loss is available as `--slope-weight`. It is useful for
  the visual shape but not sufficient by itself to recover the missing early cycles.
- A high-band loss is available as `--high-band-weight`; it compares smoothed
  high-pass energy envelopes. `--high-band-spectral-weight` adds a high-passed
  spectral FFT loss. The best high-frequency result so far still came from the
  earlier raw high-pass energy schedule, but these options are less unstable.
- A checkpoint/resume flow is now in the example. By default each run writes
  `checkpoint.json`, and `--checkpoint-in <json>` resumes matching scalar and
  tensor params.

## Learned Body-Waveshape Attempt

Added a phase-indexed learned oscillator table to the body path:
`--body-wave-points`, `--body-wave-mix-init`, and `--body-wave-lr-scale`. This
is intended to be more expressive than a sine oscillator without becoming raw
time-domain residual playback: the table is read by oscillator phase, not by
absolute sample time.

Two follow-up runs were tested:

| Attempt | Best external score | Improvement | Gates | Notes |
| --- | ---: | ---: | --- | --- |
| `bodywave1` with the accepted high-frequency support paths | 10.289998 | -79.58% | fail | MRSTFT moved slightly, but front air, waveform MSE, and transient MSE failed badly. |
| `bodywave2-bodyonly` with air paths removed | 12.426187 | -116.86% | fail | Training loss fell strongly, but the evaluator correctly penalized the missing high-frequency body and poor spectrum. |

Artifacts:

| Artifact | Path |
| --- | --- |
| Body-wave high-support run | `/tmp/dgen-train-kick808-bodywave1` |
| Body-wave high-support ranking | `/tmp/dgen-train-kick808-bodywave1-rank.txt` |
| Body-wave body-only run | `/tmp/dgen-train-kick808-bodywave2-bodyonly` |
| Body-wave body-only ranking | `/tmp/dgen-train-kick808-bodywave2-bodyonly-rank.txt` |

This topology is not a useful path to the 30% target as currently trained. The
high-support run confirms the existing failure mode: adding expressive body
motion while carrying the target-seeded air layer can make the first 5 ms too
bright and fails waveform gates. The body-only run confirms the opposite failure:
without a credible sustained air/noise layer, waveform-only training can look
busy while moving farther from the perceptual target.

Important reproducibility note: the packaged phase-warp best WAV still scores as
the best non-oracle artifact, but its checkpoint alone is not fully
self-describing after later topology/config changes. Re-rendering that checkpoint
without the exact original CLI/topology assumptions does not reproduce the
packaged score. Future successful artifacts should store or log the full render
configuration alongside `checkpoint.json`; otherwise continuation from the best
checkpoint is fragile.

## Checkpoint Config And Split-Band Follow-Ups

`checkpoint.json` now embeds the full `Config`, and render/resume automatically
loads it unless `--ignore-checkpoint-config` is passed. A smoke test confirmed
that render-only restores saved topology sizes such as `bodyWavePoints` and
`residualPoints`. This fixes the reproducibility problem for new checkpoints, but
older artifacts without embedded config still require the original CLI topology.

The split-band sizzle experiment separated the noisy top end into trainable
1.5-4 kHz and 4-12 kHz layers. The first version improved MRSTFT but failed the
front-air gate because the high-band table started too early. Added
`--sizzle-delay-ms`, `--sizzle-fade-ms`, `--air-sizzle-delay-ms`, and
`--air-sizzle-fade-ms`, then reran with a 5 ms delayed high band.

| Attempt | Best external score | Improvement | Gates | Notes |
| --- | ---: | ---: | --- | --- |
| `bandsplit1` | 7.491765 | -30.74% | fail | MRSTFT improved to 1.380865, but 0-5 ms air was +20.53 dB and waveform failed. |
| `bandsplit-delay5` | 4.948483 | 13.64% | fail | MRSTFT improved to 1.317213 and front-air passed, but waveform/transient MSE failed. |
| `bandsplit-delay5-shaperepair` | 4.948483 | 13.64% | fail | Shape repair did not beat the initial delayed checkpoint. |

This branch is the clearest evidence that the missing brightness is measurable:
MRSTFT and sustained-air metrics move in the right direction. The hard tradeoff is
that the current delayed table layer damages normalized waveform and transient
shape enough that it does not pass the anti-cheat gates.

## Phase-Warp Reset And Post-FIR Attempts

Added `--reset-phase-warp` so a resumed checkpoint can re-enable the phase-warp
body shaper instead of inheriting a zero `phaseWarpAmp`. A repair run from
`bandsplit-delay5` correctly trained the phase-warp tensor while freezing the
high-band layers, but it made the body/gate tradeoff worse: the best candidate was
still the unchanged epoch 0 checkpoint, and later checkpoints improved MRSTFT as
far as 1.202390 while increasing waveform/transient errors.

A learned post-FIR over the whole output was also tested from `bandsplit-delay5`
with `--eq-filter-size 63` while freezing the synth underneath. This behaved more
like unstable phase/comb coloration than useful EQ. The best external score again
remained the unchanged epoch 0 checkpoint; trained FIR checkpoints had worse
waveform MSE, transient MSE, or zero-crossing counts.

Finally, a high-band-only retune was attempted from the older packaged
phase-warp checkpoint. Because that checkpoint predates embedded config, the
manual render topology did not reproduce the packaged best at epoch 0. The sweep
still showed the same useful trend: increasing sustained high-frequency energy can
match `hf_delta` and `air_5_30_delta`, but it rapidly breaks zero crossings and
waveform gates. This reinforces that the target goal should be a gated composite,
not a single scalar that allows brightness to trade away the drum body.

Two upper-bound checks were run after these topology attempts:

| Check | Best score | Improvement | Gates | Interpretation |
| --- | ---: | ---: | --- | --- |
| Delayed high-passed target residual added to phase-warp best | 4.3074 | 24.8% | pass | Even an oracle-ish 30-16 kHz residual does not reach the 30% goal. Brightness alone is not enough. |
| Random ensembles of learned non-oracle candidates | 4.7617 | 16.9% | pass | Blending existing learned bodies/noise paths cannot escape the current ceiling. |

This makes the current blocker concrete: the objective needs a better
non-oracle body/transient topology, not another high-band EQ/sizzle retune. The
existing body family has a strong tradeoff: the split-band path improves MRSTFT
but fails waveform/transient gates, while the accepted phase-warp body passes
gates but cannot be pushed past roughly 20-25% even with generous high-band
assistance.

Current accepted non-oracle best remains:

| Artifact | Score | Improvement | Gates |
| --- | ---: | ---: | --- |
| `/tmp/dgen-train-kick808-phasewarp-best/learned.wav` | 4.854964 | 15.27% | pass |

## Compressed Residual And Correlated FM Follow-Ups

Two more topology branches were tried after the split-band/upper-bound analysis.

| Attempt | Best external score | Improvement | Gates | Notes |
| --- | ---: | ---: | --- | --- |
| `compressedres-gated1` | 10.292193 | -79.61% | fail | A short learned residual plus gated high paths raised high-frequency energy but badly damaged waveform shape and zero crossings. |
| `corrfm-nores1` | 10.164091 | -77.38% | fail | Removing residual playback and leaning on correlated FM/noise stayed non-oracle, but the body collapsed: later checkpoints improved raw MSE while losing cycle count and sustained air. |
| `modalbank1` | 9.148940 | -59.66% | fail | Four learned damped transient modes were active and eventually recovered the target zero-crossing count, but sustained air stayed low and waveform shape remained far outside the gate. |
| `chirpmode1` | 9.073211 | -58.34% | fail | Body-following chirped partial modes improved raw waveform/transient MSE versus fixed modes, but still lost sustained air and zero-crossing count. |
| `activesizzle1` | 9.360182 | -63.35% | fail | Delayed trainable sizzle/air-sizzle paths with nonzero amp were live, but still failed by losing cycle count and sustained air. |

The compressed-residual run also exposed a training trap: zero-initialized sizzle,
air-sizzle, and sub-body tables with zero amplitudes are dead starts. The tensor
gradient is zero until the amplitude or table is nonzero. Future runs that use
those paths should either use a nonzero amplitude, a nonzero deterministic
initializer, or stage them from a checkpoint where the path is already active.

Added `--transient-mode-*` controls for a small non-oracle modal transient bank:
four learned damped sine modes with learned amp/frequency/decay/phase, gated to
the front of the hit. The topology is parametric rather than table playback, but
the first run shows that extra low/mid modal freedom alone is not enough. It can
lower transient MSE, but without a simultaneously correct sustained air layer and
body envelope it remains perceptually worse than the phase-warp best.

Added `--transient-mode-follow-body` so the modal layer can track the falling body
phase as chirped partials rather than fixed phasors. This was a better failure
than the fixed modal bank, but still not a path to the 30% goal by itself:
`chirpmode1` reached normalized MSE 0.021310 and transient MSE 0.010043 at its
best externally ranked checkpoint, yet scored 9.073211 because sustained air
remained -8.31 dB low and zero crossings were still 16 vs target 22.

`activesizzle1` specifically tested the dead-path hypothesis from
`compressedres-gated1` by setting nonzero sizzle and air-sizzle amplitudes with
zero trainable tables. The tensors did receive gradients and learned nonzero
values. The best ranked checkpoint still scored 9.360182 with gates false:
normalized MSE was low at 0.007212, but zero crossings were 16 vs 22 and
sustained air remained -9.29 dB. This rules out "sizzle path never woke up" as
the main blocker.

I also tried to recover the exact old phase-warp render configuration from
temporary render logs so the accepted best checkpoint could become a
self-describing continuation base. The closest recovered render used
`residualFrames=4096`, `residualPoints=1024`, `fmNoisePoints=512`,
`crispNoisePoints=512`, `sizzlePoints=1025`, `airSizzlePoints=1024`,
`subBodyPoints=1024`, `eqFilterSize=17`, `crispNoiseLowHz=7000`, and
`crispNoiseDelayMs=5`. Rendering the old checkpoint with these flags improved the
reproduction over the default current topology, but still scored only 8.261756
with gates false, mainly from excessive front air and waveform/transient
regression. So the old checkpoint plus recovered flags is still not sufficient to
recreate the packaged phase-warp WAV. New successful checkpoints must remain
self-describing.

Finally, I audited every frozen-evaluator score JSON still present under
`/tmp/dgen-train-kick808*`. There are 193 scored artifacts. Only four pass all
gates:

| Artifact | Score | Improvement | Gates |
| --- | ---: | ---: | --- |
| `/tmp/dgen-train-kick808-phasewarp-best/perceptual-score.json` | 4.854964 | 15.27% | pass |
| `/tmp/dgen-train-kick808-airmode-shape-best/perceptual-score.json` | 4.862823 | 15.14% | pass |
| `/tmp/dgen-train-kick808-airmode-best/perceptual-score.json` | 4.934819 | 13.88% | pass |
| `/tmp/dgen-train-kick808-airmode-crisp/camp0.24_noise_clow7000_d5.0/perceptual-score.json` | 4.934819 | 13.88% | pass |

The nearest non-passing family is `bandsplit-delay5` at 4.948483 / 13.64%, but
it fails waveform/transient gates. This audit confirms the current search has not
found any non-oracle candidate near the required 30% gated improvement.

A quick offline EQ upper-bound check was run on the packaged phase-warp best WAV.
Random additive band EQ around the dry signal found a passing candidate at score
4.765778, a 16.83% improvement over the original baseline, with all gates true.
That is only a small improvement over the current 15.27% synth best and far below
the 30% goal. This makes the latest blocker sharper: the remaining gap is not a
simple brightness/EQ problem. The current best already sits near the useful limit
for post-spectral shaping unless the first 30 ms body/cycle structure changes.

## Runtime Notes

- Basic smooth family: roughly 12 ms per epoch at 4096 frames/window 512.
- Envelope/front family: roughly 48 ms per epoch with envelope tensors.
- Independent front mode branch currently compiles/runs around 296 ms per epoch,
  so it needs pruning or staging before long sweeps.
- Tensorized modal bank branch ran around 437 ms per epoch and is not worth
  continuing in its current form.
- Residual branch runs around 49 ms per epoch at 4096 frames/window 512, so a
  220-epoch fit takes roughly 11 seconds plus warmup/build time.
- Full residual runs around 50 ms per epoch at 4096 frames/window 512. The staged
  `fullres1` -> `fullres4` sequence took under a minute of training time and moved
  normalized MSE from 0.074191 to 0.012491.
- Slope-loss runs are slightly slower, around 52 ms per epoch in this example.
