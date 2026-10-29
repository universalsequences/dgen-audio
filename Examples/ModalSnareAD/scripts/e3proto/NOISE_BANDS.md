# Latent Snare noise filterbank: the spec the trainer and the instrument share

Status: implemented in `Latent Snare/dsp.lisp` (realtime) and `reference_render.py`
(numpy). It is **identical to E1's current `e1/noise_bands/spec.json`**, as produced by
`scripts/e1/make_noise_bands.py`: same fc, Q, stages and normalisation. Measured differences are
0 for fc and b0, and 2e-13 for norm. E1 therefore does not need to regenerate anything.
If either side changes this file, both must change.

## Decision

The instrument uses **16 band-pass filters on one live white-noise source**. The trainer
uses **fixed precomputed band-noise signals made with the same filters**. Trained
envelopes transfer for three reasons:

- The realtime band signal and the training band signal are the same random process: the
  same filter, a unit-RMS normalisation, and a zero-mean unit-variance white input.
- The envelope multiplies the *filtered* noise, `env_b(t) * N_b(t)`, in both places. The
  noise is never filtered after the envelope.
- Because each band is unit-RMS, `A_b` and `C_b` are absolute band RMS levels. They do
  not depend on filter bandwidth or on sample rate.

The realtime cost is 32 biquads (2 per band) per sample, run as `[1,16]` tensor work
inside the 19 us/block voice. This cost was not measured separately.

## Filter

Index i = 0..15:

```
fc_i  = 100 * 180^(i/15)  Hz                  (100 Hz .. 18 kHz, 0.4995 oct spacing)
Q     = 1.8   (each stage)
stages = 2    (two identical biquads in cascade, 4th order per band)
w0    = 2 pi fc_i / sr
alpha = sin(w0) / (2 Q)
b0 =  alpha / (1 + alpha),  b1 = 0,  b2 = -b0          (RBJ cookbook BPF, constant 0 dB peak gain)
a1 = -2 cos(w0) / (1 + alpha),  a2 = (1 - alpha) / (1 + alpha)
```

Each stage is Direct Form I with zero initial state:

```
y[n] = b0 (x[n] - x[n-2]) - a1 y[n-1] - a2 y[n-2]
stage 1: x = white,   stage 2: x = stage-1 output
```

Normalisation makes each band unit RMS for unit-variance white input:

```
N_b[n] = norm_b(sr) * stage2_b[n]
norm_b(sr) = 1 / sqrt( sum_n h_b[n]^2 )   h_b = impulse response of the 2-stage cascade at sr
```

`norm` depends on the sample rate. The realtime instrument carries exact tables for 44.1,
48, 88.2 and 96 kHz and picks the nearest one. A `sqrt(sr/48k)` approximation is off by
1.45 dB in the 18 kHz band at 44.1 kHz and by 4.1 dB at 96 kHz, so the instrument does not
use it. Adjacent bands cross at -2.9 dB.

## Noise source

```
white[n] = sqrt(12) * (u[n] - 0.5),   u = dgen (noise): xorshift32 uniform [0,1), seed 1 on a fresh voice
```

The trainer may use Gaussian unit-variance noise, as E1 does. After a Q=1.8 four-pole
band-pass the two sources are statistically equivalent: the per-band RMS is 0.96-1.09 for
the realtime stream, measured over 1.75 s. Only the variance must match, so **do not feed
U[0,1) un-centred**. The realtime filterbank runs freely from voice start. A first hit at
voice start therefore sees the cascade's zero-state start-up: band RMS over the first
10 ms is 0.6-1.3. Precompute the training bands with a pre-roll (E1 uses 8192 samples) so
that they are stationary from t = 0. That is the right target because hits after the
first, and voices that started earlier, see stationary bands.

## Envelope applied per band (dsp.lisp and reference_render.py)

```
t      = max(0, T - noise_onset)                 T = seconds since the hit
env_b  = (1 - e^{-t/noise_rise}) * A_b * ((1-L_b) e^{-t/d1_b} + L_b e^{-t/d2_b})  +  C_b * E(t)
E(t)   = sqrt( sum_k m_env_k(t)^2 + 1e-10 )      head amplitude-envelope RMS (wire coupling)
noise  = sum_b env_b(t) * N_b(t)
```

This is E1's `SnareRenderer` noise model. The decay knob scales d1 and d2.

## Coefficients at 48 kHz (and the 44.1 kHz norm)

| i | fc Hz | b0 | a1 | a2 | norm @48k | norm @44.1k |
|---|---|---|---|---|---|---|
| 0 | 100.0000 | 0.003622826 | -1.992583624 | 0.992754348 | 23.453409 | 22.480519 |
| 1 | 141.3681 | 0.005113712 | -1.989431900 | 0.989772575 | 19.726010 | 18.907811 |
| 2 | 199.8495 | 0.007213490 | -1.984893637 | 0.985573019 | 16.591341 | 15.903284 |
| 3 | 282.5235 | 0.010166090 | -1.978314192 | 0.979667820 | 13.955379 | 13.376840 |
| 4 | 399.3981 | 0.014308213 | -1.968689983 | 0.971383573 | 11.739168 | 11.252845 |
| 5 | 564.6216 | 0.020099269 | -1.954451181 | 0.959801463 | 9.876513 | 9.467922 |
| 6 | 798.1950 | 0.028154756 | -1.933090697 | 0.943690487 | 8.312083 | 7.969154 |
| 7 | 1128.3933 | 0.039274931 | -1.900527924 | 0.921450139 | 6.999904 | 6.712673 |
| 8 | 1595.1885 | 0.054446938 | -1.850028189 | 0.891106125 | 5.902227 | 5.662614 |
| 9 | 2255.0880 | 0.074769415 | -1.770422779 | 0.850461170 | 4.988792 | 4.790488 |
| 10 | 3187.9757 | 0.101190527 | -1.643355412 | 0.797618945 | 4.236565 | 4.075065 |
| 11 | 4506.7815 | 0.133846696 | -1.439505401 | 0.732306608 | 3.630229 | 3.503142 |
| 12 | 6371.1526 | 0.170624022 | -1.114568577 | 0.658751957 | 3.164362 | 3.072502 |
| 13 | 9006.7790 | 0.204282507 | -0.607710878 | 0.591434986 | 2.851249 | 2.802829 |
| 14 | 12732.7146 | 0.216608587 | 0.150043167 | 0.566782826 | 2.754874 | 2.787259 |
| 15 | 18000.0000 | 0.164172104 | 1.182039147 | 0.671655793 | 3.234869 | 3.664784 |

The `norm` columns are the values in `decoder_layout.json` → `noise_bands.norm`, which
also lists 88.2 and 96 kHz. They are computed by `gen_dsp.py:band_norm` from a
65536-sample impulse response, and match E1's `bands.json` `norm` values.
