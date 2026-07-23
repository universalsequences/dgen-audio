# DGen four-lane vector-math lowering

Status: Phase 2 measured decision

Date: 2026-07-23

## Decision

`dgen_runtime.h` provides clean-room inline NEON polynomial implementations
for `vsinf`, `vcosf`, `vtanhf`, `vexpf`, and `vlogf`. They keep the existing
renderer spelling while removing the per-four-sample Accelerate dependency.
The sine and cosine implementations use a three-term Cody-Waite reduction:
the nearest integer multiple of `2π` is subtracted as `hi`, `mid`, and `lo`
float32 constants with three sequential `vfmsq_f32` operations. This keeps
range-reduction error below the unchanged polynomial approximation error for
float32 arguments through `|x| = 1e6`.

The polynomial candidates beat lane-wise scalar libm at every measured block
size for all five families. Accuracy met the following audio-oriented gates:

- sin/cos: maximum absolute error at most `1e-6` on `[-2π, 2π]`
- sin/cos large arguments: maximum absolute error below `1e-5` on
  `[-1e4, 1e4]`, measured against double-precision libm on the same float32
  arguments
- tanh: maximum absolute error at most `2e-7` on `[-8, 8]`
- exp: maximum error at most 4 ULP on `[-10, 10]`
- log: maximum absolute error at most `1e-6` and at most 4 ULP on
  `[0.0001, 100]`

The less frequent `tan`, `atan`, `atan2`, `pow`, and `sqrt` vector spellings
remain lane-wise scalar libm wrappers. They were not part of this five-family
polynomial experiment, and changing their numerical behavior without a
separate measurement would not be justified.

The implementation is DGen clean-room code using standard range reduction,
IEEE-754 decomposition, and polynomial evaluation. It does not copy or adapt
SLEEF or GPL code. This provenance is also recorded in
`toolchain/THIRD-PARTY-NOTICES.txt`.

## Speed

The table reports nanoseconds per sample. vecLib is the current system-path
baseline and is not a shippable lowering because it adds Accelerate to the
generated dylib. The decision is between the scalar-libm and polynomial
columns.

| Function | Frames | vecLib | Scalar libm | Polynomial |
|---|---:|---:|---:|---:|
| sin | 64 | 1.329 | 2.298 | 1.323 |
| sin | 256 | 1.314 | 2.488 | 1.313 |
| sin | 1024 | 1.300 | 2.269 | 1.303 |
| cos | 64 | 1.299 | 2.438 | 1.312 |
| cos | 256 | 1.300 | 2.469 | 1.320 |
| cos | 1024 | 1.327 | 2.380 | 1.315 |
| tanh | 64 | 1.309 | 2.495 | 1.302 |
| tanh | 256 | 1.321 | 2.486 | 1.398 |
| tanh | 1024 | 1.314 | 2.457 | 1.300 |
| exp | 64 | 1.307 | 1.871 | 1.321 |
| exp | 256 | 1.315 | 1.813 | 1.306 |
| exp | 1024 | 1.305 | 1.805 | 1.313 |
| log | 64 | 1.372 | 2.363 | 1.314 |
| log | 256 | 1.316 | 2.317 | 1.308 |
| log | 1024 | 1.307 | 2.304 | 1.318 |

Measurements were taken on an Apple M1 Max running arm64 macOS 26.5.1 with
Apple Clang 17.0.0. Each timing processes 64 million samples in the same
four-lane load/call/accumulate loop shape used by the scalar-synth hot path.

The benchmark also retains the old single-step reducer as a timing-only
control. The table below compares it with the shipped Cody-Waite reducer in
the same process and reports `(Cody-Waite / single-step) - 1`.

| Function | Frames | Single-step | Cody-Waite | Delta |
|---|---:|---:|---:|---:|
| sin | 64 | 1.324 | 1.323 | -0.08% |
| sin | 256 | 1.315 | 1.313 | -0.14% |
| sin | 1024 | 1.304 | 1.303 | -0.09% |
| cos | 64 | 1.302 | 1.312 | +0.77% |
| cos | 256 | 1.326 | 1.320 | -0.48% |
| cos | 1024 | 1.320 | 1.315 | -0.40% |

The largest observed timing change was `0.77%`, well below the `10%`
regression threshold. Sine and cosine remain faster than lane-wise scalar
libm at every measured block size, so the original lowering decision is
unchanged.

## Accuracy

Each result is from a dense sweep of 1,048,576 evenly spaced float inputs.
libm is the reference.

| Function and domain | Maximum absolute error | Maximum ULP |
|---|---:|---:|
| sin `[-2π, 2π]` | `1.78813934e-7` | `384489` |
| cos `[-2π, 2π]` | `6.21890649e-7` | `8823447` |
| tanh `[-8, 8]` | `1.19209290e-7` | `1786` |
| exp `[-10, 10]` | `0.001953125` | `3` |
| log `[0.0001, 100]` | `4.76837158e-7` | `2` |

The large trigonometric ULP counts occur at zero crossings: a tiny
absolute error can span many representable floats when the correctly rounded
answer is zero or subnormal. For bounded audio oscillators, the absolute error
gate is the meaningful measure. Conversely, exp's largest absolute error
occurs near `exp(10)` and is only 3 ULP, so ULP is the meaningful measure
there.

The large-argument sweep uses the same 1,048,576-point density on each
`[-M, M]` domain. Each generated argument is first rounded to float32; the
reference is then `sin((double)x)` or `cos((double)x)` for that exact float32
value. This separates range-reduction error from input quantization.

| Maximum magnitude `M` | Sine max absolute error | Cosine max absolute error |
|---:|---:|---:|
| `2π` | `1.61022613e-7` | `6.16287497e-7` |
| `1e2` | `2.06193519e-7` | `6.48548764e-7` |
| `1e3` | `2.07769990e-7` | `6.68251303e-7` |
| `1e4` | `2.11966145e-7` | `6.52504377e-7` |
| `1e5` | `2.12712069e-7` | `6.60417729e-7` |
| `1e6` | `2.13917424e-7` | `6.66199041e-7` |

The `1e4` gate passes by more than an order of magnitude, and error remains
flat through the guaranteed `|x| <= 1e6` range. Beyond this point, the spacing
between adjacent float32 arguments increasingly dominates phase accuracy;
the Cody-Waite constants are no longer the limiting factor.

## Hermetic fixture proof

After restaging the updated runtime header, `scripts/prove-toolchain.sh`
passed its compile, link, ABI audit, load, nonfinite-containment, and audio
comparison checks. Compared with the Phase 2 report immediately before this
change, fixture maximum-error deltas were:

| Fixture | Before | Cody-Waite | Delta |
|---|---:|---:|---:|
| scalar synth | `9.68575478e-08` | `0` | `-9.68575478e-08` |
| wavetable instrument | `1.49011612e-08` | `1.49011612e-08` | `0` |
| spectral effect | `0` | `0` | `0` |
| feedback delay effect | `0` | `0` | `0` |

The scalar-synth system-Clang and embedded-Clang artifacts are now
bit-identical across the 8,192-sample proof. All fixtures remain below the
`2e-5` tolerance; spectral and feedback remain bit-identical.

## Reproduction

From the repository root:

```sh
scripts/benchmark-vector-math.sh
cat .toolchain/vector-math/vector-math-results.csv
```

The script builds the checked-in benchmark with the same `-O3 -ffast-math`
numerical policy used for generated artifacts, links Accelerate only into the
benchmark so it can measure the legacy baseline, and writes raw CSV results
under `.toolchain/vector-math`.
