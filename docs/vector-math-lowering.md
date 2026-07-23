# DGen four-lane vector-math lowering

Status: Phase 2 measured decision

Date: 2026-07-23

## Decision

`dgen_runtime.h` provides clean-room inline NEON polynomial implementations
for `vsinf`, `vcosf`, `vtanhf`, `vexpf`, and `vlogf`. They keep the existing
renderer spelling while removing the per-four-sample Accelerate dependency.

The polynomial candidates beat lane-wise scalar libm at every measured block
size for all five families. Accuracy met the following audio-oriented gates:

- sin/cos: maximum absolute error at most `1e-6` on `[-2π, 2π]`
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
| sin | 64 | 1.773 | 4.412 | 2.783 |
| sin | 256 | 2.302 | 3.887 | 2.694 |
| sin | 1024 | 2.537 | 3.952 | 1.898 |
| cos | 64 | 2.115 | 4.428 | 2.003 |
| cos | 256 | 1.920 | 6.776 | 2.328 |
| cos | 1024 | 2.077 | 4.290 | 2.486 |
| tanh | 64 | 2.075 | 4.684 | 2.512 |
| tanh | 256 | 2.348 | 4.040 | 2.722 |
| tanh | 1024 | 2.099 | 4.202 | 2.598 |
| exp | 64 | 2.427 | 3.354 | 2.259 |
| exp | 256 | 2.173 | 3.067 | 2.441 |
| exp | 1024 | 2.022 | 3.002 | 2.471 |
| log | 64 | 2.613 | 4.443 | 2.279 |
| log | 256 | 2.069 | 4.521 | 2.405 |
| log | 1024 | 3.189 | 4.028 | 2.208 |

Measurements were taken on an Apple M1 Max running arm64 macOS 26.5.1 with
Apple Clang 17.0.0. Each timing processes 64 million samples in the same
four-lane load/call/accumulate loop shape used by the scalar-synth hot path.

## Accuracy

Each result is from a dense sweep of 1,048,576 evenly spaced float inputs.
libm is the reference.

| Function and domain | Maximum absolute error | Maximum ULP |
|---|---:|---:|
| sin `[-2π, 2π]` | `1.78813934e-7` | `876330287` |
| cos `[-2π, 2π]` | `7.15459464e-7` | `6801247` |
| tanh `[-8, 8]` | `1.19209290e-7` | `1786` |
| exp `[-10, 10]` | `0.001953125` | `3` |
| log `[0.0001, 100]` | `4.76837158e-7` | `2` |

The large trigonometric ULP counts occur at zero crossings: a tiny
absolute error can span many representable floats when the correctly rounded
answer is zero or subnormal. For bounded audio oscillators, the absolute error
gate is the meaningful measure. Conversely, exp's largest absolute error
occurs near `exp(10)` and is only 3 ULP, so ULP is the meaningful measure
there.

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
