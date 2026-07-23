# DGen generated-C header and symbol inventory

Status: Phase 1 inventory for the embedded native toolchain contract

Inventory commit: `c3357374f71b7f33297480cd89be26e3720dfdc0`

Host used for the binary survey: arm64 macOS 26.5.1, Apple Clang 17.0.0

Code generator: `Sources/DGen/Renderer/CRenderer.swift`

This document inventories the current C renderer, not the intended Phase 2
contract. In particular, today's renderer includes Apple SDK headers and emits
Accelerate calls. Phase 1 does not change that renderer.

The four checked-in source fixtures are:

| Kind | DGenLisp fixture | Generated C |
|---|---|---|
| Scalar synth | `toolchain/fixtures/scalar-synth.lisp` | `toolchain/generated/current/scalar-synth.c` |
| Feedback/delay effect | `toolchain/fixtures/feedback-delay-effect.lisp` | `toolchain/generated/current/feedback-delay-effect.c` |
| Wavetable instrument | `toolchain/fixtures/wavetable-instrument.lisp` | `toolchain/generated/current/wavetable-instrument.c` |
| FFT/spectral effect | `toolchain/fixtures/spectral-effect.lisp` | `toolchain/generated/current/spectral-effect.c` |

The spectral fixture uses partitioned convolution so the survey covers the
renderer's entire current vDSP call surface, not only FFT setup and execution.

## Headers emitted today

`CRenderer.render` unconditionally emits the following seven includes into
every generated kernel:

| Header | Provider today | What generated code actually uses |
|---|---|---|
| `<arm_neon.h>` | Clang resource headers | AArch64 NEON types and intrinsics |
| `<stdint.h>` | Clang wrapper plus SDK `include_next` in a hosted build | `uint32_t`, `uint32x4_t` |
| `<stdio.h>` | Apple SDK | Nothing; no generated stdio call exists |
| `<string.h>` | Apple SDK | Declarations/fortify wrappers for `memcpy` and `memset` |
| `<math.h>` | Apple SDK | Scalar math declarations, `isfinite`, `INFINITY`, `M_LOG10E` |
| `<Accelerate/Accelerate.h>` | Apple SDK framework headers | vForce vector math, vDSP types/constants/functions |
| `<mach/mach_time.h>` | Apple SDK | Nothing; no generated Mach timing call exists |

`stdio.h` and `mach/mach_time.h` are dead includes. The emitted comment about
`DGEN_PROFILE` has no corresponding conditional code or calls.

The two current compilation paths both add Accelerate independently of source
contents:

- `Sources/DGenLisp/Compiler.swift` passes `-framework Accelerate`.
- `Sources/DGen/Runtime.swift` does the same in both its cached argument
  template and fresh `compileAndLoad` path.

The checked-in Phase 1 proof replaces this entire include block in fixture
copies with `toolchain/include/phase1_compat.h`. That is explicitly a fixture
patch. It is not a C renderer modification and is not the Phase 2
`dgen_runtime.h`.

## External-call inventory

Names below are C source names. Their Mach-O undefined-symbol spellings gain a
leading underscore (for example, `sinf` becomes `_sinf`). Apple header macros
or fortified lowering can use the special spellings called out below.

### A. libm/libSystem

The C renderer can directly emit all of the following scalar math calls:

| Source name | Emission site/use | Notes |
|---|---|---|
| `sinf` | scalar `sin` | libSystem |
| `cosf` | scalar `cos` | libSystem |
| `tanf` | scalar `tan` | libSystem |
| `atanf` | scalar `atan` | libSystem |
| `atan2f` | scalar `atan2` | libSystem |
| `tanhf` | scalar `tanh` | libSystem |
| `expf` | scalar `exp` and constant-base `pow` | libSystem |
| `logf` | scalar `log` and constant-base `pow` | libSystem |
| `log10f` | scalar `log10` | libSystem |
| `sqrtf` | scalar `sqrt` and exponent `0.5` | Often instruction-selected |
| `powf` | nonspecialized scalar `pow` | libSystem |
| `fmodf` | scalar remainder with nonconstant divisor | libSystem |
| `fminf` | scalar minimum | Often instruction-selected |
| `fmaxf` | scalar maximum | Often instruction-selected |
| `floorf` | scalar floor and constant-divisor remainder | Often instruction-selected |
| `ceilf` | scalar ceil | Often instruction-selected |
| `roundf` | scalar round | Often instruction-selected |
| `copysignf` | scalar sign | Often instruction-selected |
| `fabs` | scalar absolute value | The renderer currently spells this `fabs`, not `fabsf` |

Every output passes through `isfinite(v)`. With the current Apple `math.h` and
the current release flags this produced the undefined Mach-O symbol
`___isfinitef` in the feedback, wavetable, and spectral samples. It optimized
away in the scalar synth. This is a header/lowering artifact rather than a
literal function call in generated source, but it is part of today's observed
link contract.

The optimization policy determines which of the source-level calls survive as
undefined symbols. The allowlist must therefore be derived from both the
renderer source surface above and post-link inspection; an optimized four-file
sample alone is insufficient.

### B. Accelerate

#### vForce-style vector math

The renderer can directly emit:

| Source names |
|---|
| `vsinf`, `vcosf`, `vtanf`, `vatanf`, `vatan2f`, `vtanhf` |
| `vexpf`, `vlogf`, `vpowf`, `vsqrtf` |

These are not ARM NEON intrinsics despite the leading `v`. They are external
vector-math entry points declared through Accelerate on the current build
path. The scalar synth actually leaves `_vsinf`, `_vcosf`, and `_vtanhf`
undefined in its dylib. This means that merely removing explicit FFTs is not
enough to remove Accelerate from today's generated C.

The Phase 1 compatibility header implements these calls as lane-wise wrappers
over the scalar libm allowlist. Phase 2 must decide the permanent lowering in
the DGen-owned runtime header without routing per-sample math through the host
service table.

#### vDSP

The complete current vDSP surface in `CRenderer` is:

| Source name | Generated use |
|---|---|
| `vDSP_create_fftsetup` | Lazily creates an FFT setup for each FFT size |
| `vDSP_fft_zip` | In-place forward and inverse split-complex FFT |
| `vDSP_zvma` | Partitioned-convolution split-complex multiply-accumulate |

The spectral fixture leaves all three corresponding Mach-O symbols undefined:
`_vDSP_create_fftsetup`, `_vDSP_fft_zip`, and `_vDSP_zvma`.

The generated source also consumes Accelerate-only types/constants:
`FFTSetup`, `DSPSplitComplex`, `kFFTRadix2`,
`kFFTDirection_Forward`, and `kFFTDirection_Inverse`.

### C. libc, compiler-inserted, Mach, stdio, and anything else

| Source/runtime symbol | Origin | Observed result |
|---|---|---|
| `memcpy` | Noise UOp moves xorshift state between `float` storage and `uint32_t` | Direct renderer surface; a 4-byte copy is normally optimized to loads/stores |
| `memset` | Partitioned spectral MAC clears real and imaginary output arrays | Present in spectral generated C |
| `___memset_chk` | Apple `string.h` fortify lowering of the spectral fixture's `memset` | Observed undefined symbol with today's system-Clang path |
| `memcpy`, `memset`, `bzero` | Potential LLVM optimization/code-generation lowering | Must remain in the Phase 2 audit candidate set even when absent from one optimized sample |
| `dyld_stub_binder` | Mach-O lazy-binding support introduced by the linker | Required in the minimal libSystem stub even though generated C never calls it directly |
| compiler-rt helpers | Target/operation-dependent Clang lowering | No dynamic compiler-rt helper was observed in these arm64 float-only fixtures; the prototype still bundles `libclang_rt.builtins.a` as required by the spec |
| Mach timing functions | None | `mach/mach_time.h` is included, but the renderer emits no `mach_*` call |
| stdio functions | None | `stdio.h` is included, but the renderer emits no `printf`, `fprintf`, `fputs`, or file call |
| allocation/locking/dlopen/process/filesystem/networking | None | No such call is emitted by `CRenderer` |

`___memset_chk` is the expected Phase 1 surprise: its appearance is controlled
by the Apple SDK header, not visible from a search for external calls in the
renderer. In the freestanding compatibility path, `memset` is declared
directly and the undefined symbol is `_memset`; the SDK-specific fortified
symbol disappears.

All other `v...` operations in generated source—such as `vaddq_f32`,
`vmulq_f32`, `vld1q_f32`, and `vst1q_f32`—are Clang NEON intrinsics that lower
to AArch64 instructions. They are not external symbols and do not belong in
the runtime allowlist.

## Per-fixture observed undefined symbols

These are the results of the current system-Clang command embedded in
`Sources/DGenLisp/Compiler.swift` (`-Ofast`, ThinLTO, fast-math, and
`-framework Accelerate`):

| Fixture | Undefined symbols after optimization |
|---|---|
| Scalar synth | `_vcosf`, `_vsinf`, `_vtanhf` |
| Feedback/delay effect | `___isfinitef` |
| Wavetable instrument | `___isfinitef` |
| FFT/spectral effect | `___isfinitef`, `___memset_chk`, `_vDSP_create_fftsetup`, `_vDSP_fft_zip`, `_vDSP_zvma` |

All four current-path dylibs contain both an Accelerate load command and a
`/usr/lib/libSystem.B.dylib` load command. Accelerate appears even for artifacts
with no surviving Accelerate symbol because the current compiler invocation
passes the framework unconditionally.

## Current exported surface

Today's generated C does not annotate ABI functions or internal globals for
visibility. Without `-fvisibility=hidden`, each surveyed dylib exports:

- `_process`
- `_setParamValue`
- `_VOICE_COUNT`
- `_SCRATCH_STRIDE`
- `_vfmodq_f32`
- every generated `t<id>_g` scratch array

This is why the Phase 1 run proof cannot simply add
`-fvisibility=hidden`: it would also hide `_process`. Phase 2 must mark the
versioned DGen entry points with default visibility and hide everything else
before enforcing the exact export allowlist.

## Reproduction commands

Generate the four current C files and current-path dylibs:

```sh
mkdir -p toolchain/generated/current
swift run DGenLisp compile toolchain/fixtures/scalar-synth.lisp \
  -o toolchain/generated/current --name scalar-synth \
  --sample-rate 48000 --max-frames 64
swift run DGenLisp compile toolchain/fixtures/feedback-delay-effect.lisp \
  -o toolchain/generated/current --name feedback-delay-effect \
  --sample-rate 48000 --max-frames 64
swift run DGenLisp compile toolchain/fixtures/wavetable-instrument.lisp \
  -o toolchain/generated/current --name wavetable-instrument \
  --sample-rate 48000 --max-frames 64
swift run DGenLisp compile toolchain/fixtures/spectral-effect.lisp \
  -o toolchain/generated/current --name spectral-effect \
  --sample-rate 48000 --max-frames 64
```

Inspect includes, undefined symbols, and load dependencies:

```sh
rg '^#include' toolchain/generated/current/*.c
for dylib in toolchain/generated/current/*.dylib; do
  echo "### ${dylib}"
  nm -u "${dylib}" | sort -u
  nm -gU "${dylib}" | awk '{print $NF}' | sort -u
  otool -L "${dylib}"
done
```

Audit the renderer itself so an unexercised math UOp cannot silently disappear
from this inventory:

```sh
rg -n '#include|memcpy|memset|vDSP_|mach_|printf|fprintf|fputs' \
  Sources/DGen/Renderer/CRenderer.swift
rg -n 'sinf|cosf|tanf|atanf|atan2f|tanhf|expf|logf|log10f|sqrtf|powf|fmodf|fminf|fmaxf|floorf|ceilf|roundf|copysignf|fabs' \
  Sources/DGen/Renderer/CRenderer.swift
```

The hermetic proof document records the corresponding staged-toolchain
undefined symbols and load commands after the fixture-only include patch.
