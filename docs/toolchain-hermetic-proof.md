# Embedded DGen toolchain proofs

## Phase 2 closed-ABI proof

Status: complete

Authoritative run: 2026-07-23 on arm64 macOS 26.5.1

Phase 2 replaces the fixture-only header patch with the production
`dgen_runtime.h` contract and versioned `dgen_process_v1` ABI. The renderer
routes setup, forward FFT, inverse FFT, and partitioned complex
multiply-accumulate through `DGenHostServicesV1`. Generated dylibs contain no
Accelerate calls or load command; the reference host harness links Accelerate
and implements those four block-level callbacks.

Run the complete proof from the repository root:

```sh
scripts/prove-toolchain.sh
cat .toolchain/proof/proof-report.txt
```

Candidate C compilation and linking run under `env -i` with an unreachable
`PATH`, empty `DEVELOPER_DIR` and `SDKROOT`, an empty controlled sysroot,
`-nostdinc`, `-nostdlib`, explicit staged resource paths, the generated
minimal libSystem stub, and the staged compiler-rt builtins archive. The
system compiler is used only to build the comparison dylibs and the host
harness; it is not reachable during candidate compilation.

Every baseline and candidate link is audited by
`scripts/audit-dgen-dylib.sh`. The audit requires:

- an arm64 Mach-O dynamic library with macOS 11.0 deployment target;
- exactly `_dgen_process_v1` and `_dgen_set_param_value_v1` exported;
- every undefined symbol in `toolchain/abi/libsystem-symbols-v1.txt`;
- exactly `/usr/lib/libSystem.B.dylib` as a load dependency;
- an `@rpath`-relative `LC_ID_DYLIB` install name, so no build-machine staging
  path is baked into a published artifact;
- no `LC_RPATH` and no developer, workspace, user, or temporary path.

The staged distribution that hosts consume — its directory layout, which files
are guaranteed at which relative paths, and how a host selects it with
`DGenLisp --toolchain-root` — is specified in `toolchain/LAYOUT.md`.

The accepted optimization policy is explicit `-O3 -ffast-math` plus the
spelled-out DSP/vectorization flags in `DGenToolchainPolicy`. The runtime
header uses IEEE-754 exponent bits behind an optimizer barrier for finite
classification. The proof compiles a fixture containing actual NaN and
infinity bit patterns under those same flags and verifies that every output
is replaced by zero.

The authoritative run produced:

| Fixture | Duration/samples | Maximum absolute error |
|---|---:|---:|
| Scalar synth | 8,192 samples | `9.68575478e-08` |
| Feedback/delay effect | 5 seconds / 240,000 samples | `0` |
| Wavetable instrument | 8,192 samples | `1.49011612e-08` |
| FFT/spectral effect | 8,192 samples | `0` |

The tolerance is `2e-5`. It is deliberately much larger than the observed
compiler-codegen differences but small enough to catch audible algorithmic
drift. The feedback fixture runs for five seconds so its delay state wraps
and small recursive differences have time to diverge. The spectral fixture
uses the Accelerate-backed reference host table and exercises all four host
services.

The vector-math measurement and accuracy decision is separately reproducible
as documented in `docs/vector-math-lowering.md`.

## Phase 1 historical proof

Status: Phase 1 prototype

Scope: dgen-audio ownership only; no ESeq bundling, signing, cache, or hot swap

Target: arm64-apple-macos 11.0 or newer

## Outcome

The prototype builds upstream LLVM/Clang/LLD from a checksum-pinned source
release, stages only the DGen-required executables/resources, and compiles real
current DGen-generated C with no Apple SDK headers. The strict-stub variant
produces arm64 dylibs whose only load dependency is
`/usr/lib/libSystem.B.dylib`. The independent `dynamic_lookup` variant uses no
stub and has no load dependency; all of its runtime symbols remain undefined
for dyld lookup.

The checked-in harness loads both the current system-Clang artifact and the
staged-toolchain artifact, processes eight 64-frame blocks, and compares all
512 output samples. The authoritative staged-toolchain run completed on
2026-07-23. Its full report is in `.toolchain/proof/proof-report.txt`; the
command below recreates it.

The non-FFT scalar synth, feedback/delay effect, and wavetable instrument enter
the link/load/run proof. The current partitioned-convolution fixture compiles
hermetically to an arm64 object but is intentionally not linked: its direct
vDSP calls are the Phase 2 host-service-table work identified by the accepted
spec. No C generator change was made.

## Pinned toolchain

`scripts/build-toolchain.sh` pins:

- LLVM project release: 20.1.8
- Source archive SHA-256:
  `6898f963c8e938981e6c4a302e83ec5beb4630147c7311183cf61069af16333d`
- Projects: Clang and LLD
- Runtime: compiler-rt builtins only
- LLVM code-generation target: AArch64 only
- Deployment target: macOS 11.0

Upstream's default `lld` dispatcher links all five LLD ports into one
executable. `toolchain/patches/lld-macho-only.patch` is therefore applied to
the checksum-verified upstream source: it retains only the Mach-O dispatcher
and `lldMachO` library, and the build creates only the `ld64.lld` driver
alias. This is a narrow, reviewable source patch; it imports no Apple code.

Tests, examples, benchmarks, documentation, static analysis, debugger support,
unused LLVM tools and targets, sanitizers, profiling, XRay, libFuzzer, ORC
runtime support, zlib, zstd, libxml2, libedit, terminfo, curl, and httplib are
disabled. The script uses supported LLVM configuration and symbol stripping;
it does not prune an install by trial-and-error deletion.

The staged layout is:

```text
.toolchain/stage/
  VERSION.json
  bin/
    dgen-clang
    ld64.lld
  include/
    phase1_compat.h
  lib/
    libSystem.tbd
    clang/20/include/...
    clang/20/lib/darwin/libclang_rt.builtins.a
  LICENSES/
    LLVM-LICENSE.txt
    THIRD-PARTY-NOTICES.txt
```

`phase1_compat.h` is explicitly a prototype fixture header, not the Phase 2
`dgen_runtime.h`.

Build and package from the repository root:

```sh
DGEN_TOOLCHAIN_JOBS=5 scripts/build-toolchain.sh
cat toolchain/VERSION.json
cat toolchain/SIZE.txt
```

The work, download, stage, archive, metadata, size-report, and job-count paths
are overridable with the `DGEN_TOOLCHAIN_*` variables declared at the top of
the script. The default compressed archive is
`.toolchain/dgen-toolchain-20.1.8-arm64.tar.gz`. Archive entries are sorted and
normalized to uid/gid zero and mtime zero.

The measured output of the authoritative build is:

| Item | Bytes | Approximate MiB |
|---|---:|---:|
| Verified LLVM source archive | 147,242,952 | 140.4 |
| Installed staging prefix | 146,874,368 | 140.1 |
| Compressed toolchain archive | 46,976,294 | 44.8 |

The compressed archive SHA-256 is
`84721a4882abaa67708797cd01542ca22dfbe11a550eb2adfd62f5060681b5ee`.
`toolchain/SIZE.txt` is the checked-in size report. `toolchain/VERSION.json`
records the required version fields and these staged hashes:

- `dgen-clang`:
  `2ac73943c36157dbd8e4d8b730d73971b2cd6914555ab2c8b4324b41bd5b4f5e`
- `ld64.lld`:
  `acf75170895c5cad5b3c0172642163bb66d92af81d808bb6f916c45525173d5d`
- Phase 1 runtime headers:
  `4327d00a8fa00287a123328dc74837f35f7183f0f9106970db97bdd4d48177f7`

The build uses the installed development compiler only to build upstream LLVM.
No Xcode binary, library, header, or SDK stub enters the staged distribution.
The build fails if staged tool strings or load commands expose the repository,
build directory, Xcode, CommandLineTools, or `/usr/bin/clang`.

The compressed archive was also extracted under a different random prefix and
the full proof passed there, including all six load/run comparisons:

```sh
relocation_root=$(mktemp -d "$PWD/.toolchain/relocated.XXXXXX")
tar -xzf .toolchain/dgen-toolchain-20.1.8-arm64.tar.gz \
  -C "$relocation_root"
DGEN_TOOLCHAIN_STAGE_ROOT="$relocation_root/dgen-toolchain" \
  DGEN_TOOLCHAIN_PROOF_ROOT="$relocation_root/proof" \
  scripts/prove-toolchain.sh
/usr/bin/trash "$relocation_root"
```

## Current generated C and Phase 1 patch

The current renderer unconditionally emits seven SDK-facing includes. The
four original generated files are checked in under
`toolchain/generated/current`. This command makes fixture-only hermetic copies:

```sh
scripts/prepare-toolchain-fixtures.sh
```

The transformation removes only lines beginning with `#include <...>` and
prepends:

```c
#include "phase1_compat.h"
```

The compatibility header provides exact scalar declarations, NEON types from
Clang's builtin resource directory, lane-wise versions of today's Accelerate
vForce calls, and declarations that allow the current spectral code to compile
without Accelerate headers. It does not implement vDSP or change generated
operations. See `docs/toolchain-symbol-inventory.md` for the full measured
surface.

## Hermetic compile

Run the full proof:

```sh
scripts/prove-toolchain.sh
```

For every hermetic compiler and linker invocation, that script uses an
environment equivalent to:

```sh
env -i \
  PATH=/dgen-no-system-tools \
  DEVELOPER_DIR= \
  SDKROOT= \
  TMPDIR="$PWD/.toolchain/proof" \
  LC_ALL=C \
  "$PWD/.toolchain/stage/bin/dgen-clang" ...
```

The command uses absolute staged paths for `dgen-clang` and `ld64.lld`,
`-nostdinc`, an explicit staged Clang resource directory, the Phase 1
compatibility include directory, `-nostdlib`, and the DGen-authored
staged `libSystem.tbd`. It supplies `.toolchain/proof/empty-sdk`, a deliberately
empty controlled directory, as the sysroot. `/usr/bin/clang`, `xcrun`, and
CommandLineTools are not reachable through `PATH`, and `DEVELOPER_DIR`/`SDKROOT`
cannot supply a developer tree.

This is a "Command Line Tools not visible" acceptance run, implemented with a
clean environment and unreachable `PATH`; it is not a claim that the host's
developer tools were physically uninstalled. The system-Clang reference and
the harness are not inputs to the isolated candidate compile/link commands.

The compile flags deliberately mirror the current DGen system path for the
comparison:

```text
-target arm64-apple-macos11.0
-Ofast
-mcpu=apple-m1
-flto=thin
-ffast-math
-fno-math-errno
-fno-trapping-math
-ffp-contract=fast
-fvectorize
-fslp-vectorize
-funroll-loops
-fPIC
```

The proof also adds the contract flags `-ffreestanding`,
`-fno-stack-protector`, `-fno-asynchronous-unwind-tables`, and the no-SDK
include policy. It does not yet add `-fvisibility=hidden`: today's generated
entry points lack explicit default-visibility attributes, so that flag would
make `process` unresolvable. Adding the attributes and an exact export audit is
Phase 2 C-contract work. `-Ofast` is retained here only so the prototype
comparison does not silently change the current path. The release policy
should spell the equivalent intended semantics as `-O3 -ffast-math`, as
required by the accepted spec.

## Link contract decision

Standardize Phase 2 on the minimal DGen-owned `libSystem.tbd`, while retaining
the mandatory post-link undefined-symbol audit.

Why:

- With `-undefined error`, the stub rejects a misspelled or newly introduced
  symbol at link time and reports its name in the compiler diagnostic.
- With `-undefined dynamic_lookup`, the exact same deliberate typo links
  successfully. Failure moves to binary audit or, if that audit has a bug,
  `dlopen`/first use.
- Both mechanisms still need the post-link allowlist because the stub is a
  build input that can drift and because compiler/linker-introduced symbols
  must be measured from the final Mach-O.
- Maintaining the stub is not additional conceptual policy: its exports are
  generated from the same versioned allowlist the audit already requires.

The proof makes this observable rather than rhetorical:

```sh
cat .toolchain/proof/logs/stub-typo.stderr
nm -u .toolchain/proof/dynamic-lookup/link-contract-typo.dylib
```

`toolchain/harness/link-contract-typo.c` references
`dgen_deliberate_link_typo`. The strict-stub link must fail; the
`dynamic_lookup` link must succeed. The proof script treats the opposite result
as failure.

The strict real-artifact variants explicitly link the DGen stub as `-lSystem`,
so their load commands name the real runtime install name. The dynamic-lookup
variants consume neither that stub nor an SDK and have no load dependency:

```sh
for dylib in .toolchain/proof/stub/*.dylib \
             .toolchain/proof/dynamic-lookup/*.dylib; do
  otool -L "$dylib"
  nm -u "$dylib" | sort -u
done
```

The script requires exactly `/usr/lib/libSystem.B.dylib` for strict-stub
dylibs and rejects every dependency in the dynamic-lookup dylibs. It also
rejects embedded Xcode, CommandLineTools, `/usr/bin/clang`, workspace, or user
paths. Each dylib uses an `@rpath/<fixture>.dylib` install name; no absolute
staging path is recorded and no `LC_RPATH` is added.

After optimization, both modes leave the same runtime surface:

| Fixture | Undefined symbols |
|---|---|
| Scalar synth | `_cosf`, `_floorf`, `_sinf`, `_tanhf`, `dyld_stub_binder` |
| Feedback/delay effect | `_floorf`, `dyld_stub_binder` |
| Wavetable instrument | `_floorf`, `_fmaxf`, `_fminf`, `dyld_stub_binder` |

## Load/run comparison

`toolchain/harness/toolchain_harness.c`:

1. `dlopen`s the current system-Clang dylib and the candidate dylib with
   `RTLD_NOW | RTLD_LOCAL`.
2. Resolves both current DGen entry points, `process` and `setParamValue`;
   `process` drives audio while today's no-op setter is checked for presence.
3. Initializes fixture parameters, tensor data, and independent state buffers.
4. Processes eight deterministic blocks at 48 kHz.
5. Compares all samples and exits nonzero if maximum absolute error exceeds
   `2e-5`.

Run an individual comparison:

```sh
.toolchain/proof/toolchain-harness \
  feedback-delay-effect \
  .toolchain/proof/baseline/feedback-delay-effect.dylib \
  .toolchain/proof/stub/feedback-delay-effect.dylib \
  8 64 2e-5
```

The authoritative upstream-Clang build and load run measured:

| Fixture | Compared samples | Maximum absolute error |
|---|---:|---:|
| Scalar synth | 512 | `2.98023224e-08` |
| Feedback/delay effect | 512 | `0` |
| Wavetable instrument | 512 | `1.3038516e-08` |

The small nonzero scalar/wavetable differences come from the fixture-only
lane-wise libm wrapper and optimizer/code-generation choices; they are far
below the declared tolerance. Flags were not changed to force equality.
The same values were observed for both strict-stub and dynamic-lookup links;
the feedback/delay output was bit-identical.

## Surprises and Phase 2 inputs

1. Passing `-framework Accelerate` unconditionally gives even non-FFT current
   dylibs an Accelerate load command.
2. The renderer's SIMD math names (`vsinf`, `vcosf`, and peers) are external
   Accelerate vForce entry points, not NEON intrinsics. Non-FFT does not
   automatically mean no Accelerate.
3. Apple `string.h` fortified the spectral fixture's visible `memset` into
   `___memset_chk`. The no-SDK header produces the ordinary `_memset` contract.
4. The minimal stub must export `dyld_stub_binder`, an initial undefined
   introduced by Mach-O lazy binding rather than generated C.
5. ThinLTO/optimization can introduce, remove, or rewrite `memcpy`, `memset`,
   `bzero`, and compiler-rt calls. The bundled builtins archive plus final
   undefined-symbol audit are both necessary.
6. `isfinite` on an already-cast integer appears in some current generated
   bounds expressions. The Phase 1 generic macro must accept integer and
   floating inputs.
7. Fast-math causes Clang to warn that NaN/Inf classification is inconsistent
   with the selected numerical model. Phase 2 must make the sanitizer and
   numerical-semantics policy agree; this prototype preserves current flags
   and records the warning.
8. Today's ABI entry points have no explicit visibility annotations.
   `-fvisibility=hidden` cannot be enabled until Phase 2 marks the allowed
   exports.

This work stops at the repository boundary in the accepted spec. It does not
implement `DGenHostServicesV1`, change the C generator, bundle tools into
ESeq, sign code, manage caches, or hot-swap audio graph state.
