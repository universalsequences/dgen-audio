# Staged DGen toolchain layout

This file is the contract between the DGen toolchain distribution and the hosts
that embed it (today: ESeq). `scripts/build-toolchain.sh` produces a staging
prefix and packs it into
`.toolchain/dgen-toolchain-<llvm-version>-<arch>.tar.gz`. Every path below is
relative to the staged root, and every path is stable: hosts may depend on
them, and moving one is a distribution-version change.

Untarring the archive yields a single top-level directory, `dgen-toolchain/`,
which *is* the staged root:

```
tar -xzf dgen-toolchain-20.1.8-arm64.tar.gz -C <dest>
<dest>/dgen-toolchain/    # pass this to --toolchain-root
```

## Targets

A stage is target-specific by construction and carries its target in
`VERSION.json`. Two are published:

| Target | Object format | ISA | Linker | Archive |
| --- | --- | --- | --- | --- |
| `arm64-apple-macos` | Mach-O | AArch64 | `bin/ld64.lld` | `dgen-toolchain-<llvm-version>-arm64.tar.gz` |
| `x86_64-unknown-linux-gnu` | ELF | X86 | `bin/ld.lld` | `dgen-toolchain-<llvm-version>-x86_64.tar.gz` |

Both are built by the same script from the same pinned LLVM source archive,
with the same LLVM component selection: one backend, one lld driver, no
assertions, no tests, no docs. Neither is repacked from a vendor prebuilt.

## Locked principle

Nothing in the stage may reference the DGen source tree, the build tree, an
Xcode/Command Line Tools path, or any other machine-specific location. The
staged root is relocatable: it works from any directory, on any machine, with
an empty environment.

This holds for **every** target, with no exemption. It is why the stages are
built from LLVM source rather than repacked from a vendor's prebuilt LLVM
release: those binaries carry their own release CI's build paths
(`/home/runner/work/llvm-project/...`, `/opt/rh/devtoolset-7`) baked into
assertion and diagnostic strings. Such strings are inert — they never reach a
produced artifact — but "inert" is a claim that has to be re-argued for every
new prebuilt, and the point of a locked principle is that it is checked
instead.

`scripts/build-toolchain.sh` enforces it on the two staged executables of
whichever target it just built:

* a `strings` scan for the repository root, the build root, and the vendor and
  CI path prefixes above;
* a load-dependency scan — `otool -L` on Mach-O, `readelf -d` on ELF. The ELF
  side additionally rejects any `DT_RPATH`/`DT_RUNPATH` outright: a stage needs
  no rpath (every LLVM library is linked statically), so any rpath at all is a
  build-tree reference.

`scripts/audit-dgen-dylib.sh` and `scripts/audit-dgen-elf-so.sh` enforce the
equivalent property for the artifacts the toolchain *produces*.

## Layout

### Every stage

| Path | What it is | Consumed by |
| --- | --- | --- |
| `VERSION.json` | Distribution identity: distribution/ABI/codegen-policy versions, LLVM version and source digest, target triple, the platform floor (`minimum_macos` on Darwin, `glibc_floor` on Linux), and SHA-256 of the staged `clang`, `lld`, and runtime headers. | Host cache keys; `DGenToolchainPolicy.stagedVersionDigest`. Its SHA-256 is the compiler fingerprint — hosts must never shell out to a system compiler to obtain one. |
| `SIZE.txt` | Installed footprint report (LLVM source archive size, installed staging prefix size). The compressed-archive size and digest cannot live inside the archive; they are appended to the repository copy — `toolchain/SIZE.txt` for arm64, `toolchain/SIZE-<target>.txt` for every other target. | Packaging and release notes. |
| `LAYOUT.md` | This document, so the distribution is self-describing. | Humans. |
| `bin/dgen-clang` | Pinned upstream Clang driver, single-backend, stripped. The only compiler on the production path. | `DGenToolchainPolicy.embeddedInvocation`. |
| `lib/clang/20/include/` | Clang's own resource headers (freestanding subset; no SDK headers). Passed as `-resource-dir` plus `-isystem`. | `DGenToolchainPolicy.embeddedInvocation`. |
| `include/dgen_runtime.h` | The frozen ABI v1 header: `DGenProcessContextV1`, `DGenHostServicesV1`, the `dgen_process_v1` / `dgen_set_param_value_v1` prototypes, and the inline four-lane vector math. Generated C includes this and nothing else. | The compile itself (`-I<stage>/include`); hosts also read it to pin their vendored copy of the ABI structs. |
| `LICENSES/LLVM-LICENSE.txt` | Upstream LLVM license, as shipped in the LLVM source archive. | Redistribution. |
| `LICENSES/THIRD-PARTY-NOTICES.txt` | DGen's third-party notices for the staged components. | Redistribution. |

### `arm64-apple-macos` only

| Path | What it is | Consumed by |
| --- | --- | --- |
| `bin/ld64.lld` | Pinned upstream `ld64.lld`, Mach-O driver only (see `toolchain/patches/lld-macho-only.patch`), stripped. Selected with `-fuse-ld=`. | `DGenToolchainPolicy.embeddedInvocation`. |
| `lib/clang/20/lib/darwin/libclang_rt.builtins.a` | compiler-rt builtins for arm64, linked explicitly because the link is `-nostdlib`. | `DGenToolchainPolicy.embeddedInvocation`. |
| `lib/libSystem.tbd` | DGen-authored text stub for `/usr/lib/libSystem.B.dylib`, generated by `scripts/generate-libsystem-stub.sh`. Nothing is copied out of an Apple SDK. | `-L<stage>/lib -lSystem`. |
| `abi/exports-v1.txt` | Exact set of symbols an audited artifact may export. | Binary audit (`scripts/audit-dgen-dylib.sh`, and host-side reimplementations). |
| `abi/libsystem-symbols-v1.txt` | Allowlist of undefined symbols an audited artifact may reference — all resolved by libSystem. | Binary audit. |
| `empty-sdk/` | Deliberately empty directory passed as `-isysroot`, so no system SDK can leak into a compile. Created by the build script and re-created by the policy if absent. | `DGenToolchainPolicy.embeddedInvocation`. |

### `x86_64-unknown-linux-gnu` only

| Path | What it is | Consumed by |
| --- | --- | --- |
| `bin/ld.lld` | Pinned upstream `ld.lld`, ELF driver only (see `toolchain/patches/lld-elf-only.patch`), stripped. Selected with `-fuse-ld=`. | `DGenToolchainPolicy.embeddedInvocation`. |
| `lib/clang/20/lib/x86_64-unknown-linux-gnu/libclang_rt.builtins.a` | compiler-rt builtins for x86_64, linked explicitly because the link is `-nostdlib`. | `DGenToolchainPolicy.embeddedInvocation`. |
| `include/dgen_simd_compat.h` | The NEON-intrinsic shim `dgen_runtime.h` includes on non-ARM hosts. Part of the ABI header set, so it is covered by `runtime_headers_sha256`. | The compile itself, via `dgen_runtime.h`. |
| `abi/exports-v1-elf.txt` | Exact set of symbols an audited artifact may export. | Binary audit (`scripts/audit-dgen-elf-so.sh`, and host-side reimplementations). |
| `abi/libsystem-symbols-v1-elf.txt` | Allowlist of undefined symbols an audited artifact may leave for the loader. | Binary audit. |

There is no ELF counterpart to `lib/libSystem.tbd` or `empty-sdk/`: an ELF
`-shared -nostdlib` link resolves nothing against a stub, and `-nostdinc` plus
the pinned `-resource-dir` already close the door `-isysroot` closes on Darwin.

The two `abi/` files travel with the toolchain on purpose: the toolchain that
produces a binary and the contract that binary is audited against must move
together, so a toolchain update cannot leave a host auditing against a stale
allowlist.

## Building a stage

`scripts/build-toolchain.sh` picks its target from `uname`; there is no
cross-compilation path. On Darwin/arm64 it stages `arm64-apple-macos`, on
Linux/x86_64 `x86_64-unknown-linux-gnu`.

A **published** Linux archive must additionally be built inside the pinned
container image:

```sh
scripts/build-toolchain-linux-container.sh
```

The stage's glibc floor is a property of the distribution — it is fetched onto
machines the builder does not control — so it is chosen rather than inherited
from the build host. The wrapper builds inside `ubuntu:22.04` (glibc 2.35, the
same floor the published Linux DGenLisp distribution targets) and then calls
`scripts/build-toolchain.sh` unchanged. The floor actually achieved is read
back off the staged binaries into `VERSION.json`, so that field is evidence,
not an assertion. A native Linux run produces a correct stage; it is simply not
publishable, because its floor is the build host's.

## Using the stage

```sh
DGenLisp compile patch.lisp --toolchain-root <dest>/dgen-toolchain -o out --name patch
```

The `--toolchain-root` flag is the host-selected root and takes precedence over
the `DGEN_TOOLCHAIN_STAGE_ROOT` environment variable, which remains as a
development fallback. When a root is selected, `DGenToolchainPolicy`
preflight-checks `bin/dgen-clang`, the target's linker and builtins archive,
`include/dgen_runtime.h`, `VERSION.json`, and — on Darwin only —
`lib/libSystem.tbd`; an incomplete root is a hard error and never falls back to
a system compiler.

Artifacts produced through the Mach-O stage carry an `@rpath`-relative
`LC_ID_DYLIB` install name, depend on `/usr/lib/libSystem.B.dylib` alone, and
export exactly the symbols in `abi/exports-v1.txt`. Artifacts produced through
the ELF stage carry a bare `DT_SONAME`, leave undefined only the symbols in
`abi/libsystem-symbols-v1-elf.txt` for the loader to bind at `dlopen`, and
export exactly the symbols in `abi/exports-v1-elf.txt`.
