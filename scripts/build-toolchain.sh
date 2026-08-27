#!/bin/sh
set -eu

# Reproducible upstream-only DGen toolchain build.
# No executable or library is copied from a system/vendor toolchain into the
# staged distribution: every staged binary is built here from the pinned LLVM
# source archive.
#
# The build is host-native. The staged target is derived from `uname` -- there
# is no cross-compilation path, because a stage is only ever consumed on the
# machine class it was built for. Supported hosts:
#
#   Darwin/arm64  -> arm64-apple-macos          (Mach-O, AArch64, ld64.lld)
#   Linux/x86_64  -> x86_64-unknown-linux-gnu   (ELF, X86, ld.lld)
#
# A published Linux archive must be built inside the pinned container image so
# its glibc floor is a distribution property rather than an accident of the
# build host; scripts/build-toolchain-linux-container.sh does that and calls
# straight back into this script.
#
# --repack: skip the LLVM download/build/stage entirely and regenerate
# VERSION.json, SIZE.txt, and the compressed archive from the existing staged
# prefix. Use after metadata-only changes (e.g. bumping
# toolchain/COMPILER_VERSION) when the binaries are unchanged.

repack=0
if [ "${1:-}" = "--repack" ]; then
  repack=1
  shift
fi
# A stray argument must not be ignored here: the default path is a multi-hour
# from-source build, so a mistyped --repack has to fail in the first second
# rather than the last.
if [ "$#" -gt 0 ]; then
  echo "error: unexpected argument: $1" >&2
  echo "usage: $(basename "$0") [--repack]" >&2
  exit 2
fi

llvm_version=20.1.8
llvm_archive="llvm-project-${llvm_version}.src.tar.xz"
llvm_url="https://github.com/llvm/llvm-project/releases/download/llvmorg-${llvm_version}/${llvm_archive}"
llvm_sha256=6898f963c8e938981e6c4a302e83ec5beb4630147c7311183cf61069af16333d
llvm_major=${llvm_version%%.*}

# ── Host target ──
# Everything that varies with the object format or ISA is decided once, here,
# and every later step reads these variables. The Mach-O values are the
# original script's constants verbatim: the mac stage's contents and digests
# must not move because a Linux target was added beside it.
host_os=$(uname -s)
host_arch=$(uname -m)
minimum_macos=
case "${host_os}:${host_arch}" in
  Darwin:arm64)
    host_target=arm64-apple-macos
    arch_tag=arm64
    llvm_target=AArch64
    lld_binary=ld64.lld
    lld_patch_name=lld-macho-only.patch
    lld_patch_guard=DGEN_LLD_MACHO_DRIVER
    staged_builtins_relative="lib/clang/${llvm_major}/lib/darwin/libclang_rt.builtins.a"
    minimum_macos=11.0
    default_triple="${host_target}${minimum_macos}"
    ;;
  Linux:x86_64)
    host_target=x86_64-unknown-linux-gnu
    arch_tag=x86_64
    llvm_target=X86
    lld_binary=ld.lld
    lld_patch_name=lld-elf-only.patch
    lld_patch_guard=DGEN_LLD_ELF_DRIVER
    staged_builtins_relative="lib/clang/${llvm_major}/lib/${host_target}/libclang_rt.builtins.a"
    default_triple="${host_target}"
    ;;
  *)
    echo "error: no DGen toolchain build is defined for ${host_os}/${host_arch}" >&2
    echo "Supported hosts are Darwin/arm64 and Linux/x86_64." >&2
    exit 1
    ;;
esac

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
work_root=${DGEN_TOOLCHAIN_WORK_ROOT:-"${repo_root}/.toolchain-build"}
download_root=${DGEN_TOOLCHAIN_DOWNLOAD_ROOT:-"${repo_root}/.toolchain-downloads"}
stage_root=${DGEN_TOOLCHAIN_STAGE_ROOT:-"${repo_root}/.toolchain/stage"}
archive_output=${DGEN_TOOLCHAIN_ARCHIVE:-"${repo_root}/.toolchain/dgen-toolchain-${llvm_version}-${arch_tag}.tar.gz"}

# The mac stage's repository-side records keep their historical unqualified
# names so the checked-in arm64 record and the docs that quote it stay put;
# every other target is qualified by its triple.
case "${host_target}" in
  arm64-apple-macos)
    default_metadata_output="${repo_root}/toolchain/VERSION.json"
    default_size_report="${repo_root}/toolchain/SIZE.txt"
    ;;
  *)
    default_metadata_output="${repo_root}/toolchain/VERSION-${host_target}.json"
    default_size_report="${repo_root}/toolchain/SIZE-${host_target}.txt"
    ;;
esac
metadata_output=${DGEN_TOOLCHAIN_METADATA_OUTPUT:-"${default_metadata_output}"}
size_report=${DGEN_TOOLCHAIN_SIZE_REPORT:-"${default_size_report}"}

jobs=${DGEN_TOOLCHAIN_JOBS:-$(sysctl -n hw.logicalcpu 2>/dev/null || nproc 2>/dev/null || echo 4)}
# Linking clang and lld against static LLVM libraries peaks in the low
# gigabytes per link. Compilation parallelism is bounded by cores, but link
# parallelism is bounded by RAM, and an OOM-killed link late in a multi-hour
# build is the expensive failure. Two concurrent links is the safe default;
# raise it on a machine with memory to spare.
link_jobs=${DGEN_TOOLCHAIN_LINK_JOBS:-2}

source_root="${work_root}/llvm-project-${llvm_version}.src"
build_root="${work_root}/build"
archive_path="${download_root}/${llvm_archive}"
lld_patch="${repo_root}/toolchain/patches/${lld_patch_name}"

require_tool() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "error: required tool not found: $1" >&2
    exit 1
  }
}

# macOS ships `shasum`, Linux ships `sha256sum`; both print the digest first.
if command -v sha256sum >/dev/null 2>&1; then
  sha256_of() { sha256sum "$1" | awk '{print $1}'; }
elif command -v shasum >/dev/null 2>&1; then
  sha256_of() { shasum -a 256 "$1" | awk '{print $1}'; }
else
  echo "error: required tool not found: sha256sum or shasum" >&2
  exit 1
fi

# `stat -f` is BSD, `stat -c` is GNU.
case "${host_os}" in
  Darwin) file_size() { stat -f %z "$1"; } ;;
  *) file_size() { stat -c %s "$1"; } ;;
esac

require_tool python3
require_tool tar
if [ "${repack}" -eq 0 ]; then
  require_tool cmake
  require_tool curl
  require_tool patch
  require_tool strings
  case "${host_os}" in
    Darwin) require_tool otool ;;
    *) require_tool readelf ;;
  esac
fi

case "${source_root}:${build_root}:${stage_root}" in
  *":/:"*|"/:"*|*":/") echo "error: refusing a root-directory build or stage path" >&2; exit 1 ;;
esac

mkdir -p \
  "${download_root}" \
  "${work_root}" \
  "$(dirname "${stage_root}")" \
  "$(dirname "${archive_output}")" \
  "$(dirname "${metadata_output}")" \
  "$(dirname "${size_report}")"

if [ "${repack}" -eq 1 ]; then
  repack_required="${stage_root}/bin/dgen-clang ${stage_root}/bin/${lld_binary}"
  if [ "${host_target}" = "arm64-apple-macos" ]; then
    repack_required="${repack_required} ${stage_root}/lib/libSystem.tbd"
  fi
  for staged_file in ${repack_required}; do
    if [ ! -f "${staged_file}" ]; then
      echo "error: --repack requires an existing staged prefix; missing: ${staged_file}" >&2
      exit 1
    fi
  done
  if [ ! -f "${archive_path}" ]; then
    echo "error: --repack needs the LLVM source download for SIZE.txt bookkeeping: ${archive_path}" >&2
    exit 1
  fi
fi

if [ "${repack}" -eq 0 ]; then

if [ "${host_target}" = "arm64-apple-macos" ]; then
  "${repo_root}/scripts/generate-libsystem-stub.sh"
fi

if [ ! -f "${archive_path}" ]; then
  curl -fL --retry 3 -o "${archive_path}.partial" "${llvm_url}"
  mv "${archive_path}.partial" "${archive_path}"
fi

actual_archive_sha=$(sha256_of "${archive_path}")
if [ "${actual_archive_sha}" != "${llvm_sha256}" ]; then
  echo "error: LLVM archive checksum mismatch" >&2
  echo "expected: ${llvm_sha256}" >&2
  echo "actual:   ${actual_archive_sha}" >&2
  exit 1
fi

if [ ! -f "${source_root}/llvm/CMakeLists.txt" ]; then
  rm -rf "${source_root}"
  tar -xf "${archive_path}" -C "${work_root}"
fi

# Upstream's default lld dispatcher links the COFF, ELF, MinGW, Mach-O, and
# WebAssembly ports into one executable. The embedded DGen toolchain needs
# exactly one of them, so apply the checked-in, reviewable narrowing patch for
# this host's object format. This changes the driver dispatcher and linked
# libraries; it imports no vendor code.
if ! grep -q "${lld_patch_guard}" "${source_root}/lld/tools/lld/lld.cpp"; then
  patch -d "${source_root}" -p1 < "${lld_patch}"
fi

generator="Unix Makefiles"
if command -v ninja >/dev/null 2>&1; then
  generator=Ninja
fi

prefix_map="-ffile-prefix-map=${work_root}=/usr/src/dgen-llvm-${llvm_version} -fdebug-prefix-map=${work_root}=/usr/src/dgen-llvm-${llvm_version} -fmacro-prefix-map=${work_root}=/usr/src/dgen-llvm-${llvm_version}"

case "${host_target}" in
arm64-apple-macos)

cmake -S "${source_root}/llvm" -B "${build_root}" -G "${generator}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET="${minimum_macos}" \
  -DCMAKE_C_FLAGS_RELEASE="-O2 -DNDEBUG -g0 ${prefix_map}" \
  -DCMAKE_CXX_FLAGS_RELEASE="-O2 -DNDEBUG -g0 ${prefix_map}" \
  -DLLVM_ENABLE_PROJECTS="clang;lld" \
  -DLLVM_ENABLE_RUNTIMES=compiler-rt \
  -DLLVM_TARGETS_TO_BUILD="${llvm_target}" \
  -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD= \
  -DLLVM_DEFAULT_TARGET_TRIPLE="${default_triple}" \
  -DLLVM_TARGET_ARCH="${llvm_target}" \
  -DLLVM_APPEND_VC_REV=OFF \
  -DLLVM_ENABLE_ASSERTIONS=OFF \
  -DLLVM_ENABLE_BACKTRACES=OFF \
  -DLLVM_ENABLE_BINDINGS=OFF \
  -DLLVM_ENABLE_CRASH_OVERRIDES=OFF \
  -DLLVM_ENABLE_CURL=OFF \
  -DLLVM_ENABLE_HTTPLIB=OFF \
  -DLLVM_ENABLE_LIBEDIT=OFF \
  -DLLVM_ENABLE_LIBXML2=OFF \
  -DLLVM_ENABLE_OCAMLDOC=OFF \
  -DLLVM_ENABLE_TERMINFO=OFF \
  -DLLVM_ENABLE_ZLIB=OFF \
  -DLLVM_ENABLE_ZSTD=OFF \
  -DLLVM_INCLUDE_BENCHMARKS=OFF \
  -DLLVM_INCLUDE_DOCS=OFF \
  -DLLVM_INCLUDE_EXAMPLES=OFF \
  -DLLVM_INCLUDE_TESTS=OFF \
  -DLLVM_INCLUDE_UTILS=OFF \
  -DLLVM_BUILD_BENCHMARKS=OFF \
  -DLLVM_BUILD_DOCS=OFF \
  -DLLVM_BUILD_EXAMPLES=OFF \
  -DLLVM_BUILD_TESTS=OFF \
  -DLLVM_BUILD_UTILS=OFF \
  -DLLVM_BUILD_TOOLS=OFF \
  -DLLVM_INSTALL_TOOLCHAIN_ONLY=ON \
  -DLLVM_ENABLE_IDE=OFF \
  -DLLVM_ENABLE_PIC=OFF \
  -DCLANG_BUILD_EXAMPLES=OFF \
  -DCLANG_ENABLE_ARCMT=OFF \
  -DCLANG_ENABLE_BOOTSTRAP=OFF \
  -DCLANG_ENABLE_STATIC_ANALYZER=OFF \
  -DCLANG_INCLUDE_DOCS=OFF \
  -DCLANG_INCLUDE_TESTS=OFF \
  -DCLANG_PLUGIN_SUPPORT=OFF \
  -DLLD_BUILD_DOCS=OFF \
  -DLLD_INCLUDE_TESTS=OFF \
  -DLLD_SYMLINKS_TO_CREATE="${lld_binary}" \
  -DCOMPILER_RT_BUILD_BUILTINS=ON \
  -DCOMPILER_RT_BUILD_CRT=OFF \
  -DCOMPILER_RT_BUILD_GWP_ASAN=OFF \
  -DCOMPILER_RT_BUILD_LIBFUZZER=OFF \
  -DCOMPILER_RT_BUILD_MEMPROF=OFF \
  -DCOMPILER_RT_BUILD_ORC=OFF \
  -DCOMPILER_RT_BUILD_PROFILE=OFF \
  -DCOMPILER_RT_BUILD_SANITIZERS=OFF \
  -DCOMPILER_RT_BUILD_XRAY=OFF \
  -DCOMPILER_RT_DEFAULT_TARGET_ONLY=ON \
  -DCOMPILER_RT_ENABLE_IOS=OFF \
  -DCOMPILER_RT_ENABLE_TVOS=OFF \
  -DCOMPILER_RT_ENABLE_WATCHOS=OFF \
  -DCOMPILER_RT_ENABLE_XROS=OFF \
  -DDARWIN_osx_ARCHS=arm64 \
  -DDARWIN_osx_BUILTIN_ARCHS=arm64

# Darwin builds compiler-rt builtins once per (arch, os) slice, so the umbrella
# `builtins` target would build slices this stage never ships. Configure the
# runtimes sub-build, then build exactly the arm64/macOS slice.
cmake --build "${build_root}" \
  --target clang lld clang-resource-headers builtins-configure \
  --parallel "${jobs}"
cmake --build "${build_root}/runtimes/builtins-bins" \
  --target clang_rt.builtins_arm64_osx \
  --parallel "${jobs}"
;;

x86_64-unknown-linux-gnu)

# Deltas from the Mach-O configuration above, all of them consequences of the
# object format or of what a Linux host has to be relocatable against:
#   * LLVM_STATIC_LINK_CXX_STDLIB -- a published stage must run on a machine
#     whose libstdc++ is older than the build container's. Static-linking
#     libstdc++/libgcc leaves libc/libm as the only ABI surface, and the glibc
#     floor recorded in VERSION.json is then the whole compatibility story.
#     Darwin needs no equivalent: libc++ is part of the stable OS ABI there.
#   * CMAKE_SKIP_RPATH -- CMake would otherwise bake the build tree's lib
#     directory into DT_RUNPATH, which is exactly the machine-specific
#     reference LAYOUT.md's locked principle forbids. Nothing in the stage
#     needs an rpath: every LLVM library is linked statically.
#   * No Darwin slice/SDK variables, and no libSystem stub.
cmake -S "${source_root}/llvm" -B "${build_root}" -G "${generator}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_FLAGS_RELEASE="-O2 -DNDEBUG -g0 ${prefix_map}" \
  -DCMAKE_CXX_FLAGS_RELEASE="-O2 -DNDEBUG -g0 ${prefix_map}" \
  -DCMAKE_SKIP_RPATH=ON \
  -DLLVM_ENABLE_PROJECTS="clang;lld" \
  -DLLVM_ENABLE_RUNTIMES=compiler-rt \
  -DLLVM_TARGETS_TO_BUILD="${llvm_target}" \
  -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD= \
  -DLLVM_DEFAULT_TARGET_TRIPLE="${default_triple}" \
  -DLLVM_TARGET_ARCH="${llvm_target}" \
  -DLLVM_APPEND_VC_REV=OFF \
  -DLLVM_ENABLE_ASSERTIONS=OFF \
  -DLLVM_ENABLE_BACKTRACES=OFF \
  -DLLVM_ENABLE_BINDINGS=OFF \
  -DLLVM_ENABLE_CRASH_OVERRIDES=OFF \
  -DLLVM_ENABLE_CURL=OFF \
  -DLLVM_ENABLE_HTTPLIB=OFF \
  -DLLVM_ENABLE_LIBEDIT=OFF \
  -DLLVM_ENABLE_LIBPFM=OFF \
  -DLLVM_ENABLE_LIBXML2=OFF \
  -DLLVM_ENABLE_OCAMLDOC=OFF \
  -DLLVM_ENABLE_TERMINFO=OFF \
  -DLLVM_ENABLE_Z3_SOLVER=OFF \
  -DLLVM_ENABLE_ZLIB=OFF \
  -DLLVM_ENABLE_ZSTD=OFF \
  -DLLVM_STATIC_LINK_CXX_STDLIB=ON \
  -DLLVM_PARALLEL_LINK_JOBS="${link_jobs}" \
  -DLLVM_INCLUDE_BENCHMARKS=OFF \
  -DLLVM_INCLUDE_DOCS=OFF \
  -DLLVM_INCLUDE_EXAMPLES=OFF \
  -DLLVM_INCLUDE_TESTS=OFF \
  -DLLVM_INCLUDE_UTILS=OFF \
  -DLLVM_BUILD_BENCHMARKS=OFF \
  -DLLVM_BUILD_DOCS=OFF \
  -DLLVM_BUILD_EXAMPLES=OFF \
  -DLLVM_BUILD_TESTS=OFF \
  -DLLVM_BUILD_UTILS=OFF \
  -DLLVM_BUILD_TOOLS=OFF \
  -DLLVM_INSTALL_TOOLCHAIN_ONLY=ON \
  -DLLVM_ENABLE_IDE=OFF \
  -DLLVM_ENABLE_PIC=OFF \
  -DCLANG_BUILD_EXAMPLES=OFF \
  -DCLANG_ENABLE_ARCMT=OFF \
  -DCLANG_ENABLE_BOOTSTRAP=OFF \
  -DCLANG_ENABLE_STATIC_ANALYZER=OFF \
  -DCLANG_INCLUDE_DOCS=OFF \
  -DCLANG_INCLUDE_TESTS=OFF \
  -DCLANG_PLUGIN_SUPPORT=OFF \
  -DLLD_BUILD_DOCS=OFF \
  -DLLD_INCLUDE_TESTS=OFF \
  -DLLD_SYMLINKS_TO_CREATE="${lld_binary}" \
  -DCOMPILER_RT_BUILD_BUILTINS=ON \
  -DCOMPILER_RT_BUILD_CRT=OFF \
  -DCOMPILER_RT_BUILD_GWP_ASAN=OFF \
  -DCOMPILER_RT_BUILD_LIBFUZZER=OFF \
  -DCOMPILER_RT_BUILD_MEMPROF=OFF \
  -DCOMPILER_RT_BUILD_ORC=OFF \
  -DCOMPILER_RT_BUILD_PROFILE=OFF \
  -DCOMPILER_RT_BUILD_SANITIZERS=OFF \
  -DCOMPILER_RT_BUILD_XRAY=OFF \
  -DCOMPILER_RT_DEFAULT_TARGET_ONLY=ON

# ELF compiler-rt has exactly one builtins slice, so the umbrella target is the
# one this stage ships; no per-slice second pass is needed.
cmake --build "${build_root}" \
  --target clang lld clang-resource-headers builtins \
  --parallel "${jobs}"
;;
esac

rm -rf "${stage_root}"
mkdir -p \
  "${stage_root}/abi" \
  "${stage_root}/bin" \
  "${stage_root}/include" \
  "${stage_root}/lib/clang/${llvm_major}/include" \
  "$(dirname "${stage_root}/${staged_builtins_relative}")" \
  "${stage_root}/LICENSES"

cp "${build_root}/bin/clang" "${stage_root}/bin/dgen-clang"
cp "${build_root}/bin/${lld_binary}" "${stage_root}/bin/${lld_binary}"
cp -R "${build_root}/lib/clang/${llvm_major}/include/." \
  "${stage_root}/lib/clang/${llvm_major}/include/"
cp "${repo_root}/toolchain/LAYOUT.md" "${stage_root}/LAYOUT.md"
cp "${source_root}/llvm/LICENSE.TXT" "${stage_root}/LICENSES/LLVM-LICENSE.txt"
cp "${repo_root}/toolchain/THIRD-PARTY-NOTICES.txt" \
  "${stage_root}/LICENSES/THIRD-PARTY-NOTICES.txt"

# The binary-audit contract travels with the toolchain that produces the
# binaries, so a toolchain update moves both atomically (toolchain/LAYOUT.md).
# Each object format has its own pair; a stage carries only its own.
case "${host_target}" in
  arm64-apple-macos)
    mkdir -p "${stage_root}/empty-sdk"
    cp "${repo_root}/toolchain/include/dgen_runtime.h" "${stage_root}/include/"
    cp "${repo_root}/toolchain/lib/libSystem.tbd" "${stage_root}/lib/"
    cp "${repo_root}/toolchain/abi/exports-v1.txt" "${stage_root}/abi/"
    cp "${repo_root}/toolchain/abi/libsystem-symbols-v1.txt" "${stage_root}/abi/"
    builtins_archive=$(find "${build_root}/runtimes/builtins-bins" -type f \
      -name 'libclang_rt.builtins_arm64_osx.a' -print | head -n 1)
    ;;
  x86_64-unknown-linux-gnu)
    # dgen_runtime.h includes dgen_simd_compat.h on non-ARM hosts, so the shim
    # is part of the ABI header set the stage must carry.
    cp "${repo_root}/toolchain/include/dgen_runtime.h" "${stage_root}/include/"
    cp "${repo_root}/toolchain/include/dgen_simd_compat.h" "${stage_root}/include/"
    cp "${repo_root}/toolchain/abi/exports-v1-elf.txt" "${stage_root}/abi/"
    cp "${repo_root}/toolchain/abi/libsystem-symbols-v1-elf.txt" "${stage_root}/abi/"
    # COMPILER_RT_DEFAULT_TARGET_ONLY leaves exactly one builtins slice, but
    # it is written twice -- once in the runtimes sub-build, once staged into
    # the resource directory. Sort so the same one is picked every run.
    builtins_archive=$(find "${build_root}" -type f \
      -name 'libclang_rt.builtins*.a' -print | LC_ALL=C sort | head -n 1)
    ;;
esac

if [ -z "${builtins_archive}" ]; then
  echo "error: compiler-rt builtins archive was not produced" >&2
  exit 1
fi
cp "${builtins_archive}" "${stage_root}/${staged_builtins_relative}"

# Symbol stripping is a supported packaging operation. It changes neither the
# selected components nor their behavior.
if command -v strip >/dev/null 2>&1; then
  case "${host_os}" in
    Darwin)
      strip -S "${stage_root}/bin/dgen-clang"
      strip -S "${stage_root}/bin/${lld_binary}"
      ;;
    *)
      strip --strip-unneeded "${stage_root}/bin/dgen-clang"
      strip --strip-unneeded "${stage_root}/bin/${lld_binary}"
      ;;
  esac
fi

# ── Locked principle: nothing in the stage may name a build or developer path ──
# Enforced identically on every host; only the patterns and the
# dependency-listing tool are object-format specific.
case "${host_os}" in
  Darwin) forbidden_paths="${repo_root}|${work_root}|/Applications/Xcode|/Library/Developer/CommandLineTools|/usr/bin/clang" ;;
  # `/home/runner/` and `/opt/rh/` are what vendor-prebuilt LLVM bakes in from
  # its release CI; scanning for them is what makes "built from source here"
  # a checked property rather than a claim. `/usr/lib/gcc` is deliberately NOT
  # forbidden: clang's Linux driver carries it as a legitimate GCC-detection
  # search prefix, not as a reference to this machine.
  *) forbidden_paths="${repo_root}|${work_root}|/home/runner/|/opt/rh/|/usr/bin/clang" ;;
esac

for staged_binary in "${stage_root}/bin/dgen-clang" "${stage_root}/bin/${lld_binary}"; do
  if strings "${staged_binary}" | grep -E "${forbidden_paths}" >/dev/null; then
    echo "error: forbidden build/developer path embedded in ${staged_binary}" >&2
    exit 1
  fi
  case "${host_os}" in
    Darwin)
      # Skip otool's first line: it echoes the inspected file's staging path.
      if otool -L "${staged_binary}" | tail -n +2 |
        grep -E "${forbidden_paths}" >/dev/null; then
        echo "error: forbidden load dependency in ${staged_binary}" >&2
        exit 1
      fi
      ;;
    *)
      # DT_NEEDED names must stay bare sonames, and DT_RPATH/DT_RUNPATH must be
      # absent outright: either one is a machine-specific library reference.
      if readelf -d "${staged_binary}" |
        grep -E 'RPATH|RUNPATH' >/dev/null; then
        echo "error: staged binary carries an rpath: ${staged_binary}" >&2
        readelf -d "${staged_binary}" | grep -E 'RPATH|RUNPATH' >&2
        exit 1
      fi
      if readelf -d "${staged_binary}" | sed -n 's/.*NEEDED.*\[\(.*\)\]/\1/p' |
        grep '/' >/dev/null; then
        echo "error: forbidden load dependency in ${staged_binary}" >&2
        exit 1
      fi
      ;;
  esac
done

fi  # end of the full-build path skipped by --repack

clang_sha=$(sha256_of "${stage_root}/bin/dgen-clang")
lld_sha=$(sha256_of "${stage_root}/bin/${lld_binary}")
runtime_headers_sha=$(
  cd "${stage_root}/include"
  find . -type f -print | LC_ALL=C sort |
    while IFS= read -r header; do
      header_sha=$(sha256_of "${header}")
      printf '%s  %s\n' "${header_sha}" "${header}"
    done |
    sha256_of /dev/stdin
)
# The compiler version is a stable, manually-bumped identifier
# (toolchain/COMPILER_VERSION), not a git sha: VERSION.json feeds host cache
# keys, and a transient sha would churn those keys on every dgen commit.
compiler_version_file="${repo_root}/toolchain/COMPILER_VERSION"
if [ ! -f "${compiler_version_file}" ]; then
  echo "error: missing ${compiler_version_file} (single-line compiler version, e.g. abi-v1.1)" >&2
  exit 1
fi
dgen_compiler_version=$(head -n 1 "${compiler_version_file}" | tr -d '[:space:]')
if [ -z "${dgen_compiler_version}" ]; then
  echo "error: ${compiler_version_file} is empty" >&2
  exit 1
fi

# The host-varying tail of VERSION.json: Darwin records the deployment target
# it was built against, ELF records the glibc floor read back off the staged
# binaries. Both answer the same question -- the oldest system this stage runs
# on -- and both are consumed by packaging, never by the compile itself.
case "${host_target}" in
  arm64-apple-macos)
    platform_floor_json="  \"minimum_macos\": \"${minimum_macos}\","
    ;;
  x86_64-unknown-linux-gnu)
    glibc_floor=$(
      for staged_binary in "${stage_root}/bin/dgen-clang" "${stage_root}/bin/${lld_binary}"; do
        readelf -V "${staged_binary}" 2>/dev/null |
          sed -n 's/.*Name: GLIBC_\([0-9][0-9.]*\).*/\1/p'
      done | sort -t. -k1,1n -k2,2n -u | tail -n 1
    )
    if [ -z "${glibc_floor}" ]; then
      echo "error: could not read a glibc version floor off the staged binaries" >&2
      exit 1
    fi
    platform_floor_json="  \"glibc_floor\": \"${glibc_floor}\","
    ;;
esac

cat > "${stage_root}/VERSION.json" <<EOF
{
  "distribution_version": 3,
  "dgen_abi_version": 1,
  "codegen_policy_version": 1,
  "architecture_lowering_version": 1,
  "dgen_compiler_version": "${dgen_compiler_version}",
  "llvm_version": "${llvm_version}",
  "llvm_source_sha256": "${llvm_sha256}",
  "target": "${host_target}",
${platform_floor_json}
  "clang_sha256": "${clang_sha}",
  "lld_sha256": "${lld_sha}",
  "runtime_headers_sha256": "${runtime_headers_sha}"
}
EOF
cp "${stage_root}/VERSION.json" "${metadata_output}"

installed_bytes=$(du -sk "${stage_root}" | awk '{print $1 * 1024}')

# The staged report covers the installed footprint only — an archive cannot
# carry its own compressed size or digest. The repository copy below appends
# those two lines after the archive exists.
cat > "${stage_root}/SIZE.txt" <<EOF
LLVM source archive: $(file_size "${archive_path}") bytes
Installed staging prefix: ${installed_bytes} bytes
EOF

python3 - "${stage_root}" "${archive_output}" <<'PY'
import gzip
import os
import pathlib
import tarfile
import sys

stage = pathlib.Path(sys.argv[1]).resolve()
output = pathlib.Path(sys.argv[2]).resolve()
with output.open("wb") as raw:
    with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as gz:
        with tarfile.open(fileobj=gz, mode="w") as archive:
            for path in sorted(stage.rglob("*")):
                relative = pathlib.Path("dgen-toolchain") / path.relative_to(stage)
                info = archive.gettarinfo(str(path), arcname=str(relative))
                info.uid = info.gid = 0
                info.uname = info.gname = "root"
                info.mtime = 0
                if path.is_file():
                    with path.open("rb") as source:
                        archive.addfile(info, source)
                else:
                    archive.addfile(info)
PY

compressed_bytes=$(file_size "${archive_output}")
archive_sha=$(sha256_of "${archive_output}")

cp "${stage_root}/SIZE.txt" "${size_report}"
cat >> "${size_report}" <<EOF
Compressed toolchain archive: ${compressed_bytes} bytes
Compressed archive SHA-256: ${archive_sha}
EOF

echo "Staged toolchain: ${stage_root}"
echo "Target: ${host_target}"
echo "Compressed archive: ${archive_output}"
echo "Version metadata: ${metadata_output}"
echo "Size report: ${size_report}"
cat "${size_report}"
