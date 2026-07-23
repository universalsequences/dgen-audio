#!/bin/sh
set -eu

# Reproducible upstream-only DGen toolchain build.
# No executable or library is copied from Xcode into the staged distribution.

llvm_version=20.1.8
llvm_archive="llvm-project-${llvm_version}.src.tar.xz"
llvm_url="https://github.com/llvm/llvm-project/releases/download/llvmorg-${llvm_version}/${llvm_archive}"
llvm_sha256=6898f963c8e938981e6c4a302e83ec5beb4630147c7311183cf61069af16333d
minimum_macos=11.0
target_triple=arm64-apple-macos

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
work_root=${DGEN_TOOLCHAIN_WORK_ROOT:-"${repo_root}/.toolchain-build"}
download_root=${DGEN_TOOLCHAIN_DOWNLOAD_ROOT:-"${repo_root}/.toolchain-downloads"}
stage_root=${DGEN_TOOLCHAIN_STAGE_ROOT:-"${repo_root}/.toolchain/stage"}
archive_output=${DGEN_TOOLCHAIN_ARCHIVE:-"${repo_root}/.toolchain/dgen-toolchain-${llvm_version}-arm64.tar.gz"}
metadata_output=${DGEN_TOOLCHAIN_METADATA_OUTPUT:-"${repo_root}/toolchain/VERSION.json"}
size_report=${DGEN_TOOLCHAIN_SIZE_REPORT:-"${repo_root}/toolchain/SIZE.txt"}
jobs=${DGEN_TOOLCHAIN_JOBS:-$(sysctl -n hw.logicalcpu 2>/dev/null || echo 4)}

source_root="${work_root}/llvm-project-${llvm_version}.src"
build_root="${work_root}/build"
archive_path="${download_root}/${llvm_archive}"

require_tool() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "error: required tool not found: $1" >&2
    exit 1
  }
}

require_tool cmake
require_tool curl
require_tool otool
require_tool python3
require_tool patch
require_tool shasum
require_tool strings
require_tool tar

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

"${repo_root}/scripts/generate-libsystem-stub.sh"

if [ ! -f "${archive_path}" ]; then
  curl -fL --retry 3 -o "${archive_path}.partial" "${llvm_url}"
  mv "${archive_path}.partial" "${archive_path}"
fi

actual_archive_sha=$(shasum -a 256 "${archive_path}" | awk '{print $1}')
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
# WebAssembly ports into one executable. The embedded DGen toolchain needs only
# Mach-O, so apply the checked-in, reviewable narrowing patch. This changes the
# driver dispatcher and linked libraries; it does not import Apple code.
lld_patch="${repo_root}/toolchain/patches/lld-macho-only.patch"
if ! grep -q 'DGEN_LLD_MACHO_DRIVER' "${source_root}/lld/tools/lld/lld.cpp"; then
  patch -d "${source_root}" -p1 < "${lld_patch}"
fi

generator="Unix Makefiles"
if command -v ninja >/dev/null 2>&1; then
  generator=Ninja
fi

prefix_map="-ffile-prefix-map=${work_root}=/usr/src/dgen-llvm-${llvm_version} -fdebug-prefix-map=${work_root}=/usr/src/dgen-llvm-${llvm_version} -fmacro-prefix-map=${work_root}=/usr/src/dgen-llvm-${llvm_version}"

cmake -S "${source_root}/llvm" -B "${build_root}" -G "${generator}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET="${minimum_macos}" \
  -DCMAKE_C_FLAGS_RELEASE="-O2 -DNDEBUG -g0 ${prefix_map}" \
  -DCMAKE_CXX_FLAGS_RELEASE="-O2 -DNDEBUG -g0 ${prefix_map}" \
  -DLLVM_ENABLE_PROJECTS="clang;lld" \
  -DLLVM_ENABLE_RUNTIMES=compiler-rt \
  -DLLVM_TARGETS_TO_BUILD=AArch64 \
  -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD= \
  -DLLVM_DEFAULT_TARGET_TRIPLE="${target_triple}${minimum_macos}" \
  -DLLVM_TARGET_ARCH=AArch64 \
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
  -DLLD_SYMLINKS_TO_CREATE=ld64.lld \
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

cmake --build "${build_root}" \
  --target clang lld clang-resource-headers builtins-configure \
  --parallel "${jobs}"
cmake --build "${build_root}/runtimes/builtins-bins" \
  --target clang_rt.builtins_arm64_osx \
  --parallel "${jobs}"

rm -rf "${stage_root}"
mkdir -p \
  "${stage_root}/bin" \
  "${stage_root}/include" \
  "${stage_root}/lib/clang/${llvm_version%%.*}/include" \
  "${stage_root}/lib/clang/${llvm_version%%.*}/lib/darwin" \
  "${stage_root}/empty-sdk" \
  "${stage_root}/LICENSES"

cp "${build_root}/bin/clang" "${stage_root}/bin/dgen-clang"
cp "${build_root}/bin/ld64.lld" "${stage_root}/bin/ld64.lld"
cp -R "${build_root}/lib/clang/${llvm_version%%.*}/include/." \
  "${stage_root}/lib/clang/${llvm_version%%.*}/include/"
cp "${repo_root}/toolchain/include/dgen_runtime.h" "${stage_root}/include/"
cp "${repo_root}/toolchain/lib/libSystem.tbd" "${stage_root}/lib/"
cp "${source_root}/llvm/LICENSE.TXT" "${stage_root}/LICENSES/LLVM-LICENSE.txt"
cp "${repo_root}/toolchain/THIRD-PARTY-NOTICES.txt" \
  "${stage_root}/LICENSES/THIRD-PARTY-NOTICES.txt"

builtins_archive=$(find "${build_root}/runtimes/builtins-bins" -type f \
  -name 'libclang_rt.builtins_arm64_osx.a' -print | head -n 1)
if [ -z "${builtins_archive}" ]; then
  echo "error: compiler-rt builtins archive was not produced" >&2
  exit 1
fi
cp "${builtins_archive}" \
  "${stage_root}/lib/clang/${llvm_version%%.*}/lib/darwin/libclang_rt.builtins.a"

# Symbol stripping is a supported packaging operation. It changes neither the
# selected components nor their behavior.
if command -v strip >/dev/null 2>&1; then
  strip -S "${stage_root}/bin/dgen-clang"
  strip -S "${stage_root}/bin/ld64.lld"
fi

for staged_binary in "${stage_root}/bin/dgen-clang" "${stage_root}/bin/ld64.lld"; do
  if strings "${staged_binary}" |
    grep -E "${repo_root}|${work_root}|/Applications/Xcode|/Library/Developer/CommandLineTools|/usr/bin/clang" \
    >/dev/null; then
    echo "error: forbidden build/developer path embedded in ${staged_binary}" >&2
    exit 1
  fi
  # Skip otool's first line: it echoes the inspected file's staging path.
  if otool -L "${staged_binary}" | tail -n +2 |
    grep -E "${repo_root}|${work_root}|/Applications/Xcode|/Library/Developer/CommandLineTools" \
    >/dev/null; then
    echo "error: forbidden load dependency in ${staged_binary}" >&2
    exit 1
  fi
done

clang_sha=$(shasum -a 256 "${stage_root}/bin/dgen-clang" | awk '{print $1}')
lld_sha=$(shasum -a 256 "${stage_root}/bin/ld64.lld" | awk '{print $1}')
runtime_headers_sha=$(
  cd "${stage_root}/include"
  find . -type f -print | LC_ALL=C sort |
    while IFS= read -r header; do
      header_sha=$(shasum -a 256 "${header}" | awk '{print $1}')
      printf '%s  %s\n' "${header_sha}" "${header}"
    done |
    shasum -a 256 | awk '{print $1}'
)
dgen_compiler_version=$(git -C "${repo_root}" rev-parse HEAD 2>/dev/null || echo unknown)

cat > "${stage_root}/VERSION.json" <<EOF
{
  "distribution_version": 2,
  "dgen_abi_version": 1,
  "codegen_policy_version": 1,
  "architecture_lowering_version": 1,
  "dgen_compiler_version": "${dgen_compiler_version}",
  "llvm_version": "${llvm_version}",
  "llvm_source_sha256": "${llvm_sha256}",
  "target": "${target_triple}",
  "minimum_macos": "${minimum_macos}",
  "clang_sha256": "${clang_sha}",
  "lld_sha256": "${lld_sha}",
  "runtime_headers_sha256": "${runtime_headers_sha}"
}
EOF
cp "${stage_root}/VERSION.json" "${metadata_output}"

installed_bytes=$(du -sk "${stage_root}" | awk '{print $1 * 1024}')

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

compressed_bytes=$(stat -f %z "${archive_output}")
archive_sha=$(shasum -a 256 "${archive_output}" | awk '{print $1}')

cat > "${size_report}" <<EOF
LLVM source archive: $(stat -f %z "${archive_path}") bytes
Installed staging prefix: ${installed_bytes} bytes
Compressed toolchain archive: ${compressed_bytes} bytes
Compressed archive SHA-256: ${archive_sha}
EOF

echo "Staged toolchain: ${stage_root}"
echo "Compressed archive: ${archive_output}"
echo "Version metadata: ${metadata_output}"
echo "Size report: ${size_report}"
cat "${size_report}"
