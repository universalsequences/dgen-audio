#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
stage_root=${DGEN_TOOLCHAIN_STAGE_ROOT:-"${repo_root}/.toolchain/stage"}
proof_root=${DGEN_TOOLCHAIN_PROOF_ROOT:-"${repo_root}/.toolchain/proof"}
clang="${stage_root}/bin/dgen-clang"
lld="${stage_root}/bin/ld64.lld"
resource_dir="${stage_root}/lib/clang/20"
builtins="${resource_dir}/lib/darwin/libclang_rt.builtins.a"
stub_dir="${stage_root}/lib"
harness_source="${repo_root}/toolchain/harness/toolchain_harness.c"

for required_file in "${clang}" "${lld}" "${builtins}" "${stub_dir}/libSystem.tbd"; do
  if [ ! -f "${required_file}" ]; then
    echo "error: required staged file is missing: ${required_file}" >&2
    echo "run scripts/build-toolchain.sh first" >&2
    exit 1
  fi
done

case "${proof_root}" in
  ""|/) echo "error: refusing an empty or root proof path" >&2; exit 1 ;;
esac
rm -rf \
  "${proof_root}/objects" \
  "${proof_root}/baseline" \
  "${proof_root}/stub" \
  "${proof_root}/dynamic-lookup" \
  "${proof_root}/logs"
mkdir -p \
  "${proof_root}/objects" \
  "${proof_root}/baseline" \
  "${proof_root}/stub" \
  "${proof_root}/dynamic-lookup" \
  "${proof_root}/empty-sdk" \
  "${proof_root}/logs"

"${repo_root}/scripts/prepare-toolchain-fixtures.sh"

common_compile_flags="
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
-ffreestanding
-fno-stack-protector
-fno-asynchronous-unwind-tables
-std=c11
-x c
-nostdinc
-isysroot ${proof_root}/empty-sdk
-resource-dir ${resource_dir}
-isystem ${resource_dir}/include
-I ${stage_root}/include
"

compile_hermetic_object() {
  fixture_name=$1
  input_path="${repo_root}/toolchain/generated/hermetic/${fixture_name}.c"
  object_path="${proof_root}/objects/${fixture_name}.o"
  extra_compile_flags=
  if [ "${fixture_name}" = spectral-effect ]; then
    # This object is inspected but not linked in Phase 1; make it a concrete
    # arm64 Mach-O object rather than ThinLTO bitcode.
    extra_compile_flags=-fno-lto
  fi
  # Intentionally empty/PATH-less environment. The absolute staged compiler
  # and linker paths are the only executable paths in the compilation.
  # shellcheck disable=SC2086
  env -i \
    PATH=/dgen-no-system-tools \
    DEVELOPER_DIR= \
    SDKROOT= \
    TMPDIR="${proof_root}" \
    LC_ALL=C \
    "${clang}" ${common_compile_flags} ${extra_compile_flags} \
    -c "${input_path}" -o "${object_path}"
}

link_hermetic_dylib() {
  mode=$1
  fixture_name=$2
  output_path="${proof_root}/${mode}/${fixture_name}.dylib"
  if [ "${mode}" = stub ]; then
    env -i \
      PATH=/dgen-no-system-tools \
      DEVELOPER_DIR= \
      SDKROOT= \
      TMPDIR="${proof_root}" \
      LC_ALL=C \
      "${clang}" \
      -target arm64-apple-macos11.0 \
      -fuse-ld="${lld}" \
      -nostdlib \
      -isysroot "${proof_root}/empty-sdk" \
      -dynamiclib \
      -Wl,-install_name,"@rpath/${fixture_name}.dylib" \
      -Wl,-undefined,error \
      -Wl,-fatal_warnings \
      -Wl,-dead_strip \
      -L"${stub_dir}" \
      "${proof_root}/objects/${fixture_name}.o" \
      "${builtins}" \
      -lSystem \
      -o "${output_path}"
  else
    env -i \
      PATH=/dgen-no-system-tools \
      DEVELOPER_DIR= \
      SDKROOT= \
      TMPDIR="${proof_root}" \
      LC_ALL=C \
      "${clang}" \
      -target arm64-apple-macos11.0 \
      -fuse-ld="${lld}" \
      -nostdlib \
      -isysroot "${proof_root}/empty-sdk" \
      -dynamiclib \
      -Wl,-install_name,"@rpath/${fixture_name}.dylib" \
      -Wl,-undefined,dynamic_lookup \
      -Wl,-fatal_warnings \
      -Wl,-dead_strip \
      "${proof_root}/objects/${fixture_name}.o" \
      "${builtins}" \
      -o "${output_path}"
  fi
}

build_current_baseline() {
  fixture_name=$1
  /usr/bin/clang \
    -arch arm64 \
    -Ofast \
    -mcpu=apple-m1 \
    -flto=thin \
    -ffast-math \
    -fno-math-errno \
    -fno-trapping-math \
    -ffp-contract=fast \
    -fvectorize \
    -fslp-vectorize \
    -funroll-loops \
    -fPIC \
    -shared \
    -framework Accelerate \
    -std=c11 \
    -x c \
    -o "${proof_root}/baseline/${fixture_name}.dylib" \
    "${repo_root}/toolchain/generated/current/${fixture_name}.c"
}

for fixture_name in scalar-synth feedback-delay-effect wavetable-instrument; do
  build_current_baseline "${fixture_name}"
  compile_hermetic_object "${fixture_name}"
  link_hermetic_dylib stub "${fixture_name}"
  link_hermetic_dylib dynamic-lookup "${fixture_name}"
done

# The current spectral source compiles with no SDK headers. It intentionally
# does not enter the libSystem-only link/run set until Phase 2 replaces vDSP
# calls with DGenHostServicesV1.
compile_hermetic_object spectral-effect
file "${proof_root}/objects/spectral-effect.o" | grep -q 'Mach-O 64-bit object arm64'

# Link-time diagnostic proof. The minimal stub must reject the typo, while
# dynamic_lookup must accept it and defer failure to the post-link audit/load.
env -i \
  PATH=/dgen-no-system-tools \
  DEVELOPER_DIR= \
  SDKROOT= \
  TMPDIR="${proof_root}" \
  LC_ALL=C \
  "${clang}" ${common_compile_flags} \
  -c "${repo_root}/toolchain/harness/link-contract-typo.c" \
  -o "${proof_root}/objects/link-contract-typo.o"

if env -i \
  PATH=/dgen-no-system-tools \
  DEVELOPER_DIR= \
  SDKROOT= \
  TMPDIR="${proof_root}" \
  LC_ALL=C \
  "${clang}" \
  -target arm64-apple-macos11.0 \
  -fuse-ld="${lld}" \
  -nostdlib \
  -isysroot "${proof_root}/empty-sdk" \
  -dynamiclib \
  -Wl,-install_name,@rpath/link-contract-typo.dylib \
  -Wl,-undefined,error \
  -L"${stub_dir}" \
  "${proof_root}/objects/link-contract-typo.o" \
  "${builtins}" \
  -lSystem \
  -o "${proof_root}/stub/link-contract-typo.dylib" \
  >"${proof_root}/logs/stub-typo.stdout" \
  2>"${proof_root}/logs/stub-typo.stderr"; then
  echo "error: strict stub link unexpectedly accepted deliberate typo" >&2
  exit 1
fi

env -i \
  PATH=/dgen-no-system-tools \
  DEVELOPER_DIR= \
  SDKROOT= \
  TMPDIR="${proof_root}" \
  LC_ALL=C \
  "${clang}" \
  -target arm64-apple-macos11.0 \
  -fuse-ld="${lld}" \
  -nostdlib \
  -isysroot "${proof_root}/empty-sdk" \
  -dynamiclib \
  -Wl,-install_name,@rpath/link-contract-typo.dylib \
  -Wl,-undefined,dynamic_lookup \
  "${proof_root}/objects/link-contract-typo.o" \
  "${builtins}" \
  -o "${proof_root}/dynamic-lookup/link-contract-typo.dylib"

/usr/bin/clang \
  -O2 -std=c11 \
  "${harness_source}" \
  -o "${proof_root}/toolchain-harness"

: > "${proof_root}/logs/audio-comparison.txt"
for mode in stub dynamic-lookup; do
  for fixture_name in scalar-synth feedback-delay-effect wavetable-instrument; do
    comparison_output=$("${proof_root}/toolchain-harness" \
      "${fixture_name}" \
      "${proof_root}/baseline/${fixture_name}.dylib" \
      "${proof_root}/${mode}/${fixture_name}.dylib" \
      8 64 2e-5)
    printf '%s: %s\n' "${mode}" "${comparison_output}" |
      tee -a "${proof_root}/logs/audio-comparison.txt"
  done
done

for mode in stub dynamic-lookup; do
  for dylib in "${proof_root}/${mode}"/*.dylib; do
    case "${dylib}" in
      *link-contract-typo.dylib) continue ;;
    esac
    file "${dylib}" | grep -q 'Mach-O 64-bit dynamically linked shared library arm64'
    minimum_version=$(otool -l "${dylib}" |
      awk '/cmd LC_BUILD_VERSION/ { in_build_version = 1; next }
           in_build_version && $1 == "minos" { print $2; exit }')
    if [ "${minimum_version}" != 11.0 ]; then
      echo "error: expected macOS 11.0 minimum in ${dylib}, got ${minimum_version}" >&2
      exit 1
    fi
    # A dylib's first otool -L entry is LC_ID_DYLIB, not a dependency.
    dependencies=$(otool -L "${dylib}" | tail -n +3 | awk '{print $1}')
    unexpected=$(printf '%s\n' "${dependencies}" |
      grep -v '^/usr/lib/libSystem\.B\.dylib$' || true)
    if [ -n "${unexpected}" ]; then
      echo "error: unexpected load dependency in ${dylib}: ${unexpected}" >&2
      exit 1
    fi
    if [ "${mode}" = stub ] &&
      [ "${dependencies}" != /usr/lib/libSystem.B.dylib ]; then
      echo "error: strict-stub dylib does not load exactly libSystem: ${dylib}" >&2
      exit 1
    fi
    if strings "${dylib}" |
      grep -E '/Applications/Xcode|/Library/Developer/CommandLineTools|/usr/bin/clang|/Users/' \
      >"${proof_root}/logs/forbidden-paths.txt"; then
      echo "error: forbidden developer/workspace path embedded in ${dylib}" >&2
      exit 1
    fi
    if otool -l "${dylib}" | grep -q 'cmd LC_RPATH'; then
      echo "error: unexpected LC_RPATH in ${dylib}" >&2
      exit 1
    fi
  done
done

{
  echo "## Tool versions"
  "${clang}" --version
  "${lld}" --version
  echo
  echo "## Strict-stub typo diagnostic"
  cat "${proof_root}/logs/stub-typo.stderr"
  echo
  echo "## Audio comparisons"
  cat "${proof_root}/logs/audio-comparison.txt"
  echo
  echo "## Dylib dependencies"
  for dylib in "${proof_root}/stub"/*.dylib "${proof_root}/dynamic-lookup"/*.dylib; do
    case "${dylib}" in
      *link-contract-typo.dylib) continue ;;
    esac
    otool -L "${dylib}"
  done
  echo
  echo "## Undefined symbols"
  for dylib in "${proof_root}/stub"/*.dylib "${proof_root}/dynamic-lookup"/*.dylib; do
    case "${dylib}" in
      *link-contract-typo.dylib) continue ;;
    esac
    echo "### ${dylib}"
    nm -u "${dylib}" | sort -u
  done
  echo
  echo "## Architecture and deployment target"
  for dylib in "${proof_root}/stub"/*.dylib "${proof_root}/dynamic-lookup"/*.dylib; do
    case "${dylib}" in
      *link-contract-typo.dylib) continue ;;
    esac
    file "${dylib}"
    otool -l "${dylib}" |
      awk '/cmd LC_BUILD_VERSION/ { show = 1 }
           show { print }
           show && $1 == "sdk" { show = 0 }'
  done
} > "${proof_root}/proof-report.txt"

echo "Hermetic compile/link/run proof passed."
echo "Report: ${proof_root}/proof-report.txt"
