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

for required_file in \
  "${clang}" "${lld}" "${builtins}" \
  "${stub_dir}/libSystem.tbd" "${stage_root}/include/dgen_runtime.h"; do
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
  "${proof_root}/candidate" \
  "${proof_root}/logs"
mkdir -p \
  "${proof_root}/objects" \
  "${proof_root}/baseline" \
  "${proof_root}/candidate" \
  "${proof_root}/empty-sdk" \
  "${proof_root}/logs"

"${repo_root}/scripts/generate-libsystem-stub.sh" "${stage_root}/lib/libSystem.tbd"
"${repo_root}/scripts/prepare-toolchain-fixtures.sh"

common_compile_flags="
-target arm64-apple-macos11.0
-O3
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
-fvisibility=hidden
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
  input_path=$2
  object_path="${proof_root}/objects/${fixture_name}.o"
  # Intentionally empty/PATH-less environment. Only the absolute staged
  # compiler path is executable during candidate compilation.
  # shellcheck disable=SC2086
  env -i \
    PATH=/dgen-no-system-tools \
    DEVELOPER_DIR= \
    SDKROOT= \
    TMPDIR="${proof_root}" \
    LC_ALL=C \
    "${clang}" ${common_compile_flags} \
    -c "${input_path}" -o "${object_path}"
}

link_hermetic_dylib() {
  fixture_name=$1
  output_path="${proof_root}/candidate/${fixture_name}.dylib"
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
  "${repo_root}/scripts/audit-dgen-dylib.sh" "${output_path}"
}

build_system_baseline() {
  fixture_name=$1
  /usr/bin/clang \
    -target arm64-apple-macos11.0 \
    -O3 \
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
    -fvisibility=hidden \
    -fno-stack-protector \
    -fno-asynchronous-unwind-tables \
    -dynamiclib \
    -Wl,-dead_strip \
    -Wl,-install_name,"@rpath/${fixture_name}.dylib" \
    -I"${repo_root}/toolchain/include" \
    -std=c11 \
    -x c \
    -o "${proof_root}/baseline/${fixture_name}.dylib" \
    "${repo_root}/toolchain/generated/current/${fixture_name}.c"
  "${repo_root}/scripts/audit-dgen-dylib.sh" \
    "${proof_root}/baseline/${fixture_name}.dylib"
}

for fixture_name in \
  scalar-synth feedback-delay-effect wavetable-instrument spectral-effect; do
  build_system_baseline "${fixture_name}"
  compile_hermetic_object \
    "${fixture_name}" \
    "${repo_root}/toolchain/generated/hermetic/${fixture_name}.c"
  link_hermetic_dylib "${fixture_name}"
done

# Prove that the bit-level finite classifier still contains NaN and infinity
# with the production fast-math policy.
compile_hermetic_object \
  nonfinite-containment \
  "${repo_root}/toolchain/fixtures/nonfinite-containment.c"
link_hermetic_dylib nonfinite-containment

/usr/bin/clang \
  -O2 -std=c11 \
  "${harness_source}" \
  -framework Accelerate \
  -o "${proof_root}/toolchain-harness"
/usr/bin/clang \
  -O2 -std=c11 \
  "${repo_root}/toolchain/harness/nonfinite_harness.c" \
  -o "${proof_root}/nonfinite-harness"

containment_output=$("${proof_root}/nonfinite-harness" \
  "${proof_root}/candidate/nonfinite-containment.dylib")
printf '%s\n' "${containment_output}" |
  tee "${proof_root}/logs/nonfinite-containment.txt"

: > "${proof_root}/logs/audio-comparison.txt"
for fixture_name in scalar-synth wavetable-instrument spectral-effect; do
  comparison_output=$("${proof_root}/toolchain-harness" \
    "${fixture_name}" \
    "${proof_root}/baseline/${fixture_name}.dylib" \
    "${proof_root}/candidate/${fixture_name}.dylib" \
    128 64 2e-5)
  printf '%s\n' "${comparison_output}" |
    tee -a "${proof_root}/logs/audio-comparison.txt"
done

# Five seconds at 48 kHz exercises feedback divergence and delay wraparound.
comparison_output=$("${proof_root}/toolchain-harness" \
  feedback-delay-effect \
  "${proof_root}/baseline/feedback-delay-effect.dylib" \
  "${proof_root}/candidate/feedback-delay-effect.dylib" \
  3750 64 2e-5)
printf '%s\n' "${comparison_output}" |
  tee -a "${proof_root}/logs/audio-comparison.txt"

{
  echo "## Tool versions"
  "${clang}" --version
  "${lld}" --version
  echo
  echo "## NaN/Inf containment"
  cat "${proof_root}/logs/nonfinite-containment.txt"
  echo
  echo "## Audio comparisons"
  cat "${proof_root}/logs/audio-comparison.txt"
  echo
  echo "## Candidate dylib dependencies"
  for dylib in "${proof_root}/candidate"/*.dylib; do
    otool -L "${dylib}"
  done
  echo
  echo "## Candidate exports"
  for dylib in "${proof_root}/candidate"/*.dylib; do
    echo "### ${dylib}"
    nm -gU "${dylib}" | awk '{print $NF}' | sort -u
  done
  echo
  echo "## Candidate undefined symbols"
  for dylib in "${proof_root}/candidate"/*.dylib; do
    echo "### ${dylib}"
    nm -u "${dylib}" | sort -u
  done
  echo
  echo "## Architecture and deployment target"
  for dylib in "${proof_root}/candidate"/*.dylib; do
    file "${dylib}"
    otool -l "${dylib}" |
      awk '/cmd LC_BUILD_VERSION/ { show = 1 }
           show { print }
           show && $1 == "sdk" { show = 0 }'
  done
} > "${proof_root}/proof-report.txt"

echo "Phase 2 hermetic compile/link/audit/load/run proof passed."
echo "Report: ${proof_root}/proof-report.txt"
