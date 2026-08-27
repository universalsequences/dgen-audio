#!/bin/sh
set -eu

# ELF counterpart of scripts/prove-toolchain.sh, for the
# x86_64-unknown-linux-gnu stage produced by scripts/build-toolchain.sh.
#
# It proves the two things a freshly built Linux stage has to prove:
#
#   1. HERMETICITY. Every fixture compiles and links with nothing on PATH but
#      a directory that does not exist. If the staged compiler reached for a
#      system assembler, linker or header, the compile would fail here rather
#      than silently succeed on the build machine and fail on a user's.
#   2. ABI CONFORMANCE. Every produced .so passes scripts/audit-dgen-elf-so.sh:
#      exact export set, allowlisted undefined symbols only, a DT_SONAME, no
#      absolute build path anywhere in the image.
#
# It also runs the NaN/infinity containment fixture, because the production
# flag set is -ffast-math and the bit-level finite classifier in
# dgen_runtime.h is the only reason that is safe. That check is behavioural,
# so it must be re-run against every stage rather than reasoned about once.
#
# This is deliberately narrower than the Mach-O prove script: it does not do
# the system-clang A/B audio comparison, which needs the Accelerate-based
# harness. Numerical equivalence between hosts is covered by the ESeq-side
# spectral and dgen tests.

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
stage_root=${DGEN_TOOLCHAIN_STAGE_ROOT:-"${repo_root}/.toolchain/stage"}
proof_root=${DGEN_TOOLCHAIN_PROOF_ROOT:-"${repo_root}/.toolchain/proof-elf"}

clang="${stage_root}/bin/dgen-clang"
lld="${stage_root}/bin/ld.lld"
resource_dir="${stage_root}/lib/clang/20"
builtins="${resource_dir}/lib/x86_64-unknown-linux-gnu/libclang_rt.builtins.a"

for required_file in \
  "${clang}" "${lld}" "${builtins}" \
  "${stage_root}/include/dgen_runtime.h" \
  "${stage_root}/include/dgen_simd_compat.h" \
  "${stage_root}/VERSION.json"; do
  if [ ! -f "${required_file}" ]; then
    echo "error: required staged file is missing: ${required_file}" >&2
    echo "run scripts/build-toolchain.sh first" >&2
    exit 1
  fi
done

staged_target=$(sed -n 's/.*"target": *"\([^"]*\)".*/\1/p' "${stage_root}/VERSION.json")
if [ "${staged_target}" != "x86_64-unknown-linux-gnu" ]; then
  echo "error: staged toolchain targets ${staged_target:-an unknown target}," >&2
  echo "but this script proves the x86_64-unknown-linux-gnu stage." >&2
  exit 1
fi

case "${proof_root}" in
  ""|/) echo "error: refusing an empty or root proof path" >&2; exit 1 ;;
esac
rm -rf "${proof_root}/candidate" "${proof_root}/logs"
mkdir -p "${proof_root}/candidate" "${proof_root}/logs"

"${repo_root}/scripts/prepare-toolchain-fixtures.sh"

# The compile/link contract, spelled out rather than derived, so a drift
# between this proof and DGenToolchainPolicy's linux-x86_64 profile shows up
# as a diff in this file. -march=x86-64-v3 is a NUMERICAL floor: below it
# there is no FMA instruction to select and every fused multiply-add in
# dgen_simd_compat.h silently double-rounds.
common_compile_flags="
-target x86_64-unknown-linux-gnu
-O3
-march=x86-64-v3
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
-fno-stack-protector
-fno-asynchronous-unwind-tables
-ffunction-sections
-fdata-sections
-std=c11
-x c
-ffreestanding
-nostdinc
-resource-dir ${resource_dir}
-isystem ${resource_dir}/include
-I ${stage_root}/include
"

build_hermetic_so() {
  fixture_name=$1
  input_path=$2
  output_path="${proof_root}/candidate/${fixture_name}.so"
  # Intentionally empty/PATH-less environment. Only the absolute staged
  # compiler path is executable during candidate compilation; PATH names a
  # directory that does not exist, so any system-tool lookup fails loudly.
  # `-x c` above applies to every input that follows it, so the builtins
  # archive needs an explicit `-x none` to be treated as a library rather than
  # compiled as C. DGenToolchainPolicy.embeddedInvocation spells it the same
  # way; this proof is only worth running if it matches production argument
  # for argument.
  # shellcheck disable=SC2086
  env -i \
    PATH=/dgen-no-system-tools \
    TMPDIR="${proof_root}" \
    LC_ALL=C \
    "${clang}" ${common_compile_flags} \
    -fuse-ld="${lld}" \
    -nostdlib \
    -shared \
    -Wl,-soname,"${fixture_name}.so" \
    -Wl,--gc-sections \
    "${input_path}" \
    -x none "${builtins}" \
    -o "${output_path}"
  "${repo_root}/scripts/audit-dgen-elf-so.sh" "${output_path}"
}

for fixture_name in \
  scalar-synth feedback-delay-effect wavetable-instrument spectral-effect; do
  build_hermetic_so \
    "${fixture_name}" \
    "${repo_root}/toolchain/generated/hermetic/${fixture_name}.c"
done

# Prove that the bit-level finite classifier still contains NaN and infinity
# with the production fast-math policy.
build_hermetic_so \
  nonfinite-containment \
  "${repo_root}/toolchain/fixtures/nonfinite-containment.c"

# The harness is an ordinary host program, not part of the hermetic surface,
# so it is built with the system compiler exactly as the Mach-O proof does.
"${CC:-cc}" -O2 -std=c11 \
  "${repo_root}/toolchain/harness/nonfinite_harness.c" \
  -o "${proof_root}/nonfinite-harness" \
  -ldl -lm

containment_output=$("${proof_root}/nonfinite-harness" \
  "${proof_root}/candidate/nonfinite-containment.so")
printf '%s\n' "${containment_output}" |
  tee "${proof_root}/logs/nonfinite-containment.txt"

{
  echo "## Tool versions"
  "${clang}" --version
  "${lld}" --version
  echo
  echo "## Stage identity"
  cat "${stage_root}/VERSION.json"
  echo
  echo "## NaN/Inf containment"
  cat "${proof_root}/logs/nonfinite-containment.txt"
} > "${proof_root}/logs/summary.txt"

echo
echo "All fixtures compiled hermetically and passed scripts/audit-dgen-elf-so.sh."
echo "Proof root: ${proof_root}"
echo "Summary: ${proof_root}/logs/summary.txt"
