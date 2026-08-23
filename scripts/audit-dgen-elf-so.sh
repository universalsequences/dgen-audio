#!/bin/sh
set -eu

# ELF counterpart of scripts/audit-dgen-dylib.sh, for x86_64-linux artifacts.
#
# The audit inspects an already-built artifact; it is a verification step, not
# part of the hermetic compile, and it needs the system binutils. Callers
# legitimately run under a deliberately neutered PATH to prove the *compile*
# reaches no system tool, so restore a usable search path here.
PATH="${PATH:+${PATH}:}/usr/bin:/bin"
export PATH

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
sofile=${1:-}
exports_allowlist=${DGEN_EXPORT_ALLOWLIST:-"${repo_root}/toolchain/abi/exports-v1-elf.txt"}
undefined_allowlist=${DGEN_UNDEFINED_ALLOWLIST:-"${repo_root}/toolchain/abi/libsystem-symbols-v1-elf.txt"}

# DT_NEEDED allowlist. The Mach-O audit demands exactly libSystem, the single
# umbrella library Darwin offers. Linux splits the same surface across libc and
# libm, and which of the two appears depends on both the compile policy and the
# glibc version (>= 2.34 folds libm into libc, so a -lm link can resolve every
# math symbol from libc.so.6 alone and DT_NEEDED libm.so.6 never materialises).
# Every entry here must be a base-system library carrying only the symbols in
# the undefined allowlist. A hermetic -nostdlib link produces no DT_NEEDED at
# all, which is also accepted. Space-separated; override for other libcs.
allowed_needed=${DGEN_ALLOWED_NEEDED:-"libc.so.6 libm.so.6"}

# Dynamic exports the linker synthesises for any shared object. These are link
# artifacts, not DGen ABI, and which of them reach .dynsym varies by linker
# (bfd/gold/lld) and by whether crti.o/crtn.o were linked at all. They are
# filtered out here rather than listed in exports-v1-elf.txt so that the
# allowlist file stays a statement about the DGen contract and so that the
# exact-match export check yields the same verdict under every link mode.
linker_generated_exports=${DGEN_LINKER_GENERATED_EXPORTS:-"_init _fini __bss_start _edata _end _edata_end __dso_handle"}

# Absolute paths that must never be baked into a published artifact. Set empty
# to skip the scan.
forbidden_path_pattern=${DGEN_FORBIDDEN_PATH_PATTERN:-'/home/|/root/|/tmp/|/var/tmp/|/nix/store/|/usr/bin/clang|/usr/lib/llvm|/usr/lib/gcc'}

# DT_SONAME presence is the analogue of Mach-O's mandatory LC_ID_DYLIB. Set to
# 0 while the Linux compile policy does not yet pass -Wl,-soname.
require_soname=${DGEN_REQUIRE_SONAME:-1}

if [ -z "${sofile}" ] || [ ! -f "${sofile}" ]; then
  echo "usage: $0 SHARED_OBJECT" >&2
  exit 2
fi

for tool in file nm readelf; do
  command -v "${tool}" >/dev/null 2>&1 || {
    echo "error: required audit tool is unavailable: ${tool}" >&2
    exit 2
  }
done

audit_root=$(mktemp -d "${TMPDIR:-/tmp}/dgen-audit.XXXXXX")
trap 'rm -rf "${audit_root}"' EXIT HUP INT TERM

# Allowlists carry #-comments, the convention already used by
# scripts/generate-libsystem-stub.sh.
read_allowlist() {
  if [ ! -f "$1" ]; then
    echo "error: symbol allowlist is missing: $1" >&2
    exit 2
  fi
  sed -e 's/#.*//' -e 's/[[:space:]]*$//' "$1" | awk 'NF { print $1 }'
}

file_output=$(file "${sofile}")
case "${file_output}" in
  *"ELF 64-bit LSB shared object, x86-64"*) ;;
  *)
    echo "error: not an x86-64 ELF shared object: ${file_output}" >&2
    exit 1
    ;;
esac

# DT_SONAME must be a bare filename. An absolute soname bakes the build
# machine's staging path into every published artifact, exactly as an absolute
# Mach-O install name would.
soname=$(readelf -d "${sofile}" |
  awk '/\(SONAME\)/ { if (match($0, /\[[^]]*\]/)) { print substr($0, RSTART + 1, RLENGTH - 2); exit } }')
case "${soname}" in
  "")
    if [ "${require_soname}" != "0" ]; then
      echo "error: shared object has no DT_SONAME" >&2
      exit 1
    fi
    ;;
  */*)
    echo "error: DT_SONAME must be a bare filename; found: ${soname}" >&2
    exit 1
    ;;
  *) ;;
esac

# Exported dynamic symbols: GLOBAL/WEAK bindings that are defined here
# (section index other than UND) and externally visible. Both DEFAULT and
# PROTECTED visibility count as exported -- protected only forbids preemption,
# the symbol is still resolvable by dlsym -- so both are audited; HIDDEN and
# INTERNAL are not exports and should never reach .dynsym at all. readelf is
# used over `nm -D --defined-only` precisely because nm reports no visibility.
# Versioned defines carry a `@@VERSION` suffix, which is stripped.
read_allowlist "${exports_allowlist}" | LC_ALL=C sort -u \
  > "${audit_root}/expected-exports"
printf '%s\n' ${linker_generated_exports} | LC_ALL=C sort -u \
  > "${audit_root}/linker-generated"
readelf --dyn-syms -W "${sofile}" |
  awk '$1 ~ /^[0-9]+:$/ && $7 != "UND" && $8 != "" &&
       ($5 == "GLOBAL" || $5 == "WEAK") &&
       ($6 == "DEFAULT" || $6 == "PROTECTED") {
         name = $8; sub(/@.*/, "", name); print name
       }' |
  LC_ALL=C sort -u > "${audit_root}/raw-exports"
LC_ALL=C comm -23 "${audit_root}/raw-exports" "${audit_root}/linker-generated" \
  > "${audit_root}/actual-exports"
if ! cmp -s "${audit_root}/expected-exports" "${audit_root}/actual-exports"; then
  echo "error: exported symbols do not exactly match DGen ABI v1" >&2
  diff -u "${audit_root}/expected-exports" "${audit_root}/actual-exports" >&2 || true
  exit 1
fi

# Undefined dynamic symbols, WEAK ones included: a weak undefined reference is
# still a reference to something outside the artifact. Versioned imports carry
# an `@VERSION` suffix and a trailing version index, both dropped here.
read_allowlist "${undefined_allowlist}" | LC_ALL=C sort -u \
  > "${audit_root}/allowed-undefined"
readelf --dyn-syms -W "${sofile}" |
  awk '$1 ~ /^[0-9]+:$/ && $7 == "UND" && $8 != "" {
         name = $8; sub(/@.*/, "", name); print name
       }' |
  LC_ALL=C sort -u > "${audit_root}/actual-undefined"
unexpected_undefined=$(LC_ALL=C comm -23 \
  "${audit_root}/actual-undefined" "${audit_root}/allowed-undefined")
if [ -n "${unexpected_undefined}" ]; then
  echo "error: undefined symbols fall outside DGen ABI v1 allowlist:" >&2
  printf '%s\n' "${unexpected_undefined}" >&2
  exit 1
fi

printf '%s\n' ${allowed_needed} | LC_ALL=C sort -u > "${audit_root}/allowed-needed"
readelf -d "${sofile}" |
  awk '/\(NEEDED\)/ { if (match($0, /\[[^]]*\]/)) print substr($0, RSTART + 1, RLENGTH - 2) }' |
  LC_ALL=C sort -u > "${audit_root}/actual-needed"
unexpected_needed=$(LC_ALL=C comm -23 \
  "${audit_root}/actual-needed" "${audit_root}/allowed-needed")
if [ -n "${unexpected_needed}" ]; then
  echo "error: DT_NEEDED dependencies fall outside the allowed set (${allowed_needed}); found:" >&2
  printf '%s\n' "${unexpected_needed}" >&2
  exit 1
fi

if readelf -d "${sofile}" | grep -qE '\((RPATH|RUNPATH)\)'; then
  echo "error: DT_RPATH/DT_RUNPATH is forbidden in DGen artifacts" >&2
  readelf -d "${sofile}" | grep -E '\((RPATH|RUNPATH)\)' >&2
  exit 1
fi

if [ -n "${forbidden_path_pattern}" ] && command -v strings >/dev/null 2>&1; then
  if strings "${sofile}" | grep -E "${forbidden_path_pattern}" \
    > "${audit_root}/forbidden-paths"; then
    echo "error: forbidden developer, workspace, user, or temporary path embedded in shared object" >&2
    cat "${audit_root}/forbidden-paths" >&2
    exit 1
  fi
fi

echo "DGen ABI v1 audit passed: ${sofile}"
