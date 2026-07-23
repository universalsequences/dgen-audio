#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
dylib=${1:-}
exports_allowlist=${DGEN_EXPORT_ALLOWLIST:-"${repo_root}/toolchain/abi/exports-v1.txt"}
undefined_allowlist=${DGEN_UNDEFINED_ALLOWLIST:-"${repo_root}/toolchain/abi/libsystem-symbols-v1.txt"}
minimum_macos=${DGEN_MINIMUM_MACOS:-11.0}

if [ -z "${dylib}" ] || [ ! -f "${dylib}" ]; then
  echo "usage: $0 DYLIB" >&2
  exit 2
fi

for tool in file nm otool strings; do
  command -v "${tool}" >/dev/null 2>&1 || {
    echo "error: required audit tool is unavailable: ${tool}" >&2
    exit 2
  }
done

audit_root=$(mktemp -d "${TMPDIR:-/tmp}/dgen-audit.XXXXXX")
trap 'rm -rf "${audit_root}"' EXIT HUP INT TERM

file_output=$(file "${dylib}")
case "${file_output}" in
  *"Mach-O 64-bit dynamically linked shared library arm64"*) ;;
  *)
    echo "error: not an arm64 Mach-O dylib: ${file_output}" >&2
    exit 1
    ;;
esac

actual_minimum=$(otool -l "${dylib}" |
  awk '/cmd LC_BUILD_VERSION/ { in_build = 1; next }
       in_build && $1 == "minos" { print $2; exit }')
if [ "${actual_minimum}" != "${minimum_macos}" ]; then
  echo "error: expected deployment target ${minimum_macos}, got ${actual_minimum:-missing}" >&2
  exit 1
fi

LC_ALL=C sort -u "${exports_allowlist}" > "${audit_root}/expected-exports"
nm -gU "${dylib}" | awk 'NF { print $NF }' | LC_ALL=C sort -u \
  > "${audit_root}/actual-exports"
if ! cmp -s "${audit_root}/expected-exports" "${audit_root}/actual-exports"; then
  echo "error: exported symbols do not exactly match DGen ABI v1" >&2
  diff -u "${audit_root}/expected-exports" "${audit_root}/actual-exports" >&2 || true
  exit 1
fi

LC_ALL=C sort -u "${undefined_allowlist}" > "${audit_root}/allowed-undefined"
nm -u "${dylib}" | awk 'NF { print $1 }' | LC_ALL=C sort -u \
  > "${audit_root}/actual-undefined"
unexpected_undefined=$(comm -23 \
  "${audit_root}/actual-undefined" "${audit_root}/allowed-undefined")
if [ -n "${unexpected_undefined}" ]; then
  echo "error: undefined symbols fall outside DGen ABI v1 allowlist:" >&2
  printf '%s\n' "${unexpected_undefined}" >&2
  exit 1
fi

dependencies=$(otool -L "${dylib}" | tail -n +3 | awk '{print $1}')
if [ "${dependencies}" != "/usr/lib/libSystem.B.dylib" ]; then
  echo "error: dylib must depend on exactly libSystem; found:" >&2
  printf '%s\n' "${dependencies:-<none>}" >&2
  exit 1
fi

if otool -l "${dylib}" | grep -q 'cmd LC_RPATH'; then
  echo "error: LC_RPATH is forbidden in DGen artifacts" >&2
  exit 1
fi

if strings "${dylib}" |
  grep -E '/Applications/Xcode|/Library/Developer/CommandLineTools|/usr/bin/clang|/Users/|/private/var/|/tmp/' \
  > "${audit_root}/forbidden-paths"; then
  echo "error: forbidden developer, workspace, user, or temporary path embedded in dylib" >&2
  cat "${audit_root}/forbidden-paths" >&2
  exit 1
fi

echo "DGen ABI v1 audit passed: ${dylib}"
