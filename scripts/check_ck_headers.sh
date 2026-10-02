#!/usr/bin/env bash
# Fail if the CK kernel object was compiled against any ck_tile header outside CK_ROOT
# (for example the older copy in /opt/rocm/include/ck_tile). Uses the header list Ninja
# recorded for the object during the last build.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/build"
CK_ROOT="$(grep -E '^CK_ROOT:' "$BUILD/CMakeCache.txt" | cut -d= -f2-)"
CK_ROOT="$(realpath "$CK_ROOT")"

obj="$(cd "$BUILD" && ninja -t targets all | grep -oE '^[^:]*ck_grouped_gemm\.cpp\.o' | head -1)"
deps="$(cd "$BUILD" && ninja -t deps "$obj")"
ck_headers="$(echo "$deps" | grep -E '/ck_tile/' | sed 's/^ *//' | sort -u || true)"

if [[ -z "$ck_headers" ]]; then
  echo "check_ck_headers: no ck_tile headers recorded for $obj (build first)" >&2
  exit 1
fi
total=$(echo "$ck_headers" | wc -l)
bad="$(echo "$ck_headers" | while read -r h; do
  case "$(realpath "$h")" in "$CK_ROOT"/*) ;; *) echo "$h" ;; esac
done)"
if [[ -n "$bad" ]]; then
  echo "check_ck_headers: FAIL - ck_tile headers from outside $CK_ROOT:" >&2
  echo "$bad" | head -20 >&2
  exit 1
fi
echo "check_ck_headers: OK - all $total ck_tile headers come from $CK_ROOT"
