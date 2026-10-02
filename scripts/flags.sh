#!/usr/bin/env bash
# Set extra CK compile flags (empty string clears them): scripts/flags.sh "-mllvm -foo=1"
cd "$(dirname "$0")/.." && cmake -S . -B build -DCK_EXTRA_FLAGS="$1" > /dev/null && echo "CK_EXTRA_FLAGS='$1'"
