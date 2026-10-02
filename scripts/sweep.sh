#!/usr/bin/env bash
# Try CK configs one per stdin line (args for scripts/setcfg.py); appends to results/sweep.log.
#   echo "256 256 64 4 2 16 16 32 EWG Intrawave 1 false cs" | scripts/sweep.sh
# QUICK_ONLY=59 limits the run to one TestID. Leaves the last config in place.
cd "$(dirname "$0")/.."
while read -r cfg; do
  [ -z "$cfg" ] && continue
  python3 scripts/setcfg.py $cfg
  rm -f /tmp/quick.csv
  out=$(scripts/quick.sh 2>&1)
  if grep -q "error" /tmp/quick_build.log; then
    echo "$cfg | BUILD FAILED: $(grep -m1 -oE 'error: .{0,120}' /tmp/quick_build.log)" | tee -a results/sweep.log
    continue
  fi
  spill=$(grep -oE "VGPRs Spill: [0-9]+" /tmp/quick_build.log | awk '{print $3}' | sort -n | tail -1)
  tf=$(grep -E "^[0-9]+,.*ck_tile" /tmp/quick.csv 2>/dev/null | awk -F, '{s+=$12; n++; if($10!="PASS")bad=1} END{if(n) printf "%.0f%s", s/n, bad?" (FAIL/NA)":""}')
  echo "$cfg | maxspill=$spill | meanTF=${tf:-none}" | tee -a results/sweep.log
done
