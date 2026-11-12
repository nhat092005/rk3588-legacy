#!/usr/bin/env bash
# On-board run of the a4 experiment with the C bench (no PC link, no rknn_server). Run on the board, detached, from the folder that holds
# rknn_perf_bench, models/*.rknn and this script. Writes out/<model>_m<mask>_r<rep>.csv (latency) and out/<model>_m<mask>_detail_detail.txt
# (per-layer table with WorkLoad), out/<model>.done per model, out/DONE at the end or out/FAIL on the first error. Never interrupt it.
set -u
cd "$(dirname "$0")" || exit 1
mkdir -p out
CPU=${CPU:-6}                       # application on a big core, as in the fast level of the E2 notes
WARM=20; LOOPS=100; DWARM=3; DLOOPS=10
ROT=("1 3 7" "3 7 1" "7 1 3")       # core_mask 1 = CORE_0, 3 = CORE_0_1, 7 = CORE_0_1_2, rotated per repetition
fail() { echo "FAIL $*" > out/FAIL; exit 1; }
rm -f out/FAIL
for m in models/*.rknn; do
  n=$(basename "$m" .rknn)
  [ -f "out/$n.done" ] && continue
  for rep in 0 1 2; do
    for mask in ${ROT[$rep]}; do
      taskset -c "$CPU" ./rknn_perf_bench "$m" "$mask" "$WARM" "$LOOPS" "out/${n}_m${mask}_r${rep}" - > "out/${n}_m${mask}_r${rep}.log" 2>&1 || fail "$n mask $mask rep $rep"
    done
  done
  for mask in 1 7; do
    taskset -c "$CPU" ./rknn_perf_bench "$m" "$mask" "$DWARM" "$DLOOPS" "out/${n}_m${mask}_detail" detail > "out/${n}_m${mask}_detail.log" 2>&1 || fail "$n mask $mask detail"
  done
  echo "$n $(cat /sys/class/thermal/thermal_zone*/temp | tr '\n' ' ')" >> out/temps.txt
  touch "out/$n.done"
done
touch out/DONE
