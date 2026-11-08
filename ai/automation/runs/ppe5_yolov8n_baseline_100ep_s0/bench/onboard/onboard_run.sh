#!/bin/bash
# Runs as root on the board (needs /sys/kernel/debug/rknpu). Output in ~/onboard/out, marker out/done.
cd /home/minhnhat/onboard || exit 1
mkdir -p out; rm -f out/done out/stop_* out/*.pid
M=best-rk3588.rknn
echo "label,cpu,state,usage,time_us" > out/cpuidle.csv
snap() { for c in /sys/devices/system/cpu/cpu[0-9]*; do for s in $c/cpuidle/state*; do echo "$1,$(basename $c),$(basename $s),$(cat $s/usage),$(cat $s/time)" >> out/cpuidle.csv; done; done; }
state() { echo "$1 $(date +%T) npu=$(cat /sys/class/devfreq/fdab0000.npu/cur_freq) gov=$(cat /sys/class/devfreq/fdab0000.npu/governor) ddr=$(cat /sys/class/devfreq/dmc/cur_freq) soc=$(cat /sys/class/thermal/thermal_zone0/temp) power=$(cat /sys/kernel/debug/rknpu/power)" >> out/progress.txt; }
state start
# 1. back-to-back, 3 independent processes; batch 1 also samples rknpu load and power every 0.1 s
for b in 1 2 3; do
  snap pre_hot_$b; state hot_$b
  if [ $b = 1 ]; then
    echo "ts,power,core0_load" > out/load_hot_1.csv
    ( while [ ! -e out/stop_load ]; do echo "$(date +%s.%N),$(cat /sys/kernel/debug/rknpu/power),$(grep -o 'Core0: *[0-9]*' /sys/kernel/debug/rknpu/load | grep -o '[0-9]*$')" >> out/load_hot_1.csv; sleep 0.1; done ) &
    SP=$!
  fi
  ./rknn_perf_bench $M 1 50 1000 out/hot_$b > out/hot_$b.log 2>&1
  if [ $b = 1 ]; then touch out/stop_load; wait $SP; fi
  snap post_hot_$b; sleep 60
done
# 2. cold start test: gaps between runs, same process
snap pre_gap2s; state gap2s;  ./rknn_perf_bench $M 1 5 40 out/gap2s 0 2000 > out/gap2s.log 2>&1;  snap post_gap2s; sleep 30
snap pre_gap10s; state gap10s; ./rknn_perf_bench $M 1 0 20 out/gap10s 0 10000 > out/gap10s.log 2>&1; snap post_gap10s; sleep 30
# 3. per-layer detail, 3 independent processes
for b in 1 2 3; do snap pre_detail_$b; state detail_$b; ./rknn_perf_bench $M 1 20 200 out/detail_$b detail > out/detail_$b.log 2>&1; snap post_detail_$b; sleep 60; done
state end
chown -R minhnhat:minhnhat out
echo ONBOARD_DONE > out/done
