#!/bin/bash
# board-side sampler: 1 line per second
f=/tmp/board_sampler.csv; echo "ts,npu_hz,npu_gov,dmc_hz,dmc_gov,cpu0_khz,cpu4_khz,cpu6_khz,soc_mC,npu_mC" > $f
while [ ! -e /tmp/board_sampler.stop ]; do
  echo "$(date +%s.%N),$(cat /sys/class/devfreq/fdab0000.npu/cur_freq),$(cat /sys/class/devfreq/fdab0000.npu/governor),$(cat /sys/class/devfreq/dmc/cur_freq),$(cat /sys/class/devfreq/dmc/governor),$(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq),$(cat /sys/devices/system/cpu/cpufreq/policy4/scaling_cur_freq),$(cat /sys/devices/system/cpu/cpufreq/policy6/scaling_cur_freq),$(cat /sys/class/thermal/thermal_zone0/temp),$(cat /sys/class/thermal/thermal_zone6/temp)" >> $f
  sleep 1
done
