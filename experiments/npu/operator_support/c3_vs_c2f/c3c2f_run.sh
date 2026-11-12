#!/bin/bash
# C3 versus C2f board session (rule.txt of this folder). Runs on the board as user minhnhat, root only through the two passwordless helpers.
# Models: ~/c3c2f/A.rknn (C2f, ppe5_yolov8n_scratch_200ep_s0) and ~/c3c2f/B.rknn (C3, ppe5_yolov8n_c3_scratch_200ep_s0). Output: ~/c3c2f/out, marker out/done (ALL_DONE) or out/failed.
# Interleaved rounds: odd rounds A then B, even rounds B then A. Each round = 50 warm-up + 1000 back-to-back runs of ONE model in a fresh process.
D=/home/minhnhat/c3c2f; cd $D || exit 1
ROUNDS=${ROUNDS:-8}; ROUNDS7=${ROUNDS7:-4}; NDET=${NDET:-2}
mkdir -p out out/p3; rm -f out/done out/failed out/p3/stop out/p3/done out/p3/ready
BENCH=/home/minhnhat/onboard/rknn_perf_bench
log() { echo "$(date +%T) $*" >> out/progress.txt; }
state() { log "$1 npu=$(cat /sys/class/devfreq/fdab0000.npu/cur_freq) npugov=$(cat /sys/class/devfreq/fdab0000.npu/governor) ddr=$(cat /sys/class/devfreq/dmc/cur_freq) ddrgov=$(cat /sys/class/devfreq/dmc/governor) cpu4=$(cat /sys/devices/system/cpu/cpufreq/policy4/scaling_cur_freq) cpu0=$(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq) soc=$(cat /sys/class/thermal/thermal_zone0/temp)"; }
fail() { log "FAIL: $*"; echo "$*" > out/failed; cleanup; exit 1; }
cleanup() {
  touch /home/minhnhat/p3/stop 2>/dev/null
  for i in $(seq 1 20); do [ -f /home/minhnhat/p3/done ] && break; sleep 1; done
  cp /home/minhnhat/p3/progress.txt out/p3/ 2>/dev/null; cp /home/minhnhat/p3/log.txt out/p3/ 2>/dev/null
  sudo -n /usr/local/sbin/lock_freq.sh unlock > out/unlock_status.txt 2>&1
  state unlocked
}
echo "bench md5 $(md5sum $BENCH | cut -d' ' -f1) ; A $(md5sum A.rknn | cut -d' ' -f1) ; B $(md5sum B.rknn | cut -d' ' -f1) ; kernel $(uname -r)" >> out/progress.txt
# P3 helper: NPU interrupts to CPU4, CPU5, CPU6, irqbalance stopped. The helper creates /home/minhnhat/p3.
rm -rf /home/minhnhat/p3; mkdir -p /home/minhnhat/p3   # created by the user first: a root-owned dir could not receive the stop file
setsid nohup sudo -n /usr/local/sbin/p3_hold.sh 7200 > /dev/null 2>&1 < /dev/null &
for i in $(seq 1 30); do [ -f /home/minhnhat/p3/ready ] && break; sleep 1; done
[ -f /home/minhnhat/p3/ready ] || fail "p3 helper not ready"
sudo -n /usr/local/sbin/lock_freq.sh lock > out/lock_status.txt 2>&1 || fail "lock_freq failed"
state locked
grep -i npu /proc/interrupts > out/irq_before.txt
run() { # arm mask cpus tag detail
  local arm=$1 mask=$2 cpus=$3 tag=$4 det=$5
  state "start $tag"
  if [ "$det" = detail ]; then taskset -c $cpus $BENCH $arm.rknn $mask 20 200 out/$tag detail > out/$tag.log 2>&1 || fail "bench $tag"
  else taskset -c $cpus $BENCH $arm.rknn $mask 50 1000 out/$tag > out/$tag.log 2>&1 || fail "bench $tag"; fi
  state "end $tag"
}
# CORE_0 under taskset -c 4 (cell a4i4)
for r in $(seq 1 $ROUNDS); do
  if [ $((r % 2)) = 1 ]; then O="A B"; else O="B A"; fi
  for arm in $O; do run $arm 1 4 c0_${arm}_r$r; sleep 20; done
done
# per-layer detail, CORE_0, interleaved
for r in $(seq 1 $NDET); do
  if [ $((r % 2)) = 1 ]; then O="A B"; else O="B A"; fi
  for arm in $O; do run $arm 1 4 c0_${arm}_d$r detail; sleep 20; done
done
# secondary: CORE_0_1_2 under taskset -c 4-7 (P3)
for r in $(seq 1 $ROUNDS7); do
  if [ $((r % 2)) = 1 ]; then O="A B"; else O="B A"; fi
  for arm in $O; do run $arm 7 4-7 c7_${arm}_r$r; sleep 20; done
done
for arm in A B; do run $arm 7 4-7 c7_${arm}_d1 detail; sleep 20; done
grep -i npu /proc/interrupts > out/irq_after.txt
cleanup
echo ALL_DONE > out/done
