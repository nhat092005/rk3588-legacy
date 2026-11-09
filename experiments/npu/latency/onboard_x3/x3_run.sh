#!/bin/bash
# X3 (root): C API back to back (gap 0) versus paced (gap 220 ms), six cells, irqbalance stopped.
# Per cell two blocks of 300 timed runs after 50 warm-up: gap 0 then gap 220. Cells list then reversed (24 blocks).
# Restores IRQ affinity, cpuidle and irqbalance (started again only if it was active) on exit.
cd /home/minhnhat/onboard || exit 1
mkdir -p out_x3; rm -f out_x3/done
M=best-rk3588.rknn
IRQS=$(grep -i npu /proc/interrupts | awk -F: '{gsub(/ /,"",$1); print $1}')
IRQBAL=$(systemctl is-active irqbalance)
echo "npu irqs: $IRQS ; irqbalance was: $IRQBAL" > out_x3/progress.txt
{ echo "state: kernel $(uname -r) ; $(uptime -p) ; rknn_server pid '$(pgrep -x rknn_server)' ; adbd pid '$(pgrep -x adbd)' ; adb port 5555 listeners $(ss -ltn | grep -c ':5555 ')"; bash /home/minhnhat/rk3588/ai/board/sh/lock_freq.sh status; md5sum $M rknn_perf_bench; } >> out_x3/progress.txt 2>&1
declare -A SAVED_IRQ SAVED_IDLE
for i in $IRQS; do SAVED_IRQ[$i]=$(cat /proc/irq/$i/smp_affinity_list); done
for s in /sys/devices/system/cpu/cpu[0-9]*/cpuidle/state1/disable; do SAVED_IDLE[$s]=$(cat $s); done
restore() {
  for i in $IRQS; do echo "${SAVED_IRQ[$i]}" > /proc/irq/$i/smp_affinity_list; done
  for s in "${!SAVED_IDLE[@]}"; do echo "${SAVED_IDLE[$s]}" > $s; done
}
readback() { for i in $IRQS; do echo "irq $i smp=$(cat /proc/irq/$i/smp_affinity_list) effective=$(cat /proc/irq/$i/effective_affinity_list 2>/dev/null || echo n/a)"; done | tr '\n' ' '; }
trap 'restore; if [ "$IRQBAL" = active ]; then systemctl start irqbalance; fi' EXIT
for i in $IRQS; do echo "saved irq $i affinity ${SAVED_IRQ[$i]}" >> out_x3/progress.txt; done
systemctl stop irqbalance
echo "irqbalance now: $(systemctl is-active irqbalance)" >> out_x3/progress.txt
if [ "$(systemctl is-active irqbalance)" != inactive ]; then echo "ABORT: irqbalance not inactive" >> out_x3/progress.txt; exit 1; fi
irqsnap() { grep -i npu /proc/interrupts > out_x3/irq_$1.txt; }
setcond() {
  for i in $IRQS; do echo "${SAVED_IRQ[$i]}" > /proc/irq/$i/smp_affinity_list; done
  if [ "$1" = A ]; then PIN=""; else
    app=${1#a}; app=${app%%i*}; irqc=${1##*i}
    for i in $IRQS; do echo $irqc > /proc/irq/$i/smp_affinity_list; done; PIN="taskset -c $app"
  fi
}
L=(a4i4 a0i0 a4i0 a0i4 a1i0 a1i4)
ALL=("${L[@]}"); for ((k=${#L[@]}-1; k>=0; k--)); do ALL+=("${L[$k]}"); done
echo "cells: ${#ALL[@]}" >> out_x3/progress.txt
block=0
half_n=0
for c in "${ALL[@]}"; do
  setcond $c
  half_n=$((half_n+1))
  if [ $half_n -le 6 ]; then GAPS="0 220"; else GAPS="220 0"; fi
  for g in $GAPS; do
    block=$((block+1)); irqsnap pre_${block}_${c}_g$g
    echo "block $block cond $c gap $g $(date +%T) pin='$PIN' irqbalance=$(systemctl is-active irqbalance) readback: $(readback)" >> out_x3/progress.txt
    $PIN ./rknn_perf_bench $M 1 50 300 out_x3/b${block}_${c}_g$g - $g > out_x3/b${block}_${c}_g$g.log 2>&1
    irqsnap post_${block}_${c}_g$g; echo "block $block post readback: $(readback)" >> out_x3/progress.txt
  done
  sleep 20
done
restore
trap - EXIT
echo "restored (read before irqbalance restarts): $(readback) idle_state1_disable=$(cat /sys/devices/system/cpu/cpu0/cpuidle/state1/disable)" >> out_x3/progress.txt
if [ "$IRQBAL" = active ]; then systemctl start irqbalance; fi
echo "irqbalance now: $(systemctl is-active irqbalance) ; readback after restart: $(readback)" >> out_x3/progress.txt
chown -R minhnhat:minhnhat out_x3
echo X3_DONE > out_x3/done
