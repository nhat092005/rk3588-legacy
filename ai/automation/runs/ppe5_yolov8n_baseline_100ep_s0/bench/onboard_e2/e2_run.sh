#!/bin/bash
# E2 (root): 4 conditions x 2 blocks (ABCD then DCBA), 1000 runs each after 50 warm-up. Restores IRQ affinity and cpuidle on exit.
cd /home/minhnhat/onboard || exit 1
mkdir -p out_e2; rm -f out_e2/done
M=best-rk3588.rknn
IRQS=$(grep -i npu /proc/interrupts | awk -F: '{gsub(/ /,"",$1); print $1}')
echo "npu irqs: $IRQS" > out_e2/progress.txt
declare -A SAVED_IRQ SAVED_IDLE
for i in $IRQS; do SAVED_IRQ[$i]=$(cat /proc/irq/$i/smp_affinity_list); done
for s in /sys/devices/system/cpu/cpu[0-9]*/cpuidle/state1/disable; do SAVED_IDLE[$s]=$(cat $s); done
restore() { for i in $IRQS; do echo "${SAVED_IRQ[$i]}" > /proc/irq/$i/smp_affinity_list; done; for s in "${!SAVED_IDLE[@]}"; do echo "${SAVED_IDLE[$s]}" > $s; done; }
trap restore EXIT
for i in $IRQS; do echo "saved irq $i affinity ${SAVED_IRQ[$i]}" >> out_e2/progress.txt; done
irqsnap() { grep -i npu /proc/interrupts > out_e2/irq_$1.txt; }
setcond() { restore
  case $1 in
    A) PIN="";;
    B) for i in $IRQS; do echo 4 > /proc/irq/$i/smp_affinity_list; done; PIN="taskset -c 4";;
    C) for i in $IRQS; do echo 0 > /proc/irq/$i/smp_affinity_list; done; PIN="taskset -c 0";;
    D) for s in /sys/devices/system/cpu/cpu[0-9]*/cpuidle/state1/disable; do echo 1 > $s; done; PIN="";;
  esac; }
block=0
for c in A B C D D C B A; do
  block=$((block+1)); setcond $c; irqsnap pre_${block}_$c
  echo "block $block cond $c $(date +%T) pin='$PIN' state1_disable=$(cat /sys/devices/system/cpu/cpu0/cpuidle/state1/disable) irq_aff=$(for i in $IRQS; do cat /proc/irq/$i/smp_affinity_list; done | tr '\n' ' ')" >> out_e2/progress.txt
  $PIN ./rknn_perf_bench $M 1 50 1000 out_e2/b${block}_$c > out_e2/b${block}_$c.log 2>&1
  irqsnap post_${block}_$c; sleep 20
done
restore
echo "restored: $(for i in $IRQS; do cat /proc/irq/$i/smp_affinity_list; done | tr '\n' ' ') idle_state1_disable=$(cat /sys/devices/system/cpu/cpu0/cpuidle/state1/disable)" >> out_e2/progress.txt
chown -R minhnhat:minhnhat out_e2
echo E2_DONE > out_e2/done
