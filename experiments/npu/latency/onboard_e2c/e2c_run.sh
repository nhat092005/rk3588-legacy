#!/bin/bash
# E2c (root): split test of app placement and NPU interrupt placement with irqbalance stopped.
# Blocks A B C E F G G F E C B A, 1000 runs each after 50 warm-up.
# Restores IRQ affinity, cpuidle and irqbalance (started again only if it was active) on exit. Every block's /proc/interrupts is saved before and after.
cd /home/minhnhat/onboard || exit 1
mkdir -p out_e2c; rm -f out_e2c/done
M=best-rk3588.rknn
IRQS=$(grep -i npu /proc/interrupts | awk -F: '{gsub(/ /,"",$1); print $1}')
IRQBAL=$(systemctl is-active irqbalance)
echo "npu irqs: $IRQS ; irqbalance was: $IRQBAL" > out_e2c/progress.txt
{ echo "state: kernel $(uname -r) ; $(uptime -p) ; rknn_server pid '$(pgrep -x rknn_server)' ; adbd pid '$(pgrep -x adbd)' ; adb port 5555 listeners $(ss -ltn | grep -c ':5555 ')"; bash /home/minhnhat/rk3588/ai/board/sh/lock_freq.sh status; md5sum $M rknn_perf_bench; } >> out_e2c/progress.txt 2>&1
declare -A SAVED_IRQ SAVED_IDLE
for i in $IRQS; do SAVED_IRQ[$i]=$(cat /proc/irq/$i/smp_affinity_list); done
for s in /sys/devices/system/cpu/cpu[0-9]*/cpuidle/state1/disable; do SAVED_IDLE[$s]=$(cat $s); done
restore() {
  for i in $IRQS; do echo "${SAVED_IRQ[$i]}" > /proc/irq/$i/smp_affinity_list; done
  for s in "${!SAVED_IDLE[@]}"; do echo "${SAVED_IDLE[$s]}" > $s; done
}
readback() { for i in $IRQS; do echo "irq $i smp=$(cat /proc/irq/$i/smp_affinity_list) effective=$(cat /proc/irq/$i/effective_affinity_list 2>/dev/null || echo n/a)"; done | tr '\n' ' '; }
trap 'restore; if [ "$IRQBAL" = active ]; then systemctl start irqbalance; fi' EXIT
for i in $IRQS; do echo "saved irq $i affinity ${SAVED_IRQ[$i]}" >> out_e2c/progress.txt; done
systemctl stop irqbalance
echo "irqbalance now: $(systemctl is-active irqbalance)" >> out_e2c/progress.txt
if [ "$(systemctl is-active irqbalance)" != inactive ]; then echo "ABORT: irqbalance not inactive" >> out_e2c/progress.txt; exit 1; fi
irqsnap() { grep -i npu /proc/interrupts > out_e2c/irq_$1.txt; }
setcond() {
  for i in $IRQS; do echo "${SAVED_IRQ[$i]}" > /proc/irq/$i/smp_affinity_list; done
  case $1 in
    A) PIN="";;
    B) for i in $IRQS; do echo 4 > /proc/irq/$i/smp_affinity_list; done; PIN="taskset -c 4";;
    C) for i in $IRQS; do echo 0 > /proc/irq/$i/smp_affinity_list; done; PIN="taskset -c 0";;
    E) for i in $IRQS; do echo 0 > /proc/irq/$i/smp_affinity_list; done; PIN="taskset -c 4";;
    F) for i in $IRQS; do echo 4 > /proc/irq/$i/smp_affinity_list; done; PIN="taskset -c 0";;
    G) for i in $IRQS; do echo 0 > /proc/irq/$i/smp_affinity_list; done; PIN="taskset -c 1";;
  esac
}
block=0
for c in A B C E F G G F E C B A; do
  block=$((block+1)); setcond $c; irqsnap pre_${block}_$c
  echo "block $block cond $c $(date +%T) pin='$PIN' irqbalance=$(systemctl is-active irqbalance) readback: $(readback)" >> out_e2c/progress.txt
  $PIN ./rknn_perf_bench $M 1 50 1000 out_e2c/b${block}_$c > out_e2c/b${block}_$c.log 2>&1
  irqsnap post_${block}_$c; echo "block $block post readback: $(readback)" >> out_e2c/progress.txt; sleep 20
done
restore
trap - EXIT
echo "restored (read before irqbalance restarts): $(readback) idle_state1_disable=$(cat /sys/devices/system/cpu/cpu0/cpuidle/state1/disable)" >> out_e2c/progress.txt
if [ "$IRQBAL" = active ]; then systemctl start irqbalance; fi
echo "irqbalance now: $(systemctl is-active irqbalance) ; readback after restart: $(readback)" >> out_e2c/progress.txt
chown -R minhnhat:minhnhat out_e2c
echo E2C_DONE > out_e2c/done
