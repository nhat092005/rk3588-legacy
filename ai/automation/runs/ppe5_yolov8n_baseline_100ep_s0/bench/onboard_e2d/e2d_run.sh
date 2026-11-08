#!/bin/bash
# E2d (root): per-CPU map of app placement x NPU interrupt placement with irqbalance stopped.
# 40 blocks: list L then L reversed, 1000 runs each after 50 warm-up. Condition names: A, or a<app cpu>i<interrupt cpu>.
# Restores IRQ affinity, cpuidle and irqbalance (started again only if it was active) on exit. Every block's /proc/interrupts is saved before and after.
cd /home/minhnhat/onboard || exit 1
mkdir -p out_e2d; rm -f out_e2d/done
M=best-rk3588.rknn
IRQS=$(grep -i npu /proc/interrupts | awk -F: '{gsub(/ /,"",$1); print $1}')
IRQBAL=$(systemctl is-active irqbalance)
echo "npu irqs: $IRQS ; irqbalance was: $IRQBAL" > out_e2d/progress.txt
{ echo "state: kernel $(uname -r) ; $(uptime -p) ; rknn_server pid '$(pgrep -x rknn_server)' ; adbd pid '$(pgrep -x adbd)' ; adb port 5555 listeners $(ss -ltn | grep -c ':5555 ')"; bash /home/minhnhat/rk3588/ai/board/lock_freq.sh status; md5sum $M rknn_perf_bench; } >> out_e2d/progress.txt 2>&1
declare -A SAVED_IRQ SAVED_IDLE
for i in $IRQS; do SAVED_IRQ[$i]=$(cat /proc/irq/$i/smp_affinity_list); done
for s in /sys/devices/system/cpu/cpu[0-9]*/cpuidle/state1/disable; do SAVED_IDLE[$s]=$(cat $s); done
restore() {
  for i in $IRQS; do echo "${SAVED_IRQ[$i]}" > /proc/irq/$i/smp_affinity_list; done
  for s in "${!SAVED_IDLE[@]}"; do echo "${SAVED_IDLE[$s]}" > $s; done
}
readback() { for i in $IRQS; do echo "irq $i smp=$(cat /proc/irq/$i/smp_affinity_list) effective=$(cat /proc/irq/$i/effective_affinity_list 2>/dev/null || echo n/a)"; done | tr '\n' ' '; }
trap 'restore; if [ "$IRQBAL" = active ]; then systemctl start irqbalance; fi' EXIT
for i in $IRQS; do echo "saved irq $i affinity ${SAVED_IRQ[$i]}" >> out_e2d/progress.txt; done
systemctl stop irqbalance
echo "irqbalance now: $(systemctl is-active irqbalance)" >> out_e2d/progress.txt
if [ "$(systemctl is-active irqbalance)" != inactive ]; then echo "ABORT: irqbalance not inactive" >> out_e2d/progress.txt; exit 1; fi
irqsnap() { grep -i npu /proc/interrupts > out_e2d/irq_$1.txt; }
setcond() {
  for i in $IRQS; do echo "${SAVED_IRQ[$i]}" > /proc/irq/$i/smp_affinity_list; done
  if [ "$1" = A ]; then PIN=""; else
    app=${1#a}; app=${app%%i*}; irqc=${1##*i}
    for i in $IRQS; do echo $irqc > /proc/irq/$i/smp_affinity_list; done; PIN="taskset -c $app"
  fi
}
L=(A); for a in 0 1 2 3 4 5 6 7; do L+=(a${a}i0); done; for a in 0 1 2 3 4 5 6 7; do L+=(a${a}i4); done; L+=(a5i5 a6i6 a7i7)
ALL=("${L[@]}"); for ((k=${#L[@]}-1; k>=0; k--)); do ALL+=("${L[$k]}"); done
echo "blocks: ${#ALL[@]}" >> out_e2d/progress.txt
block=0
for c in "${ALL[@]}"; do
  block=$((block+1)); setcond $c; irqsnap pre_${block}_$c
  echo "block $block cond $c $(date +%T) pin='$PIN' irqbalance=$(systemctl is-active irqbalance) readback: $(readback)" >> out_e2d/progress.txt
  $PIN ./rknn_perf_bench $M 1 50 1000 out_e2d/b${block}_$c > out_e2d/b${block}_$c.log 2>&1
  irqsnap post_${block}_$c; echo "block $block post readback: $(readback)" >> out_e2d/progress.txt; sleep 20
done
restore
trap - EXIT
echo "restored (read before irqbalance restarts): $(readback) idle_state1_disable=$(cat /sys/devices/system/cpu/cpu0/cpuidle/state1/disable)" >> out_e2d/progress.txt
if [ "$IRQBAL" = active ]; then systemctl start irqbalance; fi
echo "irqbalance now: $(systemctl is-active irqbalance) ; readback after restart: $(readback)" >> out_e2d/progress.txt
chown -R minhnhat:minhnhat out_e2d
echo E2D_DONE > out_e2d/done
