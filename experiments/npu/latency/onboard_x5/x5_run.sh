#!/bin/bash
# X5 (root): cell a4i4, inputs zeros, random, img1, img2 with model s0, model s1 with zeros and img1, and a second s0 zeros block. 14 blocks.
# Restores IRQ affinity and irqbalance on exit.
cd /home/minhnhat/onboard || exit 1
mkdir -p out_x5; rm -f out_x5/done
M=best-rk3588.rknn
IRQS=$(grep -i npu /proc/interrupts | awk -F: '{gsub(/ /,"",$1); print $1}')
IRQBAL=$(systemctl is-active irqbalance)
echo "npu irqs: $IRQS ; irqbalance was: $IRQBAL" > out_x5/progress.txt
{ echo "state: kernel $(uname -r) ; $(uptime -p) ; rknn_server pid '$(pgrep -x rknn_server)' ; adbd pid '$(pgrep -x adbd)' ; adb port 5555 listeners $(ss -ltn | grep -c ':5555 ')"; bash /home/minhnhat/rk3588/ai/board/sh/lock_freq.sh status; md5sum $M s1.rknn rknn_perf_bench_in inputs/*; } >> out_x5/progress.txt 2>&1
declare -A SAVED_IRQ SAVED_IDLE
for i in $IRQS; do SAVED_IRQ[$i]=$(cat /proc/irq/$i/smp_affinity_list); done
for s in /sys/devices/system/cpu/cpu[0-9]*/cpuidle/state1/disable; do SAVED_IDLE[$s]=$(cat $s); done
restore() {
  for i in $IRQS; do echo "${SAVED_IRQ[$i]}" > /proc/irq/$i/smp_affinity_list; done
  for s in "${!SAVED_IDLE[@]}"; do echo "${SAVED_IDLE[$s]}" > $s; done
}
readback() { for i in $IRQS; do echo "irq $i smp=$(cat /proc/irq/$i/smp_affinity_list) effective=$(cat /proc/irq/$i/effective_affinity_list 2>/dev/null || echo n/a)"; done | tr '\n' ' '; }
trap 'restore; if [ "$IRQBAL" = active ]; then systemctl start irqbalance; fi' EXIT
for i in $IRQS; do echo "saved irq $i affinity ${SAVED_IRQ[$i]}" >> out_x5/progress.txt; done
systemctl stop irqbalance
echo "irqbalance now: $(systemctl is-active irqbalance)" >> out_x5/progress.txt
if [ "$(systemctl is-active irqbalance)" != inactive ]; then echo "ABORT: irqbalance not inactive" >> out_x5/progress.txt; exit 1; fi
irqsnap() { grep -i npu /proc/interrupts > out_x5/irq_$1.txt; }
setcond() {
  for i in $IRQS; do echo "${SAVED_IRQ[$i]}" > /proc/irq/$i/smp_affinity_list; done
  if [ "$1" = A ]; then PIN=""; else
    app=${1#a}; app=${app%%i*}; irqc=${1##*i}
    for i in $IRQS; do echo $irqc > /proc/irq/$i/smp_affinity_list; done; PIN="taskset -c $app"
  fi
}
L=(s0-zeros s0-random s0-img1 s0-img2 s1-zeros s1-img1 s0-zerosb)
ALL=("${L[@]}"); for ((k=${#L[@]}-1; k>=0; k--)); do ALL+=("${L[$k]}"); done
block=0
setcond a4i4
for c in "${ALL[@]}"; do
  block=$((block+1)); irqsnap pre_${block}_$c
  model=${c%%-*}; inp=${c#*-}
  if [ $model = s0 ]; then MF=$M; else MF=s1.rknn; fi
  case $inp in zeros*) IN="";; random) IN="random";; *) IN="inputs/$inp.raw";; esac
  echo "block $block cond $c model $MF input '$IN' $(date +%T) pin='$PIN' irqbalance=$(systemctl is-active irqbalance) readback: $(readback)" >> out_x5/progress.txt
  $PIN ./rknn_perf_bench_in $MF 1 50 300 out_x5/b${block}_$c - 0 $IN > out_x5/b${block}_$c.log 2>&1
  irqsnap post_${block}_$c; echo "block $block post readback: $(readback)" >> out_x5/progress.txt; sleep 20
done
restore
trap - EXIT
echo "restored (read before irqbalance restarts): $(readback)" >> out_x5/progress.txt
if [ "$IRQBAL" = active ]; then systemctl start irqbalance; fi
echo "irqbalance now: $(systemctl is-active irqbalance)" >> out_x5/progress.txt
chown -R minhnhat:minhnhat out_x5
echo X5_DONE > out_x5/done
