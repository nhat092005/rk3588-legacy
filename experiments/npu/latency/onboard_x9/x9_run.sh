#!/bin/bash
# X9 (root): E2 condition B with irqbalance active, 3 trials of 3000 runs, affinity logged every second.
cd /home/minhnhat/onboard || exit 1
O=out_x9; mkdir -p $O; rm -f $O/done
M=best-rk3588.rknn
IRQS=$(grep -i npu /proc/interrupts | awk -F: '{gsub(/ /,"",$1); print $1}')
declare -A SAVED_IRQ
for i in $IRQS; do SAVED_IRQ[$i]=$(cat /proc/irq/$i/smp_affinity_list); done
say() { echo "$(date +%T) $*" >> $O/progress.txt; }
readback() { for i in $IRQS; do echo -n "$i:$(cat /proc/irq/$i/smp_affinity_list)/$(cat /proc/irq/$i/effective_affinity_list) "; done; }
cleanup() { for i in $IRQS; do echo "${SAVED_IRQ[$i]}" > /proc/irq/$i/smp_affinity_list; done; say "restored: $(readback) irqbalance $(systemctl is-active irqbalance)"; chown -R minhnhat:minhnhat $O; }
trap cleanup EXIT
echo "npu irqs: $IRQS" > $O/progress.txt
say "state: kernel $(uname -r) ; $(uptime -p) ; irqbalance $(systemctl is-active irqbalance) ; md5 $(md5sum $M rknn_perf_bench | tr '\n' ' ')"
[ "$(systemctl is-active irqbalance)" = active ] || { say "ABORT: irqbalance not active"; exit 1; }
for t in 1 2 3; do
  for i in $IRQS; do echo 4 > /proc/irq/$i/smp_affinity_list; done
  grep -i npu /proc/interrupts > $O/irq_pre_$t.txt
  say "trial $t start: $(readback)"
  taskset -c 4 ./rknn_perf_bench $M 1 50 3000 $O/t$t > $O/t$t.log 2>&1 &
  B=$!
  while kill -0 $B 2> /dev/null; do echo "$(date +%s) $(readback) irqbal=$(systemctl is-active irqbalance)" >> $O/aff_$t.txt; sleep 1; done
  grep -i npu /proc/interrupts > $O/irq_post_$t.txt
  say "trial $t end: $(readback)"
  sleep 30
done
trap - EXIT
cleanup
echo X9_DONE > $O/done
