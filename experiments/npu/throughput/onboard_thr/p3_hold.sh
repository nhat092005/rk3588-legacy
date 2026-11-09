#!/bin/bash
# P3 helper (root, on little CPU3). NPU interrupts: first line to CPU4, second to CPU5, third to CPU6, irqbalance stopped, set once.
# Every second: epoch, irqbalance state, smp_affinity_list of each NPU line. Stops on /home/minhnhat/p3/stop or after the timeout.
# Restores the saved affinities and irqbalance on exit and reads them back. Never touches rknn_server.
D=/home/minhnhat/p3; mkdir -p $D; cd $D || exit 1
taskset -p -c 3 $$ > /dev/null
TIMEOUT=${1:-5400}
IRQS=$(grep -i npu /proc/interrupts | awk -F: '{gsub(/ /,"",$1); print $1}')
IRQBAL=$(systemctl is-active irqbalance)
declare -A SAVED_IRQ
for i in $IRQS; do SAVED_IRQ[$i]=$(cat /proc/irq/$i/smp_affinity_list); done
say() { echo "$(date +%T) $*" >> $D/progress.txt; }
readback() { for i in $IRQS; do echo -n "irq $i smp=$(cat /proc/irq/$i/smp_affinity_list) effective=$(cat /proc/irq/$i/effective_affinity_list) "; done; }
cleanup() {
  for i in $IRQS; do echo "${SAVED_IRQ[$i]}" > /proc/irq/$i/smp_affinity_list; done
  if [ "$IRQBAL" = active ]; then systemctl start irqbalance; fi
  say "restored: $(readback) irqbalance $(systemctl is-active irqbalance)"
  chown -R minhnhat:minhnhat $D
  echo HOLD_DONE > $D/done
}
trap cleanup EXIT
say "state: kernel $(uname -r) ; $(uptime -p) ; irqbalance was $IRQBAL ; saved: $(readback) ; frequencies: $(bash /home/minhnhat/rk3588/ai/board/lock_freq.sh status | tr '\n' ';')"
systemctl stop irqbalance
cpu=4
for i in $IRQS; do echo $cpu > /proc/irq/$i/smp_affinity_list; cpu=$((cpu+1)); done
say "pinned: $(readback) irqbalance $(systemctl is-active irqbalance)"
touch $D/ready
n=0
while [ ! -f $D/stop ] && [ $n -lt $TIMEOUT ]; do
  echo "$(date +%s) $(systemctl is-active irqbalance) $(for i in $IRQS; do echo -n "$(cat /proc/irq/$i/smp_affinity_list) "; done)" >> $D/log.txt
  sleep 1; n=$((n+1))
done
[ -f $D/stop ] || say "timeout after $TIMEOUT s"
