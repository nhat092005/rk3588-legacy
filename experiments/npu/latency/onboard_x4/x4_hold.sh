#!/bin/bash
# X4 helper (root, on CPU7). Mode 'pin': rknn_server (all threads) and NPU interrupts on CPU4, irqbalance stopped.
# Mode 'unpin': saved interrupt affinity, rknn_server on 0-7, irqbalance as before. The PC writes the wanted mode to
# /home/minhnhat/x4/mode and waits for mode_ack. Every second: epoch, mode, number of threads, number not on CPU4.
# Stops on /home/minhnhat/x4/stop or after the timeout. Never kills rknn_server. Restores and reads back on exit.
D=/home/minhnhat/x4; mkdir -p $D; cd $D || exit 1
taskset -p -c 7 $$ > /dev/null
TIMEOUT=${1:-5400}
IRQS=$(grep -i npu /proc/interrupts | awk -F: '{gsub(/ /,"",$1); print $1}')
IRQBAL=$(systemctl is-active irqbalance)
declare -A SAVED_IRQ
for i in $IRQS; do SAVED_IRQ[$i]=$(cat /proc/irq/$i/smp_affinity_list); done
say() { echo "$(date +%T) $*" >> $D/progress.txt; }
P=$(pgrep -x rknn_server | head -1)
readback() { for i in $IRQS; do echo -n "irq $i smp=$(cat /proc/irq/$i/smp_affinity_list) effective=$(cat /proc/irq/$i/effective_affinity_list) "; done; }
allowed_summary() { for t in /proc/$P/task/*; do grep Cpus_allowed_list $t/status | cut -f2; done | sort | uniq -c | tr '\n' ' '; }
unpin() {
  for i in $IRQS; do echo "${SAVED_IRQ[$i]}" > /proc/irq/$i/smp_affinity_list; done
  [ -n "$P" ] && taskset -a -p -c 0-7 $P > /dev/null
  if [ "$IRQBAL" = active ]; then systemctl start irqbalance; fi
}
pin() {
  systemctl stop irqbalance
  for i in $IRQS; do echo 4 > /proc/irq/$i/smp_affinity_list; done
  taskset -a -p -c 4 $P > /dev/null
}
cleanup() {
  unpin
  say "restored: $(readback) rknn_server allowed: $(allowed_summary) irqbalance $(systemctl is-active irqbalance)"
  chown -R minhnhat:minhnhat $D
  echo HOLD_DONE > $D/done
}
trap cleanup EXIT
say "state: kernel $(uname -r) ; $(uptime -p) ; rknn_server pid $P ; adbd pid $(pgrep -x adbd | head -1) ; irqbalance was $IRQBAL ; frequencies: $(bash /home/minhnhat/rk3588/ai/board/lock_freq.sh status | tr '\n' ';')"
[ -z "$P" ] && { say "ABORT: no rknn_server"; exit 1; }
MODE=unpin; unpin; echo unpin > $D/mode_ack
touch $D/ready
n=0
while [ ! -f $D/stop ] && [ $n -lt $TIMEOUT ]; do
  W=$(cat $D/mode 2>/dev/null || echo unpin)
  if [ "$W" != "$MODE" ]; then
    if [ "$W" = pin ]; then pin; else unpin; fi
    MODE=$W
    say "mode $MODE: $(readback) irqbalance $(systemctl is-active irqbalance) rknn_server allowed: $(allowed_summary)"
    echo $MODE > $D/mode_ack
  fi
  nt=0; nv=0
  for t in /proc/$P/task/*; do nt=$((nt+1)); a=$(grep Cpus_allowed_list $t/status 2>/dev/null | cut -f2); [ "$a" != 4 ] && nv=$((nv+1)); done
  echo "$(date +%s) $MODE $nt $nv" >> $D/log.txt
  sleep 1; n=$((n+1))
done
[ -f $D/stop ] || say "timeout after $TIMEOUT s"
