#!/bin/bash
# X1 (root): time from each NPU interrupt to the moment rknn_perf_bench runs again, with ftrace, 5 conditions x 2 blocks.
# Restores interrupt affinity, irqbalance and every tracing setting on exit, and reads them back into progress.txt.
cd /home/minhnhat/onboard || exit 1
O=out_x1
mkdir -p $O; rm -f $O/done
M=best-rk3588.rknn
T=/sys/kernel/debug/tracing
IRQS=$(grep -i npu /proc/interrupts | awk -F: '{gsub(/ /,"",$1); print $1}')
IRQBAL=$(systemctl is-active irqbalance)
OLDCLK=$(sed -n 's/.*\[\([a-z_]*\)\].*/\1/p' $T/trace_clock)
OLDBUF=$(awk '{print $1}' $T/buffer_size_kb)
declare -A SAVED_IRQ
for i in $IRQS; do SAVED_IRQ[$i]=$(cat /proc/irq/$i/smp_affinity_list); done
say() { echo "$(date +%T) $*" >> $O/progress.txt; }
readback() { for i in $IRQS; do echo -n "irq $i smp=$(cat /proc/irq/$i/smp_affinity_list) effective=$(cat /proc/irq/$i/effective_affinity_list) "; done; }
trace_off() {
  echo 0 > $T/events/enable
  for e in irq/irq_handler_entry irq/irq_handler_exit sched/sched_wakeup sched/sched_switch; do echo 0 > $T/events/$e/filter; done
  echo > $T/trace
}
restore() { for i in $IRQS; do echo "${SAVED_IRQ[$i]}" > /proc/irq/$i/smp_affinity_list; done; }
cleanup() {
  trace_off; echo $OLDCLK > $T/trace_clock; echo $OLDBUF > $T/buffer_size_kb; echo 1 > $T/tracing_on
  restore
  say "restored (read before irqbalance restarts): $(readback) trace_clock=$(cat $T/trace_clock) buffer_kb=$(cat $T/buffer_size_kb) events_enable=$(cat $T/events/enable)"
  if [ "$IRQBAL" = active ]; then systemctl start irqbalance; fi
  say "irqbalance now: $(systemctl is-active irqbalance)"
  chown -R minhnhat:minhnhat $O
}
trap cleanup EXIT
echo "npu irqs: $IRQS ; irqbalance was: $IRQBAL ; trace_clock was $OLDCLK ; buffer_kb was $OLDBUF" > $O/progress.txt
say "state: kernel $(uname -r) ; $(uptime -p) ; rknn_server pid '$(pgrep -x rknn_server)' ; adbd pid '$(pgrep -x adbd)'"
say "frequencies: $(bash /home/minhnhat/rk3588/ai/board/lock_freq.sh status | tr '\n' ';')"
say "md5: $(md5sum $M rknn_perf_bench | tr '\n' ' ')"
systemctl stop irqbalance
if [ "$(systemctl is-active irqbalance)" != inactive ]; then say "ABORT: irqbalance not inactive"; exit 1; fi
trace_off
echo mono > $T/trace_clock
echo 32768 > $T/buffer_size_kb
BUF=$(awk '{print $1}' $T/buffer_size_kb); say "buffer_size_kb read back: $BUF"
if [ "$BUF" -lt 32768 ]; then say "ABORT: buffer_size_kb below 32768"; exit 1; fi
ovr() { grep -h '^overrun' $T/per_cpu/cpu*/stats | awk '{s+=$2} END {print s+0}'; }
echo "irq == 36" > $T/events/irq/irq_handler_entry/filter
echo "irq == 36" > $T/events/irq/irq_handler_exit/filter
echo 'comm == "rknn_perf_bench"' > $T/events/sched/sched_wakeup/filter
echo 'next_comm == "rknn_perf_bench" || prev_comm == "rknn_perf_bench"' > $T/events/sched/sched_switch/filter
setcond() {
  restore
  if [ "$1" = A ]; then PIN=""; else
    app=${1#a}; app=${app%%i*}; irqc=${1##*i}
    for i in $IRQS; do echo $irqc > /proc/irq/$i/smp_affinity_list; done; PIN="taskset -c $app"
  fi
}
L=(A a1i0 a0i0 a4i0 a1i4 a4i4)
ALL=("${L[@]}"); for ((k=${#L[@]}-1; k>=0; k--)); do ALL+=("${L[$k]}"); done
block=0
for c in "${ALL[@]}"; do
  block=$((block+1)); setcond $c
  grep -i npu /proc/interrupts > $O/irq_pre_${block}_$c.txt
  for s in /sys/devices/system/cpu/cpu[0-7]/cpuidle/state[01]; do echo "$s $(cat $s/usage) $(cat $s/time)"; done > $O/idle_pre_${block}_$c.txt
  say "block $block cond $c pin='$PIN' irqbalance=$(systemctl is-active irqbalance) readback: $(readback)"
  echo > $T/trace
  OV0=$(ovr)
  for e in irq/irq_handler_entry irq/irq_handler_exit sched/sched_wakeup sched/sched_switch power/cpu_idle; do echo 1 > $T/events/$e/enable; done
  $PIN ./rknn_perf_bench $M 1 50 300 $O/b${block}_$c > $O/b${block}_$c.log 2>&1
  echo 0 > $T/events/enable
  gzip -c $T/trace > $O/trace_${block}_$c.txt.gz
  grep -i npu /proc/interrupts > $O/irq_post_${block}_$c.txt
  for s in /sys/devices/system/cpu/cpu[0-7]/cpuidle/state[01]; do echo "$s $(cat $s/usage) $(cat $s/time)"; done > $O/idle_post_${block}_$c.txt
  say "block $block post readback: $(readback) overrun_delta: $(( $(ovr) - OV0 ))"
  sleep 20
done
trap - EXIT
cleanup
echo X1_DONE > $O/done
