#!/bin/bash
# E2e board helper (root, pinned to CPU7). Takes one command file per block from /home/minhnhat/e2e and never kills rknn_server.
# Protocol per block N (files in the same directory):
#   PC writes cmd_N (a condition name)                -> daemon sets the interrupt CPU and pins every rknn_server thread, writes ready_N
#   PC runs init_runtime, then writes inited_N         -> daemon re-pins every thread (a session may add threads), records Cpus_allowed_list,
#                                                         per-thread CPU time and /proc/interrupts, writes started_N
#   PC runs the timed eval_perf calls, writes end_N    -> daemon records the same again with the final CPU of each thread, writes done_N
# Only the seven whitelisted conditions are accepted. A block that lasts longer than the timeout makes the daemon restore and exit.
# On exit (also on error): interrupt affinity, rknn_server affinity (0-7) and irqbalance are restored and read back into progress.txt.
D=/home/minhnhat/e2e
cd $D || exit 1
taskset -p -c 7 $$ > /dev/null
M=/home/minhnhat/onboard/best-rk3588.rknn
WHITELIST=" A s4i4 s0i0 s4i0 s0i4 s1i0 s1i4 "
INIT_TIMEOUT=120
RUN_TIMEOUT=400
IRQS=$(grep -i npu /proc/interrupts | awk -F: '{gsub(/ /,"",$1); print $1}')
IRQBAL=$(systemctl is-active irqbalance)
declare -A SAVED_IRQ
for i in $IRQS; do SAVED_IRQ[$i]=$(cat /proc/irq/$i/smp_affinity_list); done
say() { echo "$(date +%T) $*" >> $D/progress.txt; }
readback() { for i in $IRQS; do echo -n "irq $i smp=$(cat /proc/irq/$i/smp_affinity_list) effective=$(cat /proc/irq/$i/effective_affinity_list 2>/dev/null || echo n/a) "; done; }
srv_pid() { pgrep -x rknn_server | head -1; }
srv_allowed() { local p; p=$(srv_pid); for t in /proc/$p/task/*; do grep Cpus_allowed_list $t/status | cut -f2; done | sort | uniq -c | tr '\n' ' '; }
restore() {
  for i in $IRQS; do echo "${SAVED_IRQ[$i]}" > /proc/irq/$i/smp_affinity_list; done
  local p; p=$(srv_pid); [ -n "$p" ] && taskset -a -p -c 0-7 $p > /dev/null
}
cleanup() {
  restore
  say "exit: restored (read before irqbalance restarts): $(readback) rknn_server tasks allowed: $(srv_allowed)"
  if [ "$IRQBAL" = active ]; then systemctl start irqbalance; fi
  say "exit: irqbalance now $(systemctl is-active irqbalance), $(readback)"
  chown -R minhnhat:minhnhat $D
  echo EXITED > $D/daemon_exit
}
trap cleanup EXIT
systemctl stop irqbalance
if [ "$(systemctl is-active irqbalance)" != inactive ]; then say "ABORT: irqbalance not inactive"; exit 1; fi
P=$(srv_pid)
[ -z "$P" ] && { say "ABORT: no rknn_server"; exit 1; }
say "state: kernel $(uname -r) ; $(uptime -p) ; rknn_server pid $P tasks $(ls /proc/$P/task | wc -l) ; adbd pid $(pgrep -x adbd | head -1) ; port 5555 listeners $(ss -ltn | grep -c ':5555 ')"
say "librknnrt mapped by rknn_server: $(grep -o '/[^ ]*librknn[^ ]*' /proc/$P/maps | sort -u | tr '\n' ' ') ; md5 $(md5sum /usr/lib/librknnrt.so | cut -c1-32) ; model md5 $(md5sum $M | cut -c1-32)"
say "frequencies: $(bash /home/minhnhat/rk3588/ai/board/lock_freq.sh status | tr '\n' ';')"
say "saved irq affinities: $(for i in $IRQS; do echo -n "$i=${SAVED_IRQ[$i]} "; done) ; irqbalance was $IRQBAL"
snap() {  # tid utime stime last-cpu name
  local p; p=$(srv_pid)
  for t in /proc/$p/task/*; do
    tid=${t##*/}
    rest=$(sed 's/.*) //' $t/stat 2> /dev/null) || continue
    set -- $rest
    echo "$tid ${12} ${13} ${37} $(tr ' ' '_' < $t/comm)"
  done
}
wait_for() {  # file timeout_s
  local n=0; local limit=$(( $2 * 5 ))
  while [ ! -f "$1" ]; do
    [ -f $D/stop ] && return 2
    sleep 0.2; n=$((n+1)); [ $n -ge $limit ] && return 1
  done
  return 0
}
say "daemon ready"
touch $D/daemon_ready
while [ ! -f $D/stop ]; do
  for f in $D/cmd_*; do
    [ -e "$f" ] || continue
    N=${f##*cmd_}
    [ -f $D/ready_$N ] && continue
    cond=$(cat $f | tr -d '[:space:]')
    case "$WHITELIST" in *" $cond "*) ;; *) say "block $N: condition '$cond' not in whitelist"; echo BAD > $D/ready_$N; continue ;; esac
    restore
    if [ "$cond" != A ]; then
      app=${cond#s}; app=${app%%i*}; irqc=${cond##*i}
      for i in $IRQS; do echo $irqc > /proc/irq/$i/smp_affinity_list; done
      taskset -a -p -c $app $(srv_pid) > /dev/null
    fi
    say "block $N cond $cond pid $(srv_pid) readback: $(readback) rknn_server tasks allowed: $(srv_allowed) ; governors: $(bash /home/minhnhat/rk3588/ai/board/lock_freq.sh status | awk '{printf "%s ", $2}')"
    echo "pid=$(srv_pid) cond=$cond" > $D/ready_$N
    wait_for $D/inited_$N $INIT_TIMEOUT || { say "ABORT: block $N no inited_N within ${INIT_TIMEOUT} s or stop"; exit 1; }
    P=$(srv_pid)
    [ "$cond" != A ] && taskset -a -p -c $app $P > /dev/null
    for t in /proc/$P/task/*; do echo "${t##*/} $(grep Cpus_allowed_list $t/status | cut -f2)"; done > $D/allowed1_$N.txt
    snap > $D/cpu0_$N.txt
    grep -i npu /proc/interrupts > $D/irq_pre_$N.txt
    bash /home/minhnhat/rk3588/ai/board/lock_freq.sh status > $D/gov_pre_$N.txt
    say "block $N inited: tasks $(ls /proc/$P/task | wc -l), allowed $(srv_allowed), readback: $(readback)"
    touch $D/started_$N
    wait_for $D/end_$N $RUN_TIMEOUT || { say "ABORT: block $N no end_N within ${RUN_TIMEOUT} s or stop"; exit 1; }
    grep -i npu /proc/interrupts > $D/irq_post_$N.txt
    for t in /proc/$P/task/*; do echo "${t##*/} $(grep Cpus_allowed_list $t/status | cut -f2)"; done > $D/allowed2_$N.txt
    snap > $D/cpu1_$N.txt
    bash /home/minhnhat/rk3588/ai/board/lock_freq.sh status > $D/gov_post_$N.txt
    say "block $N end: allowed $(srv_allowed), readback: $(readback)"
    restore
    touch $D/done_$N
  done
  sleep 0.2
done
say "stop file seen"
