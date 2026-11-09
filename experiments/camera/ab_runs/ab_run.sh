#!/bin/bash
# A/B of mallopt pin and gc collect, locked clocks, 5 min each. Marker /tmp/ab.done.
cd ~/rk3588
M=experiments/runs/ppe5_yolov8n_baseline_100ep_s0/weights/rknn_int8
rm -f /tmp/ab.done /tmp/ab_*.json /tmp/ab_*.rows.jsonl /tmp/ab.log; rc=99
trap 'sudo -n /usr/local/sbin/lock_freq.sh unlock >/tmp/ab_unlock.log 2>&1; echo "rc=$rc $(date +%T)" > /tmp/ab.done' EXIT
sudo -n /usr/local/sbin/lock_freq.sh lock > /tmp/ab_lock.log 2>&1 || { rc=lockfail; exit 1; }
for v in "v0_none 0 0" "v3_pin_gc 1 1" "v1_pin_only 1 0" "v2_gc_only 0 1" "v0b_none 0 0"; do
  set -- $v
  echo "run $1 $(date +%T)" >> /tmp/ab.log
  PYTHONPATH=. taskset -c 4-7 .venv-board/bin/python /tmp/ab_wrap.py $2 $3 --model $M --core-mask CORE_0 --duration-s 300 --out /tmp/ab_$1.json --run-id ab --git-commit d84836b09d07bbd07e1ebc20172b24ef40347d5d --git-dirty true >> /tmp/ab.log 2>&1 || { rc=fail_$1; exit 1; }
done
rc=0
