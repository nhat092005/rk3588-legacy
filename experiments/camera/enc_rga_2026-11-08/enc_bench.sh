#!/bin/bash
# 4K encode bench: H.264 vs H.265 (mpp), real camera frames looped x4 (120 frames), CBR, locked clocks
cd /dev/shm/enc || exit 1
F=f4k.nv12; FS=12441600
SRC="cat $F $F $F $F"
RAW="rawvideoparse format=nv12 width=3840 height=2160 framerate=30/1"
OUT=/dev/shm/enc/results.tsv
echo -e "codec\tbps\trun\twall_s\tuser_s\tsys_s\tbytes\tdmc_load_mean\tdmc_load_max" > $OUT
sample() { rm -f dmc.log; while [ -f run.flag ]; do cat /sys/class/devfreq/dmc/load | cut -d@ -f1 >> dmc.log; sleep 0.05; done; }
run1() { # codec bps run outfile
  local c=$1 b=$2 r=$3 o=$4
  if [ "$c" = base ]; then ENC="fakesink"; else ENC="mpph${c}enc rc-mode=cbr bps=$b gop=30 ! filesink location=$o"; fi
  touch run.flag; sample & SP=$!
  local t0=$(date +%s.%N)
  { TIMEFORMAT="%U %S"; time bash -c "$SRC | taskset -c 4-7 gst-launch-1.0 -q fdsrc blocksize=$FS ! $RAW ! $ENC" >/dev/null 2>&1; } 2> tm.txt
  local t1=$(date +%s.%N)
  rm -f run.flag; wait $SP
  local wall=$(echo "$t1 - $t0" | bc -l)
  read u s < tm.txt
  local sz=0; [ -f "$o" ] && sz=$(stat -c %s "$o")
  local mean=$(awk '{s+=$1;n++; if($1>m)m=$1} END{printf "%.1f\t%d", s/n, m}' dmc.log)
  echo -e "$c\t$b\t$r\t$wall\t$u\t$s\t$sz\t$mean" | tee -a $OUT
}
for r in 1 2 3; do run1 base 0 $r /dev/null; done
for c in 264 265; do for b in 4000000 8000000 16000000 32000000; do for r in 1 2 3; do run1 $c $b $r out_${c}_${b}.bin; done; done; done
touch enc_done.flag
