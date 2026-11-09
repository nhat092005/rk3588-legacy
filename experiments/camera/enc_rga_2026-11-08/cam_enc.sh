#!/bin/bash
# camera 4K mainpath (dma-buf) -> mpp encoder, 10 s @30fps, CBR 16 Mbps, real-time source
cd /dev/shm/enc
echo -e "codec\trun\twall_s\tuser_s\tsys_s\tbytes\tdmc_mean\tframes" > cam.tsv
sample() { rm -f dmc2.log; while [ -f run2.flag ]; do cut -d@ -f1 /sys/class/devfreq/dmc/load >> dmc2.log; sleep 0.05; done; }
for r in 1 2 3; do for c in 264 265; do
  v4l2-ctl -d /dev/video11 --set-selection=target=crop,top=380,left=0,width=4224,height=2376 --set-fmt-video=width=3840,height=2160,pixelformat=NV12 >/dev/null 2>&1
  touch run2.flag; sample & SP=$!
  t0=$(date +%s.%N)
  { TIMEFORMAT="%U %S"; time taskset -c 4-7 gst-launch-1.0 -q v4l2src device=/dev/video11 io-mode=dmabuf num-buffers=300 ! video/x-raw,format=NV12,width=3840,height=2160 ! mpph${c}enc rc-mode=cbr bps=16000000 gop=30 ! filesink location=cam_${c}_$r.bin >gst.out 2>&1; } 2> tm2.txt
  t1=$(date +%s.%N); rm -f run2.flag; wait $SP
  read u s < tm2.txt; w=$(echo "$t1 - $t0"|bc -l); sz=$(stat -c %s cam_${c}_$r.bin)
  m=$(awk '{s+=$1;n++} END{printf "%.1f", s/n}' dmc2.log)
  echo -e "$c\t$r\t$w\t$u\t$s\t$sz\t$m\t300" | tee -a cam.tsv
done; done
touch cam_done.flag
