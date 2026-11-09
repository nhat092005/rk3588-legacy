#!/usr/bin/env bash
# Run ON THE BOARD. Start both ISP streams as background load for condition C2 (same node setup order as rec_both.sh).
# 4K mainpath -> VPU mpph265enc -> /dev/null, selfpath 640x360 -> /dev/null. Pinned to CPU0-3. Writes pids to ~/camload/pids.
M=/dev/video11; S=/dev/video12
D=$HOME/camload; rm -rf $D; mkdir -p $D
for d in $M $S; do v4l2-ctl -d $d --set-selection=target=crop,top=380,left=0,width=4224,height=2376 >/dev/null || exit 1; done
v4l2-ctl -d $M --set-fmt-video=width=3840,height=2160,pixelformat=NV12 >/dev/null || exit 1
v4l2-ctl -d $S --set-fmt-video=width=640,height=360,pixelformat=NV12 >/dev/null || exit 1
setsid taskset -c 0-3 gst-launch-1.0 -e v4l2src device=$M ! video/x-raw,format=NV12,width=3840,height=2160,framerate=30/1 ! mpph265enc bps=20000000 ! fakesink sync=false >$D/gst.log 2>&1 </dev/null & echo $! > $D/gst.pid
sleep 2
setsid taskset -c 0-3 v4l2-ctl -d $S --stream-mmap=4 --stream-to=/dev/null >$D/v4l2.log 2>&1 </dev/null & echo $! > $D/v4l2.pid
sleep 1
echo started
