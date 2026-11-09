#!/usr/bin/env bash
# Run ON THE BOARD. Stop the C2 background load and restore the crops. gst gets one SIGINT (-e) so it exits cleanly.
D=$HOME/camload
[ -f $D/gst.pid ] && kill -INT $(cat $D/gst.pid) 2>/dev/null
[ -f $D/v4l2.pid ] && kill $(cat $D/v4l2.pid) 2>/dev/null
sleep 2
for d in /dev/video11 /dev/video12; do v4l2-ctl -d $d --set-selection=target=crop,top=0,left=0,width=4224,height=3136 >/dev/null 2>&1; done
echo stopped
