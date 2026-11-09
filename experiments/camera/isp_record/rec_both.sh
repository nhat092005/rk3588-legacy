#!/usr/bin/env bash
# Record BOTH ISP streams at the same time, same scene, to compare their quality:
#   /dev/video11 mainpath: crop 4224x2376 (16:9) -> 3840x2160 (4K UHD)  -> /tmp/cam_4k.mp4
#   /dev/video12 selfpath: crop 4224x2376 (16:9) -> 640x360 (AI input)  -> /tmp/cam_640x360.mp4
# Run: ./rec_both.sh   Stop with ONE Ctrl+C and wait for "Crop restored".
# Optional: BPS4K=30000000 BPS360=16000000 OUT4K=... OUT360=... ./rec_both.sh
M=/dev/video11; S=/dev/video12
BPS4K=${BPS4K:-20000000}; BPS360=${BPS360:-8000000}
OUT4K=${OUT4K:-/tmp/cam_4k.mp4}; OUT360=${OUT360:-/tmp/cam_640x360.mp4}
F2=/tmp/rec_both_360.fifo
restore() { for d in $M $S; do v4l2-ctl -d $d --set-selection=target=crop,top=0,left=0,width=4224,height=3136 >/dev/null 2>&1; done; rm -f $F2; }
stop() { kill -INT $G1 2>/dev/null; kill $V2 2>/dev/null; }
trap stop INT TERM
# gst, v4l2-ctl run in their own session (setsid): the terminal's Ctrl+C reaches only this script, which sends gst ONE SIGINT.
# A second SIGINT makes gst quit before it writes the MP4 index (moov atom not found).
# 1. Configure BOTH nodes first: VIDIOC_S_FMT is refused (EBUSY) on one node while the other one streams.
for d in $M $S; do v4l2-ctl -d $d --set-selection=target=crop,top=380,left=0,width=4224,height=2376 >/dev/null || { restore; exit 1; }; done
v4l2-ctl -d $M --set-fmt-video=width=3840,height=2160,pixelformat=NV12 >/dev/null || { restore; exit 1; }
v4l2-ctl -d $S --set-fmt-video=width=640,height=360,pixelformat=NV12 >/dev/null || { restore; exit 1; }
# 2. 4K: gst reads the mainpath directly (a pipe at 373 MB/s dropped frames). Start it first, its own S_FMT must come before selfpath streams.
setsid gst-launch-1.0 -e v4l2src device=$M ! video/x-raw,format=NV12,width=3840,height=2160,framerate=30/1 ! mpph264enc bps=$BPS4K ! h264parse ! mp4mux ! filesink location=$OUT4K >/dev/null 2>&1 & G1=$!
sleep 2
# 3. 640x360: v4l2-ctl captures (gst's v4l2src would reset this node's crop), the VPU encodes through a FIFO.
rm -f $F2; mkfifo $F2
setsid gst-launch-1.0 -e filesrc location=$F2 ! rawvideoparse format=nv12 width=640 height=360 framerate=30/1 ! mpph264enc bps=$BPS360 ! h264parse ! mp4mux ! filesink location=$OUT360 >/dev/null 2>&1 & G2=$!
echo "Recording 4K -> $OUT4K and 640x360 -> $OUT360 (one Ctrl+C to stop)"
setsid stdbuf -o0 v4l2-ctl -d $S --stream-mmap=4 --stream-to=$F2 >/dev/null 2>&1 & V2=$!
wait $G1
wait $V2 $G2
restore
echo; echo "Crop restored to the full frame."
for f in $OUT4K $OUT360; do
  [ -s "$f" ] && echo "$f: $(ffprobe -v error -show_entries stream=width,height,nb_frames -of csv=p=0 "$f" | tr '\n' ' ') $(du -h "$f" | cut -f1)" && echo "  scp $USER@$(hostname -I | awk '{print $1}'):$f ."
done
