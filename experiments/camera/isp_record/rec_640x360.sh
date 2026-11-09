#!/usr/bin/env bash
# Record the AI input stream: ISP crop 4224x3136 -> 4224x2376 (16:9), scale -> 640x360 (the stream that feeds the AI model), H.264 by the VPU.
# Run: ./rec_640x360.sh    Stop with Ctrl+C. Optional: BPS=16000000 ./rec_640x360.sh   (default 8 Mbit/s)
DEV=/dev/video12
BPS=${BPS:-8000000}
OUT=${OUT:-/tmp/cam_640x360.mp4}
cleanup() {
  v4l2-ctl -d "$DEV" --set-selection=target=crop,top=0,left=0,width=4224,height=3136 >/dev/null 2>&1
  echo; echo "Crop restored to the full frame."
  [ -s "$OUT" ] && ffprobe -v error -show_entries stream=width,height,nb_frames,duration -of default=nw=1 "$OUT" && ls -l "$OUT" && echo "Copy to your PC:  scp $USER@$(hostname -I | awk '{print $1}'):$OUT ."
}
trap ':' INT                       # Ctrl+C must reach gst-launch, which then finishes the file (-e), then cleanup runs
v4l2-ctl -d "$DEV" --set-selection=target=crop,top=380,left=0,width=4224,height=2376 >/dev/null || exit 1
v4l2-ctl -d "$DEV" --set-fmt-video=width=640,height=360,pixelformat=NV12 >/dev/null || { cleanup; exit 1; }
echo "Recording 640x360 @30 to $OUT (Ctrl+C to stop)"
# v4l2-ctl captures (keeps the crop above), gst only encodes: gst's v4l2src re-sets the format and the driver then resets the crop to 640x360 at (0,0)
stdbuf -o0 v4l2-ctl -d "$DEV" --stream-mmap=4 --stream-to=- 2>/dev/null \
  | gst-launch-1.0 -e fdsrc fd=0 ! rawvideoparse format=nv12 width=640 height=360 framerate=30/1 \
  ! mpph264enc bps="$BPS" ! h264parse ! mp4mux ! filesink location="$OUT" >/dev/null 2>&1
cleanup
