#!/usr/bin/env bash
# Record the main video stream: ISP crop 4224x3136 -> 4224x2376 (16:9), scale -> 3840x2160 (4K UHD), H.264 by the VPU.
# Run: ./rec_4k.sh    Stop with Ctrl+C. Optional: BPS=30000000 ./rec_4k.sh   (default 20 Mbit/s)
DEV=/dev/video11
BPS=${BPS:-20000000}
OUT=${OUT:-/tmp/cam_4k.mp4}
cleanup() {
  v4l2-ctl -d "$DEV" --set-selection=target=crop,top=0,left=0,width=4224,height=3136 >/dev/null 2>&1
  echo; echo "Crop restored to the full frame."
  [ -s "$OUT" ] && ffprobe -v error -show_entries stream=width,height,nb_frames,duration -of default=nw=1 "$OUT" && ls -l "$OUT" && echo "Copy to your PC:  scp $USER@$(hostname -I | awk '{print $1}'):$OUT ."
}
trap ':' INT                       # Ctrl+C must reach gst-launch, which then finishes the file (-e), then cleanup runs
v4l2-ctl -d "$DEV" --set-selection=target=crop,top=380,left=0,width=4224,height=2376 >/dev/null || exit 1
v4l2-ctl -d "$DEV" --set-fmt-video=width=3840,height=2160,pixelformat=NV12 >/dev/null || { cleanup; exit 1; }
echo "Recording 3840x2160 @30 to $OUT (Ctrl+C to stop)"
gst-launch-1.0 -e v4l2src device="$DEV" ! video/x-raw,format=NV12,width=3840,height=2160,framerate=30/1 \
  ! mpph264enc bps="$BPS" ! h264parse ! mp4mux ! filesink location="$OUT" >/dev/null 2>&1
cleanup
