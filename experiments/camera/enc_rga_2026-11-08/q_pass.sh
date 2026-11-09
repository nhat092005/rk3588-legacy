#!/bin/bash
cd /dev/shm/enc
: > quality.tsv
for c in 264 265; do for b in 4000000 8000000 16000000 32000000; do
  f=out_${c}_${b}.bin; fmt=h264; [ $c = 265 ] && fmt=hevc
  cat f4k.nv12 f4k.nv12 f4k.nv12 f4k.nv12 | ffmpeg -hide_banner -nostats -loglevel info -f rawvideo -pix_fmt nv12 -video_size 3840x2160 -framerate 30 -i pipe:0 -framerate 30 -f $fmt -i $f \
    -filter_complex "[0:v]format=yuv420p,split[a][b];[1:v]format=yuv420p,split[c][d];[a][c]psnr;[b][d]ssim" -f null - > q_${c}_${b}.log 2>&1
  p=$(grep -o "PSNR y:[0-9.]* u:[0-9.]* v:[0-9.]* average:[0-9.]*" q_${c}_${b}.log | tail -1)
  s=$(grep -o "SSIM Y:[0-9.]* .*All:[0-9.]*" q_${c}_${b}.log | tail -1)
  echo -e "$c\t$b\t$(stat -c %s $f)\t$p\t$s" | tee -a quality.tsv
done; done
touch q_done.flag
