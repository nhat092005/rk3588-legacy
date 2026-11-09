# Scripts đo luồng camera trên Orange Pi 5 (2026-11-07)

Chạy trên board `minhnhat@192.168.137.49`, không cần sudo. Sau khi đo đã xóa khỏi `/tmp` của board.

- `rga_bench.c`: đo RGA (NV12 sang RGB888). Chép lên board, `gcc -O2 -o rga_bench rga_bench.c -I/usr/include/rga -lrga`, chạy `taskset -c 6 ./rga_bench`.
- `board_pipeline_test.py`: đo NV12 sang BGR, `preprocess`, suy luận RKNNLite, `postprocess`, lấy mẫu tải GPU. Cần `ai/board/common.py` và `split_decode.py` đặt tại `/tmp/bc/ai/board/`, khung NV12 chụp bằng `v4l2-ctl` vào `/tmp/f<WxH>.nv12`, chạy với `PYTHONPATH=/tmp/bc ~/rk3588/.venv-board/bin/python`.
- Cắt 16:9 ở ISP: `v4l2-ctl -d /dev/video12 --set-selection=target=crop,top=380,left=0,width=4224,height=2376 --set-fmt-video=width=1920,height=1080,pixelformat=NV12 --stream-mmap=4 --stream-count=150 --stream-to=/dev/null`. Crop giữ nguyên giữa các lần mở nút, phải đặt lại `top=0,left=0,width=4224,height=3136`.
- Mã hóa: `gst-launch-1.0 -q videotestsrc num-buffers=300 ! video/x-raw,format=NV12,width=1920,height=1080,framerate=30/1 ! mpph264enc ! fakesink sync=false` và `ffmpeg -f lavfi -i testsrc2=size=1920x1080:rate=30 -frames:v 300 -pix_fmt yuv420p -c:v libx264 -crf 20 -f null -`.
- `npu_loop.py`: vòng suy luận NPU nền (CORE_0) trong N giây, in latency. Dùng để đo RGA khi NPU bận.
- `board_pipeline_test2.py`: giống `board_pipeline_test.py`, thêm ca ảnh có người (`/tmp/test_img.jpg`, phóng lên 1080p) và ba lần chạy liên tiếp. Chạy lần lượt không ghim và `taskset -c 4-7`.
- Khóa xung: `sudo -n /usr/local/sbin/lock_freq.sh lock`, trả lại `unlock`, kiểm `status`.
- `build_trk_pkg.py`, `track_bench.py`: đo ByteTrack. Tạo gói tracker không cần torch (`python3 build_trk_pkg.py <ultralytics_dir> <out_dir> scripts/demo/video/track.py`), nén và chép lên board `/tmp/trk_pkg`, cài `pip --target /tmp/trk_deps lap`, chạy `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 taskset -c 6 python track_bench.py 600`. Không đặt `OPENBLAS_NUM_THREADS=1` thì `update` chậm khoảng 100 lần.
- `ai/board/camera.py` (trong `ai/board`): harness chạy liên tục camera, ISP, NPU, hậu xử lý (và ByteTrack với `--track`). `camera_runs/cam_A.json` (không khóa xung, 60 s), `cam_B.json` (khóa xung, 120 s), `cam_C.json` (khóa xung, `--track`, 60 s) là ba lần kiểm harness ngày 2026-11-07, ghi ra `/tmp` chứ không vào `runs/<id>/bench/camera_<ngày>.json` của Phase 9.
- `camera.py --save-video <mp4>`: ghi video annotated qua `ffmpeg` (H.264, MP4 phân mảnh), `ffmpeg` ghim sang các lõi tiến trình chính không dùng. `make bench-camera ... VIDEO=1`.
