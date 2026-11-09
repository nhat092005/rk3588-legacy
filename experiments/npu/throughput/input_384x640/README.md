# input_384x640

Input 384x640 against 640x640 for the ISP stream of 640x360 (12 padded rows each side against 140). Measured 2026-11-08.
Documentation: `docs/contents/00-tong-quan/04-gioi-han-va-huong-phat-trien.rst` (decision 6).

## Kept

| File | Content |
|---|---|
| `rknn_<H>x640/model/best-rk3588.rknn`, `dataset.txt`, `metadata.yaml`, `rknn_<H>x640/calib.cache` | INT8 builds of `ppe5_yolov8n_baseline_100ep_s0` at 640x640 and 384x640. `dataset.txt` lists the calibration images and still holds the old `tmp/npu-split/...` paths, kept verbatim |
| `dump_A640.npz`, `dump_B384.npz`, `dump_B384_012.npz`, `dump_Aproj.npz` | Raw NPU outputs on the 262 bands, read by `band_int8_eval.py` and `band_edge.py` |
| `export_*.log`, `band/band.yaml` | Export logs, dataset file of the bands |
| `export_hw.py`, `band_eval.py`, `band_dump.py`, `band_int8_eval.py`, `band_edge.py` | Scripts of the experiment |

## Deleted on 2026-11-10, regenerated with

| Deleted | Regenerate |
|---|---|
| `band/images`, `band/labels` (262 bands cut from `test` and `test_ext`) | `.venv/bin/python experiments/npu/throughput/input_384x640/band_eval.py test,test_ext` |
| `rknn_<H>x640/calib/` (100 train images of `data/ppe5/processed/images/train`, shuffled with `random.Random(0)`, letterboxed) | `.venv/bin/python experiments/npu/throughput/input_384x640/export_hw.py <H> 640` with H = 384 or 640 |
| `rknn_640x640/w/best.pt` | Same file as `experiments/runs/ppe5_yolov8n_baseline_100ep_s0/weights/best.pt` (equal SHA-256) |

The calibration images are regenerated, but the `.rknn` files were not rebuilt to compare bytes, so the kept `.rknn` files are the measured ones.
