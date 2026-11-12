# modular: detector as Backbone, Neck, Head

Goal: replace one part without touching the other two or the trainer, loss and evaluator, so that an A/B comparison against the existing runs stays valid.

## 2026-11-12: empty frame

- Files: `backbone.py`, `neck.py`, `head.py` (contracts only, no architecture), `register.py` (makes parts usable by name in a yaml, wiring example in its docstring). The three modules do not import each other or `all_core`.
- Tested with stub parts in `tests/test_modular_frame.py`: a yaml with Backbone, Neck, three `Index` rows and Detect builds, gives strides 8, 16, 32, and runs a loss forward and backward on CPU with dummy tensors. This is a mechanical check, not a training run.
- No architecture, no model folder in `ai/models/`, no config in `ai/automation/configs/`. Nothing trained or measured.
- Limits: the head must subclass Ultralytics `Detect` and be last. `register()` changes `ultralytics.nn.tasks` for the whole process. Channel counts are written in the yaml, a mismatch fails when the model is built.

## 2026-11-12: ngân sách ms

Quyết định của người dùng: thêm trần mới cho phần mạng trên NPU, giữ mốc cũ để so sánh. Không sửa số liệu cũ.

| Mức | ms (3 core, INT8, raw-head cut) | FPS (1000 / ms, chỉ phần mạng) | Dùng để |
|---|---|---|---|
| Mốc cũ | 6.77 đến 6.79 (YOLOv8n ReLU) | khoảng 147 | So sánh công bằng với các kết quả đã đo |
| Trần mới | 25 | 40 | Giới hạn thực tế cho 30 FPS của camera |

Cách tính trần 25 ms, nguồn `docs/contents/10-luong-camera-den-phat-hien/07-chay-lien-tuc-tren-board.rst` mục 3 và `06-sau-npu.rst` mục 2, `01-buc-tranh-tong-the.rst` mục 5:

- Chu kỳ camera 33.39 ms (29.95 FPS).
- Các bước ngoài NPU khi khóa xung: khoảng 0.9 ms (cảnh trống), khoảng 8 ms trường hợp xấu nhất đã đo (postprocess 2.043 ms với 34 hộp, ByteTrack 5.6 ms với 34 hộp giả).
- Cách hiểu nối tiếp: 33.39 trừ khoảng 8 là khoảng 25 ms. Lấy trường hợp xấu nhất để cảnh trống chắc chắn đạt.

Điều kiện không đổi: A/B = 1 (mọi op trên 3 core NPU), cùng eval set và seed với baseline.

Chưa biết, cần đo trước khi kết luận một mô hình đạt trần:
- Vòng camera ở `CORE_0_1_2` chưa đo (chỉ có `CORE_0`).
- ByteTrack với hộp thật nhiều người chưa đo, trên lõi A55 lên tới 51.2 ms.
- ISP `delay:29ms` không nằm trong số đầu cuối, ý nghĩa unknown.
- Mô hình lớn hơn nóng hơn, cần chạy lại kiểm nhiệt Phase 9.
- Trần chỉ đúng khi khóa xung. Không khóa thì đầu cuối 26.1 ms so với 18.97 ms.
