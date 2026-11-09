import unicodedata
L, C, R = 12, 30, 52
RT = 46  # right column text start
rows = {}
def put(r, col, text):
    text = unicodedata.normalize("NFC", text)
    line = rows.setdefault(r, [" "] * 150)
    for i, ch in enumerate(text):
        if ch != " ":
            assert line[col + i] in (" ", ch), f"overlap row {r} col {col+i}: '{line[col+i]}' vs '{ch}'"
            line[col + i] = ch
def hline(r, a, b, joints):
    put(r, a, "+" + "-" * (b - a - 1) + "+")
    for j in joints: rows[r][j] = "+"
def vbar(r0, r1, col, arrow=True):
    for r in range(r0, r1): put(r, col, "|")
    if arrow: put(r1, col, "v")

put(0, 16, "Cảm biến OV13855: RAW10 4224x3136, 30 FPS   (mục 2)")
put(1, C, "|  MIPI CSI-2, VICAP"); put(2, C, "v")
put(3, 12, "ISP: cắt 16:9 -> 4224x2376 (bỏ 380 hàng trên, 380 hàng dưới)   (mục 4)")
put(4, C, "|")
hline(5, L, R, [C]); put(6, L, "v"); put(6, R, "v")
put(7, 2, "LUỒNG 1: video chính"); put(7, RT, "LUỒNG 2: đưa vào AI")
put(8, 2, "mainpath, thu nhỏ -> 3840x2160 (4K UHD)"); put(8, RT, "selfpath, thu nhỏ -> 640x360")
put(9, L, "|"); put(9, R, "|"); put(10, L, "v"); put(10, R, "v")
put(11, 2, "xoay theo hướng camera (mục 5)"); put(11, RT, "xoay theo hướng camera (mục 5)")
put(12, 2, "quay camera vật lý hoặc RGA"); put(12, RT, "quay camera vật lý hoặc RGA")
put(13, L, "|"); put(13, R, "|"); put(14, R, "v")
put(15, RT, "RGA3: NV12 -> RGB888, đặt vào khung 640x384   (mục 6)")
put(16, RT, "viền xám 114 (12 hàng trên, 12 hàng dưới),")
put(17, RT, "tô sẵn một lần bằng RGA2")
put(18, R, "|  [1, 384, 640, 3] NHWC"); put(19, R, "v")
put(20, RT, "NPU: ba lõi (intra-frame), zero-copy qua C API   (mục 7, 8)")
put(21, RT, "model INT8, hiệu chuẩn 300 ảnh   (mục 6)")
put(22, R, "|  tensor (1, 9, 5040)"); put(23, R, "v")
put(24, RT, "CPU, ghim lõi A76   (mục 9, 11)")
put(25, RT, "hậu xử lý (giải mã hộp, NMS) -> ByteTrack")
put(26, RT, "-> hộp và mã số, dựng chữ nhãn (dùng NEON)")
put(27, RT, "quy tắc vi phạm   (mục 15)")
put(28, R, "|  x4K = x640 * 6, y4K = (y384 - 12) * 6")
# left stem runs down; queue box on the left
for r in range(13, 21): put(r, L, "|")
put(21, L, "|"); put(22, L, "|"); put(23, L, "v")
put(24, 2, "hàng đợi: giữ 1 đến 2 khung 4K")
put(25, 2, "chờ kết quả AI   (mục 14)")
for r in range(26, 29): put(r, L, "|")
put(29, R, "|"); put(29, L, "|")
hline(30, L, R, [C]); put(31, C, "v")
put(32, 6, "RGA2: vẽ khung; RGA3 hoặc RGA2: chèn nhãn (imblend), lên khung 4K   (mục 12)")
put(33, C, "|  khung 4K đã vẽ")
hline(34, L, R, [C]); put(35, L, "v"); put(35, C, "v"); put(35, R, "v")
put(36, 7, "VPU: H.264"); put(36, 22, "bộ JPEG: ảnh vi phạm"); put(36, R, "GPU: giao diện ứng dụng")
put(37, 7, "lưu video"); put(37, 22, "lấy từ khung 4K"); put(37, R, "|")
put(38, 7, "(mục 16)"); put(38, 22, "(mục 15, 16)"); put(38, R, "v")
put(39, R, "VOP2: hiển thị"); put(40, R, "(khi có màn hình, mục 12)")
out = ["".join(rows[r]).rstrip() for r in sorted(rows)]
out += ["", "Khung đi giữa các khối bằng dma-buf (mục 13). Chương trình chính viết bằng C hoặc C++ (mục 10).", "IEP không dùng (mục 12)."]
open("diagram.txt", "w", encoding="utf-8").write("\n".join(out) + "\n")
print("\n".join(out)); print("max width", max(len(x) for x in out))
