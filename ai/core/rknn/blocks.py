"""Map a row of the RKNN per-layer table to a YOLOv8 layer (L0-L22) and a part inside it, from the ONNX node name.

The name is the FullName column, e.g. "Conv:/model/model.2/cv1/conv/Conv". Rows without "model.<N>" are the graph
input/output (group "io") and the decode ops after Detect (group "post_detect"), see docs 03-phuong-phap/02-tang-npu-nen-do/01-ke-hoach-do.
"""
import re

NAME_RE = re.compile(r"model\.(\d+)/?(.*)")
C2F = {2, 4, 6, 8, 12, 15, 18, 21}
CONV = {0, 1, 3, 5, 7, 16, 19}
RESIZE = {10, 13}
CONCAT = {11, 14, 17, 20}
IO_OPS = {"InputOperator", "OutputOperator"}


def part_of(layer: int, rest: str) -> str:
    head = rest.split("/")[0]
    if layer in CONV:
        return "conv"
    if layer in RESIZE:
        return "resize"
    if layer in CONCAT:
        return "concat"
    if layer in C2F:
        return {"cv1": "cv1", "Split": "split", "Concat": "concat", "cv2": "cv2"}.get(head, "bottleneck" if head.startswith("m.") else f"other:{head}")
    if layer == 9:
        return {"cv1": "cv1", "Concat": "concat", "cv2": "cv2"}.get(head, "maxpool" if head.startswith("m") else f"other:{head}")
    if layer == 22:
        return "box_branch" if head.startswith("cv2.") else "cls_branch" if head.startswith("cv3.") else "dfl" if head == "dfl" else "decode"
    return f"other:{head}"


def assign(row: dict) -> tuple[str, str]:
    """Returns (block, part): block is "L<N>", "io" or "post_detect"."""
    if row["op"] in IO_OPS:
        return "io", row["op"]
    m = NAME_RE.search(row["name"])
    if m is None:
        return "post_detect", row["op"]
    layer = int(m.group(1))
    return f"L{layer}", part_of(layer, m.group(2))
