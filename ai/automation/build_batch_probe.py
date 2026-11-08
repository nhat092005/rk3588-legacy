"""CLI: build FP16 RKNN of the toolkit-consumed ONNX with rknn_batch_size 1 and 4 for probe M7 (bench/onboard_q2e/rule_probe.txt).

    .venv/bin/python -m ai.automation.build_batch_probe
Writes bench/onboard_q2e/m7/{b1,b4}/best-rk3588.rknn under the baseline seed 0 run.
"""
from pathlib import Path

RUN = Path(__file__).resolve().parent / "runs" / "ppe5_yolov8n_baseline_100ep_s0"
ONNX = RUN / "bench" / "onboard_q2c" / "diag" / "ppe5_yolov8n_baseline_100ep_s0" / "int8_input.onnx"


def main() -> None:
    from rknn.api import RKNN

    for b in (1, 4):
        out = RUN / "bench" / "onboard_q2e" / "m7" / f"b{b}"
        out.mkdir(parents=True, exist_ok=True)
        rknn = RKNN(verbose=False)
        assert rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform="rk3588") == 0
        assert rknn.load_onnx(model=str(ONNX)) == 0
        assert rknn.build(do_quantization=False, rknn_batch_size=b) == 0, f"build batch {b} failed"
        assert rknn.export_rknn(str(out / "best-rk3588.rknn")) == 0
        rknn.release()
        print("built", out)


if __name__ == "__main__":
    main()
