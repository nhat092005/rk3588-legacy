"""CLI: QAT fine-tuning of the split-head SiLU model (phase 2d, candidate B), then TorchScript export and RKNN build.

    .venv/bin/python -m ai.automation.qat_split <run_id> --epochs 6 [--batch 16] [--limit-batches N]
Starts from weights/best.pt, fine-tunes with prepare_qat_fx (observers frozen at epoch E-2, BN statistics frozen at E-1), SGD, the Ultralytics
detection loss on the raw outputs, Ultralytics train augmentation. Writes weights/qat_split/{qat.pt, train_log.json} and
weights/rknn_int8_qat/{best-rk3588.rknn, metadata.yaml (split_head: true), recipe.json} (rknn build with do_quantization=False).
"""
import argparse
import copy
import json
import time
from pathlib import Path

import torch
import yaml

RUNS = Path(__file__).resolve().parent / "runs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--limit-batches", type=int, default=0, help="smoke test: stop each epoch after N batches")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-rknn", action="store_true")
    args = ap.parse_args()
    from rknn.api import RKNN
    from torch.ao.quantization import disable_observer
    from torch.ao.quantization.quantize_fx import convert_fx
    from torch.nn.intrinsic.qat import freeze_bn_stats
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.utils import DEFAULT_CFG
    from ultralytics.utils.loss import v8DetectionLoss

    from ai.core.dataset import dataset_yaml_path
    from ai.core.rknn.qat_split import prepare

    torch.manual_seed(args.seed)
    run = RUNS / args.run_id
    cfg_run = yaml.safe_load((run / "config.yaml").read_text())
    det = YOLO(str(run / "weights" / "best.pt")).model.float()
    for prm in det.parameters():  # checkpoints store the weights with requires_grad off
        prm.requires_grad_(True)
    qm = prepare(det).cuda().train()
    ref = copy.deepcopy(det).float().cuda().eval()
    ref.args = get_cfg(DEFAULT_CFG)
    criterion = v8DetectionLoss(ref)
    data = check_det_dataset(str(dataset_yaml_path(cfg_run["dataset"])))
    hyp = get_cfg(DEFAULT_CFG, overrides=dict(imgsz=640, batch=args.batch, seed=args.seed))
    dataset = build_yolo_dataset(hyp, data["train"], args.batch, data, mode="train", rect=False, stride=32)
    loader = build_dataloader(dataset, args.batch, 8, shuffle=True)
    opt = torch.optim.SGD([p for p in qm.parameters() if p.requires_grad], lr=args.lr, momentum=0.937, nesterov=True, weight_decay=5e-4)
    log = []
    for ep in range(args.epochs):
        if ep == args.epochs - 2:
            qm.apply(disable_observer)
        if ep == args.epochs - 1:
            qm.apply(freeze_bn_stats)
        t0, tot, n = time.time(), 0.0, 0
        for i, batch in enumerate(loader):
            if args.limit_batches and i >= args.limit_batches:
                break
            img = batch["img"].cuda(non_blocking=True).float() / 255
            batch = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            boxes, scores = qm(img)
            feats = [torch.zeros(img.shape[0], 1, img.shape[2] // s, img.shape[3] // s, device=img.device) for s in (8, 16, 32)]
            loss, _ = criterion({"boxes": boxes, "scores": scores, "feats": feats}, batch)
            opt.zero_grad()
            loss.sum().backward()
            torch.nn.utils.clip_grad_norm_(qm.parameters(), 10.0)
            opt.step()
            tot += float(loss.sum().detach() / img.shape[0])
            n += 1
        log.append(dict(epoch=ep + 1, mean_loss=tot / max(n, 1), batches=n, seconds=round(time.time() - t0, 1)))
        print(log[-1], flush=True)
    out = run / "weights" / "qat_split"
    out.mkdir(parents=True, exist_ok=True)
    qm.cpu().eval()
    q = convert_fx(copy.deepcopy(qm))
    ts = torch.jit.trace(q, torch.rand(1, 3, 640, 640))
    ts.save(str(out / "qat.pt"))
    (out / "train_log.json").write_text(json.dumps(dict(args=vars(args), log=log), indent=1) + "\n")
    if args.no_rknn:
        return
    rk = run / "weights" / "rknn_int8_qat"
    rk.mkdir(parents=True, exist_ok=True)
    rknn = RKNN(verbose=True, verbose_file=str(rk / "build.log"))
    assert rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform="rk3588") == 0
    assert rknn.load_pytorch(model=str(out / "qat.pt"), input_size_list=[[1, 3, 640, 640]]) == 0
    assert rknn.build(do_quantization=False) == 0
    assert rknn.export_rknn(str(rk / "best-rk3588.rknn")) == 0
    rknn.release()
    meta = yaml.safe_load((run / "weights" / "rknn_int8" / "metadata.yaml").read_text())
    meta["split_head"] = True
    (rk / "metadata.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    (rk / "recipe.json").write_text(json.dumps(dict(run_id=args.run_id, mode="qat", args=vars(args), log=log), indent=1) + "\n")
    print(f"Done: {rk}")


if __name__ == "__main__":
    main()
