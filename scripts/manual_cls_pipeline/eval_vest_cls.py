# -*- coding: utf-8 -*-
"""方向性门禁：以 no_vest 召回为准评估分类模型（SOP：漏报 > 误报）。

背景：top1 是聚合指标，在 vest 占多数时会把 no_vest 漏报藏起来（现场实测
过一个 val top1=0.958 但大面积高置信漏报的候选）。上线前必须用本脚本过门禁。

用法：
  python scripts/manual_cls_pipeline/eval_vest_cls.py \
      --model runs/.../weights/best.pt \
      --data dataset/manual_cls_aug_split \
      --target-recall 0.95

退出码：0=PASS，1=FAIL。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_pairs(val_dir: Path, class_names: list[str]) -> tuple[list[Path], np.ndarray]:
    files, labels = [], []
    for idx, name in enumerate(class_names):
        d = val_dir / name
        if not d.exists():
            sys.exit(f"类别目录不存在: {d}")
        for f in sorted(d.glob("*.jpg")):
            files.append(f)
            labels.append(idx)
    return files, np.array(labels)


def main() -> None:
    ap = argparse.ArgumentParser(description="no_vest 召回门禁")
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=str(PROJECT_ROOT / "dataset" / "manual_cls_aug_split"))
    ap.add_argument("--val-dir", default=None, help="默认 {data}/val")
    ap.add_argument("--target-recall", type=float, default=0.95, help="no_vest 召回门禁线 @0.5")
    ap.add_argument("--min-vest-recall", type=float, default=0.85,
                    help="推荐阈值时要求的最小 vest 召回（防阈值过调到大量误报）")
    ap.add_argument("--imgsz", type=int, default=224, help="须与训练/部署一致（0904 基线为 320）")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default=0)
    args = ap.parse_args()

    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    from ultralytics import YOLO

    model = YOLO(args.model)
    class_names = [model.names[i] for i in sorted(model.names)]
    val_dir = Path(args.val_dir) if args.val_dir else Path(args.data) / "val"
    if "no_vest" not in class_names or "vest" not in class_names:
        sys.exit(f"模型类别异常: {class_names}")
    nv_idx, v_idx = class_names.index("no_vest"), class_names.index("vest")

    files, labels = load_pairs(val_dir, class_names)
    print(f"模型: {args.model}")
    print(f"验证: {val_dir}  n={len(files)}  "
          f"(no_vest={int((labels == nv_idx).sum())} vest={int((labels == v_idx).sum())})")

    pvest = np.empty(len(files), dtype=np.float32)
    for i in range(0, len(files), args.batch):
        chunk = [str(f) for f in files[i:i + args.batch]]
        results = model.predict(chunk, imgsz=args.imgsz, verbose=False, device=args.device)
        for j, r in enumerate(results):
            pvest[i + j] = float(r.probs.data.cpu().numpy()[v_idx])

    is_nv = labels == nv_idx

    def metrics(th: float) -> tuple[float, float, float]:
        pred_v = pvest >= th
        nv_rec = float((~pred_v[is_nv]).mean()) if is_nv.any() else float("nan")
        v_rec = float((pred_v[~is_nv]).mean()) if (~is_nv).any() else float("nan")
        acc = float((pred_v == ~is_nv).mean())
        return nv_rec, v_rec, acc

    print(f"\n{'thr':>5s} {'nv_recall':>10s} {'vest_recall':>12s} {'acc':>7s}")
    for th in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        nv, v, acc = metrics(th)
        print(f"{th:5.2f} {nv:10.3f} {v:12.3f} {acc:7.3f}")

    base_nv, base_v, base_acc = metrics(0.5)
    print(f"\n@0.5: no_vest 召回={base_nv:.3f}  vest 召回={base_v:.3f}  acc={base_acc:.3f}")

    # 推荐阈值：在保证 vest 召回的前提下最大化 no_vest 召回
    best_th, best_nv = 0.5, base_nv
    for th in np.arange(0.30, 0.91, 0.05):
        nv, v, _ = metrics(float(th))
        if v >= args.min_vest_recall and nv > best_nv:
            best_nv, best_th = nv, float(th)
    print(f"推荐阈值(vest_recall>={args.min_vest_recall}): {best_th:.2f} "
          f"-> no_vest 召回={best_nv:.3f}")

    ok = base_nv >= args.target_recall
    print(f"\n门禁 no_vest 召回@0.5 >= {args.target_recall:.2f} : "
          f"{'PASS' if ok else 'FAIL'} (实测 {base_nv:.3f})")
    if not ok:
        print("提示：不要靠更多 epoch 硬修；优先补现场难例(--extra-crops) + 均衡采样 + 换阈值参考。")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
